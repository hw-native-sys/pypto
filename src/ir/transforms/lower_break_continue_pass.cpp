/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Check if a statement is a BreakStmt
bool IsBreak(const StmtPtr& stmt) { return As<BreakStmt>(stmt) != nullptr; }

/// Check if a statement is a ContinueStmt
bool IsContinue(const StmtPtr& stmt) { return As<ContinueStmt>(stmt) != nullptr; }

/// Check if a statement tree contains a BreakStmt (at any depth)
bool ContainsBreak(const StmtPtr& stmt) {
  if (IsBreak(stmt)) return true;
  if (auto seq = As<SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
      if (ContainsBreak(s)) return true;
    }
  }
  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (ContainsBreak(if_stmt->then_body_)) return true;
    if (if_stmt->else_body_.has_value() && ContainsBreak(*if_stmt->else_body_)) return true;
  }
  if (auto op_stmts = As<OpStmts>(stmt)) {
    for (const auto& s : op_stmts->stmts_) {
      if (ContainsBreak(s)) return true;
    }
  }
  // Don't recurse into nested loops — break only affects innermost loop
  return false;
}

/// Check if a statement tree contains a ContinueStmt (at any depth, not crossing loop boundaries)
bool ContainsContinue(const StmtPtr& stmt) {
  if (IsContinue(stmt)) return true;
  if (auto seq = As<SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
      if (ContainsContinue(s)) return true;
    }
  }
  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (ContainsContinue(if_stmt->then_body_)) return true;
    if (if_stmt->else_body_.has_value() && ContainsContinue(*if_stmt->else_body_)) return true;
  }
  if (auto op_stmts = As<OpStmts>(stmt)) {
    for (const auto& s : op_stmts->stmts_) {
      if (ContainsContinue(s)) return true;
    }
  }
  return false;
}

/// Check if the then_body of an IfStmt is (or ends with) a ContinueStmt
bool ThenBodyEndsWith(const StmtPtr& then_body, bool (*check)(const StmtPtr&)) {
  if (check(then_body)) return true;
  if (auto seq = As<SeqStmts>(then_body)) {
    if (!seq->stmts_.empty() && check(seq->stmts_.back())) return true;
  }
  return false;
}

/// Remove trailing ContinueStmt from a statement body
StmtPtr RemoveTrailingContinue(const StmtPtr& body, const Span& span) {
  if (IsContinue(body)) {
    // The entire body is just a continue — return an empty SeqStmts
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  if (auto seq = As<SeqStmts>(body)) {
    if (!seq->stmts_.empty() && IsContinue(seq->stmts_.back())) {
      std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
      return std::make_shared<SeqStmts>(new_stmts, span);
    }
  }
  return body;
}

/// Remove trailing BreakStmt from a statement body
StmtPtr RemoveTrailingBreak(const StmtPtr& body, const Span& span) {
  if (IsBreak(body)) {
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  if (auto seq = As<SeqStmts>(body)) {
    if (!seq->stmts_.empty() && IsBreak(seq->stmts_.back())) {
      std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
      return std::make_shared<SeqStmts>(new_stmts, span);
    }
  }
  return body;
}

/// Create a Not(expr) expression
ExprPtr MakeLogicalNot(const ExprPtr& expr, const Span& span) {
  return std::make_shared<Not>(expr, DataType::BOOL, span);
}

/// Create an And(left, right) expression
ExprPtr MakeLogicalAnd(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return std::make_shared<And>(left, right, DataType::BOOL, span);
}

/// Create a Lt(left, right) comparison expression
ExprPtr MakeLogicalLt(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return std::make_shared<Lt>(left, right, DataType::BOOL, span);
}

/**
 * @brief Lower continue statements in a sequence of statements.
 *
 * When encountering `if (cond) continue;` followed by remaining statements,
 * transforms into `if (!cond) { remaining }`.
 *
 * @param stmts The sequence of statements to process
 * @param span Source location for new nodes
 * @return New sequence with continue lowered
 */
std::vector<StmtPtr> LowerContinueInSequence(const std::vector<StmtPtr>& stmts, const Span& span) {
  std::vector<StmtPtr> result;

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare ContinueStmt — discard it and all remaining statements
    if (IsContinue(stmt)) {
      break;
    }

    // Case 2: IfStmt whose then-body is or ends with ContinueStmt
    auto if_stmt = As<IfStmt>(stmt);
    if (if_stmt && ThenBodyEndsWith(if_stmt->then_body_, IsContinue)) {
      // Collect remaining statements after this if
      std::vector<StmtPtr> remaining;
      for (size_t j = i + 1; j < stmts.size(); ++j) {
        remaining.push_back(stmts[j]);
      }

      // Recursively lower continue in remaining statements
      remaining = LowerContinueInSequence(remaining, span);

      // Build the else body: merge original else (if any) with remaining
      std::vector<StmtPtr> else_stmts;
      if (if_stmt->else_body_.has_value()) {
        else_stmts.push_back(*if_stmt->else_body_);
      }
      for (const auto& r : remaining) {
        else_stmts.push_back(r);
      }

      // Remove the trailing ContinueStmt from then_body
      auto new_then = RemoveTrailingContinue(if_stmt->then_body_, span);

      // Build the new IfStmt
      std::optional<StmtPtr> new_else;
      if (!else_stmts.empty()) {
        new_else = std::make_shared<SeqStmts>(else_stmts, span);
      }

      result.push_back(
          std::make_shared<IfStmt>(if_stmt->condition_, new_then, new_else, if_stmt->return_vars_, span));
      return result;  // consumed all remaining
    }

    // Case 3: IfStmt whose then-body is or ends with BreakStmt
    // (handled by LowerBreak — skip here)

    // Default: keep the statement
    result.push_back(stmt);
  }

  return result;
}

/**
 * @brief Lower break in a for loop body using arithmetic alive flag.
 *
 * Instead of converting for→while, keeps the for loop and transforms:
 *   if (cond) break; remaining → alive = alive AND NOT(cond); if (alive) { remaining }
 *   bare break → alive = false
 *
 * This avoids generating WhileStmt, which some backends (PTO, CCE) cannot handle.
 * The alive flag becomes a for-loop iter_arg after ConvertToSSA.
 */
std::vector<StmtPtr> LowerBreakInForBody(const std::vector<StmtPtr>& stmts, const std::string& alive_var_name,
                                         const Span& span) {
  std::vector<StmtPtr> result;
  auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare BreakStmt — set alive = false, discard remaining
    if (IsBreak(stmt)) {
      auto alive_var = std::make_shared<Var>(alive_var_name, bool_type, span);
      auto false_const = std::make_shared<ConstBool>(false, span);
      result.push_back(std::make_shared<AssignStmt>(alive_var, false_const, span));
      break;
    }

    // Case 2: IfStmt whose then-body is or ends with BreakStmt
    auto if_stmt = As<IfStmt>(stmt);
    if (if_stmt && ThenBodyEndsWith(if_stmt->then_body_, IsBreak)) {
      // If then_body has code before break, keep it under original condition
      auto trimmed_then = RemoveTrailingBreak(if_stmt->then_body_, span);
      bool has_then_code = false;
      if (auto seq = As<SeqStmts>(trimmed_then)) {
        has_then_code = !seq->stmts_.empty();
      } else {
        has_then_code = true;
      }
      if (has_then_code) {
        result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, trimmed_then, std::nullopt,
                                                  std::vector<VarPtr>{}, span));
      }

      // alive = alive AND NOT(cond)
      auto alive_ref = std::make_shared<Var>(alive_var_name, bool_type, span);
      auto not_cond = MakeLogicalNot(if_stmt->condition_, span);
      auto new_alive_expr = MakeLogicalAnd(alive_ref, not_cond, span);
      auto alive_assign = std::make_shared<AssignStmt>(std::make_shared<Var>(alive_var_name, bool_type, span),
                                                       new_alive_expr, span);
      result.push_back(alive_assign);

      // Collect remaining statements (else body + subsequent statements)
      std::vector<StmtPtr> remaining;
      if (if_stmt->else_body_.has_value()) {
        remaining.push_back(*if_stmt->else_body_);
      }
      for (size_t j = i + 1; j < stmts.size(); ++j) {
        remaining.push_back(stmts[j]);
      }

      // Recursively lower break in remaining
      remaining = LowerBreakInForBody(remaining, alive_var_name, span);

      // Wrap remaining in if (alive) — side-effect only, no yield
      if (!remaining.empty()) {
        auto alive_guard = std::make_shared<Var>(alive_var_name, bool_type, span);
        auto remaining_body = std::make_shared<SeqStmts>(remaining, span);
        result.push_back(
            std::make_shared<IfStmt>(alive_guard, remaining_body, std::nullopt, std::vector<VarPtr>{}, span));
      }
      return result;  // consumed all remaining
    }

    // Default: keep the statement
    result.push_back(stmt);
  }

  return result;
}

/**
 * @brief Replace BreakStmt with AssignStmt(alive_var, false) in a statement tree.
 *
 * Also wraps remaining statements after the break-containing if in the else branch.
 * Used for break in WhileStmt (where the while loop is kept).
 */
std::vector<StmtPtr> LowerBreakInSequence(const std::vector<StmtPtr>& stmts,
                                          const std::string& alive_var_name, const Span& span) {
  std::vector<StmtPtr> result;
  auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare BreakStmt — replace with alive = false, discard remaining
    if (IsBreak(stmt)) {
      auto alive_var = std::make_shared<Var>(alive_var_name, bool_type, span);
      auto false_const = std::make_shared<ConstBool>(false, span);
      result.push_back(std::make_shared<AssignStmt>(alive_var, false_const, span));
      break;
    }

    // Case 2: IfStmt whose then-body is or ends with BreakStmt
    auto if_stmt = As<IfStmt>(stmt);
    if (if_stmt && ThenBodyEndsWith(if_stmt->then_body_, IsBreak)) {
      // Collect remaining statements after this if
      std::vector<StmtPtr> remaining;
      for (size_t j = i + 1; j < stmts.size(); ++j) {
        remaining.push_back(stmts[j]);
      }

      // Recursively lower break in remaining
      remaining = LowerBreakInSequence(remaining, alive_var_name, span);

      // Remove the trailing BreakStmt from then_body, add alive = false
      auto new_then_body = RemoveTrailingBreak(if_stmt->then_body_, span);
      auto alive_var = std::make_shared<Var>(alive_var_name, bool_type, span);
      auto false_const = std::make_shared<ConstBool>(false, span);
      auto alive_assign = std::make_shared<AssignStmt>(alive_var, false_const, span);

      // Build then body: original (minus break) + alive = false
      std::vector<StmtPtr> then_stmts;
      if (auto seq = As<SeqStmts>(new_then_body)) {
        then_stmts = seq->stmts_;
      } else {
        then_stmts.push_back(new_then_body);
      }
      then_stmts.push_back(alive_assign);
      auto new_then = std::make_shared<SeqStmts>(then_stmts, span);

      // Build else body: merge original else (if any) with remaining
      std::vector<StmtPtr> else_stmts;
      if (if_stmt->else_body_.has_value()) {
        else_stmts.push_back(*if_stmt->else_body_);
      }
      for (const auto& r : remaining) {
        else_stmts.push_back(r);
      }

      std::optional<StmtPtr> new_else;
      if (!else_stmts.empty()) {
        new_else = std::make_shared<SeqStmts>(else_stmts, span);
      }

      result.push_back(
          std::make_shared<IfStmt>(if_stmt->condition_, new_then, new_else, if_stmt->return_vars_, span));
      return result;  // consumed all remaining
    }

    // Default: keep the statement
    result.push_back(stmt);
  }

  return result;
}

/**
 * @brief Mutator that lowers break and continue statements to structured control flow.
 *
 * Transformations:
 * 1. `continue` → wrap remaining loop body statements in `if (!cond) { ... }`
 * 2. `break` in `for` → convert to `while` loop with alive flag
 * 3. `break` in `while` → add alive flag to condition
 *
 * Must run BEFORE ConvertToSSA so that new variables are converted to SSA form.
 */
class LowerBreakContinueMutator : public IRMutator {
 public:
  LowerBreakContinueMutator() = default;

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First, recursively process the body (handles nested structures)
    auto new_body = VisitStmt(op->body_);

    bool has_continue = ContainsContinue(new_body);
    bool has_break = ContainsBreak(new_body);

    if (!has_continue && !has_break) {
      if (new_body.get() != op->body_.get()) {
        return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                         new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                         op->chunk_policy_, op->loop_origin_);
      }
      return op;
    }

    Span span = op->span_;

    // Step 1: Lower continue in the body
    if (has_continue) {
      std::vector<StmtPtr> body_stmts;
      if (auto seq = As<SeqStmts>(new_body)) {
        body_stmts = seq->stmts_;
      } else {
        body_stmts = {new_body};
      }
      auto lowered = LowerContinueInSequence(body_stmts, span);
      new_body = std::make_shared<SeqStmts>(lowered, span);
    }

    // Step 2: If no break, keep as ForStmt with lowered body
    if (!has_break) {
      return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                       new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                       op->chunk_policy_, op->loop_origin_);
    }

    // Step 3: Break present — keep as ForStmt with alive flag
    // Instead of converting to WhileStmt (which backends may not support),
    // use arithmetic alive flag: alive = alive AND NOT(break_cond)
    std::string alive_name = "_alive_" + std::to_string(alive_counter_++);
    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);

    // Lower break in the body using for-loop pattern
    std::vector<StmtPtr> body_stmts;
    if (auto seq = As<SeqStmts>(new_body)) {
      body_stmts = LowerBreakInForBody(seq->stmts_, alive_name, span);
    } else {
      body_stmts = LowerBreakInForBody({new_body}, alive_name, span);
    }

    new_body = std::make_shared<SeqStmts>(body_stmts, span);

    // Keep as ForStmt with the lowered body
    auto for_stmt = std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                              new_body, op->return_vars_, op->span_, op->kind_,
                                              op->chunk_size_, op->chunk_policy_, op->loop_origin_);

    // Initialize alive = true before the for loop
    StmtPtr init_alive = std::make_shared<const AssignStmt>(
        std::make_shared<Var>(alive_name, bool_type, span), std::make_shared<ConstBool>(true, span), span);

    std::vector<StmtPtr> result_stmts = {init_alive, for_stmt};
    return std::make_shared<SeqStmts>(result_stmts, span);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // First, recursively process the body
    auto new_body = VisitStmt(op->body_);

    bool has_continue = ContainsContinue(new_body);
    bool has_break = ContainsBreak(new_body);

    if (!has_continue && !has_break) {
      if (new_body.get() != op->body_.get()) {
        return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, new_body, op->return_vars_,
                                           op->span_);
      }
      return op;
    }

    Span span = op->span_;

    // Lower continue in the body
    if (has_continue) {
      std::vector<StmtPtr> body_stmts;
      if (auto seq = As<SeqStmts>(new_body)) {
        body_stmts = seq->stmts_;
      } else {
        body_stmts = {new_body};
      }
      auto lowered = LowerContinueInSequence(body_stmts, span);
      new_body = std::make_shared<SeqStmts>(lowered, span);
    }

    // Lower break: add alive flag and AND into condition
    if (has_break) {
      std::string alive_name = "_alive_" + std::to_string(alive_counter_++);
      auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);

      std::vector<StmtPtr> body_stmts;
      if (auto seq = As<SeqStmts>(new_body)) {
        body_stmts = LowerBreakInSequence(seq->stmts_, alive_name, span);
      } else {
        body_stmts = LowerBreakInSequence({new_body}, alive_name, span);
      }
      new_body = std::make_shared<SeqStmts>(body_stmts, span);

      // AND the alive flag into the while condition
      auto alive_ref = std::make_shared<Var>(alive_name, bool_type, span);
      auto new_cond = MakeLogicalAnd(op->condition_, alive_ref, span);

      // Create init statement for alive
      StmtPtr init_alive = std::make_shared<const AssignStmt>(
          std::make_shared<Var>(alive_name, bool_type, span), std::make_shared<ConstBool>(true, span), span);

      auto while_stmt =
          std::make_shared<WhileStmt>(new_cond, op->iter_args_, new_body, op->return_vars_, span);

      std::vector<StmtPtr> result_stmts = {init_alive, while_stmt};
      return std::make_shared<SeqStmts>(result_stmts, span);
    }

    return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, new_body, op->return_vars_, span);
  }

 private:
  int alive_counter_ = 0;
};

FunctionPtr TransformLowerBreakContinue(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "LowerBreakContinue cannot run on null function";

  LowerBreakContinueMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

namespace pass {
Pass LowerBreakContinue() { return CreateFunctionPass(TransformLowerBreakContinue, "LowerBreakContinue"); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
