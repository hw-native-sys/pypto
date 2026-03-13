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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Helpers
// ============================================================================

/// Try to extract a compile-time integer from a ConstInt or Neg(ConstInt).
static std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto neg = std::dynamic_pointer_cast<const Neg>(expr)) {
    if (auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_)) {
      return -inner->value_;
    }
  }
  return std::nullopt;
}

static ExprPtr MakeConstBool(bool value, const Span& span) {
  return std::make_shared<ConstBool>(value, span);
}

static ExprPtr MakeNot(const ExprPtr& operand, const Span& span) {
  return std::make_shared<Not>(operand, DataType::BOOL, span);
}

static ExprPtr MakeAndExpr(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return std::make_shared<And>(left, right, DataType::BOOL, span);
}

static StmtPtr MakeSeq(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return std::make_shared<SeqStmts>(std::move(stmts), span);
}

/// Build the loop condition for converting ForStmt to WhileStmt.
/// For positive step: loop_var < stop
/// For negative step: loop_var > stop
/// For dynamic step:  (step > 0 and loop_var < stop) or (step < 0 and loop_var > stop)
static ExprPtr BuildForToWhileCondition(const ExprPtr& loop_var_expr, const ExprPtr& stop,
                                        const ExprPtr& step, const Span& span) {
  auto step_val = TryGetConstInt(step);
  if (step_val.has_value()) {
    if (*step_val > 0) {
      return MakeLt(loop_var_expr, stop, span);
    }
    return MakeGt(loop_var_expr, stop, span);
  }
  // Dynamic step: compound condition
  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto step_pos = MakeGt(step, zero, span);
  auto lt_cond = MakeLt(loop_var_expr, stop, span);
  auto pos_branch = MakeAndExpr(step_pos, lt_cond, span);

  auto step_neg = MakeLt(step, zero, span);
  auto gt_cond = MakeGt(loop_var_expr, stop, span);
  auto neg_branch = MakeAndExpr(step_neg, gt_cond, span);

  return std::make_shared<Or>(pos_branch, neg_branch, DataType::BOOL, span);
}

/// Flatten a statement into a vector of statements (unwrapping SeqStmts).
static std::vector<StmtPtr> FlattenToVec(const StmtPtr& stmt) {
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  return {stmt};
}

/// Simple mutator that substitutes one Var for another by unique_id.
class VarSubstituter : public IRMutator {
 public:
  VarSubstituter(uint64_t old_id, ExprPtr replacement)
      : old_id_(old_id), replacement_(std::move(replacement)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    if (op->UniqueId() == old_id_) {
      return replacement_;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    if (op->UniqueId() == old_id_) {
      return replacement_;
    }
    return op;
  }

 private:
  uint64_t old_id_;
  ExprPtr replacement_;
};

// ============================================================================
// Scanner: detect break/continue in a subtree without entering nested loops
// ============================================================================

struct ScanResult {
  bool has_break = false;
  bool has_continue = false;
};

class BreakContinueScanner : public IRVisitor {
 public:
  ScanResult Scan(const StmtPtr& stmt) {
    result_ = {};
    VisitStmt(stmt);
    return result_;
  }

 protected:
  void VisitStmt_(const BreakStmtPtr& /*op*/) override { result_.has_break = true; }
  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { result_.has_continue = true; }
  // Don't recurse into nested loops
  void VisitStmt_(const ForStmtPtr& /*op*/) override {}
  void VisitStmt_(const WhileStmtPtr& /*op*/) override {}

 private:
  ScanResult result_;
};

// ============================================================================
// Backward Resolution: compute yield values at a continue point
// ============================================================================

/// Given the original yield values and the statements that precede the continue
/// point, compute what values to yield.
///
/// For each yield value:
///   - If it's available at the continue point (defined in pre_continue_stmts
///     or is an iter_arg), use it.
///   - Otherwise, use the corresponding iter_arg value.
static std::vector<ExprPtr> ResolveYieldAtContinue(const std::vector<ExprPtr>& original_yield_values,
                                                   const std::vector<StmtPtr>& pre_continue_stmts,
                                                   const std::vector<IterArgPtr>& iter_args) {
  // Collect unique IDs of all variables defined before the continue point
  std::unordered_set<uint64_t> available_vars;
  for (const auto& stmt : pre_continue_stmts) {
    auto flat = FlattenToVec(stmt);
    for (const auto& s : flat) {
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
        available_vars.insert(assign->var_->UniqueId());
      }
    }
  }

  // Iter args are always available within the loop body
  for (const auto& ia : iter_args) {
    available_vars.insert(ia->UniqueId());
  }

  std::vector<ExprPtr> resolved;
  resolved.reserve(original_yield_values.size());

  for (size_t j = 0; j < original_yield_values.size(); ++j) {
    const auto& val = original_yield_values[j];
    auto var = std::dynamic_pointer_cast<const Var>(val);
    if (!var) {
      // Not a variable reference (e.g., a constant) — use as-is
      resolved.push_back(val);
      continue;
    }

    if (available_vars.count(var->UniqueId())) {
      resolved.push_back(val);
      continue;
    }

    // Variable not available — use corresponding iter_arg
    if (j < iter_args.size()) {
      resolved.push_back(iter_args[j]);
    } else {
      resolved.push_back(val);
    }
  }

  return resolved;
}

// ============================================================================
// Body processing with phi-node approach
//
// Instead of putting YieldStmt inside IfStmt branches for the loop,
// we use IfStmt's own return_vars (phi nodes) to merge values from
// the continue path and normal path, then have a single trailing
// YieldStmt at the loop body's top level.
// ============================================================================

/// Result of processing a loop body for break/continue elimination.
struct BodyResult {
  std::vector<StmtPtr> stmts;         // Processed body (no trailing yield)
  std::vector<ExprPtr> yield_values;  // Values for the trailing yield
};

/// Find the YieldStmt at the end of a loop body.
static std::shared_ptr<const YieldStmt> FindTrailingYield(const StmtPtr& body) {
  if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(body)) {
    return yield;
  }
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(body)) {
    if (!seq->stmts_.empty()) {
      return FindTrailingYield(seq->stmts_.back());
    }
  }
  return nullptr;
}

/// Remove the trailing YieldStmt from a body.
static std::vector<StmtPtr> RemoveTrailingYieldToVec(const StmtPtr& body) {
  auto flat = FlattenToVec(body);
  if (!flat.empty()) {
    if (std::dynamic_pointer_cast<const YieldStmt>(flat.back())) {
      flat.pop_back();
    }
  }
  return flat;
}

/// Create phi Vars for an IfStmt's return_vars.
static std::vector<VarPtr> CreatePhiVars(const std::vector<ExprPtr>& values, int& counter, const Span& span) {
  std::vector<VarPtr> phis;
  phis.reserve(values.size());
  for (const auto& val : values) {
    auto type = val->GetType();
    auto name = "__phi_" + std::to_string(counter++);
    phis.push_back(std::make_shared<Var>(name, type, span));
  }
  return phis;
}

/// Convert a vector of VarPtr to ExprPtr.
static std::vector<ExprPtr> VarsToExprs(const std::vector<VarPtr>& vars) {
  std::vector<ExprPtr> exprs;
  exprs.reserve(vars.size());
  for (const auto& v : vars) {
    exprs.push_back(v);
  }
  return exprs;
}

/// Collect the statements from the "normal" (non-escape) branch of an IfStmt,
/// followed by the statements after the IfStmt.
/// When escape_in_then=true, the normal path is: else body + post.
/// When escape_in_then=false, the normal path is: then body + post.
static std::vector<StmtPtr> CollectNormalPath(const std::shared_ptr<const IfStmt>& if_stmt,
                                              bool escape_in_then, const std::vector<StmtPtr>& post) {
  std::vector<StmtPtr> normal_stmts;
  if (escape_in_then) {
    if (if_stmt->else_body_.has_value()) {
      auto flat = FlattenToVec(*if_stmt->else_body_);
      normal_stmts.insert(normal_stmts.end(), flat.begin(), flat.end());
    }
  } else {
    auto flat = FlattenToVec(if_stmt->then_body_);
    normal_stmts.insert(normal_stmts.end(), flat.begin(), flat.end());
  }
  normal_stmts.insert(normal_stmts.end(), post.begin(), post.end());
  return normal_stmts;
}

/// Build an IfStmt that routes the escape branch (break/continue) to yield
/// escape_values, and the normal branch to yield normal_result values.
/// escape_in_then=true means the then-branch escapes; false means else-branch escapes.
/// Returns BodyResult with pre stmts + the new IfStmt, and phi_exprs as yield values.
static BodyResult BuildEscapeIfStmt(const std::shared_ptr<const IfStmt>& if_stmt, bool escape_in_then,
                                    const std::vector<StmtPtr>& pre,
                                    const std::vector<ExprPtr>& escape_values, BodyResult normal_result,
                                    int& name_counter, const Span& span) {
  auto phi_vars = CreatePhiVars(escape_values, name_counter, span);
  auto phi_exprs = VarsToExprs(phi_vars);

  auto escape_yield = std::make_shared<YieldStmt>(escape_values, if_stmt->span_);

  std::vector<StmtPtr> normal_parts(normal_result.stmts.begin(), normal_result.stmts.end());
  normal_parts.push_back(std::make_shared<YieldStmt>(std::move(normal_result.yield_values), if_stmt->span_));
  auto normal_body = MakeSeq(std::move(normal_parts), if_stmt->span_);

  StmtPtr then_body = escape_in_then ? static_cast<StmtPtr>(escape_yield) : normal_body;
  StmtPtr else_body = escape_in_then ? normal_body : static_cast<StmtPtr>(escape_yield);

  auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, then_body, std::make_optional(else_body),
                                         phi_vars, if_stmt->span_);

  std::vector<StmtPtr> result(pre.begin(), pre.end());
  result.push_back(new_if);
  return BodyResult{std::move(result), std::move(phi_exprs)};
}

/// Process a list of statements (loop body without trailing yield) to eliminate
/// continue statements. Returns BodyResult with restructured stmts and
/// the values that should be yielded at the end.
///
/// Uses the phi-node approach: IfStmt branches yield to return_vars,
/// and the final yield uses those return_vars.
static BodyResult ProcessBodyForContinue(const std::vector<StmtPtr>& stmts,
                                         const std::vector<IterArgPtr>& iter_args,
                                         const std::vector<ExprPtr>& original_yield_values, int& name_counter,
                                         const Span& span) {
  BreakContinueScanner scanner;

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare ContinueStmt
    if (std::dynamic_pointer_cast<const ContinueStmt>(stmt)) {
      std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
      auto continue_values = ResolveYieldAtContinue(original_yield_values, pre, iter_args);
      return BodyResult{std::move(pre), std::move(continue_values)};
    }

    // Case 2: IfStmt containing continue in a branch
    auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt);
    if (!if_stmt) continue;

    auto then_scan = scanner.Scan(if_stmt->then_body_);
    bool else_has_continue = false;
    if (if_stmt->else_body_.has_value()) {
      else_has_continue = scanner.Scan(*if_stmt->else_body_).has_continue;
    }

    if (!then_scan.has_continue && !else_has_continue) continue;

    bool escape_in_then = then_scan.has_continue;
    std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
    std::vector<StmtPtr> post(stmts.begin() + static_cast<ptrdiff_t>(i) + 1, stmts.end());

    auto continue_values = ResolveYieldAtContinue(original_yield_values, pre, iter_args);
    auto normal_stmts = CollectNormalPath(if_stmt, escape_in_then, post);
    auto normal_result =
        ProcessBodyForContinue(normal_stmts, iter_args, original_yield_values, name_counter, span);

    if (original_yield_values.empty()) {
      // No iter_args — no yields or phi needed
      auto empty_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, if_stmt->span_);
      auto filled_body = MakeSeq(std::move(normal_result.stmts), if_stmt->span_);
      StmtPtr then_body = escape_in_then ? static_cast<StmtPtr>(empty_body) : filled_body;
      StmtPtr else_body = escape_in_then ? filled_body : static_cast<StmtPtr>(empty_body);
      auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, then_body, std::make_optional(else_body),
                                             std::vector<VarPtr>{}, if_stmt->span_);
      std::vector<StmtPtr> result(pre.begin(), pre.end());
      result.push_back(new_if);
      return BodyResult{std::move(result), {}};
    }

    return BuildEscapeIfStmt(if_stmt, escape_in_then, pre, continue_values, std::move(normal_result),
                             name_counter, span);
  }

  // No continue found — return as-is
  return BodyResult{std::vector<StmtPtr>(stmts.begin(), stmts.end()), original_yield_values};
}

/// Process a list of statements to eliminate break statements.
/// Uses the phi-node approach analogous to ProcessBodyForContinue.
static BodyResult ProcessBodyForBreak(const std::vector<StmtPtr>& stmts, size_t break_flag_index,
                                      const std::vector<IterArgPtr>& while_iter_args,
                                      const std::vector<ExprPtr>& original_yield_values, int& name_counter,
                                      const Span& span) {
  BreakContinueScanner scanner;

  // Helper: build the yield values for the break path.
  // Sets break_flag=True, keeps loop_var advancement, uses current iter_arg values.
  auto build_break_values = [&](const std::vector<StmtPtr>& pre_stmts) -> std::vector<ExprPtr> {
    auto user_iter_args = std::vector<IterArgPtr>(
        while_iter_args.begin() + static_cast<ptrdiff_t>(break_flag_index) + 1, while_iter_args.end());
    auto user_orig_values =
        std::vector<ExprPtr>(original_yield_values.begin() + static_cast<ptrdiff_t>(break_flag_index) + 1,
                             original_yield_values.end());
    auto resolved_user = ResolveYieldAtContinue(user_orig_values, pre_stmts, user_iter_args);

    std::vector<ExprPtr> break_values;
    for (size_t j = 0; j < break_flag_index; ++j) {
      break_values.push_back(original_yield_values[j]);
    }
    break_values.push_back(MakeConstBool(true, span));
    break_values.insert(break_values.end(), resolved_user.begin(), resolved_user.end());
    return break_values;
  };

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare BreakStmt
    if (std::dynamic_pointer_cast<const BreakStmt>(stmt)) {
      std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
      auto break_values = build_break_values(pre);
      return BodyResult{std::move(pre), std::move(break_values)};
    }

    // Case 2: IfStmt containing break in a branch
    auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt);
    if (!if_stmt) continue;

    auto then_scan = scanner.Scan(if_stmt->then_body_);
    bool else_has_break = false;
    if (if_stmt->else_body_.has_value()) {
      else_has_break = scanner.Scan(*if_stmt->else_body_).has_break;
    }

    if (!then_scan.has_break && !else_has_break) continue;

    bool escape_in_then = then_scan.has_break;
    std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
    std::vector<StmtPtr> post(stmts.begin() + static_cast<ptrdiff_t>(i) + 1, stmts.end());

    auto break_values = build_break_values(pre);
    auto normal_stmts = CollectNormalPath(if_stmt, escape_in_then, post);
    auto normal_result = ProcessBodyForBreak(normal_stmts, break_flag_index, while_iter_args,
                                             original_yield_values, name_counter, span);

    return BuildEscapeIfStmt(if_stmt, escape_in_then, pre, break_values, std::move(normal_result),
                             name_counter, span);
  }

  // No break found — return as-is
  return BodyResult{std::vector<StmtPtr>(stmts.begin(), stmts.end()), original_yield_values};
}

// ============================================================================
// Main Mutator
// ============================================================================

class LowerBreakContinueMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First, recurse into the body to handle nested loops
    auto new_body = VisitStmt(op->body_);

    BreakContinueScanner scanner;
    auto scan = scanner.Scan(new_body);

    if (!scan.has_break && !scan.has_continue) {
      if (new_body.get() == op->body_.get()) {
        return op;
      }
      return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                       new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                       op->chunk_policy_, op->loop_origin_);
    }

    if (scan.has_break) {
      return LowerForWithBreak(op, new_body, scan.has_continue);
    }

    // Only continue, no break
    return LowerForWithContinue(op, new_body);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // First, recurse into the body to handle nested loops
    auto new_body = VisitStmt(op->body_);

    BreakContinueScanner scanner;
    auto scan = scanner.Scan(new_body);

    if (!scan.has_break && !scan.has_continue) {
      if (new_body.get() == op->body_.get()) {
        return op;
      }
      return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, new_body, op->return_vars_,
                                         op->span_);
    }

    if (scan.has_break) {
      return LowerWhileWithBreak(op, new_body, scan.has_continue);
    }

    // Only continue
    return LowerWhileWithContinue(op, new_body);
  }

 private:
  int name_counter_ = 0;

  std::string FreshName(const std::string& prefix) { return prefix + "_" + std::to_string(name_counter_++); }

  /// Extract yield values from a trailing YieldStmt (empty vector if none).
  static std::vector<ExprPtr> GetTrailingYieldValues(const std::shared_ptr<const YieldStmt>& trailing_yield) {
    return trailing_yield ? trailing_yield->value_ : std::vector<ExprPtr>{};
  }

  // --------------------------------------------------------------------------
  // ForStmt with only continue
  // --------------------------------------------------------------------------
  StmtPtr LowerForWithContinue(const ForStmtPtr& op, const StmtPtr& body) {
    auto trailing_yield = FindTrailingYield(body);
    auto body_stmts = RemoveTrailingYieldToVec(body);
    auto orig_yield_values = GetTrailingYieldValues(trailing_yield);

    auto result =
        ProcessBodyForContinue(body_stmts, op->iter_args_, orig_yield_values, name_counter_, op->span_);

    std::vector<StmtPtr> final_stmts(result.stmts.begin(), result.stmts.end());
    if (!result.yield_values.empty() || trailing_yield) {
      final_stmts.push_back(std::make_shared<YieldStmt>(std::move(result.yield_values), op->span_));
    }
    auto final_body = MakeSeq(std::move(final_stmts), op->span_);

    return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                     final_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                     op->chunk_policy_, op->loop_origin_);
  }

  // --------------------------------------------------------------------------
  // ForStmt with break (and possibly continue)
  // --------------------------------------------------------------------------
  StmtPtr LowerForWithBreak(const ForStmtPtr& op, const StmtPtr& body, bool also_has_continue) {
    Span span = op->span_;

    // Create break_flag and loop_var iter_args
    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    auto break_flag_name = FreshName("__brk_flag");
    auto break_flag_init = MakeConstBool(false, span);
    auto break_flag_iter = std::make_shared<IterArg>(break_flag_name, bool_type, break_flag_init, span);

    auto loop_var_type = op->loop_var_->GetType();
    auto loop_var_name = FreshName("__lv");
    auto loop_var_init = op->start_;
    auto loop_var_iter = std::make_shared<IterArg>(loop_var_name, loop_var_type, loop_var_init, span);

    // Substitute old ForStmt loop_var with the new loop_var iter_arg in the body
    VarSubstituter sub(op->loop_var_->UniqueId(), loop_var_iter);
    auto substituted_body = sub.VisitStmt(body);

    // Build new iter_args: [break_flag, loop_var, ...original_iter_args...]
    std::vector<IterArgPtr> new_iter_args = {break_flag_iter, loop_var_iter};
    new_iter_args.insert(new_iter_args.end(), op->iter_args_.begin(), op->iter_args_.end());

    // Build return vars
    auto break_flag_ret = std::make_shared<Var>(break_flag_name + "_ret", bool_type, span);
    auto loop_var_ret = std::make_shared<Var>(loop_var_name + "_ret", loop_var_type, span);
    std::vector<VarPtr> new_return_vars = {break_flag_ret, loop_var_ret};
    new_return_vars.insert(new_return_vars.end(), op->return_vars_.begin(), op->return_vars_.end());

    // Build while condition: (loop_var < stop) and (not break_flag)
    ExprPtr loop_cond = BuildForToWhileCondition(loop_var_iter, op->stop_, op->step_, span);
    loop_cond = MakeAndExpr(loop_cond, MakeNot(break_flag_iter, span), span);

    // Process the substituted body
    auto trailing_yield = FindTrailingYield(substituted_body);
    auto body_stmts = RemoveTrailingYieldToVec(substituted_body);
    auto orig_yield_values = GetTrailingYieldValues(trailing_yield);

    // Build augmented yield values: [break_flag=False, loop_var+step, ...original_values...]
    ExprPtr loop_var_next = MakeAdd(loop_var_iter, op->step_, span);
    std::vector<ExprPtr> normal_yield_values = {MakeConstBool(false, span), loop_var_next};
    normal_yield_values.insert(normal_yield_values.end(), orig_yield_values.begin(), orig_yield_values.end());

    // Process break (and continue) in the body
    BodyResult processed;
    if (also_has_continue) {
      auto continue_result =
          ProcessBodyForContinue(body_stmts, op->iter_args_, orig_yield_values, name_counter_, span);
      // Rebuild augmented yield values with continue's result
      std::vector<ExprPtr> augmented = {MakeConstBool(false, span), loop_var_next};
      augmented.insert(augmented.end(), continue_result.yield_values.begin(),
                       continue_result.yield_values.end());

      processed =
          ProcessBodyForBreak(continue_result.stmts, 0, new_iter_args, augmented, name_counter_, span);
    } else {
      processed = ProcessBodyForBreak(body_stmts, 0, new_iter_args, normal_yield_values, name_counter_, span);
    }

    // Build final body: processed stmts + trailing yield
    std::vector<StmtPtr> final_stmts(processed.stmts.begin(), processed.stmts.end());
    final_stmts.push_back(std::make_shared<YieldStmt>(std::move(processed.yield_values), span));
    auto final_body = MakeSeq(std::move(final_stmts), span);

    return std::make_shared<WhileStmt>(loop_cond, new_iter_args, final_body, new_return_vars, span);
  }

  // --------------------------------------------------------------------------
  // WhileStmt with only continue
  // --------------------------------------------------------------------------
  StmtPtr LowerWhileWithContinue(const WhileStmtPtr& op, const StmtPtr& body) {
    auto trailing_yield = FindTrailingYield(body);
    auto body_stmts = RemoveTrailingYieldToVec(body);
    auto orig_yield_values = GetTrailingYieldValues(trailing_yield);

    auto result =
        ProcessBodyForContinue(body_stmts, op->iter_args_, orig_yield_values, name_counter_, op->span_);

    std::vector<StmtPtr> final_stmts(result.stmts.begin(), result.stmts.end());
    if (!result.yield_values.empty() || trailing_yield) {
      final_stmts.push_back(std::make_shared<YieldStmt>(std::move(result.yield_values), op->span_));
    }
    auto final_body = MakeSeq(std::move(final_stmts), op->span_);

    return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, final_body, op->return_vars_,
                                       op->span_);
  }

  // --------------------------------------------------------------------------
  // WhileStmt with break (and possibly continue)
  // --------------------------------------------------------------------------
  StmtPtr LowerWhileWithBreak(const WhileStmtPtr& op, const StmtPtr& body, bool also_has_continue) {
    Span span = op->span_;

    // Add break_flag as additional iter_arg
    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    auto break_flag_name = FreshName("__brk_flag");
    auto break_flag_init = MakeConstBool(false, span);
    auto break_flag_iter = std::make_shared<IterArg>(break_flag_name, bool_type, break_flag_init, span);

    std::vector<IterArgPtr> new_iter_args = {break_flag_iter};
    new_iter_args.insert(new_iter_args.end(), op->iter_args_.begin(), op->iter_args_.end());

    auto break_flag_ret = std::make_shared<Var>(break_flag_name + "_ret", bool_type, span);
    std::vector<VarPtr> new_return_vars = {break_flag_ret};
    new_return_vars.insert(new_return_vars.end(), op->return_vars_.begin(), op->return_vars_.end());

    // Augment condition with: and (not break_flag)
    ExprPtr new_condition = MakeAndExpr(op->condition_, MakeNot(break_flag_iter, span), span);

    // Process body
    auto trailing_yield = FindTrailingYield(body);
    auto body_stmts = RemoveTrailingYieldToVec(body);
    auto orig_yield_values = GetTrailingYieldValues(trailing_yield);

    // Build augmented yield values: [break_flag=False, ...original_values...]
    std::vector<ExprPtr> normal_yield_values = {MakeConstBool(false, span)};
    normal_yield_values.insert(normal_yield_values.end(), orig_yield_values.begin(), orig_yield_values.end());

    BodyResult processed;
    if (also_has_continue) {
      auto continue_result =
          ProcessBodyForContinue(body_stmts, op->iter_args_, orig_yield_values, name_counter_, span);
      std::vector<ExprPtr> augmented = {MakeConstBool(false, span)};
      augmented.insert(augmented.end(), continue_result.yield_values.begin(),
                       continue_result.yield_values.end());
      processed =
          ProcessBodyForBreak(continue_result.stmts, 0, new_iter_args, augmented, name_counter_, span);
    } else {
      processed = ProcessBodyForBreak(body_stmts, 0, new_iter_args, normal_yield_values, name_counter_, span);
    }

    std::vector<StmtPtr> final_stmts(processed.stmts.begin(), processed.stmts.end());
    final_stmts.push_back(std::make_shared<YieldStmt>(std::move(processed.yield_values), span));
    auto final_body = MakeSeq(std::move(final_stmts), span);

    return std::make_shared<WhileStmt>(new_condition, new_iter_args, final_body, new_return_vars, span);
  }
};

// ============================================================================
// Pass entry point
// ============================================================================

FunctionPtr TransformLowerBreakContinue(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "LowerBreakContinue cannot run on null function";

  // Only process InCore/AIC/AIV functions
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

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
Pass LowerBreakContinue() {
  return CreateFunctionPass(TransformLowerBreakContinue, "LowerBreakContinue", kLowerBreakContinueProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
