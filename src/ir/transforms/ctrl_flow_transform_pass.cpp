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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

class BreakContinueDetector : public IRVisitor {
 public:
  bool has_break = false;
  bool has_continue = false;

  void VisitStmt_(const BreakStmtPtr& /*op*/) override { has_break = true; }
  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { has_continue = true; }

  // Don't recurse into nested loops — break/continue in nested loops
  // belong to that inner loop, not the current one.
  void VisitStmt_(const ForStmtPtr& /*op*/) override {}
  void VisitStmt_(const WhileStmtPtr& /*op*/) override {}
};

BreakContinueDetector DetectBreakContinue(const StmtPtr& stmt) {
  BreakContinueDetector detector;
  detector.VisitStmt(stmt);
  return detector;
}

// ============================================================================
// Helpers: flatten / unflatten statement lists
// ============================================================================

/// Flatten a stmt into a vector. If it's a SeqStmts, extract its children;
/// otherwise return a single-element vector.
std::vector<StmtPtr> FlattenToVector(const StmtPtr& stmt) {
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  return {stmt};
}

/// Convert a vector of statements into a single StmtPtr.
StmtPtr MakeStmt(const std::vector<StmtPtr>& stmts, const Span& span) {
  INTERNAL_CHECK(!stmts.empty()) << "Internal error: cannot make statement from empty list";
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return std::make_shared<SeqStmts>(stmts, span);
}

// ============================================================================
// Helpers: check if a statement list ends with Break or Continue
// ============================================================================

/// Check if the last statement in a list IS the target kind (not nested).
bool DirectlyEndsWith(const std::vector<StmtPtr>& stmts, ObjectKind target) {
  if (stmts.empty()) return false;
  return stmts.back()->GetKind() == target;
}

/// Check if a branch (as a stmt) ends with the target.
/// Flattens SeqStmts to check the last element.
bool BranchEndsWith(const StmtPtr& branch, ObjectKind target) {
  auto stmts = FlattenToVector(branch);
  return DirectlyEndsWith(stmts, target);
}

/// Strip trailing target from a stmt list. Returns the list without the last element.
std::vector<StmtPtr> StripTrailingTarget(const std::vector<StmtPtr>& stmts) {
  INTERNAL_CHECK(!stmts.empty()) << "Internal error: cannot strip from empty list";
  return {stmts.begin(), stmts.end() - 1};
}

// ============================================================================
// Core: EliminateTarget
// ============================================================================

/// Append early-exit statements: set break flag (if break) and yield (if iter_args).
void AppendEarlyExit(std::vector<StmtPtr>& stmts, ObjectKind target, const VarPtr& break_var,
                     const std::vector<ExprPtr>& yield_vars, const Span& span) {
  if (target == ObjectKind::BreakStmt) {
    stmts.push_back(std::make_shared<AssignStmt>(break_var, std::make_shared<ConstBool>(true, span), span));
  }
  if (!yield_vars.empty()) {
    stmts.push_back(std::make_shared<YieldStmt>(yield_vars, span));
  }
}

/// Convert a statement vector to a single StmtPtr, using an empty SeqStmts for empty lists.
StmtPtr MakeBranchBody(const std::vector<StmtPtr>& stmts, const Span& span) {
  if (stmts.empty()) {
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  return MakeStmt(stmts, span);
}

/// Eliminate all occurrences of a target (BreakStmt or ContinueStmt) from a
/// statement list by restructuring into if-else.
///
/// @param stmts      Flat list of statements (loop body)
/// @param target     ObjectKind::BreakStmt or ObjectKind::ContinueStmt
/// @param break_var  For Break: the __break flag variable. nullptr for Continue.
/// @param yield_vars For loops with iter_args: expressions to yield on early exit.
///                   Empty if the loop has no iter_args.
/// @param span       Span to use for synthesized nodes
std::vector<StmtPtr> EliminateTarget(const std::vector<StmtPtr>& stmts, ObjectKind target,
                                     const VarPtr& break_var, const std::vector<ExprPtr>& yield_vars,
                                     const Span& span) {
  std::vector<StmtPtr> result;

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& s = stmts[i];
    std::vector<StmtPtr> remaining(stmts.begin() + static_cast<int64_t>(i) + 1, stmts.end());

    // CASE 1: Statement IS the target directly
    if (s->GetKind() == target) {
      AppendEarlyExit(result, target, break_var, yield_vars, span);
      return result;
    }

    // CASE 2-4: Statement is IfStmt with target in a branch
    auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(s);
    if (if_stmt) {
      bool then_ends = BranchEndsWith(if_stmt->then_body_, target);
      bool else_ends = if_stmt->else_body_.has_value() && BranchEndsWith(*if_stmt->else_body_, target);

      // CASE 4: Both branches end with target
      if (then_ends && else_ends) {
        auto then_stmts = StripTrailingTarget(FlattenToVector(if_stmt->then_body_));
        auto else_stmts = StripTrailingTarget(FlattenToVector(*if_stmt->else_body_));

        AppendEarlyExit(then_stmts, target, break_var, yield_vars, span);
        AppendEarlyExit(else_stmts, target, break_var, yield_vars, span);

        result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBranchBody(then_stmts, span),
                                                  std::optional<StmtPtr>(MakeBranchBody(else_stmts, span)),
                                                  if_stmt->return_vars_, span));
        // Remaining is dead code (both branches exit)
        return result;
      }

      // CASE 2: then_body ends with target
      if (then_ends) {
        auto then_stmts = StripTrailingTarget(FlattenToVector(if_stmt->then_body_));
        AppendEarlyExit(then_stmts, target, break_var, yield_vars, span);

        // Absorb remaining into else branch
        std::vector<StmtPtr> else_content;
        if (if_stmt->else_body_.has_value()) {
          auto existing_else = FlattenToVector(*if_stmt->else_body_);
          else_content.insert(else_content.end(), existing_else.begin(), existing_else.end());
        }
        else_content.insert(else_content.end(), remaining.begin(), remaining.end());

        // Recursively eliminate target in the new else content
        auto new_else_stmts = EliminateTarget(else_content, target, break_var, yield_vars, span);

        result.push_back(std::make_shared<IfStmt>(
            if_stmt->condition_, MakeBranchBody(then_stmts, span),
            std::optional<StmtPtr>(MakeBranchBody(new_else_stmts, span)), if_stmt->return_vars_, span));
        return result;
      }

      // CASE 3: else_body ends with target
      if (else_ends) {
        auto else_stmts = StripTrailingTarget(FlattenToVector(*if_stmt->else_body_));
        AppendEarlyExit(else_stmts, target, break_var, yield_vars, span);

        // Absorb remaining into then branch
        auto then_stmts = FlattenToVector(if_stmt->then_body_);
        then_stmts.insert(then_stmts.end(), remaining.begin(), remaining.end());

        // Recursively eliminate target in the new then content
        auto new_then_stmts = EliminateTarget(then_stmts, target, break_var, yield_vars, span);

        result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBranchBody(new_then_stmts, span),
                                                  std::optional<StmtPtr>(MakeBranchBody(else_stmts, span)),
                                                  if_stmt->return_vars_, span));
        return result;
      }
    }

    // DEFAULT: Statement doesn't contain target at top level → keep it
    result.push_back(s);
  }

  return result;
}

// ============================================================================
// Mutator: CtrlFlowTransformMutator
// ============================================================================

class CtrlFlowTransformMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First recurse into body to handle nested loops
    auto new_body = VisitStmt(op->body_);

    // Only process sequential for loops
    if (op->kind_ != ForKind::Sequential) {
      if (new_body.get() != op->body_.get()) {
        return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                         new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                         op->chunk_policy_, op->loop_origin_);
      }
      return op;
    }

    auto detected = DetectBreakContinue(new_body);

    if (!detected.has_break && !detected.has_continue) {
      if (new_body.get() != op->body_.get()) {
        return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                         new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                         op->chunk_policy_, op->loop_origin_);
      }
      return op;
    }

    const Span& span = op->span_;
    auto stmts = FlattenToVector(new_body);

    // Collect current iter_arg exprs for yield insertion on early exit
    std::vector<ExprPtr> yield_vars(op->iter_args_.begin(), op->iter_args_.end());

    // Phase 1: Eliminate continue
    if (detected.has_continue) {
      stmts = EliminateTarget(stmts, ObjectKind::ContinueStmt, nullptr, yield_vars, span);
    }

    // Phase 2: Eliminate break (convert ForStmt to WhileStmt)
    if (detected.has_break) {
      auto break_var = MakeBreakVar(span);

      stmts = EliminateTarget(stmts, ObjectKind::BreakStmt, break_var, yield_vars, span);

      // Append iter_adv guarded by if (!__break)
      auto not_break = std::make_shared<Not>(break_var, DataType::BOOL, span);
      auto loop_var_type = As<ScalarType>(op->loop_var_->GetType());
      auto add_dtype = loop_var_type ? loop_var_type->dtype_ : DataType::INDEX;
      auto iter_adv = std::make_shared<AssignStmt>(
          op->loop_var_, std::make_shared<Add>(op->loop_var_, op->step_, add_dtype, span), span);
      auto guarded_adv =
          std::make_shared<IfStmt>(not_break, iter_adv, std::nullopt, std::vector<VarPtr>{}, span);
      stmts.push_back(guarded_adv);

      StmtPtr while_body = MakeStmt(stmts, span);

      // While condition: Lt(loop_var, stop) && !__break
      auto loop_cond = std::make_shared<Lt>(op->loop_var_, op->stop_, DataType::BOOL, span);
      auto while_cond = std::make_shared<And>(
          loop_cond, std::make_shared<Not>(break_var, DataType::BOOL, span), DataType::BOOL, span);

      auto while_stmt =
          std::make_shared<WhileStmt>(while_cond, op->iter_args_, while_body, op->return_vars_, span);

      // Build init statements: loop_var = start, __break = false
      std::vector<StmtPtr> init_stmts;
      init_stmts.push_back(std::make_shared<AssignStmt>(op->loop_var_, op->start_, span));
      init_stmts.push_back(
          std::make_shared<AssignStmt>(break_var, std::make_shared<ConstBool>(false, span), span));
      init_stmts.push_back(while_stmt);

      return std::make_shared<SeqStmts>(init_stmts, span);
    }

    // Continue-only: return modified ForStmt with transformed body
    return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                     MakeStmt(stmts, span), op->return_vars_, span, op->kind_,
                                     op->chunk_size_, op->chunk_policy_, op->loop_origin_);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // First recurse into body to handle nested loops
    auto new_body = VisitStmt(op->body_);

    auto detected = DetectBreakContinue(new_body);

    if (!detected.has_break && !detected.has_continue) {
      if (new_body.get() != op->body_.get()) {
        return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, new_body, op->return_vars_,
                                           op->span_);
      }
      return op;
    }

    const Span& span = op->span_;
    auto stmts = FlattenToVector(new_body);

    // Collect current iter_arg exprs for yield insertion
    std::vector<ExprPtr> yield_vars(op->iter_args_.begin(), op->iter_args_.end());

    // Phase 1: Eliminate continue
    if (detected.has_continue) {
      stmts = EliminateTarget(stmts, ObjectKind::ContinueStmt, nullptr, yield_vars, span);
    }

    // Phase 2: Eliminate break
    if (detected.has_break) {
      auto break_var = MakeBreakVar(span);

      stmts = EliminateTarget(stmts, ObjectKind::BreakStmt, break_var, yield_vars, span);

      StmtPtr while_body = MakeStmt(stmts, span);

      // Augment while condition: original_cond && !__break
      auto while_cond = std::make_shared<And>(
          op->condition_, std::make_shared<Not>(break_var, DataType::BOOL, span), DataType::BOOL, span);

      // Build: __break = false; while (cond && !__break) { body }
      std::vector<StmtPtr> init_stmts;
      init_stmts.push_back(
          std::make_shared<AssignStmt>(break_var, std::make_shared<ConstBool>(false, span), span));
      init_stmts.push_back(
          std::make_shared<WhileStmt>(while_cond, op->iter_args_, while_body, op->return_vars_, span));
      return std::make_shared<SeqStmts>(init_stmts, span);
    }

    // Continue-only: return modified WhileStmt
    return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, MakeStmt(stmts, span),
                                       op->return_vars_, span);
  }

 private:
  int break_var_counter_ = 0;

  VarPtr MakeBreakVar(const Span& span) {
    std::string name = "__break_" + std::to_string(break_var_counter_++);
    return std::make_shared<Var>(name, std::make_shared<ScalarType>(DataType::BOOL), span);
  }
};

/// Transform a function by eliminating break/continue.
FunctionPtr TransformCtrlFlow(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "CtrlFlowTransform cannot run on null function";

  // Only transform InCore-type functions (InCore, AIC, AIV).
  // Host/Orchestration code can use break/continue natively.
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  CtrlFlowTransformMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;  // No changes
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass CtrlFlowTransform() {
  return CreateFunctionPass(TransformCtrlFlow, "CtrlFlowTransform", kCtrlFlowTransformProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
