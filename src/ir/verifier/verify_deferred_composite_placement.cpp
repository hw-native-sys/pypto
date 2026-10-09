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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/deferred_wait_contract.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Deferred-placement check: ``defer=True`` mesh composites must sit in a task-level
/// ``pl.at(CORE_GROUP)`` body — the same placement OutlineIncoreScopes later
/// stamps with ``deferred_completion_waiter``. Without this check, a bare
/// Orchestration call can carry ``defer=True`` and be skipped by
/// LowerCompositeOps on non-InCore rails, silently ignoring the flag.
///
/// Lifecycle: listed in both ``GetStructuralProperties()`` and
/// ``GetVerifiedProperties()`` so Default Basic verifies it at pipeline input,
/// and ``VerificationInstrument`` re-checks after every pass. After
/// OutlineIncoreScopes the Call lives in a stamped InCore/AIV function body
/// (no surrounding ``InCoreScopeStmt``); that shape is accepted via the stamp.
class DeferredCompositePlacementChecker : public IRVisitor {
 public:
  DeferredCompositePlacementChecker(std::vector<Diagnostic>& diagnostics, std::string func_name,
                                    FunctionType func_type, bool stamped_deferred_waiter)
      : diagnostics_(diagnostics),
        func_name_(std::move(func_name)),
        func_type_(func_type),
        stamped_deferred_waiter_(stamped_deferred_waiter) {}

  void VisitExpr_(const CallPtr& op) override {
    if (outline_utils::IsDeferredCompositeCall(op)) {
      CheckDeferredComposite(op);
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    if (!op) return;
    const bool prev_inside = inside_incore_scope_;
    const bool prev_task_level = incore_is_task_level_;
    const bool prev_early = incore_allow_early_resolve_;
    const bool prev_pred = incore_has_predicate_;

    inside_incore_scope_ = true;
    // Matches OutlineIncoreScopes: a waiter nested under another launch scope
    // (spmd / cluster / …) is illegal because the outer launch owns dispatch
    // predicate and early-resolve semantics.
    incore_is_task_level_ = (launch_scope_nesting_ == 0);
    incore_allow_early_resolve_ = op->GetAttr<bool>("allow_early_resolve", false);
    incore_has_predicate_ = op->GetAttr<ExprPtr>(kAttrPredicate, nullptr) != nullptr;

    IRVisitor::VisitStmt_(op);

    inside_incore_scope_ = prev_inside;
    incore_is_task_level_ = prev_task_level;
    incore_allow_early_resolve_ = prev_early;
    incore_has_predicate_ = prev_pred;
  }

  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitLaunchScope(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitLaunchScope(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitLaunchScope(op); }
  void VisitStmt_(const GraphScopeStmtPtr& op) override { VisitLaunchScope(op); }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { VisitLaunchScope(op); }
  // RuntimeScope (``pl.manual_scope`` / ``pl.scope``) and CommDomainScope are
  // not outline targets and do not flip OutlineIncoreScopes'
  // ``inside_nested_scope_body_`` — leave nesting unchanged so a legal
  // ``manual_scope`` + task-level ``pl.at`` placement still passes.

  void VisitStmt_(const IfStmtPtr& op) override { VisitControlFlow(op); }
  void VisitStmt_(const ForStmtPtr& op) override { VisitControlFlow(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { VisitControlFlow(op); }

 private:
  template <typename ScopeT>
  void VisitLaunchScope(const std::shared_ptr<const ScopeT>& op) {
    if (!op) return;
    ++launch_scope_nesting_;
    IRVisitor::VisitStmt_(op);
    --launch_scope_nesting_;
  }

  template <typename StmtT>
  void VisitControlFlow(const std::shared_ptr<const StmtT>& op) {
    if (!op) return;
    // Only control flow *inside* the deferred task body is illegal for split.
    // Orchestration may wrap whole ``pl.at`` tasks in if/for (``deps=[tid]``
    // across branches) — that must remain legal. Post-outline stamped waiter
    // functions have no surrounding InCoreScopeStmt, so treat their whole body
    // as the task body.
    const bool count_nesting = inside_incore_scope_ || stamped_deferred_waiter_;
    if (count_nesting) {
      ++control_flow_nesting_;
      IRVisitor::VisitStmt_(op);
      --control_flow_nesting_;
      return;
    }
    IRVisitor::VisitStmt_(op);
  }

  void CheckDeferredComposite(const CallPtr& call) {
    const std::string op_name = call->op_ ? call->op_->name_ : "pld.tensor.*";

    // Post-outline: OutlineIncoreScopes moved the Call into a stamped InCore /
    // AIV waiter function. Accept that shape so VerificationInstrument does not
    // false-fail between outline and LowerCompositeOps — but still reject
    // nesting under if/for/while (SplitDeferredCompositeKernels cannot peel a
    // compound wait region into registration-only).
    const bool stamped_ok = !inside_incore_scope_ && stamped_deferred_waiter_ &&
                            (func_type_ == FunctionType::InCore || func_type_ == FunctionType::AIV) &&
                            launch_scope_nesting_ == 0;
    if (stamped_ok) {
      if (control_flow_nesting_ != 0) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "DeferredCompositePlacementValid", /*error_code=*/4,
            op_name + "(..., defer=True) in function '" + func_name_ +
                "' cannot sit under if/for/while; keep it as the straight-line body of its "
                "own task-level pl.at(CORE_GROUP) so push / wait / epilogue can be split.",
            call->span_);
      }
      return;
    }

    if (!inside_incore_scope_) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "DeferredCompositePlacementValid", /*error_code=*/0,
          op_name + "(..., defer=True) in function '" + func_name_ +
              "' must be the body of a task-level `with pl.at(level=pl.Level.CORE_GROUP)` "
              "scope. Bare Orchestration / HOST / InCore-function placement cannot register "
              "deferred completion; capture the TaskId and gate consumers with `deps=[tid]`.",
          call->span_);
      return;
    }
    if (!incore_is_task_level_) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "DeferredCompositePlacementValid", /*error_code=*/1,
          op_name + "(..., defer=True) in function '" + func_name_ +
              "' must be in a task-level pl.at(CORE_GROUP) scope; nesting under pl.spmd or "
              "another task-launch scope is unsupported because the outer launch owns "
              "dispatch predicate and early-resolve semantics.",
          call->span_);
      return;
    }
    if (control_flow_nesting_ != 0) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "DeferredCompositePlacementValid", /*error_code=*/4,
          op_name + "(..., defer=True) in function '" + func_name_ +
              "' cannot sit under if/for/while inside pl.at; hoist it to its own straight-line "
              "pl.at(CORE_GROUP) task (use deps=[tid] across branches instead).",
          call->span_);
      return;
    }
    if (incore_allow_early_resolve_) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "DeferredCompositePlacementValid", /*error_code=*/2,
          "pl.at(...) containing " + op_name + "(..., defer=True) in function '" + func_name_ +
              "' cannot use allow_early_resolve=True; the waiter's TaskId must remain "
              "unresolved until its registered signal condition is satisfied.",
          call->span_);
      return;
    }
    if (incore_has_predicate_) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "DeferredCompositePlacementValid",
                                /*error_code=*/3,
                                op_name + "(..., defer=True) task in function '" + func_name_ +
                                    "' cannot use a dispatch predicate; every submitted waiter must register "
                                    "its completion condition.",
                                call->span_);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  FunctionType func_type_;
  bool stamped_deferred_waiter_ = false;
  int launch_scope_nesting_ = 0;
  int control_flow_nesting_ = 0;
  bool inside_incore_scope_ = false;
  bool incore_is_task_level_ = false;
  bool incore_allow_early_resolve_ = false;
  bool incore_has_predicate_ = false;
};

class DeferredCompositePlacementValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "DeferredCompositePlacementValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      DeferredCompositePlacementChecker checker(diagnostics, func->name_, func->func_type_,
                                                func->GetAttr<bool>(kAttrDeferredCompletionWaiter, false));
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateDeferredCompositePlacementValidPropertyVerifier() {
  return std::make_shared<DeferredCompositePlacementValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
