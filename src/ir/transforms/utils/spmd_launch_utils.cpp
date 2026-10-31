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

#include "pypto/ir/transforms/utils/spmd_launch_utils.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {

std::vector<FunctionPtr> DirectCallees(const FunctionPtr& fn, const ProgramPtr& program) {
  CallSubmitCollector collector;
  if (fn->body_) collector.VisitStmt(fn->body_);
  std::vector<FunctionPtr> out;
  std::unordered_set<const Function*> seen;
  auto add = [&](const OpPtr& op) {
    if (!op) return;
    auto callee = program->GetFunction(op->name_);
    if (callee && seen.insert(callee.get()).second) out.push_back(callee);
  };
  for (const auto& call : collector.calls) add(call->op_);
  for (const auto& submit : collector.submits) add(submit->op_);
  return out;
}

bool IsLaunchWrapper(const FunctionPtr& fn) {
  if (!fn) return false;
  return fn->func_type_ == FunctionType::Spmd ||
         (fn->func_type_ == FunctionType::Group && fn->GetAttr<bool>(kAttrSpmdUnwrapped, false));
}

namespace {

/// A Call/Submit found in one function body, plus the width of the innermost
/// enclosing ``SpmdScopeStmt`` when that scope has not been outlined yet.
struct ScopedDispatch {
  CallPtr call;
  SubmitPtr submit;
  ExprPtr scope_core_num;  ///< null when the dispatch is not inside a raw scope
  bool scope_sync_start = false;
  /// True when the innermost carrier between the enclosing scope and this
  /// dispatch is an ``InCoreScopeStmt``. ``OutlineIncoreScopes`` moves such a
  /// body into the launched kernel, so a collective dispatched there runs once
  /// per block like any kernel body — not in the launch-wrapper body, which is
  /// what a scope body without that carrier becomes.
  bool in_incore = false;
};

/// ``CallSubmitCollector`` that also remembers the innermost enclosing SPMD
/// scope. Outlining copies that scope's width onto the dispatch's attrs, so on
/// pre-outline IR — a direct pass call, or a pipeline that runs its consumers
/// before ``OutlineClusterScopes`` — the width exists only on the scope.
/// A stack, not a single value: an inner ``pl.spmd(1)`` must not inherit an
/// outer ``pl.spmd(4)``.
class ScopedDispatchCollector : public IRVisitor {
 public:
  std::vector<ScopedDispatch> dispatches;

 protected:
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    if (!op) return;
    // Everything outside the body evaluates at the caller's level: a launch
    // shape query inside `core_num_` is not itself a dispatch in this launch.
    VisitExpr(op->core_num_);
    VisitScopeAttrs(op);
    scopes_.push_back({op->core_num_, op->sync_start_});
    VisitStmt(op->body_);
    scopes_.pop_back();
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    if (!op) return;
    VisitScopeAttrs(op);
    carriers_.push_back(true);
    VisitStmt(op->body_);
    carriers_.pop_back();
  }

  void VisitStmt_(const ClusterScopeStmtPtr& op) override {
    if (!op) return;
    // A cluster body outlines to a Group wrapper, whose non-launch statements
    // codegen drops, so it is not a kernel carrier.
    VisitScopeAttrs(op);
    carriers_.push_back(false);
    VisitStmt(op->body_);
    carriers_.pop_back();
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op) dispatches.push_back({op, nullptr, ScopeWidth(), ScopeSyncStart(), InInCoreCarrier()});
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    if (op) dispatches.push_back({nullptr, op, ScopeWidth(), ScopeSyncStart(), InInCoreCarrier()});
    IRVisitor::VisitExpr_(op);
  }

 private:
  struct ScopeState {
    ExprPtr core_num;
    bool sync_start;
  };

  [[nodiscard]] ExprPtr ScopeWidth() const { return scopes_.empty() ? nullptr : scopes_.back().core_num; }
  [[nodiscard]] bool ScopeSyncStart() const { return !scopes_.empty() && scopes_.back().sync_start; }
  [[nodiscard]] bool InInCoreCarrier() const { return !carriers_.empty() && carriers_.back(); }

  std::vector<ScopeState> scopes_;
  std::vector<bool> carriers_;
};

}  // namespace

void ForEachSpmdLaunch(const ProgramPtr& program, const std::function<void(const SpmdLaunchSite&)>& visit) {
  if (!program) return;
  // DirectCallees is a full body walk; without the cache, S dispatches to one
  // wrapper would re-walk it S times (pass-complexity.md).
  std::unordered_map<const Function*, std::vector<FunctionPtr>> direct_callee_cache;
  auto direct_callees_cached = [&](const FunctionPtr& f) -> const std::vector<FunctionPtr>& {
    auto [it, inserted] = direct_callee_cache.try_emplace(f.get());
    if (inserted) it->second = DirectCallees(f, program);
    return it->second;
  };
  auto visit_dispatch = [&](const FunctionPtr& caller, const OpPtr& op, const ExprPtr& own_core_num,
                            bool own_sync_start, const ExprPtr& scope_core_num, bool scope_sync_start,
                            bool in_incore_carrier, const CallPtr& dispatch_call) {
    if (!op) return;
    auto callee = program->GetFunction(op->name_);
    // Precedence: the enclosing (not yet outlined) scope first — it is the
    // carrier that outlining turns into the launch, and codegen reads the width
    // it stamps there, so a stale spec already on the child dispatch must not
    // win. Then the dispatch's own spec, then the callee's Function attrs.
    ExprPtr core_num = scope_core_num;
    bool sync_start = scope_sync_start;
    if (!core_num) {
      core_num = own_core_num;
      sync_start = own_sync_start;
    }
    if (!core_num && callee) {
      core_num = callee->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
      sync_start = callee->GetAttr<bool>(kAttrSyncStart, false);
    }
    if (!core_num) return;
    // A legacy Function-level spec predates kAttrSpmdUnwrapped, so a Spmd
    // wrapper or ANY Group launches its own callees there. Gating on the marker
    // would demote an old cluster-unwrapped Group to a launched kernel. A scope
    // width is a real launch spec, so it is not legacy.
    const bool legacy_spec = !own_core_num && !scope_core_num && callee;
    const bool descend =
        legacy_spec ? (callee->func_type_ == FunctionType::Spmd || callee->func_type_ == FunctionType::Group)
                    : IsLaunchWrapper(callee);
    SpmdLaunchSite site{caller, core_num, sync_start, {}};
    site.unoutlined = scope_core_num != nullptr;
    site.unoutlined_kernel_body = in_incore_carrier;
    site.dispatch_call = dispatch_call;
    if (callee) {
      if (descend) {
        // The wrapper body itself is part of the launch too: a dispatch body
        // may hold a statement (e.g. a collective) that no callee owns.
        site.launch_wrapper = callee;
        site.kernels = direct_callees_cached(callee);
      } else {
        site.kernels = std::vector<FunctionPtr>{callee};
      }
    }
    visit(site);
  };
  for (const auto& [gv, fn] : program->functions_) {
    (void)gv;
    if (!fn || !fn->body_) continue;
    ScopedDispatchCollector collector;
    collector.VisitStmt(fn->body_);
    for (const auto& dispatch : collector.dispatches) {
      if (dispatch.call) {
        visit_dispatch(fn, dispatch.call->op_, dispatch.call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr),
                       dispatch.call->GetAttr<bool>(kAttrSyncStart, false), dispatch.scope_core_num,
                       dispatch.scope_sync_start, dispatch.in_incore, dispatch.call);
      } else if (dispatch.submit) {
        visit_dispatch(fn, dispatch.submit->op_, dispatch.submit->core_num_.value_or(nullptr),
                       dispatch.submit->sync_start_, dispatch.scope_core_num, dispatch.scope_sync_start,
                       dispatch.in_incore, nullptr);
      }
    }
  }
}

}  // namespace pypto::ir
