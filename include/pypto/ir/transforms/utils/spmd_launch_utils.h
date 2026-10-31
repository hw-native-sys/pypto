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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SPMD_LAUNCH_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SPMD_LAUNCH_UTILS_H_

#include <functional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir {

/// Collects every Call and Submit within one function body, recursing through
/// nested statements but never into other functions.
class CallSubmitCollector : public IRVisitor {
 public:
  std::vector<CallPtr> calls;
  std::vector<SubmitPtr> submits;

  void VisitExpr_(const CallPtr& op) override {
    if (op) calls.push_back(op);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) submits.push_back(op);
    IRVisitor::VisitExpr_(op);
  }
};

/// The distinct functions directly called (Call or Submit) by `fn`, in first-use
/// order. Op calls such as tile.load resolve to no function and are dropped.
std::vector<FunctionPtr> DirectCallees(const FunctionPtr& fn, const ProgramPtr& program);

/// True when dispatching `fn` launches the kernels its body calls rather than
/// `fn` itself: an outlined Spmd wrapper (scope-based `with pl.spmd(...)`), or a
/// Group whose body was a `pl.cluster(): with pl.spmd(...)` region
/// (kAttrSpmdUnwrapped). Anything else, including a mixed-kernel Group reached
/// by `pl.spmd_submit(self.kernel, ..., core_num=N)`, is itself the launched kernel.
bool IsLaunchWrapper(const FunctionPtr& fn);

/// One SPMD launch site: a dispatch that carries a launch width.
struct SpmdLaunchSite {
  FunctionPtr caller;                ///< Function holding the dispatch.
  ExprPtr core_num;                  ///< Block count expression (never null).
  bool sync_start = false;           ///< Whether all blocks must be co-resident.
  std::vector<FunctionPtr> kernels;  ///< Launched kernels (empty if the callee is unknown).
  /// The launch wrapper (see IsLaunchWrapper) whose body the dispatch descends
  /// into, when it launches the wrapper's callees rather than the wrapper
  /// itself; null when the callee is the launched kernel. The wrapper body
  /// runs at the launch site but belongs to no single launched kernel, so a
  /// guard over "what runs inside this launch" must scan it alongside `kernels`.
  FunctionPtr launch_wrapper;
  /// True when the width came from an enclosing `SpmdScopeStmt` that has not
  /// been outlined yet, rather than from the dispatch's own spec.
  bool unoutlined = false;
  /// For an `unoutlined` site: the dispatch sits under an `InCoreScopeStmt`, so
  /// `OutlineIncoreScopes` will move the scope body into the launched kernel
  /// rather than leave it as the launch wrapper outlining would synthesize.
  bool unoutlined_kernel_body = false;
  /// The dispatched Call when the site came from one (null for a Submit). An
  /// op-level dispatch resolves to no callee, leaving `kernels` empty.
  CallPtr dispatch_call;
};

/// Visit every SPMD launch site in `program`.
///
/// A dispatch carries its spec as kAttrCoreNum / kAttrSyncStart Call attrs (an
/// outlined `with/for pl.spmd`, or a `pl.cluster()`-nested `pl.spmd`) or in the
/// first-class Submit fields (`pl.spmd_submit`, or an `as tid` scope). The width
/// is resolved in this order: the innermost enclosing `SpmdScopeStmt` when there
/// is one (it is the launch carrier before `OutlineClusterScopes` copies it onto
/// the dispatch, and codegen reads the width it stamps there), then the
/// dispatch's own spec, then the callee's Function attrs as a fallback for a
/// dispatch that carries none. Such a legacy Function-level spec launches the
/// callees of any Spmd or Group callee. Reporting pre-outline scopes keeps the
/// visit independent of the outlining order. Callee walks are memoized, so the
/// whole visit is O(N).
void ForEachSpmdLaunch(const ProgramPtr& program, const std::function<void(const SpmdLaunchSite&)>& visit);

/// Codegen preflight: every SPMD compute launch must be provably positive in
/// its enclosing arithmetic context. Reuses the legalization pass's analysis.
void VerifySpmdLaunchGuards(const ProgramPtr& program, const FunctionPtr& function);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SPMD_LAUNCH_UTILS_H_
