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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_GRAPH_REPLAY_INVARIANT_H_
#define PYPTO_IR_TRANSFORMS_UTILS_GRAPH_REPLAY_INVARIANT_H_

#include <unordered_set>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace graph_replay {

/// True when @p type is `Scalar[TASK_ID]`.
///
/// A TaskId is never a boundary scalar, so it is exempt from every
/// boundary-scalar rule. The recording captures the dependency structure
/// itself — an edge is a node index, not a value to patch — and
/// `graph_boundary_matches` refuses outright any call carrying an explicit
/// dependency (`args.explicit_dep_count() != 0`), so an id produced outside the
/// region can never reach a replay in the first place.
[[nodiscard]] bool IsTaskIdScalar(const TypePtr& type);

/// The values inside one Graph function that are the same on every replay.
///
/// This is a *weaker* property than the one Step A hoists, and the distinction
/// is the whole point of this file:
///
/// | Property | Question it answers | Used for |
/// | -------- | ------------------- | -------- |
/// | hoistable | can the call site recompute this? | moving a value out of the region |
/// | replay-invariant | is this the same on every call? | letting a value stay frozen |
///
/// Every hoistable value is replay-invariant; the converse fails, and each value
/// in the gap is one the pass used to reject even though the runtime would have
/// been happy with it. A frozen value is only wrong when it can *differ* between
/// calls, so replay-invariance is the property the runtime actually needs.
///
/// Three seeds, each with its own reason for being constant across replays:
///
/// * a **literal**, trivially;
/// * a **constant-trip loop induction variable**. Recording walks the loop once
///   and bakes each iteration's literal into that iteration's own node, and
///   constant bounds mean every later call walks the identical sequence. This is
///   also the only thing a tiled kernel's slab offset can be — `i * TILE` cannot
///   be hoisted, because the value does not exist at the call site at all;
/// * **`tensor.dim` of a boundary tensor parameter**. `graph_boundary_matches`
///   compares each boundary tensor's `ndims`, `shapes` and `strides` against the
///   recorded `GraphBoundarySignature` and declines the cached graph on any
///   mismatch, so within one recording a boundary shape cannot change.
///
/// Closed under scalar arithmetic and over names bound to an invariant value.
///
/// **Scalar parameters are deliberately excluded.** The runtime patches a
/// boundary scalar's slot on every call, which is exactly what makes one a legal
/// *task argument* and an illegal *frozen view offset*. Admitting them here
/// would turn the distinction this class exists to draw back into the one it
/// replaces.
class ReplayInvariantSet {
 public:
  /// @param graph_func the Graph function whose body will be collected; its
  ///        tensor parameters seed the `tensor.dim` rule.
  explicit ReplayInvariantSet(const FunctionPtr& graph_func);

  /// Collect the invariant names bound in @p body.
  ///
  /// One forward walk. The body is SSA in definition order, so a name is always
  /// classified before anything that reads it — the same property Step A's own
  /// collector relies on.
  void Collect(const StmtPtr& body);

  /// True when @p expr evaluates to the same value on every replay.
  [[nodiscard]] bool IsInvariant(const ExprPtr& expr) const;

  /// True when @p call is `tensor.dim(<boundary tensor>, <literal axis>)`.
  ///
  /// Exposed so `IsInvariant` can treat such a call as a leaf rather than
  /// rejecting it with every other call; a shape read written inline in an
  /// argument reaches the checker without ever being bound to a name.
  [[nodiscard]] bool IsBoundaryDimRead(const CallPtr& call) const;

 private:
  /// Record a tensor name that aliases a boundary parameter, directly or
  /// through an earlier alias.
  void TrackTensorAlias(const AssignStmtPtr& op);

  /// Tensor parameters, plus every bare alias of one.
  std::unordered_set<const Var*> boundary_tensors_;
  /// Names bound to a replay-invariant value, plus constant-trip loop vars.
  std::unordered_set<const Var*> invariant_;
};

}  // namespace graph_replay
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_GRAPH_REPLAY_INVARIANT_H_
