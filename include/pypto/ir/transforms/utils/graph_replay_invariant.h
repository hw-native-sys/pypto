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

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
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
/// `graph_begin_inner` refuses any call carrying an explicit
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
/// A derivable scalar may be hoisted and refreshed per call. A value from
/// a constant-trip loop can stay frozen even though the caller cannot derive
/// it. The two analyses keep those cases separate.
///
/// Three seeds, each with its own reason for being constant across replays:
///
/// * a **literal**, trivially;
/// * a **constant-trip loop induction variable**. Recording walks the loop once
///   and bakes each iteration's literal into that iteration's own node, and
///   constant bounds mean every later call walks the identical sequence. This is
///   also the only thing a tiled kernel's slab offset can be — `i * TILE` cannot
///   be hoisted, because the value does not exist at the call site at all;
/// * **`tensor.dim` of a boundary tensor parameter**. The runtime matches
///   its rank, shape and strides against the recorded GraphBoundaryMatchInfo,
///   so within one recording the boundary shape cannot change.
///
/// Metadata-only slice/reshape/view chains inherit the shape fact only after
/// their source, shape/offset operands and result extents are invariant.
/// Closed under scalar arithmetic and over names bound to an invariant value.
///
/// Scalar parameters are mutable by default. The runtime patches a boundary
/// scalar's slot on every call, which is exactly what makes one a legal
/// task argument and an illegal frozen view offset. Call-site metadata proofs
/// seed topology checks separately. Values consumed by graph-body codegen
/// are also part of the GraphConfig key.
class ReplayInvariantSet {
 public:
  /// @param graph_func the Graph function whose body will be collected; its
  ///        tensor parameters seed the `tensor.dim` rule.
  // Extra scalar seeds are compiler-internal proofs retained across hoisting.
  // Only actual scalar parameters in this identity set qualify; callers must
  // prove their original defining expressions invariant before rewriting.
  explicit ReplayInvariantSet(const FunctionPtr& graph_func,
                              const std::unordered_set<const Var*>& proven_fixed_scalar_params = {});

  /// Seed only free shape symbols actually bound from selected boundary descriptors.
  /// Scalar parameters and local definitions shadow descriptor symbols by emitted name.
  void SeedBoundaryDimensionSymbols(const FunctionPtr& func);

  /// Collect the invariant names bound in @p body.
  ///
  /// One forward walk. The body is SSA in definition order, so a name is always
  /// classified before anything that reads it — the same property Step A's own
  /// collector relies on.
  void Collect(const StmtPtr& body, bool include_constant_trip_indices = true);

  /// True when @p expr evaluates to the same value on every replay.
  [[nodiscard]] bool IsInvariant(const ExprPtr& expr) const;

  /// True for `tensor.dim` of a boundary tensor or a proven metadata-only
  /// view of one, with a literal axis.
  ///
  /// Exposed so `IsInvariant` can treat such a call as a leaf rather than
  /// rejecting it with every other call; a shape read written inline in an
  /// argument reaches the checker without ever being bound to a name.
  [[nodiscard]] bool IsBoundaryDimRead(const CallPtr& call) const;

  /// Re-prove scalar origins from every current Graph Call/Submit.
  /// Only actual tensor arguments at each site seed shape metadata. GraphConfig
  /// distinguishes the resulting scalar values used by the Graph body.
  using FixedScalarIndex = std::unordered_map<const Function*, std::unordered_set<const Var*>>;
  [[nodiscard]] static FixedScalarIndex IndexFixedScalarsFromCallSites(const ProgramPtr& program);
  [[nodiscard]] static const std::unordered_set<const Var*>& FixedScalarsForGraph(
      const FixedScalarIndex& index, const FunctionPtr& graph);

 private:
  /// Record a bare alias or a metadata-only view with a proven invariant
  /// source, complete operand list and result shape.
  void TrackTensorAlias(const AssignStmtPtr& op);

  // The empty set is a locally proven invariant. Nonempty sets record the
  // exact tensor arguments a caller must forward in a Graph's layout key.
  // A disengaged optional is unknown.
  using Roots = std::vector<const Var*>;
  using Proof = std::optional<Roots>;

  [[nodiscard]] Proof ProveExpr(const ExprPtr& expr) const;
  [[nodiscard]] Proof ProveDimRead(const CallPtr& call) const;

  /// Physical extent facts only: never descriptor, address or tensor data facts.
  std::unordered_map<const Var*, std::vector<Proof>> fixed_local_axes_;
  std::unordered_map<const Var*, Roots> boundary_tensors_;
  std::unordered_map<const Var*, Roots> invariant_;
};

}  // namespace graph_replay
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_GRAPH_REPLAY_INVARIANT_H_
