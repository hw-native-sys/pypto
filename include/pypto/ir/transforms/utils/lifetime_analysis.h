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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"

namespace pypto {
namespace ir {

/**
 * @brief Lifetime of one variable within an allocation sharing group.
 *
 * PyPTO orders reads before writes at the same statement point, so a variable
 * whose last use is point ``p`` may share storage with a variable defined at
 * point ``p``. The DSA adapter preserves that ordering when it converts these
 * points to half-open standalone-solver intervals.
 */
struct VariableLifetime {
  int def_point;
  int last_use_point;
};

/**
 * @brief Lifetime interval for one allocation (a base-group of TileType vars).
 *
 * One interval per physical allocation: views and semantic must-aliases that
 * share a ``base_`` Ptr are collapsed into a single interval whose [def, last_use]
 * is the union over the group's members (topological order).  This is the unit
 * the reuse packer — and the DSA adapter — treats as one buffer.
 */
struct LifetimeInterval {
  VarPtr variable;           ///< Representative variable of the sharing group.
  int def_point;             ///< Group's earliest definition point (topological order).
  int last_use_point;        ///< Group's latest last-use point (topological order).
  MemorySpace memory_space;  ///< Memory space (== DSA pool).
  uint64_t size;             ///< Slot size in bytes (largest member).
  /// Per-variable ranges before the conservative group hull is formed. This
  /// lets a structured DSA solver preserve holes in a semantics-required
  /// allocation identity instead of treating the whole hull as continuously
  /// live. MemoryReuse continues to use def_point/last_use_point as its fast
  /// group-level bound.
  std::vector<VariableLifetime> live_ranges;
};

/**
 * @brief Per-allocation lifetimes + hard separations for a DSA solver.
 *
 * ``intervals``: one LifetimeInterval per allocation (must-aliases + views already
 * collapsed via ``base_`` identity; opportunistic reuse is the solver's job).
 *
 * ``separations``: index pairs into ``intervals`` that must NOT share an address
 * even when lifetime-disjoint.  Three sources, the same constraints MemoryReuse
 * honors: (1) pipeline double-buffer clones (same group, different stage) — so
 * stages ping-pong instead of serializing; (2) the Ascend910B load+tpop_from_aic
 * in-place hazard (backend-gated); (3) op-semantic forbid-alias (e.g. tile.sel's
 * mask/tmp must not share the output's buffer). Pipeline separation is reduced
 * to the backend-capacity-gated stage residue count computed by the shared
 * analysis; the standalone solver still enforces the resulting pairs as hard
 * constraints.
 */
struct AllocationPlan {
  std::vector<LifetimeInterval> intervals;
  std::vector<std::pair<size_t, size_t>> separations;
};

/**
 * @brief Compute the per-allocation lifetime + separation inputs for a DSA solve.
 *
 * Thin, IR-facing entry point over the reuse pass's (phi/loop-aware) lifetime
 * analysis + hazard/forbid-alias collectors, exposed so the DSA adapter can build
 * a DsaProblem without duplicating them.
 *
 * @param func The function to analyze (needed for the backend-gated hazard guard).
 * @return Intervals (one per allocation) + all separations; empty if no tiles.
 */
[[nodiscard]] AllocationPlan ComputeAllocationPlan(const FunctionPtr& func);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
