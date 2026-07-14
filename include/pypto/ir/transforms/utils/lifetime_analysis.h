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
#include <map>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"

namespace pypto {
namespace ir {

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
  /// Stable source-level names of every view/must-alias member collapsed into
  /// this allocation identity. The representative is included for singleton
  /// classes as well, so an exported corpus can reconstruct the normalized
  /// alias classes without depending on IR pointer identity.
  std::vector<std::string> alias_members;
};

enum class AllocationSeparationReason : uint8_t {
  Generic,
  PipelineStage,
  TargetHazard,
  SemanticNoAlias,
};

struct AllocationSeparation {
  size_t first;
  size_t second;
  std::vector<AllocationSeparationReason> reasons;
};

struct PipelineAllocationMember {
  size_t interval_index;
  int32_t stage;
  uint32_t residue;
};

/**
 * @brief One normalized pipeline-buffering group before DSA solving.
 *
 * ``depth`` is the number of distinct source stages. ``effective_depth`` is
 * the capacity-gated number of physical residues; members in different
 * residues receive hard separations, while same-residue chronological reuse
 * is represented by a sparse cost overlay in the standalone document.
 */
struct PipelineAllocationGroup {
  MemorySpace memory_space;
  int32_t group;
  uint64_t slot_size;
  uint32_t depth;
  uint32_t effective_depth;
  std::vector<PipelineAllocationMember> members;
};

/**
 * @brief Per-allocation lifetimes + hard separations for a DSA solver.
 *
 * ``intervals``: one LifetimeInterval per allocation (must-aliases + views already
 * collapsed via ``base_`` identity; opportunistic reuse is the solver's job).
 *
 * ``separations``: typed index pairs into ``intervals`` that must NOT share an
 * address even when lifetime-disjoint. Three sources, the same constraints MemoryReuse
 * honors: (1) pipeline double-buffer clones (same group, different stage) — so
 * stages ping-pong instead of serializing; (2) the Ascend910B load+tpop_from_aic
 * in-place hazard (backend-gated); (3) op-semantic forbid-alias (e.g. tile.sel's
 * mask/tmp must not share the output's buffer). Pipeline separation is reduced
 * to the backend-capacity-gated stage residue count computed by the shared
 * analysis; the standalone solver still enforces the resulting pairs as hard
 * constraints. ``pipeline_groups`` retains the normalized depth/stage/residue
 * relation used to derive those pairs and sparse reuse costs.
 */
struct AllocationPlan {
  std::vector<LifetimeInterval> intervals;
  std::vector<AllocationSeparation> separations;
  std::vector<PipelineAllocationGroup> pipeline_groups;
};

/**
 * @brief Compute the per-allocation lifetime + separation inputs for a DSA solve.
 *
 * Thin, IR-facing entry point over the reuse pass's (phi/loop-aware) lifetime
 * analysis + hazard/forbid-alias collectors, exposed so the DSA adapter can build
 * a DsaProblem without duplicating them.
 *
 * @param func The function to analyze (needed for the backend-gated hazard guard).
 * @param reserved_end_by_space Leading reserved extent per memory space. The
 *        shared packer's whole-space dry run uses it when shedding pipeline depth.
 * @return Intervals (one per allocation) + all separations; empty if no tiles.
 */
[[nodiscard]] AllocationPlan ComputeAllocationPlan(
    const FunctionPtr& func, const std::map<MemorySpace, uint64_t>& reserved_end_by_space = {});

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
