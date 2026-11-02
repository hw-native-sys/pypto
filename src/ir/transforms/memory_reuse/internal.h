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

#ifndef SRC_IR_TRANSFORMS_MEMORY_REUSE_INTERNAL_H_
#define SRC_IR_TRANSFORMS_MEMORY_REUSE_INTERNAL_H_

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

// Shared by alias analysis and yield repair: unknown bounds conservatively
// may execute; after_first asks whether another iteration may run.
bool LoopMayExecute(const StmtPtr& statement, bool after_first);

// Private stage boundaries. The pass entry orders packing and writeback repair;
// semantic_aliases.cpp also uses writeback repair for planners that skip reuse.
// Each call owns its analysis/mutator state; no state survives a function pass.
FunctionPtr TransformMaterializeSemanticAliases(const FunctionPtr& func);
StmtPtr RefreshStorageTypes(const StmtPtr& body);
StmtPtr CoalesceAccumulatorIfPhis(const StmtPtr& body);

// Function-local plan for descriptor selection. Keys are original Tile Vars;
// indices also maps replacement Tile Vars to the same scalar slot expression.
struct StorageSelectionPlan {
  std::map<const Var*, VarPtr> tiles;
  std::map<const Var*, ExprPtr> indices;
  std::vector<StmtPtr> allocations;
};
StmtPtr MaterializeStorageSelections(const StmtPtr& body, const std::vector<VarPtr>& parameters,
                                     StorageSelectionPlan plan);

using HazardInputs = AllocationHazardInputs;
using ForbidAliasMap = AllocationForbidAliasMap;

// Offsets are bytes relative to the representative's existing MemRef window.
// The representative owns the physical allocation; members retain their views.
struct ReusePlacement {
  VarPtr representative;
  uint64_t byte_offset = 0;
  // Forwarding a logical pipeline input also selects its slot. Ordinary byte
  // reuse preserves the destination's own pool/slot identity.
  bool forward_slot = false;
};
using ReuseMap = std::map<VarPtr, ReusePlacement>;

ReuseMap IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const std::map<const Var*, std::set<int>>& phi_family_ids,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::pair<int, int>>& var_liveness,
    const std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>>& pipeline_membership,
    const std::set<const Var*>& pipeline_load_tiles, const std::set<const Var*>& subrange_unsafe_groups,
    const std::map<MemorySpace, uint64_t>& reserved_end_by_space, const std::set<const Var*>& pinned_bases,
    const FunctionPtr& func, std::vector<Diagnostic>* out_hints);

StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const ReuseMap& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups);

// Forward eligible computed results into their last-used software-pipeline input slot.
StmtPtr ReuseSoftwarePipelineInputs(const StmtPtr& body, const LifetimeAnalysisResult& lifetimes,
                                    const HazardInputs& hazard, const std::set<const Var*>& declared_bases);

// Align carries after packing, then normalize aliases both before and after
// yield repair. Keeping these separate preserves the planner-specific ordering.
StmtPtr AlignLoopCarriesToInit(const StmtPtr& body);
StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body);
StmtPtr NormalizeIdentityCopyBuffers(const StmtPtr& body);
// Includes any scratch allocations required to reconcile cyclic carry copies.
StmtPtr FixupYields(const StmtPtr& body, const std::vector<VarPtr>& parameters, bool fixup_if_stmts,
                    bool canonical_branch_storage = false);
StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body, const std::set<const Var*>& used_bases);
StmtPtr StripPipelineMembership(const StmtPtr& body);

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_MEMORY_REUSE_INTERNAL_H_
