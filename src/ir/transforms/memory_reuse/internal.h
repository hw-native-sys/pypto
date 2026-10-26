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

using HazardInputs = AllocationHazardInputs;
using ForbidAliasMap = AllocationForbidAliasMap;

// Pipeline membership prevents relocation; hardware hazards also forbid a new
// mutable alias. Reuse the constraint collectors on the same frozen snapshot.
struct BranchStorageProtection {
  std::set<const Var*> immovable_bases;
  std::set<const Var*> no_mutable_alias_bases;
};
BranchStorageProtection CollectBranchStorageProtection(const FunctionPtr& function, const StmtPtr& body);

// Packing may eliminate exact whole-root transfers. Subviews retain separate
// byte windows, and caller-owned storage never joins this optional coalescing.
struct TransferCoalescing {
  std::set<const Var*> definitions;
  ForbidAliasMap subrange_separation;
};
TransferCoalescing CollectCoalescibleTransfers(const StmtPtr& body, const LifetimeAnalysisResult& analysis,
                                               const std::vector<VarPtr>& parameters);
StmtPtr RemoveIdentityStorageTransfers(const StmtPtr& body, const std::set<const Var*>& fixed_bases);

// Offsets are bytes relative to the representative's existing MemRef window.
// The representative owns the physical allocation; members retain their views.
struct ReusePlacement {
  VarPtr representative;
  uint64_t byte_offset = 0;
};
using ReuseMap = std::map<VarPtr, ReusePlacement>;

ReuseMap IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const ForbidAliasMap& copy_separation,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>>& pipeline_membership,
    const std::set<const Var*>& pipeline_load_tiles, const std::set<const Var*>& subrange_unsafe_groups,
    const std::map<MemorySpace, uint64_t>& reserved_end_by_space, const std::set<const Var*>& pinned_bases,
    const FunctionPtr& func, std::vector<Diagnostic>* out_hints);

StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const ReuseMap& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups);

// Align carries after packing, then normalize aliases both before and after
// yield repair. Keeping these separate preserves the planner-specific ordering.
StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body);
StmtPtr NormalizeIdentityCopyBuffers(const StmtPtr& body);
// Includes any scratch allocations required to reconcile cyclic carry copies.
// All destinations are settled before packing. An if-only run leaves legacy
// PyPTO loop writeback scheduling to the post-placement call.
StmtPtr FixupYields(const StmtPtr& body, const std::vector<VarPtr>& parameters, bool canonical_loop_copies,
                    bool if_results_only = false);
StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body, const std::set<const Var*>& used_bases);
StmtPtr StripPipelineMembership(const StmtPtr& body);

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_MEMORY_REUSE_INTERNAL_H_
