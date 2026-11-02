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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/buffer_ir_selection.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/reserve_buffer_utils.h"
#include "src/ir/transforms/memory_reuse/internal.h"

namespace pypto {
namespace ir {
namespace {

struct ReusePlan {
  LifetimeAnalysisResult analysis;
  AllocationConstraintAnalysis constraints;
  memory_reuse::ReuseMap placements;
  std::vector<Diagnostic> hints;
  std::set<MemorySpace> overflow_spaces;
};

// Keep the trial and committed plans independent: every retry must rebuild
// provenance, operand restrictions and lifetimes from the rewritten IR.
ReusePlan PlanReuse(const FunctionPtr& func, LifetimeAnalysisResult analysis,
                    const std::map<MemorySpace, uint64_t>& reserved_end_by_space) {
  ReusePlan plan;
  plan.analysis = std::move(analysis);
  plan.constraints = AnalyzeAllocationConstraints(func, plan.analysis, "MemoryReuse");
  plan.placements = memory_reuse::IdentifyReuseOpportunities(
      plan.analysis.lifetimes, plan.constraints.target_hazard_inputs, plan.constraints.forbid_alias,
      plan.analysis.phi_family_ids, plan.analysis.var_sharing_groups, plan.analysis.var_liveness,
      plan.analysis.pipeline_membership, plan.analysis.pipeline_load_tiles,
      plan.analysis.subrange_unsafe_groups, reserved_end_by_space, plan.constraints.declared_allocation_bases,
      func, &plan.hints, &plan.overflow_spaces);
  return plan;
}

/**
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities by walking the full
 * IR tree to compute variable lifetimes, then applying greedy MemRef sharing.
 * Variables that can share memory will point to the same MemRef object.
 * After sharing, redundant alloc operations are removed.
 */
FunctionPtr TransformMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MemoryReusePass cannot run on null function";
  BufferIRFunctionScope buffer_scope(func);

  // Orchestration bodies submit tasks and never hold TileType variables, so
  // there is nothing for memory reuse to do — skip them silently. A Graph body
  // is orchestration too.
  if (IsOrchestrationLike(func->func_type_)) return func;

  // Step 0 (semantic must-alias retarget) now runs in the preceding
  // MaterializeSemanticAliases pass, so the body here is already retargeted.
  StmtPtr new_body = func->body_;

  // Step 1: Compute lifetimes by walking full IR tree
  // Include on-chip parameters: a capacity retry must account for storage
  // already live at function entry, not only AssignStmt-defined allocations.
  auto analysis_result = AnalyzeAllocationLifetimes(func);

  if (analysis_result.lifetimes.empty()) {
    LOG_DEBUG << "No TileType variables found in function '" << func->name_ << "', skipping memory reuse";
    return func;
  }

  auto planned_func = func;
  if (auto* context = PassContext::Current(); context && context->GetEnableSoftwarePipeline()) {
    const auto constraint_analysis = AnalyzeAllocationConstraints(func, analysis_result, "MemoryReuse");
    new_body = memory_reuse::ReuseSoftwarePipelineInputs(new_body, analysis_result,
                                                         constraint_analysis.target_hazard_inputs,
                                                         constraint_analysis.declared_allocation_bases);
    if (new_body != func->body_) {
      auto updated = MutableCopy(func);
      updated->body_ = new_body;
      planned_func = updated;
      analysis_result = AnalyzeAllocationLifetimes(planned_func);
    }
  }

  // Step 2: Identify reuse opportunities.  On Ascend910 split-AIV functions,
  // collect the load + tpop_from_aic hazard inputs so the reuse decision never
  // forms the hazardous in-place sharing (folds in the former
  // LegalizePTOBufferReuse responsibility).  Off-910B the inputs stay empty and
  // reuse behaviour is unchanged.
  // Per-space reserved end (the SpaceFootprint reserved_start for the exact fit check). Only meaningful
  // with a configured backend; empty otherwise ⇒ reserved_start defaults to 0.
  std::map<MemorySpace, uint64_t> reserved_end_by_space;
  if (backend::BackendConfig::IsConfigured()) {
    auto policy = backend::GetBackend()->CreateMemoryAllocatorPolicy();
    if (policy) {
      // Shared with AllocateMemoryAddr — the reserved start is parity-by-construction, not comment-synced.
      const auto resolution = ResolveReserveBufferBases(func, *policy);
      for (const auto& [space, end] : resolution.reserved_end_by_space) reserved_end_by_space[space] = end;
    }
  }
  // Bases the author declared via a one-argument `pl.MemRef(...)`, materialized by
  // InitMemRef as a pinned alloc. Their membership is off-limits to the packer.
  // InitMemRef hoists every alloc to the body's top-level SeqStmts, so this scans
  // that list in place rather than flattening a copy of the whole body. Missing
  // that shape must not fail open: an empty `pinned_bases` silently disables both
  // the packer isolation and the co-liveness check below, handing the author back
  // exactly the coalescing the binding was written to prevent.
  auto plan = PlanReuse(planned_func, std::move(analysis_result), reserved_end_by_space);
  const auto* context = PassContext::Current();
  if ((context == nullptr || !context->GetEnableBufferIR()) && plan.overflow_spaces.count(MemorySpace::Vec)) {
    auto body = memory_reuse::MaterializeForwardedVecSelections(planned_func,
                                                                plan.constraints.declared_allocation_bases);
    if (body != new_body) {
      auto trial = MutableCopy(planned_func);
      trial->body_ = memory_reuse::RefreshStorageTypes(body);
      auto candidate = PlanReuse(trial, AnalyzeAllocationLifetimes(trial), reserved_end_by_space);
      // A bounded, single retry. Keep the original IR if explicit copies do
      // not make Vec fit, or if another space would acquire a new overflow.
      // Independent old-value uses and all original hazard/pinning rules still
      // participate in this exact re-plan; none are waived for capacity.
      if (!candidate.overflow_spaces.count(MemorySpace::Vec) &&
          std::includes(plan.overflow_spaces.begin(), plan.overflow_spaces.end(),
                        candidate.overflow_spaces.begin(), candidate.overflow_spaces.end())) {
        new_body = trial->body_;
        plan = std::move(candidate);
        plan.hints.emplace_back(DiagnosticSeverity::PerfHint, "MemoryReuse", 0,
                                "materialized forwarded Vec selections to fit on-chip capacity; "
                                "unchanged branches now copy into independent output storage",
                                func->span_);
      }
    }
  }
  analysis_result = std::move(plan.analysis);
  const auto& reuse_map = plan.placements;
  // Surface capacity-forced pipeline-depth reductions (perf hints) and legacy-fallback overflows
  // (warnings) through the unified diagnostic channel → perf_hints.log / stderr.
  if (!plan.hints.empty()) EmitDiagnostics(plan.hints, "MemoryReuse");

  // Step 3: Apply MemRef sharing (skip if no reuse candidates)
  if (!reuse_map.empty()) {
    new_body = memory_reuse::ApplyMemRefSharing(new_body, reuse_map, analysis_result.var_sharing_groups);

    // Step 3.5: Re-align loop-carried iter_arg/return_var MemRefs to their
    // (now-reused) initValue, top-down.  Reuse retypes producer/init AssignStmt
    // vars but leaves loop-carry nodes stale; for nested pipelined accumulators
    // that split chain would otherwise force YieldFixupMutator to emit invalid
    // acc->acc tile.move ops (#1352).
    const auto* context = PassContext::Current();
    if (context != nullptr && context->GetEnableBufferIR()) {
      new_body = memory_reuse::RefreshStorageTypes(new_body);
    } else {
      new_body = memory_reuse::AlignLoopCarriesToInit(new_body);
    }
  }

  // Step 3.75: Coalesce peeled loop-carried accumulator if-phis so YieldFixupMutator
  // does not reconcile them with an illegal Acc->Acc tile.move.  See
  // TopDownRetargeter::CoalesceAccumulatorIfPhis.  Must run after all ForStmt-carry
  // coalescing (Steps 0/3/3.5) so the accumulator branch is on its final buffer, and
  // before YieldFixupMutator (Step 4) so it observes one buffer per phi.  A no-op when
  // no accumulator if-phi exists (e.g. non-pipelined kernels).
  new_body = memory_reuse::CoalesceAccumulatorIfPhis(new_body);

  // Step 3.9 / 4.5: Reconcile bare-Var SSA identity copies whose buffers diverged
  // from the value they rename.  No-op when no mismatch exists.
  //
  // This runs on both sides of YieldFixup because both sides create such a
  // divergence, from opposite directions:
  //
  //  - before, because Step 3.75's accumulator-if-phi retarget leaves a rename
  //    stranded on the pre-coalesce buffer, and because a carry rename such as
  //    `lag2 = lag1` only lands on lag1's buffer here. YieldFixup has to see the
  //    settled buffer to order the carry writebacks against each other: against
  //    the pre-normalization buffers every rename still sits on a buffer of its
  //    own, so no conflict is visible and a shift register silently collapses
  //    ([#2481]).
  //  - after, because YieldFixup refreshes an If result's representative MemRef
  //    after reuse. Its aliases must follow that hint as well; the descriptor
  //    still selects its actual storage at runtime.
  //
  // The mutator is idempotent, so the second run is a no-op whenever the first
  // already settled everything.
  new_body = memory_reuse::NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);

  // Step 4: Reconcile fixed carry storage and refresh native If result hints
  const auto* ctx = PassContext::Current();
  // A carry-copy cycle needs a scratch buffer; its alloc belongs on the body head
  // like every other allocation, and Step 5 below keeps it because it is in use.
  new_body = memory_reuse::FixupYields(
      new_body, func->params_, /*canonical_branch_storage=*/ctx != nullptr && ctx->GetEnableBufferIR());

  // Step 4.5: the second half of the bracket described at Step 3.9.
  new_body = memory_reuse::NormalizeIdentityCopyBuffers(new_body);

  // Step 5: Remove alloc statements for MemRefs no longer in use
  auto used_bases = memref_collectors::CollectUsedBasePtrs(new_body);
  new_body = memory_reuse::RemoveUnusedAllocStatements(new_body, used_bases);

  // Step 6: Strip the now-consumed pipeline_membership attr so it does not ride
  // downstream into later passes / codegen. It was only needed to carry stage
  // identity from LowerPipelineLoops to the reuse decision above.
  new_body = memory_reuse::StripPipelineMembership(new_body);

  auto result = std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_,
                                                 func->requires_runtime_binding_, func->ir_stage_);
  return result;
}

}  // namespace

namespace pass {
Pass MaterializeSemanticAliases() {
  return CreateFunctionPass(memory_reuse::TransformMaterializeSemanticAliases, "MaterializeSemanticAliases",
                            kMaterializeSemanticAliasesProperties);
}
Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, "MemoryReuse", kMemoryReuseProperties); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
