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

#ifndef PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
#define PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_

#include "pypto/ir/transforms/ir_property.h"

namespace pypto {
namespace ir {
namespace pass {

/// @brief Central registry of PassProperties for all built-in passes.
///
/// Each constant declares the required, produced, and invalidated IRProperties
/// for one pass.  Using `inline const` (not `constexpr`) because
/// IRPropertySet's initializer_list constructor is not constexpr in C++17.

// -- Inline-function elimination pass (runs first, before everything) ---------

inline const PassProperties kInlineFunctionsProperties{.produced = {IRProperty::InlineFunctionsEliminated}};

// -- SynthesizeAllReduceSignals and MaterializeCommDomainScopes passes (run
//    late in the pipeline, after phase-fence expansion and immediately before
//    LowerHostTensorCollectives).
//    Nothing between InlineFunctions and here removes the host_orch
//    alloc/window/dispatch/allreduce chain (host_orch is never tile-lowered), so
//    alloc/view/dispatch/allreduce sites are still discoverable. The synthesizer
//    first normalizes host allreduce calls to explicit-signal IR.
//    Traces pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r),
//    materialises WindowBuffer back-references on every DistributedTensorType view,
//    and wraps the host_orch body in nested CommDomainScopeStmts (one per
//    inferred comm domain).

inline const PassProperties kSynthesizeAllReduceSignalsProperties{};

inline const PassProperties kMaterializeCommDomainScopesProperties{
    .produced = {IRProperty::CommDomainScopesMaterialized}};

inline const PassProperties kLowerHostTensorCollectivesProperties{
    .required = {IRProperty::CommDomainScopesMaterialized},
    .produced = {IRProperty::CommDomainScopesMaterialized}};

inline const PassProperties kMaterializeDistTensorCtxProperties{
    .required = {IRProperty::CommDomainScopesMaterialized},
    .produced = {IRProperty::CommDomainScopesMaterialized}};

// -- MaterializeRuntimeScopes pass (runs last, after the final Simplify) ------
//    Inserts explicit AUTO RuntimeScopeStmt nodes for the orchestration function
//    body and for/if bodies so codegen emits PTO2_SCOPE 1:1 from the IR.
inline const PassProperties kMaterializeRuntimeScopesProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::CallDirectionsResolved},
    .produced = {IRProperty::RuntimeScopesMaterialized}};

// -- ClassifyIterArgCarry pass (runs last, after MaterializeRuntimeScopes) ----
//    Classifies each Orchestration ForStmt iter_arg as a trivial alias or a
//    materialised rebind carry and sizes manual-scope TaskId array carries,
//    stamping the plan onto ForStmt::attrs_. Runs on the final IR shape so the
//    stamped plan is exactly what the orchestration codegen lowers.
inline const PassProperties kClassifyIterArgCarryProperties{
    .required = {IRProperty::CallDirectionsResolved, IRProperty::RuntimeScopesMaterialized},
    .produced = {IRProperty::IterArgCarryClassified, IRProperty::RuntimeScopesMaterialized}};

// -- Loop unrolling pass (runs before SSA) ------------------------------------

inline const PassProperties kUnrollLoopsProperties{.produced = {IRProperty::UnrollResolved}};

// -- Control flow structuring pass (runs before SSA, after unrolling) ---------

inline const PassProperties kCtrlFlowTransformProperties{.produced = {IRProperty::StructuredCtrlFlow}};

// -- SSA conversion pass ------------------------------------------------------

inline const PassProperties kConvertToSSAProperties{.produced = {IRProperty::SSAForm},
                                                    .invalidated = {IRProperty::NormalizedStmtStructure}};

// -- Expression / statement normalisation passes ------------------------------

inline const PassProperties kFlattenCallExprProperties{
    .required = {IRProperty::SSAForm, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::NoNestedCalls, IRProperty::NormalizedStmtStructure}};

inline const PassProperties kNormalizeStmtStructureProperties{
    .produced = {IRProperty::NormalizedStmtStructure}};

// -- Simplification pass ------------------------------------------------------

inline const PassProperties kSimplifyProperties{};

// -- Composite op lowering pass (tile.sin / tile.cos / InCore allreduce -> primitives, etc.) -----
//
// LowerCompositeOps decomposes composite tile/distributed ops into primitive
// ops. Today it handles tile.sin / tile.cos (Cody-Waite range reduction +
// degree-9 Horner polynomial) and explicit-signal InCore pld.tensor.allreduce;
// host-level allreduce is skipped and lowered later by LowerHostTensorCollectives.
// Future composite ops add a rule to the file-local dispatch table in
// lower_composite_ops_pass.cpp. The pass operates within existing op
// vocabularies, so it neither requires nor produces nor invalidates any
// IRProperty.

inline const PassProperties kLowerCompositeOpsProperties{};

// -- Outlining pass -----------------------------------------------------------

// OutlineIncoreScopes opens the AivSplitValid verification window: it preserves
// the first-class SplitAivScopeStmt regions inside each outlined InCore function,
// so the structural region verifier can run from here until LowerAutoVectorSplit
// erases the node (pass 21).
inline const PassProperties kOutlineIncoreScopesProperties{
    .required = {IRProperty::SSAForm},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::AivSplitValid}};

// -- Cluster outlining pass ---------------------------------------------------

inline const PassProperties kOutlineClusterScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::ClusterOutlined}};

// -- Hierarchy outlining pass -------------------------------------------------

inline const PassProperties kOutlineHierarchyScopesProperties{
    .required = {IRProperty::SSAForm},
    .produced = {IRProperty::SSAForm, IRProperty::HierarchyOutlined,
                 IRProperty::OrchestrationReferencesResolved}};

// -- Tensor-to-tile conversion pass ------------------------------------------

// Re-opens the AivSplitValid window. OutlineIncoreScopes establishes the property
// while the AIV-split boundary is still `tensor.aiv_shard` / `tensor.aic_gather` —
// a TensorType carries no memory space, so the verifier's boundary memory contract
// check (d) is necessarily skipped there. This pass rewrites those ops to their
// tile form and attaches the declared boundary memory, which is exactly what
// check (d) inspects, so it invalidates and re-produces the property to force a
// second verification at a point where the memory sides are observable.
inline const PassProperties kConvertTensorToTileOpsProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::NormalizedStmtStructure,
                 IRProperty::AivSplitValid},
    .invalidated = {IRProperty::AivSplitValid}};

// -- Orchestration tensor optimization pass -----------------------------------

inline const PassProperties kOptimizeOrchTensorsProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps},
    .produced = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps}};

// -- Tile ND-to-2D flattening pass --------------------------------------------

inline const PassProperties kFlattenTileNdTo2DProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure}};

// -- Auto L0 matmul tiling pass -----------------------------------------------
//
// Property-preserving rewrite: replaces ``tile.matmul[_acc]`` over Mat-resident
// operands with a small loop nest of L0-sized matmuls.  Runs between
// FlattenTileNdTo2D and InferTileMemorySpace, so all tile ops are already 2D
// and memory spaces have not yet been inferred.  No properties are produced
// or invalidated beyond what the input IR already guarantees.

inline const PassProperties kAutoTileMatmulL0Properties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure}};

// -- Canonicalize Mat-resident tile.slice into tile.extract -------------------
// Runs right after AutoTileMatmulL0, before InferTileMemorySpace.  A
// property-preserving rewrite: it only folds Mat-resident tile.slice ops into
// their tile.extract / tile.matmul consumers, so it requires and produces the
// same property set as AutoTileMatmulL0.

inline const PassProperties kCanonicalizeTileSliceProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure}};

// -- Tile memory space inference pass -----------------------------------------

// Also re-verifies AivSplitValid (same rationale as ConvertTensorToTileOps): this
// pass is what finally resolves every tile memory space, so an AIV-split boundary
// whose operand space was still unresolved at pass 10 — and therefore skipped by
// check (d) — becomes observable here, the last verification point before
// LowerAutoVectorSplit erases the region node.
inline const PassProperties kInferTileMemorySpaceProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure,
                 IRProperty::AivSplitValid},
    .invalidated = {IRProperty::AivSplitValid}};

// -- Materialize tensor strides pass (RFC #1300 §2.4) ------------------------

inline const PassProperties kMaterializeTensorStridesProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure,
                 IRProperty::TensorViewCanonical}};

// -- Resolve backend op layouts pass ------------------------------------------

// The pass self-normalizes statement structure before returning, so
// NormalizedStmtStructure is preserved across the pass and the pipeline
// no longer needs a standalone NormalizeStmtStructure call after it.
inline const PassProperties kResolveBackendOpLayoutsProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure}};

// -- Auto vector-split lowering pass (RFC #1300; live, always-on) --------------
//
// Converts AUTO pl.split mixed InCore functions into the explicit split_aiv form
// (aiv_shard / aic_gather + halved vector sub-region) BEFORE ExpandMixedKernel.
// Runs unconditionally in the Default strategy. Same pre/post properties as
// ExpandMixedKernel's required set: it rewrites the still-mixed InCore body in
// place without changing the structural property set (and is a no-op for
// functions with no split mode or already in explicit split_aiv form).
//
// This pass closes the AivSplitValid verification window: it consumes and erases
// the first-class SplitAivScopeStmt regions (so the structural region verifier
// can no longer run afterwards), hence it requires AivSplitValid on entry and
// invalidates it on exit.
inline const PassProperties kLowerAutoVectorSplitProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure,
                 IRProperty::AivSplitValid},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .invalidated = {IRProperty::AivSplitValid}};

// -- Mixed kernel expansion pass ----------------------------------------------

// HardSyncallOccupancyValid is produced here (not by a transformation ExpandMixedKernel
// performs, but because this pass resolves each kernel's FunctionType to AIV/AIC/Group —
// the precondition the hard-syncall occupancy verifier depends on). The verifier fires
// once, right after this pass.
inline const PassProperties kExpandMixedKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded, IRProperty::NormalizedStmtStructure,
                 IRProperty::HardSyncallOccupancyValid}};

// -- GM pipe buffer injection pass (backend-gated; extracted from ExpandMixedKernel) --

inline const PassProperties kInjectGMPipeBufferProperties{
    .required = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded, IRProperty::NormalizedStmtStructure}};

// -- Split vector kernel pass -------------------------------------------------

inline const PassProperties kSplitVectorKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded},
    .produced = {IRProperty::SSAForm, IRProperty::VectorKernelSplit, IRProperty::NormalizedStmtStructure}};

// -- Memory / codegen passes --------------------------------------------------

inline const PassProperties kInitMemRefProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred},
    .produced = {IRProperty::HasMemRefs, IRProperty::NormalizedStmtStructure},
    .invalidated = {IRProperty::SSAForm}};

// Semantic must-alias materialization (Step 0 formerly inside MemoryReuse).
// Same requirements as MemoryReuse; retargets MemRefs in place without adding or
// removing structural IR properties.
inline const PassProperties kMaterializeSemanticAliasesProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::NormalizedStmtStructure}};

inline const PassProperties kMemoryReuseProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::NormalizedStmtStructure}};

inline const PassProperties kAllocateMemoryAddrProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::AllocatedMemoryAddr}};

// -- Return order normalization pass ------------------------------------------

inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps},
    .produced = {IRProperty::ReturnParamsExplicit}};

// -- Pipeline lowering + IO-order canonicalization passes (tile-level, before InitMemRef) ------

// SkewCrossCorePipeline runs immediately before LowerPipelineLoops and rewrites
// cross-core (cube/vector) pipeline loops into prologue/steady/epilogue skew or a
// Sequential demotion; same tile-level property set as the unroll pass
// (structural rewrite, no property added or removed).
inline const PassProperties kSkewCrossCorePipelineProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure}};

inline const PassProperties kLowerPipelineLoopsProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure}};

inline const PassProperties kCanonicalizeIOOrderProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure,
                 IRProperty::PipelineResolved}};

// -- Call-site direction pass -----------------------------------------------
//
// The integrity of ``Call::attrs_["arg_directions"]`` is checked by the
// ``CallDirectionsResolved`` PropertyVerifier (registered in
// PropertyVerifierRegistry), so PassPipeline auto-verifies it whenever this
// pass produces the property — no separate verify pass is needed.

// DeriveCallDirections also performs manual-scope lowering as Phase 2 (the
// former DeriveManualScopeDeps pass). Phase 2 reads the arg_directions
// populated by Phase 1; manual dependency edges stay in the typed
// ``Submit::deps_`` field throughout (ManualDepsOnSubmitOnly invariant), so
// no separate IRProperty is needed here.
inline const PassProperties kDeriveCallDirectionsProperties{.required = {IRProperty::SplitIncoreOrch},
                                                            .produced = {IRProperty::CallDirectionsResolved}};

inline const PassProperties kExpandManualPhaseFenceProperties{
    .required = {IRProperty::NoNestedCalls, IRProperty::NormalizedStmtStructure,
                 IRProperty::CallDirectionsResolved},
    .produced = {IRProperty::NoNestedCalls, IRProperty::NormalizedStmtStructure,
                 IRProperty::CallDirectionsResolved}};

// -- Automatic runtime-scope task dependency pass -----------------------------
//
// Reads ``Call.attrs_["arg_directions"]`` and writes
// ``Call.attrs_["compiler_manual_dep_edges"]`` for runtime scopes. MANUAL
// scopes are analyzed in the default pipeline. AUTO-scope analysis is
// controlled by the pass option and remains off by default at high-level
// pipeline entry points. The pass preserves CallDirectionsResolved because it
// does not rewrite call args or direction attrs.
inline const PassProperties kAutoDeriveTaskDependenciesProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::CallDirectionsResolved},
    .produced = {IRProperty::CallDirectionsResolved}};

// -- No-op tile.reshape folding pass -----------------------------------------

inline const PassProperties kFoldNoOpReshapeProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D}};

// -- Stamp tpop split/id onto tfree ops --------------------------------------

inline const PassProperties kStampTfreeSplitProperties{.required = {IRProperty::SplitIncoreOrch}};

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
