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

// -- Loop unrolling pass (runs before SSA) ------------------------------------

inline const PassProperties kUnrollLoopsProperties{};

// -- Control flow structuring pass (runs before SSA, after unrolling) ---------

inline const PassProperties kCtrlFlowTransformProperties{.produced = {IRProperty::StructuredCtrlFlow}};

// -- Loop chunking pass (runs after SSA) --------------------------------------

inline const PassProperties kSplitChunkedLoopsProperties{
    .required = {IRProperty::SSAForm, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::NormalizedStmtStructure}};

// -- Chunk loop interchange pass (runs after SplitChunkedLoops) ---------------

inline const PassProperties kInterchangeChunkLoopsProperties{
    .required = {IRProperty::SSAForm, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::NormalizedStmtStructure}};

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

// -- Composite op lowering pass (tile.sin / tile.cos -> primitives, etc.) -----
//
// LowerCompositeOps decomposes composite tile ops into primitive arithmetic
// ops. Today the only composite ops handled are tile.sin / tile.cos (Cody-Waite
// range reduction + degree-9 Horner polynomial); future composite ops add a
// rule to the file-local dispatch table in lower_composite_ops_pass.cpp. The
// pass operates purely within the existing tile-op vocabulary, so it neither
// requires nor produces nor invalidates any IRProperty.

inline const PassProperties kLowerCompositeOpsProperties{};

// -- Outlining pass -----------------------------------------------------------

inline const PassProperties kOutlineIncoreScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch}};

// -- Cluster outlining pass ---------------------------------------------------

inline const PassProperties kOutlineClusterScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::ClusterOutlined}};

// -- Hierarchy outlining pass -------------------------------------------------

inline const PassProperties kOutlineHierarchyScopesProperties{
    .required = {IRProperty::SSAForm},
    .produced = {IRProperty::SSAForm, IRProperty::HierarchyOutlined,
                 IRProperty::OrchestrationReferencesResolved}};

// -- Tensor-to-tile conversion pass ------------------------------------------

inline const PassProperties kConvertTensorToTileOpsProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::NormalizedStmtStructure}};

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

// -- Tile memory space inference pass -----------------------------------------

inline const PassProperties kInferTileMemorySpaceProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure}};

// -- Lower transpose-load parameter layout pass (RFC #1300 P6) ----------------

inline const PassProperties kLowerTransposeLoadParamLayoutProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D}};

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

// -- Mixed kernel expansion pass ----------------------------------------------

inline const PassProperties kExpandMixedKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded, IRProperty::NormalizedStmtStructure}};

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
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps}};

// -- Pipeline lowering + IO-order canonicalization passes (tile-level, before InitMemRef) ------

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
// populated by Phase 1 and writes the per-Call ``manual_dep_edges`` attr
// inside ``RuntimeScopeStmt(manual=true)`` regions; codegen reads that attr
// directly, so no separate IRProperty is needed.
inline const PassProperties kDeriveCallDirectionsProperties{.required = {IRProperty::SplitIncoreOrch},
                                                            .produced = {IRProperty::CallDirectionsResolved}};

// -- No-op tile.reshape folding pass -----------------------------------------

inline const PassProperties kFoldNoOpReshapeProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D}};

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
