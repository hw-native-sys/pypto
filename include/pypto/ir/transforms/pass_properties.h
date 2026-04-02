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

// -- Expression simplification pass -------------------------------------------

inline const PassProperties kSimplifyExprProperties{};

// -- Outlining pass -----------------------------------------------------------

inline const PassProperties kOutlineIncoreScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch}};

// -- Cluster outlining pass ---------------------------------------------------

inline const PassProperties kOutlineClusterScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::ClusterOutlined}};

// -- Hierarchy outlining pass -------------------------------------------------

inline const PassProperties kOutlineHierarchyScopesProperties{
    .required = {IRProperty::SSAForm}, .produced = {IRProperty::SSAForm, IRProperty::HierarchyOutlined}};

// -- Tensor-to-tile conversion pass ------------------------------------------

inline const PassProperties kConvertTensorToTileOpsProperties{
    .required = {IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::NormalizedStmtStructure}};

// -- Tile ND-to-2D flattening pass --------------------------------------------

inline const PassProperties kFlattenTileNdTo2DProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::TileOps2D, IRProperty::NormalizedStmtStructure}};

// -- Tile memory space inference pass -----------------------------------------

inline const PassProperties kInferTileMemorySpaceProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::NormalizedStmtStructure},
    .produced = {IRProperty::SSAForm, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure}};

// -- Resolve transpose layout pass --------------------------------------------

inline const PassProperties kResolveTransposeLayoutProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D}};

// -- Resolve backend op layouts pass ------------------------------------------

inline const PassProperties kResolveBackendOpLayoutsProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D},
    .invalidated = {IRProperty::NormalizedStmtStructure}};

// -- Mixed kernel expansion pass ----------------------------------------------

inline const PassProperties kExpandMixedKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::IncoreTileOps, IRProperty::SplitIncoreOrch,
                 IRProperty::TileOps2D, IRProperty::TileMemoryInferred, IRProperty::NormalizedStmtStructure},
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

inline const PassProperties kInsertSyncProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D}};

inline const PassProperties kAllocateMemoryAddrProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps, IRProperty::HasMemRefs,
                 IRProperty::TileOps2D},
    .produced = {IRProperty::AllocatedMemoryAddr}};

// -- Return order normalization pass ------------------------------------------

inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps}};

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
