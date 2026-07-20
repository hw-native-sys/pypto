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

#ifndef PYPTO_IR_PTO_TARGET_LOWERING_H_
#define PYPTO_IR_PTO_TARGET_LOWERING_H_

#include <array>
#include <string_view>

namespace pypto {
namespace ir {

inline constexpr const char* kAttrPTOControlFlowHandles = "pto.control_flow_handles";
inline constexpr const char* kAttrPTOTargetLoweringDeferred = "pto.target_lowering_deferred";
inline constexpr const char* kAttrPTOUsesSpmdBlockParams = "pto.uses_spmd_block_params";
inline constexpr const char* kAttrPTOUsesSubblockParam = "pto.uses_subblock_param";
inline constexpr const char* kPTOSpmdBlockIdxParam = "__pypto_spmd_block_idx";
inline constexpr const char* kPTOSpmdBlockNumParam = "__pypto_spmd_block_num";
inline constexpr const char* kPTOSpmdSubblockIdxParam = "__pypto_spmd_subblock_idx";

inline bool IsSyntheticPTOTargetParamName(std::string_view name) {
  return name == kPTOSpmdBlockIdxParam || name == kPTOSpmdBlockNumParam || name == kPTOSpmdSubblockIdxParam;
}

enum class PTOSimpleOpKind { AllocationOnly, ScalarFill, Unary, TileBinary, TileScalarBinary };

struct PTOSimpleOpLowering {
  std::string_view logical_name;
  std::string_view target_name;
  PTOSimpleOpKind kind;
};

inline constexpr std::array<PTOSimpleOpLowering, 23> kPTOSimpleOpLowerings{{
    {"tile.create", "", PTOSimpleOpKind::AllocationOnly},
    // Alloc-backed reshapes are metadata-only in PTO: the memory planner has
    // already assigned the result view the same physical storage.  Keeping a
    // distinct typed handle makes that alias decision explicit without a
    // runtime data-movement operation.
    {"tile.reshape", "", PTOSimpleOpKind::AllocationOnly},
    {"tile.full", "pto.texpands", PTOSimpleOpKind::ScalarFill},
    {"tile.abs", "pto.tabs", PTOSimpleOpKind::Unary},
    {"tile.exp", "pto.texp", PTOSimpleOpKind::Unary},
    {"tile.log", "pto.tlog", PTOSimpleOpKind::Unary},
    {"tile.sqrt", "pto.tsqrt", PTOSimpleOpKind::Unary},
    {"tile.recip", "pto.trecip", PTOSimpleOpKind::Unary},
    {"tile.neg", "pto.tneg", PTOSimpleOpKind::Unary},
    {"tile.not", "pto.tnot", PTOSimpleOpKind::Unary},
    {"tile.relu", "pto.trelu", PTOSimpleOpKind::Unary},
    {"tile.move", "pto.tmov", PTOSimpleOpKind::Unary},
    {"tile.fillpad", "pto.tfillpad", PTOSimpleOpKind::Unary},
    {"tile.add", "pto.tadd", PTOSimpleOpKind::TileBinary},
    {"tile.sub", "pto.tsub", PTOSimpleOpKind::TileBinary},
    {"tile.mul", "pto.tmul", PTOSimpleOpKind::TileBinary},
    {"tile.div", "pto.tdiv", PTOSimpleOpKind::TileBinary},
    {"tile.matmul", "pto.tmatmul", PTOSimpleOpKind::TileBinary},
    {"tile.row_sum", "pto.trowsum", PTOSimpleOpKind::TileBinary},
    {"tile.adds", "pto.tadds", PTOSimpleOpKind::TileScalarBinary},
    {"tile.subs", "pto.tsubs", PTOSimpleOpKind::TileScalarBinary},
    {"tile.muls", "pto.tmuls", PTOSimpleOpKind::TileScalarBinary},
    {"tile.divs", "pto.tdivs", PTOSimpleOpKind::TileScalarBinary},
}};

inline const PTOSimpleOpLowering* FindPTOSimpleOpLowering(std::string_view logical_name) {
  for (const auto& lowering : kPTOSimpleOpLowerings) {
    if (lowering.logical_name == logical_name) return &lowering;
  }
  return nullptr;
}

inline bool IsPTOHandlePlanOp(std::string_view logical_name) {
  return logical_name == "tile.load" || logical_name == "tile.store" || logical_name == "tile.slice" ||
         FindPTOSimpleOpLowering(logical_name) != nullptr;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PTO_TARGET_LOWERING_H_
