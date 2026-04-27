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

#ifndef PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_
#define PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_

#include <cstdint>
#include <string>

#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/// Convert DataType to MLIR type string (e.g., FP32 -> "f32", INT32 -> "i32")
std::string DataTypeToMLIR(DataType dtype);

/// Convert MemorySpace to PTO address space string (e.g., Vec -> "vec", DDR -> "gm")
std::string MemorySpaceToMLIR(ir::MemorySpace space);

/// Convert TileLayout to its string name (e.g., row_major -> "row_major")
const char* TileLayoutToStr(ir::TileLayout layout);

/// Format a complete !pto.tile_buf<...> type string from individual components.
/// v_row/v_col are the valid shape dimensions (may differ from rows/cols).
std::string FormatTileBufTypeString(const std::string& loc, const std::string& dtype_str, int64_t rows,
                                    int64_t cols, ir::TileLayout blayout, ir::TileLayout slayout,
                                    uint64_t fractal, ir::PadValue pad, int64_t v_row, int64_t v_col,
                                    bool v_row_dynamic = false, bool v_col_dynamic = false);

/// Intermediate result holder for ExtractTileTypeInfo.
struct TileTypeComponents {
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::PadValue pad = ir::PadValue::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
};

/// Extract dtype, shape, and layout from a TileType into a TileTypeComponents struct.
///
/// `v_row_dynamic` / `v_col_dynamic` are always set when the corresponding rank is
/// present, so the resulting `!pto.tile_buf<...>` type string always reads
/// `v_row=?, v_col=?`. The actual extents are conveyed via the `valid_row` /
/// `valid_col` operands on `pto.alloc_tile` (see ComputeAllocTileFields).
///
/// @param dtype_str_override Optional override for the dtype string (e.g.,
///                           PTOCodegen::GetTypeString); empty falls back to
///                           DataTypeToMLIR(tile_type.dtype_).
TileTypeComponents ExtractTileTypeInfo(const ir::TileType& tile_type,
                                       const std::string& dtype_str_override = "");

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_
