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

#include "pypto/codegen/pto/pto_type_utils.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;

std::string DataTypeToMLIR(DataType dtype) {
  if (dtype == DataType::FP32) {
    return "f32";
  } else if (dtype == DataType::FP16) {
    return "f16";
  } else if (dtype == DataType::BF16) {
    return "bf16";
  } else if (dtype == DataType::INT32) {
    return "i32";
  } else if (dtype == DataType::UINT32) {
    return "ui32";
  } else if (dtype == DataType::INDEX) {
    return "index";
  } else if (dtype == DataType::INT64) {
    return "i64";
  } else if (dtype == DataType::UINT64) {
    return "ui64";
  } else if (dtype == DataType::INT8) {
    return "i8";
  } else if (dtype == DataType::UINT8) {
    return "ui8";
  } else if (dtype == DataType::INT16) {
    return "i16";
  } else if (dtype == DataType::UINT16) {
    return "ui16";
  } else if (dtype == DataType::BOOL) {
    return "i1";
  } else {
    throw ValueError("Invalid DataType value");
  }
}

std::string MemorySpaceToMLIR(ir::MemorySpace space) {
  if (space == ir::MemorySpace::DDR) {
    return "gm";
  } else if (space == ir::MemorySpace::Vec) {
    return "vec";
  } else if (space == ir::MemorySpace::Mat) {
    return "mat";
  } else if (space == ir::MemorySpace::Left) {
    return "left";
  } else if (space == ir::MemorySpace::Right) {
    return "right";
  } else if (space == ir::MemorySpace::Acc) {
    return "acc";
  } else if (space == ir::MemorySpace::Bias) {
    return "bias";
  } else {
    throw ValueError("Invalid MemorySpace value");
  }
}

const char* TileLayoutToStr(ir::TileLayout layout) {
  switch (layout) {
    case ir::TileLayout::none_box:
      return "none_box";
    case ir::TileLayout::row_major:
      return "row_major";
    case ir::TileLayout::col_major:
      return "col_major";
    default:
      INTERNAL_CHECK(false) << "Unknown TileLayout: " << static_cast<int>(layout);
      return "";
  }
}

std::string FormatTileBufTypeString(const std::string& loc, const std::string& dtype_str, int64_t rows,
                                    int64_t cols, ir::TileLayout blayout, ir::TileLayout slayout,
                                    uint64_t fractal, ir::PadValue pad, int64_t v_row, int64_t v_col,
                                    bool v_row_dynamic, bool v_col_dynamic) {
  std::ostringstream oss;
  oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
  oss << ", rows=" << rows << ", cols=" << cols;
  oss << ", v_row=" << (v_row_dynamic ? "?" : std::to_string(v_row));
  oss << ", v_col=" << (v_col_dynamic ? "?" : std::to_string(v_col));
  oss << ", blayout=" << TileLayoutToStr(blayout);
  oss << ", slayout=" << TileLayoutToStr(slayout);
  oss << ", fractal=" << fractal;
  oss << ", pad=" << static_cast<int>(pad) << ">";
  return oss.str();
}

TileTypeComponents ExtractTileTypeInfo(const ir::TileType& tile_type, const std::string& dtype_str_override) {
  TileTypeComponents c;
  c.dtype_str = dtype_str_override.empty() ? DataTypeToMLIR(tile_type.dtype_) : dtype_str_override;

  if (tile_type.shape_.size() >= 2) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) c.rows = c0->value_;
    if (auto c1 = As<ir::ConstInt>(tile_type.shape_[1])) c.cols = c1->value_;
  } else if (tile_type.shape_.size() == 1) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) {
      c.rows = 1;
      c.cols = c0->value_;
    }
  }
  // Valid extent is always conveyed dynamically via `valid_row` / `valid_col`
  // operands on `pto.alloc_tile`; the type string therefore always reads
  // `v_row=?, v_col=?`. The numeric `v_row` / `v_col` fields below are kept
  // for symmetry but are ignored by the formatter when *_dynamic is true.
  c.v_row = c.rows;
  c.v_col = c.cols;
  c.v_row_dynamic = true;
  c.v_col_dynamic = true;

  if (tile_type.tile_view_.has_value()) {
    const auto& tv = *tile_type.tile_view_;
    c.blayout = tv.blayout;
    c.slayout = tv.slayout;
    c.fractal = tv.fractal;
    c.pad = tv.pad;
  } else if (c.cols == 1 && c.rows > 1) {
    c.blayout = ir::TileLayout::col_major;
  }
  return c;
}

}  // namespace codegen
}  // namespace pypto
