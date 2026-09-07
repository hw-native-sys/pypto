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
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/printer.h"
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
  } else if (dtype == DataType::FP8E4M3FN) {
    // PTOAS v0.48 tile_buf dtype spelling (lowercase f8e4m3 does not parse).
    return "f8E4M3FN";
  } else if (dtype == DataType::FP8E5M2) {
    return "f8E5M2";
  } else if (dtype == DataType::FP8E8M0) {
    // Bare `f8E8M0` does not parse; PTOAS v0.48+ requires the dialect type.
    // EmitC maps loc=scaling + !pto.f8E8M0 → TileType::ScaleLeft/ScaleRight
    // (ui8 would wrongly become Fixpipe TileType::Scaling).
    return "!pto.f8E8M0";
  } else if (dtype == DataType::HF8) {
    return "!pto.hif8";
  } else if (dtype == DataType::FP4) {
    // MXFP4 E2M1 packed form used by pto-isa / PTOAS for MX matmul. Bare
    // `f4E2M1x2` does not parse in PTOAS (the bare-keyword parser lacks it);
    // the dialect type `!pto.f4E2M1x2` (TableGen mnemonic) is accepted in all
    // emit contexts (ptr<>, tile_buf dtype=, tensor_view element).
    return "!pto.f4E2M1x2";
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

std::string FormatLocalArrayTypeString(const ir::ArrayType& array_type) {
  auto extent = As<ir::ConstInt>(array_type.extent());
  CHECK(extent) << "array element extent must be a compile-time ConstInt for incore codegen";
  CHECK(array_type.dtype_ != DataType::TASK_ID)
      << "TASK_ID arrays are an orchestration-only construct (runtime dependency tracking) "
         "and cannot be lowered to an incore !pto.local_array";
  std::ostringstream oss;
  oss << "!pto.local_array<" << extent->value_ << "x" << DataTypeToMLIR(array_type.dtype_) << ">";
  return oss.str();
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
  } else if (space == ir::MemorySpace::LeftScale || space == ir::MemorySpace::RightScale) {
    // PTOAS v0.48 exposes a single MLIR loc `scaling` for scale / fixpipe buffers
    // (TileType::ScaleLeft/ScaleRight are not yet distinct tile_buf locs).
    return "scaling";
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
                                    uint64_t fractal, ir::PadValue pad, ir::CompactMode compact,
                                    int64_t v_row, int64_t v_col, bool v_row_dynamic, bool v_col_dynamic) {
  std::ostringstream oss;
  oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
  oss << ", rows=" << rows << ", cols=" << cols;
  oss << ", v_row=" << (v_row_dynamic ? "?" : std::to_string(v_row));
  oss << ", v_col=" << (v_col_dynamic ? "?" : std::to_string(v_col));
  oss << ", blayout=" << TileLayoutToStr(blayout);
  oss << ", slayout=" << TileLayoutToStr(slayout);
  oss << ", fractal=" << fractal;
  oss << ", pad=" << static_cast<int>(pad);
  if (compact != ir::CompactMode::null) {
    oss << ", compact=" << static_cast<int>(compact);
  }
  oss << ">";
  return oss.str();
}

std::string FormatMultiTileBufTypeString(const std::string& slot_type_str, uint64_t count) {
  INTERNAL_CHECK(count >= kMinMultiTileBufSlots && count <= kMaxMultiTileBufSlots)
      << "Internal error: multi_tile_buf count must be in [" << kMinMultiTileBufSlots << ", "
      << kMaxMultiTileBufSlots << "], got " << count;
  std::ostringstream oss;
  oss << "!pto.multi_tile_buf<" << slot_type_str << ", count=" << count << ">";
  return oss.str();
}

namespace {

/// Bytes a boxed fractal row is aligned to, mirroring PTOAS' `kAlignedBytes`.
constexpr int64_t kBoxAlignedBytes = 32;

/// Box granularity PTOAS derives for @p fractal / @p slayout, or nullopt when
/// the layout is one this rule does not cover (see `CheckBoxedTileExtents`).
std::optional<std::pair<int64_t, int64_t>> BoxGranularity(uint64_t fractal, ir::TileLayout slayout,
                                                          int64_t elem_bytes) {
  if (fractal == ir::tile_view_semantics::kAccFractal) {
    return std::pair<int64_t, int64_t>{16, 16};
  }
  if (fractal != 512 || elem_bytes <= 0 || kBoxAlignedBytes % elem_bytes != 0) {
    // The MX-scale fractal carries its own contract, and a carrier wider than
    // the alignment has no whole-box grid; PTOAS diagnoses both itself.
    return std::nullopt;
  }
  const int64_t packed = kBoxAlignedBytes / elem_bytes;
  if (slayout == ir::TileLayout::row_major) return std::pair<int64_t, int64_t>{16, packed};
  if (slayout == ir::TileLayout::col_major) return std::pair<int64_t, int64_t>{packed, 16};
  return std::nullopt;
}

}  // namespace

void CheckBoxedTileExtents(const ir::TileType& tile_type, const TileTypeComponents& components,
                           const ir::Span* span) {
  // A non-boxed tile is addressed as a flat run of bytes; the box rule is not
  // about it. (PTOAS checks a byte-size alignment there instead.)
  if (components.slayout == ir::TileLayout::none_box) return;

  const DataType& dtype = tile_type.dtype_;
  const auto space = tile_type.GetMemorySpace();

  // `ExtractTileTypeInfo` falls back to its struct default for a dimension that
  // is not a `ConstInt`, so a dynamic physical extent would be checked -- and
  // emitted -- as that placeholder instead of as itself. `InitMemRef` (pass 34)
  // already refuses a dynamic `TileType::shape_`, so reaching codegen with one
  // is a pass bug, not user input.
  for (const auto& dim : tile_type.shape_) {
    INTERNAL_CHECK(As<ir::ConstInt>(dim))
        << "Internal error: a boxed tile reached codegen with a dynamic physical extent ("
        << ir::PythonPrint(dim)
        << "); InitMemRef requires a static TileType::shape_, with any runtime extent in TileView";
  }

  const int64_t bits = static_cast<int64_t>(ir::storage_size::GetStorageBitWidth(dtype));
  if (bits <= 0 || bits % 8 != 0) return;  // sub-byte carrier: not this rule
  auto box = BoxGranularity(components.fractal, components.slayout, bits / 8);
  if (!box) return;
  // Plain locals rather than a structured binding: the lambda below captures
  // them, which C++17 does not allow for binding names.
  const int64_t inner_rows = box->first;
  const int64_t inner_cols = box->second;

  const std::string where = space.has_value() ? ir::MemorySpaceToString(*space) : std::string("unresolved");
  auto report = [&](const char* axis, int64_t extent, int64_t inner) {
    const int64_t padded = (extent + inner - 1) / inner * inner;
    // The span is optional: some allocations are hoisted out of any single
    // statement, and a location-less message still names the tile.
    CHECK_SPAN(false, span != nullptr ? *span : ir::Span("", 0, 0))
        << "a " << where << " tile of physical shape [" << components.rows << ", " << components.cols
        << "] and dtype " << dtype.ToString() << " must be a whole number of " << inner_rows << "x"
        << inner_cols << " fractal boxes, but its " << axis << " extent " << extent
        << " is not a multiple of " << inner
        << ". PTO addresses a boxed tile one box at a time, so a partial box has no address. The "
           "*logical* extent is free -- allocate "
        << padded << " on that axis and declare " << extent
        << " as the tile's valid_shape (`valid_shape=` on pl.load / pl.tile.create + "
           "pl.set_validshape), which moves and computes only the real data. A tensor-level "
           "pl.matmul / pl.matmul_acc does this for its M axis automatically.";
  };

  // PTOAS exempts the row axis for Vec (unboxed rows) and for a single-row tile
  // (the NZ map degenerates); the column rule always applies.
  const bool row_exempt = (space.has_value() && *space == ir::MemorySpace::Vec) || components.rows == 1;
  if (!row_exempt && inner_rows > 0 && components.rows % inner_rows != 0) {
    report("row", components.rows, inner_rows);
  }
  if (inner_cols > 0 && components.cols % inner_cols != 0) {
    report("column", components.cols, inner_cols);
  }
}

TileTypeComponents ExtractTileTypeInfo(const ir::TileType& tile_type, const std::string& dtype_str_override) {
  TileTypeComponents c;
  c.dtype_str = dtype_str_override.empty() ? DataTypeToMLIR(tile_type.dtype_) : dtype_str_override;

  // Effective view encodes implicit defaults for the memory space (Mat/Right/Acc),
  // so read it before lowering the shape. PTOAS represents FP4 Vec tiles in
  // physical x2-carrier coordinates: the packed BLayout axis is half the PyPTO
  // logical nibble extent. Matrix spaces deliberately keep their logical MX
  // dimensions because PTOAS/TMATMUL_MX use a separate packed-matrix contract.
  ir::TileView view = ir::tile_view_semantics::GetEffectiveTileView(tile_type);
  const bool packed_fp4_vec =
      tile_type.dtype_ == DataType::FP4 && tile_type.GetMemorySpace() == ir::MemorySpace::Vec;

  if (tile_type.shape_.size() >= 2) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) c.rows = c0->value_;
    if (auto c1 = As<ir::ConstInt>(tile_type.shape_[1])) c.cols = c1->value_;
  } else if (tile_type.shape_.size() == 1) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) {
      c.rows = 1;
      c.cols = c0->value_;
    }
  }
  if (packed_fp4_vec) {
    int64_t* packed_dim = view.blayout == ir::TileLayout::col_major ? &c.rows : &c.cols;
    CHECK(*packed_dim > 0 && *packed_dim % 2 == 0)
        << "FP4 Vec tile packed dimension must be a positive even logical extent for PTOAS, got "
        << *packed_dim;
    *packed_dim /= 2;
  }
  // Valid extent is always conveyed dynamically via `valid_row` / `valid_col`
  // operands on `pto.alloc_tile`; the type string therefore always reads
  // `v_row=?, v_col=?`.  Subview result types infer static valid dims via
  // InferSubviewTileTypeComponents (in pto_ops_common.cpp) separately.
  c.v_row = c.rows;
  c.v_col = c.cols;
  c.v_row_dynamic = true;
  c.v_col_dynamic = true;

  c.blayout = view.blayout;
  c.slayout = view.slayout;
  c.fractal = view.fractal;
  c.pad = view.pad;
  c.compact = view.compact;
  return c;
}

}  // namespace codegen
}  // namespace pypto
