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

/**
 * @file scatter.cpp
 * @brief Scatter tile operations (DPS form).
 *
 * Mirrors the gather family but writes per-row instead of reading per-row:
 * - tile.scatter:      index-based row scatter (pto.tscatter index form)
 * - tile.scatter_mask: mask-pattern row scatter (PyPTO codegen form; no
 *                      pto.tscatter mask ISA op — unlike tile.gather_mask)
 *
 * Both ops are DPS — `dst` is the in/out buffer; the IR result aliases `dst`
 * via `set_output_reuses_input(...)`. There is no compare form for scatter.
 *
 * Duplicate-index ordering: when two index entries map to the same destination
 * slot, pto.tscatter resolves the collision in ascending element order (the
 * later/higher-index write wins), matching torch `scatter_`'s last-wins
 * semantics along the scan axis. Callers that build flat indices
 * (ConvertTensorToTileOps) and the ST reference both rely on this order; it is
 * a pto.tscatter ABI guarantee, not a PyPTO-side choice.
 */

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// pto.tscatter element-type whitelist for `src` / `dst` operands.
bool IsScatterElementDtype(const DataType& dt) {
  return dt == DataType::FP16 || dt == DataType::FP32 || dt == DataType::BF16 || dt == DataType::INT16 ||
         dt == DataType::INT32 || dt == DataType::INT8;
}

// pto.tscatter element-type whitelist for the `indexes` operand.
bool IsScatterIndexDtype(const DataType& dt) { return dt == DataType::INT16 || dt == DataType::INT32; }

// Hardware element-size matching rule between `dst` and `indexes`:
//   sizeof(dst) == 4 bytes  →  sizeof(indexes) == 4 bytes (i32)
//   sizeof(dst) == 2 bytes  →  sizeof(indexes) == 2 bytes (i16)
//   sizeof(dst) == 1 byte   →  sizeof(indexes) == 2 bytes (i16)
//
// Returns the required indexes-byte-width given a destination element width.
int RequiredIndexBytes(int dst_bytes) { return dst_bytes == 1 ? 2 : dst_bytes; }

void CheckScatterDtypeSizing(const DataType& dst_dtype, const DataType& idx_dtype,
                             const std::string& op_name) {
  const int dst_bytes = static_cast<int>(dst_dtype.GetBit()) / 8;
  const int idx_bytes = static_cast<int>(idx_dtype.GetBit()) / 8;
  const int required = RequiredIndexBytes(dst_bytes);
  CHECK(idx_bytes == required) << "The operator " << op_name << " with dst dtype " << dst_dtype.ToString()
                               << " (" << dst_bytes << " bytes) requires indexes dtype of " << required
                               << " bytes, but got " << idx_dtype.ToString() << " (" << idx_bytes
                               << " bytes)";
}

// Build the DPS-style output type that inherits `dst`'s shape/dtype/memory
// metadata. Mirrors tile.scatter_update / tile.store: the IR result is an
// alias of `dst` so downstream passes that consume the result see the post-
// scatter buffer with the right tile_view / memory space.
TypePtr MakeScatterResultType(const std::shared_ptr<const TileType>& dst_type) {
  TileView tile_view;
  if (dst_type->tile_view_.has_value()) {
    tile_view = *dst_type->tile_view_;
  }
  if (tile_view.valid_shape.empty()) {
    tile_view.valid_shape = dst_type->shape_;
  }
  return std::make_shared<TileType>(dst_type->shape_, dst_type->dtype_, std::nullopt, tile_view,
                                    dst_type->memory_space_);
}

}  // namespace

// ============================================================================
// tile.scatter — index form of pto.tscatter (DPS)
// ============================================================================
//
// Args (3 inputs):
//   dst     : destination tile (in/out via DPS), same dtype as `src`. Its
//             shape is independent of `src` — only the flattened indices need
//             to address valid `dst` elements.
//   src     : source tile, [rows, cols]
//   indexes : per-element *flattened* destination indices, with the **same
//             [rows, cols] shape as `src`** (dtype constrained by the
//             element-size matching rule). pto.tscatter loops over the index
//             tile's valid shape and writes dst.flat[indexes[i, j]] = src[i, j].
//             A column scatter dst[i, c] = src[i, j] is expressed with
//             indexes[i, j] = i * dst_cols + c. The tensor.scatter lowering
//             builds these flat indices from the user's gather-style column
//             index tile (same shape as `src`).
//
// Result: TileType matching `dst` (aliased via set_output_reuses_input(0)).
//
// Semantics: dst.flat[indexes[i, j]] = src[i, j]   for (i, j) in src valid shape

static TypePtr DeduceTileScatterType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& /*kwargs*/,
                                     const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (dst, src, indexes), but got " << args.size();

  auto dst_type = As<TileType>(args[0]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(dst_type->dtype_))
      << "The operator " << op_name << " requires dst dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << dst_type->dtype_.ToString();

  auto src_type = As<TileType>(args[1]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(src_type->dtype_))
      << "The operator " << op_name << " requires src dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << src_type->dtype_.ToString();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires dst dtype (" << dst_type->dtype_.ToString()
      << ") to match src dtype (" << src_type->dtype_.ToString() << ")";

  auto idx_type = As<TileType>(args[2]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires indexes to be a TileType, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(IsScatterIndexDtype(idx_type->dtype_))
      << "The operator " << op_name << " requires indexes dtype in {INT16, INT32}, but got "
      << idx_type->dtype_.ToString();

  CheckScatterDtypeSizing(dst_type->dtype_, idx_type->dtype_, op_name);

  CHECK(src_type->shape_.size() == 2 && dst_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D src/dst tiles, but got src rank "
      << src_type->shape_.size() << " and dst rank " << dst_type->shape_.size();
  CHECK(idx_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D indexes tile, but got rank " << idx_type->shape_.size();

  // pto.tscatter loops over the index tile's valid [rows, cols], reading the
  // matching src element each step, so `indexes` must have exactly the same
  // shape as `src`. `dst` is addressed by the flattened indices and may have an
  // independent shape (its column count need not equal src's).
  auto src_rows = As<ConstInt>(src_type->shape_[0]);
  auto idx_rows = As<ConstInt>(idx_type->shape_[0]);
  if (src_rows && idx_rows) {
    CHECK(src_rows->value_ == idx_rows->value_)
        << "The operator " << op_name << " requires indexes.shape[0] == src.shape[0], got src rows "
        << src_rows->value_ << " vs indexes rows " << idx_rows->value_;
  }
  auto src_cols = As<ConstInt>(src_type->shape_[1]);
  auto idx_cols = As<ConstInt>(idx_type->shape_[1]);
  if (src_cols && idx_cols) {
    CHECK(src_cols->value_ == idx_cols->value_)
        << "The operator " << op_name << " requires indexes.shape[1] == src.shape[1], got src cols "
        << src_cols->value_ << " vs indexes cols " << idx_cols->value_;
  }

  return MakeScatterResultType(dst_type);
}

REGISTER_OP("tile.scatter")
    .set_op_category("TileOp")
    .set_description(
        "Scatter src elements into dst at flattened destination indices "
        "(maps to pto.tscatter index form; DPS: dst is in/out)")
    .add_argument("dst", "Destination tile (same dtype as src; rewritten in-place via DPS)")
    .add_argument("src", "Source tile (FP16/FP32/BF16/INT8/INT16/INT32, 2D)")
    .add_argument("indexes",
                  "Per-element flattened destination index tile (INT16 or INT32; same shape as src)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileScatterType(args, kwargs, "tile.scatter");
    });

// ============================================================================
// tile.scatter_mask — mask-pattern form of pto.tscatter (DPS, A3 / CPU-sim only)
// ============================================================================
//
// Args (2 inputs):
//   dst : destination tile (DPS, written via the mask pattern)
//   src : source tile (compact rows)
//
// Attrs:
//   mask_pattern : 1-7 (P0101 / P1010 / P0001 / P0010 / P0100 / P1000 / P1111)
//
// Semantics: write each `src` element into the next mask-marked column of the
// corresponding `dst` row. For non-P1111 patterns dst.cols equals src.cols *
// stride (stride=2 for patterns 1-2, stride=4 for patterns 3-6). For P1111
// dst.cols == src.cols.

static TypePtr DeduceTileScatterMaskType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                                         const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (dst, src), but got "
                          << args.size();

  auto dst_type = As<TileType>(args[0]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(dst_type->dtype_))
      << "The operator " << op_name << " requires dst dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << dst_type->dtype_.ToString();

  auto src_type = As<TileType>(args[1]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(src_type->dtype_))
      << "The operator " << op_name << " requires src dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << src_type->dtype_.ToString();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires dst and src to have the same dtype, got "
      << dst_type->dtype_.ToString() << " vs " << src_type->dtype_.ToString();

  CHECK(src_type->shape_.size() == 2 && dst_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D src/dst tiles, but got src rank "
      << src_type->shape_.size() << " and dst rank " << dst_type->shape_.size();

  int pattern = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "mask_pattern") {
      pattern = AnyCast<int>(value, "kwarg key: mask_pattern");
      break;
    }
  }
  CHECK(pattern >= 1 && pattern <= 7)
      << "The operator " << op_name << " requires mask_pattern in range [1, 7], but got " << pattern;

  // Row count must match between src and dst.
  auto src_rows = As<ConstInt>(src_type->shape_[0]);
  auto dst_rows = As<ConstInt>(dst_type->shape_[0]);
  if (src_rows && dst_rows) {
    CHECK(src_rows->value_ == dst_rows->value_)
        << "The operator " << op_name << " requires src.shape[0] == dst.shape[0], got src rows "
        << src_rows->value_ << " vs dst rows " << dst_rows->value_;
  }

  // Column expansion: dst.cols == src.cols * stride (or equal for P1111).
  auto src_cols_const = As<ConstInt>(src_type->shape_[1]);
  auto dst_cols_const = As<ConstInt>(dst_type->shape_[1]);
  if (src_cols_const && dst_cols_const) {
    const int64_t stride = (pattern == 7) ? 1 : ((pattern <= 2) ? 2 : 4);
    CHECK(dst_cols_const->value_ == src_cols_const->value_ * stride)
        << "The operator " << op_name << " with mask_pattern=" << pattern << " requires dst.shape[1] ("
        << dst_cols_const->value_ << ") == src.shape[1] (" << src_cols_const->value_ << ") * " << stride;
  }

  return MakeScatterResultType(dst_type);
}

REGISTER_OP("tile.scatter_mask")
    .set_op_category("TileOp")
    .set_description(
        "Scatter src rows into dst columns selected by a mask pattern (DPS: dst "
        "is in/out). PyPTO codegen-level form lowered to a pto.tscatter mask "
        "emission — not a distinct pto-isa instruction (unlike tile.gather_mask); "
        "emitted for A2/A3 / CPU-sim style lowering paths.")
    .add_argument("dst", "Destination tile (DPS; columns are rewritten on mask-marked positions)")
    .add_argument("src", "Source tile (compact rows; same dtype as dst)")
    .set_attr<int>("mask_pattern")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileScatterMaskType(args, kwargs, "tile.scatter_mask");
    });

}  // namespace ir
}  // namespace pypto
