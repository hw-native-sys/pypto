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
 * @file gather.cpp
 * @brief Gather tile operations
 *
 * This file implements gather operators:
 * - tile.gather: index-based element gathering (pto.tgather index form)
 * - tile.gather_mask: mask-pattern element selection (pto.tgather mask form)
 * - tile.gather_compare: compare-form element gathering, two outputs
 *   (pto.tgather compare form, returns TupleType{dst, cdst})
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
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

static TypePtr DeduceTileGatherType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (src, indices, tmp), but got " << args.size();

  // First arg: src tile (f16, f32, i16, or i32)
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::INT16 || src_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires src dtype to be FP16, FP32, INT16, or INT32, but got "
      << src_type->dtype_.ToString();

  // Second arg: indices tile (must be i32)
  auto idx_type = As<TileType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(idx_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires indices dtype to be INT32, but got "
      << idx_type->dtype_.ToString();

  // Third arg: tmp workspace tile (must be i32, same shape as indices)
  auto tmp_type = As<TileType>(args[2]->GetType());
  CHECK(tmp_type) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(tmp_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires tmp dtype to be INT32, but got "
      << tmp_type->dtype_.ToString();
  CHECK(tmp_type->shape_.size() == idx_type->shape_.size())
      << "The operator " << op_name << " requires tmp shape rank to match indices rank ("
      << idx_type->shape_.size() << "), but got " << tmp_type->shape_.size();

  // Output: shape from indices tile, dtype from src tile, propagate tile_view
  TileView tile_view;
  tile_view.valid_shape = idx_type->shape_;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(idx_type->shape_, src_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Registration for Gather Operations
// ============================================================================

REGISTER_OP("tile.gather")
    .set_op_category("TileOp")
    .set_description("Gather elements by index (maps to pto.tgather)")
    .add_argument("src", "Source tile (FP16, FP32, INT16, or INT32)")
    .add_argument("indices", "Index tile (INT32)")
    .add_argument("tmp", "Temporary workspace tile (INT32)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGatherType(args, kwargs, "tile.gather");
    });

// ============================================================================
// Gather Mask: mask-pattern form of pto.tgather
// ============================================================================

static TypePtr DeduceTileGatherMaskType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument (src), but got "
                          << args.size();

  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::INT16 || src_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires src dtype to be FP16, FP32, INT16, or INT32, but got "
      << src_type->dtype_.ToString();

  // Validate mask_pattern kwarg (values 1-7 per PTOAS MaskPattern enum)
  int pattern = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "mask_pattern") {
      pattern = std::any_cast<int>(value);
      break;
    }
  }
  CHECK(pattern >= 1 && pattern <= 7)
      << "The operator " << op_name << " requires mask_pattern in range [1, 7], but got " << pattern;

  // Output shape: mask selects a subset of columns per row, producing a compacted tile.
  //   P0101 (1), P1010 (2) — stride 2: each row contributes cols/2 elements
  //   P0001 (3)..P1000 (6) — stride 4: each row contributes cols/4 elements
  //   P1111 (7)            — no stride: all cols kept
  const auto& src_shape = src_type->shape_;
  INTERNAL_CHECK_SPAN(src_shape.size() == 2, args[0]->span_)
      << "Internal error: tile.gather_mask requires 2D src shape, got rank " << src_shape.size();

  const ExprPtr& col_expr = src_shape[1];
  ExprPtr out_col_expr;
  if (pattern == 7) {
    out_col_expr = col_expr;  // P1111: all cols
  } else {
    int64_t divisor = (pattern <= 2) ? 2 : 4;
    if (auto const_col = As<ConstInt>(col_expr)) {
      int64_t out_cols = const_col->value_ / divisor;
      CHECK(const_col->value_ % divisor == 0)
          << "The operator " << op_name << " with mask_pattern=" << pattern
          << " requires src columns divisible by " << divisor << ", got " << const_col->value_;
      out_col_expr = std::make_shared<ConstInt>(out_cols, DataType::INDEX, Span::unknown());
    } else {
      auto div_expr = std::make_shared<ConstInt>(divisor, DataType::INDEX, Span::unknown());
      out_col_expr = std::make_shared<FloorDiv>(col_expr, div_expr, DataType::INDEX, Span::unknown());
    }
  }

  std::vector<ExprPtr> out_shape = {src_shape[0], out_col_expr};
  TileView tile_view;
  tile_view.valid_shape = out_shape;
  InheritTileViewLayout(tile_view, src_type);

  // Read optional output_dtype kwarg for cross-type bit extraction (e.g. FP32→UINT32).
  // Hardware TGATHER mask form only requires sizeof(dst) == sizeof(src), not same type.
  bool has_output_dtype = false;
  DataType out_dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "output_dtype") {
      if (value.type() == typeid(DataType)) {
        out_dtype = AnyCast<DataType>(value, "kwarg key: output_dtype");
      } else if (value.type() == typeid(int)) {
        out_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: output_dtype"));
      }
      has_output_dtype = true;
      break;
    }
  }
  if (!has_output_dtype) {
    out_dtype = src_type->dtype_;
  } else {
    CHECK(out_dtype.GetBit() == src_type->dtype_.GetBit())
        << "The operator " << op_name << " output_dtype must have the same bit width as src dtype ("
        << src_type->dtype_.ToString() << " = " << src_type->dtype_.GetBit() << " bits), but got "
        << out_dtype.ToString() << " = " << out_dtype.GetBit() << " bits";
  }

  return std::make_shared<TileType>(out_shape, out_dtype, std::nullopt, tile_view);
}

REGISTER_OP("tile.gather_mask")
    .set_op_category("TileOp")
    .set_description("Gather elements by mask pattern (maps to pto.tgather with maskPattern)")
    .add_argument("src", "Source tile (FP16, FP32, INT16, or INT32)")
    .set_attr<int>("mask_pattern")
    .set_attr<DataType>("output_dtype")  // optional: cross-type output (sizeof equality required)
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGatherMaskType(args, kwargs, "tile.gather_mask");
    });

// ============================================================================
// Gather Compare: compare-form of pto.tgather with two destination outputs.
// ============================================================================
//
// Args (3 inputs):
//   src    : source tile (FP16/FP32/INT16/INT32)
//   kvalue : scalar threshold (ScalarType, dtype must match src — applied
//            to every row of src)
//   tmp    : workspace tile (UINT8, sized by codegen kernel)
//
// Attrs:
//   cmp_mode    : int in [0, 5] matching {eq, ne, lt, le, gt, ge}
//   offset      : int starting index offset
//   out_cols    : int — output column count per row for dst
//   count_dtype : DataType — INT32 or UINT32 (defaults to INT32 in caller)
//
// Output: TupleType{ TileType_dst, TileType_cdst } (2 outputs)
//   dst  shape  = [rows, out_cols], dtype = INT32 (gathered indices)
//   cdst shape  = [1, rows], dtype = count_dtype (per-row match count)
//
// `cdst` is kept 2D (`[1, rows]`) so that the rows-many counts are
// physically contiguous (row_major + cols * sizeof(count_dtype) % 32 == 0),
// which is required by the PTOAS pto.tgather verifier and the PTO TileConfig
// 32-byte alignment static_assert. The compare-form ISA writes
// `cdstPtr + i` for i in [0, srcValidRow), and the [1, rows] row_major
// layout provides exactly that contiguity.
//
// The DPS buffers backing dst/cdst are allocated by the downstream
// init_memref / memory_reuse passes from the deduced TupleType, so this
// op stays purely 3-input-2-output at the IR/DSL surface.
//
// The DSL form `d, c = pl.tile.gather_compare(src, kvalue, tmp, ...)` is
// unpacked by the parser into `tmp_tuple = call; d = tmp_tuple[0];
// c = tmp_tuple[1]`, so callers receive a (Tile, Tile) pair.

static TypePtr DeduceTileGatherCompareType(const std::vector<ExprPtr>& args,
                                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (src, kvalue, tmp), but got " << args.size();

  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::INT16 || src_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires src dtype in {FP16, FP32, INT16, INT32}, but got "
      << src_type->dtype_.ToString();
  CHECK(src_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D src, but got rank " << src_type->shape_.size();

  auto kv_type = As<ScalarType>(args[1]->GetType());
  CHECK(kv_type) << "The operator " << op_name << " requires kvalue to be a ScalarType, but got "
                 << args[1]->GetType()->TypeName();
  CHECK(kv_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires kvalue dtype equal to src dtype "
      << src_type->dtype_.ToString() << " , but got " << kv_type->dtype_.ToString();

  auto tmp_type = As<TileType>(args[2]->GetType());
  CHECK(tmp_type) << "The operator " << op_name << " requires tmp to be a TileType, but got "
                  << args[2]->GetType()->TypeName();

  int cmp_mode = -1;
  bool cmp_mode_seen = false;
  int out_cols = -1;
  bool out_cols_seen = false;
  DataType count_dtype = DataType::INT32;
  for (const auto& [key, value] : kwargs) {
    if (key == "cmp_mode") {
      cmp_mode = AnyCast<int>(value, "kwarg key: cmp_mode");
      cmp_mode_seen = true;
    } else if (key == "out_cols") {
      out_cols = AnyCast<int>(value, "kwarg key: out_cols");
      out_cols_seen = true;
    } else if (key == "count_dtype") {
      if (value.type() == typeid(DataType)) {
        count_dtype = AnyCast<DataType>(value, "kwarg key: count_dtype");
      } else {
        count_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: count_dtype"));
      }
    }
  }
  CHECK(cmp_mode_seen) << "The operator " << op_name << " requires a 'cmp_mode' keyword argument";
  CHECK(cmp_mode >= 0 && cmp_mode <= 5)
      << "The operator " << op_name << " requires cmp_mode in [0, 5] (eq/ne/lt/le/gt/ge), but got "
      << cmp_mode;
  CHECK(out_cols_seen) << "The operator " << op_name << " requires an 'out_cols' keyword argument";
  CHECK(out_cols > 0) << "The operator " << op_name << " requires out_cols > 0, but got " << out_cols;
  CHECK(count_dtype == DataType::INT32 || count_dtype == DataType::UINT32)
      << "The operator " << op_name << " requires count_dtype to be INT32 or UINT32, but got "
      << count_dtype.ToString();

  // Deduce dst/cdst tile types from src.shape[0] (rows) + attrs.
  const ExprPtr& rows_expr = src_type->shape_[0];
  auto out_cols_expr = std::make_shared<ConstInt>(out_cols, DataType::INDEX, Span::unknown());

  TileView dst_view;
  dst_view.valid_shape = {rows_expr, out_cols_expr};
  InheritTileViewLayout(dst_view, src_type);
  std::vector<ExprPtr> dst_shape = {rows_expr, out_cols_expr};
  auto dst_type = std::make_shared<TileType>(dst_shape, DataType::INT32, std::nullopt, dst_view);

  // cdst is shaped as [1, rows] (one count per src-row, laid out as a single
  // row of `rows` int32s) — required by the PTOAS pto.tgather verifier
  // (cdst must use row_major blayout) AND the PTO TileConfig 32-byte
  // alignment static_assert (row_major + cols*sizeof(int) % 32 == 0).
  // The compare-form ISA writes `cdstPtr + i` for i in [0, srcValidRow),
  // so the rows-many counts must be physically contiguous, which the
  // [1, rows] row_major layout provides.
  auto one_expr = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  TileView cdst_view;
  cdst_view.valid_shape = {one_expr, rows_expr};
  cdst_view.blayout = TileLayout::row_major;
  std::vector<ExprPtr> cdst_shape = {one_expr, rows_expr};
  auto cdst_type = std::make_shared<TileType>(cdst_shape, count_dtype, std::nullopt, cdst_view);

  std::vector<TypePtr> elements{dst_type, cdst_type};
  return std::make_shared<TupleType>(std::move(elements));
}

REGISTER_OP("tile.gather_compare")
    .set_op_category("TileOp")
    .set_description(
        "Compare-form gather: scan src per-row against kvalue, produce gathered indices "
        "tile (dst) and per-row match count tile (cdst). Maps to pto.tgather compare-form. "
        "Returns TupleType{dst, cdst}.")
    .add_argument("src", "Source tile (FP16/FP32/INT16/INT32, 2D)")
    .add_argument("kvalue", "Scalar threshold (ScalarType; dtype must match src; applied per row)")
    .add_argument("tmp", "Workspace tile (UINT8)")
    .set_attr<int>("cmp_mode")
    .set_attr<int>("offset")
    .set_attr<int>("out_cols")
    .set_attr<DataType>("count_dtype")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    // Output is a TupleType{TileType_dst, TileType_cdst}. set_output_memory applies
    // Vec to every TileType element inside the TupleType.
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGatherCompareType(args, kwargs, "tile.gather_compare");
    });

}  // namespace ir
}  // namespace pypto
