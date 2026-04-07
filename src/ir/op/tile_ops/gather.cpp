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
  INTERNAL_CHECK(src_shape.size() == 2)
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

}  // namespace ir
}  // namespace pypto
