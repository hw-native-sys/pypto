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
 * @file sort.cpp
 * @brief Sorting tile operations (Sort32)
 *
 * This file implements sort operations for tile-level programming.
 * Sort32 sorts fixed-size 32-element blocks and maps to pto.tsort32.
 */

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTileSort32Type(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name
                          << " requires 2 arguments (src, idx), but got " << args.size();

  // First arg: src tile (f16 or f32)
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src_type->dtype_.ToString();

  // Second arg: idx tile
  auto idx_type = As<TileType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Build output shape: double the last dimension for value-index pairs
  const auto& input_shape = src_type->shape_;
  CHECK(!input_shape.empty()) << "The operator " << op_name << " requires non-empty input shape";

  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  auto last_dim = input_shape.back();
  // Try constant evaluation for the common case (sort32 always uses cols=32 -> 64)
  if (auto const_dim = As<ConstInt>(last_dim)) {
    int64_t doubled = const_dim->value_ * 2;
    output_shape.push_back(std::make_shared<ConstInt>(doubled, DataType::INDEX, Span::unknown()));
  } else {
    auto two = std::make_shared<ConstInt>(2, DataType::INDEX, Span::unknown());
    output_shape.push_back(std::make_shared<Mul>(last_dim, two, DataType::INDEX, Span::unknown()));
  }

  TileView tile_view;
  tile_view.valid_shape = output_shape;
  return std::make_shared<TileType>(output_shape, src_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Registration for Sort Operations
// ============================================================================

REGISTER_OP("tile.sort32")
    .set_op_category("TileOp")
    .set_description("Sort fixed 32-element blocks (maps to pto.tsort32)")
    .add_argument("src", "Input value tile (TileType, f16 or f32)")
    .add_argument("idx", "Input index tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSort32Type(args, kwargs, "tile.sort32");
    });

}  // namespace ir
}  // namespace pypto
