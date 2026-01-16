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
 * @file reduction.cpp
 * @brief Reduction block operations (Sum)
 *
 * This file implements reduction operations for block-level programming.
 * Reduction operations can reduce a TileType along specified axes.
 */

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockSumType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  // block.sum requires 2 or 3 arguments: (tile, axes, keepdim?)
  CHECK(args.size() >= 2 && args.size() <= 3)
      << "The operator " << op_name << " requires 2 or 3 arguments, but got " << args.size();

  // First argument must be TileType
  auto tile_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must specify reduction axes
  // Get the input shape
  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Determine which axes to reduce
  std::set<int64_t> reduce_axes;

  // Extract axis from second argument
  // For now, we support a single ConstInt representing a single axis
  // In the future, this could be extended to support a list of axes
  const auto& axis_expr = args[1];
  auto const_axis = std::dynamic_pointer_cast<const ConstInt>(axis_expr);

  CHECK(const_axis) << "The operator " << op_name
                    << " requires second argument to be a ConstInt representing the reduction axis";

  // Single axis specified
  int axis_value = const_axis->value_;
  if (axis_value < 0) {
    // Negative axis: convert to positive
    axis_value = static_cast<int>(input_ndim) + axis_value;
  }
  CHECK(axis_value >= 0 && static_cast<int64_t>(axis_value) < input_ndim)
      << "The operator " << op_name << " axis " << axis_value << " is out of range for shape with "
      << input_ndim << " dimensions";
  reduce_axes.insert(static_cast<int64_t>(axis_value));

  // Extract keepdim from third argument (optional, default to false)
  bool keepdim = false;
  if (args.size() == 3) {
    const auto& keepdim_expr = args[2];
    auto const_keepdim = std::dynamic_pointer_cast<const ConstInt>(keepdim_expr);
    CHECK(const_keepdim) << "The operator " << op_name
                         << " requires third argument to be a ConstInt representing keepdim (0 or 1)";
    CHECK(const_keepdim->value_ == 0 || const_keepdim->value_ == 1)
        << "The operator " << op_name << " requires keepdim to be 0 or 1, got " << const_keepdim->value_;
    keepdim = (const_keepdim->value_ == 1);
  }

  // If all axes are reduced and keepdim is false, return ScalarType
  if (static_cast<int64_t>(reduce_axes.size()) == input_ndim && !keepdim) {
    return std::make_shared<ScalarType>(tile_type->dtype_);
  }

  // Build output shape
  std::vector<ExprPtr> output_shape;
  if (keepdim) {
    // When keepdim is true, keep all dimensions but set reduced axes to 1
    for (int64_t i = 0; i < input_ndim; ++i) {
      if (reduce_axes.find(i) != reduce_axes.end()) {
        // Reduced axis: set to 1
        output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown()));
      } else {
        // Keep this dimension
        output_shape.push_back(input_shape[i]);
      }
    }
  } else {
    // When keepdim is false, remove reduced axes
    for (int64_t i = 0; i < input_ndim; ++i) {
      if (reduce_axes.find(i) == reduce_axes.end()) {
        // Keep this dimension
        output_shape.push_back(input_shape[i]);
      }
    }
  }

  // If output shape is empty, return ScalarType
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(tile_type->dtype_);
  }

  // Return TileType with reduced shape
  return std::make_shared<TileType>(tile_type->dtype_, output_shape);
}

// ============================================================================
// Registration Function for Block Reduction Operations
// ============================================================================

REGISTER_OP("block.sum")
    .set_op_category("BlockOp")
    .set_description("Sum reduction of a tile along specified axes")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("axes", "Reduction axes (required)")
    .add_argument("keepdim", "Keep reduced dimensions as 1 (optional, default false)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) { return DeduceBlockSumType(args, "block.sum"); });

}  // namespace ir
}  // namespace pypto
