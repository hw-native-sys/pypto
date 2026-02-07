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
 * @brief Reduction block operations (Sum, Max, Min)
 *
 * This file implements reduction operations for block-level programming.
 * Reduction operations can reduce a TileType along specified axes.
 */

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceBlockReductionType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  // block.sum and block.max require 1 argument (tile) and 2 attributes (axis, keepdim)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Get the input shape
  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Determine which axes to reduce
  std::set<int64_t> reduce_axes;

  // Extract axis from kwargs (required)
  int axis_value = GetKwarg<int>(kwargs, "axis");
  if (axis_value < 0) {
    // Negative axis: convert to positive
    axis_value = static_cast<int>(input_ndim) + axis_value;
  }
  CHECK(axis_value >= 0 && static_cast<int64_t>(axis_value) < input_ndim)
      << "The operator " << op_name << " axis " << axis_value << " is out of range for shape with "
      << input_ndim << " dimensions";
  reduce_axes.insert(static_cast<int64_t>(axis_value));

  // Extract keepdim from kwargs (optional, default to false)
  bool keepdim = GetKwarg<bool>(kwargs, "keepdim", false);

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
  return std::make_shared<TileType>(output_shape, tile_type->dtype_);
}

// Type deduction for row reduction operations (reduces along last axis with keepdim=True)
TypePtr DeduceBlockRowReductionType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  // Row reduction operations require 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Get the input shape
  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Row reduction requires at least 2D tile
  CHECK(input_ndim >= 2) << "The operator " << op_name << " requires at least a 2D tile, but got "
                         << input_ndim << " dimensions";

  // Output shape is [...batch_dims, rows, 1] - reduce along last axis with keepdim=True
  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown()));

  return std::make_shared<TileType>(output_shape, tile_type->dtype_);
}

// ============================================================================
// Registration Function for Block Reduction Operations
// ============================================================================

REGISTER_OP("block.sum")
    .set_op_category("BlockOp")
    .set_description("Sum reduction of a tile along specified axis")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockReductionType(args, kwargs, "block.sum");
    });

REGISTER_OP("block.max")
    .set_op_category("BlockOp")
    .set_description("Max reduction of a tile along specified axis")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockReductionType(args, kwargs, "block.max");
    });
REGISTER_OP("block.min")
    .set_op_category("BlockOp")
    .set_description("Min reduction of a tile along specified axis")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockReductionType(args, kwargs, "block.min");
    });

// ============================================================================
// Row Reduction Operations (TROWSUM, TROWMAX, TROWMIN)
// ============================================================================

REGISTER_OP("block.row_sum")
    .set_op_category("BlockOp")
    .set_description("Row-wise sum reduction (reduces along axis=1, maps to TROWSUM)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowReductionType(args, kwargs, "block.row_sum");
    });

REGISTER_OP("block.row_max")
    .set_op_category("BlockOp")
    .set_description("Row-wise max reduction (reduces along axis=1, maps to TROWMAX)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowReductionType(args, kwargs, "block.row_max");
    });

REGISTER_OP("block.row_min")
    .set_op_category("BlockOp")
    .set_description("Row-wise min reduction (reduces along axis=1, maps to TROWMIN)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowReductionType(args, kwargs, "block.row_min");
    });

}  // namespace ir
}  // namespace pypto
