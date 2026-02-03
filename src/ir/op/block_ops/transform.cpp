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
 * @file transform.cpp
 * @brief Shape transformation block operations (view, reshape, transpose)
 *
 * This file implements shape transformation operations for tiles including
 * view, reshape and transpose operations.
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {
// ============================================================================
// Helper Functions (file-local)
// ============================================================================

/**
 * @brief Normalize axis index to handle negative indexing
 *
 * @param axis The axis index (can be negative)
 * @param ndim The number of dimensions
 * @return The normalized axis index
 */
int NormalizeAxis(int axis, size_t ndim) {
  if (axis < 0) {
    axis += static_cast<int>(ndim);
  }
  CHECK(axis >= 0 && axis < static_cast<int>(ndim))
      << "Axis " << axis << " is out of range for " << ndim << "D tile";
  return axis;
}

/**
 * @brief Compute the product of shape dimensions (for static shapes)
 *
 * @param shape The shape dimensions
 * @return The product if all dimensions are ConstInt, -1 otherwise
 */
int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;  // Dynamic shape, cannot compute product
    }
    product *= const_dim->value_;
  }
  return product;
}

}  // anonymous namespace

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTileViewType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.view requires at least 2 arguments: input tile and shape_ndim
  // Followed by shape dimensions and offset dimensions
  CHECK(args.size() >= 2) << "tile.view requires at least 2 arguments (input, shape_ndim), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.view requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument is the number of shape dimensions (ConstInt)
  auto shape_ndim_const = As<ConstInt>(args[1]);
  CHECK(shape_ndim_const)
      << "tile.view requires second argument to be a ConstInt indicating number of shape dimensions";

  size_t shape_ndim = static_cast<size_t>(shape_ndim_const->value_);
  CHECK(shape_ndim > 0) << "tile.view requires at least 1 shape dimension";

  // Check we have enough arguments: input + shape_ndim + shape_dims + offset_dims
  CHECK(args.size() >= 2 + shape_ndim)
      << "tile.view requires at least " << (2 + shape_ndim) << " arguments for shape_ndim=" << shape_ndim
      << ", but got " << args.size();

  // Extract new shape dimensions (args[2] to args[2 + shape_ndim - 1])
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_ndim);
  for (size_t i = 0; i < shape_ndim; ++i) {
    new_shape.emplace_back(args[2 + i]);
  }

  // The remaining arguments are offset dimensions (not used for type deduction)
  // View preserves dtype but has new shape (which can have different rank than input)
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileReshapeType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.reshape requires at least 2 arguments: input tile and shape_ndim
  // Followed by shape dimensions
  CHECK(args.size() >= 2) << "tile.reshape requires at least 2 arguments (input, shape_ndim), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.reshape requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument is the number of shape dimensions (ConstInt)
  auto shape_ndim_const = As<ConstInt>(args[1]);
  CHECK(shape_ndim_const)
      << "tile.reshape requires second argument to be a ConstInt indicating number of shape dimensions";

  size_t shape_ndim = static_cast<size_t>(shape_ndim_const->value_);
  CHECK(shape_ndim > 0) << "tile.reshape requires at least 1 shape dimension";

  // Check we have correct number of arguments: input + shape_ndim + shape_dims
  CHECK(args.size() == 2 + shape_ndim)
      << "tile.reshape requires exactly " << (2 + shape_ndim) << " arguments for shape_ndim=" << shape_ndim
      << ", but got " << args.size();

  // Extract new shape dimensions (args[2] to args[2 + shape_ndim - 1])
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_ndim);
  for (size_t i = 0; i < shape_ndim; ++i) {
    new_shape.emplace_back(args[2 + i]);
  }

  // For static shapes, verify that the total number of elements matches
  int64_t old_product = ComputeShapeProduct(tile_type->shape_);
  int64_t new_product = ComputeShapeProduct(new_shape);

  if (old_product > 0 && new_product > 0) {
    CHECK(old_product == new_product) << "tile.reshape: cannot reshape tile of size " << old_product
                                      << " into shape with size " << new_product;
  }

  // Return new TileType with reshaped dimensions and same dtype
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileTransposeType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.transpose requires exactly 3 arguments: input tile, axis1, axis2
  CHECK(args.size() == 3) << "tile.transpose requires exactly 3 arguments (input, axis1, axis2), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.transpose requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const auto& input_shape = tile_type->shape_;
  size_t ndim = input_shape.size();

  CHECK(ndim >= 2) << "tile.transpose requires at least 2 dimensions, but got " << ndim;

  // Second argument is axis1 (ConstInt)
  auto axis1_const = As<ConstInt>(args[1]);
  CHECK(axis1_const) << "tile.transpose requires second argument (axis1) to be a ConstInt";

  // Third argument is axis2 (ConstInt)
  auto axis2_const = As<ConstInt>(args[2]);
  CHECK(axis2_const) << "tile.transpose requires third argument (axis2) to be a ConstInt";

  // Normalize axes (handle negative indexing)
  int axis1 = NormalizeAxis(static_cast<int>(axis1_const->value_), ndim);
  int axis2 = NormalizeAxis(static_cast<int>(axis2_const->value_), ndim);

  CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                        << ", axis2=" << axis2;

  // Create new shape by swapping the specified dimensions
  std::vector<ExprPtr> new_shape = input_shape;
  std::swap(new_shape[axis1], new_shape[axis2]);

  // Return new TileType with transposed shape and same dtype
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

// ============================================================================
// Registration Function for Tile Transform Operations
// ============================================================================

REGISTER_OP("block.view")
    .set_op_category("BlockOp")
    .set_description("Create a view/slice of a tile with new shape and offset")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape_ndim", "Number of shape dimensions (ConstInt)")
    .add_argument("shape_dims", "New shape dimensions (variable number)")
    .add_argument("offset_dims", "Offset dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileViewType(args, kwargs);
    });

REGISTER_OP("block.reshape")
    .set_op_category("BlockOp")
    .set_description("Reshape tile to new shape")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape_ndim", "Number of shape dimensions (ConstInt)")
    .add_argument("shape_dims", "New shape dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReshapeType(args, kwargs);
    });

REGISTER_OP("block.transpose")
    .set_op_category("BlockOp")
    .set_description("Transpose tile by swapping two axes")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTransposeType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
