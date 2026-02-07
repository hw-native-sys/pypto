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
#include "pypto/ir/expr.h"
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
  // tile.view requires exactly 3 arguments: input tile, shape tuple, and offset tuple
  CHECK(args.size() == 3) << "tile.view requires exactly 3 arguments (input, shape, offset), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.view requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tile.view requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tile.view shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tile.view shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tile.view requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < offset_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(offset_tuple_type->types_[i]);
    CHECK(scalar_type) << "tile.view offset tuple element " << i << " must be ScalarType, but got "
                       << offset_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tile.view offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract shape dimensions
  // If args[1] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[1])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    new_shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      new_shape.emplace_back(
          std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
  }

  // View preserves dtype but has new shape (which can have different rank than input)
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileReshapeType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.reshape requires exactly 2 arguments: input tile and shape tuple
  CHECK(args.size() == 2) << "tile.reshape requires exactly 2 arguments (input, shape), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.reshape requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tile.reshape requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tile.reshape shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tile.reshape shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract new shape dimensions
  // If args[1] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[1])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    new_shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      new_shape.emplace_back(
          std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
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
    .set_pipe(PipeType::V)
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileViewType(args, kwargs);
    });

REGISTER_OP("block.reshape")
    .set_op_category("BlockOp")
    .set_description("Reshape tile to new shape")
    .set_pipe(PipeType::V)
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReshapeType(args, kwargs);
    });

REGISTER_OP("block.transpose")
    .set_op_category("BlockOp")
    .set_description("Transpose tile by swapping two axes")
    .set_pipe(PipeType::V)
    .add_argument("input", "Input tile (TileType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTransposeType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
