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
 * @brief Shape transformation tile operations (slice, reshape, transpose)
 *
 * This file implements shape transformation operations for tiles including
 * slice, reshape and transpose operations.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

/**
 * @brief Check whether a DataType is a valid index-like integer type
 *
 * INDEX, INT64, and UINT64 are all accepted as dimension/offset types
 * in tile operations.
 */
bool IsIndexLikeDtype(DataType dtype) {
  return dtype == DataType::INT64 || dtype == DataType::UINT64 || dtype == DataType::INDEX;
}

/**
 * @brief Validate that all elements of a TupleType are ScalarType with an index-like dtype
 *
 * @param tuple_type The tuple type whose elements to validate
 * @param op_name Name of the operation (for error messages)
 * @param arg_name Name of the argument (for error messages), e.g. "shape" or "offset"
 */
void ValidateIndexTupleElements(const TupleTypePtr& tuple_type, const std::string& op_name,
                                const std::string& arg_name) {
  for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(tuple_type->types_[i]);
    CHECK(scalar_type) << op_name << " " << arg_name << " tuple element " << i
                       << " must be ScalarType, but got " << tuple_type->types_[i]->TypeName();
    CHECK(IsIndexLikeDtype(scalar_type->dtype_))
        << op_name << " " << arg_name << " tuple element " << i
        << " must have dtype INT64, UINT64, or INDEX, but got " << scalar_type->dtype_.ToString();
  }
}

}  // anonymous namespace

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTileSliceType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.slice requires exactly 3 arguments: input tile, shape tuple, and offset tuple
  CHECK(args.size() == 3) << "tile.slice requires exactly 3 arguments (input, shape, offset), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.slice requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tile.slice requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(shape_tuple_type, "tile.slice", "shape");

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tile.slice requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(offset_tuple_type, "tile.slice", "offset");

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
  TileView tile_view;
  tile_view.valid_shape = new_shape;

  // Infer blayout from new shape: column vectors [N, 1] use col_major
  if (new_shape.size() == 2) {
    auto rows_const = As<ConstInt>(new_shape[0]);
    auto cols_const = As<ConstInt>(new_shape[1]);
    if (rows_const && cols_const) {
      if (cols_const->value_ == 1 && rows_const->value_ > 1) {
        tile_view.blayout = TileLayout::col_major;
      } else {
        tile_view.blayout = TileLayout::row_major;
      }
    } else {
      // Dynamic shape: default to row_major
      tile_view.blayout = TileLayout::row_major;
    }
  }

  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
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

  // Validate all shape elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(shape_tuple_type, "tile.reshape", "shape");

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
  TileView tile_view;
  tile_view.valid_shape = new_shape;

  // Infer blayout from new shape: column vectors [N, 1] use col_major
  if (new_shape.size() == 2) {
    auto rows_const = As<ConstInt>(new_shape[0]);
    auto cols_const = As<ConstInt>(new_shape[1]);
    if (rows_const && cols_const) {
      if (cols_const->value_ == 1 && rows_const->value_ > 1) {
        tile_view.blayout = TileLayout::col_major;
      } else {
        tile_view.blayout = TileLayout::row_major;
      }
    } else {
      // Dynamic shape: default to row_major
      tile_view.blayout = TileLayout::row_major;
    }
  }

  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
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
  TileView tile_view;
  tile_view.valid_shape = new_shape;
  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Registration Function for Tile Transform Operations
// ============================================================================

REGISTER_OP("tile.slice")
    .set_op_category("TileOp")
    .set_description("Create a slice of a tile with new shape and offset")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSliceType(args, kwargs);
    });

REGISTER_OP("tile.reshape")
    .set_op_category("TileOp")
    .set_description("Reshape tile to new shape")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReshapeType(args, kwargs);
    });

REGISTER_OP("tile.transpose")
    .set_op_category("TileOp")
    .set_description("Transpose tile by swapping two axes")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTransposeType(args, kwargs);
    });

TypePtr DeduceTileAssembleType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "tile.assemble requires exactly 3 arguments (target, source, offset), but got "
                          << args.size();

  auto target_type = As<TileType>(args[0]->GetType());
  CHECK(target_type) << "tile.assemble requires first argument (target) to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

  auto source_type = As<TileType>(args[1]->GetType());
  CHECK(source_type) << "tile.assemble requires second argument (source) to be a TileType, but got "
                     << args[1]->GetType()->TypeName();

  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tile.assemble requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  ValidateIndexTupleElements(offset_tuple_type, "tile.assemble", "offset");

  CHECK(target_type->dtype_ == source_type->dtype_)
      << "tile.assemble requires target and source to have the same dtype, but got "
      << target_type->dtype_.ToString() << " and " << source_type->dtype_.ToString();

  return std::make_shared<TileType>(target_type->shape_, target_type->dtype_);
}

REGISTER_OP("tile.assemble")
    .set_op_category("TileOp")
    .set_description("Write source tile data into target tile at specified offset")
    .add_argument("target", "Target tile (TileType)")
    .add_argument("source", "Source tile to write (TileType)")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileAssembleType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
