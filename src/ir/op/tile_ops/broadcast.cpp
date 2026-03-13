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
 * @file broadcast.cpp
 * @brief Row broadcast tile operations
 *
 * This file implements row-wise broadcast operations for tile-level programming.
 * These operations broadcast a row vector [M, 1] to match a tile [M, N].
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTileRowExpandType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType (the main tile)
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TileType (the row vector)
  auto row_type = As<TileType>(args[1]->GetType());
  CHECK(row_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Get shapes
  const auto& tile_shape = tile_type->shape_;
  const auto& row_shape = row_type->shape_;

  // Both must have at least 2D (last 2 dimensions are used for broadcasting)
  CHECK(tile_shape.size() >= 2) << "The operator " << op_name
                                << " requires first argument to have at least 2 dimensions, but got "
                                << tile_shape.size() << " dimensions";
  CHECK(row_shape.size() >= 2) << "The operator " << op_name
                               << " requires second argument to have at least 2 dimensions, but got "
                               << row_shape.size() << " dimensions";

  // Last dimension of row vector must be 1
  auto row_col_const = As<ConstInt>(row_shape[row_shape.size() - 1]);
  CHECK(row_col_const && row_col_const->value_ == 1)
      << "The operator " << op_name << " requires second argument's last dimension to be 1, but got "
      << row_shape[row_shape.size() - 1];

  // Second-to-last dimension (rows) must match
  auto tile_rows_const = As<ConstInt>(tile_shape[tile_shape.size() - 2]);
  auto row_rows_const = As<ConstInt>(row_shape[row_shape.size() - 2]);

  if (tile_rows_const && row_rows_const) {
    CHECK(tile_rows_const->value_ == row_rows_const->value_)
        << "The operator " << op_name
        << " requires matching row dimensions, but got tile rows=" << tile_rows_const->value_
        << " and row_vec rows=" << row_rows_const->value_;
  }

  // Promote data types
  auto result_dtype = PromoteDataTypes(tile_type->dtype_, row_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type->dtype_.ToString() << " and " << row_type->dtype_.ToString();

  // Output has the same shape as the main tile
  TileView tile_view;
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, *result_dtype, std::nullopt, tile_view);
}

// Type deduction for column expand operations
TypePtr DeduceTileColExpandType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument is the target tile (shape to expand to)
  auto target_type = As<TileType>(args[0]->GetType());
  CHECK(target_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument is the column tile to expand (shape [1, cols])
  auto col_type = As<TileType>(args[1]->GetType());
  CHECK(col_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Result has same shape as target, with promoted dtype
  auto result_dtype = PromoteDataTypes(target_type->dtype_, col_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  TileView tile_view;
  tile_view.valid_shape = target_type->shape_;
  return std::make_shared<TileType>(target_type->shape_, *result_dtype, std::nullopt, tile_view);
}

// Type deduction for scalar expand operations
TypePtr DeduceTileExpandScalarType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument is the target tile
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument is the scalar to expand
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  // Result has same shape as tile, with promoted dtype
  auto result_dtype = PromoteDataTypes(tile_type->dtype_, scalar_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  TileView tile_view;
  tile_view.valid_shape = tile_type->shape_;
  return std::make_shared<TileType>(tile_type->shape_, *result_dtype, std::nullopt, tile_view);
}

// ============================================================================
// Registration Function for Block Row Broadcast Operations
// ============================================================================

REGISTER_OP("tile.row_expand")
    .set_op_category("TileOp")
    .set_description(
        "Broadcast first element of each source row across the destination row: dst[i,j] = src[i,0]")
    .add_argument("src", "Input tile (TileType, 2D [M, N])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.row_expand requires exactly 1 argument, but got "
                              << args.size();
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator tile.row_expand requires argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();
      CHECK(tile_type->shape_.size() >= 2)
          << "The operator tile.row_expand requires input tile"
          << " to have at least 2 dimensions, but got " << tile_type->shape_.size() << " dimensions";
      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
    });

REGISTER_OP("tile.row_expand_sub")
    .set_op_category("TileOp")
    .set_description("Row-wise broadcast subtraction: tile - row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowExpandType(args, kwargs, "tile.row_expand_sub");
    });

REGISTER_OP("tile.row_expand_div")
    .set_op_category("TileOp")
    .set_description("Row-wise broadcast division: tile / row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowExpandType(args, kwargs, "tile.row_expand_div");
    });

REGISTER_OP("tile.row_expand_mul")
    .set_op_category("TileOp")
    .set_description("Row-wise broadcast multiplication: tile * row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowExpandType(args, kwargs, "tile.row_expand_mul");
    });

REGISTER_OP("tile.row_expand_add")
    .set_op_category("TileOp")
    .set_description("Row-wise broadcast addition: tile + row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowExpandType(args, kwargs, "tile.row_expand_add");
    });

REGISTER_OP("tile.col_expand")
    .set_op_category("TileOp")
    .set_description("Expand column tile [1, cols] to target shape [rows, cols]")
    .add_argument("target", "Target tile defining output shape (TileType)")
    .add_argument("col_tile", "Column tile to expand (TileType, shape [1, cols])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileColExpandType(args, kwargs, "tile.col_expand");
    });

REGISTER_OP("tile.col_expand_mul")
    .set_op_category("TileOp")
    .set_description("Expand column tile and multiply with target tile")
    .add_argument("target", "Target tile (TileType)")
    .add_argument("col_tile", "Column tile to expand and multiply (TileType, shape [1, cols])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileColExpandType(args, kwargs, "tile.col_expand_mul");
    });

REGISTER_OP("tile.col_expand_div")
    .set_op_category("TileOp")
    .set_description("Expand column tile and divide target tile by it")
    .add_argument("target", "Target tile (TileType)")
    .add_argument("col_tile", "Column tile to expand and divide by (TileType, shape [1, cols])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileColExpandType(args, kwargs, "tile.col_expand_div");
    });

REGISTER_OP("tile.col_expand_sub")
    .set_op_category("TileOp")
    .set_description("Expand column tile and subtract from target tile")
    .add_argument("target", "Target tile (TileType)")
    .add_argument("col_tile", "Column tile to expand and subtract (TileType, shape [1, cols])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileColExpandType(args, kwargs, "tile.col_expand_sub");
    });

REGISTER_OP("tile.expands")
    .set_op_category("TileOp")
    .set_description("Expand scalar to target tile shape")
    .add_argument("target", "Target tile defining output shape (TileType)")
    .add_argument("scalar", "Scalar to expand (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileExpandScalarType(args, kwargs, "tile.expands");
    });

}  // namespace ir
}  // namespace pypto
