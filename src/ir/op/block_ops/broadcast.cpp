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
 * @brief Row broadcast block operations
 *
 * This file implements row-wise broadcast operations for block-level programming.
 * These operations broadcast a row vector [M, 1] to match a tile [M, N].
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockRowExpandType(const std::vector<ExprPtr>& args,
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
      << (row_col_const ? std::to_string(row_col_const->value_) : "?");

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
  return std::make_shared<TileType>(tile_shape, *result_dtype);
}

// ============================================================================
// Registration Function for Block Row Broadcast Operations
// ============================================================================

REGISTER_OP("block.row_expand_sub")
    .set_op_category("BlockOp")
    .set_description("Row-wise broadcast subtraction: tile - row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_sub");
    });

REGISTER_OP("block.row_expand_div")
    .set_op_category("BlockOp")
    .set_description("Row-wise broadcast division: tile / row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_div");
    });

REGISTER_OP("block.row_expand_mul")
    .set_op_category("BlockOp")
    .set_description("Row-wise broadcast multiplication: tile * row_vec (broadcasted)")
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_mul");
    });

REGISTER_OP("block.row_expand_add")
    .set_op_category("BlockOp")
    .set_description("Row-wise broadcast addition: tile + row_vec (broadcasted)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType, 2D [M, N])")
    .add_argument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_add");
    });

}  // namespace ir
}  // namespace pypto
