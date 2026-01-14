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
 * @file elementwise.cpp
 * @brief Element-wise tile operations (Add, Sub, Mul, Div)
 *
 * This file implements element-wise tile operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTileOpElementwiseBinaryType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Try TileType
  auto tile_type1 = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  auto tile_type2 = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());

  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << tile_type1->shape_ << " and " << tile_type2->shape_;

  return std::make_shared<TileType>(*result_dtype, broadcast_result.shape);
}

// ============================================================================
// Registration Function for Tile Element-wise Operations
// ============================================================================

REGISTER_OP("tile.add")
    .set_op_category("TileOp")
    .set_description("Element-wise addition of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTileOpElementwiseBinaryType(args, "tile.add");
    });

REGISTER_OP("tile.sub")
    .set_op_category("TileOp")
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTileOpElementwiseBinaryType(args, "tile.sub");
    });

REGISTER_OP("tile.mul")
    .set_op_category("TileOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTileOpElementwiseBinaryType(args, "tile.mul");
    });

REGISTER_OP("tile.div")
    .set_op_category("TileOp")
    .set_description("Element-wise division of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTileOpElementwiseBinaryType(args, "tile.div");
    });

}  // namespace ir
}  // namespace pypto
