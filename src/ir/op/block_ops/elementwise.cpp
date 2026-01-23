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
 * @brief Element-wise block operations (Mul, Add, Div, Sub, and scalar variants)
 *
 * This file implements element-wise block operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Operations are divided into:
 * - Tile-Tile operations (mul, add, div, sub): TileType + TileType
 * - Tile-Scalar operations (muls, adds, divs, subs): TileType + ScalarType
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto tile_type1 = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  auto tile_type2 = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());

  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();

  // Use broadcasting
  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << tile_type1->shape_ << " and " << tile_type2->shape_;

  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceBlockOpScalarBinaryType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs,
                                      const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument MUST be ScalarType
  auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  // Result has same shape as tile, with promoted dtype
  auto result_dtype = PromoteDataTypes(tile_type->dtype_, scalar_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type->dtype_.ToString() << " and " << scalar_type->dtype_.ToString();

  return std::make_shared<TileType>(tile_type->shape_, *result_dtype);
}

// ============================================================================
// Registration Function for Block Element-wise Operations
// ============================================================================

REGISTER_OP("block.mul")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    });

REGISTER_OP("block.add")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
    });

REGISTER_OP("block.div")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.div");
    });

REGISTER_OP("block.sub")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.sub");
    });

REGISTER_OP("block.maximum")
    .set_op_category("BlockOp")
    .set_description("Element-wise maximum of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.maximum");
    });

REGISTER_OP("block.muls")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.muls");
    });

REGISTER_OP("block.adds")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.adds");
    });

REGISTER_OP("block.divs")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.divs");
    });

REGISTER_OP("block.subs")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.subs");
    });

}  // namespace ir
}  // namespace pypto
