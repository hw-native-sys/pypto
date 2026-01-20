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
 * @brief Element-wise block operations (Mul, Add, Div)
 *
 * This file implements element-wise block operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Block operations work on TileType, similar to tile operations.
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

  // First argument must be TileType
  auto tile_type1 = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

  // Second argument can be TileType or ScalarType
  auto tile_type2 = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());
  auto scalar_type2 = std::dynamic_pointer_cast<const ScalarType>(args[1]->GetType());

  if (tile_type2) {
    // Both are TileType - use broadcasting
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << tile_type1->shape_ << " and " << tile_type2->shape_;

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
  } else if (scalar_type2) {
    // TileType + ScalarType - result is TileType with same shape as first argument
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, scalar_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    return std::make_shared<TileType>(tile_type1->shape_, *result_dtype);
  } else {
    CHECK(false) << "The operator " << op_name
                 << " requires second argument to be a TileType or ScalarType, but got "
                 << args[1]->GetType()->TypeName();
    return nullptr;
  }
}

// ============================================================================
// Registration Function for Block Element-wise Operations
// ============================================================================

REGISTER_OP("block.mul")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of two tiles or tile and scalar with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    });

REGISTER_OP("block.add")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of two tiles or tile and scalar with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
    });

REGISTER_OP("block.div")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of two tiles or tile and scalar with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.div");
    });

}  // namespace ir
}  // namespace pypto
