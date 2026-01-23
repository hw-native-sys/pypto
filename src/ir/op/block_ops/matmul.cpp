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
 * @file matmul.cpp
 * @brief Matrix multiplication block operations
 *
 * This file implements matrix multiplication for block-level programming.
 * Block matmul operates on 2D TileTypes.
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

TypePtr DeduceBlockMatMulType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto lhs_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  auto rhs_type = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Extract shapes
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  // For block matmul, we require 2D tiles
  CHECK(lhs_shape.size() == 2) << "The operator " << op_name << " requires lhs to be 2D, but got "
                               << lhs_shape.size() << " dimensions";
  CHECK(rhs_shape.size() == 2) << "The operator " << op_name << " requires rhs to be 2D, but got "
                               << rhs_shape.size() << " dimensions";

  // Matrix multiplication: [M, K] @ [K, N] -> [M, N]
  // We need to verify that K dimensions match
  // Note: In PTO ISA, we see [M, K] @ [K, N] -> [M, N]

  ExprPtr m_dim = lhs_shape[0];
  ExprPtr k_dim_lhs = lhs_shape[1];
  ExprPtr k_dim_rhs = rhs_shape[0];
  ExprPtr n_dim = rhs_shape[1];

  // Try to verify K dimensions match if they are constant
  auto k_lhs_const = std::dynamic_pointer_cast<const ConstInt>(k_dim_lhs);
  auto k_rhs_const = std::dynamic_pointer_cast<const ConstInt>(k_dim_rhs);

  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  // Promote data types
  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  // Output shape is [M, N]
  std::vector<ExprPtr> output_shape = {m_dim, n_dim};

  return std::make_shared<TileType>(output_shape, *result_dtype);
}

// ============================================================================
// Registration Function for Block Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("block.matmul")
    .set_op_category("BlockOp")
    .set_description("Matrix multiplication of two tiles")
    .set_pipe(PipeType::MTE3)
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulType(args, kwargs, "block.matmul");
    });

}  // namespace ir
}  // namespace pypto
