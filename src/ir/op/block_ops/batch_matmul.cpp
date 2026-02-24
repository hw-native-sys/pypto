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
 * @file batch_matmul.cpp
 * @brief Batch matrix multiplication operations for block-level programming
 *
 * This file implements batch matrix multiplication operations for TileType,
 * supporting multi-dimensional tensors with batch dimensions.
 */

#include <any>
#include <cstddef>
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
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

/**
 * @brief Deduce type for batch matrix multiplication
 *
 * Batch matmul operates on multi-dimensional TileTypes with batch dimensions.
 * For inputs with shape [...batch_dims, M, K] and [...batch_dims, K, N],
 * the output has shape [...broadcast_batch_dims, M, N].
 *
 * @param args Arguments: [lhs_tile, rhs_tile]
 * @param kwargs Keyword arguments (unused)
 * @param op_name Operator name for error messages
 * @return TileType with output shape
 */
TypePtr DeduceBlockBatchMatMulType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto lhs_type = As<TileType>(args[0]->GetType());
  auto rhs_type = As<TileType>(args[1]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Extract shapes
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  // For batch matmul, we require at least 2D tiles
  CHECK(lhs_shape.size() >= 2) << "The operator " << op_name
                               << " requires lhs to have at least 2 dimensions, but got " << lhs_shape.size()
                               << " dimensions";
  CHECK(rhs_shape.size() >= 2) << "The operator " << op_name
                               << " requires rhs to have at least 2 dimensions, but got " << rhs_shape.size()
                               << " dimensions";

  size_t lhs_ndim = lhs_shape.size();
  size_t rhs_ndim = rhs_shape.size();

  // Extract matrix dimensions (last 2 dimensions)
  ExprPtr m_dim = lhs_shape[lhs_ndim - 2];
  ExprPtr k_dim_lhs = lhs_shape[lhs_ndim - 1];
  ExprPtr k_dim_rhs = rhs_shape[rhs_ndim - 2];
  ExprPtr n_dim = rhs_shape[rhs_ndim - 1];

  // Try to verify K dimensions match if they are constant
  auto k_lhs_const = As<ConstInt>(k_dim_lhs);
  auto k_rhs_const = As<ConstInt>(k_dim_rhs);

  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  // Handle batch dimensions
  std::vector<ExprPtr> output_shape;

  if (lhs_ndim == 2 && rhs_ndim == 2) {
    // Simple 2D x 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
    output_shape = {m_dim, n_dim};
  } else {
    // Batch matrix multiplication
    // Extract batch dimensions (all except last 2)
    std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
    std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);

    // Broadcast batch dimensions
    auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
    CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for " << op_name;

    output_shape = broadcast_result.shape;

    // Append matrix dimensions: [M, N]
    output_shape.push_back(m_dim);
    output_shape.push_back(n_dim);
  }

  // Promote data types
  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  return std::make_shared<TileType>(output_shape, *result_dtype);
}

// ============================================================================
// Registration Function for Block Batch Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("block.batch_matmul")
    .set_op_category("BlockOp")
    .set_description("Batch matrix multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType, at least 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, at least 2D)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockBatchMatMulType(args, kwargs, "block.batch_matmul");
    });

}  // namespace ir
}  // namespace pypto
