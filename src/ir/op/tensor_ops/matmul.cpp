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
 * @brief Matrix multiplication tensor operations
 *
 * This file implements matrix multiplication operations for tensors,
 * supporting transpose options and output dtype control.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorMatMulType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  // tensor.matmul requires at least 2 arguments (lhs, rhs)
  // Optional args: outDtype, aTrans, bTrans, cMatrixNz
  CHECK(args.size() >= 2) << "The operator " << op_name << " requires at least 2 arguments, but got "
                          << args.size();

  // First two arguments must be TensorType
  auto lhs_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs_type = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  // Extract shapes
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(lhs_shape.size() >= 1) << "The operator " << op_name << " requires lhs to have at least 1 dimension";
  CHECK(rhs_shape.size() >= 1) << "The operator " << op_name << " requires rhs to have at least 1 dimension";

  // Determine output dtype
  DataType out_dtype;
  if (args.size() >= 3) {
    // outDtype is provided as third argument (ConstInt representing DataType enum)
    auto dtype_const = std::dynamic_pointer_cast<const ConstInt>(args[2]);
    if (dtype_const) {
      out_dtype = static_cast<DataType>(dtype_const->value_);
    } else {
      // If not provided or not a ConstInt, promote from input types
      auto promoted = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
      CHECK(promoted) << "Cannot promote data types for " << op_name;
      out_dtype = *promoted;
    }
  } else {
    // Default: promote from input types
    auto promoted = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
    CHECK(promoted) << "Cannot promote data types for " << op_name;
    out_dtype = *promoted;
  }

  // Extract transpose flags (args[3] and args[4] are aTrans and bTrans)
  bool a_trans = false;
  bool b_trans = false;

  if (args.size() >= 5) {
    auto a_trans_const = std::dynamic_pointer_cast<const ConstInt>(args[3]);
    auto b_trans_const = std::dynamic_pointer_cast<const ConstInt>(args[4]);
    if (a_trans_const) {
      a_trans = a_trans_const->value_ != 0;
    }
    if (b_trans_const) {
      b_trans = b_trans_const->value_ != 0;
    }
  }

  // Compute output shape based on transpose flags
  // For 2D: lhs [M, K] x rhs [K, N] -> [M, N]
  // With transpose: lhs [K, M]^T x rhs [N, K]^T -> [M, N]

  std::vector<ExprPtr> output_shape;

  if (lhs_shape.size() == 2 && rhs_shape.size() == 2) {
    // 2D x 2D matrix multiplication
    ExprPtr m_dim = a_trans ? lhs_shape[1] : lhs_shape[0];
    ExprPtr n_dim = b_trans ? rhs_shape[0] : rhs_shape[1];
    output_shape = {m_dim, n_dim};
  } else if (lhs_shape.size() == 2 && rhs_shape.size() == 1) {
    // Matrix x vector: [M, K] x [K] -> [M]
    output_shape = {lhs_shape[0]};
  } else if (lhs_shape.size() == 1 && rhs_shape.size() == 2) {
    // Vector x matrix: [K] x [K, N] -> [N]
    output_shape = {rhs_shape[1]};
  } else {
    // For higher-dimensional tensors, use batched matmul semantics
    // Output shape is broadcast of batch dimensions + [M, N]
    size_t lhs_ndim = lhs_shape.size();
    size_t rhs_ndim = rhs_shape.size();

    // Extract batch dimensions (all except last 2)
    std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
    std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);

    // Broadcast batch dimensions
    auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
    CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for " << op_name;

    output_shape = broadcast_result.shape;

    // Append matrix dimensions
    ExprPtr m_dim = a_trans ? lhs_shape[lhs_ndim - 1] : lhs_shape[lhs_ndim - 2];
    ExprPtr n_dim = b_trans ? rhs_shape[rhs_ndim - 2] : rhs_shape[rhs_ndim - 1];
    output_shape.push_back(m_dim);
    output_shape.push_back(n_dim);
  }

  return std::make_shared<TensorType>(out_dtype, output_shape);
}

// ============================================================================
// Registration Function for Tensor Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication of two tensors with optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("outDtype", "Output data type (optional, ConstInt)")
    .add_argument("aTrans", "Transpose lhs (optional, ConstInt bool)")
    .add_argument("bTrans", "Transpose rhs (optional, ConstInt bool)")
    .add_argument("cMatrixNz", "C matrix non-zero flag (optional, ConstInt bool)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorMatMulType(args, "tensor.matmul");
    });

}  // namespace ir
}  // namespace pypto
