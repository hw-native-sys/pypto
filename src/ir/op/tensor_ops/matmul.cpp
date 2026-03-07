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

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceTensorMatMulType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.matmul requires exactly 2 Expr arguments (lhs, rhs)
  CHECK(args.size() == 2) << "tensor.matmul requires exactly 2 arguments (lhs, rhs), but got " << args.size();

  // First two arguments must be TensorType
  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());

  CHECK(lhs_type) << "tensor.matmul requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "tensor.matmul requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  // Extract shapes
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(lhs_shape.size() >= 1) << "tensor.matmul requires lhs to have at least 1 dimension";
  CHECK(rhs_shape.size() >= 1) << "tensor.matmul requires rhs to have at least 1 dimension";

  // Read kwargs (with defaults)
  DataType out_dtype;
  try {
    out_dtype = GetKwarg<DataType>(kwargs, "out_dtype");
  } catch (const ValueError& e) {
    auto promoted = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
    CHECK(promoted) << "Cannot promote data types for tensor.matmul";
    out_dtype = *promoted;
  } catch (const TypeError& e) {
    throw TypeError("Invalid kwarg type for out_dtype: " + std::string(e.what()));
  }

  bool a_trans = GetKwarg<bool>(kwargs, "a_trans", false);
  bool b_trans = GetKwarg<bool>(kwargs, "b_trans", false);

  // Compute output shape based on transpose flags
  // For 2D: lhs [M, K] x rhs [K, N] -> [M, N]
  // With transpose: lhs [K, M]^T x rhs [N, K]^T -> [M, N]

  std::vector<ExprPtr> output_shape;

  if (lhs_shape.size() == 1 && rhs_shape.size() == 1) {
    // Vector x vector (dot product): [K] x [K] -> scalar (0D tensor)
    output_shape = {};
  } else if (lhs_shape.size() == 2 && rhs_shape.size() == 1) {
    // Matrix x vector: [M, K] x [K] -> [M]
    output_shape = {lhs_shape[0]};
  } else if (lhs_shape.size() == 1 && rhs_shape.size() == 2) {
    // Vector x matrix: [K] x [K, N] -> [N]
    output_shape = {rhs_shape[1]};
  } else if (lhs_shape.size() == 2 && rhs_shape.size() == 2) {
    // 2D x 2D matrix multiplication
    ExprPtr m_dim = a_trans ? lhs_shape[1] : lhs_shape[0];
    ExprPtr n_dim = b_trans ? rhs_shape[0] : rhs_shape[1];
    output_shape = {m_dim, n_dim};
  } else {
    // For higher-dimensional tensors (both must have at least 2 dimensions),
    // use batched matmul semantics
    size_t lhs_ndim = lhs_shape.size();
    size_t rhs_ndim = rhs_shape.size();

    // Ensure both tensors have at least 2 dimensions for batched matmul
    CHECK(lhs_ndim >= 2 && rhs_ndim >= 2)
        << "tensor.matmul requires both tensors to have at least 2 dimensions "
        << "for batched matmul, but got lhs shape size " << lhs_ndim << " and rhs shape size " << rhs_ndim;

    // Extract batch dimensions (all except last 2)
    std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
    std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);

    // Broadcast batch dimensions
    auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
    CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for tensor.matmul";

    output_shape = broadcast_result.shape;

    // Append matrix dimensions
    ExprPtr m_dim = a_trans ? lhs_shape[lhs_ndim - 1] : lhs_shape[lhs_ndim - 2];
    ExprPtr n_dim = b_trans ? rhs_shape[rhs_ndim - 2] : rhs_shape[rhs_ndim - 1];
    output_shape.push_back(m_dim);
    output_shape.push_back(n_dim);
  }

  return std::make_shared<TensorType>(output_shape, out_dtype);
}

TypePtr DeduceTensorMatMulAccType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  auto acc_type = As<TensorType>(args[0]->GetType());
  auto lhs_type = As<TensorType>(args[1]->GetType());
  auto rhs_type = As<TensorType>(args[2]->GetType());
  CHECK(acc_type) << op_name << " requires acc to be a TensorType, but got " << args[0]->GetType()->TypeName();
  CHECK(lhs_type) << op_name << " requires lhs to be a TensorType, but got " << args[1]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires rhs to be a TensorType, but got " << args[2]->GetType()->TypeName();

  auto matmul_type = As<TensorType>(DeduceTensorMatMulType({args[1], args[2]}, kwargs));
  CHECK(matmul_type) << op_name << " internal error: expected TensorType matmul result";
  auto broadcast_result = BroadcastShapes(acc_type->shape_, matmul_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires accumulator shape compatible with matmul output";
  auto result_dtype = PromoteDataTypes(acc_type->dtype_, matmul_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible accumulator dtype";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorMatMulBiasType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());
  auto bias_type = As<TensorType>(args[2]->GetType());
  CHECK(lhs_type) << op_name << " requires lhs to be a TensorType, but got " << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires rhs to be a TensorType, but got " << args[1]->GetType()->TypeName();
  CHECK(bias_type) << op_name << " requires bias to be a TensorType, but got " << args[2]->GetType()->TypeName();

  auto matmul_type = As<TensorType>(DeduceTensorMatMulType({args[0], args[1]}, kwargs));
  CHECK(matmul_type) << op_name << " internal error: expected TensorType matmul result";
  auto broadcast_result = BroadcastShapes(matmul_type->shape_, bias_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires bias shape compatible with matmul output";
  auto result_dtype = PromoteDataTypes(matmul_type->dtype_, bias_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible bias dtype";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

// ============================================================================
// Registration Function for Tensor Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication of two tensors with optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulType(args, kwargs);
    });

REGISTER_OP("tensor.matmul_acc")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication with accumulation")
    .add_argument("acc", "Accumulator tensor (TensorType)")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulAccType(args, kwargs, "tensor.matmul_acc");
    });

REGISTER_OP("tensor.matmul_bias")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication with bias add")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("bias", "Bias tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulBiasType(args, kwargs, "tensor.matmul_bias");
    });

REGISTER_OP("tensor.gemv")
    .set_op_category("TensorOp")
    .set_description("General matrix-vector multiplication")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulType(args, kwargs);
    });

REGISTER_OP("tensor.gemv_acc")
    .set_op_category("TensorOp")
    .set_description("General matrix-vector multiplication with accumulation")
    .add_argument("acc", "Accumulator tensor (TensorType)")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulAccType(args, kwargs, "tensor.gemv_acc");
    });

REGISTER_OP("tensor.gemv_bias")
    .set_op_category("TensorOp")
    .set_description("General matrix-vector multiplication with bias")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("bias", "Bias tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulBiasType(args, kwargs, "tensor.gemv_bias");
    });

}  // namespace ir
}  // namespace pypto
