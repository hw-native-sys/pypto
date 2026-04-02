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
#include "pypto/ir/scalar_expr.h"
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

DataType DeduceTensorMatMulOutDType(const TensorTypePtr& lhs_type, const TensorTypePtr& rhs_type,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  try {
    return GetKwarg<DataType>(kwargs, "out_dtype");
  } catch (const ValueError&) {
    auto promoted = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
    CHECK(promoted) << "Cannot promote data types for " << op_name;
    return *promoted;
  } catch (const TypeError& e) {
    throw TypeError("Invalid kwarg type for out_dtype: " + std::string(e.what()));
  }
}

void CheckStaticInnerDimsMatch(const ExprPtr& lhs_k, const ExprPtr& rhs_k, const std::string& op_name) {
  auto lhs_k_const = As<ConstInt>(lhs_k);
  auto rhs_k_const = As<ConstInt>(rhs_k);
  if (lhs_k_const && rhs_k_const) {
    CHECK(lhs_k_const->value_ == rhs_k_const->value_)
        << op_name << " requires matching inner dimensions, but got lhs K=" << lhs_k_const->value_
        << " and rhs K=" << rhs_k_const->value_;
  }
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

  CHECK(lhs_shape.size() == 2) << "tensor.matmul requires lhs to be 2D, but got " << lhs_shape.size()
                               << "D. Use tensor.batch_matmul for batched inputs.";
  CHECK(rhs_shape.size() == 2) << "tensor.matmul requires rhs to be 2D, but got " << rhs_shape.size()
                               << "D. Use tensor.batch_matmul for batched inputs.";

  // Read kwargs (with defaults)
  DataType out_dtype = DeduceTensorMatMulOutDType(lhs_type, rhs_type, kwargs, "tensor.matmul");
  bool a_trans = GetKwarg<bool>(kwargs, "a_trans", false);
  bool b_trans = GetKwarg<bool>(kwargs, "b_trans", false);

  ExprPtr m_dim = a_trans ? lhs_shape[1] : lhs_shape[0];
  ExprPtr lhs_k = a_trans ? lhs_shape[0] : lhs_shape[1];
  ExprPtr rhs_k = b_trans ? rhs_shape[1] : rhs_shape[0];
  ExprPtr n_dim = b_trans ? rhs_shape[0] : rhs_shape[1];
  CheckStaticInnerDimsMatch(lhs_k, rhs_k, "tensor.matmul");

  return std::make_shared<TensorType>(std::vector<ExprPtr>{m_dim, n_dim}, out_dtype);
}

TypePtr DeduceTensorBatchMatMulType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "tensor.batch_matmul requires exactly 2 arguments (lhs, rhs), but got "
                          << args.size();

  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());

  CHECK(lhs_type) << "tensor.batch_matmul requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "tensor.batch_matmul requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(lhs_shape.size() >= 3) << "tensor.batch_matmul requires lhs to have at least 3 dimensions, but got "
                               << lhs_shape.size() << "D. Use tensor.matmul for 2D matrix multiplication.";
  CHECK(rhs_shape.size() >= 3) << "tensor.batch_matmul requires rhs to have at least 3 dimensions, but got "
                               << rhs_shape.size() << "D. Use tensor.matmul for 2D matrix multiplication.";

  DataType out_dtype = DeduceTensorMatMulOutDType(lhs_type, rhs_type, kwargs, "tensor.batch_matmul");
  bool a_trans = GetKwarg<bool>(kwargs, "a_trans", false);
  bool b_trans = GetKwarg<bool>(kwargs, "b_trans", false);

  std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
  std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
  CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for tensor.batch_matmul";

  size_t lhs_ndim = lhs_shape.size();
  size_t rhs_ndim = rhs_shape.size();
  ExprPtr m_dim = a_trans ? lhs_shape[lhs_ndim - 1] : lhs_shape[lhs_ndim - 2];
  ExprPtr lhs_k = a_trans ? lhs_shape[lhs_ndim - 2] : lhs_shape[lhs_ndim - 1];
  ExprPtr rhs_k = b_trans ? rhs_shape[rhs_ndim - 1] : rhs_shape[rhs_ndim - 2];
  ExprPtr n_dim = b_trans ? rhs_shape[rhs_ndim - 2] : rhs_shape[rhs_ndim - 1];
  CheckStaticInnerDimsMatch(lhs_k, rhs_k, "tensor.batch_matmul");

  std::vector<ExprPtr> output_shape = broadcast_result.shape;
  output_shape.push_back(m_dim);
  output_shape.push_back(n_dim);
  return std::make_shared<TensorType>(output_shape, out_dtype);
}

// ============================================================================
// Registration Function for Tensor Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("2D matrix multiplication of two tensors with optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType, 2D)")
    .add_argument("rhs", "Right-hand side tensor (TensorType, 2D)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulType(args, kwargs);
    });

REGISTER_OP("tensor.batch_matmul")
    .set_op_category("TensorOp")
    .set_description("Batched matrix multiplication of tensors with rank >= 3 and optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType, rank >= 3)")
    .add_argument("rhs", "Right-hand side tensor (TensorType, rank >= 3)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorBatchMatMulType(args, kwargs);
    });

// ============================================================================
// tensor.matmul_acc: Matrix multiplication with accumulation
// ============================================================================

TypePtr DeduceTensorMatMulAccType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "tensor.matmul_acc requires exactly 3 arguments (acc, lhs, rhs), but got "
                          << args.size();

  auto acc_type = As<TensorType>(args[0]->GetType());
  auto lhs_type = As<TensorType>(args[1]->GetType());
  auto rhs_type = As<TensorType>(args[2]->GetType());

  CHECK(acc_type) << "tensor.matmul_acc requires first argument (acc) to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(lhs_type) << "tensor.matmul_acc requires second argument (lhs) to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(rhs_type) << "tensor.matmul_acc requires third argument (rhs) to be a TensorType, but got "
                  << args[2]->GetType()->TypeName();

  const auto& acc_shape = acc_type->shape_;
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(acc_shape.size() == 2) << "tensor.matmul_acc requires acc to be 2D, but got " << acc_shape.size()
                               << "D";
  CHECK(lhs_shape.size() == 2) << "tensor.matmul_acc requires lhs to be 2D, but got " << lhs_shape.size()
                               << "D";
  CHECK(rhs_shape.size() == 2) << "tensor.matmul_acc requires rhs to be 2D, but got " << rhs_shape.size()
                               << "D";

  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "tensor.matmul_acc requires identical lhs and rhs dtypes, but got " << lhs_type->dtype_.ToString()
      << " and " << rhs_type->dtype_.ToString();

  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;
  CHECK(acc_type->dtype_ == result_dtype)
      << "tensor.matmul_acc requires accumulator dtype " << result_dtype.ToString() << ", but got "
      << acc_type->dtype_.ToString();

  bool a_trans = GetKwarg<bool>(kwargs, "a_trans", false);
  bool b_trans = GetKwarg<bool>(kwargs, "b_trans", false);

  // acc[M, N] += lhs[M, K] @ rhs[K, N] (with optional transpose)
  ExprPtr m_dim = a_trans ? lhs_shape[1] : lhs_shape[0];
  ExprPtr lhs_k = a_trans ? lhs_shape[0] : lhs_shape[1];
  ExprPtr rhs_k = b_trans ? rhs_shape[1] : rhs_shape[0];
  ExprPtr n_dim = b_trans ? rhs_shape[0] : rhs_shape[1];

  CheckStaticInnerDimsMatch(lhs_k, rhs_k, "tensor.matmul_acc");

  // Verify acc shape matches [M, N]
  auto m_acc = As<ConstInt>(acc_shape[0]);
  auto m_lhs = As<ConstInt>(m_dim);
  if (m_acc && m_lhs) {
    CHECK(m_acc->value_ == m_lhs->value_)
        << "tensor.matmul_acc: acc M=" << m_acc->value_ << " != matmul M=" << m_lhs->value_;
  }
  auto n_acc = As<ConstInt>(acc_shape[1]);
  auto n_rhs = As<ConstInt>(n_dim);
  if (n_acc && n_rhs) {
    CHECK(n_acc->value_ == n_rhs->value_)
        << "tensor.matmul_acc: acc N=" << n_acc->value_ << " != matmul N=" << n_rhs->value_;
  }

  return std::make_shared<TensorType>(acc_shape, result_dtype);
}

REGISTER_OP("tensor.matmul_acc")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication with accumulation: acc = acc + lhs @ rhs")
    .add_argument("acc", "Accumulator tensor (TensorType)")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMatMulAccType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
