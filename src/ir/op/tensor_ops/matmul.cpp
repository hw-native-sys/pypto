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
    auto k_lhs_const = As<ConstInt>(lhs_shape[0]);
    auto k_rhs_const = As<ConstInt>(rhs_shape[0]);
    if (k_lhs_const && k_rhs_const) {
      CHECK(k_lhs_const->value_ == k_rhs_const->value_)
          << "tensor.matmul dot product requires matching dimensions, but got lhs K=" << k_lhs_const->value_
          << " and rhs K=" << k_rhs_const->value_;
    }
    output_shape = {};
  } else if (lhs_shape.size() == 2 && rhs_shape.size() == 1) {
    // Matrix x vector: [M, K] x [K] -> [M]
    auto k_lhs_const = As<ConstInt>(lhs_shape[1]);
    auto k_rhs_const = As<ConstInt>(rhs_shape[0]);
    if (k_lhs_const && k_rhs_const) {
      CHECK(k_lhs_const->value_ == k_rhs_const->value_)
          << "tensor.matmul requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
          << " and rhs K=" << k_rhs_const->value_;
    }
    output_shape = {lhs_shape[0]};
  } else if (lhs_shape.size() == 1 && rhs_shape.size() == 2) {
    // Vector x matrix: [K] x [K, N] -> [N]
    auto k_lhs_const = As<ConstInt>(lhs_shape[0]);
    auto k_rhs_const = As<ConstInt>(rhs_shape[0]);
    if (k_lhs_const && k_rhs_const) {
      CHECK(k_lhs_const->value_ == k_rhs_const->value_)
          << "tensor.matmul requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
          << " and rhs K=" << k_rhs_const->value_;
    }
    output_shape = {rhs_shape[1]};
  } else if (lhs_shape.size() == 2 && rhs_shape.size() == 2) {
    // 2D x 2D matrix multiplication
    ExprPtr m_dim = a_trans ? lhs_shape[1] : lhs_shape[0];
    ExprPtr k_lhs = a_trans ? lhs_shape[0] : lhs_shape[1];
    ExprPtr k_rhs = b_trans ? rhs_shape[1] : rhs_shape[0];
    ExprPtr n_dim = b_trans ? rhs_shape[0] : rhs_shape[1];

    // Verify K dimensions match (when statically known)
    auto k_lhs_const = As<ConstInt>(k_lhs);
    auto k_rhs_const = As<ConstInt>(k_rhs);
    if (k_lhs_const && k_rhs_const) {
      CHECK(k_lhs_const->value_ == k_rhs_const->value_)
          << "tensor.matmul requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
          << " and rhs K=" << k_rhs_const->value_;
    }

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
    ExprPtr k_lhs = a_trans ? lhs_shape[lhs_ndim - 2] : lhs_shape[lhs_ndim - 1];
    ExprPtr k_rhs = b_trans ? rhs_shape[rhs_ndim - 1] : rhs_shape[rhs_ndim - 2];
    ExprPtr n_dim = b_trans ? rhs_shape[rhs_ndim - 2] : rhs_shape[rhs_ndim - 1];

    // Verify K dimensions match (when statically known)
    auto k_lhs_const = As<ConstInt>(k_lhs);
    auto k_rhs_const = As<ConstInt>(k_rhs);
    if (k_lhs_const && k_rhs_const) {
      CHECK(k_lhs_const->value_ == k_rhs_const->value_)
          << "tensor.matmul requires matching inner dimensions for batched matmul, but got lhs K="
          << k_lhs_const->value_ << " and rhs K=" << k_rhs_const->value_;
    }

    output_shape.push_back(m_dim);
    output_shape.push_back(n_dim);
  }

  return std::make_shared<TensorType>(output_shape, out_dtype);
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

  CHECK(acc_shape.size() >= 2) << "tensor.matmul_acc requires acc to be at least 2D, but got "
                               << acc_shape.size() << "D";
  CHECK(lhs_shape.size() >= 2) << "tensor.matmul_acc requires lhs to be at least 2D, but got "
                               << lhs_shape.size() << "D";
  CHECK(rhs_shape.size() >= 2) << "tensor.matmul_acc requires rhs to be at least 2D, but got "
                               << rhs_shape.size() << "D";

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

  // Trailing matrix dims (after applying transpose).
  size_t lhs_ndim = lhs_shape.size();
  size_t rhs_ndim = rhs_shape.size();
  ExprPtr m_dim = a_trans ? lhs_shape[lhs_ndim - 1] : lhs_shape[lhs_ndim - 2];
  ExprPtr k_lhs = a_trans ? lhs_shape[lhs_ndim - 2] : lhs_shape[lhs_ndim - 1];
  ExprPtr k_rhs = b_trans ? rhs_shape[rhs_ndim - 1] : rhs_shape[rhs_ndim - 2];
  ExprPtr n_dim = b_trans ? rhs_shape[rhs_ndim - 2] : rhs_shape[rhs_ndim - 1];

  // Verify K dimensions match.
  auto k_lhs_const = As<ConstInt>(k_lhs);
  auto k_rhs_const = As<ConstInt>(k_rhs);
  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "tensor.matmul_acc: lhs K=" << k_lhs_const->value_ << " != rhs K=" << k_rhs_const->value_;
  }

  // Verify trailing M, N match acc.
  size_t acc_ndim = acc_shape.size();
  auto m_acc = As<ConstInt>(acc_shape[acc_ndim - 2]);
  auto n_acc = As<ConstInt>(acc_shape[acc_ndim - 1]);
  auto m_lhs_const = As<ConstInt>(m_dim);
  auto n_rhs_const = As<ConstInt>(n_dim);
  if (m_acc && m_lhs_const) {
    CHECK(m_acc->value_ == m_lhs_const->value_)
        << "tensor.matmul_acc: acc M=" << m_acc->value_ << " != matmul M=" << m_lhs_const->value_;
  }
  if (n_acc && n_rhs_const) {
    CHECK(n_acc->value_ == n_rhs_const->value_)
        << "tensor.matmul_acc: acc N=" << n_acc->value_ << " != matmul N=" << n_rhs_const->value_;
  }

  // Batch dim handling: lhs and rhs batch dims broadcast against each other; the
  // result must equal the acc batch dims (acc is the in-place accumulation target
  // and is not broadcast).
  std::vector<ExprPtr> acc_batch(acc_shape.begin(), acc_shape.end() - 2);
  std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
  std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
  CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for tensor.matmul_acc";
  CHECK(broadcast_result.shape.size() == acc_batch.size())
      << "tensor.matmul_acc: acc batch rank (" << acc_batch.size()
      << ") must equal broadcast(lhs, rhs) batch rank (" << broadcast_result.shape.size() << ")";
  for (size_t i = 0; i < acc_batch.size(); ++i) {
    auto acc_const = As<ConstInt>(acc_batch[i]);
    auto bcast_const = As<ConstInt>(broadcast_result.shape[i]);
    if (acc_const && bcast_const) {
      CHECK(acc_const->value_ == bcast_const->value_)
          << "tensor.matmul_acc: acc batch dim " << i << " (" << acc_const->value_
          << ") must equal broadcast(lhs, rhs) batch dim " << i << " (" << bcast_const->value_ << ")";
    }
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
