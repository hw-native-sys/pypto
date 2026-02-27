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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
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
  auto lhs_type = As<TileType>(args[0]->GetType());
  auto rhs_type = As<TileType>(args[1]->GetType());

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
  auto k_lhs_const = As<ConstInt>(k_dim_lhs);
  auto k_rhs_const = As<ConstInt>(k_dim_rhs);

  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  // A2A3 only support float or int32_t output, and input type must be same
  CHECK(lhs_type->dtype_ == rhs_type->dtype_);
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  // Output shape is [M, N]
  std::vector<ExprPtr> output_shape = {m_dim, n_dim};

  return std::make_shared<TileType>(output_shape, result_dtype);
}

TypePtr DeduceBlockMatMulAccType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  // All arguments must be TileType
  auto acc_type = As<TileType>(args[0]->GetType());
  auto lhs_type = As<TileType>(args[1]->GetType());
  auto rhs_type = As<TileType>(args[2]->GetType());

  CHECK(acc_type) << "The operator " << op_name << " requires first argument (acc) to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(lhs_type) << "The operator " << op_name
                  << " requires second argument (lhs) to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires third argument (rhs) to be a TileType, but got "
                  << args[2]->GetType()->TypeName();

  // Extract shapes
  const auto& acc_shape = acc_type->shape_;
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  // For block matmul_acc, we require 2D tiles
  CHECK(acc_shape.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                               << acc_shape.size() << " dimensions";
  CHECK(lhs_shape.size() == 2) << "The operator " << op_name << " requires lhs to be 2D, but got "
                               << lhs_shape.size() << " dimensions";
  CHECK(rhs_shape.size() == 2) << "The operator " << op_name << " requires rhs to be 2D, but got "
                               << rhs_shape.size() << " dimensions";

  // Matrix multiplication with accumulation: acc[M, N] += lhs[M, K] @ rhs[K, N]
  ExprPtr m_dim_acc = acc_shape[0];
  ExprPtr n_dim_acc = acc_shape[1];

  // Verify dimensions match
  auto m_acc_const = As<ConstInt>(m_dim_acc);
  auto m_lhs_const = As<ConstInt>(lhs_shape[0]);
  auto n_acc_const = As<ConstInt>(n_dim_acc);
  auto n_rhs_const = As<ConstInt>(rhs_shape[1]);
  auto k_lhs_const = As<ConstInt>(lhs_shape[1]);
  auto k_rhs_const = As<ConstInt>(rhs_shape[0]);

  if (m_acc_const && m_lhs_const) {
    CHECK(m_acc_const->value_ == m_lhs_const->value_)
        << "The operator " << op_name
        << " requires matching M dimensions, but got acc M=" << m_acc_const->value_
        << " and lhs M=" << m_lhs_const->value_;
  }

  if (n_acc_const && n_rhs_const) {
    CHECK(n_acc_const->value_ == n_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching N dimensions, but got acc N=" << n_acc_const->value_
        << " and rhs N=" << n_rhs_const->value_;
  }

  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching K dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  // Promote data types
  auto lhs_rhs_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(lhs_rhs_dtype) << "The operator " << op_name
                       << " requires compatible lhs and rhs data types, but got "
                       << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  auto result_dtype = PromoteDataTypes(acc_type->dtype_, *lhs_rhs_dtype);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible accumulator data type, but got "
                      << acc_type->dtype_.ToString() << " and " << lhs_rhs_dtype->ToString();

  // Output shape is [M, N] (same as accumulator)
  std::vector<ExprPtr> output_shape = {m_dim_acc, n_dim_acc};

  return std::make_shared<TileType>(output_shape, *result_dtype);
}

TypePtr DeduceBlockMatMulBiasType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto lhs_type = As<TileType>(args[0]->GetType());
  auto rhs_type = As<TileType>(args[1]->GetType());
  auto bias_type = As<TileType>(args[2]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires first argument (lhs) to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name
                  << " requires second argument (rhs) to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(bias_type) << "The operator " << op_name
                   << " requires third argument (bias) to be a TileType, but got "
                   << args[2]->GetType()->TypeName();

  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;
  const auto& bias_shape = bias_type->shape_;

  CHECK(lhs_shape.size() == 2) << "The operator " << op_name << " requires lhs to be 2D, but got "
                               << lhs_shape.size() << " dimensions";
  CHECK(rhs_shape.size() == 2) << "The operator " << op_name << " requires rhs to be 2D, but got "
                               << rhs_shape.size() << " dimensions";
  CHECK(bias_shape.size() == 2) << "The operator " << op_name << " requires bias to be 2D, but got "
                                << bias_shape.size() << " dimensions";

  auto k_lhs_const = As<ConstInt>(lhs_shape[1]);
  auto k_rhs_const = As<ConstInt>(rhs_shape[0]);
  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  std::vector<ExprPtr> output_shape = {lhs_shape[0], rhs_shape[1]};

  // Hardware requires bias to be [1, N]
  auto bias_row_const = As<ConstInt>(bias_shape[0]);
  CHECK(bias_row_const && bias_row_const->value_ == 1)
      << "The operator " << op_name << " requires bias to have shape [1, N], but got "
      << FormatShape(bias_shape);
  auto bias_n_const = As<ConstInt>(bias_shape[1]);
  auto rhs_n_const = As<ConstInt>(rhs_shape[1]);
  if (bias_n_const && rhs_n_const) {
    CHECK(bias_n_const->value_ == rhs_n_const->value_)
        << "The operator " << op_name
        << " requires bias N dimension to match output N=" << rhs_n_const->value_
        << ", but got bias N=" << bias_n_const->value_;
  }

  auto lhs_rhs_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(lhs_rhs_dtype) << "The operator " << op_name << " requires compatible lhs/rhs data types, but got "
                       << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  auto result_dtype = PromoteDataTypes(*lhs_rhs_dtype, bias_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible bias data type, but got "
                      << lhs_rhs_dtype->ToString() << " and " << bias_type->dtype_.ToString();
  return std::make_shared<TileType>(output_shape, *result_dtype);
}

// ============================================================================
// Registration Function for Block Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("block.matmul")
    .set_op_category("BlockOp")
    .set_description("Matrix multiplication of two tiles")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulType(args, kwargs, "block.matmul");
    });

REGISTER_OP("block.matmul_acc")
    .set_op_category("BlockOp")
    .set_description("Matrix multiplication with accumulation: acc = acc + lhs @ rhs")
    .add_argument("acc", "Accumulator tile (TileType, 2D)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulAccType(args, kwargs, "block.matmul_acc");
    });

REGISTER_OP("block.matmul_bias")
    .set_op_category("BlockOp")
    .set_description("Matrix multiplication with bias add: C = lhs @ rhs + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .add_argument("bias", "Bias tile (TileType, [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulBiasType(args, kwargs, "block.matmul_bias");
    });

REGISTER_OP("block.gemv")
    .set_op_category("BlockOp")
    .set_description("General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulType(args, kwargs, "block.gemv");
    });

REGISTER_OP("block.gemv_acc")
    .set_op_category("BlockOp")
    .set_description("GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N]")
    .add_argument("acc", "Accumulator tile (TileType, 2D [1, N])")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulAccType(args, kwargs, "block.gemv_acc");
    });

REGISTER_OP("block.gemv_bias")
    .set_op_category("BlockOp")
    .set_description("GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .add_argument("bias", "Bias tile (TileType, [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMatMulBiasType(args, kwargs, "block.gemv_bias");
    });

}  // namespace ir
}  // namespace pypto
