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
 * @brief Matrix multiplication tile operations
 *
 * This file implements matrix multiplication for tile-level programming.
 * Block matmul operates on 2D TileTypes.
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTileMatMulType(const std::vector<ExprPtr>& args,
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

  // For tile matmul, we require 2D tiles
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
  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  // Output shape is [M, N]
  std::vector<ExprPtr> output_shape = {m_dim, n_dim};

  // Acc layout: Nz
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = output_shape;

  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileMatMulAccType(const std::vector<ExprPtr>& args,
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

  // For tile matmul_acc, we require 2D tiles
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

  // A2A3 only support float or int32_t output, and input type must be same
  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  CHECK(acc_type->dtype_ == result_dtype)
      << "The operator " << op_name << " requires accumulator dtype " << result_dtype.ToString()
      << ", but got " << acc_type->dtype_.ToString();

  // Output shape is [M, N] (same as accumulator)
  std::vector<ExprPtr> output_shape = {m_dim_acc, n_dim_acc};

  // Acc layout: Nz
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = output_shape;

  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileMatMulBiasType(const std::vector<ExprPtr>& args,
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

  TileView tile_view;
  tile_view.valid_shape = output_shape;
  return std::make_shared<TileType>(output_shape, *result_dtype, std::nullopt, tile_view);
}

// ============================================================================
// Registration Function for Block Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tile.matmul")
    .set_op_category("TileOp")
    .set_description("Matrix multiplication of two tiles")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulType(args, kwargs, "tile.matmul");
    });

REGISTER_OP("tile.matmul_acc")
    .set_op_category("TileOp")
    .set_description("Matrix multiplication with accumulation: acc = acc + lhs @ rhs")
    .add_argument("acc", "Accumulator tile (TileType, 2D)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulAccType(args, kwargs, "tile.matmul_acc");
    });

REGISTER_OP("tile.matmul_bias")
    .set_op_category("TileOp")
    .set_description("Matrix multiplication with bias add: C = lhs @ rhs + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .add_argument("bias", "Bias tile (TileType, [1, N])")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_input_memory(2, MemorySpace::Bias)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulBiasType(args, kwargs, "tile.matmul_bias");
    });

static bool IsMxDataDtype(DataType dtype) {
  return dtype == DataType::FP8E4M3FN || dtype == DataType::FP8E5M2 || dtype == DataType::FP4;
}

static void CheckMxScaleTile(const TileTypePtr& scale_type, const ExprPtr& expected_rows,
                             const ExprPtr& expected_cols, const std::string& op_name,
                             const char* scale_name) {
  CHECK(scale_type) << "The operator " << op_name << " requires " << scale_name << " to be a TileType";
  CHECK(scale_type->dtype_ == DataType::FP8E8M0)
      << "The operator " << op_name << " requires " << scale_name << " dtype FP8E8M0, but got "
      << scale_type->dtype_.ToString();
  CHECK(scale_type->shape_.size() == 2)
      << "The operator " << op_name << " requires " << scale_name << " to be 2D, but got "
      << scale_type->shape_.size() << " dimensions";

  auto er = As<ConstInt>(expected_rows);
  auto ec = As<ConstInt>(expected_cols);
  auto sr = As<ConstInt>(scale_type->shape_[0]);
  auto sc = As<ConstInt>(scale_type->shape_[1]);
  if (er && sr) {
    CHECK(er->value_ == sr->value_) << "The operator " << op_name << " requires " << scale_name
                                    << " rows=" << er->value_ << ", but got " << sr->value_;
  }
  if (ec && sc) {
    CHECK(ec->value_ == sc->value_) << "The operator " << op_name << " requires " << scale_name
                                    << " cols=" << ec->value_ << ", but got " << sc->value_;
  }
}

static ExprPtr MxScaleKFromK(const ExprPtr& k_dim, const std::string& op_name) {
  auto k_const = As<ConstInt>(k_dim);
  if (k_const) {
    CHECK(k_const->value_ > 0 && k_const->value_ % 32 == 0)
        << "The operator " << op_name
        << " requires K divisible by 32 for MX block-scale, but got K=" << k_const->value_;
    return std::make_shared<ConstInt>(k_const->value_ / 32, DataType::INDEX, Span::unknown());
  }
  // Dynamic K: scaleK is still K/32 conceptually; leave as symbolic floor-div is not
  // available here — callers with dynamic K must supply matching scale shapes.
  return nullptr;
}

TypePtr DeduceTileMatMulMxType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires exactly 4 arguments (lhs, lhs_scale, rhs, rhs_scale), but got "
                          << args.size();

  auto lhs_type = As<TileType>(args[0]->GetType());
  auto lhs_scale_type = As<TileType>(args[1]->GetType());
  auto rhs_type = As<TileType>(args[2]->GetType());
  auto rhs_scale_type = As<TileType>(args[3]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires lhs to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires rhs to be a TileType, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(lhs_type->shape_.size() == 2 && rhs_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D lhs/rhs tiles";
  CHECK(IsMxDataDtype(lhs_type->dtype_))
      << "The operator " << op_name << " requires lhs dtype in {FP8E4M3FN, FP8E5M2, FP4}, but got "
      << lhs_type->dtype_.ToString();
  CHECK(IsMxDataDtype(rhs_type->dtype_))
      << "The operator " << op_name << " requires rhs dtype in {FP8E4M3FN, FP8E5M2, FP4}, but got "
      << rhs_type->dtype_.ToString();

  ExprPtr m_dim = lhs_type->shape_[0];
  ExprPtr k_dim_lhs = lhs_type->shape_[1];
  ExprPtr k_dim_rhs = rhs_type->shape_[0];
  ExprPtr n_dim = rhs_type->shape_[1];

  auto k_lhs_const = As<ConstInt>(k_dim_lhs);
  auto k_rhs_const = As<ConstInt>(k_dim_rhs);
  if (k_lhs_const && k_rhs_const) {
    CHECK(k_lhs_const->value_ == k_rhs_const->value_)
        << "The operator " << op_name << " requires matching K, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  ExprPtr scale_k = MxScaleKFromK(k_dim_lhs, op_name);
  if (!scale_k) {
    // Fall back to lhs_scale cols when K is dynamic.
    CHECK(lhs_scale_type && lhs_scale_type->shape_.size() == 2);
    scale_k = lhs_scale_type->shape_[1];
  }
  CheckMxScaleTile(lhs_scale_type, m_dim, scale_k, op_name, "lhs_scale");
  CheckMxScaleTile(rhs_scale_type, scale_k, n_dim, op_name, "rhs_scale");

  std::vector<ExprPtr> output_shape = {m_dim, n_dim};
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = output_shape;
  return std::make_shared<TileType>(output_shape, DataType::FP32, std::nullopt, tile_view);
}

TypePtr DeduceTileMatMulMxAccType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires exactly 5 arguments (acc, lhs, lhs_scale, rhs, rhs_scale), but got "
                          << args.size();
  auto acc_type = As<TileType>(args[0]->GetType());
  CHECK(acc_type) << "The operator " << op_name << " requires acc to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(acc_type->shape_.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                                      << acc_type->shape_.size() << " dimensions";
  CHECK(acc_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires acc dtype FP32, but got " << acc_type->dtype_.ToString();

  std::vector<ExprPtr> mx_args = {args[1], args[2], args[3], args[4]};
  auto out_type = DeduceTileMatMulMxType(mx_args, kwargs, op_name);
  auto out_tile = As<TileType>(out_type);
  auto m_acc = As<ConstInt>(acc_type->shape_[0]);
  auto n_acc = As<ConstInt>(acc_type->shape_[1]);
  auto m_out = As<ConstInt>(out_tile->shape_[0]);
  auto n_out = As<ConstInt>(out_tile->shape_[1]);
  if (m_acc && m_out) {
    CHECK(m_acc->value_ == m_out->value_)
        << "The operator " << op_name << " requires acc rows to match output M";
  }
  if (n_acc && n_out) {
    CHECK(n_acc->value_ == n_out->value_)
        << "The operator " << op_name << " requires acc cols to match output N";
  }
  return out_type;
}

TypePtr DeduceTileMatMulMxBiasType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires exactly 5 arguments (lhs, lhs_scale, rhs, rhs_scale, bias), but got "
                          << args.size();
  auto bias_type = As<TileType>(args[4]->GetType());
  CHECK(bias_type) << "The operator " << op_name << " requires bias to be a TileType, but got "
                   << args[4]->GetType()->TypeName();
  CHECK(bias_type->shape_.size() == 2) << "The operator " << op_name << " requires bias to be 2D, but got "
                                       << bias_type->shape_.size() << " dimensions";
  CHECK(bias_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires bias dtype FP32, but got " << bias_type->dtype_.ToString();

  std::vector<ExprPtr> mx_args = {args[0], args[1], args[2], args[3]};
  auto out_type = DeduceTileMatMulMxType(mx_args, kwargs, op_name);
  auto out_tile = As<TileType>(out_type);
  auto n_out = As<ConstInt>(out_tile->shape_[1]);
  auto bias_rows = As<ConstInt>(bias_type->shape_[0]);
  auto bias_cols = As<ConstInt>(bias_type->shape_[1]);
  if (bias_rows) {
    CHECK(bias_rows->value_ == 1) << "The operator " << op_name << " requires bias shape [1, N]";
  }
  if (n_out && bias_cols) {
    CHECK(n_out->value_ == bias_cols->value_)
        << "The operator " << op_name << " requires bias cols to match N";
  }
  return out_type;
}

REGISTER_OP("tile.matmul_mx")
    .set_op_category("TileOp")
    .set_description("MX block-scale matrix multiplication: C = matmul_mx(A, A_scale, B, B_scale)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0, [M, K/32])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0, [K/32, N])")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxType(args, kwargs, "tile.matmul_mx");
    });

REGISTER_OP("tile.matmul_mx_acc")
    .set_op_category("TileOp")
    .set_description("MX block-scale matmul with accumulation: acc += matmul_mx(...)")
    .add_argument("acc", "Accumulator tile (TileType, 2D, FP32)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::LeftScale)
    .set_input_memory(3, MemorySpace::Right)
    .set_input_memory(4, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxAccType(args, kwargs, "tile.matmul_mx_acc");
    });

REGISTER_OP("tile.matmul_mx_bias")
    .set_op_category("TileOp")
    .set_description("MX block-scale matmul with bias: C = matmul_mx(...) + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, MXFP8/MXFP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("bias", "Bias tile (TileType, [1, N], FP32)")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_input_memory(4, MemorySpace::Bias)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxBiasType(args, kwargs, "tile.matmul_mx_bias");
    });

REGISTER_OP("tile.gemv")
    .set_op_category("TileOp")
    .set_description("General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulType(args, kwargs, "tile.gemv");
    });

REGISTER_OP("tile.gemv_acc")
    .set_op_category("TileOp")
    .set_description("GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N]")
    .add_argument("acc", "Accumulator tile (TileType, 2D [1, N])")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulAccType(args, kwargs, "tile.gemv_acc");
    });

REGISTER_OP("tile.gemv_bias")
    .set_op_category("TileOp")
    .set_description("GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .add_argument("bias", "Bias tile (TileType, [1, N])")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_input_memory(2, MemorySpace::Bias)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulBiasType(args, kwargs, "tile.gemv_bias");
    });

}  // namespace ir
}  // namespace pypto
