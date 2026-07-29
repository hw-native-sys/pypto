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
#include <cstdint>
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

static void CheckDimMatch(const ExprPtr& expected, const ExprPtr& actual, const std::string& op_name,
                          const char* scale_name, const char* which, const char* axis) {
  auto e = As<ConstInt>(expected);
  auto a = As<ConstInt>(actual);
  if (e && a) {
    CHECK(e->value_ == a->value_) << "The operator " << op_name << " requires " << scale_name << " " << which
                                  << " " << axis << "=" << e->value_ << ", but got " << a->value_;
  }
}

/// Validate MX scale tile physical shape and valid_shape separately (PTOAS contract).
static void CheckMxScaleTile(const TileTypePtr& scale_type, const ExprPtr& phys_rows,
                             const ExprPtr& phys_cols, const ExprPtr& valid_rows, const ExprPtr& valid_cols,
                             const std::string& op_name, const char* scale_name) {
  CHECK(scale_type) << "The operator " << op_name << " requires " << scale_name << " to be a TileType";
  CHECK(scale_type->dtype_ == DataType::FP8E8M0)
      << "The operator " << op_name << " requires " << scale_name << " dtype FP8E8M0, but got "
      << scale_type->dtype_.ToString();
  CHECK(scale_type->shape_.size() == 2)
      << "The operator " << op_name << " requires " << scale_name << " to be 2D, but got "
      << scale_type->shape_.size() << " dimensions";

  CheckDimMatch(phys_rows, scale_type->shape_[0], op_name, scale_name, "physical", "rows");
  CheckDimMatch(phys_cols, scale_type->shape_[1], op_name, scale_name, "physical", "cols");

  ExprPtr scale_valid_rows = scale_type->shape_[0];
  ExprPtr scale_valid_cols = scale_type->shape_[1];
  if (scale_type->tile_view_.has_value() && scale_type->tile_view_->valid_shape.size() >= 2) {
    if (scale_type->tile_view_->valid_shape[0]) scale_valid_rows = scale_type->tile_view_->valid_shape[0];
    if (scale_type->tile_view_->valid_shape[1]) scale_valid_cols = scale_type->tile_view_->valid_shape[1];
  }
  CheckDimMatch(valid_rows, scale_valid_rows, op_name, scale_name, "valid", "rows");
  CheckDimMatch(valid_cols, scale_valid_cols, op_name, scale_name, "valid", "cols");
}

/// MX scale groups along K: ceil(K / 32). Valid K need not be a multiple of 32.
static ExprPtr MxScaleKCeil(const ExprPtr& k_dim) {
  auto k_const = As<ConstInt>(k_dim);
  if (!k_const) return nullptr;
  CHECK(k_const->value_ > 0) << "MX scale K must be positive, got K=" << k_const->value_;
  const int64_t groups = (k_const->value_ + 31) / 32;
  return std::make_shared<ConstInt>(groups, DataType::INDEX, Span::unknown());
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
  CHECK(lhs_type->dtype_ == DataType::FP8E4M3FN)
      << "The operator " << op_name << " requires lhs dtype FP8E4M3FN, but got "
      << lhs_type->dtype_.ToString();
  CHECK(rhs_type->dtype_ == DataType::FP8E4M3FN)
      << "The operator " << op_name << " requires rhs dtype FP8E4M3FN, but got "
      << rhs_type->dtype_.ToString();

  ExprPtr m_phys = lhs_type->shape_[0];
  ExprPtr k_phys_lhs = lhs_type->shape_[1];
  ExprPtr k_phys_rhs = rhs_type->shape_[0];
  ExprPtr n_phys = rhs_type->shape_[1];

  ExprPtr m_valid = m_phys;
  ExprPtr k_valid_lhs = k_phys_lhs;
  ExprPtr k_valid_rhs = k_phys_rhs;
  ExprPtr n_valid = n_phys;
  if (lhs_type->tile_view_.has_value()) {
    const auto& vs = lhs_type->tile_view_->valid_shape;
    if (vs.size() >= 1 && vs[0]) m_valid = vs[0];
    if (vs.size() >= 2 && vs[1]) k_valid_lhs = vs[1];
  }
  if (rhs_type->tile_view_.has_value()) {
    const auto& vs = rhs_type->tile_view_->valid_shape;
    if (vs.size() >= 1 && vs[0]) k_valid_rhs = vs[0];
    if (vs.size() >= 2 && vs[1]) n_valid = vs[1];
  }

  auto m_phys_c = As<ConstInt>(m_phys);
  auto n_phys_c = As<ConstInt>(n_phys);
  auto k_phys_lhs_c = As<ConstInt>(k_phys_lhs);
  auto k_phys_rhs_c = As<ConstInt>(k_phys_rhs);
  if (m_phys_c) {
    CHECK(m_phys_c->value_ > 0 && m_phys_c->value_ % 16 == 0)
        << "The operator " << op_name
        << " requires physical M divisible by 16 (ISA/PTOAS tmatmul_mx), but got M=" << m_phys_c->value_;
  }
  if (n_phys_c) {
    CHECK(n_phys_c->value_ > 0 && n_phys_c->value_ % 32 == 0)
        << "The operator " << op_name
        << " requires physical N divisible by 32 (ISA/PTOAS tmatmul_mx fp8), but got N=" << n_phys_c->value_;
  }
  if (k_phys_lhs_c && k_phys_rhs_c) {
    CHECK(k_phys_lhs_c->value_ == k_phys_rhs_c->value_)
        << "The operator " << op_name
        << " requires matching physical K, but got lhs K=" << k_phys_lhs_c->value_
        << " and rhs K=" << k_phys_rhs_c->value_;
  }
  if (k_phys_lhs_c) {
    CHECK(k_phys_lhs_c->value_ > 0 && k_phys_lhs_c->value_ % 64 == 0)
        << "The operator " << op_name
        << " requires physical K divisible by 64 (ISA/PTOAS tmatmul_mx), but got K=" << k_phys_lhs_c->value_;
  }

  auto k_valid_lhs_c = As<ConstInt>(k_valid_lhs);
  auto k_valid_rhs_c = As<ConstInt>(k_valid_rhs);
  if (k_valid_lhs_c && k_valid_rhs_c) {
    CHECK(k_valid_lhs_c->value_ == k_valid_rhs_c->value_)
        << "The operator " << op_name
        << " requires matching valid K, but got lhs valid K=" << k_valid_lhs_c->value_
        << " and rhs valid K=" << k_valid_rhs_c->value_;
  }
  if (k_valid_lhs_c) {
    CHECK(k_valid_lhs_c->value_ > 0) << "The operator " << op_name << " requires positive valid K, but got "
                                     << k_valid_lhs_c->value_;
  }
  // PTOAS v0.48: matmul_mx derives the scale physical group count from physical K
  // (ceil(K/32)), while tget_scale_addr derives it from valid K (ceil(validK/32)).
  // They must agree, otherwise no scale tile satisfies both verifiers — valid K
  // must round up to the same scale-group count as physical K.
  if (k_phys_lhs_c && k_valid_lhs_c) {
    const int64_t phys_groups = (k_phys_lhs_c->value_ + 31) / 32;
    const int64_t valid_groups = (k_valid_lhs_c->value_ + 31) / 32;
    CHECK(valid_groups == phys_groups)
        << "The operator " << op_name
        << " requires valid K to round up to the same scale-group count as physical K "
           "(ceil(validK/32) == ceil(physicalK/32)); otherwise the PTOAS matmul_mx and tget_scale_addr "
           "verifiers conflict. Got physical K="
        << k_phys_lhs_c->value_ << ", valid K=" << k_valid_lhs_c->value_;
  }

  ExprPtr scale_k_phys = MxScaleKCeil(k_phys_lhs);
  ExprPtr scale_k_valid = MxScaleKCeil(k_valid_lhs);
  if (!scale_k_phys) {
    CHECK(lhs_scale_type && lhs_scale_type->shape_.size() == 2);
    scale_k_phys = lhs_scale_type->shape_[1];
  }
  if (!scale_k_valid) {
    scale_k_valid = scale_k_phys;
  }
  CheckMxScaleTile(lhs_scale_type, m_phys, scale_k_phys, m_valid, scale_k_valid, op_name, "lhs_scale");
  CheckMxScaleTile(rhs_scale_type, scale_k_phys, n_phys, scale_k_valid, n_valid, op_name, "rhs_scale");

  std::vector<ExprPtr> output_shape = {m_phys, n_phys};
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = {m_valid, n_valid};
  return std::make_shared<TileType>(output_shape, DataType::FP32, std::nullopt, tile_view);
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

REGISTER_OP("tile.matmul_mx")
    .set_op_category("TileOp")
    .set_description("MX block-scale matrix multiplication: C = matmul_mx(A, A_scale, B, B_scale)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, MXFP8 E4M3); physical M % 16 == 0, K % 64 == 0")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0); physical/valid [M, ceil(K/32)]")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, MXFP8 E4M3); physical K % 64 == 0, N % 32 == 0")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0); physical/valid [ceil(K/32), N]")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxType(args, kwargs, "tile.matmul_mx");
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
