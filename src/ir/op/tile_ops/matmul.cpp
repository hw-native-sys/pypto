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
#include <cstddef>
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
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/printer.h"
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

  // Matrix multiplication: [M, K] @ [K, N] -> [M, N]. Physical boxed
  // storage may be wider than the valid computation window, so dimensional
  // compatibility follows valid_shape while the result allocation follows
  // the operands' physical M/N extents.
  const auto lhs_valid = GetValidShape(lhs_type);
  const auto rhs_valid = GetValidShape(rhs_type);
  const ExprPtr& k_dim_lhs = lhs_valid[1];
  const ExprPtr& k_dim_rhs = rhs_valid[0];

  // Physical boxes must remain compatible even when their logical valid
  // windows are narrower. Downstream extraction and L0 tiling index the
  // physical K extent directly.
  auto physical_k_lhs_const = As<ConstInt>(lhs_shape[1]);
  auto physical_k_rhs_const = As<ConstInt>(rhs_shape[0]);
  if (physical_k_lhs_const && physical_k_rhs_const) {
    CHECK(physical_k_lhs_const->value_ == physical_k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching physical inner dimensions, but got lhs K=" << physical_k_lhs_const->value_
        << " and rhs K=" << physical_k_rhs_const->value_;
  }

  // PTO takes M/K from lhs and N from rhs. The rhs may expose a wider valid K
  // window than lhs (for example a physically padded GM-to-Mat load), but it
  // must contain every K element PTO will read.
  CHECK(ProveValidExtentLessEqual(k_dim_lhs, k_dim_rhs) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires rhs valid K to cover lhs valid K, but got lhs K=" << PythonPrint(k_dim_lhs)
      << " and rhs K=" << PythonPrint(k_dim_rhs);

  // A2A3 only support float or int32_t output, and input type must be same
  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  // Physical output shape follows the boxed operands; only their valid M/N
  // rectangle contains computed values.
  std::vector<ExprPtr> output_shape = {lhs_shape[0], rhs_shape[1]};
  std::vector<ExprPtr> output_valid_shape = {lhs_valid[0], rhs_valid[1]};

  // Acc layout (Nz), taken from the destination space's implicit layout rather
  // than a hand-written triple. fractal is the inner box size in *bytes* — 16
  // rows x (1024 / dtype_bytes / 16) cols, i.e. a 16x16 box for the 4-byte
  // (FP32/INT32) accumulator.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(output_shape, MemorySpace::Acc));
  tile_view.valid_shape = std::move(output_valid_shape);

  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view, MemorySpace::Acc);
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

  // Matrix multiplication with accumulation: acc[M, N] += lhs[M, K] @ rhs[K, N].
  // Match the logical valid rectangle, not padding in the boxed allocations.
  const auto acc_valid = GetValidShape(acc_type);
  const auto lhs_valid = GetValidShape(lhs_type);
  const auto rhs_valid = GetValidShape(rhs_type);
  const ExprPtr& m_dim_acc = acc_valid[0];
  const ExprPtr& n_dim_acc = acc_valid[1];

  // The aliased Acc result, lhs, and rhs must agree in physical M/N/K. Valid
  // windows are checked separately below and may be narrower than these boxes.
  auto physical_m_acc_const = As<ConstInt>(acc_shape[0]);
  auto physical_m_lhs_const = As<ConstInt>(lhs_shape[0]);
  auto physical_n_acc_const = As<ConstInt>(acc_shape[1]);
  auto physical_n_rhs_const = As<ConstInt>(rhs_shape[1]);
  auto physical_k_lhs_const = As<ConstInt>(lhs_shape[1]);
  auto physical_k_rhs_const = As<ConstInt>(rhs_shape[0]);

  if (physical_m_acc_const && physical_m_lhs_const) {
    CHECK(physical_m_acc_const->value_ == physical_m_lhs_const->value_)
        << "The operator " << op_name
        << " requires matching physical M dimensions, but got acc M=" << physical_m_acc_const->value_
        << " and lhs M=" << physical_m_lhs_const->value_;
  }
  if (physical_n_acc_const && physical_n_rhs_const) {
    CHECK(physical_n_acc_const->value_ == physical_n_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching physical N dimensions, but got acc N=" << physical_n_acc_const->value_
        << " and rhs N=" << physical_n_rhs_const->value_;
  }
  if (physical_k_lhs_const && physical_k_rhs_const) {
    CHECK(physical_k_lhs_const->value_ == physical_k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching physical K dimensions, but got lhs K=" << physical_k_lhs_const->value_
        << " and rhs K=" << physical_k_rhs_const->value_;
  }

  // PTO derives the computed M/K/N rectangle from lhs/lhs/rhs. The in-place
  // accumulator and rhs K window may be larger, but must contain that complete
  // rectangle. Unknown symbolic relations remain legal, matching the previous
  // static-only validation while rejecting every provably out-of-bounds case.
  CHECK(ProveValidExtentLessEqual(lhs_valid[0], m_dim_acc) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires acc valid M to cover lhs valid M, but got acc M=" << PythonPrint(m_dim_acc)
      << " and lhs M=" << PythonPrint(lhs_valid[0]);
  CHECK(ProveValidExtentLessEqual(rhs_valid[1], n_dim_acc) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires acc valid N to cover rhs valid N, but got acc N=" << PythonPrint(n_dim_acc)
      << " and rhs N=" << PythonPrint(rhs_valid[1]);
  CHECK(ProveValidExtentLessEqual(lhs_valid[1], rhs_valid[0]) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires rhs valid K to cover lhs valid K, but got lhs K=" << PythonPrint(lhs_valid[1])
      << " and rhs K=" << PythonPrint(rhs_valid[0]);

  // A2A3 only support float or int32_t output, and input type must be same
  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  CHECK(acc_type->dtype_ == result_dtype)
      << "The operator " << op_name << " requires accumulator dtype " << result_dtype.ToString()
      << ", but got " << acc_type->dtype_.ToString();

  // The output aliases the accumulator's physical storage and valid region.
  std::vector<ExprPtr> output_shape = acc_shape;

  // Acc layout (Nz) — as in tile.matmul, from the destination's implicit layout.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(output_shape, MemorySpace::Acc));
  tile_view.valid_shape = acc_valid;

  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view, MemorySpace::Acc);
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

  // Acc layout (Nz) — as in tile.matmul. This deducer previously left the view at
  // the struct default (row_major/none_box) and reached the Acc layout only
  // because a fully-valid view collapses to nullopt and the registry's
  // memory-space stamp re-canonicalized it against Acc's implicit view.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(output_shape, MemorySpace::Acc));
  tile_view.valid_shape = output_shape;
  return std::make_shared<TileType>(output_shape, *result_dtype, std::nullopt, tile_view, MemorySpace::Acc);
}

namespace {

TileTypePtr BuildGemvResultType(const TypePtr& inferred, const TileTypePtr& lhs_type,
                                const TileTypePtr& rhs_type) {
  auto inferred_type = As<TileType>(inferred);
  INTERNAL_CHECK(inferred_type) << "Internal error: GEMV type inference must produce TileType";

  auto physical_rows = std::make_shared<ConstInt>(16, DataType::INDEX, lhs_type->shape_[0]->span_);
  std::vector<ExprPtr> output_shape = {physical_rows, rhs_type->shape_[1]};
  const auto lhs_valid = GetValidShape(lhs_type);
  const auto rhs_valid = GetValidShape(rhs_type);

  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = {lhs_valid[0], rhs_valid[1]};
  return std::make_shared<TileType>(output_shape, inferred_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileGemvType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  TypePtr inferred = DeduceTileMatMulType(args, kwargs, op_name);
  return BuildGemvResultType(inferred, As<TileType>(args[0]->GetType()), As<TileType>(args[1]->GetType()));
}

TypePtr DeduceTileGemvAccType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();
  auto acc_type = As<TileType>(args[0]->GetType());
  CHECK(acc_type) << "The operator " << op_name << " requires first argument (acc) to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(acc_type->shape_.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                                      << acc_type->shape_.size() << " dimensions";

  std::vector<ExprPtr> product_args = {args[1], args[2]};
  TypePtr product = DeduceTileMatMulType(product_args, kwargs, op_name);
  auto expected_type =
      BuildGemvResultType(product, As<TileType>(args[1]->GetType()), As<TileType>(args[2]->GetType()));

  for (std::size_t axis = 0; axis < 2; ++axis) {
    auto acc_extent = As<ConstInt>(acc_type->shape_[axis]);
    auto expected_extent = As<ConstInt>(expected_type->shape_[axis]);
    if (acc_extent && expected_extent) {
      CHECK(acc_extent->value_ == expected_extent->value_)
          << "The operator " << op_name << " requires accumulator physical shape "
          << FormatShape(expected_type->shape_) << ", but got " << FormatShape(acc_type->shape_);
    }
  }
  CHECK(acc_type->dtype_ == expected_type->dtype_)
      << "The operator " << op_name << " requires accumulator dtype " << expected_type->dtype_.ToString()
      << ", but got " << acc_type->dtype_.ToString();
  return expected_type;
}

TypePtr DeduceTileGemvBiasType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  TypePtr inferred = DeduceTileMatMulBiasType(args, kwargs, op_name);
  return BuildGemvResultType(inferred, As<TileType>(args[0]->GetType()), As<TileType>(args[1]->GetType()));
}

}  // namespace

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
      return DeduceTileGemvType(args, kwargs, "tile.gemv");
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
      return DeduceTileGemvAccType(args, kwargs, "tile.gemv_acc");
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
      return DeduceTileGemvBiasType(args, kwargs, "tile.gemv_bias");
    });

}  // namespace ir
}  // namespace pypto
