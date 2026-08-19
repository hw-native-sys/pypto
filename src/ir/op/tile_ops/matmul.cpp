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

#include "pypto/core/dtype.h"
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

namespace {

struct MatmulProductInfo {
  std::vector<ExprPtr> physical_shape;
  std::vector<ExprPtr> valid_shape;
  DataType accumulator_dtype;
};

/// Validate the matrix-product contract shared by matmul-family ops.
/// Physical boxes must agree in K because downstream L0 extraction indexes
/// them directly. The rhs valid K may be wider than lhs, but must contain every
/// K element the cube reads. Output storage follows physical M/N while its
/// logical rectangle follows lhs/rhs valid M/N.
MatmulProductInfo DeduceMatmulProductInfo(const TileTypePtr& lhs_type, const TileTypePtr& rhs_type,
                                          const std::string& op_name) {
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(lhs_shape.size() == 2) << "The operator " << op_name << " requires lhs to be 2D, but got "
                               << lhs_shape.size() << " dimensions";
  CHECK(rhs_shape.size() == 2) << "The operator " << op_name << " requires rhs to be 2D, but got "
                               << rhs_shape.size() << " dimensions";

  auto physical_k_lhs_const = As<ConstInt>(lhs_shape[1]);
  auto physical_k_rhs_const = As<ConstInt>(rhs_shape[0]);
  if (physical_k_lhs_const && physical_k_rhs_const) {
    CHECK(physical_k_lhs_const->value_ == physical_k_rhs_const->value_)
        << "The operator " << op_name
        << " requires matching physical inner dimensions (physical K), but got lhs K="
        << physical_k_lhs_const->value_ << " and rhs K=" << physical_k_rhs_const->value_;
  }

  const auto lhs_valid = GetValidShape(lhs_type);
  const auto rhs_valid = GetValidShape(rhs_type);
  CHECK(ProveValidExtentLessEqual(lhs_valid[1], rhs_valid[0]) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires rhs valid K to cover lhs valid K, but got lhs K=" << PythonPrint(lhs_valid[1])
      << " and rhs K=" << PythonPrint(rhs_valid[0]);

  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  const auto accumulator_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  return MatmulProductInfo{{lhs_shape[0], rhs_shape[1]}, {lhs_valid[0], rhs_valid[1]}, accumulator_dtype};
}

}  // namespace

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

  auto geometry = DeduceMatmulProductInfo(lhs_type, rhs_type, op_name);

  // Acc layout (Nz), taken from the destination space's implicit layout rather
  // than a hand-written triple. fractal is the inner box size in *bytes* — 16
  // rows x (1024 / dtype_bytes / 16) cols, i.e. a 16x16 box for the 4-byte
  // (FP32/INT32) accumulator.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(geometry.physical_shape, MemorySpace::Acc));
  tile_view.valid_shape = geometry.valid_shape;

  return std::make_shared<TileType>(std::move(geometry.physical_shape), geometry.accumulator_dtype,
                                    std::nullopt, tile_view, MemorySpace::Acc);
}

TypePtr DeduceTileMatMulAccType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 3 || args.size() == 4)
      << "The operator " << op_name << " requires 3 arguments (acc, lhs, rhs) or 4 with the optional "
      << "init_cond predicate, but got " << args.size();
  CheckMatmulInitCond(args, 3, op_name);

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

  auto geometry = DeduceMatmulProductInfo(lhs_type, rhs_type, op_name);
  const auto& acc_shape = acc_type->shape_;

  CHECK(acc_shape.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                               << acc_shape.size() << " dimensions";

  // Matrix multiplication with accumulation: acc[M, N] += lhs[M, K] @ rhs[K, N].
  // Match the logical valid rectangle, not padding in the boxed allocations.
  const auto acc_valid = GetValidShape(acc_type);
  const ExprPtr& m_dim_acc = acc_valid[0];
  const ExprPtr& n_dim_acc = acc_valid[1];

  // The aliased Acc result must agree with the matrix product's physical M/N.
  // Valid windows are checked separately below and may be narrower.
  auto physical_m_acc_const = As<ConstInt>(acc_shape[0]);
  auto physical_m_product_const = As<ConstInt>(geometry.physical_shape[0]);
  auto physical_n_acc_const = As<ConstInt>(acc_shape[1]);
  auto physical_n_product_const = As<ConstInt>(geometry.physical_shape[1]);

  if (physical_m_acc_const && physical_m_product_const) {
    CHECK(physical_m_acc_const->value_ == physical_m_product_const->value_)
        << "The operator " << op_name
        << " requires matching physical M dimensions, but got acc M=" << physical_m_acc_const->value_
        << " and product M=" << physical_m_product_const->value_;
  }
  if (physical_n_acc_const && physical_n_product_const) {
    CHECK(physical_n_acc_const->value_ == physical_n_product_const->value_)
        << "The operator " << op_name
        << " requires matching physical N dimensions, but got acc N=" << physical_n_acc_const->value_
        << " and product N=" << physical_n_product_const->value_;
  }

  // PTO derives the computed M/K/N rectangle from lhs/lhs/rhs. The in-place
  // accumulator and rhs K window may be larger, but must contain that complete
  // rectangle. Unknown symbolic relations remain legal, matching the previous
  // static-only validation while rejecting every provably out-of-bounds case.
  CHECK(ProveValidExtentLessEqual(geometry.valid_shape[0], m_dim_acc) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires acc valid M to cover lhs valid M, but got acc M=" << PythonPrint(m_dim_acc)
      << " and lhs M=" << PythonPrint(geometry.valid_shape[0]);
  CHECK(ProveValidExtentLessEqual(geometry.valid_shape[1], n_dim_acc) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires acc valid N to cover rhs valid N, but got acc N=" << PythonPrint(n_dim_acc)
      << " and rhs N=" << PythonPrint(geometry.valid_shape[1]);

  CHECK(acc_type->dtype_ == geometry.accumulator_dtype)
      << "The operator " << op_name << " requires accumulator dtype " << geometry.accumulator_dtype.ToString()
      << ", but got " << acc_type->dtype_.ToString();

  // The output aliases the accumulator's physical storage and valid region.
  std::vector<ExprPtr> output_shape = acc_shape;

  // Acc layout (Nz) — as in tile.matmul, from the destination's implicit layout.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(output_shape, MemorySpace::Acc));
  tile_view.valid_shape = acc_valid;

  return std::make_shared<TileType>(output_shape, geometry.accumulator_dtype, std::nullopt, tile_view,
                                    MemorySpace::Acc);
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

  auto geometry = DeduceMatmulProductInfo(lhs_type, rhs_type, op_name);
  const auto& rhs_shape = rhs_type->shape_;
  const auto& bias_shape = bias_type->shape_;

  CHECK(bias_shape.size() == 2) << "The operator " << op_name << " requires bias to be 2D, but got "
                                << bias_shape.size() << " dimensions";

  const auto rhs_valid = GetValidShape(rhs_type);
  const auto bias_valid = GetValidShape(bias_type);

  // Hardware requires bias to be [1, N]
  auto bias_row_const = As<ConstInt>(bias_shape[0]);
  CHECK(bias_row_const && bias_row_const->value_ == 1)
      << "The operator " << op_name << " requires bias to have shape [1, N], but got "
      << FormatShape(bias_shape);
  const auto one = std::make_shared<ConstInt>(1, DataType::INDEX, bias_valid[0]->span_);
  CHECK(ProveValidExtentLessEqual(one, bias_valid[0]) != ProofResult::kFalse)
      << "The operator " << op_name << " requires bias valid rows to cover one broadcast row, but got "
      << PythonPrint(bias_valid[0]);
  auto bias_n_const = As<ConstInt>(bias_shape[1]);
  auto rhs_n_const = As<ConstInt>(rhs_shape[1]);
  if (bias_n_const && rhs_n_const) {
    CHECK(bias_n_const->value_ == rhs_n_const->value_)
        << "The operator " << op_name
        << " requires bias N dimension to match output N=" << rhs_n_const->value_
        << ", but got bias N=" << bias_n_const->value_;
  }
  CHECK(ProveValidExtentLessEqual(rhs_valid[1], bias_valid[1]) != ProofResult::kFalse)
      << "The operator " << op_name
      << " requires bias valid N to cover output valid N=" << PythonPrint(rhs_valid[1])
      << ", but got bias valid N=" << PythonPrint(bias_valid[1]);

  CHECK(bias_type->dtype_ == geometry.accumulator_dtype)
      << "The operator " << op_name << " requires bias dtype " << geometry.accumulator_dtype.ToString()
      << " to match the accumulator, but got " << bias_type->dtype_.ToString();

  // Acc layout (Nz) — as in tile.matmul. This deducer previously left the view at
  // the struct default (row_major/none_box) and reached the Acc layout only
  // because a fully-valid view collapses to nullopt and the registry's
  // memory-space stamp re-canonicalized it against Acc's implicit view.
  TileView tile_view;
  tile_view_semantics::SetTileLayout(
      tile_view, tile_view_semantics::GetImplicitTileLayout(geometry.physical_shape, MemorySpace::Acc));
  tile_view.valid_shape = geometry.valid_shape;
  return std::make_shared<TileType>(std::move(geometry.physical_shape), geometry.accumulator_dtype,
                                    std::nullopt, tile_view, MemorySpace::Acc);
}

namespace {

void ValidateGemvAccPhase(const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  const auto acc_phase = GetKwargOr<std::string>(kwargs, "acc_phase", "unspecified");
  CHECK(acc_phase == "unspecified" || acc_phase == "partial" || acc_phase == "final")
      << "The operator " << op_name
      << " requires acc_phase to be one of {unspecified, partial, final}, but got " << acc_phase;
}

DataType ValidateGemvInputDtypes(const TileTypePtr& lhs_type, const TileTypePtr& rhs_type,
                                 const std::string& op_name) {
  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  const auto input_dtype = lhs_type->dtype_;
  CHECK(input_dtype == DataType::INT8 || input_dtype == DataType::FP16 || input_dtype == DataType::BF16 ||
        input_dtype == DataType::FP32)
      << "The operator " << op_name
      << " supports only INT8 x INT8 -> INT32 and same-type FP16/BF16/FP32 inputs -> FP32, but got "
      << input_dtype.ToString();
  return input_dtype == DataType::INT8 ? DataType::INT32 : DataType::FP32;
}

TileTypePtr BuildGemvResultType(const TypePtr& inferred, const TileTypePtr& lhs_type,
                                const TileTypePtr& rhs_type) {
  auto inferred_type = As<TileType>(inferred);
  INTERNAL_CHECK(inferred_type) << "Internal error: GEMV type inference must produce TileType";

  auto lhs_physical_rows = As<ConstInt>(lhs_type->shape_[0]);
  CHECK(lhs_physical_rows && lhs_physical_rows->value_ == 1)
      << "GEMV requires lhs physical row extent to be exactly 1, but got shape "
      << FormatShape(lhs_type->shape_);
  auto output_physical_rows = std::make_shared<ConstInt>(16, DataType::INDEX, lhs_type->shape_[0]->span_);
  std::vector<ExprPtr> output_shape = {output_physical_rows, rhs_type->shape_[1]};
  const auto lhs_valid = GetValidShape(lhs_type);
  const auto rhs_valid = GetValidShape(rhs_type);
  auto logical_row = std::make_shared<ConstInt>(1, DataType::INDEX, lhs_valid[0]->span_);
  CHECK(ProveValidExtentEqual(lhs_valid[0], logical_row) == ProofResult::kTrue)
      << "GEMV requires lhs logical row extent to be exactly 1, but got valid_shape "
      << FormatShape(lhs_valid);
  CHECK(ProveValidExtentLessEqual(lhs_valid[1], rhs_valid[0]) != ProofResult::kFalse)
      << "GEMV requires rhs logical K to cover lhs logical K, but got lhs K=" << PythonPrint(lhs_valid[1])
      << " and rhs K=" << PythonPrint(rhs_valid[0]);

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
  ValidateGemvAccPhase(kwargs, op_name);
  TypePtr inferred = DeduceTileMatMulType(args, kwargs, op_name);
  auto lhs_type = As<TileType>(args[0]->GetType());
  auto rhs_type = As<TileType>(args[1]->GetType());
  ValidateGemvInputDtypes(lhs_type, rhs_type, op_name);
  return BuildGemvResultType(inferred, lhs_type, rhs_type);
}

TypePtr DeduceTileGemvAccType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  ValidateGemvAccPhase(kwargs, op_name);
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();
  auto acc_type = As<TileType>(args[0]->GetType());
  CHECK(acc_type) << "The operator " << op_name << " requires first argument (acc) to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(acc_type->shape_.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                                      << acc_type->shape_.size() << " dimensions";

  std::vector<ExprPtr> product_args = {args[1], args[2]};
  TypePtr product = DeduceTileMatMulType(product_args, kwargs, op_name);
  ValidateGemvInputDtypes(As<TileType>(args[1]->GetType()), As<TileType>(args[2]->GetType()), op_name);
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
  const auto acc_valid = GetValidShape(acc_type);
  const auto expected_valid = GetValidShape(expected_type);
  for (std::size_t axis = 0; axis < 2; ++axis) {
    CHECK(ProveValidExtentEqual(acc_valid[axis], expected_valid[axis]) == ProofResult::kTrue)
        << "The operator " << op_name << " requires accumulator valid_shape " << FormatShape(expected_valid)
        << ", but got " << FormatShape(acc_valid);
  }
  return expected_type;
}

TypePtr DeduceTileGemvBiasType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  ValidateGemvAccPhase(kwargs, op_name);
  TypePtr inferred = DeduceTileMatMulBiasType(args, kwargs, op_name);
  auto lhs_type = As<TileType>(args[0]->GetType());
  auto rhs_type = As<TileType>(args[1]->GetType());
  auto bias_type = As<TileType>(args[2]->GetType());
  const auto result_dtype = ValidateGemvInputDtypes(lhs_type, rhs_type, op_name);
  CHECK(bias_type->dtype_ == result_dtype)
      << "The operator " << op_name << " requires bias dtype " << result_dtype.ToString()
      << " to match the accumulator output, but got " << bias_type->dtype_.ToString();
  const auto bias_valid = GetValidShape(bias_type);
  auto logical_row = std::make_shared<ConstInt>(1, DataType::INDEX, bias_valid[0]->span_);
  CHECK(ProveValidExtentEqual(bias_valid[0], logical_row) == ProofResult::kTrue)
      << "The operator " << op_name << " requires bias to have exactly one valid row, but got "
      << PythonPrint(bias_valid[0]);
  return BuildGemvResultType(inferred, lhs_type, rhs_type);
}

}  // namespace

// ============================================================================
// Registration Function for Block Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tile.matmul")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
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
    .functional_execution_memory_access()
    .set_description("Matrix multiplication with accumulation: acc = acc + lhs @ rhs")
    .add_argument("acc", "Accumulator tile (TileType, 2D)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .add_argument("init_cond",
                  "Optional BOOL scalar; where it holds the accumulator is overwritten with "
                  "lhs @ rhs instead of accumulated into (the split-K `k == 0` step)")
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
    .functional_execution_memory_access()
    .set_description("Matrix multiplication with bias add: C = lhs @ rhs + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D)")
    .add_argument("bias", "Accumulator-typed bias tile (TileType, [1, N])")
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
    .functional_execution_memory_access()
    .set_description("General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .set_attr<std::string>("acc_phase")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGemvType(args, kwargs, "tile.gemv");
    });

REGISTER_OP("tile.gemv_acc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N]")
    .add_argument("acc", "Accumulator tile (TileType, 2D [1, N])")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .set_attr<std::string>("acc_phase")
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
    .functional_execution_memory_access()
    .set_description("GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N]")
    .add_argument("lhs", "Row vector tile (TileType, 2D [1, K])")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D [K, N])")
    .add_argument("bias", "Accumulator-typed bias tile (TileType, [1, N])")
    .set_attr<std::string>("acc_phase")
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
