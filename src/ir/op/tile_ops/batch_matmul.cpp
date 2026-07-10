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
 * @brief Batch matrix multiplication operations for tile-level programming
 *
 * This file implements batch matrix multiplication operations for TileType,
 * supporting multi-dimensional tensors with batch dimensions.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Propagate batch-dim valid extents through a (possibly broadcasting) batch matmul.
 *
 * Mirrors ``BroadcastShapes`` on the *physical* batch dims but resolves the *valid*
 * extent per output batch dim: a size-1 (broadcast) operand dim contributes its
 * counterpart's valid extent; two full-extent dims must agree, and a provable
 * ``ConstInt`` disagreement is a user error (symbolic dims defer, matching the K
 * rule — never widen, but never falsely reject a dynamic extent either).
 *
 * Precondition: the physical batch shapes already broadcast (the caller verifies via
 * ``BroadcastShapes``); this only selects each output dim's valid extent. Right-aligned
 * like ``BroadcastShapes``, so ``lhs_batch_valid[i]`` pairs with ``lhs_batch_shape[i]``.
 */
std::vector<ExprPtr> BroadcastBatchValidShape(const std::vector<ExprPtr>& lhs_batch_shape,
                                              const std::vector<ExprPtr>& lhs_batch_valid,
                                              const std::vector<ExprPtr>& rhs_batch_shape,
                                              const std::vector<ExprPtr>& rhs_batch_valid, const Span& span,
                                              const std::string& op_name) {
  const size_t nl = lhs_batch_shape.size();
  const size_t nr = rhs_batch_shape.size();
  const size_t n = std::max(nl, nr);
  std::vector<ExprPtr> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const int64_t il = static_cast<int64_t>(nl) - 1 - static_cast<int64_t>(i);
    const int64_t ir = static_cast<int64_t>(nr) - 1 - static_cast<int64_t>(i);
    ExprPtr l_shape = (il >= 0) ? lhs_batch_shape[il] : nullptr;
    ExprPtr l_valid = (il >= 0) ? lhs_batch_valid[il] : nullptr;
    ExprPtr r_shape = (ir >= 0) ? rhs_batch_shape[ir] : nullptr;
    ExprPtr r_valid = (ir >= 0) ? rhs_batch_valid[ir] : nullptr;
    if (!l_shape) {  // dim present only on rhs
      out.push_back(r_valid);
      continue;
    }
    if (!r_shape) {  // dim present only on lhs
      out.push_back(l_valid);
      continue;
    }
    auto lc = As<ConstInt>(l_shape);
    auto rc = As<ConstInt>(r_shape);
    if (lc && lc->value_ == 1) {  // lhs broadcasts along this dim
      // The sole lhs batch matrix is replicated across the counterpart's valid extent.
      // But if that single matrix is itself padding (valid extent provably 0), every
      // replicated output batch is garbage — the output valid extent is 0, NOT the
      // counterpart's (which would widen padding into claimed-valid output batches).
      auto lv = As<ConstInt>(l_valid);
      out.push_back((lv && lv->value_ == 0) ? std::make_shared<ConstInt>(0, DataType::INDEX, span) : r_valid);
      continue;
    }
    if (rc && rc->value_ == 1) {  // rhs broadcasts along this dim
      auto rv = As<ConstInt>(r_valid);
      out.push_back((rv && rv->value_ == 0) ? std::make_shared<ConstInt>(0, DataType::INDEX, span) : l_valid);
      continue;
    }
    // Non-broadcast: both carry the full physical extent, so their valid extents must
    // agree. Only a static-vs-static disagreement is provable; symbolic dims defer.
    auto lv = As<ConstInt>(l_valid);
    auto rv = As<ConstInt>(r_valid);
    if (lv && rv) {
      CHECK_SPAN(lv->value_ == rv->value_, span)
          << op_name << ": lhs and rhs disagree on the valid extent of a non-broadcast batch "
          << "dimension (lhs valid=" << lv->value_ << ", rhs valid=" << rv->value_ << ")";
    }
    out.push_back(l_valid);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace

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
TypePtr DeduceTileBatchMatMulType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  (void)kwargs;
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

  // Extract matrix dimensions from the trailing matrix axes.
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

  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  // Hardware matmul accumulates to FP32 for float inputs, INT32 for integer inputs.
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  // Propagate the valid region: for C[..,M,N] = A[..,M,K] @ B[..,K,N] the valid output
  // is [broadcast(batch valid).., valid(A)[M-axis], valid(B)[N-axis]] — never the full
  // physical shape, which would claim padding carries real data. M is lhs axis -2, N is
  // rhs axis -1 (see the m_dim/n_dim assignment above).
  const std::vector<ExprPtr> lhs_valid = GetValidShape(lhs_type);
  const std::vector<ExprPtr> rhs_valid = GetValidShape(rhs_type);
  CheckMatMulValidKCompat(lhs_valid[lhs_ndim - 1], rhs_valid[rhs_ndim - 2], args[0]->span_, op_name);
  const ExprPtr& m_valid = lhs_valid[lhs_ndim - 2];
  const ExprPtr& n_valid = rhs_valid[rhs_ndim - 1];

  std::vector<ExprPtr> output_valid;
  if (lhs_ndim == 2 && rhs_ndim == 2) {
    output_valid = {m_valid, n_valid};
  } else {
    std::vector<ExprPtr> lhs_batch_shape(lhs_shape.begin(), lhs_shape.end() - 2);
    std::vector<ExprPtr> rhs_batch_shape(rhs_shape.begin(), rhs_shape.end() - 2);
    std::vector<ExprPtr> lhs_batch_valid(lhs_valid.begin(), lhs_valid.end() - 2);
    std::vector<ExprPtr> rhs_batch_valid(rhs_valid.begin(), rhs_valid.end() - 2);
    output_valid = BroadcastBatchValidShape(lhs_batch_shape, lhs_batch_valid, rhs_batch_shape,
                                            rhs_batch_valid, args[0]->span_, op_name);
    output_valid.push_back(m_valid);
    output_valid.push_back(n_valid);
  }

  // The matmul output tile uses the hardware's native accumulator layout:
  // - blayout=col_major, slayout=row_major: hardware's column-major block / row-major sub-block
  // - fractal=1024: 32x32 sub-tile fractal size (standard for this hardware's matrix unit)
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = std::move(output_valid);
  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view);
}

/**
 * @brief Deduce type for batch matrix multiplication with accumulation
 *
 * Computes acc[..batch, M, N] += lhs[..batch_lhs, M, K] @ rhs[..batch_rhs, K, N].
 * batch dims of lhs/rhs are broadcast against each other; the resulting batch shape
 * must match the acc batch dims exactly (acc is an in-place target and is not
 * broadcast).
 *
 * @param args Arguments: [acc_tile, lhs_tile, rhs_tile]
 * @param kwargs Keyword arguments (unused)
 * @param op_name Operator name for error messages
 * @return TileType with output shape (same as acc)
 */
TypePtr DeduceTileBatchMatMulAccType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  (void)kwargs;
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

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

  const auto& acc_shape = acc_type->shape_;
  const auto& lhs_shape = lhs_type->shape_;
  const auto& rhs_shape = rhs_type->shape_;

  CHECK(acc_shape.size() >= 2) << "The operator " << op_name
                               << " requires acc to have at least 2 dimensions, but got " << acc_shape.size()
                               << " dimensions";
  CHECK(lhs_shape.size() >= 2) << "The operator " << op_name
                               << " requires lhs to have at least 2 dimensions, but got " << lhs_shape.size()
                               << " dimensions";
  CHECK(rhs_shape.size() >= 2) << "The operator " << op_name
                               << " requires rhs to have at least 2 dimensions, but got " << rhs_shape.size()
                               << " dimensions";

  size_t acc_ndim = acc_shape.size();
  size_t lhs_ndim = lhs_shape.size();
  size_t rhs_ndim = rhs_shape.size();

  // Trailing matrix dims.
  ExprPtr m_dim_acc = acc_shape[acc_ndim - 2];
  ExprPtr n_dim_acc = acc_shape[acc_ndim - 1];
  ExprPtr m_dim_lhs = lhs_shape[lhs_ndim - 2];
  ExprPtr k_dim_lhs = lhs_shape[lhs_ndim - 1];
  ExprPtr k_dim_rhs = rhs_shape[rhs_ndim - 2];
  ExprPtr n_dim_rhs = rhs_shape[rhs_ndim - 1];

  // Verify M / N / K when statically known.
  auto m_acc_const = As<ConstInt>(m_dim_acc);
  auto m_lhs_const = As<ConstInt>(m_dim_lhs);
  auto n_acc_const = As<ConstInt>(n_dim_acc);
  auto n_rhs_const = As<ConstInt>(n_dim_rhs);
  auto k_lhs_const = As<ConstInt>(k_dim_lhs);
  auto k_rhs_const = As<ConstInt>(k_dim_rhs);

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
        << " requires matching inner dimensions, but got lhs K=" << k_lhs_const->value_
        << " and rhs K=" << k_rhs_const->value_;
  }

  // Broadcast batch dims of lhs and rhs; require result to equal acc's batch dims.
  std::vector<ExprPtr> acc_batch(acc_shape.begin(), acc_shape.end() - 2);
  std::vector<ExprPtr> lhs_batch(lhs_shape.begin(), lhs_shape.end() - 2);
  std::vector<ExprPtr> rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);

  auto broadcast_result = BroadcastShapes(lhs_batch, rhs_batch);
  CHECK(broadcast_result.success) << "Cannot broadcast batch dimensions for " << op_name;

  CHECK(broadcast_result.shape.size() == acc_batch.size())
      << "The operator " << op_name << " requires acc batch rank (" << acc_batch.size()
      << ") to equal broadcast(lhs, rhs) batch rank (" << broadcast_result.shape.size() << ")";

  // Acc is in-place: every batch dim must match exactly when statically known.
  for (size_t i = 0; i < acc_batch.size(); ++i) {
    auto acc_const = As<ConstInt>(acc_batch[i]);
    auto bcast_const = As<ConstInt>(broadcast_result.shape[i]);
    if (acc_const && bcast_const) {
      CHECK(acc_const->value_ == bcast_const->value_)
          << "The operator " << op_name << " requires acc batch dim " << i
          << " to equal broadcast(lhs, rhs) batch dim " << i << ", but got acc=" << acc_const->value_
          << " and broadcast=" << bcast_const->value_;
    }
  }

  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires identical lhs and rhs data types, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  // Hardware accumulates to FP32 for float inputs, INT32 for integer inputs.
  auto result_dtype =
      (lhs_type->dtype_.IsFloat() && rhs_type->dtype_.IsFloat()) ? DataType::FP32 : DataType::INT32;

  CHECK(acc_type->dtype_ == result_dtype)
      << "The operator " << op_name << " requires accumulator dtype " << result_dtype.ToString()
      << ", but got " << acc_type->dtype_.ToString();

  // Output shape = acc shape (in-place accumulation).
  std::vector<ExprPtr> output_shape = acc_shape;

  // acc[..,M,N] += lhs[..,M,K] @ rhs[..,K,N] writes the matmul result in place at the
  // origin. The written region is [broadcast(batch valid).., valid(lhs)[M], valid(rhs)[N]]
  // (M is lhs axis -2, N is rhs axis -1); the result's valid region is the bounding box
  // (union) of the accumulator's existing region and that written rectangle — never the
  // full physical shape. Reuse the assemble union rule at offset = origin, which clamps
  // to the physical shape and rejects a non-representable union. A fully-valid
  // accumulator unions back to the full shape (seed via create_tile(valid_shape=0) to
  // narrow).
  const std::vector<ExprPtr> acc_valid = GetValidShape(acc_type);
  const std::vector<ExprPtr> lhs_valid = GetValidShape(lhs_type);
  const std::vector<ExprPtr> rhs_valid = GetValidShape(rhs_type);
  CheckMatMulValidKCompat(lhs_valid[lhs_ndim - 1], rhs_valid[rhs_ndim - 2], args[0]->span_, op_name);

  std::vector<ExprPtr> lhs_batch_valid(lhs_valid.begin(), lhs_valid.end() - 2);
  std::vector<ExprPtr> rhs_batch_valid(rhs_valid.begin(), rhs_valid.end() - 2);
  std::vector<ExprPtr> written_valid = BroadcastBatchValidShape(lhs_batch, lhs_batch_valid, rhs_batch,
                                                                rhs_batch_valid, args[0]->span_, op_name);
  written_valid.push_back(lhs_valid[lhs_ndim - 2]);  // M valid
  written_valid.push_back(rhs_valid[rhs_ndim - 1]);  // N valid

  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, args[0]->span_);
  std::vector<ExprPtr> zero_offset(acc_ndim, zero);
  std::vector<ExprPtr> output_valid = ComputeAssembleUnionValidShape(acc_valid, written_valid, zero_offset,
                                                                     acc_shape, args[0]->span_, op_name);

  // Acc layout (Nz) — same as 2D matmul_acc.
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = std::move(output_valid);
  return std::make_shared<TileType>(output_shape, result_dtype, std::nullopt, tile_view);
}

// ============================================================================
// Registration Function for Block Batch Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tile.batch_matmul")
    .set_op_category("TileOp")
    .set_description("Batch matrix multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType, at least 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, at least 2D)")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileBatchMatMulType(args, kwargs, "tile.batch_matmul");
    });

REGISTER_OP("tile.batch_matmul_acc")
    .set_op_category("TileOp")
    .set_description(
        "Batch matrix multiplication with accumulation: acc = acc + lhs @ rhs (with batch broadcast)")
    .add_argument("acc", "Accumulator tile (TileType, at least 2D)")
    .add_argument("lhs", "Left-hand side tile (TileType, at least 2D)")
    .add_argument("rhs", "Right-hand side tile (TileType, at least 2D)")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::Right)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileBatchMatMulAccType(args, kwargs, "tile.batch_matmul_acc");
    });

}  // namespace ir
}  // namespace pypto
