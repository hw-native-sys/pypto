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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/rewrite_internal.h"

namespace pypto {
namespace ir {

using transform_utils::Substitute;

namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

// ============================================================================
// Standalone N-D transpose lowering
// ============================================================================
//
// A standalone >2D tile.transpose (not consumed by a tile.batch_matmul) has no
// dedicated 2D hardware op: the backend pto.ttrans is strictly 2D. The frontend
// auto-allocates the transpose scratch (tmp) at INPUT rank (see
// python/pypto/ir/op/tile_ops.py), so a 3D input yields a 3D tmp. The generic
// re-create path would substitute a flattened 2D tmp while leaving the input
// rank at 3, tripping the input-rank == tmp-rank CHECK in
// DeduceTileTransposeType.
//
// This lowering mirrors LowerBatchMatmul's non-fused path: it handles only the
// last-two-axes swap (axes {ndim-2, ndim-1}) with leading batch dims. The
// already-flattened 2D input [batch_count*A, B] is sliced per batch into an
// [A, B] page, transposed via a genuine 2D tile.transpose into [B, A], and
// assembled into a flat [batch_count*B, A] output tile. Every emitted transpose
// is 2D, so no codegen change is needed. Transposes that move a batch axis
// cannot be expressed as a per-batch 2D ttrans and are rejected with a clear
// user-facing error.

/// Lower a standalone >2D last-two-axes tile.transpose into per-batch 2D
/// tile.transpose calls assembled into a flat 2D output.
NdTransposeResult LowerNdTranspose(const AssignStmtPtr& assign, const CallPtr& call,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span) {
  NdTransposeResult out;

  // Read the ORIGINAL (pre-substitution) input type: its >2D dims are intact.
  auto input_type = As<TileType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(input_type, span)
      << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";
  auto input_dims = ToStaticDims(input_type->shape_, "tile.transpose");
  size_t ndim = input_dims.size();
  INTERNAL_CHECK_SPAN(ndim > 2, span)
      << "Internal error: LowerNdTranspose called on a tile of rank " << ndim << " (expected >2)";

  // Normalize axes and require a last-two-axes swap.
  auto axis1_const = As<ConstInt>(call->args_[1]);
  auto axis2_const = As<ConstInt>(call->args_[2]);
  INTERNAL_CHECK_SPAN(axis1_const && axis2_const, span)
      << "Internal error: tile.transpose axes must be ConstInt in FlattenTileNdTo2D";
  int64_t axis1 = NormalizeAxisIndex(axis1_const->value_, ndim, "tile.transpose");
  int64_t axis2 = NormalizeAxisIndex(axis2_const->value_, ndim, "tile.transpose");
  CHECK_SPAN(IsTrailingMatrixAxisSwap(axis1, axis2, ndim), span)
      << "FlattenTileNdTo2D: only last-two-axes tile.transpose on >2D tiles is supported; "
      << "transpose involving a batch axis (axes " << axis1 << ", " << axis2 << " of rank " << ndim
      << ") is not yet lowered. Reshape the tile so the transposed axes are the last two.";

  std::vector<int64_t> batch_dims(input_dims.begin(), input_dims.end() - 2);
  int64_t a = input_dims[ndim - 2];
  int64_t b = input_dims[ndim - 1];
  int64_t batch_count = MultiplyStaticDims(batch_dims, "tile.transpose batch size");

  // Resolve the input operand. After var_map substitution it is usually the
  // already-flattened 2D tile [batch_count*A, B]. But a producer that this pass
  // leaves at rank>2 (e.g. a tile.reshape to a 3D shape, which the generic path
  // re-creates without flattening) yields a still-ND operand. In that case emit
  // a tile.reshape down to the merged 2D [batch_count*A, B] so the per-batch
  // tile.slice below has a 2D parent.
  auto operand = Substitute(call->args_[0], ctx.var_map);
  auto operand_type = As<TileType>(operand->GetType());
  INTERNAL_CHECK_SPAN(operand_type, span)
      << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";
  MemorySpace target_mem = operand_type->memory_space_.value_or(MemorySpace::Vec);

  if (operand_type->shape_.size() > 2) {
    auto [merged, last] = ComputeMergedShape(operand_type->shape_, "tile.transpose input");
    INTERNAL_CHECK_SPAN(merged == batch_count * a && last == b, span)
        << "Internal error: tile.transpose flattened input shape [" << merged << ", " << last
        << "] does not match expected [" << (batch_count * a) << ", " << b << "]";
    auto reshape_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(merged, last, span), span);
    auto reshape = op_registry.Create("tile.reshape", {operand, reshape_shape}, span);
    auto reshape_var = std::make_shared<Var>("trans_in_2d", reshape->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(reshape_var, reshape, span));
    operand = reshape_var;
    operand_type = As<TileType>(operand->GetType());
  }

  // Pre-create the flat output tile [batch_count*B, A].
  auto out_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * b, a, span), span);
  std::vector<std::pair<std::string, std::any>> create_kw = {
      {"dtype", operand_type->dtype_},
      {"target_memory", target_mem},
  };
  auto create_out = op_registry.Create("tile.create", {out_shape}, create_kw, span);
  VarPtr out_var = std::make_shared<Var>(assign->var_->name_hint_, create_out->GetType(), span);
  out.stmts.push_back(std::make_shared<AssignStmt>(out_var, create_out, span));

  // Pre-create one flat scratch pool [batch_count*A, B] sliced per batch.
  // pto.ttrans requires a scratch operand whose type matches the source page's;
  // its codegen reuses the SOURCE's type for BOTH ins operands
  // (MakeTileTransposeCodegenPTO emits "src_type, src_type"). The source page is
  // a tile.slice -> pto.subview, which produces a NEW SSA value with STATIC valid
  // [A, B]. The scratch must be the same kind of value, so it is sliced from this
  // pool per batch (a partial tile.slice -> pto.subview), NOT
  // created+set_validshape: set_validshape mutates the alloc in place (dynamic
  // valid ?x?) without renaming it, so ttrans would see the same SSA value typed
  // both dynamic (at its def/set_validshape) and static (at the ttrans use) ->
  // ptoas type conflict. The pool lives across the loop; being a single
  // allocation it is cheap relative to per-batch scratch churn.
  auto tmp_pool_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * a, b, span), span);
  std::vector<std::pair<std::string, std::any>> tmp_pool_kw = {
      {"dtype", operand_type->dtype_},
      {"target_memory", target_mem},
  };
  auto tmp_pool_create = op_registry.Create("tile.create", {tmp_pool_shape}, tmp_pool_kw, span);
  VarPtr tmp_pool_var = std::make_shared<Var>("trans_tmp_pool", tmp_pool_create->GetType(), span);
  out.stmts.push_back(std::make_shared<AssignStmt>(tmp_pool_var, tmp_pool_create, span));

  auto axis0_expr = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto axis1_expr = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  for (int64_t i = 0; i < batch_count; ++i) {
    auto suffix = std::to_string(i);

    // Extract the i-th dense 2D input page [A, B] from the flat operand.
    auto in_offset = MakeShapeTupleFromInts({i * a, 0}, span);
    auto in_shape = MakeShapeTupleFromInts({a, b}, span);
    auto slice = op_registry.Create("tile.slice", {operand, in_shape, in_offset}, span);
    ExprPtr src_page = std::make_shared<Var>("trans_page_" + suffix, slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(As<Var>(src_page), slice, span));

    // Slice the i-th 2D scratch page [A, B] from the flat tmp pool (subview with
    // STATIC valid [A, B], matching the source page's type exactly).
    auto tmp_offset = MakeShapeTupleFromInts({i * a, 0}, span);
    auto tmp_shape = MakeShapeTupleFromInts({a, b}, span);
    auto tmp_slice = op_registry.Create("tile.slice", {tmp_pool_var, tmp_shape, tmp_offset}, span);
    ExprPtr scratch_page = std::make_shared<Var>("trans_tmp_" + suffix, tmp_slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(As<Var>(scratch_page), tmp_slice, span));

    // Transpose the page [A, B] -> [B, A]. Ranks match, CHECK passes.
    auto transpose =
        op_registry.Create("tile.transpose", {src_page, axis0_expr, axis1_expr, scratch_page}, span);
    auto transpose_var = std::make_shared<Var>("trans_" + suffix, transpose->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(transpose_var, transpose, span));

    // Assemble the [B, A] result into the flat output at row offset i*B.
    auto out_offset = MakeShapeTupleFromInts({i * b, 0}, span);
    auto assemble = op_registry.Create("tile.assemble", {out_var, transpose_var, out_offset}, span);
    out_var = std::make_shared<Var>(out_var->name_hint_, assemble->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(out_var, assemble, span));
  }

  out.output_var = out_var;
  return out;
}

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto
