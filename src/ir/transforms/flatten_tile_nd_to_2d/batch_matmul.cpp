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
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/rewrite_internal.h"

namespace pypto {
namespace ir {

using transform_utils::Substitute;

namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

// ============================================================================
// Batch matmul lowering
// ============================================================================
//
// tile.batch_matmul performs batched matrix multiplication on rank>2 tiles:
//   lhs [..., M, K] x rhs [..., K, N] -> result [..., M, N]
// where "..." are broadcast-compatible batch dimensions.
//
// The 2D backend only supports tile.matmul on rank-2 tiles. This lowering
// eliminates tile.batch_matmul by unrolling the batch dimensions at compile
// time (all shapes are static) into a flat sequence of 2D tile.matmul calls.
//
// Overall flow:
//
//   1. Normalize operands — peel safe batch-only tile.reshape wrappers and flag a
//      tile.transpose_view operand (the b_trans / a_trans form: the view already
//      presents [.., K, N], so batch_matmul itself carries no transpose semantic).
//
//   2. Broadcast batch dimensions — compute the output batch shape via
//      NumPy-style broadcasting (e.g. [2,1] x [1,3] -> [2,3]).
//
//   3. Detect direct-store fusion — if the very next statement is a tile.store
//      consuming this result, fuse per-batch stores directly instead of
//      assembling into a temporary tile. This avoids an intermediate buffer.
//
//   4. Capacity gate — decide once whether both operands' whole tiles fit Mat
//      together (BatchOperandsWholeFit).
//
//   5. Unroll — for each flat batch index 0..batch_count-1:
//      a. Decompose the flat index into per-dim indices for lhs and rhs,
//         respecting broadcast (size-1 dims always map to index 0).
//      b. Extract the 2D [M,K] / [K,N] page (ExtractBatchPage). When the whole
//         tiles fit, every operand is sliced from its kept whole Mat tile (row
//         slice for plain operands, column slice for tile.transpose_view); when
//         they do not, each operand is loaded per batch instead. Transposed
//         operands are realised by a zero-copy tile.transpose_view — never a copy.
//      c. Emit tile.matmul(lhs_2d, rhs_2d).
//      d. Cast dtype if matmul output (FP32) differs from expected result dtype.
//      e. Either tile.store (fused path) or tile.assemble into output tile.
//
// The result is a flat 2D tile [batch_count*M, N] (non-fused) or a chain
// of per-batch tile.store calls (fused), with no tile.batch_matmul remaining.
//

AssignDefMap BuildAssignDefMap(const std::vector<StmtPtr>& stmts) {
  AssignDefMap map;
  for (const auto& stmt : stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      map[assign->var_.get()] = assign;
    }
  }
  return map;
}

/// Parsed information about a batch_matmul operand.
struct BatchOperandInfo {
  ExprPtr operand;                   ///< After var_map substitution
  ExprPtr original_operand;          ///< Before substitution (for def lookup)
  TileTypePtr operand_type;          ///< Type after substitution
  TileTypePtr original_type;         ///< Type before substitution
  bool from_transpose_view = false;  ///< True if the operand is a tile.transpose_view result.
                                     ///< Its trailing two dims are already swapped to the matmul
                                     ///< orientation, but when flattened the batch is concatenated
                                     ///< on the COLUMN axis ([K, B*N]), so per-batch extraction is a
                                     ///< column slice (offset {0, b*N}) rather than a row slice.
  bool whole_fits = true;            ///< Whether both operands' whole tiles fit Mat together. When
                                     ///< false, ExtractBatchPage loads this operand per batch (from
                                     ///< base_load) instead of slicing a kept whole tile.
  CallPtr base_load;                 ///< Underlying natural tile.load for this operand (traced through
                                     ///< the tile.transpose_view when from_transpose_view). Used by the
                                     ///< !fit per-batch path to re-emit a per-batch load.
};

/// Resolve an inline or single-definition `op_name` wrapper around a batch_matmul operand.
CallPtr ResolveBatchOperandCall(const ExprPtr& operand_expr, const AssignDefMap& def_map,
                                const std::string& op_name) {
  if (auto call = As<Call>(operand_expr)) {
    if (call->op_ && call->op_->name_ == op_name) return call;
  }
  if (auto operand_var = As<Var>(operand_expr)) {
    auto def_it = def_map.find(operand_var.get());
    if (def_it != def_map.end()) {
      if (auto call = As<Call>(def_it->second->value_)) {
        if (call->op_ && call->op_->name_ == op_name) return call;
      }
    }
  }
  return nullptr;
}

/// Check if a `tile.reshape` call is safe to peel when feeding `tile.batch_matmul`.
///
/// A reshape is "safe to peel" when it only reinterprets the batch portion of
/// the shape and leaves the trailing (M, N) matrix dims untouched:
///   * input and output ranks are both >= 2,
///   * the last two dims (the matmul page) are identical static values,
///   * the product of the leading batch dims is the same on both sides.
bool IsSafePeelableBatchMatmulReshape(const CallPtr& reshape_call) {
  if (!reshape_call || !reshape_call->op_ || !IsOp(reshape_call, "tile.reshape")) {
    return false;
  }
  if (reshape_call->args_.size() != 2) return false;

  auto out_type = As<TileType>(reshape_call->GetType());
  auto in_type = As<TileType>(reshape_call->args_[0]->GetType());
  if (!out_type || !in_type) return false;
  if (out_type->shape_.size() < 2 || in_type->shape_.size() < 2) return false;

  // Trailing matmul page must be preserved.
  auto in_rows = As<ConstInt>(in_type->shape_[in_type->shape_.size() - 2]);
  auto in_cols = As<ConstInt>(in_type->shape_.back());
  auto out_rows = As<ConstInt>(out_type->shape_[out_type->shape_.size() - 2]);
  auto out_cols = As<ConstInt>(out_type->shape_.back());
  if (!in_rows || !in_cols || !out_rows || !out_cols) return false;
  if (in_rows->value_ != out_rows->value_ || in_cols->value_ != out_cols->value_) return false;

  // Batch element count must be preserved (so the reshape is a pure batch reinterpretation).
  auto static_batch_product = [](const std::vector<ExprPtr>& shape) -> std::optional<int64_t> {
    int64_t product = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      auto ci = As<ConstInt>(shape[i]);
      if (!ci) return std::nullopt;
      product *= ci->value_;
    }
    return product;
  };
  auto in_batch = static_batch_product(in_type->shape_);
  auto out_batch = static_batch_product(out_type->shape_);
  if (!in_batch || !out_batch || *in_batch != *out_batch) return false;
  return true;
}

/// Peel safe tile.reshape wrappers around a batch_matmul operand.
///
/// Peeling lets `LowerBatchMatmul` look through e.g. `tile.reshape([1, M, N],
/// [1, 1, M, N])` and reuse the upstream `tile.load` operand directly. The
/// alternative (the rank>2 fallback in `ExtractBatchPage`) would otherwise emit a
/// redundant ND `tile.slice` + `tile.reshape` chain per batch element, which
/// can lower to invalid degenerate tiles for zero-valid sub-blocks.
///
/// Iterates so nested reshapes (e.g. two consecutive safe reshapes) all peel.
/// Returns the deepest safe operand, or the input unchanged when no reshape
/// is found / the reshape fails the safety conditions.
ExprPtr PeelSafeBatchReshape(const ExprPtr& operand_expr, const AssignDefMap& def_map) {
  ExprPtr current = operand_expr;
  while (true) {
    CallPtr reshape_call;
    if (auto call = As<Call>(current)) {
      if (IsOp(call, "tile.reshape")) {
        reshape_call = call;
      }
    }
    if (!reshape_call) {
      if (auto var = As<Var>(current)) {
        auto def_it = def_map.find(var.get());
        if (def_it != def_map.end()) {
          if (auto call = As<Call>(def_it->second->value_)) {
            if (IsOp(call, "tile.reshape")) {
              reshape_call = call;
            }
          }
        }
      }
    }
    if (!reshape_call) return current;
    if (!IsSafePeelableBatchMatmulReshape(reshape_call)) return current;
    current = reshape_call->args_[0];
  }
}

/// Whether a natural `tile.load`'s whole source window collapses to a contiguous
/// 2D row axis (the precondition the codegen ND2NZ collapse enforces). A
/// non-contiguous whole load (a partial middle dim under a non-singleton outer
/// dim, e.g. a multi-batch slice that also cuts the matrix-row dim) cannot be
/// legalized as one 2D ND2NZ load, so the operand must instead be re-emitted per
/// batch (ExtractBatchPage !fit path). Returns true (keep whole) when the load is
/// absent / 2D / dynamic-shaped. The contiguity rule itself is shared with the
/// codegen guard via `IsRowMajorCollapseContiguous`, so routing and guard agree.
bool WholeLoadContiguous(const CallPtr& base_load) {
  if (!base_load || base_load->args_.size() < 4) return true;
  auto tensor_type = AsTensorTypeLike(base_load->args_[0]->GetType());
  auto valid = As<MakeTuple>(base_load->args_[3]);
  if (!tensor_type || !valid) return true;
  const size_t ndim = valid->elements_.size();
  if (ndim <= 2 || tensor_type->shape_.size() != ndim) return true;
  return tile_conversion_utils::IsRowMajorCollapseContiguous(valid->elements_, tensor_type->shape_);
}

/// The per-operand whole-vs-per-batch routing decision, shared by LowerBatchMatmul,
/// LowerBatchMatmulAcc, and the dead-load drop pre-scan so all three stay in sync:
/// keep this operand whole only when both operands' whole tiles fit Mat together
/// (the joint `capacity_fits` gate) AND its whole load collapses contiguously;
/// otherwise it is re-emitted per batch.
bool KeepOperandWhole(bool capacity_fits, const CallPtr& base_load) {
  return capacity_fits && WholeLoadContiguous(base_load);
}

/// Trace a batch_matmul operand var to its underlying natural `tile.load` (through
/// safe batch reshape wrappers and a tile.transpose_view), mirroring
/// NormalizeBatchMatmulOperand's base_load resolution. Used by the drop pre-scan
/// to apply the same whole-vs-per-batch routing decision as LowerBatchMatmul.
CallPtr TraceOperandBaseLoad(const ExprPtr& operand_expr, const AssignDefMap& def_map) {
  ExprPtr base = PeelSafeBatchReshape(operand_expr, def_map);
  if (auto tv = ResolveBatchOperandCall(base, def_map, "tile.transpose_view")) {
    if (!tv->args_.empty()) base = tv->args_[0];
  }
  return ResolveBatchOperandCall(base, def_map, "tile.load");
}

/// Normalize one batch_matmul operand:
///  - peel safe batch-only tile.reshape wrappers that only reinterpret batch dims
///  - recognize a tile.transpose_view operand (the canonical b_trans/a_trans form):
///    its trailing dims are already swapped to the matmul orientation, so no
///    per-batch transpose is needed; we only record that per-batch extraction must
///    column-slice (the flattened view concatenates batch on the column axis).
///  - return the base operand plus type information.
BatchOperandInfo NormalizeBatchMatmulOperand(const ExprPtr& operand_expr, const std::string& operand_name,
                                             const AssignDefMap& def_map, const FlattenContext& ctx) {
  BatchOperandInfo info;
  // Peel safe batch-only tile.reshape wrappers first so the transpose_view check
  // below sees the underlying operand directly.
  ExprPtr base_operand = PeelSafeBatchReshape(operand_expr, def_map);

  // A b_trans/a_trans operand arrives as a tile.transpose_view (issues #1776 / ND
  // extension): the view already presents the operand in [.., K, N] orientation, so
  // batch_matmul carries no transpose semantic. Keep the view as the operand (it is
  // a whole Mat tile we slice per batch); just flag column-slicing.
  if (ResolveBatchOperandCall(base_operand, def_map, "tile.transpose_view") != nullptr) {
    info.from_transpose_view = true;
  }

  info.original_operand = base_operand;
  info.original_type = As<TileType>(base_operand->GetType());
  CHECK(info.original_type) << "FlattenTileNdTo2D: tile.batch_matmul " << operand_name
                            << " expects TileType operand, but got " << base_operand->GetType()->TypeName();

  // Trace the underlying NATURAL tile.load (through the tile.transpose_view when
  // transposed). The !fit per-batch path re-emits a per-batch load from it.
  ExprPtr load_src = base_operand;
  if (info.from_transpose_view) {
    auto tv = ResolveBatchOperandCall(base_operand, def_map, "tile.transpose_view");
    if (tv && !tv->args_.empty()) load_src = tv->args_[0];
  }
  info.base_load = ResolveBatchOperandCall(load_src, def_map, "tile.load");

  info.operand = Substitute(base_operand, ctx.var_map);
  info.operand_type = As<TileType>(info.operand->GetType());
  CHECK(info.operand_type) << "FlattenTileNdTo2D: tile.batch_matmul substituted " << operand_name
                           << " expects TileType operand, but got " << info.operand->GetType()->TypeName();
  return info;
}

/// Build batch-adjusted offset elements: add batch indices to the batch dimensions
/// of base offsets, then append the trailing matrix-dimension offsets unchanged.
std::vector<ExprPtr> BuildBatchAdjustedOffsets(const std::vector<ExprPtr>& base_offset_elems,
                                               const std::vector<int64_t>& batch_indices, size_t batch_rank,
                                               const Span& span) {
  std::vector<ExprPtr> adjusted;
  adjusted.reserve(base_offset_elems.size());
  for (size_t dim = 0; dim < batch_rank; ++dim) {
    if (batch_indices[dim] == 0) {
      adjusted.push_back(base_offset_elems[dim]);
    } else {
      auto offset = std::make_shared<ConstInt>(batch_indices[dim], DataType::INDEX, span);
      adjusted.push_back(MakeCanonicalIndexAdd(base_offset_elems[dim], offset, span));
    }
  }
  for (size_t dim = batch_rank; dim < base_offset_elems.size(); ++dim) {
    adjusted.push_back(base_offset_elems[dim]);
  }
  return adjusted;
}

/// Result of extracting a 2D batch page from a rank>2 operand.
struct BatchPageResult {
  VarPtr var;                  ///< The 2D variable (possibly transposed)
  std::vector<StmtPtr> stmts;  ///< Statements emitted to produce it
};

/// Extract the 2D matrix page for one batch index from a batch_matmul operand.
///
/// Every operand — lhs or rhs, transposed or not, load- or move-sourced — is
/// handled identically:
///  * whole_fits (default): the operand's whole tile is already in Mat; take its
///    [source_rows, source_cols] page with a single tile.slice — a ROW slice for a
///    plain (row-batched) operand, a COLUMN slice for a tile.transpose_view
///    (column-batched) operand. A broadcast operand reuses its single page.
///  * !whole_fits (large operands): load THIS operand per batch from its
///    underlying natural tile.load, adding a per-batch tile.transpose_view when the
///    operand is transposed. The dead whole load/view is dropped during rewriting.
///
/// The operand is always 2D by the time it reaches here (loads flatten to 2D,
/// transpose_views are 2D, safe batch-only reshapes are peeled to the 2D load).
BatchPageResult ExtractBatchPage(const BatchOperandInfo& info, const std::vector<int64_t>& operand_dims,
                                 const std::vector<int64_t>& operand_batch_shape, int64_t batch_index,
                                 const std::string& base_name, const FlattenContext& ctx,
                                 const OpRegistry& op_registry, const Span& span) {
  BatchPageResult page;
  const auto& operand = info.operand;
  const auto& operand_type = info.operand_type;

  int64_t source_rows = operand_dims[operand_dims.size() - 2];
  int64_t source_cols = operand_dims.back();
  std::string suffix = std::to_string(batch_index);

  VarPtr current;

  if (!info.whole_fits && info.base_load) {
    // !fit + load (GM): the operands' whole tiles do not fit Mat together, so load
    // THIS operand PER BATCH from its underlying natural tile.load (a per-batch
    // [1,..,X,Y] window → 2D [X,Y], which the hardware ND2NZ path accepts as the
    // leading dims are 1). A transposed operand then gets a per-batch
    // tile.transpose_view — same as the whole-tile path, just one batch at a time.
    // The whole load/view is dropped upstream (no longer referenced) so it does not
    // occupy L1.
    //
    // NOTE: this only covers load-sourced (GM) operands. A move-sourced operand
    // (V2C mixed kernel: a Vec compute result moved to Mat) has base_load == null,
    // so a !fit V2C operand falls through to the whole-slice path below — correct
    // only while the whole moved tile fits the fixed cross-core ring. A per-batch
    // V2C move for large shapes is a deferred follow-up (see BatchOperandsWholeFit).
    auto load_tensor = info.base_load->args_[0];
    auto load_tensor_var = AsVarLike(load_tensor);
    auto load_tensor_type = AsTensorTypeLike(load_tensor->GetType());
    auto base_offsets = As<MakeTuple>(info.base_load->args_[1]);
    auto base_shapes = As<MakeTuple>(info.base_load->args_[2]);
    INTERNAL_CHECK_SPAN(load_tensor_var && load_tensor_type && base_offsets && base_shapes &&
                            load_tensor_type->shape_.size() >= 2 &&
                            base_shapes->elements_.size() == load_tensor_type->shape_.size(),
                        span)
        << "FlattenTileNdTo2D: !fit per-batch load expects a tensor-backed tile.load with rank >= 2";
    // Use the load's WINDOW matrix dims (the actual sliced tile), not the source
    // tensor's full trailing dims — they differ when the operand is a partial
    // sub-tile of a larger tensor (e.g. a multi-batch slice that also cuts the
    // matrix-row dim, the non-contiguous case routed here).
    const size_t win_rank = base_shapes->elements_.size();
    auto x_dim = As<ConstInt>(base_shapes->elements_[win_rank - 2]);
    auto y_dim = As<ConstInt>(base_shapes->elements_.back());
    INTERNAL_CHECK_SPAN(x_dim && y_dim, span)
        << "FlattenTileNdTo2D: !fit per-batch load needs static trailing dims";

    auto batch_indices = BuildBatchIndices(batch_index, operand_batch_shape);
    auto load_offset_elems =
        BuildBatchAdjustedOffsets(base_offsets->elements_, batch_indices, operand_batch_shape.size(), span);
    std::vector<int64_t> load_shape_values(operand_batch_shape.size(), 1);
    load_shape_values.push_back(x_dim->value_);
    load_shape_values.push_back(y_dim->value_);
    auto load_offsets = std::make_shared<MakeTuple>(load_offset_elems, span);
    auto load_shape = As<MakeTuple>(MakeShapeTupleFromInts(load_shape_values, span));
    INTERNAL_CHECK_SPAN(load_shape, span) << "FlattenTileNdTo2D: !fit per-batch load shape must be a tuple";
    auto view_call = CreateCollapsedTensorView(load_tensor_var, load_tensor_type, span);
    auto view_var = std::make_shared<Var>(base_name + "_pbview2d_" + suffix, view_call->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(view_var, view_call, span));

    auto row_offset = CollapseLeadingOffsetsToRow(load_offsets->elements_, load_tensor_type->shape_, span);
    auto load_2d_shape = CollapseLeadingDimsTo2D(load_shape->elements_, span);
    auto load_2d_offsets =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{row_offset, load_offsets->elements_.back()}, span);
    auto load_2d_shape_tuple = std::make_shared<MakeTuple>(load_2d_shape, span);
    std::vector<ExprPtr> load_args;
    load_args.reserve(4);
    load_args.push_back(view_var);
    load_args.push_back(load_2d_offsets);
    load_args.push_back(load_2d_shape_tuple);
    load_args.push_back(load_2d_shape_tuple);
    auto deduced_load = op_registry.Create("tile.load", load_args, info.base_load->kwargs_, span);
    auto load_2d =
        std::make_shared<Call>(deduced_load->op_, deduced_load->args_, deduced_load->kwargs_,
                               info.base_load->attrs_, deduced_load->GetType(), deduced_load->span_);

    // The source tensor view and load window are both collapsed to 2D here, so
    // codegen sees a regular 2D ND2NZ Mat load instead of a rank>2 source view.
    auto load_2d_type = As<TileType>(load_2d->GetType());
    INTERNAL_CHECK_SPAN(load_2d_type && load_2d_type->shape_.size() == 2, span)
        << "FlattenTileNdTo2D: !fit per-batch collapsed load must produce a 2D tile";
    current = std::make_shared<Var>(base_name + "_pbload_" + suffix, load_2d_type, span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, load_2d, span));

    if (info.from_transpose_view) {
      auto view = op_registry.Create("tile.transpose_view", {current}, {}, span);
      auto view_var = std::make_shared<Var>(base_name + "_pbview_" + suffix, view->GetType(), span);
      page.stmts.push_back(std::make_shared<AssignStmt>(view_var, view, span));
      current = view_var;
    }
    page.var = current;
    return page;

  } else {
    // Whole-fit: slice the [source_rows, source_cols] page from the kept whole 2D
    // tile. The operand is ALWAYS 2D here — a load flattens to a 2D result, a
    // tile.transpose_view is 2D, and a safe batch-only reshape is peeled to its 2D
    // load — so no rank>2 fallback is needed (verified: this assert fires on no
    // test across the full UT + ST suites). A plain operand is row-batched
    // ([B*rows, cols]) so the page is a row slice at row b*source_rows; a
    // tile.transpose_view operand is COLUMN-batched ([source_rows, B*source_cols] =
    // [K, B*N]: the whole-tile transpose concatenates the batches along the column
    // axis), so its page is a column slice at col b*source_cols. Either way the
    // page is [source_rows, source_cols].
    INTERNAL_CHECK_SPAN(operand_type->shape_.size() == 2, span)
        << "FlattenTileNdTo2D: batch_matmul operand must be flattened to 2D before "
           "ExtractBatchPage, got rank "
        << operand_type->shape_.size();
    std::vector<int64_t> offset_values = info.from_transpose_view
                                             ? std::vector<int64_t>{0, batch_index * source_cols}
                                             : std::vector<int64_t>{batch_index * source_rows, 0};
    auto offset = MakeShapeTupleFromInts(offset_values, span);
    auto shape = MakeShapeTupleFromInts({source_rows, source_cols}, span);
    auto slice = op_registry.Create("tile.slice", {operand, shape, offset}, span);
    current = std::make_shared<Var>(base_name + "_slice_" + suffix, slice->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, slice, span));
  }

  // No per-batch transpose: a transposed (b_trans/a_trans) operand arrives as a
  // tile.transpose_view whose page is already in the matmul orientation (the
  // column-slice above extracts batch_b^T directly).
  page.var = current;
  return page;
}

/// Detect whether the next statement is a tile.store consuming the batch_matmul result.
struct DirectStoreInfo {
  bool detected = false;
  AssignStmtPtr store_assign;
  CallPtr store_call;
};

DirectStoreInfo DetectDirectStore(const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                  const VarPtr& result_var) {
  DirectStoreInfo info;
  if (stmt_index + 1 >= stmts.size()) return info;

  auto store_assign = As<AssignStmt>(stmts[stmt_index + 1]);
  auto store_call = store_assign ? As<Call>(store_assign->value_) : nullptr;
  if (!store_call || !IsOp(store_call, "tile.store")) return info;

  auto store_input = !store_call->args_.empty() ? As<Var>(store_call->args_[0]) : nullptr;
  if (!store_input || store_input.get() != result_var.get()) return info;

  info.detected = true;
  info.store_assign = store_assign;
  info.store_call = store_call;
  return info;
}

/// Result of lowering a tile.batch_matmul operation.

/// Lower tile.batch_matmul into unrolled 2D tile.matmul calls.
///
/// Enumerates every batch index combination, extracts the 2D matrix page from each
/// operand, emits a tile.matmul per batch element, and either assembles results into
/// a flat 2D output tile or fuses directly into per-batch tile.store when possible.
BatchMatmulResult LowerBatchMatmul(const AssignStmtPtr& assign, const CallPtr& call,
                                   const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span) {
  BatchMatmulResult out;
  auto def_map = BuildAssignDefMap(stmts);

  // Normalize operands.
  auto lhs_info = NormalizeBatchMatmulOperand(call->args_[0], "lhs", def_map, ctx);
  auto rhs_info = NormalizeBatchMatmulOperand(call->args_[1], "rhs", def_map, ctx);
  // Route each operand to whole-slice vs per-batch independently: keep whole only
  // when the operands' whole tiles fit Mat together (capacity) AND this operand's
  // whole load collapses contiguously. A non-contiguous whole load (multi-batch +
  // partial matrix-row dim) is re-emitted per batch — each per-batch page is a
  // [1, X, Y] window that collapses cleanly — instead of erroring in codegen.
  const bool capacity_fits = BatchOperandsWholeFit(lhs_info.original_type, rhs_info.original_type);
  lhs_info.whole_fits = KeepOperandWhole(capacity_fits, lhs_info.base_load);
  rhs_info.whole_fits = KeepOperandWhole(capacity_fits, rhs_info.base_load);
  auto orig_result_type = As<TileType>(call->GetType());
  CHECK(orig_result_type) << "FlattenTileNdTo2D: tile.batch_matmul expects TileType result";

  // Extract static dimensions.
  auto lhs_dims = ToStaticDims(lhs_info.original_type->shape_, "tile.batch_matmul lhs");
  auto rhs_dims = ToStaticDims(rhs_info.original_type->shape_, "tile.batch_matmul rhs");
  CHECK(lhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul lhs must be at least 2D";
  CHECK(rhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul rhs must be at least 2D";

  // Compute broadcast batch dimensions.
  std::vector<ExprPtr> lhs_batch_exprs(lhs_info.original_type->shape_.begin(),
                                       lhs_info.original_type->shape_.end() - 2);
  std::vector<ExprPtr> rhs_batch_exprs(rhs_info.original_type->shape_.begin(),
                                       rhs_info.original_type->shape_.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch_exprs, rhs_batch_exprs);
  CHECK(broadcast_result.success) << "FlattenTileNdTo2D: tile.batch_matmul batch dimensions must broadcast";

  auto output_batch_dims = ToStaticDims(broadcast_result.shape, "tile.batch_matmul output batch");
  int64_t batch_count = MultiplyStaticDims(output_batch_dims, "tile.batch_matmul output batch size");

  std::vector<int64_t> lhs_batch_dims(lhs_dims.begin(), lhs_dims.end() - 2);
  std::vector<int64_t> rhs_batch_dims(rhs_dims.begin(), rhs_dims.end() - 2);

  // Matrix dimensions. A transposed operand already arrives in matmul orientation
  // via its tile.transpose_view (original_type is the post-transpose [.., K, N]),
  // so the trailing two dims are used directly — no swap.
  int64_t lhs_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_cols = lhs_dims.back();
  int64_t rhs_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_cols = rhs_dims.back();

  // K-match is validated user-facing at op construction (DeduceTileBatchMatMulType);
  // a mismatch here would be a compiler bug in an earlier pass.
  INTERNAL_CHECK_SPAN(lhs_cols == rhs_rows, span)
      << "Internal error: tile.batch_matmul inner dimensions must match, but got " << lhs_cols << " and "
      << rhs_rows;

  // Detect direct-store fusion opportunity.
  auto direct_store = DetectDirectStore(stmts, stmt_index, assign->var_);

  // Fast path: batch_count == 1, non-fused, and no dtype cast required. The
  // result tile is exactly what a single tile.matmul produces (2D, Acc). Skip
  // the create + per-batch move-to-Vec + tile.assemble dance and let the Acc
  // tile flow directly to the consumer. This is essential when the consumer is
  // tile.matmul_acc / tile.batch_matmul_acc — those need an Acc accumulator,
  // and a Vec-staged tile would force an illegal cross-core Vec→Acc move at
  // codegen time. Any downstream Vec consumer can still insert its own Acc→Vec
  // move.
  //
  // Skip the fast path when the deduced tile.matmul accumulator dtype differs
  // from the requested orig_result_type dtype: returning the raw Acc tile
  // would leak the wider accumulator dtype (e.g. fp32/int32) instead of the
  // expected output dtype, and the cast must be inserted in Vec memory by the
  // general path below.
  if (batch_count == 1 && !direct_store.detected) {
    auto output_batch_indices = BuildBatchIndices(0, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    auto matmul = op_registry.Create("tile.matmul", {lhs_page.var, rhs_page.var}, span);
    auto matmul_type = As<TileType>(matmul->GetType());
    bool needs_cast = matmul_type && matmul_type->dtype_ != orig_result_type->dtype_;
    if (!needs_cast) {
      out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
      out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());
      auto matmul_var = std::make_shared<Var>(assign->var_->name_hint_, matmul->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul, span));
      out.output_var = matmul_var;
      return out;
    }
    // Discard the speculative pages and matmul (no out.stmts modification yet);
    // fall through to the general path which inserts the required tile.cast.
  }

  // Allocate output tile (non-fused path only).
  VarPtr out_var;
  if (!direct_store.detected) {
    auto out_shape =
        std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * lhs_rows, rhs_cols, span), span);
    // Per-batch matmul results are moved to Vec via tile.move(target_memory=Vec)
    // before being assembled into this tile, so allocate the staging tile in Vec
    // up-front. This keeps the printed/parsed IR consistent (parser otherwise
    // backfills target_memory=Vec from the assemble consumer chain).
    std::vector<std::pair<std::string, std::any>> create_kw = {
        {"dtype", orig_result_type->dtype_},
        {"target_memory", MemorySpace::Vec},
    };
    auto create_out = op_registry.Create("tile.create", {out_shape}, create_kw, span);
    out_var = std::make_shared<Var>(assign->var_->name_hint_, create_out->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(out_var, create_out, span));
  }

  // Prepare direct-store state.
  ExprPtr current_store_tensor;
  MakeTuplePtr direct_store_offsets;
  std::vector<ExprPtr> direct_store_shape;
  if (direct_store.detected) {
    current_store_tensor = Substitute(direct_store.store_call->args_[2], ctx.var_map);
    direct_store_offsets = As<MakeTuple>(Substitute(direct_store.store_call->args_[1], ctx.var_map));
    auto store_tensor_type = As<TensorType>(current_store_tensor->GetType());
    CHECK(store_tensor_type) << "FlattenTileNdTo2D: tile.batch_matmul direct store target must be TensorType";
    CHECK(direct_store_offsets) << "FlattenTileNdTo2D: tile.store offsets must be a MakeTuple";
    CHECK(direct_store_offsets->elements_.size() == output_batch_dims.size() + 2)
        << "FlattenTileNdTo2D: tile.store offsets rank must match batch_matmul result rank";
    if (store_tensor_type->shape_.size() > 2) {
      // Build the original tensor-rank partition shape:
      // [1, ..., 1, M, N] (left-padded with 1s for batch dims)
      const size_t tensor_rank = store_tensor_type->shape_.size();
      const size_t tile_rank = 2;  // matmul result is always 2D
      direct_store_shape.reserve(tensor_rank);
      for (size_t i = tile_rank; i < tensor_rank; ++i) {
        direct_store_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
      }
      direct_store_shape.push_back(std::make_shared<ConstInt>(lhs_rows, DataType::INDEX, span));
      direct_store_shape.push_back(std::make_shared<ConstInt>(rhs_cols, DataType::INDEX, span));
    }
  }

  // Unroll batch dimensions.
  for (int64_t i = 0; i < batch_count; ++i) {
    auto output_batch_indices = BuildBatchIndices(i, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    // Extract 2D pages.
    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    // Emit tile.matmul.
    auto matmul = op_registry.Create("tile.matmul", {lhs_page.var, rhs_page.var}, span);
    auto matmul_var = std::make_shared<Var>("matmul_" + std::to_string(i), matmul->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul, span));

    // Move matmul result from Acc to Vec, then cast dtype if needed.
    // The explicit tile.move is always required for the non-fused (assemble) path so
    // that ExpandMixedKernel sees a clear AIC→AIV boundary. For the fused (direct
    // store) path, the tile.store codegen handles the Acc→DDR transfer directly.
    ExprPtr batch_result = matmul_var;
    auto batch_result_type = As<TileType>(matmul_var->GetType());
    bool needs_cast = batch_result_type && batch_result_type->dtype_ != orig_result_type->dtype_;
    if (!direct_store.detected || needs_cast) {
      std::vector<std::pair<std::string, std::any>> move_kw = {
          {"target_memory", MemorySpace::Vec},
      };
      auto move = op_registry.Create("tile.move", {matmul_var}, move_kw, span);
      auto move_var = std::make_shared<Var>("matmul_vec_" + std::to_string(i), move->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(move_var, move, span));
      batch_result = move_var;
    }
    if (needs_cast) {
      std::vector<std::pair<std::string, std::any>> cast_kw = {
          {"target_type", orig_result_type->dtype_},
          {"mode", 2},
      };
      auto cast = op_registry.Create("tile.cast", {batch_result}, cast_kw, span);
      auto cast_var = std::make_shared<Var>("matmul_cast_" + std::to_string(i), cast->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(cast_var, cast, span));
      batch_result = cast_var;
    }

    if (direct_store.detected) {
      // Fused path: emit per-batch tile.store.
      // Keep the original tensor-rank offsets — codegen reconstructs the
      // corresponding partition_view from that window description.
      auto store_offset_elems = BuildBatchAdjustedOffsets(
          direct_store_offsets->elements_, output_batch_indices, output_batch_dims.size(), span);
      auto store_offset = std::make_shared<MakeTuple>(store_offset_elems, span);

      std::vector<ExprPtr> store_args = {batch_result, store_offset, current_store_tensor};
      if (!direct_store_shape.empty()) {
        store_args.push_back(std::make_shared<MakeTuple>(direct_store_shape, span));
      }
      auto batch_store = op_registry.Create("tile.store", store_args, span);
      auto batch_store_var =
          std::make_shared<Var>(direct_store.store_assign->var_->name_hint_ + "_" + std::to_string(i),
                                batch_store->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(batch_store_var, batch_store, span));
      current_store_tensor = batch_store_var;
    } else {
      // Non-fused path: assemble into output tile.
      auto out_offset = MakeShapeTupleFromInts({i * lhs_rows, 0}, span);
      auto assemble = op_registry.Create("tile.assemble", {out_var, batch_result, out_offset}, span);
      out_var = std::make_shared<Var>(out_var->name_hint_, assemble->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(out_var, assemble, span));
    }
  }

  if (direct_store.detected) {
    auto final_store_var = As<Var>(current_store_tensor);
    CHECK(final_store_var) << "FlattenTileNdTo2D: expected final direct store result to be a Var";
    out.fused_store = true;
    out.store_result_var = final_store_var;
    out.store_orig_var = direct_store.store_assign->var_;
  } else {
    out.output_var = out_var;
  }

  return out;
}

// ============================================================================
// Batch matmul_acc lowering
// ============================================================================
//
// tile.batch_matmul_acc semantics:
//   acc[..., M, N] += lhs[..., M, K] @ rhs[..., K, N]   (with batch broadcast)
//
// The 2D backend only supports tile.matmul_acc on rank-2 tiles. After earlier
// flattening (which has already turned the original ND acc into its flat 2D form
// [batch_count*M, N]), this lowering unrolls the batch dim into a sequence of
// per-batch tile.matmul_acc calls writing into the corresponding row-band of acc.
//
// Direct-store fusion is intentionally not applied here — the canonical use is
// "y_acc = matmul; for k: y_acc = matmul_acc(y_acc, ...); store(y_acc)" where
// the store consumes the loop-carried accumulator after the loop, not the
// individual matmul_acc results. The acc operand itself is the in-place target.
//

/// Lower tile.batch_matmul_acc into unrolled 2D tile.matmul_acc calls.
BatchMatmulAccResult LowerBatchMatmulAcc(const AssignStmtPtr& assign, const CallPtr& call,
                                         const std::vector<StmtPtr>& stmts, const FlattenContext& ctx,
                                         const OpRegistry& op_registry, const Span& span) {
  (void)assign;
  BatchMatmulAccResult out;
  auto def_map = BuildAssignDefMap(stmts);

  // The acc operand has already been flattened (or is naturally 2D) by earlier
  // statement processing; substitute via var_map to pick up any rewrites.
  // Accept both Var and IterArg (loop-carried accumulator) — both are Var-like
  // in the IR and downstream code only needs name_hint_ + a stable Expr.
  auto acc_operand = Substitute(call->args_[0], ctx.var_map);
  auto acc_var = AsVarLike(acc_operand);
  CHECK(acc_var) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc must be a Var/IterArg after "
                    "substitution, got "
                 << acc_operand->TypeName();
  auto acc_type = As<TileType>(acc_operand->GetType());
  CHECK(acc_type) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc must be TileType";
  CHECK(acc_type->shape_.size() == 2)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc expects acc to be 2D after flatten, got rank "
      << acc_type->shape_.size();

  // Normalize lhs/rhs operands (peel safe reshape, flag tile.transpose_view).
  auto lhs_info = NormalizeBatchMatmulOperand(call->args_[1], "lhs", def_map, ctx);
  auto rhs_info = NormalizeBatchMatmulOperand(call->args_[2], "rhs", def_map, ctx);
  // Route each operand to whole-slice vs per-batch independently: keep whole only
  // when the operands' whole tiles fit Mat together (capacity) AND this operand's
  // whole load collapses contiguously. A non-contiguous whole load (multi-batch +
  // partial matrix-row dim) is re-emitted per batch — each per-batch page is a
  // [1, X, Y] window that collapses cleanly — instead of erroring in codegen.
  const bool capacity_fits = BatchOperandsWholeFit(lhs_info.original_type, rhs_info.original_type);
  lhs_info.whole_fits = KeepOperandWhole(capacity_fits, lhs_info.base_load);
  rhs_info.whole_fits = KeepOperandWhole(capacity_fits, rhs_info.base_load);

  // Extract original (pre-flatten) static dimensions for batch + matrix axes.
  auto lhs_dims = ToStaticDims(lhs_info.original_type->shape_, "tile.batch_matmul_acc lhs");
  auto rhs_dims = ToStaticDims(rhs_info.original_type->shape_, "tile.batch_matmul_acc rhs");
  CHECK(lhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul_acc lhs must be at least 2D";
  CHECK(rhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul_acc rhs must be at least 2D";

  // Compute broadcast batch dims (must equal acc's batch by op contract).
  std::vector<ExprPtr> lhs_batch_exprs(lhs_info.original_type->shape_.begin(),
                                       lhs_info.original_type->shape_.end() - 2);
  std::vector<ExprPtr> rhs_batch_exprs(rhs_info.original_type->shape_.begin(),
                                       rhs_info.original_type->shape_.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch_exprs, rhs_batch_exprs);
  CHECK(broadcast_result.success)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc batch dimensions must broadcast";

  auto output_batch_dims = ToStaticDims(broadcast_result.shape, "tile.batch_matmul_acc output batch");
  int64_t batch_count = MultiplyStaticDims(output_batch_dims, "tile.batch_matmul_acc output batch size");

  std::vector<int64_t> lhs_batch_dims(lhs_dims.begin(), lhs_dims.end() - 2);
  std::vector<int64_t> rhs_batch_dims(rhs_dims.begin(), rhs_dims.end() - 2);

  // Transposed operands arrive in matmul orientation via tile.transpose_view, so
  // the trailing two dims are used directly (no swap).
  int64_t lhs_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_cols = lhs_dims.back();
  int64_t rhs_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_cols = rhs_dims.back();

  // K-match is validated user-facing at op construction (DeduceTileBatchMatMulAccType);
  // a mismatch here would be a compiler bug in an earlier pass.
  INTERNAL_CHECK_SPAN(lhs_cols == rhs_rows, span)
      << "Internal error: tile.batch_matmul_acc inner dimensions must match, got " << lhs_cols << " and "
      << rhs_rows;

  // Sanity check on flat acc shape: should be [batch_count*M, N].
  auto acc_rows_const = As<ConstInt>(acc_type->shape_[0]);
  auto acc_cols_const = As<ConstInt>(acc_type->shape_[1]);
  CHECK(acc_rows_const && acc_cols_const)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc expects static acc dims after flatten";
  CHECK(acc_rows_const->value_ == batch_count * lhs_rows)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc acc rows " << acc_rows_const->value_ << " != batch_count("
      << batch_count << ") * M(" << lhs_rows << ")";
  CHECK(acc_cols_const->value_ == rhs_cols) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc cols "
                                            << acc_cols_const->value_ << " != N(" << rhs_cols << ")";

  // Memory-space concerns (Vec/Acc round-trips on the acc operand, retargetable
  // producer promotion of the loop-carried tile.create, and matching TileView
  // layout refresh) belong to InferTileMemorySpace (pass 17, runs immediately
  // after this pass). See:
  //   * DemandCollector — propagates the matmul_acc Acc input_constraint back
  //     through inherit-input ops.
  //   * TileMemorySpaceAnalyzer::VisitStmt_(ForStmtPtr) — explicitly
  //     back-propagates the yield's memory space to the iter_arg AND its
  //     initValue (covering the dummy tile.create dummy-acc-init pattern in
  //     issue #1235).
  //   * TileMemorySpaceMutator::VisitStmt_(AssignStmtPtr) — rewrites the
  //     retargetable producer's target_memory kwarg and refreshes its
  //     TileView via GetImplicitTileView.
  // FlattenTileNdTo2D's job here is purely the shape lowering: pass the acc
  // operand through and emit 2D tile.matmul_acc (and per-batch
  // tile.slice/tile.assemble in the general path). Any required tile.move
  // calls are inserted by InferTileMemorySpace's MoveCollector. This avoids
  // the cross-core Vec→Acc move that previously failed verification in mixed
  // CUBE/VECTOR kernels (issue #1235).
  VarPtr current_acc = acc_var;

  // Fast path: batch_count == 1. The acc is already [M, N] and per-batch slicing
  // would be identity. Emit a single tile.matmul_acc directly. This also avoids
  // tile.slice/tile.assemble in Acc memory which is the standard codegen path
  // covered by the existing 2D tile.matmul_acc handling.
  if (batch_count == 1) {
    auto output_batch_indices = BuildBatchIndices(0, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    auto matmul_acc = op_registry.Create("tile.matmul_acc", {current_acc, lhs_page.var, rhs_page.var}, span);
    auto new_acc = std::make_shared<Var>(current_acc->name_hint_, matmul_acc->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(new_acc, matmul_acc, span));
    out.output_var = new_acc;
    return out;
  }

  // General path: unroll batch dims using slice + tile.matmul_acc + assemble on acc.
  for (int64_t i = 0; i < batch_count; ++i) {
    auto output_batch_indices = BuildBatchIndices(i, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    auto suffix = std::to_string(i);
    auto acc_offset = MakeShapeTupleFromInts({i * lhs_rows, 0}, span);
    auto acc_shape = MakeShapeTupleFromInts({lhs_rows, rhs_cols}, span);
    auto acc_slice = op_registry.Create("tile.slice", {current_acc, acc_shape, acc_offset}, span);
    auto acc_page_var = std::make_shared<Var>("acc_page_" + suffix, acc_slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(acc_page_var, acc_slice, span));

    auto matmul_acc = op_registry.Create("tile.matmul_acc", {acc_page_var, lhs_page.var, rhs_page.var}, span);
    auto matmul_var = std::make_shared<Var>("matmul_acc_" + suffix, matmul_acc->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul_acc, span));

    auto assemble = op_registry.Create("tile.assemble", {current_acc, matmul_var, acc_offset}, span);
    current_acc = std::make_shared<Var>(current_acc->name_hint_, assemble->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(current_acc, assemble, span));
  }

  out.output_var = current_acc;
  return out;
}

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto
