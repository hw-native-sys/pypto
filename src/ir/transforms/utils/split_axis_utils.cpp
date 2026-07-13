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

#include "pypto/ir/transforms/utils/split_axis_utils.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace split_axis {

int SplitDimension(SplitMode mode) {
  INTERNAL_CHECK(mode == SplitMode::UpDown || mode == SplitMode::LeftRight)
      << "Internal error: SplitDimension expects UpDown or LeftRight, got SplitMode("
      << static_cast<int>(mode) << ")";
  return (mode == SplitMode::UpDown) ? 0 : 1;
}

bool IsReduceOnSplitAxis(const CallPtr& call, int split_dim) {
  // Submits carry a GlobalVar callee and no op_; reduce ops are always plain
  // Calls with a non-null op_, so this guard correctly skips Submits (no
  // SubmitPtr handling needed — see pass-submit-awareness.md).
  if (!call->op_) return false;

  auto input_tile_type = [&]() -> std::shared_ptr<const TileType> {
    if (call->args_.empty()) return nullptr;
    return std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
  };

  if (IsOp(call, "tile.row_sum") || IsOp(call, "tile.row_max") || IsOp(call, "tile.row_min") ||
      IsOp(call, "tile.row_prod") || IsOp(call, "tile.row_argmax") || IsOp(call, "tile.row_argmin")) {
    auto tt = input_tile_type();
    int last_axis = tt ? static_cast<int>(tt->shape_.size()) - 1 : 1;
    return split_dim == last_axis;
  }
  // Column reductions collapse the first axis (axis 0). Splitting on that axis
  // (SplitMode::UpDown) would leave each lane with a partial reduction.
  if (IsOp(call, "tile.col_sum") || IsOp(call, "tile.col_max") || IsOp(call, "tile.col_min") ||
      IsOp(call, "tile.col_prod") || IsOp(call, "tile.col_argmax") || IsOp(call, "tile.col_argmin")) {
    return split_dim == 0;
  }
  if (IsOp(call, "tile.sum") || IsOp(call, "tile.max") || IsOp(call, "tile.min")) {
    int axis = call->GetKwarg<int>("axis", -1);
    auto tt = input_tile_type();
    if (axis < 0 && tt) {
      axis = static_cast<int>(tt->shape_.size()) + axis;
    }
    return axis == split_dim;
  }
  return false;
}

namespace {

bool IsSingletonDim(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return ci->value_ == 1;
  }
  return false;
}

// Half-dim computation. Throws on odd ConstInt because silently floor-dividing
// odd dims would drop data, and reliably padding the box requires
// producer/consumer/slot-size co-ordination that lives outside this pass.
// Users who need odd extents should pad the tile box to a multiple of the
// producer's innerDim and narrow back with pl.tile.set_validshape; see
// docs/en/dev/passes/22-split_vector_kernel.md.
ExprPtr ComputeHalfDimSize(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    if ((ci->value_ % 2) != 0) {
      throw pypto::ValueError(
          "SplitVectorKernel requires an even split dimension, got " + std::to_string(ci->value_) +
          ". Pad the tile box such that the halved dimension stays a multiple of the producer's "
          "innerDim — i.e. pad the full box to a multiple of (2 * innerDim) (e.g. 32 for Acc "
          "fractal=1024, or 64/elem_bytes for fractal=512) — and use pl.tile.set_validshape(...) "
          "with the original odd extent.");
    }
    return std::make_shared<ConstInt>(ci->value_ / 2, ci->dtype(), ci->span_);
  }
  auto two = std::make_shared<ConstInt>(2, GetScalarDtype(dim_size), dim_size->span_);
  return MakeFloorDiv(dim_size, two, dim_size->span_);
}

ExprPtr MakeConstLike(const ExprPtr& ref, int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, GetScalarDtype(ref), span);
}

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr LocalizeValidDimForSplit(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                                 const ExprPtr& half_dim_size, const ExprPtr& subblock_idx) {
  if (!valid_dim) return valid_dim;
  if (!subblock_idx) {
    return half_dim_size;
  }
  if (AreExprsEqual(valid_dim, original_dim)) {
    return half_dim_size;
  }

  auto span = valid_dim->span_;
  auto zero = MakeConstLike(valid_dim, 0, span);
  auto subblock_offset = MakeMul(subblock_idx, half_dim_size, span);
  auto remaining = MakeSub(valid_dim, subblock_offset, span);
  return MakeMax(MakeMin(remaining, half_dim_size, span), zero, span);
}

// Whether a tile.set_validshape split-axis operand must be localized to the
// current subblock. Localization (subtracting half on lane 1) is only correct
// when the valid extent genuinely spans both lanes -- i.e. it equals the full
// pre-split extent, or it provably overflows the halved physical box (in which
// case leaving it unlocalized would also trip the PTOAS "operand <= shape dim"
// verifier). A smaller operand is a *replicated* valid extent both AIV lanes
// share (e.g. a fused-attention head count, valid_row=5 on a [16]->[8] split):
// localizing it would collapse lane 1 to 0 and silently corrupt that lane.
bool ValidOperandNeedsLocalize(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                               const ExprPtr& half_dim_size) {
  if (!valid_dim) return false;
  if (AreExprsEqual(valid_dim, original_dim)) return true;
  auto valid_const = std::dynamic_pointer_cast<const ConstInt>(valid_dim);
  auto half_const = std::dynamic_pointer_cast<const ConstInt>(half_dim_size);
  return valid_const != nullptr && half_const != nullptr && valid_const->value_ > half_const->value_;
}

CallPtr RebuildCallWithSplit(const CallPtr& call, int split_int) {
  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_int));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_int));
  }
  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->GetType(), call->span_);
}

TypePtr HalveTileShape(const TypePtr& type, int dim, const ExprPtr& subblock_idx) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = ComputeHalfDimSize(tt->shape_[dim]);

  // Keep TileView.valid_shape consistent with halved physical shape, and for
  // partial valid regions localize the split dimension to the current subblock.
  std::optional<TileView> new_tile_view = tt->tile_view_;
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], new_shape[dim], subblock_idx);
    }
    new_tile_view = std::move(tv);
  }

  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

ExprPtr HalveTupleElement(const ExprPtr& tuple_expr, int dim) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] = ComputeHalfDimSize(new_elements[dim]);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

ExprPtr LocalizeTupleElementForSplit(const ExprPtr& tuple_expr, int dim, const ExprPtr& original_dim,
                                     const ExprPtr& half_dim_size, const ExprPtr& subblock_idx) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] =
      LocalizeValidDimForSplit(tuple->elements_[dim], original_dim, half_dim_size, subblock_idx);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

CallPtr RebuildTpopWithHalvedShape(const CallPtr& call, int split_int, int split_dim,
                                   const ExprPtr& subblock_idx) {
  auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx);

  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_int));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_int));
  }

  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), new_result_type, call->span_);
}

ExprPtr AdjustOffsets(const ExprPtr& offsets_expr, int split_dim, const ExprPtr& half_size,
                      const ExprPtr& subblock_idx) {
  auto offsets = std::dynamic_pointer_cast<const MakeTuple>(offsets_expr);
  if (!offsets || split_dim < 0 || split_dim >= static_cast<int>(offsets->elements_.size())) {
    return offsets_expr;
  }

  std::vector<ExprPtr> new_elements = offsets->elements_;
  auto original_offset = offsets->elements_[split_dim];

  ExprPtr adjusted;
  if (auto subblock_const = std::dynamic_pointer_cast<const ConstInt>(subblock_idx)) {
    if (subblock_const->value_ == 0) {
      adjusted = original_offset;
    } else if (subblock_const->value_ == 1) {
      if (auto original_const = std::dynamic_pointer_cast<const ConstInt>(original_offset);
          original_const && original_const->value_ == 0) {
        adjusted = half_size;
      } else {
        adjusted = MakeAdd(original_offset, half_size, original_offset->span_);
      }
    }
  }

  if (!adjusted) {
    // offset = original + get_subblock_idx() * half_size
    auto adjustment = MakeMul(subblock_idx, half_size, original_offset->span_);
    adjusted = MakeAdd(original_offset, adjustment, original_offset->span_);
  }
  new_elements[split_dim] = adjusted;

  return std::make_shared<MakeTuple>(std::move(new_elements), offsets->span_);
}

TypePtr ApplyTrackedTileShape(const TypePtr& type, int dim, const ExprPtr& half_dim_size,
                              const ExprPtr& subblock_idx) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = half_dim_size;

  std::optional<TileView> new_tile_view = tt->tile_view_;
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], half_dim_size, subblock_idx);
    }
    new_tile_view = std::move(tv);
  }

  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

// Product of static tile dims in [lo, hi). Returns -1 if any dim is non-const
// (real products are >= 1, so -1 is an unambiguous "not static" sentinel).
int64_t StaticDimProduct(const std::vector<ExprPtr>& shape, int lo, int hi) {
  int64_t p = 1;
  for (int d = lo; d < hi; ++d) {
    auto ci = std::dynamic_pointer_cast<const ConstInt>(shape[d]);
    if (!ci) return -1;
    p *= ci->value_;
  }
  return p;
}

// Handle a tile.reshape whose input is an already-split tile. Reshape preserves
// row-major element order, so the split partition (first half vs second half of
// the input's split dim) lands on a specific result dimension; this finds it and
// halves that dimension, re-tracking the (possibly migrated) split axis.
//
// Returns: the rewritten statement when the split axis migrates to a *different*
// result dim (e.g. the rms_norm [N,1]->[1,N] column reshape); nullptr when it
// stays on the same dim index OR the row-major partition cannot be flat-tracked
// (caller falls through to generic halving, which also covers dynamic dims and
// the non-contiguous LEFT_RIGHT prefix); throws (reject) only when flat-tracking
// applies but no result dim carries the halved split cleanly -- rather than
// silently miscompile.
StmtPtr TryMigrateReshapeSplit(const CallPtr& call, const std::shared_ptr<const AssignStmt>& assign,
                               const std::shared_ptr<const TileType>& in_tt, int in_split_dim,
                               const ExprPtr& subblock_idx,
                               std::unordered_map<const Var*, TileInfo>& tile_vars,
                               std::unordered_map<const Var*, VarPtr>& var_replacements) {
  auto res_tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
  if (!res_tt || call->args_.size() < 2) return nullptr;
  INTERNAL_CHECK_SPAN(in_split_dim >= 0 && in_split_dim < static_cast<int>(in_tt->shape_.size()), call->span_)
      << "Internal error: input split dim " << in_split_dim << " out of bounds for rank "
      << in_tt->shape_.size();

  // Flat-offset tracking only applies when the split partition is a contiguous
  // prefix (every input dim before the split dim is 1 -- always true for the
  // dim-0 UP_DOWN split, not for a LEFT_RIGHT col split of a multi-row tile) and
  // all extents are static. Otherwise defer to generic halving (same-axis path).
  const int64_t prefix_in = StaticDimProduct(in_tt->shape_, 0, in_split_dim);
  auto orig_c = std::dynamic_pointer_cast<const ConstInt>(in_tt->shape_[in_split_dim]);
  const int64_t inner_in =
      StaticDimProduct(in_tt->shape_, in_split_dim + 1, static_cast<int>(in_tt->shape_.size()));
  if (prefix_in != 1 || !orig_c || (orig_c->value_ % 2) != 0 || inner_in < 0) return nullptr;

  // Number of elements in the first half (row-major) of the split partition.
  const int64_t split_flat = (orig_c->value_ / 2) * inner_in;

  // Find the result dim whose first half matches that flat prefix exactly.
  int d_out = -1;
  int64_t prefix_out = 1;
  for (int d = 0; d < static_cast<int>(res_tt->shape_.size()); ++d) {
    auto out_c = std::dynamic_pointer_cast<const ConstInt>(res_tt->shape_[d]);
    if (!out_c) return nullptr;  // dynamic result dim -> defer to generic
    const int64_t inner_out =
        StaticDimProduct(res_tt->shape_, d + 1, static_cast<int>(res_tt->shape_.size()));
    if (inner_out < 0) return nullptr;
    if (prefix_out == 1 && (out_c->value_ % 2) == 0 && (out_c->value_ / 2) * inner_out == split_flat) {
      d_out = d;
      break;
    }
    prefix_out *= out_c->value_;
  }
  // Flat-tracking applies but no clean per-dim halving exists -> reject.
  if (d_out < 0) {
    throw pypto::ValueError(
        "SplitVectorKernel: tile.reshape moves the split axis (dim " + std::to_string(in_split_dim) +
        ") across a layout this pass cannot track under split; keep the reduction/reshape out of the "
        "split scope.");
  }

  // Split axis stays on the same dim index -> generic halving handles it; only
  // migrate when it actually moves.
  if (d_out == in_split_dim) return nullptr;

  // Halve the migrated dim on both the reshape target arg and the result type.
  ExprPtr half_dim_size = ComputeHalfDimSize(res_tt->shape_[d_out]);
  auto new_result_type = HalveTileShape(call->GetType(), d_out, subblock_idx);
  std::vector<ExprPtr> new_args = call->args_;
  new_args[1] = HalveTupleElement(call->args_[1], d_out);

  auto new_call =
      std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
  auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
  TileInfo info{half_dim_size, d_out};
  tile_vars[assign->var_.get()] = info;
  tile_vars[new_var.get()] = info;
  var_replacements[assign->var_.get()] = new_var;
  return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
}

StmtPtr ProcessStmt(const StmtPtr& stmt, SplitMode mode, int split_int, int split_dim,
                    std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                    const ExprPtr& subblock_idx, std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call || !call->op_) return stmt;

    const auto& op_name = call->op_->name_;

    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpush_to_aic")) {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }

    // tpop_from_aic: AIV consumes from cube — halve the popped tile to match split vector lanes.
    // tpop_from_aiv: AIC consumes from vector — keep full tile shape; only sync split attribute
    // (vector-side split affects AIV compute, not the matmul operand tile delivered to cube).
    if (IsOp(call, "tile.tpop_from_aiv")) {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }
    if (IsOp(call, "tile.tpop_from_aic")) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      auto new_call = RebuildTpopWithHalvedShape(call, split_int, split_dim, subblock_idx);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        TileInfo info{ComputeHalfDimSize(tt->shape_[split_dim]), split_dim};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
      }
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.load — halve result shape, halve shape/valid_shape args, adjust offset.
    // Singleton split-dim tiles (e.g. broadcast [1, 128] under UP_DOWN) are preserved as-is.
    if (is_aiv && IsOp(call, "tile.load") && call->args_.size() >= 4) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      bool is_singleton =
          tt && split_dim < static_cast<int>(tt->shape_.size()) && IsSingletonDim(tt->shape_[split_dim]);

      if (is_singleton) {
        return stmt;
      }

      // Rank-1 (and rank-0) loads carry no 2D split axis: which physical axis is
      // "the split axis" only becomes defined once the tile is reshaped to 2D.
      // Halving them here is unsafe -- under UP_DOWN it would split a rank-1
      // column vector along the wrong axis (e.g. a [128] scale later reshaped to
      // [1, 128] would be halved to [64] and then fail to reshape). Bypass them
      // under every split mode and let the consuming reshape introduce and slice
      // the split axis (see the tile.reshape handling below). LEFT_RIGHT already
      // bypassed rank-1 loads via split_dim >= rank; this also covers UP_DOWN.
      if (!tt || static_cast<int>(tt->shape_.size()) < 2 ||
          split_dim >= static_cast<int>(tt->shape_.size())) {
        return stmt;
      }
      ExprPtr half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);

      auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = AdjustOffsets(call->args_[1], split_dim, half_dim_size, subblock_idx);
      new_args[2] = HalveTupleElement(call->args_[2], split_dim);
      new_args[3] = LocalizeTupleElementForSplit(call->args_[3], split_dim, tt->shape_[split_dim],
                                                 half_dim_size, subblock_idx);

      auto new_call =
          std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
      TileInfo info{half_dim_size, split_dim};
      tile_vars[assign->var_.get()] = info;
      tile_vars[new_var.get()] = info;
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.store — adjust offset using tracked tile info
    if (is_aiv && IsOp(call, "tile.store") && call->args_.size() >= 3) {
      auto tile_var = std::dynamic_pointer_cast<const Var>(call->args_[0]);
      if (tile_var) {
        auto it = tile_vars.find(tile_var.get());
        if (it != tile_vars.end()) {
          auto new_offsets =
              AdjustOffsets(call->args_[1], it->second.split_dim, it->second.half_dim_size, subblock_idx);
          std::vector<ExprPtr> new_args = call->args_;
          new_args[1] = new_offsets;
          auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_,
                                                 call->GetType(), call->span_);
          return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
        }
      }
    }

    // AIV only: any other op producing TileType — halve result shape (and static shape args when present).
    // Reject reduce ops that reduce on the split axis (partial reduction is semantically incorrect).
    // Skip halving when the output split-dim is singleton (broadcast / degenerate tiles).
    if (is_aiv) {
      // Find the primary tracked (already-split) tile input. Its split dim can
      // differ from the global split_dim once a reshape has migrated the split
      // axis across dimensions (the rms_norm [N,1]<->[1,N] column reshape), so the
      // reduce guard and elementwise halving must follow the input's dim.
      int in_split_dim = -1;
      std::shared_ptr<const TileType> in_tt;
      for (const auto& a : call->args_) {
        if (auto v = AsVarLike(a)) {
          auto it = tile_vars.find(v.get());
          if (it != tile_vars.end()) {
            in_split_dim = it->second.split_dim;
            in_tt = std::dynamic_pointer_cast<const TileType>(a->GetType());
            break;
          }
        }
      }

      // Reduce on the (possibly migrated) split axis is a partial reduction —
      // reject it on the input's tracked dim, not just the global split_dim.
      const int reduce_split_dim = (in_split_dim >= 0) ? in_split_dim : split_dim;
      if (IsReduceOnSplitAxis(call, reduce_split_dim)) {
        throw pypto::ValueError("SplitVectorKernel: reduce op '" + op_name +
                                "' reduces on the split axis (dim " + std::to_string(reduce_split_dim) +
                                "); partial reduction in a split kernel is not supported");
      }

      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());

      // tile.reshape that moves the split axis to a different result dim.
      if (tt && IsOp(call, "tile.reshape") && in_split_dim >= 0 && in_tt) {
        if (auto migrated = TryMigrateReshapeSplit(call, assign, in_tt, in_split_dim, subblock_idx, tile_vars,
                                                   var_replacements)) {
          return migrated;
        }
        // nullptr -> split extent stays in place; fall through to generic halving.
      }

      // Result split dim: follow the tracked input's (possibly migrated) dim; root
      // ops with no tracked input use the global split dim.
      const int result_split_dim = (in_split_dim >= 0) ? in_split_dim : split_dim;
      if (tt && result_split_dim < static_cast<int>(tt->shape_.size())) {
        if (IsSingletonDim(tt->shape_[result_split_dim])) {
          return stmt;
        }
        auto half_dim_size = ComputeHalfDimSize(tt->shape_[result_split_dim]);

        // tile.reshape lifts a full (un-split) source tile -- typically a rank-1
        // load that bypassed the split-specific load rewrite -- onto a 2D shape
        // whose split axis spans the full width. Reshape is an offsetless view, so
        // halving only its result type leaves BOTH AIV lanes reading the first
        // half of the full buffer; lane 1 then silently reuses lane 0's data
        // (observed as lane 1 applying the wrong half of the per-channel dequant
        // scale in dsv4 proj_b's INT8 GEMM epilogue). Emit the reshape at full
        // width and follow it with a per-subblock column slice so each lane reads
        // its own half. Reshapes whose input is already split fall through to the
        // plain result-halving below (their producer already partitioned the data).
        if (IsOp(call, "tile.reshape")) {
          auto input_var = AsVarLike(call->args_[0]);
          bool input_is_split = input_var && tile_vars.count(input_var.get()) != 0;
          auto half_const = std::dynamic_pointer_cast<const ConstInt>(half_dim_size);
          if (!input_is_split && half_const != nullptr) {
            auto full_var =
                std::make_shared<Var>(assign->var_->name_hint_, call->GetType(), assign->var_->span_);
            auto full_reshape = std::make_shared<AssignStmt>(full_var, call, assign->span_);

            std::vector<ExprPtr> shape_elems;
            std::vector<ExprPtr> offset_elems;
            shape_elems.reserve(tt->shape_.size());
            offset_elems.reserve(tt->shape_.size());
            for (int d = 0; d < static_cast<int>(tt->shape_.size()); ++d) {
              if (d == result_split_dim) {
                shape_elems.push_back(MakeIndexConst(half_const->value_, assign->span_));
                offset_elems.push_back(
                    MakeMul(subblock_idx, MakeIndexConst(half_const->value_, assign->span_), assign->span_));
              } else {
                auto dim_const = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[d]);
                INTERNAL_CHECK_SPAN(dim_const != nullptr, assign->span_)
                    << "SplitVectorKernel: tile.reshape non-split result dim " << d
                    << " must be static to slice the split axis";
                shape_elems.push_back(MakeIndexConst(dim_const->value_, assign->span_));
                offset_elems.push_back(MakeIndexConst(0, assign->span_));
              }
            }
            auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elems), assign->span_);
            auto offset_tuple = std::make_shared<MakeTuple>(std::move(offset_elems), assign->span_);
            auto slice_call = OpRegistry::GetInstance().Create(
                "tile.slice", {full_var, shape_tuple, offset_tuple}, {}, assign->span_);
            auto slice_var =
                std::make_shared<Var>(assign->var_->name_hint_, slice_call->GetType(), assign->var_->span_);
            auto slice_assign = std::make_shared<AssignStmt>(slice_var, slice_call, assign->span_);

            TileInfo info{half_dim_size, result_split_dim};
            // Track both the original var and the slice replacement, matching the
            // other tile-producing branches: a later tile.store / loop init that
            // references the original var (before the final Substitute) must still
            // find the tile info to adjust its split-dim offset.
            tile_vars[assign->var_.get()] = info;
            tile_vars[slice_var.get()] = info;
            var_replacements[assign->var_.get()] = slice_var;
            return std::make_shared<SeqStmts>(std::vector<StmtPtr>{full_reshape, slice_assign},
                                              assign->span_);
          }
        }

        auto new_result_type = HalveTileShape(call->GetType(), result_split_dim, subblock_idx);
        std::vector<ExprPtr> new_args = call->args_;
        if ((IsOp(call, "tile.full") || IsOp(call, "tile.create")) && call->args_.size() >= 1) {
          new_args[0] = HalveTupleElement(call->args_[0], result_split_dim);
        } else if (IsOp(call, "tile.reshape") && call->args_.size() >= 2) {
          new_args[1] = HalveTupleElement(call->args_[1], result_split_dim);
        } else if (IsOp(call, "tile.slice") && call->args_.size() >= 3) {
          // tile.slice = (src, shape, offset[, valid_shape[, drop_dims]]). The
          // generic result-type halving above shrinks the split dim of the
          // result TileType, but the static shape tuple (arg[1]) is left at full
          // width unless it is rewritten here -- codegen then emits a
          // pto.subview whose sizes (full) disagree with the partition the
          // tstore expects (half), the qk_pv strided sub-slice miscompile that
          // motivated the explicit-AIV-split RFC. Halve the shape tuple so it
          // tracks the halved result type.
          new_args[1] = HalveTupleElement(call->args_[1], result_split_dim);

          // Offset (arg[2]) localization mirrors the reshape->slice path above:
          // only add the per-subblock base when the SOURCE tile is NOT already
          // split. A split-tracked source has already been partitioned by its
          // producer, so its offset is in lane-local coordinates and must be
          // left untouched; an unsplit (full-width) source needs
          // +subblock_idx*half so each lane reads its own half.
          auto slice_src = AsVarLike(call->args_[0]);
          bool slice_src_is_split = slice_src && tile_vars.count(slice_src.get()) != 0;
          if (!slice_src_is_split) {
            new_args[2] = AdjustOffsets(call->args_[2], result_split_dim, half_dim_size, subblock_idx);
          }

          // Optional explicit valid_shape (arg[3]) must stay consistent with the
          // result type's valid_shape, which HalveTileShape already localized to
          // this subblock regardless of src split state. An empty MakeTuple
          // sentinel (the "no valid_shape" form paired with drop_dims) and the
          // optional drop_dims (arg[4]) are passed through unchanged --
          // LocalizeTupleElementForSplit is a no-op on a tuple whose split_dim
          // is out of range.
          if (call->args_.size() >= 4) {
            new_args[3] = LocalizeTupleElementForSplit(
                call->args_[3], result_split_dim, tt->shape_[result_split_dim], half_dim_size, subblock_idx);
          }
        } else if (IsOp(call, "tile.set_validshape") && call->args_.size() == 3) {
          // args = (tile, valid_row, valid_col). Halving the result type alone
          // leaves the split-dim valid operand at its full pre-split extent, so
          // a full/overflowing operand exceeds the halved physical box (PTOAS
          // rejects it with "row/col operand <= shape dim"). Localize the
          // split-dim operand the same way HalveTileShape localizes the type's
          // valid_shape -- but ONLY when it genuinely spans both lanes. A smaller
          // operand is a replicated extent both AIV lanes share; localizing it
          // would collapse lane 1 to 0 and silently corrupt that lane.
          // set_validshape carries only (tile, valid_row, valid_col), so the
          // operand index is valid only for a 2D split dim; guard against a
          // migrated/higher result_split_dim that would index past the args.
          const int operand_idx = 1 + result_split_dim;  // dim 0 -> row, 1 -> col
          if (operand_idx < static_cast<int>(call->args_.size()) &&
              ValidOperandNeedsLocalize(call->args_[operand_idx], tt->shape_[result_split_dim],
                                        half_dim_size)) {
            new_args[operand_idx] = LocalizeValidDimForSplit(
                call->args_[operand_idx], tt->shape_[result_split_dim], half_dim_size, subblock_idx);
          }
        }
        auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type,
                                               call->span_);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
        TileInfo info{half_dim_size, result_split_dim};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
        var_replacements[assign->var_.get()] = new_var;
        return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
      }
    }

    return stmt;
  }

  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    if (!call || !call->op_) return stmt;

    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpush_to_aic")) {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<EvalStmt>(new_call, eval->span_);
    }

    if (is_aiv && IsOp(call, "tile.store") && call->args_.size() >= 3) {
      auto tile_var = std::dynamic_pointer_cast<const Var>(call->args_[0]);
      if (tile_var) {
        auto it = tile_vars.find(tile_var.get());
        if (it != tile_vars.end()) {
          auto new_offsets =
              AdjustOffsets(call->args_[1], it->second.split_dim, it->second.half_dim_size, subblock_idx);
          std::vector<ExprPtr> new_args = call->args_;
          new_args[1] = new_offsets;
          auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_,
                                                 call->GetType(), call->span_);
          return std::make_shared<EvalStmt>(new_call, eval->span_);
        }
      }
    }

    return stmt;
  }

  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    // Eagerly substitute initValues while rebuilding iter_args. If this is
    // deferred to the final Substitute pass, it can create a second IterArg
    // instance whose pointer diverges from the one referenced by the rebuilt
    // loop body, breaking structural equality.
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(for_stmt->iter_args_.size());
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    // Propagate tile_vars from init values to iter_args BEFORE processing body.
    // Iter_args carry the init_value into the loop; if the init is a tracked
    // halved tile, the iter_arg must also be tracked so that operations on it
    // inside the loop body are correctly recognized.
    for (const auto& ia : for_stmt->iter_args_) {
      auto new_init_value = ia->initValue_;
      if (new_init_value && !var_replacements.empty()) {
        new_init_value = transform_utils::Substitute(new_init_value, var_replacements);
      }
      TypePtr new_type = ia->GetType();
      bool has_tracked_tile = false;
      TileInfo tracked_info;
      if (ia->initValue_) {
        if (auto init_var = AsVarLike(ia->initValue_)) {
          auto it = tile_vars.find(init_var.get());
          if (it != tile_vars.end()) {
            has_tracked_tile = true;
            tracked_info = it->second;
            tile_vars[ia.get()] = it->second;
            new_type = ApplyTrackedTileShape(ia->GetType(), it->second.split_dim, it->second.half_dim_size,
                                             subblock_idx);
          }
        }
      }

      if (new_type != ia->GetType() || new_init_value != ia->initValue_) {
        auto new_iter_arg = std::make_shared<IterArg>(ia->name_hint_, new_type, new_init_value, ia->span_);
        new_iter_args.push_back(new_iter_arg);
        var_replacements[ia.get()] = new_iter_arg;
        if (has_tracked_tile) {
          tile_vars[new_iter_arg.get()] = tracked_info;
        }
      } else {
        new_iter_args.push_back(ia);
      }
    }

    auto flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(for_stmt->body_)) {
      flat = seq->stmts_;
    } else {
      flat.push_back(for_stmt->body_);
    }
    auto new_body_stmts =
        ProcessStmts(flat, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements);
    StmtPtr new_body = (new_body_stmts.size() == 1)
                           ? new_body_stmts[0]
                           : std::make_shared<SeqStmts>(new_body_stmts, for_stmt->span_);

    // Propagate tile_vars tracking from iter_args to return_vars.
    // ForStmt return_vars are the loop-exit versions of the corresponding
    // iter_args.  If an iter_arg carries a halved tile, the return_var must
    // inherit the tile info so that downstream tile.store gets the correct
    // subblock offset adjustment.
    INTERNAL_CHECK_SPAN(for_stmt->iter_args_.size() == for_stmt->return_vars_.size(), for_stmt->span_)
        << "Internal error: ForStmt iter_args and return_vars sizes must match, got "
        << for_stmt->iter_args_.size() << " vs " << for_stmt->return_vars_.size();
    for (size_t i = 0; i < new_iter_args.size() && i < new_return_vars.size(); ++i) {
      auto it = tile_vars.find(new_iter_args[i].get());
      if (it != tile_vars.end()) {
        tile_vars[new_return_vars[i].get()] = it->second;
        auto new_type = ApplyTrackedTileShape(new_return_vars[i]->GetType(), it->second.split_dim,
                                              it->second.half_dim_size, subblock_idx);
        if (new_type != new_return_vars[i]->GetType()) {
          auto new_return_var =
              std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
          new_return_vars[i] = new_return_var;
          tile_vars[new_return_var.get()] = it->second;
          var_replacements[for_stmt->return_vars_[i].get()] = new_return_var;
        }
      }
    }

    return loop_repair::RebuildForStmt(for_stmt, new_iter_args, new_body, new_return_vars);
  }

  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto then_flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(if_stmt->then_body_)) {
      then_flat = seq->stmts_;
    } else {
      then_flat.push_back(if_stmt->then_body_);
    }
    auto new_then = ProcessStmts(then_flat, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx,
                                 var_replacements);
    StmtPtr new_then_body =
        (new_then.size() == 1) ? new_then[0] : std::make_shared<SeqStmts>(new_then, if_stmt->span_);

    std::optional<StmtPtr> new_else;
    if (const auto& else_body_opt = if_stmt->else_body_; else_body_opt.has_value()) {
      const StmtPtr& else_body = else_body_opt.value();
      auto else_flat = std::vector<StmtPtr>();
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(else_body)) {
        else_flat = seq->stmts_;
      } else {
        else_flat.push_back(else_body);
      }
      auto new_else_stmts = ProcessStmts(else_flat, mode, split_int, split_dim, tile_vars, is_aiv,
                                         subblock_idx, var_replacements);
      new_else = (new_else_stmts.size() == 1) ? new_else_stmts[0]
                                              : std::make_shared<SeqStmts>(new_else_stmts, if_stmt->span_);
    }
    auto new_if = MutableCopy(if_stmt);
    new_if->then_body_ = new_then_body;
    new_if->else_body_ = new_else;
    return new_if;
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto new_stmts = ProcessStmts(seq->stmts_, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx,
                                  var_replacements);
    return std::make_shared<SeqStmts>(new_stmts, seq->span_);
  }

  return stmt;
}

std::string ReserveFreshName(std::unordered_set<std::string>& used_names, const std::string& base_name) {
  std::string name = base_name;
  if (used_names.count(name) != 0) {
    name = auto_name::GenerateFreshNameLike(base_name, used_names);
  }
  used_names.insert(name);
  return name;
}

}  // namespace

std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_int,
                                  int split_dim, std::unordered_map<const Var*, TileInfo>& tile_vars,
                                  bool is_aiv, const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(
        ProcessStmt(stmt, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements));
  }
  return result;
}

SubblockInjectionResult InjectSubblockIdx(const FunctionPtr& func, bool is_aiv) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(func->body_);
  }

  std::unordered_set<std::string> used_names;
  for (const auto& p : func->params_) {
    used_names.insert(p->name_hint_);
  }
  std::vector<VarPtr> def_vars;
  transform_utils::CollectDefVars(func->body_, def_vars);
  for (const auto& v : def_vars) {
    used_names.insert(v->name_hint_);
  }

  if (!is_aiv) {
    return {nullptr, std::move(body_stmts), std::move(used_names)};
  }

  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::string subblock_var_name = ReserveFreshName(used_names, "subblock_idx");

  auto& op_reg = OpRegistry::GetInstance();
  auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
  auto subblock_call =
      std::make_shared<Call>(subblock_op, std::vector<ExprPtr>{},
                             std::vector<std::pair<std::string, std::any>>{}, idx_type, func->span_);
  auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, func->span_);
  auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, func->span_);
  body_stmts.insert(body_stmts.begin(), assign_stmt);
  return {subblock_idx_var, std::move(body_stmts), std::move(used_names)};
}

SubblockInjectionResult InjectSubblockIdxIntoStmts(const std::vector<StmtPtr>& region_stmts,
                                                   const std::unordered_set<std::string>& used_names) {
  // An empty region (DCE-emptied, or a ``pass``-only body whose sole binding was
  // dropped) carries no compute to localize, so there is nothing to inject a
  // per-lane index for. Return a no-op result (null index, empty body) the caller
  // splices as nothing — i.e. it erases the region rather than crashing.
  if (region_stmts.empty()) {
    return {nullptr, {}, used_names};
  }
  std::vector<StmtPtr> body_stmts = region_stmts;

  // Seed the name set with the caller-supplied names plus the region's own def
  // vars so the injected subblock index never clashes with an existing binding.
  std::unordered_set<std::string> names = used_names;
  for (const auto& s : region_stmts) {
    std::vector<VarPtr> def_vars;
    transform_utils::CollectDefVars(s, def_vars);
    for (const auto& v : def_vars) {
      names.insert(v->name_hint_);
    }
  }

  const Span& span = region_stmts.front()->span_;
  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::string subblock_var_name = ReserveFreshName(names, "subblock_idx");

  auto& op_reg = OpRegistry::GetInstance();
  auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
  auto subblock_call = std::make_shared<Call>(
      subblock_op, std::vector<ExprPtr>{}, std::vector<std::pair<std::string, std::any>>{}, idx_type, span);
  auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, span);
  auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, span);
  body_stmts.insert(body_stmts.begin(), assign_stmt);
  return {subblock_idx_var, std::move(body_stmts), std::move(names)};
}

namespace {

// Mirrors the (formerly file-local) hazard finder in ExpandMixedKernel: records
// the first tile.transpose whose source carries the split axis and whose
// transpose actually swaps it. Shared so the explicit per-region check in pass 21
// and the AUTO whole-function check in pass 22 use one detector.
class TransposeSplitHazardFinder : public IRVisitor {
 public:
  explicit TransposeSplitHazardFinder(int split_dim) : split_dim_(split_dim) {}
  [[nodiscard]] CallPtr Offending() const { return offending_; }
  [[nodiscard]] const std::string& ResultName() const { return result_name_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    Consider(As<Call>(op->value_), op->var_ ? op->var_->name_hint_ : "");
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const EvalStmtPtr& op) override {
    Consider(As<Call>(op->expr_), "");
    IRVisitor::VisitStmt_(op);
  }

 private:
  // Whether the transpose actually swaps the split axis (so the split data
  // migrates). tile.transpose carries the two axis indices as args[1]/args[2].
  [[nodiscard]] bool SwapsSplitAxis(const CallPtr& call) const {
    if (call->args_.size() < 3) return true;  // conservative if the axes are absent
    auto a0 = std::dynamic_pointer_cast<const ConstInt>(call->args_[1]);
    auto a1 = std::dynamic_pointer_cast<const ConstInt>(call->args_[2]);
    if (!a0 || !a1) return true;
    return static_cast<int>(a0->value_) == split_dim_ || static_cast<int>(a1->value_) == split_dim_;
  }

  void Consider(const CallPtr& call, const std::string& result_name) {
    if (offending_ || !call || !call->op_ || !IsOp(call, "tile.transpose") || call->args_.empty()) {
      return;
    }
    auto tt = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
    if (!tt || split_dim_ < 0 || split_dim_ >= static_cast<int>(tt->shape_.size())) return;
    if (!SwapsSplitAxis(call)) return;  // split axis not transposed -> stays put, typed correctly
    // The split axis carries real data unless it is statically 1. A dynamic
    // (non-ConstInt) extent is treated as non-singleton: it cannot be proven
    // safe, so flag it conservatively.
    auto dim = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim_]);
    if (!dim || dim->value_ != 1) {
      offending_ = call;
      result_name_ = result_name;
    }
  }

  int split_dim_;
  CallPtr offending_;
  std::string result_name_;
};

}  // namespace

TransposeSplitHazard FindTransposeSplitHazard(const StmtPtr& body, int split_dim) {
  if (!body) return {};
  TransposeSplitHazardFinder finder(split_dim);
  finder.VisitStmt(body);
  return {finder.Offending(), finder.ResultName()};
}

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto
