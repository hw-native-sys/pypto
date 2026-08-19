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

#include <algorithm>
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
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
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

  // Row reductions collapse the last axis. Splitting on that axis
  // (SplitMode::LeftRight for a 2D tile) would leave each lane with a partial
  // reduction. Reduce ops always take the tile first, so a missing / non-Tile
  // arg0 can only be malformed IR — assume the 2D last axis.
  if (IsOp(call, "tile.row_sum") || IsOp(call, "tile.row_max") || IsOp(call, "tile.row_min") ||
      IsOp(call, "tile.row_prod") || IsOp(call, "tile.row_argmax") || IsOp(call, "tile.row_argmin")) {
    std::shared_ptr<const TileType> input_tile;
    if (!call->args_.empty() && call->args_[0]) {
      input_tile = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
    }
    const int last_axis = input_tile ? static_cast<int>(input_tile->shape_.size()) - 1 : 1;
    return split_dim == last_axis;
  }
  // Column reductions collapse the first axis (axis 0). Splitting on that axis
  // (SplitMode::UpDown) would leave each lane with a partial reduction.
  if (IsOp(call, "tile.col_sum") || IsOp(call, "tile.col_max") || IsOp(call, "tile.col_min") ||
      IsOp(call, "tile.col_prod") || IsOp(call, "tile.col_argmax") || IsOp(call, "tile.col_argmin")) {
    return split_dim == 0;
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

bool IsUnsupportedAutoSplitGenerator(const CallPtr& call) {
  return IsOp(call, "tile.ci") || IsOp(call, "tile.random");
}

// Whether a split-axis extent is statically ODD, i.e. the two AIV lanes hold
// DIFFERENT extents (lane 0 = ceil, lane 1 = floor). A dynamic extent has no
// compile-time parity and is treated as even (the pre-existing floordiv
// behaviour); ShardSplitCode keeps such a boundary on the even split code.
bool IsOddSplitExtent(const ExprPtr& dim_size) {
  auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size);
  return ci != nullptr && (ci->value_ % 2) != 0;
}

// How far apart the two lanes' data sits for a tile whose physical half is
// @p box_half: the body's partition stride when ResolveLaneStride found one,
// else the box half itself (the default box partition).
ExprPtr LaneStep(const ExprPtr& lane_stride, const ExprPtr& box_half) {
  return lane_stride ? lane_stride : box_half;
}

ExprPtr MakeConstLike(const ExprPtr& ref, int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, GetScalarDtype(ref), span);
}

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

// Lane L's valid extent on the split axis: ``clamp(V - L * S, 0, S)``, where
// @p lane_extent is the partition stride S (the tile's physical half under the
// default box partition, the balanced ``ceil(V / 2)`` under a rebalanced body).
ExprPtr LocalizeValidDimForSplit(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                                 const ExprPtr& lane_extent, const ExprPtr& subblock_idx) {
  if (!valid_dim) return valid_dim;
  if (!subblock_idx) {
    return lane_extent;
  }
  // Shortcut: a fully-valid axis whose stride is EXACTLY half of it splits into
  // two identical lanes, so the stride is already the per-lane extent. Anything
  // else falls through to the clamp — an odd axis (the ceil stride over-states
  // lane 1 by one cell, which is what pto-isa reads the runtime extents for),
  // and a rebalanced stride (which is narrower than the full axis by design).
  // A dynamic axis keeps the shortcut, as it had before parity mattered.
  if (AreExprsEqual(valid_dim, original_dim)) {
    auto original_const = std::dynamic_pointer_cast<const ConstInt>(original_dim);
    auto extent_const = std::dynamic_pointer_cast<const ConstInt>(lane_extent);
    const bool exact_half = original_const == nullptr || extent_const == nullptr ||
                            extent_const->value_ * 2 == original_const->value_;
    if (exact_half) return lane_extent;
  }

  auto span = valid_dim->span_;
  auto subblock_offset = MakeMul(subblock_idx, lane_extent, span);
  // Saturating subtract, NOT `max(valid - offset, 0)`: a valid extent can carry
  // an UNSIGNED dtype (tile.set_validshape accepts UINT64), where `valid -
  // offset` wraps to a huge value for the lane the extent does not reach and the
  // clamp would then hand that lane a FULL half instead of nothing. Clamping the
  // minuend first is correct for both signednesses.
  auto reached = MakeMax(valid_dim, subblock_offset, span);
  auto remaining = MakeSub(reached, subblock_offset, span);
  return MakeMin(remaining, lane_extent, span);
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

CallPtr RebuildCallWithSplit(const CallPtr& call, int split_code) {
  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_code));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_code));
  }
  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->GetType(), call->span_);
}

// The halved tile's TileView.
//
// An explicit pre-split view is carried through with its split-axis extent
// localized to the current lane. A tile with NO view is fully valid, which
// needs no view of its own after an EVEN halving — both lanes fill their box.
// An ODD halving is the exception: the ceil box over-states lane 1 by one cell,
// so the lane's true extent has to be materialized as a view even though the
// pre-split tile carried none. Downstream that extent is what reaches the
// store's row count and pto-isa's per-lane TPOP valid operands.
std::optional<TileView> HalvedTileView(const std::shared_ptr<const TileType>& tt, int dim,
                                       const std::vector<ExprPtr>& new_shape, const ExprPtr& subblock_idx,
                                       const ExprPtr& lane_stride) {
  const ExprPtr step = LaneStep(lane_stride, new_shape[dim]);
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], step, subblock_idx);
    }
    return tv;
  }
  // A tile with no view is fully valid; on the box partition that stays true
  // per lane unless the axis is odd. A rebalanced body always needs the view:
  // its stride is narrower than the box, so neither lane fills its box.
  if (!subblock_idx || (!lane_stride && !IsOddSplitExtent(tt->shape_[dim]))) return std::nullopt;
  TileView tv = tile_view_semantics::GetEffectiveTileView(*tt);
  tv.valid_shape = new_shape;
  tv.valid_shape[dim] = LocalizeValidDimForSplit(tt->shape_[dim], tt->shape_[dim], step, subblock_idx);
  return tv;
}

TypePtr HalveTileShape(const TypePtr& type, int dim, const ExprPtr& subblock_idx,
                       const ExprPtr& lane_stride) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = ComputeHalfDimSize(tt->shape_[dim]);

  auto new_tile_view = HalvedTileView(tt, dim, new_shape, subblock_idx, lane_stride);
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

CallPtr RebuildTpopWithHalvedShape(const CallPtr& call, int split_code, int split_dim,
                                   const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx, lane_stride);

  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_code));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_code));
  }

  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), new_result_type, call->span_);
}

// The two AIV lanes' split-axis extents for a boundary tile, when the box, the
// valid extent and the partition stride are all compile-time constants.
//
// ``eL = clamp(V - L * S, 0, S)`` — the same clamp LocalizeValidDimForSplit
// materializes into the popped tile's valid_shape, and therefore exactly what
// PTOAS hands pto-isa as the tile's runtime valid extent.
struct LaneExtents {
  int64_t lane0 = 0;
  int64_t lane1 = 0;
  int64_t stride = 0;    ///< the partition stride S
  int64_t box_half = 0;  ///< ceil(box / 2), the per-lane physical box
  int64_t valid = 0;     ///< V, the pre-split valid extent on the split axis
};

std::optional<LaneExtents> ComputeLaneExtents(const std::shared_ptr<const TileType>& tt, int split_dim,
                                              const ExprPtr& lane_stride) {
  if (!tt || split_dim < 0 || split_dim >= static_cast<int>(tt->shape_.size())) return std::nullopt;
  auto box = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim]);
  if (!box) return std::nullopt;
  const auto valid_shape = tile_view_semantics::GetEffectiveTileView(*tt).valid_shape;
  if (split_dim >= static_cast<int>(valid_shape.size())) return std::nullopt;
  auto valid = std::dynamic_pointer_cast<const ConstInt>(valid_shape[split_dim]);
  if (!valid) return std::nullopt;

  LaneExtents extents;
  extents.box_half = (box->value_ + 1) / 2;
  extents.valid = valid->value_;
  extents.stride = extents.box_half;
  if (lane_stride) {
    auto stride_const = std::dynamic_pointer_cast<const ConstInt>(lane_stride);
    if (!stride_const) return std::nullopt;
    extents.stride = stride_const->value_;
  }
  extents.lane0 = std::min(valid->value_, extents.stride);
  extents.lane1 = std::min(std::max(valid->value_ - extents.stride, static_cast<int64_t>(0)), extents.stride);
  return extents;
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
                              const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = half_dim_size;

  auto new_tile_view = HalvedTileView(tt, dim, new_shape, subblock_idx, lane_stride);
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

bool HasAutoEquivalentReinterpretShape(const CallPtr& call) {
  INTERNAL_CHECK(call) << "Internal error: auto-shape comparison requires a non-null Call";
  INTERNAL_CHECK_SPAN(IsOp(call, "tile.reinterpret_view"), call->span_)
      << "Internal error: auto-shape comparison expects tile.reinterpret_view";
  INTERNAL_CHECK_SPAN(call->args_.size() == 2, call->span_)
      << "Internal error: explicit tile.reinterpret_view must carry data and shape arguments";

  auto explicit_type = std::dynamic_pointer_cast<const TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(explicit_type, call->span_)
      << "Internal error: tile.reinterpret_view must produce TileType";
  auto auto_call =
      OpRegistry::GetInstance().Create("tile.reinterpret_view", {call->args_[0]}, call->kwargs_, call->span_);
  auto auto_type = std::dynamic_pointer_cast<const TileType>(auto_call->GetType());
  INTERNAL_CHECK_SPAN(auto_type, call->span_)
      << "Internal error: auto-shaped tile.reinterpret_view must produce TileType";
  if (explicit_type->shape_.size() != auto_type->shape_.size()) return false;
  for (size_t i = 0; i < explicit_type->shape_.size(); ++i) {
    if (!AreExprsEqual(explicit_type->shape_[i], auto_type->shape_[i])) return false;
  }
  return true;
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
                               std::unordered_map<const Var*, VarPtr>& var_replacements,
                               const ExprPtr& lane_stride) {
  auto res_tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
  if (!res_tt || call->args_.size() < 2) return nullptr;
  INTERNAL_CHECK_SPAN(in_split_dim >= 0 && in_split_dim < static_cast<int>(in_tt->shape_.size()), call->span_)
      << "Internal error: input split dim " << in_split_dim << " out of bounds for rank "
      << in_tt->shape_.size();

  // Flat-offset tracking only applies when the split partition is a contiguous
  // prefix (every input dim before the split dim is 1 -- always true for the
  // dim-0 UP_DOWN split, not for a LEFT_RIGHT col split of a multi-row tile),
  // all extents are static, and the input split axis is EVEN -- an odd axis has
  // no exact element count to match a result dim's first half against. Anything
  // else defers to generic halving (same-axis path), which ceil-halves.
  const int64_t prefix_in = StaticDimProduct(in_tt->shape_, 0, in_split_dim);
  auto orig_c = std::dynamic_pointer_cast<const ConstInt>(in_tt->shape_[in_split_dim]);
  const int64_t inner_in =
      StaticDimProduct(in_tt->shape_, in_split_dim + 1, static_cast<int>(in_tt->shape_.size()));

  // An ODD input split axis has no exact element count to match a result dim's
  // first half against, so migration cannot be expressed. Falling through is
  // only safe when the generic (same-axis) path can still carry the split: if
  // the result's own split dim is gone or singleton -- `[15, 1] -> [1, 15]` --
  // that path leaves the reshape FULL width while each lane holds a half, so
  // both lanes would read the same rows. Reject instead of miscompiling.
  if (orig_c && (orig_c->value_ % 2) != 0) {
    const bool same_axis_carries_split = in_split_dim < static_cast<int>(res_tt->shape_.size()) &&
                                         !IsSingletonDim(res_tt->shape_[in_split_dim]);
    CHECK_SPAN(same_axis_carries_split, call->span_)
        << "SplitVectorKernel: tile.reshape moves an ODD split axis (dim " << in_split_dim << ", extent "
        << orig_c->value_ << ") onto a result whose dim " << in_split_dim
        << " cannot carry it, and an odd axis cannot be tracked through the reshape (its two lanes hold "
           "different extents). Pad that axis to an even extent and narrow it back with "
           "pl.tile.set_validshape(...), or keep the reshape out of the split scope.";
    return nullptr;
  }
  if (prefix_in != 1 || !orig_c || inner_in < 0) return nullptr;

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

  // A migrated axis carries its own half, which is derived from the RESULT dim's
  // box — it cannot express a partition balanced on another axis's valid extent.
  CHECK_SPAN(!lane_stride, call->span_)
      << "SplitVectorKernel: tile.reshape moves the split axis from dim " << in_split_dim << " to dim "
      << d_out
      << " inside a split region whose ragged boundary was balanced across the two AIV lanes. The two "
         "partitions cannot be reconciled: pad the boundary tile's valid extent to its physical box "
         "with pl.tile.set_validshape(...), or move the reshape out of the split scope.";

  // Halve the migrated dim on both the reshape target arg and the result type.
  ExprPtr half_dim_size = ComputeHalfDimSize(res_tt->shape_[d_out]);
  auto new_result_type = HalveTileShape(call->GetType(), d_out, subblock_idx, /*lane_stride=*/nullptr);
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

StmtPtr ProcessStmt(const StmtPtr& stmt, SplitMode mode, int split_dim,
                    std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                    const ExprPtr& subblock_idx, std::unordered_map<const Var*, VarPtr>& var_replacements,
                    const ExprPtr& lane_stride) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call || !call->op_) return stmt;

    const auto& op_name = call->op_->name_;

    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpush_to_aic")) {
      INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
          << "Internal error: " << op_name << " must carry the pushed tile";
      const int push_code =
          IsOp(call, "tile.tpush_to_aic")
              ? GatherSplitCode(mode, call->args_[0]->GetType(), split_dim, op_name, call->span_)
              : ShardSplitCode(mode, call->args_[0]->GetType(), split_dim, lane_stride, op_name, call->span_);
      auto new_call = RebuildCallWithSplit(call, push_code);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }

    // tpop_from_aic: AIV consumes from cube — halve the popped tile to match split vector lanes.
    // tpop_from_aiv: AIC consumes from vector — keep full tile shape; only sync split attribute
    // (vector-side split affects AIV compute, not the matmul operand tile delivered to cube).
    if (IsOp(call, "tile.tpop_from_aiv")) {
      auto new_call =
          RebuildCallWithSplit(call, GatherSplitCode(mode, call->GetType(), split_dim, op_name, call->span_));
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }
    if (IsOp(call, "tile.tpop_from_aic")) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      auto new_call = RebuildTpopWithHalvedShape(
          call, ShardSplitCode(mode, call->GetType(), split_dim, lane_stride, op_name, call->span_),
          split_dim, subblock_idx, lane_stride);
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

      auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx, lane_stride);
      const ExprPtr load_step = LaneStep(lane_stride, half_dim_size);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = AdjustOffsets(call->args_[1], split_dim, load_step, subblock_idx);
      new_args[2] = HalveTupleElement(call->args_[2], split_dim);
      new_args[3] = LocalizeTupleElementForSplit(call->args_[3], split_dim, tt->shape_[split_dim], load_step,
                                                 subblock_idx);

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
          auto new_offsets = AdjustOffsets(call->args_[1], it->second.split_dim,
                                           LaneStep(lane_stride, it->second.half_dim_size), subblock_idx);
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

      // An explicit reinterpret shape may redistribute bytes across dimensions.
      // The AIV split machinery deliberately tracks a physical axis, not a flat
      // element interval, so only the canonical auto shape is safe to halve. An
      // explicit spelling of that same shape is accepted and rewritten below;
      // arbitrary byte-equivalent shapes must stay outside the split scope.
      if (tt && IsOp(call, "tile.reinterpret_view") && call->args_.size() == 2) {
        CHECK_SPAN(HasAutoEquivalentReinterpretShape(call), call->span_)
            << "SplitVectorKernel: tile.reinterpret_view with an explicit shape inside a split kernel "
               "must match its auto-inferred shape. Omit shape= (recommended), or move an arbitrary "
               "byte-equivalent reinterpret_view outside the split scope.";
      }

      // tile.reshape that moves the split axis to a different result dim.
      // Do not route reinterpret_view through flat element-count migration: its
      // accepted auto shape keeps the physical split axis at the same index.
      if (tt && IsOp(call, "tile.reshape") && in_split_dim >= 0 && in_tt) {
        if (auto migrated = TryMigrateReshapeSplit(call, assign, in_tt, in_split_dim, subblock_idx, tile_vars,
                                                   var_replacements, lane_stride)) {
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
        CHECK_SPAN(!IsUnsupportedAutoSplitGenerator(call), call->span_)
            << "SplitVectorKernel: automatic split-axis halving of '" << op_name
            << "' is not supported because its generated values depend on position. Move the operation "
               "outside the automatically-halved split region.";
        auto half_dim_size = ComputeHalfDimSize(tt->shape_[result_split_dim]);

        // tile.reshape or tile.reinterpret_view lifts a full (un-split) source
        // tile -- typically a rank-1 load for reshape, or an untracked tile
        // parameter for reinterpret_view -- onto a shape whose split axis spans
        // the full width. These are offsetless views, so halving only the result
        // type leaves BOTH AIV lanes reading the first
        // half of the full buffer; lane 1 then silently reuses lane 0's data
        // (observed as lane 1 applying the wrong half of the per-channel dequant
        // scale in dsv4 proj_b's INT8 GEMM epilogue). Emit the view at full
        // width and follow it with a per-subblock column slice so each lane reads
        // its own half. Views whose input is already split fall through to the
        // plain result-halving below (their producer already partitioned the data).
        if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) {
          auto input_var = AsVarLike(call->args_[0]);
          bool input_is_split = input_var && tile_vars.count(input_var.get()) != 0;
          auto half_const = std::dynamic_pointer_cast<const ConstInt>(half_dim_size);
          if (!input_is_split && IsOp(call, "tile.reinterpret_view")) {
            CHECK_SPAN(half_const != nullptr, call->span_)
                << "SplitVectorKernel: tile.reinterpret_view over a full-width source requires a static "
                   "split extent so the pass can materialize a per-lane slice. Split/load the source "
                   "before reinterpret_view, or move reinterpret_view outside the split scope.";
          }
          if (!input_is_split && half_const != nullptr) {
            auto full_var =
                std::make_shared<Var>(assign->var_->name_hint_, call->GetType(), assign->var_->span_);
            auto full_view = std::make_shared<AssignStmt>(full_var, call, assign->span_);

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
                if (IsOp(call, "tile.reinterpret_view")) {
                  CHECK_SPAN(dim_const != nullptr, call->span_)
                      << "SplitVectorKernel: tile.reinterpret_view over a full-width source requires a "
                         "static target shape so the pass can materialize a per-lane slice. Split/load "
                         "the source before reinterpret_view, or move reinterpret_view outside the split "
                         "scope.";
                }
                INTERNAL_CHECK_SPAN(dim_const != nullptr, assign->span_)
                    << "Internal error: tile.reshape non-split result dim " << d
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
            return std::make_shared<SeqStmts>(std::vector<StmtPtr>{full_view, slice_assign}, assign->span_);
          }
        }

        auto new_result_type = HalveTileShape(call->GetType(), result_split_dim, subblock_idx, lane_stride);
        const ExprPtr lane_step = LaneStep(lane_stride, half_dim_size);
        std::vector<ExprPtr> new_args = call->args_;
        if ((IsOp(call, "tile.full") || IsOp(call, "tile.create")) && call->args_.size() >= 1) {
          new_args[0] = HalveTupleElement(call->args_[0], result_split_dim);
        } else if ((IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) &&
                   call->args_.size() >= 2) {
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
            new_args[2] = AdjustOffsets(call->args_[2], result_split_dim, lane_step, subblock_idx);
          }

          // Optional explicit valid_shape (arg[3]) must stay consistent with the
          // result type's valid_shape, which HalveTileShape already localized to
          // this subblock regardless of src split state. An empty MakeTuple
          // sentinel (the "no valid_shape" form paired with drop_dims) and the
          // optional drop_dims (arg[4]) are passed through unchanged --
          // LocalizeTupleElementForSplit is a no-op on a tuple whose split_dim
          // is out of range.
          if (call->args_.size() >= 4) {
            new_args[3] = LocalizeTupleElementForSplit(call->args_[3], result_split_dim,
                                                       tt->shape_[result_split_dim], lane_step, subblock_idx);
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
              ValidOperandNeedsLocalize(call->args_[operand_idx], tt->shape_[result_split_dim], lane_step)) {
            new_args[operand_idx] = LocalizeValidDimForSplit(
                call->args_[operand_idx], tt->shape_[result_split_dim], lane_step, subblock_idx);
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
      INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
          << "Internal error: " << call->op_->name_ << " must carry the pushed tile";
      const int push_code =
          IsOp(call, "tile.tpush_to_aic")
              ? GatherSplitCode(mode, call->args_[0]->GetType(), split_dim, call->op_->name_, call->span_)
              : ShardSplitCode(mode, call->args_[0]->GetType(), split_dim, lane_stride, call->op_->name_,
                               call->span_);
      auto new_call = RebuildCallWithSplit(call, push_code);
      return std::make_shared<EvalStmt>(new_call, eval->span_);
    }

    if (is_aiv && IsOp(call, "tile.store") && call->args_.size() >= 3) {
      auto tile_var = std::dynamic_pointer_cast<const Var>(call->args_[0]);
      if (tile_var) {
        auto it = tile_vars.find(tile_var.get());
        if (it != tile_vars.end()) {
          auto new_offsets = AdjustOffsets(call->args_[1], it->second.split_dim,
                                           LaneStep(lane_stride, it->second.half_dim_size), subblock_idx);
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
                                             subblock_idx, lane_stride);
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
        ProcessStmts(flat, mode, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements, lane_stride);
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
                                              it->second.half_dim_size, subblock_idx, lane_stride);
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
    auto new_then = ProcessStmts(then_flat, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                 var_replacements, lane_stride);
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
      auto new_else_stmts = ProcessStmts(else_flat, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                         var_replacements, lane_stride);
      new_else = (new_else_stmts.size() == 1) ? new_else_stmts[0]
                                              : std::make_shared<SeqStmts>(new_else_stmts, if_stmt->span_);
    }
    auto new_if = MutableCopy(if_stmt);
    new_if->then_body_ = new_then_body;
    new_if->else_body_ = new_else;
    return new_if;
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto new_stmts = ProcessStmts(seq->stmts_, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                  var_replacements, lane_stride);
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

std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_dim,
                                  std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                                  const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements,
                                  const ExprPtr& lane_stride) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(
        ProcessStmt(stmt, mode, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements, lane_stride));
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

CallPtr AsCall(const ExprPtr& expr) { return std::dynamic_pointer_cast<const Call>(expr); }

// Replace one axis of a TileType's valid_shape, preserving everything else about
// the type (layout, memref, memory space) — the localization is metadata-only.
TypePtr WithValidDim(const std::shared_ptr<const TileType>& tt, int dim, const ExprPtr& valid_dim) {
  TileView tv = tile_view_semantics::GetEffectiveTileView(*tt);
  INTERNAL_CHECK(dim >= 0 && dim < static_cast<int>(tv.valid_shape.size()))
      << "Internal error: split dim " << dim << " out of range for a rank-" << tv.valid_shape.size()
      << " valid_shape";
  tv.valid_shape[dim] = valid_dim;
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, std::move(tv), tt->memory_space_);
}

// A consumer PASSES THROUGH the split-axis extent when its result is the same
// physical box as the operand and carries the operand's own (pre-localization)
// valid extents — an elementwise op, a cast, a move. A consumer that reshapes
// the logical rectangle (a reduction, a slice, an extract) reports a different
// valid_shape, and rewriting its split-axis extent to the lane's would be a lie
// about what it computes; those are rejected by the caller instead.
bool PassesThroughValidShape(const std::shared_ptr<const TileType>& operand_before,
                             const std::shared_ptr<const TileType>& result) {
  if (!operand_before || !result) return false;
  if (!tile_view_semantics::ShapeExprListsEquivalent(operand_before->shape_, result->shape_)) {
    return false;
  }
  return tile_view_semantics::ShapeExprListsEquivalent(
      tile_view_semantics::GetEffectiveTileView(*operand_before).valid_shape,
      tile_view_semantics::GetEffectiveTileView(*result).valid_shape);
}

bool IsProvablyPositive(const ExprPtr& extent) {
  auto c = std::dynamic_pointer_cast<const ConstInt>(extent);
  return c != nullptr && c->value_ > 0;
}

// What a tracked (localized) var carries: the per-lane extent now on its split
// axis, plus the type it had BEFORE localization, which is what a consumer's
// deduced valid_shape is compared against to decide pass-through.
struct LocalizedTile {
  ExprPtr lane_extent;                     ///< clamp(V - idx*half, 0, half)
  ExprPtr joined_extent;                   ///< V, the pre-shard extent a gather restores
  std::shared_ptr<const TileType> before;  ///< the type before localization
};

}  // namespace

namespace {

// Scan for the balanced partition stride (see ResolveLaneStride in the header).
//
// Walks the PRE-lowering body in order, tracking which vars carry
// boundary-derived data. It answers one question — "does every split-axis tile
// in this body come from one ragged Cube -> Vector boundary?" — and any doubt
// (a second boundary shape, a gather, an independently split tile, a dynamic
// extent) makes it decline, leaving the universal box partition in place.
class LaneStrideScanner : public IRVisitor {
 public:
  explicit LaneStrideScanner(int split_dim) : split_dim_(split_dim) {}

  /// The boundary's ``(box, valid)`` on the split axis, or nullopt to decline.
  [[nodiscard]] std::optional<std::pair<int64_t, int64_t>> GetBoundary() const {
    if (blocked_ || !boundary_.has_value()) return std::nullopt;
    return boundary_;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    Consider(As<Call>(op->value_), op->var_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    Consider(As<Call>(op->expr_), nullptr);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void Consider(const CallPtr& call, const VarPtr& result) {
    if (blocked_ || !call || !call->op_) return;

    // The Vector -> Cube direction re-joins the lanes positionally at their own
    // extents, which only abut when the two are equal.
    if (IsOp(call, "tile.aic_gather") || IsOp(call, "tile.tpush_to_aic") ||
        IsOp(call, "tile.tpop_from_aiv") ||
        core_affinity::ClassifyMoveDirection(call) == core_affinity::CVDirection::VECTOR_TO_CUBE) {
      blocked_ = true;
      return;
    }

    // Cube -> Vector boundary: the tile it partitions is the cube-side operand
    // (a move / shard) or the popped result (an already-lowered tpop).
    const bool is_move_boundary =
        core_affinity::ClassifyMoveDirection(call) == core_affinity::CVDirection::CUBE_TO_VECTOR;
    if (is_move_boundary || IsOp(call, "tile.aiv_shard") || IsOp(call, "tile.tpop_from_aic")) {
      auto full_type = IsOp(call, "tile.tpop_from_aic")
                           ? call->GetType()
                           : (call->args_.empty() ? nullptr : call->args_[0]->GetType());
      RecordBoundary(full_type, result);
      return;
    }

    // Everything else that produces a tile split on this axis must derive from
    // the boundary; a fresh one (a load, a generator) spans the full box.
    auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
    if (!tt || split_dim_ >= static_cast<int>(tt->shape_.size())) return;
    if (IsSingletonDim(tt->shape_[split_dim_])) return;
    // Cube-affine ops keep their full width (the affinity gate never halves
    // them), so they neither derive from nor conflict with the partition.
    if (core_affinity::ClassifyCallAffinity(call) == core_affinity::CoreAffinity::CUBE) return;

    // EVERY operand the split axis partitions must come from the boundary, not
    // just one of them: `tile.add(shard, vec_param)` mixes a boundary-derived
    // half with a full-width parameter, and a balanced stride would give the
    // two operands different rows. A singleton operand is replicated across the
    // lanes rather than partitioned, so it neither derives nor conflicts.
    bool partitioned_operand = false;
    for (const auto& arg : call->args_) {
      auto arg_tt = std::dynamic_pointer_cast<const TileType>(arg->GetType());
      if (!arg_tt || split_dim_ >= static_cast<int>(arg_tt->shape_.size())) continue;
      if (IsSingletonDim(arg_tt->shape_[split_dim_])) continue;
      partitioned_operand = true;
      auto var = AsVarLike(arg);
      if (!var || derived_.count(var.get()) == 0) {
        blocked_ = true;
        return;
      }
    }
    // No partitioned operand at all: this op mints its own split-axis tile.
    if (!partitioned_operand) {
      blocked_ = true;
      return;
    }
    if (result) derived_.insert(result.get());
  }

  void RecordBoundary(const TypePtr& full_type, const VarPtr& result) {
    auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
    auto extents = ComputeLaneExtents(tt, split_dim_, /*lane_stride=*/nullptr);
    if (!extents.has_value()) {
      blocked_ = true;
      return;
    }
    auto box = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim_]);
    const std::pair<int64_t, int64_t> shape{box->value_, extents->valid};
    if (boundary_.has_value() && *boundary_ != shape) {
      blocked_ = true;  // two boundaries of different shape: no single partition
      return;
    }
    boundary_ = shape;
    if (result) derived_.insert(result.get());
  }

  int split_dim_;
  bool blocked_ = false;
  std::optional<std::pair<int64_t, int64_t>> boundary_;
  std::unordered_set<const Var*> derived_;
};

}  // namespace

ExprPtr ResolveLaneStride(const std::vector<StmtPtr>& stmts, int split_dim) {
  if (split_dim < 0) return nullptr;
  LaneStrideScanner scanner(split_dim);
  for (const auto& stmt : stmts) {
    scanner.VisitStmt(stmt);
  }
  auto boundary = scanner.GetBoundary();
  if (!boundary.has_value()) return nullptr;
  const auto [box, valid] = *boundary;
  const int64_t balanced = (valid + 1) / 2;
  // Nothing to rebalance when the balanced stride IS the box half — a fully
  // valid boundary, or one short by a single cell. Leaving the stride null
  // there keeps the IR of every such kernel byte-identical to before. An EMPTY
  // boundary (valid == 0) has no partition to balance either, and a zero stride
  // is not a legal one: keep the box partition, where both lanes are empty.
  if (balanced <= 0 || valid >= box || balanced == (box + 1) / 2) return nullptr;
  return std::make_shared<ConstInt>(balanced, DataType::INDEX, Span::unknown());
}

// The CEIL half, so an odd extent 2k+1 gives both lanes the same (k+1)-cell BOX
// and lets the per-lane valid extent carry the raggedness (lane 0 fills k+1,
// lane 1 fills k). That mirrors pto-isa's TILE_UP_DOWN_ODD /
// TILE_LEFT_RIGHT_ODD ("AIV0 = rows/2 + 1, AIV1 = rows/2"), which derives lane
// 1's slot offset from the lanes' RUNTIME valid extents. An even extent is
// unaffected: ceil(2k / 2) == k.
//
// A dynamic extent keeps floordiv(dim, 2) — its parity is unknown at compile
// time, so no odd split code can be stamped for it either (see ShardSplitCode).
ExprPtr ComputeHalfDimSize(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return std::make_shared<ConstInt>((ci->value_ + 1) / 2, ci->dtype(), ci->span_);
  }
  auto two = std::make_shared<ConstInt>(2, GetScalarDtype(dim_size), dim_size->span_);
  return MakeFloorDiv(dim_size, two, dim_size->span_);
}

std::optional<std::pair<int64_t, int64_t>> StaticLaneExtents(const TypePtr& full_type, int split_dim,
                                                             const ExprPtr& lane_stride) {
  auto extents =
      ComputeLaneExtents(std::dynamic_pointer_cast<const TileType>(full_type), split_dim, lane_stride);
  if (!extents.has_value()) return std::nullopt;
  return std::make_pair(extents->lane0, extents->lane1);
}

int ShardSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const ExprPtr& lane_stride,
                   const std::string& op_name, const Span& span) {
  if (mode == SplitMode::None) return kSplitNone;
  auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
  auto extents = ComputeLaneExtents(tt, split_dim, lane_stride);
  // No compile-time lane extents (a runtime box, valid extent or stride): emit
  // the even code, which is what this path emitted before the odd codes existed
  // and what the device is measured to accept.
  //
  // Reading pto-isa's pop suggests otherwise — it derives lane 1's band from the
  // popped tile's runtime valid extent, so a runtime extent that leaves the
  // lanes unequal (15 of a 16-wide axis gives 8 and 7) looks like it would need
  // the odd code. The device says the even code is right: rejecting these
  // boundaries was tried and it took down
  // tests/st/runtime/cross_core/test_cross_core_split_parity.py, whose
  // LEFT_RIGHT cases (lanes 8 / 7 and 8 / 1) pass an elementwise on-device
  // comparison — re-verified by perturbing the golden on lane 1's columns
  // alone, which does fail. Which reading is the contract is asked upstream in
  // hw-native-sys/pto-isa#263; until that answers, the encoding follows the
  // measured behaviour rather than the source reading.
  if (!extents.has_value()) return SplitCodeFor(mode, /*odd_extent=*/false);

  if (extents->lane0 == extents->lane1) return SplitCodeFor(mode, /*odd_extent=*/false);
  if (extents->lane0 == extents->lane1 + 1) return SplitCodeFor(mode, /*odd_extent=*/true);
  // Lane 1 pops nothing, so pto-isa never dereferences its band — the even code
  // is exact whatever lane 0 holds. This is the empty-tail shard (a valid extent
  // that does not reach the second lane at all).
  if (extents->lane1 == 0) return SplitCodeFor(mode, /*odd_extent=*/false);

  // Only the BOX partition can land here: the balanced stride is ceil(V / 2) by
  // construction, so its lanes never differ by more than one. ResolveLaneStride
  // declined to rebalance this body, and the reason is what the fix must target.
  CHECK_SPAN(false, span)
      << op_name << ": the Cube -> Vector boundary tile's valid split-axis extent (" << extents->valid
      << " of a " << PythonPrint(tt->shape_[split_dim]) << "-wide dim " << split_dim
      << ") leaves the two AIV lanes " << extents->lane0 << " and " << extents->lane1
      << " cells. pto-isa places lane 1's band at its own extent (TILE_UP_DOWN / TILE_LEFT_RIGHT) or one "
         "past it (the _ODD modes), so the lanes must be equal, differ by exactly one, or leave lane 1 "
         "empty. The compiler balances a ragged boundary across the lanes automatically, but only when "
         "the whole split body derives from it — this one also holds an independently split value (a "
         "tile.load, a generator, a full-width operand) or a Vector -> Cube gather, which spans the full "
         "box. Either move that value out of the split scope, or widen the valid extent to "
      << (2 * extents->box_half) << " / " << (2 * extents->box_half - 1)
      << " with pl.tile.set_validshape(...), or narrow it to at most " << extents->box_half
      << " so the whole value stays on lane 0.";
  return kSplitNone;  // unreachable: CHECK_SPAN(false, ...) always throws
}

int GatherSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const std::string& op_name,
                    const Span& span) {
  if (mode == SplitMode::None) return kSplitNone;
  auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
  // A gather is never rebalanced (ResolveLaneStride declines a body that holds
  // one), so its lanes always sit on the box partition.
  auto extents = ComputeLaneExtents(tt, split_dim, /*lane_stride=*/nullptr);
  if (extents.has_value() && extents->lane0 != extents->lane1 && extents->lane1 != 0) {
    CHECK_SPAN(false, span)
        << op_name << ": the Vector -> Cube boundary tile's split axis (dim " << split_dim
        << ") leaves the two AIV lanes " << extents->lane0 << " and " << extents->lane1
        << " cells. Each lane hands the cube a band placed at its OWN extent, so the two only abut when "
           "the lanes are equal — the gather has no odd form. Pad that axis to "
        << (2 * extents->box_half) << " and narrow the cube-side result with pl.tile.set_validshape(...).";
  }
  return SplitCodeFor(mode, /*odd_extent=*/false);
}

TypePtr LocalizeShardValidForLane(const TypePtr& shard_type, const TypePtr& operand_type, int split_dim,
                                  const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  auto shard = std::dynamic_pointer_cast<const TileType>(shard_type);
  auto operand = std::dynamic_pointer_cast<const TileType>(operand_type);
  if (!shard || !operand || !subblock_idx || split_dim < 0 ||
      split_dim >= static_cast<int>(shard->shape_.size()) ||
      split_dim >= static_cast<int>(operand->shape_.size())) {
    return shard_type;
  }
  const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
  // A fully-valid EVEN split axis on the box partition needs no repair: both
  // lanes hold `half`, which is exactly what ReshapeSplitAxis already produced.
  // A fully-valid ODD axis still does (its lanes hold ceil and floor), and so
  // does any rebalanced body (its stride is narrower than the box half) — only
  // the lane index, in scope here, can tell the lanes apart.
  if (static_cast<int>(operand_valid.size()) <= split_dim ||
      (!lane_stride && AreExprsEqual(operand_valid[split_dim], operand->shape_[split_dim]) &&
       !IsOddSplitExtent(operand->shape_[split_dim]))) {
    return shard_type;
  }
  auto lane_extent = LocalizeValidDimForSplit(operand_valid[split_dim], operand->shape_[split_dim],
                                              LaneStep(lane_stride, shard->shape_[split_dim]), subblock_idx);
  return WithValidDim(shard, split_dim, lane_extent);
}

namespace {

/// Mutable state threaded through the (recursive) localization walk.
struct LocalizeState {
  std::unordered_map<const Var*, LocalizedTile> tracked;
  std::unordered_map<const Var*, VarPtr> replacements;
  ExprPtr lane_index;                                  ///< the region's aiv_id, resolved lazily
  std::unordered_set<const Var*> chained_store_dests;  ///< region-wide, indexed once
};

std::vector<StmtPtr> LocalizeStmts(const std::vector<StmtPtr>& stmts, int split_dim, const Span& span,
                                   LocalizeState& state);

/// Recurse into a nested body, preserving its statement kind.
StmtPtr LocalizeNestedBody(const StmtPtr& body, int split_dim, const Span& span, LocalizeState& state) {
  if (!body) return body;
  auto inner = LocalizeStmts(transform_utils::FlattenToStmts(body), split_dim, span, state);
  if (inner.size() == 1) return inner[0];
  return std::make_shared<SeqStmts>(inner, body->span_);
}

/// Whether another ``tile.store`` in this body writes THROUGH `var`, i.e. takes
/// it as its destination tensor (arg 2) — a chained store.
///
/// This is the one read that makes the empty-lane guard unsafe, because it
/// orders two stores against each other: skipping the first would leave the
/// second writing through a tensor version that was never produced. Every other
/// read of a store's SSA result is benign. It is a destination-passing alias of
/// a tensor the region does not own, so a phi that carries it out of a branch,
/// or the enclosing function threading it to a return, still denotes the SAME
/// buffer — an empty lane that stored nothing leaves exactly the value any
/// reader must see, and ExpandMixedKernel drops the alias from the AIV half.
/// Indexed ONCE over the whole region, recursively: a chained store nested in a
/// branch or a loop orders against an outer store just as a sibling one does, so
/// a sibling-only scan would miss it and wrongly allow the empty-lane guard.
/// Building the index up front also keeps the walk linear in the region size
/// instead of quadratic in its store count (see `.claude/rules/pass-complexity.md`).
///
/// A store's own destination is an operand, never its SSA result, so a store can
/// never place its own result in this set; no self-exclusion is needed.
void CollectChainedStoreDestinations(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>* out) {
  for (const auto& s : stmts) {
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(for_stmt->body_), out);
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(while_stmt->body_), out);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(if_stmt->then_body_), out);
      if (if_stmt->else_body_.has_value()) {
        CollectChainedStoreDestinations(transform_utils::FlattenToStmts(*if_stmt->else_body_), out);
      }
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
      CollectChainedStoreDestinations(seq->stmts_, out);
      continue;
    }
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(s);
    auto call = assign ? AsCall(assign->value_) : nullptr;
    if (!call || !IsOp(call, "tile.store") || call->args_.size() < 3) continue;
    if (auto dest = AsVarLike(call->args_[2])) out->insert(dest.get());
  }
}

/// A loop carry seeded by a localized (per-lane) tile would need its `IterArg`
/// and the paired return var retyped to the lane's extent. Those are separate
/// SSA definitions that this walk does not rebuild, so the carry would keep the
/// deducer's lane-agnostic type while its init value carries the lane's — the
/// silent-wrong-data shape this pass exists to remove. Refuse it explicitly
/// instead, naming the authoring that avoids it.
void RejectLocalizedCarry(const std::vector<IterArgPtr>& iter_args, const char* loop_form, const Span& span,
                          const LocalizeState& state) {
  for (const auto& iter_arg : iter_args) {
    if (!iter_arg) continue;
    auto init = AsVarLike(iter_arg->initValue_);
    if (!init) continue;
    auto replaced = state.replacements.find(init.get());
    const Var* key = (replaced != state.replacements.end()) ? replaced->second.get() : init.get();
    auto it = state.tracked.find(key);
    if (it == state.tracked.end()) continue;
    CHECK_SPAN(false, span)
        << "pl.split_aiv: carrying a per-lane cross-core value across a " << loop_form
        << " boundary is not supported. '" << iter_arg->name_hint_
        << "' is seeded by a value whose split-axis extent differs per lane ("
        << PythonPrint(it->second.lane_extent)
        << "), but a loop carry is a separate binding that would keep the lane-agnostic extent.\n"
        << "Author one of these instead:\n"
        << "  * keep the shard and everything that consumes it inside the loop body\n"
        << "  * make the split axis fully valid before the crossing (pl.set_validshape to the full "
           "extent) and treat the padding as don't-care\n"
        << "  * store the per-lane shard before the loop and reload it inside";
  }
}

std::vector<StmtPtr> LocalizeStmts(const std::vector<StmtPtr>& stmts, int split_dim, const Span& span,
                                   LocalizeState& state) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    // --- Control flow: the repair must reach as deep as the region does. ------
    // A shard, a consumer or a store nested in a loop or a branch is ordinary
    // authoring (a per-lane tail stored under `if`, a shard inside `pl.range`),
    // and a flat walk would leave it on the deducer's lane-agnostic ceil(V/2).
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto new_body = LocalizeNestedBody(for_stmt->body_, split_dim, span, state);
      RejectLocalizedCarry(for_stmt->iter_args_, "pl.range", for_stmt->span_, state);
      result.push_back(new_body == for_stmt->body_
                           ? stmt
                           : loop_repair::RebuildForStmt(for_stmt, for_stmt->iter_args_, new_body,
                                                         for_stmt->return_vars_));
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto new_body = LocalizeNestedBody(while_stmt->body_, split_dim, span, state);
      RejectLocalizedCarry(while_stmt->iter_args_, "pl.while_", while_stmt->span_, state);
      if (new_body == while_stmt->body_) {
        result.push_back(stmt);
      } else {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = new_body;
        result.push_back(new_while);
      }
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = LocalizeNestedBody(if_stmt->then_body_, split_dim, span, state);
      std::optional<StmtPtr> new_else = if_stmt->else_body_;
      if (if_stmt->else_body_.has_value()) {
        new_else = LocalizeNestedBody(*if_stmt->else_body_, split_dim, span, state);
      }
      result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, new_then, new_else,
                                                if_stmt->return_vars_, if_stmt->span_));
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      for (auto& s : LocalizeStmts(seq->stmts_, split_dim, span, state)) result.push_back(s);
      continue;
    }

    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto call = assign ? AsCall(assign->value_) : nullptr;
    if (!assign || !call || !call->op_) {
      result.push_back(stmt);
      continue;
    }

    if (IsOp(call, "tile.get_subblock_idx")) {
      state.lane_index = assign->var_;
      result.push_back(stmt);
      continue;
    }

    // --- The boundary itself: repair the deducer's ceil-div guess. -----------
    if (IsOp(call, "tile.aiv_shard") && !call->args_.empty()) {
      auto operand = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
      auto shard = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (!operand || !shard || split_dim < 0 || split_dim >= static_cast<int>(shard->shape_.size())) {
        result.push_back(stmt);
        continue;
      }
      const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
      // A fully-valid EVEN split axis needs no repair: both lanes hold `half`,
      // which is exactly what ReshapeSplitAxis already produced. A fully-valid
      // ODD one does: its lanes hold ceil and floor, the transport picks the odd
      // code for exactly that reason, and pto-isa then reads the per-lane extents
      // off the popped tile. Leaving both lanes at the ceil guess would place
      // lane 1's band one cell too far.
      if (static_cast<int>(operand_valid.size()) <= split_dim ||
          (AreExprsEqual(operand_valid[split_dim], operand->shape_[split_dim]) &&
           !IsOddSplitExtent(operand->shape_[split_dim]))) {
        result.push_back(stmt);
        continue;
      }
      INTERNAL_CHECK_SPAN(state.lane_index, call->span_)
          << "Internal error: a pl.split_aiv region with a partially-valid split axis has no "
             "tile.get_subblock_idx binding, so the per-lane extent cannot be materialized";
      auto new_type = LocalizeShardValidForLane(shard, operand, split_dim, state.lane_index);
      auto lane_extent =
          tile_view_semantics::GetEffectiveTileView(*std::dynamic_pointer_cast<const TileType>(new_type))
              .valid_shape[split_dim];
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_type, call->span_);
      state.replacements[assign->var_.get()] = new_var;
      state.tracked[new_var.get()] = LocalizedTile{lane_extent, operand_valid[split_dim], shard};
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      continue;
    }

    // --- Consumers: carry the per-lane extent, or say why we cannot. ---------
    const LocalizedTile* source = nullptr;
    for (const auto& arg : call->args_) {
      auto arg_var = AsVarLike(arg);
      if (!arg_var) continue;
      auto replaced = state.replacements.find(arg_var.get());
      const Var* key = (replaced != state.replacements.end()) ? replaced->second.get() : arg_var.get();
      auto it = state.tracked.find(key);
      if (it != state.tracked.end()) {
        source = &it->second;
        break;
      }
    }
    // tile.aic_gather is the region's EXIT: it re-joins the two lanes' bands and
    // hands a FULL tile back to the cube. This is the one place the join can be
    // typed correctly, because it is the only place the lanes' TRUE extents are
    // known.
    //
    // Under the per-lane clamp the bands always abut, so the joined extent is
    // exactly the pre-shard V:
    //
    //   V >  half : lane 0 saturates to half, lane 1 holds V - half
    //               -> [0, half) U [half, V) = [0, V)
    //   V <= half : lane 0 holds V, lane 1 is empty
    //               -> [0, V) U {} = [0, V)
    //
    // The deducer cannot compute that. It runs before the lanes exist and can
    // only double its own lane-agnostic guess (2 * ceil(V / 2)), which claims
    // the hole between the bands as real data. So restore V here.
    if (IsOp(call, "tile.aic_gather")) {
      auto gathered = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (source != nullptr && gathered && split_dim < static_cast<int>(gathered->shape_.size())) {
        auto joined_type = WithValidDim(gathered, split_dim, source->joined_extent);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, joined_type, assign->var_->span_);
        auto new_call =
            std::make_shared<Call>(call->op_, call->args_, call->kwargs_, joined_type, call->span_);
        state.replacements[assign->var_.get()] = new_var;
        // The result is a FULL tile again, so it leaves the per-lane dataflow.
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        continue;
      }
      // Nothing localized this operand, so a provable partial is one the author
      // gave BOTH lanes: the bands do not abut and no valid_shape describes the
      // union. This is the check the deducer used to make on a value it could
      // not see the truth of.
      auto operand = call->args_.empty()
                         ? nullptr
                         : std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
      if (operand && split_dim < static_cast<int>(operand->shape_.size())) {
        const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
        auto valid_const = split_dim < static_cast<int>(operand_valid.size())
                               ? std::dynamic_pointer_cast<const ConstInt>(operand_valid[split_dim])
                               : nullptr;
        auto half_const = std::dynamic_pointer_cast<const ConstInt>(operand->shape_[split_dim]);
        const bool provably_partial = valid_const && half_const && valid_const->value_ < half_const->value_;
        const char* axis_name = (split_dim == 0) ? "row" : "column";
        CHECK_SPAN(!provably_partial, call->span_)
            << "pl.split_aiv: tile.aic_gather re-joins the two lanes positionally, but each lane's "
               "valid "
            << axis_name << " extent (" << valid_const->value_ << " of " << half_const->value_
            << ") covers only part of its half, so the lanes contribute the disjoint bands [0, "
            << valid_const->value_ << ") and [" << half_const->value_ << ", "
            << (half_const->value_ + valid_const->value_)
            << ") of the re-joined tile. No valid_shape describes that union.\n"
            << "Author one of these instead:\n"
            << "  * make each lane's half fully valid before the gather -- pl.fillpad(v) to fill the "
               "padding with data, or pl.set_validshape(v, <full half extent>, ...) to declare it "
               "data -- so the bands abut\n"
            << "  * let the shard's own per-lane extent flow into the gather instead of overriding "
               "it: a shard-then-gather round trip re-joins exactly the pre-shard extent";
      }
      result.push_back(stmt);
      continue;
    }
    if (source == nullptr) {
      result.push_back(stmt);
      continue;
    }

    auto consumer_result = std::dynamic_pointer_cast<const TileType>(call->GetType());
    if (!consumer_result) {
      // Not a tile result (tile.store returns a Tensor, system.tfree returns
      // nothing): the op reads the extent but does not carry it onward, which is
      // exactly where the per-lane extent is meant to land.
      //
      // The store is the one consumer that must not see an EMPTY lane. When the
      // ragged extent does not reach the second lane (V <= half) that lane's
      // extent is 0, and a zero-row store is outside pto-isa's contract:
      // TSTORE_IMPL asserts ``src.GetValidRow() > 0 && src.GetValidCol() > 0``
      // (npu/a2a3/TStore.hpp) and PTOAS's PartitionViewOp rejects a statically
      // non-positive size. A release build compiles the assert out and the DMA
      // moves nothing, but a debug / CPU-sim / CA-model build traps. So guard
      // the store on a non-empty lane. The tpop and the tfree stay
      // UNCONDITIONAL: both lanes must still pop and free the slot or the pipe
      // desynchronizes.
      if (IsOp(call, "tile.store") && !IsProvablyPositive(source->lane_extent)) {
        // The guard can be a plain IfStmt because a store's SSA result is a
        // destination-passing alias (see CollectChainedStoreDestinations). Only a
        // CHAINED store makes that unsafe — skipping the first store would leave
        // the second writing through a tensor version that was never produced —
        // and guarding it would need a return-carrying if (a phi over stored /
        // not-stored). Report that shape instead of emitting an unguarded
        // zero-row store.
        CHECK_SPAN(state.chained_store_dests.count(assign->var_.get()) == 0, call->span_)
            << "pl.split_aiv: a store whose result feeds another store cannot be guarded against an "
               "empty lane. The split axis is only partially valid, so one lane's extent can be 0 at "
               "runtime ("
            << PythonPrint(source->lane_extent) << "), and a zero-row store is outside the ISA contract.\n"
            << "Author one of these instead:\n"
            << "  * write the per-lane shard to its destination with a single subscript assignment "
               "inside the region\n"
            << "  * make the split axis fully valid before the crossing (pl.set_validshape to the "
               "full extent) so every lane is non-empty";
        auto zero = std::make_shared<ConstInt>(0, GetScalarDtype(source->lane_extent), call->span_);
        auto non_empty = MakeGt(source->lane_extent, zero, call->span_);
        result.push_back(
            std::make_shared<IfStmt>(non_empty, stmt, std::nullopt, std::vector<VarPtr>{}, call->span_));
        continue;
      }
      result.push_back(stmt);
      continue;
    }

    // A consumer that WIDENS the logical region back to the whole physical box
    // (a set_validshape to the full extent) deliberately drops the per-lane
    // extent — the author is declaring the padding to be data. Nothing to carry.
    if (tile_view_semantics::ShapeExprListsEquivalent(
            tile_view_semantics::GetEffectiveTileView(*consumer_result).valid_shape,
            consumer_result->shape_)) {
      result.push_back(stmt);
      continue;
    }
    CHECK_SPAN(PassesThroughValidShape(source->before, consumer_result), call->span_)
        << "pl.split_aiv: '" << call->op_->name_
        << "' reshapes the logical valid region of a per-lane cross-core value, which is not "
           "supported. The shard's split-axis extent differs per lane ("
        << PythonPrint(source->lane_extent)
        << "), and that extent can only be carried by ops that pass the valid_shape through "
           "unchanged — the boundary offers no way to re-narrow a popped tile afterwards.\n"
        << "Author one of these instead:\n"
        << "  * do the reshaping compute on the CUBE side, before pl.aiv_shard\n"
        << "  * make the split axis fully valid before the crossing "
           "(pl.set_validshape to the full extent) and treat the padding as don't-care\n"
        << "  * store the per-lane shard first, and reshape in a later kernel";
    auto new_type = WithValidDim(consumer_result, split_dim, source->lane_extent);
    auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_type, call->span_);
    state.replacements[assign->var_.get()] = new_var;
    state.tracked[new_var.get()] = LocalizedTile{source->lane_extent, source->joined_extent, consumer_result};
    result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
  }

  return result;
}

}  // namespace

std::vector<StmtPtr> LocalizeExplicitBoundaryValid(const std::vector<StmtPtr>& stmts, int split_dim,
                                                   const Span& region_span) {
  LocalizeState state;
  CollectChainedStoreDestinations(stmts, &state.chained_store_dests);
  auto result = LocalizeStmts(stmts, split_dim, region_span, state);

  if (state.replacements.empty()) {
    return result;
  }
  // One final substitution re-points every downstream use (including the ones
  // inside nested control flow) at the retyped vars, mirroring how the AUTO
  // halving path finishes in ProcessStmts.
  StmtPtr body = (result.size() == 1) ? result[0] : std::make_shared<SeqStmts>(result, region_span);
  return transform_utils::FlattenToStmts(transform_utils::Substitute(body, state.replacements));
}

namespace {

// Mirrors the (formerly file-local) hazard finder in ExpandMixedKernel: records
// the first tile.transpose whose source carries the split axis and whose
// transpose actually swaps it. Shared so the explicit per-region check in pass 20
// and the AUTO whole-function check in pass 21 use one detector.
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
