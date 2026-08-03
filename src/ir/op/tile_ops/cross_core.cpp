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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

// Read the required "split" int attr shared by the split-axis reshape ops
// (reuses the tpush/tpop encoding: 1 = UP_DOWN/axis0, 2 = LEFT_RIGHT/axis1).
int ReadSplitAttr(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
                  const Span& span) {
  std::optional<int> split_opt;
  for (const auto& [key, value] : kwargs) {
    if (key == "split") {
      split_opt = AnyCast<int>(value, "kwarg key: split");
      break;
    }
  }
  CHECK_SPAN(split_opt.has_value(), span)
      << op_name << " requires a 'split' attr (1 = UP_DOWN/axis0, 2 = LEFT_RIGHT/axis1)";
  const int split = *split_opt;
  CHECK_SPAN(split == 1 || split == 2, span)
      << op_name << " split must be 1 (UP_DOWN/axis0) or 2 (LEFT_RIGHT/axis1), but got " << split;
  return split;
}

// Shared split-axis reshape core for both the tile ops (tile.aiv_shard /
// tile.aic_gather) and the tensor ops (tensor.aiv_shard / tensor.aic_gather).
// Halves (shard, `halve` = true) or doubles (gather, `halve` = false) the
// split-axis extent of `shape` and `valid`.
//
// Static (ConstInt) extents are halved/doubled directly; for the halving
// direction a static split-axis extent must be even. Dynamic (non-ConstInt)
// extents are reshaped symbolically (floordiv(dim, 2) on shard, dim * 2 on
// gather) so the result type reflects the shard/gather along the split axis
// rather than an identity reshape.
//
// The even-extent requirement applies to the PHYSICAL split-axis extent only;
// the per-lane valid_shape is reshaped with ceil-div on halve (floordiv(dim + 1,
// 2), keeping valid <= physical) since the true per-lane valid region is
// localized later at lowering time, which knows the subblock (lane) index. This
// avoids rejecting an input whose physical extent is even but whose partial
// valid_shape happens to be odd.
struct SplitReshaped {
  std::vector<ExprPtr> shape;
  std::vector<ExprPtr> valid;
};

SplitReshaped ReshapeSplitAxis(std::vector<ExprPtr> shape, std::vector<ExprPtr> valid, size_t axis,
                               bool halve, const std::string& op_name, const Span& span) {
  if (auto c = As<ConstInt>(shape[axis])) {
    if (halve) {
      CHECK_SPAN(c->value_ % 2 == 0, span)
          << op_name << ": split-axis static extent " << c->value_ << " must be even to shard in half";
      shape[axis] = std::make_shared<ConstInt>(c->value_ / 2, c->dtype(), shape[axis]->span_);
    } else {
      shape[axis] = std::make_shared<ConstInt>(c->value_ * 2, c->dtype(), shape[axis]->span_);
    }
  } else {
    // Dynamic split-axis extent: symbolic half / double. Per-lane evenness is
    // resolved at lowering time, which knows the subblock index.
    auto two = std::make_shared<ConstInt>(2, GetScalarDtype(shape[axis]), shape[axis]->span_);
    shape[axis] = halve ? MakeFloorDiv(shape[axis], two, shape[axis]->span_)
                        : MakeMul(shape[axis], two, shape[axis]->span_);
  }
  if (axis < valid.size()) {
    if (auto vc = As<ConstInt>(valid[axis])) {
      const auto new_extent = halve ? (vc->value_ + 1) / 2 : vc->value_ * 2;
      valid[axis] = std::make_shared<ConstInt>(new_extent, vc->dtype(), valid[axis]->span_);
    } else {
      // Dynamic valid extent: ceil-div on halve (floordiv(dim + 1, 2)), double on
      // gather — mirroring the physical reshape; the exact per-lane valid region
      // is re-derived at lowering time.
      auto vspan = valid[axis]->span_;
      auto dt = GetScalarDtype(valid[axis]);
      auto two = std::make_shared<ConstInt>(2, dt, vspan);
      if (halve) {
        auto one = std::make_shared<ConstInt>(1, dt, vspan);
        valid[axis] = MakeFloorDiv(MakeAdd(valid[axis], one, vspan), two, vspan);
      } else {
        valid[axis] = MakeMul(valid[axis], two, vspan);
      }
    }
  }
  return {std::move(shape), std::move(valid)};
}

// Deducer for the tile-level split-axis reshape ops tile.aiv_shard (full ->
// half) and tile.aic_gather (half -> full). The single positional tile argument
// is reshaped along the split axis selected by the "split" int attr.
//
// 2D-vocab constraint: the input must be rank-2 and the split attr must be 1 or 2.
TypePtr DeduceSplitReshape(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name, bool halve) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 tile argument, but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const int split = ReadSplitAttr(kwargs, op_name, args[0]->span_);
  CHECK_SPAN(tile_type->shape_.size() == 2, args[0]->span_)
      << op_name << " requires a 2D tile, but got rank " << tile_type->shape_.size();

  const size_t axis = (split == 1) ? 0 : 1;
  auto reshaped =
      ReshapeSplitAxis(tile_type->shape_, GetValidShape(tile_type), axis, halve, op_name, args[0]->span_);

  // The result is a fresh per-lane (shard) / re-joined (gather) tile along the
  // split axis. Only the halved/doubled valid_shape is carried; the source's
  // explicit blayout/slayout is intentionally NOT inherited. Inheriting a
  // non-implicit layout (e.g. an Acc operand's col_major) makes the result type
  // diverge from the deduction fixpoint that downstream elementwise consumers
  // (which re-derive layout from their inputs) and a print->parse round-trip
  // reconstruct — the boundary's true memory layout is re-attached by the
  // lowering pass (ReshapeTypeWithMemory) and normalized downstream.
  TileView tile_view;
  tile_view.valid_shape = std::move(reshaped.valid);
  return std::make_shared<TileType>(std::move(reshaped.shape), tile_type->dtype_, std::nullopt,
                                    std::move(tile_view));
}

// Tensor-level counterpart of DeduceSplitReshape for tensor.aiv_shard /
// tensor.aic_gather — the @pl.jit / pl.spmd author-facing form, where producers
// (pl.matmul, elementwise) return Tensor. Mirrors the tile deducer exactly but
// over a TensorType, and enforces rank-2: UP_DOWN / LEFT_RIGHT are only
// well-defined on the 2D physical tile view. An N-D tensor flattens to
// [product(leading), last] (FlattenTileNdTo2D), so a pre-flatten row-axis split
// would not match the contiguous half the lowering physically takes — reject
// with a reshape hint rather than silently miscompiling.
TypePtr DeduceSplitReshapeTensor(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name, bool halve) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 tensor argument, but got "
                          << args.size();

  // Exact TensorType match: rejects TileType (the tile op's domain) AND
  // DistributedTensorType (out of scope for AIV/AIC split).
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name
                     << " requires argument to be a (non-distributed) TensorType, but got "
                     << args[0]->GetType()->TypeName();

  const int split = ReadSplitAttr(kwargs, op_name, args[0]->span_);
  CHECK_SPAN(tensor_type->shape_.size() == 2, args[0]->span_)
      << op_name << " requires a 2D tensor, but got rank " << tensor_type->shape_.size()
      << ". Reshape the operand to 2D (pl.reshape) before the shard / gather so the "
         "UP_DOWN / LEFT_RIGHT split axis is unambiguous.";

  const size_t axis = (split == 1) ? 0 : 1;

  // Valid shape: TensorView::valid_shape if set, otherwise the static shape
  // (mirrors GetValidShape for tiles).
  std::vector<ExprPtr> valid = (tensor_type->tensor_view_ && !tensor_type->tensor_view_->valid_shape.empty())
                                   ? tensor_type->tensor_view_->valid_shape
                                   : tensor_type->shape_;
  auto reshaped =
      ReshapeSplitAxis(tensor_type->shape_, std::move(valid), axis, halve, op_name, args[0]->span_);

  // Fresh per-lane (shard) / re-joined (gather) tensor along the split axis; only
  // the halved/doubled valid_shape is carried (no layout inheritance — same
  // rationale as the tile deducer). Memory space is a tile-level concept and is
  // re-attached when ConvertTensorToTileOps lowers this to tile.aiv_shard.
  //
  // Canonicalize a redundant view away, mirroring the tile path: TileType's
  // constructor drops a tile_view whose valid_shape matches the shape (the
  // implicit view), but TensorType performs no such canonicalization. So only
  // attach a tensor_view when the reshaped valid_shape is a genuine partial
  // (differs from the reshaped shape). A redundant valid_shape == shape view
  // otherwise breaks the print -> parse round-trip: the printer collapses it to
  // a bare ``pl.TensorView()`` presence marker that reparses to an empty
  // valid_shape (structurally != the shape-sized valid_shape).
  if (tile_view_semantics::ShapeExprListsEquivalent(reshaped.valid, reshaped.shape)) {
    return std::make_shared<TensorType>(std::move(reshaped.shape), tensor_type->dtype_, std::nullopt);
  }
  TensorView tensor_view({}, TensorLayout::ND, std::move(reshaped.valid));
  return std::make_shared<TensorType>(std::move(reshaped.shape), tensor_type->dtype_, std::nullopt,
                                      std::make_optional(std::move(tensor_view)));
}

}  // namespace

// ============================================================================
// Cross-Core Tile Transfer Operations (tpush / tpop)
// ============================================================================

// Push tile data to AIV (from AIC)
REGISTER_OP("tile.tpush_to_aiv")
    .set_description("Push tile data from AIC to AIV via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Push tile data to AIC (from AIV)
REGISTER_OP("tile.tpush_to_aic")
    .set_description("Push tile data from AIV to AIC via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIC (into AIV)
REGISTER_OP("tile.tpop_from_aic")
    .set_description("Pop tile data from AIC cross-core pipe into AIV")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIV (into AIC)
REGISTER_OP("tile.tpop_from_aiv")
    .set_description("Pop tile data from AIV cross-core pipe into AIC")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// ============================================================================
// Split-axis reshape ops (aiv_shard / aic_gather)
// ============================================================================

// Boundary memory contract (see AivSplitValidPropertyVerifier check (d)).
//
// Both ops ARE the cross-core transfer, so the value has two memory spaces: one
// on the producing lane and one on the consuming lane. The declared type
// describes the CONSUMING side — that is the lane the result Var is read on, and
// it is what ExpandMixedKernel materializes as the boundary tpop:
//
//   tile.aiv_shard : Acc (cube produces into L0C) -> Vec (AIV pops into UB)
//   tile.aic_gather: Vec (vector produces into UB) -> Mat (AIC pops into L1)
//
// The operand side is NOT declared via set_input_memory: a violated input
// constraint makes InferTileMemorySpace *insert a tile.move* to the required
// space (infer_tile_memory_space_pass.cpp MoveCollector), which for a Vec
// operand would synthesize a physically impossible UB -> L0C move instead of
// reporting the authoring error. The operand contract is enforced by the
// AivSplitValid verifier, which reports it as a user diagnostic with a span.
//
// Acc (not Mat) is the cube side of aiv_shard: the shard's operand is pushed
// across the c2v pipe, and only an L0C tile is a supported tpush producer — a
// Mat/L1 tile is rejected by ptoas ("'pto.tpush' op tile type must map to a
// supported producer pipe"). aic_gather is the mirror image but its cube side is
// the tpop DESTINATION, which is Mat (GetBoundaryTpopMemory(CoreSide::AIC)).

// Shard a full tile into half along the split axis (cube -> vector vocabulary).
REGISTER_OP("tile.aiv_shard")
    .set_op_category("CrossCoreOp")
    .set_description("Shard a 2D tile into half along the split axis (full -> half)")
    .add_argument("tile", "Tile data to shard (TileType, 2D)")
    .set_attr<int>("split")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshape(args, kwargs, "tile.aiv_shard", /*halve=*/true);
    });

// Gather two half tiles back into a full tile along the split axis (inverse of aiv_shard).
REGISTER_OP("tile.aic_gather")
    .set_op_category("CrossCoreOp")
    .set_description("Gather a 2D tile into full along the split axis (half -> full)")
    .add_argument("tile", "Tile data to gather (TileType, 2D)")
    .set_attr<int>("split")
    .set_output_memory(MemorySpace::Mat)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshape(args, kwargs, "tile.aic_gather", /*halve=*/false);
    });

// ============================================================================
// Tensor-level split-axis reshape ops (tensor.aiv_shard / tensor.aic_gather)
// ============================================================================
// High-level (@pl.jit / pl.spmd) author-facing form: producers such as
// pl.matmul and elementwise ops return Tensor, so an explicit shard / gather
// inside a ``for aiv_id in pl.split_aiv(...)`` region takes a TensorType. These
// are lowered 1:1 to tile.aiv_shard / tile.aic_gather in ConvertTensorToTileOps
// (pass 10), where the boundary memory space is re-attached.

// Shard a full 2D tensor into half along the split axis (cube -> vector vocabulary).
REGISTER_OP("tensor.aiv_shard")
    .set_op_category("CrossCoreOp")
    .set_description("Shard a 2D tensor into half along the split axis (full -> half)")
    .add_argument("tensor", "Tensor data to shard (TensorType, 2D)")
    .set_attr<int>("split")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshapeTensor(args, kwargs, "tensor.aiv_shard", /*halve=*/true);
    });

// Gather a half 2D tensor back into full along the split axis (inverse of aiv_shard).
REGISTER_OP("tensor.aic_gather")
    .set_op_category("CrossCoreOp")
    .set_description("Gather a 2D tensor into full along the split axis (half -> full)")
    .add_argument("tensor", "Tensor data to gather (TensorType, 2D)")
    .set_attr<int>("split")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshapeTensor(args, kwargs, "tensor.aic_gather", /*halve=*/false);
    });

}  // namespace ir
}  // namespace pypto
