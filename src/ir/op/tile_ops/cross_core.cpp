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

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/printer.h"
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
// (0 = NONE/no split axis, 1 = UP_DOWN/axis0, 2 = LEFT_RIGHT/axis1).
//
// These two ops carry the AUTHORED MODE, not the pto-isa split code: the odd
// codes (kSplitUpDownOdd / kSplitLeftRightOdd, see include/pypto/ir/stmt.h)
// describe how the two lanes' RUNTIME extents relate, which is a property of
// the transport rather than of the author's choice of axis. ExpandMixedKernel
// derives the code from this mode plus the boundary tile's extents when it
// mints the tpush / tpop pair (split_axis::ShardSplitCode).
//
// 0 is the task-parallel (``mode=pl.SplitMode.NONE``) region: both AIV lanes run
// the full body, so there is no axis to halve — the op still marks the AIC/AIV
// boundary crossing, but shape-preservingly. See the deducers below.
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
      << op_name << " requires a 'split' attr (0 = NONE/no split axis, 1 = UP_DOWN/axis0, "
      << "2 = LEFT_RIGHT/axis1)";
  const int split = *split_opt;
  CHECK_SPAN(split == kSplitNone || split == kSplitUpDown || split == kSplitLeftRight, span)
      << op_name << " split must be 0 (NONE/no split axis), 1 (UP_DOWN/axis0) or 2 (LEFT_RIGHT/axis1), "
      << "but got " << split;
  return split;
}

// The optional "lane_stride" attr: how far apart the two AIV lanes' data sits on
// the split axis. Absent (the common case) means the default box partition,
// where the stride is the tile's own physical half. LowerAutoVectorSplit stamps
// it when it rebalances a ragged boundary across the lanes, and ExpandMixedKernel
// reads it back to pick the transport's pto-isa split code — so it is the
// compiler's own bookkeeping, never something an author writes.
void CheckLaneStrideAttr(const std::vector<std::pair<std::string, std::any>>& kwargs,
                         const ExprPtr& split_axis_extent, const std::string& op_name, const Span& span) {
  for (const auto& [key, value] : kwargs) {
    if (key != "lane_stride") continue;
    const int stride = AnyCast<int>(value, "kwarg key: lane_stride");
    CHECK_SPAN(stride > 0, span) << op_name << ": 'lane_stride' must be a positive partition stride, but got "
                                 << stride;
    if (auto extent = As<ConstInt>(split_axis_extent)) {
      const int64_t box_half = (extent->value_ + 1) / 2;
      CHECK_SPAN(stride <= box_half, span)
          << op_name << ": 'lane_stride' " << stride << " exceeds the per-lane physical half " << box_half
          << " of a " << extent->value_ << "-wide split axis";
    }
    return;
  }
}

// Shared split-axis reshape core for both the tile ops (tile.aiv_shard /
// tile.aic_gather) and the tensor ops (tensor.aiv_shard / tensor.aic_gather).
// Halves (shard, `halve` = true) or doubles (gather, `halve` = false) the
// split-axis extent of `shape` and `valid`.
//
// The physical half is the CEIL half, which is what makes an ODD split axis
// representable: 2k+1 gives BOTH lanes a (k+1)-cell box, and the raggedness is
// carried by the per-lane valid extent that LowerAutoVectorSplit materializes
// once the lane index is in scope (lane 0 fills k+1, lane 1 fills k). That is
// pto-isa's TILE_UP_DOWN_ODD / TILE_LEFT_RIGHT_ODD contract, which
// ExpandMixedKernel selects when it mints the transport ops
// (split_axis::ShardSplitCode). An even extent is unaffected: ceil(2k/2) == k.
//
// Dynamic (non-ConstInt) extents are reshaped symbolically (floordiv(dim, 2) on
// shard, dim * 2 on gather) so the result type reflects the shard/gather along
// the split axis rather than an identity reshape.
//
// The per-lane valid_shape is reshaped with ceil-div on halve (floordiv(dim + 1,
// 2), keeping valid <= physical), since the true per-lane valid region is
// localized later at lowering time.
struct SplitReshaped {
  std::vector<ExprPtr> shape;
  std::vector<ExprPtr> valid;
};

SplitReshaped ReshapeSplitAxis(std::vector<ExprPtr> shape, std::vector<ExprPtr> valid, size_t axis,
                               bool halve, const std::string& op_name, const Span& span) {
  if (auto c = As<ConstInt>(shape[axis])) {
    // Ceil half: exact for an even extent, lane 1's spare cell for an odd one.
    const int64_t reshaped = halve ? (c->value_ + 1) / 2 : c->value_ * 2;
    shape[axis] = std::make_shared<ConstInt>(reshaped, c->dtype(), shape[axis]->span_);
  } else {
    // Dynamic split-axis extent: symbolic half / double. The per-lane extents
    // are resolved at lowering time, which knows the subblock index.
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

// ============================================================================
// Which partial valid_shapes the AIC/AIV boundary can actually carry
// ============================================================================
//
// THE SHARD (Cube -> Vector). The C2V FIFO transports a physically box-strided
// slot. pto-isa builds the GM slot view from the popped tile's COMPILE-TIME
// rows/cols and the producer's box row pitch, then strides that view with the
// tile's RUNTIME validCol (a2a3 tload_common.hpp TLoadGm2ubNd2nd:
// lenBurst = validCol, but gmGap = gStride3 - gShape4). Requiring
// "GM advance per burst == one producer-box row" gives
//
//     validCol * s + (ProdN - ConsN) * s == ProdN * s   =>   validCol == ConsN
//
// where ProdN cancels -- so the requirement is the SAME for UP_DOWN and
// LEFT_RIGHT. The row field is free in both: the DMA always moves
// nBurst == gShape3 == the physical row count, whatever validRow says.
//
// This derivation is the Vec/UB pop's, and does NOT carry over to the gather:
// a V2C pop lands in an NZ Mat tile through TLoadGm2L1Nd2nz, which takes its
// nValue/dValue from the GM tensor shape and reads neither validRow nor
// validCol. The gather is constrained by geometry instead (see below).
//
// That leaves exactly one carrier for a narrowed COLUMN extent: transport the
// full box and rebuild the logical extents afterwards with pto.treshape
// (MakeTpopCodegenPTO in src/backend/common/pto_ops_crosscore.cpp). treshape
// takes no operands, so it can only restore COMPILE-TIME STATIC extents, and it
// rebuilds BOTH axes from one target type. Hence, per axis of the popped tile:
//
//   column full              -> rides the transport unchanged; the row field is
//                               then free (static OR per-lane)
//   column static narrowing  -> full-box transport + treshape, but only if the
//                               row extent is static too
//   column per-lane          -> no carrier  (LEFT_RIGHT narrows the split axis)
//   column runtime-valued    -> no carrier
//
// A split-axis narrowing is what makes an extent per-lane: lane L holds
// clamp(V - L*half, 0, half), so the two lanes differ whenever V < the physical
// extent. On the shard (AIC -> AIV) that per-lane value is legal on the row
// field and is materialized by LowerAutoVectorSplit.
//
// THE GATHER (Vector -> Cube) is limited by geometry, not by the DMA. Every V2C
// transport places lane l at offset l*half, never compacted, so the gathered
// real data is [0, v0) union [half, half + v1) -- a rectangle only when the two
// bands abut (v0 == half) or lane 1 is empty (v1 == 0). Under the correct
// per-lane extents that always holds and the gathered extent is v0 + v1 == V,
// which is exactly what a shard-then-gather round trip must report. What has no
// rectangle is a HAND-AUTHORED partial that both lanes share (v < half on each):
// the deducer's 2*v would then claim the hole between the bands as real data.
// Only that provable case is rejected here; a lane-dependent extent is the
// compiler's own localized clamp and is left alone.
//
// Docs: docs/en/dev/codegen/00-pto_codegen.md,
//       docs/en/dev/passes/21-lower_auto_vector_split.md
bool IsFullExtent(const ExprPtr& valid_dim, const ExprPtr& dim) {
  return ProveValidExtentEqual(valid_dim, dim) == ProofResult::kTrue;
}

std::string DescribeExtent(const ExprPtr& valid_dim, const ExprPtr& dim) {
  return PythonPrint(valid_dim) + " of " + PythonPrint(dim);
}

// Deducer for the tile-level split-axis reshape ops tile.aiv_shard (full ->
// half) and tile.aic_gather (half -> full). The single positional tile argument
// is reshaped along the split axis selected by the "split" int attr.
//
// SHAPE-PRESERVING AT split=0. A task-parallel (``mode=pl.SplitMode.NONE``)
// region has no split axis: both AIV lanes run the full body, so nothing is
// halved and nothing is re-joined. There the op still means "this value crosses
// the AIC/AIV boundary" — that is the whole of its meaning in manual mode — and
// the crossing does not change the value's shape. So split=0 reshapes nothing
// and the rank-2 requirement is dropped with it: rank 2 exists only to make
// UP_DOWN / LEFT_RIGHT unambiguous, and neither applies.
TypePtr DeduceSplitReshape(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name, bool halve) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 tile argument, but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const int split = ReadSplitAttr(kwargs, op_name, args[0]->span_);
  if (split == 0) {
    // No split axis: preserve shape and valid_shape exactly. The type is still
    // rebuilt (rather than returned as-is) so the boundary result keeps the same
    // "fresh tile, no inherited layout / memref" shape the halving path produces
    // — the memory space comes from set_output_memory, and the layout is
    // re-attached downstream.
    TileView no_split_view;
    no_split_view.valid_shape = GetValidShape(tile_type);
    // No split axis, but the column field is still pinned by the FIFO transport.
    CheckSplitBoundaryCarriesValid(op_name, tile_type->shape_, no_split_view.valid_shape,
                                   /*split_axis=*/-1, halve, args[0]->span_);
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt,
                                      std::move(no_split_view));
  }

  CHECK_SPAN(tile_type->shape_.size() == 2, args[0]->span_)
      << op_name << " requires a 2D tile, but got rank " << tile_type->shape_.size();

  const size_t axis = static_cast<size_t>(SplitAxisFromSplitCode(split));
  CheckLaneStrideAttr(kwargs, tile_type->shape_[axis], op_name, args[0]->span_);
  CheckSplitBoundaryCarriesValid(op_name, tile_type->shape_, GetValidShape(tile_type), static_cast<int>(axis),
                                 halve, args[0]->span_);
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
  if (split == 0) {
    // Task-parallel (NONE) region: the op marks the crossing and preserves the
    // shape (see DeduceSplitReshape). Return the operand's type unchanged — its
    // view is already canonical, so re-wrapping it could only break the
    // print -> parse round-trip the halving path has to work around below.
    return args[0]->GetType();
  }

  CHECK_SPAN(tensor_type->shape_.size() == 2, args[0]->span_)
      << op_name << " requires a 2D tensor, but got rank " << tensor_type->shape_.size()
      << ". Reshape the operand to 2D (pl.reshape) before the shard / gather so the "
         "UP_DOWN / LEFT_RIGHT split axis is unambiguous.";

  const size_t axis = static_cast<size_t>(SplitAxisFromSplitCode(split));

  // Valid shape: TensorView::valid_shape if set, otherwise the static shape
  // (mirrors GetValidShape for tiles).
  std::vector<ExprPtr> valid = (tensor_type->tensor_view_ && !tensor_type->tensor_view_->valid_shape.empty())
                                   ? tensor_type->tensor_view_->valid_shape
                                   : tensor_type->shape_;
  // Same boundary contract as the tile form this lowers to (pass 10), checked
  // here so the diagnostic carries the author's own @pl.jit span.
  CheckSplitBoundaryCarriesValid(op_name, tensor_type->shape_, valid, static_cast<int>(axis), halve,
                                 args[0]->span_);
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

// `split_axis` is 0 (UP_DOWN), 1 (LEFT_RIGHT), or -1 for the shape-preserving
// split=0 crossing, which has no split axis and therefore no per-lane extent.
void CheckSplitBoundaryCarriesValid(const std::string& op_name, const std::vector<ExprPtr>& shape,
                                    const std::vector<ExprPtr>& valid, int split_axis, bool halve,
                                    const Span& span) {
  if (shape.size() != 2 || valid.size() != 2) {
    return;
  }
  // (1) The gather is exempt from the column contract at EVERY split, including
  // the shape-preserving split=0 crossing: a V2C pop lands in an NZ Mat tile
  // through TLoadGm2L1Nd2nz, which reads neither validRow nor validCol, and it
  // never takes the pto.treshape path (use_full_box is gated on the C2V pop).
  //
  // Its own rule — that the two lanes' bands must abut — is NOT checked here.
  // This deducer runs before the per-lane extents exist, so the only extent it
  // can see is its own lane-agnostic ceil-div guess; judging the join on that
  // both misreports the extents and rejects shapes that are representable once
  // the lanes are localized. LowerAutoVectorSplit owns that rule instead, where
  // the true extents are known (split_axis_utils.cpp).
  if (!halve) {
    return;
  }

  // (2) The column field is the contested one. A full column extent needs no
  // restore at all, which also leaves the row field free.
  if (IsFullExtent(valid[1], shape[1])) {
    return;
  }

  // (3) LEFT_RIGHT narrows the split axis, so the column extent is per-lane.
  const bool column_is_per_lane = (split_axis == 1);
  CHECK_SPAN(!column_is_per_lane, span)
      << op_name << ": LEFT_RIGHT splits the column axis, but this tile's valid column extent ("
      << DescribeExtent(valid[1], shape[1])
      << ") does not cover the full box, so the two AIV lanes would hold different amounts of real "
         "data. A per-lane column extent cannot cross this boundary: the Cube<->Vector FIFO pins the "
         "transported column extent to the physical one, and the only way to restore a narrower "
         "extent afterwards (pto.treshape) is compile-time static and so cannot express a per-lane "
         "value.\n"
      << "Author one of these instead:\n"
      << "  * split the row axis: mode=pl.SplitMode.UP_DOWN (a per-lane ROW extent IS carried)\n"
      << "  * make the columns fully valid before the crossing and let the padding be don't-care at "
         "the store:\n"
      << "        acc = pl.set_validshape(acc, <valid_rows>, " << PythonPrint(shape[1]) << ")\n"
      << "        out[..., c0 : c0 + <half>] = shard   # c0 = aiv_id * <half>\n"
      << "  * keep the ragged column tail in its own matmul outside the pl.split_aiv region";

  // (4) The narrowed column extent must be rebuilt after the full-box transport,
  // and pto.treshape can only rebuild static extents -- on BOTH axes at once.
  const bool column_is_static = static_cast<bool>(As<ConstInt>(valid[1]));
  CHECK_SPAN(column_is_static, span)
      << op_name << ": this tile's valid column extent (" << DescribeExtent(valid[1], shape[1])
      << ") is a runtime value narrower than the physical box. A narrowed column extent has to be "
         "rebuilt after the boundary's full-box transport, and the only rebuild available "
         "(pto.treshape) is compile-time static.\n"
      << "Fix: make the columns fully valid before the crossing --\n"
      << "        acc = pl.set_validshape(acc, <valid_rows>, " << PythonPrint(shape[1]) << ")\n"
      << "    and store the full column box; the padded columns are don't-care.";

  // A row extent that is per-lane (split-axis narrowing on an UP_DOWN shard) or
  // runtime-valued cannot survive the static treshape that the narrowed column
  // extent forces, because treshape rewrites both axes from one target type.
  const bool row_is_per_lane = (split_axis == 0) && !IsFullExtent(valid[0], shape[0]);
  const bool row_is_static = static_cast<bool>(As<ConstInt>(valid[0]));
  CHECK_SPAN(!row_is_per_lane, span)
      << op_name << ": UP_DOWN split with BOTH a per-lane row extent (" << DescribeExtent(valid[0], shape[0])
      << ", so the lanes hold different row counts) and a "
      << "narrowed column extent (" << DescribeExtent(valid[1], shape[1]) << ") is not supported. "
      << "The per-lane row extent rides on the TPOP valid_row operand, but the narrowed column "
         "extent can only be restored by pto.treshape, which rebuilds BOTH axes from one static "
         "type and would overwrite that per-lane row.\n"
      << "Fix: widen the columns before the crossing and store the full column box --\n"
      << "        acc = pl.set_validshape(acc, " << PythonPrint(valid[0]) << ", " << PythonPrint(shape[1])
      << ")\n"
      << "        out[r0 : r0 + <half>, 0 : " << PythonPrint(shape[1]) << "] = shard";
  CHECK_SPAN(row_is_static, span)
      << op_name << ": this tile's valid row extent (" << DescribeExtent(valid[0], shape[0])
      << ") is a runtime value and its valid column extent (" << DescribeExtent(valid[1], shape[1])
      << ") is narrower than the physical box. The narrowed column extent forces a full-box "
         "transport whose logical extents are rebuilt by the static pto.treshape, which cannot "
         "express a runtime row extent.\n"
      << "Fix: make the columns fully valid before the crossing --\n"
      << "        acc = pl.set_validshape(acc, <valid_rows>, " << PythonPrint(shape[1]) << ")\n"
      << "    and store the full column box; the padded columns are don't-care.";
}

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
    // Optional partition stride (see tile.aiv_shard); consumed by the torch
    // reference runtime, ignored by PTO codegen.
    .set_attr<int>("lane_stride")
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
    // Optional partition stride (see tile.aiv_shard); consumed by the torch
    // reference runtime, ignored by PTO codegen.
    .set_attr<int>("lane_stride")
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
//
// The memory contract is MODE-INDEPENDENT: it describes which lane produces the
// value and which consumes it, and a task-parallel (split=0) region crosses the
// same two lanes as a data-parallel one. Only the SHAPE differs — at split=0 the
// crossing preserves it (see DeduceSplitReshape).

// Shard a full tile into half along the split axis (cube -> vector vocabulary).
REGISTER_OP("tile.aiv_shard")
    .set_op_category("CrossCoreOp")
    .set_description(
        "Cross the AIC->AIV boundary; halve the 2D tile along the split axis (split=1/2), or preserve its "
        "shape (split=0)")
    .add_argument("tile", "Tile data to shard (TileType, 2D)")
    .set_attr<int>("split")
    // Optional; stamped by LowerAutoVectorSplit when it balances a ragged
    // boundary across the two lanes (see CheckLaneStrideAttr).
    .set_attr<int>("lane_stride")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshape(args, kwargs, "tile.aiv_shard", /*halve=*/true);
    });

// Gather two half tiles back into a full tile along the split axis (inverse of aiv_shard).
REGISTER_OP("tile.aic_gather")
    .set_op_category("CrossCoreOp")
    .set_description(
        "Cross the AIV->AIC boundary; rejoin the 2D tile along the split axis (split=1/2), or preserve its "
        "shape (split=0)")
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
    .set_description(
        "Cross the AIC->AIV boundary; halve the 2D tensor along the split axis (split=1/2), or preserve its "
        "shape (split=0)")
    .add_argument("tensor", "Tensor data to shard (TensorType, 2D)")
    .set_attr<int>("split")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshapeTensor(args, kwargs, "tensor.aiv_shard", /*halve=*/true);
    });

// Gather a half 2D tensor back into full along the split axis (inverse of aiv_shard).
REGISTER_OP("tensor.aic_gather")
    .set_op_category("CrossCoreOp")
    .set_description(
        "Cross the AIV->AIC boundary; rejoin the 2D tensor along the split axis (split=1/2), or preserve its "
        "shape (split=0)")
    .add_argument("tensor", "Tensor data to gather (TensorType, 2D)")
    .set_attr<int>("split")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshapeTensor(args, kwargs, "tensor.aic_gather", /*halve=*/false);
    });

}  // namespace ir
}  // namespace pypto
