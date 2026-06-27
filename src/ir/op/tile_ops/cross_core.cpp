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
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

// Shared deducer for the split-axis reshape ops tile.aiv_shard (full -> half) and
// tile.aic_gather (half -> full). The single positional tile argument is reshaped
// along the split axis selected by the "split" int attr (same encoding as
// tpush/tpop: 1 = UP_DOWN/axis0, 2 = LEFT_RIGHT/axis1). When `halve` is true the
// split-axis extent is halved (aiv_shard); otherwise it is doubled (aic_gather).
//
// 2D-vocab constraint: the input must be rank-2 and the split attr must be 1 or 2.
// For the halving direction a static (ConstInt) split-axis extent must be even;
// dynamic (non-ConstInt) extents are reshaped symbolically (floordiv(dim, 2) on
// shard, dim * 2 on gather).
TypePtr DeduceSplitReshape(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name, bool halve) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 tile argument, but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Read the required "split" int attr (reuses the tpush/tpop encoding).
  std::optional<int> split_opt;
  for (const auto& [key, value] : kwargs) {
    if (key == "split") {
      split_opt = AnyCast<int>(value, "kwarg key: split");
      break;
    }
  }
  CHECK_SPAN(split_opt.has_value(), args[0]->span_)
      << op_name << " requires a 'split' attr (1 = UP_DOWN/axis0, 2 = LEFT_RIGHT/axis1)";
  const int split = *split_opt;
  CHECK_SPAN(split == 1 || split == 2, args[0]->span_)
      << op_name << " split must be 1 (UP_DOWN/axis0) or 2 (LEFT_RIGHT/axis1), but got " << split;
  CHECK_SPAN(tile_type->shape_.size() == 2, args[0]->span_)
      << op_name << " requires a 2D tile, but got rank " << tile_type->shape_.size();

  const size_t axis = (split == 1) ? 0 : 1;

  // Reshape the physical shape and the valid_shape along the split axis so
  // downstream codegen sees a consistent view. Static (ConstInt) extents are
  // halved/doubled directly; dynamic extents become symbolic floordiv(dim, 2) /
  // dim * 2 so the result type reflects the shard/gather along the split axis
  // (ExpandMixedKernel consumes it as the authoritative half/full boundary size)
  // rather than typing as an identity reshape.
  std::vector<ExprPtr> new_shape = tile_type->shape_;
  std::vector<ExprPtr> new_valid = GetValidShape(tile_type);

  // The even-extent requirement applies to the PHYSICAL split-axis extent only;
  // the per-lane valid_shape is reshaped with ceil-div (keeping valid <= physical)
  // since the true per-lane valid region is localized later at lowering time, which
  // knows the subblock (lane) index. This avoids rejecting a tile whose physical
  // extent is even but whose partial valid_shape happens to be odd.
  if (auto c = As<ConstInt>(new_shape[axis])) {
    if (halve) {
      CHECK_SPAN(c->value_ % 2 == 0, args[0]->span_)
          << op_name << ": split-axis static extent " << c->value_ << " must be even to shard in half";
      new_shape[axis] = std::make_shared<ConstInt>(c->value_ / 2, c->dtype(), new_shape[axis]->span_);
    } else {
      new_shape[axis] = std::make_shared<ConstInt>(c->value_ * 2, c->dtype(), new_shape[axis]->span_);
    }
  } else {
    // Dynamic split-axis extent: symbolic half / double. Per-lane evenness is
    // resolved at lowering time, which knows the subblock index.
    auto two = std::make_shared<ConstInt>(2, GetScalarDtype(new_shape[axis]), new_shape[axis]->span_);
    new_shape[axis] = halve ? MakeFloorDiv(new_shape[axis], two, new_shape[axis]->span_)
                            : MakeMul(new_shape[axis], two, new_shape[axis]->span_);
  }
  if (axis < new_valid.size()) {
    if (auto vc = As<ConstInt>(new_valid[axis])) {
      const auto new_extent = halve ? (vc->value_ + 1) / 2 : vc->value_ * 2;
      new_valid[axis] = std::make_shared<ConstInt>(new_extent, vc->dtype(), new_valid[axis]->span_);
    } else {
      // Dynamic valid extent: ceil-div on halve (floordiv(dim + 1, 2)), double on
      // gather — mirroring the physical reshape; the exact per-lane valid region
      // is re-derived at lowering time.
      auto vspan = new_valid[axis]->span_;
      auto dt = GetScalarDtype(new_valid[axis]);
      auto two = std::make_shared<ConstInt>(2, dt, vspan);
      if (halve) {
        auto one = std::make_shared<ConstInt>(1, dt, vspan);
        new_valid[axis] = MakeFloorDiv(MakeAdd(new_valid[axis], one, vspan), two, vspan);
      } else {
        new_valid[axis] = MakeMul(new_valid[axis], two, vspan);
      }
    }
  }

  // The result is a fresh per-lane (shard) / re-joined (gather) tile along the
  // split axis. Only the halved/doubled valid_shape is carried; the source's
  // explicit blayout/slayout is intentionally NOT inherited. Inheriting a
  // non-implicit layout (e.g. an Acc operand's col_major) makes the result type
  // diverge from the deduction fixpoint that downstream elementwise consumers
  // (which re-derive layout from their inputs) and a print->parse round-trip
  // reconstruct — the boundary's true memory layout is re-attached by the
  // lowering pass (ReshapeTypeWithMemory) and normalized downstream.
  TileView tile_view;
  tile_view.valid_shape = std::move(new_valid);
  return std::make_shared<TileType>(std::move(new_shape), tile_type->dtype_, std::nullopt,
                                    std::move(tile_view));
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

// Shard a full tile into half along the split axis (cube -> vector vocabulary).
REGISTER_OP("tile.aiv_shard")
    .set_op_category("CrossCoreOp")
    .set_description("Shard a 2D tile into half along the split axis (full -> half)")
    .add_argument("tile", "Tile data to shard (TileType, 2D)")
    .set_attr<int>("split")
    .set_output_memory_inherit_input()
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
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSplitReshape(args, kwargs, "tile.aic_gather", /*halve=*/false);
    });

}  // namespace ir
}  // namespace pypto
