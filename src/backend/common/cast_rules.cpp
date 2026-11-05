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

#include "pypto/backend/common/cast_rules.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::backend {
namespace {

bool Reject(DirectCastInfo& info, const std::string& reason) {
  info.reason = reason;
  return false;
}

/// Check one endpoint against its own conditions. `role` only names the
/// endpoint in the diagnostic, so a rejection says which side failed.
bool CheckEndpoint(const TcvtEndpointConditions& rule, const ir::TileTypePtr& tile, const char* role,
                   DirectCastInfo& info) {
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  const std::string prefix = std::string("direct cast ") + role + " ";
  if (rule.blayout && view.blayout != *rule.blayout) {
    return Reject(info, prefix + "block layout is unsupported");
  }
  if (rule.slayout && view.slayout != *rule.slayout) {
    return Reject(info, prefix + "scatter layout is unsupported");
  }
  if (rule.fractal && view.fractal != *rule.fractal) {
    return Reject(info, prefix + "boxing granularity is unsupported");
  }
  if (rule.null_compact && view.compact != ir::CompactMode::null) {
    return Reject(info, prefix + "requires compact=null");
  }
  if (rule.vec_only && tile->memory_space_ && *tile->memory_space_ != ir::MemorySpace::Vec) {
    return Reject(info, prefix + "requires Vec memory");
  }
  if (rule.static_shape || rule.row_alignment != 1) {
    if (tile->shape_.size() != 2) return Reject(info, prefix + "requires static 2D geometry");
    const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
    const auto cols = ir::As<ir::ConstInt>(tile->shape_[1]);
    if (!rows || !cols || rows->value_ <= 0 || cols->value_ <= 0 || rows->value_ > rule.max_rows ||
        cols->value_ > rule.max_columns ||
        rows->value_ > rule.max_byte_span / cols->value_ / tile->dtype_.GetByte()) {
      return Reject(info, prefix + "physical shape exceeds its static dimension or byte-span limits");
    }
    // A boxed endpoint addresses whole fractal boxes, so a partial row of boxes
    // has no representable storage. Checked on the *physical* shape: PTOAS pads
    // the tile up to the box grid and tracks the narrower extent as the valid
    // region, which `static_valid` covers separately.
    if (rows->value_ % rule.row_alignment != 0) {
      return Reject(info, prefix + "physical rows are not a whole number of boxes");
    }
  }
  if (rule.static_valid) {
    for (const auto& dim : view.valid_shape) {
      const auto valid = ir::As<ir::ConstInt>(dim);
      if (!valid || valid->value_ <= 0) {
        return Reject(info, prefix + "requires static positive valid extents");
      }
    }
  }
  if (rule.full_valid && !ir::tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, tile->shape_)) {
    return Reject(info, prefix + "requires full static validity");
  }
  if (rule.dense_stride && !ir::tile_view_semantics::HasCanonicalDenseStrides(*tile, view)) {
    return Reject(info, prefix + "requires canonical dense 2D strides");
  }
  if (rule.zero_start_offset && view.start_offset) {
    const auto offset = ir::As<ir::ConstInt>(view.start_offset);
    if (!offset || offset->value_ != 0) return Reject(info, prefix + "requires zero tile start_offset");
  }
  return true;
}

bool CheckConditions(const TcvtConditions& rule, const ir::TileTypePtr& source, const ir::TileTypePtr& result,
                     DirectCastInfo& info) {
  if (rule.matching_valid_shape &&
      !ir::CastExtentsEqual(source, result, ir::GetValidShape(source), ir::GetValidShape(result))) {
    return Reject(info, "cast source/result logical valid extents disagree");
  }
  const bool empty = ir::tile_view_semantics::IsZeroValidShapeSentinel(ir::GetValidShape(source)) &&
                     ir::tile_view_semantics::IsZeroValidShapeSentinel(ir::GetValidShape(result));
  // This is an internal no-data replay after splitting, not a relaxation for
  // arbitrary zero-valid user tiles. The dtype edge must explicitly allow it.
  if (rule.allow_empty_replay && empty) {
    info.empty_replay = true;
    info.descriptor = rule.descriptor;
    info.preserve_input_layout = rule.preserve_input_layout;
    info.preserve_output_layout = rule.preserve_output_layout;
    return true;
  }
  if (!CheckEndpoint(rule.source, source, "source", info)) return false;
  if (!CheckEndpoint(rule.result, result, "result", info)) return false;
  if (rule.source_col_alignment != 1 || rule.max_source_registers != std::numeric_limits<int64_t>::max()) {
    if (source->shape_.size() != 2) return Reject(info, "direct cast requires 2D source geometry");
    const auto cols = ir::As<ir::ConstInt>(source->shape_[1]);
    if (!cols || cols->value_ % rule.source_col_alignment != 0 ||
        cols->value_ / rule.source_register_bytes > rule.max_source_registers) {
      return Reject(info, "direct cast source row alignment or register count is unsupported");
    }
  }
  info.descriptor = rule.descriptor;
  info.preserve_input_layout = rule.preserve_input_layout;
  info.preserve_output_layout = rule.preserve_output_layout;
  info.reason.clear();
  return true;
}

}  // namespace

bool IsDirectCast(const ir::CallPtr& call, const TcvtAdjacency& rules, DirectCastInfo* details) {
  DirectCastInfo local;
  auto& info = details ? *details : local;
  info = {};
  if (!ir::IsOp(call, "tile.cast") || call->args_.empty()) {
    return Reject(info, "expected tile.cast");
  }
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto result = ir::As<ir::TileType>(call->GetType());
  if (!source || !result) return Reject(info, "cast endpoints must be TileTypes");
  for (const auto& edge : rules.edges) {
    if (edge.source != source->dtype_ || edge.destination != result->dtype_) continue;
    if (edge.alternatives.empty()) return CheckConditions({}, source, result, info);
    for (const auto& conditions : edge.alternatives) {
      DirectCastInfo candidate;
      if (CheckConditions(conditions, source, result, candidate)) {
        info = candidate;
        return true;
      }
      info = candidate;
    }
    return false;
  }
  return Reject(info, "dtype pair has no native TCVT edge on this backend");
}

}  // namespace pypto::backend
