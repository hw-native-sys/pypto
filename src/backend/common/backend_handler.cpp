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

#include "pypto/backend/common/backend_handler.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace backend {

namespace {

bool Reject(DirectCastInfo& info, const std::string& reason) {
  info.reason = reason;
  return false;
}

// Casts preserve logical value geometry. Packed x2 shapes count carriers, so
// compare after expanding the contiguous axis; dtype alone cannot prove this.
bool LogicalExtentsMatch(const ir::TileTypePtr& source, const ir::TileTypePtr& result,
                         const std::vector<ir::ExprPtr>& source_shape,
                         const std::vector<ir::ExprPtr>& result_shape) {
  if (source_shape.size() != result_shape.size() || source_shape.empty()) return false;
  auto normalized_source = source_shape;
  auto normalized_result = result_shape;
  if (source->dtype_.IsPackedFp4() != result->dtype_.IsPackedFp4()) {
    const bool packed_source = source->dtype_.IsPackedFp4();
    auto& expanded = packed_source ? normalized_source : normalized_result;
    const auto packed = packed_source ? source : result;
    const auto view = ir::tile_view_semantics::GetEffectiveTileView(*packed);
    size_t axis = expanded.size() - 1;
    if (view.blayout == ir::TileLayout::col_major) {
      if (expanded.size() != 2) return false;
      axis = 0;
    }
    const auto two = std::make_shared<ir::ConstInt>(2, DataType::INDEX, ir::Span::unknown());
    if (const auto extent = ir::As<ir::ConstInt>(expanded[axis])) {
      // Guard before doubling malformed raw IR: a boolean capability query must
      // not overflow while deciding that a descriptor is unsupported.
      if (extent->value_ < 0 || extent->value_ > std::numeric_limits<int64_t>::max() / 2) return false;
      expanded[axis] = std::make_shared<ir::ConstInt>(extent->value_ * 2, DataType::INDEX, extent->span_);
    } else {
      expanded[axis] = std::make_shared<ir::Mul>(expanded[axis], two, DataType::INDEX, ir::Span::unknown());
    }
  }
  // Inference can retain (n + 1) * 2 while simplification expands it to
  // n * 2 + 2. Normalize both endpoints before structural comparison.
  ir::arith::Analyzer analyzer;
  for (size_t axis = 0; axis < normalized_source.size(); ++axis) {
    normalized_source[axis] = analyzer.Simplify(normalized_source[axis]);
    normalized_result[axis] = analyzer.Simplify(normalized_result[axis]);
  }
  return ir::tile_view_semantics::ShapeExprListsEquivalent(normalized_source, normalized_result);
}

bool CheckConditions(const TcvtConditions& rule, const ir::TileTypePtr& source, const ir::TileTypePtr& result,
                     DirectCastInfo& info) {
  if (rule.matching_valid_shape &&
      !LogicalExtentsMatch(source, result, ir::GetValidShape(source), ir::GetValidShape(result))) {
    return Reject(info, "cast source/result logical valid extents disagree");
  }
  const bool empty = ir::tile_view_semantics::IsZeroValidShapeSentinel(ir::GetValidShape(source)) &&
                     ir::tile_view_semantics::IsZeroValidShapeSentinel(ir::GetValidShape(result));
  // This is an internal no-data replay after splitting, not a relaxation for
  // arbitrary zero-valid user tiles. The dtype edge must explicitly allow it.
  if (rule.allow_empty_replay && empty) {
    info.empty_replay = true;
    info.descriptor = rule.descriptor;
    info.preserve_layout = rule.preserve_layout;
    return true;
  }
  for (const auto& tile : {source, result}) {
    const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
    if (rule.layout && view.blayout != *rule.layout) return Reject(info, "cast layout is unsupported");
    if (rule.unboxed && view.slayout != ir::TileLayout::none_box) {
      return Reject(info, "packed cast requires none_box storage");
    }
    if (rule.null_compact && view.compact != ir::CompactMode::null) {
      return Reject(info, "direct cast requires compact=null");
    }
    if (rule.vec_only && tile->memory_space_ && *tile->memory_space_ != ir::MemorySpace::Vec) {
      return Reject(info, "direct cast requires Vec memory");
    }
    if (rule.static_shape) {
      if (tile->shape_.size() != 2) return Reject(info, "direct cast requires static 2D geometry");
      const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
      const auto cols = ir::As<ir::ConstInt>(tile->shape_[1]);
      if (!rows || !cols || rows->value_ <= 0 || cols->value_ <= 0 || rows->value_ > rule.max_rows ||
          cols->value_ > rule.max_columns ||
          rows->value_ > rule.max_byte_span / cols->value_ / tile->dtype_.GetByte()) {
        return Reject(info, "direct cast physical shape exceeds its static dimension or byte-span limits");
      }
    }
    if (rule.static_valid) {
      if (view.valid_shape.size() != tile->shape_.size()) return Reject(info, "cast valid rank mismatch");
      for (size_t axis = 0; axis < view.valid_shape.size(); ++axis) {
        const auto valid = ir::As<ir::ConstInt>(view.valid_shape[axis]);
        const auto extent = ir::As<ir::ConstInt>(tile->shape_[axis]);
        if (!valid || !extent || valid->value_ <= 0 || valid->value_ > extent->value_) {
          return Reject(info, "direct cast requires static positive valid extents within physical shape");
        }
      }
    }
    if (rule.full_valid &&
        !ir::tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, tile->shape_)) {
      return Reject(info, "direct DN cast requires full static validity");
    }
    if (rule.dense_stride && !view.stride.empty()) {
      if (tile->shape_.size() != 2 || view.stride.size() != 2) {
        return Reject(info, "direct cast requires dense 2D strides");
      }
      const auto first = ir::As<ir::ConstInt>(view.stride[0]);
      const auto second = ir::As<ir::ConstInt>(view.stride[1]);
      const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
      const auto cols = ir::As<ir::ConstInt>(tile->shape_[1]);
      const bool column_major = view.blayout == ir::TileLayout::col_major;
      if (!first || !second || !rows || !cols || first->value_ != (column_major ? 1 : cols->value_) ||
          second->value_ != (column_major ? rows->value_ : 1)) {
        return Reject(info, "direct cast requires canonical dense strides");
      }
    }
    if (rule.zero_start_offset && view.start_offset) {
      const auto offset = ir::As<ir::ConstInt>(view.start_offset);
      if (!offset || offset->value_ != 0) return Reject(info, "direct cast requires zero tile start_offset");
    }
  }
  if (rule.source_col_alignment != 1 || rule.max_source_registers != std::numeric_limits<int64_t>::max()) {
    if (source->shape_.size() != 2) return Reject(info, "direct cast requires 2D source geometry");
    const auto cols = ir::As<ir::ConstInt>(source->shape_[1]);
    if (!cols || cols->value_ % rule.source_col_alignment != 0 ||
        cols->value_ / rule.source_register_bytes > rule.max_source_registers) {
      return Reject(info, "direct cast source row alignment or register count is unsupported");
    }
  }
  info.descriptor = rule.descriptor;
  info.preserve_layout = rule.preserve_layout;
  info.reason.clear();
  return true;
}

}  // namespace

bool BackendHandler::IsDirectCast(const ir::CallPtr& call, DirectCastInfo* details) const {
  DirectCastInfo local;
  auto& info = details ? *details : local;
  info = {};
  if (!ir::IsOp(call, "tile.cast") || call->args_.empty() || call->args_.size() > 2) {
    return Reject(info, "expected tile.cast(src[, tmp])");
  }
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto result = ir::As<ir::TileType>(call->GetType());
  if (!source || !result) return Reject(info, "cast endpoints must be TileTypes");
  if (!call->HasKwarg("target_type") || !call->HasKwarg("mode")) {
    return Reject(info, "cast requires target_type and mode attributes");
  }
  DataType target;
  int mode = 0;
  int saturation = 0;
  try {
    target = call->GetKwarg<DataType>("target_type");
    mode = call->GetKwarg<int>("mode");
    saturation = call->GetKwarg<int>("saturation_mode", 0);
  } catch (const TypeError&) {
    return Reject(info, "cast attributes have unsupported types");
  }
  if (target != result->dtype_) return Reject(info, "cast target_type disagrees with its result dtype");
  if (mode < 0 || mode > 6 || (result->dtype_.IsPackedFp4() && mode == 6)) {
    return Reject(info, "cast rounding mode is unsupported");
  }
  if (!LogicalExtentsMatch(source, result, source->shape_, result->shape_)) {
    return Reject(info, "cast source/result logical shapes disagree");
  }
  // Ordinary TCVT iterates the destination's valid rectangle. A cross-core
  // transport can expose a full source box while the consumer keeps its
  // runtime lane extent; equal validity is an edge-specific requirement.
  for (const auto& tile : {source, result}) {
    const auto valid = ir::GetValidShape(tile);
    if (valid.size() != tile->shape_.size()) return Reject(info, "cast valid rank mismatch");
    for (size_t axis = 0; axis < tile->shape_.size(); ++axis) {
      const auto extent = ir::As<ir::ConstInt>(tile->shape_[axis]);
      const auto count = ir::As<ir::ConstInt>(valid[axis]);
      if ((extent && extent->value_ <= 0) || (count && count->value_ < 0) ||
          (extent && count && count->value_ > extent->value_)) {
        return Reject(info, "cast valid extents must fit a positive physical shape");
      }
    }
  }
  if (saturation != 0 && saturation != 1) return Reject(info, "cast saturation mode is unsupported");
  if (call->args_.size() == 2) {
    const auto scratch = ir::As<ir::TileType>(call->args_[1]->GetType());
    if (!scratch || (scratch->memory_space_ && *scratch->memory_space_ != ir::MemorySpace::Vec)) {
      return Reject(info, "cast scratch must be a Vec tile");
    }
  }
  for (const auto& edge : GetTcvtAdjacency().edges) {
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

uint64_t BackendHandler::GetCapacityBytes(ir::MemorySpace memory_space) const {
  return backend_.GetMemSize(memory_space);
}

}  // namespace backend
}  // namespace pypto
