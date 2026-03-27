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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TILE_VIEW_SEMANTICS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TILE_VIEW_SEMANTICS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto::ir::tile_view_semantics {

/// Return whether two shape-like expression lists are statically identical.
inline bool ShapeExprListsEquivalent(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] == rhs[i]) {
      continue;
    }
    auto lhs_const = As<ConstInt>(lhs[i]);
    auto rhs_const = As<ConstInt>(rhs[i]);
    if (!lhs_const || !rhs_const || lhs_const->value_ != rhs_const->value_) {
      return false;
    }
  }
  return true;
}

/// Infer the implicit block layout used when Python syntax omits TileView.
inline TileLayout InferImplicitTileLayoutFromShape(const std::vector<ExprPtr>& shape) {
  if (shape.size() != 2) {
    return TileLayout::row_major;
  }

  auto rows_const = As<ConstInt>(shape[0]);
  auto cols_const = As<ConstInt>(shape[1]);
  if (!rows_const || !cols_const) {
    return TileLayout::row_major;
  }
  return (cols_const->value_ == 1 && rows_const->value_ > 1) ? TileLayout::col_major : TileLayout::row_major;
}

/// Build the implicit TileView semantics represented by omitted Python syntax.
inline TileView GetImplicitTileView(const std::vector<ExprPtr>& shape,
                                    const std::optional<MemorySpace>& memory_space = std::nullopt) {
  TileView implicit_view;
  implicit_view.valid_shape = shape;
  implicit_view.blayout = InferImplicitTileLayoutFromShape(shape);

  if (memory_space.has_value()) {
    switch (*memory_space) {
      case MemorySpace::Mat:
      case MemorySpace::Left:
        implicit_view.blayout = TileLayout::col_major;
        implicit_view.slayout = TileLayout::row_major;
        break;
      case MemorySpace::Right:
        implicit_view.slayout = TileLayout::col_major;
        break;
      case MemorySpace::Acc:
        implicit_view.blayout = TileLayout::col_major;
        implicit_view.slayout = TileLayout::row_major;
        implicit_view.fractal = 1024;
        break;
      default:
        break;
    }
  }

  return implicit_view;
}

/// Return whether TileView matches the printer's raw TileView() defaults.
inline bool IsDefaultPrintedTileView(const TileView& tile_view, const std::vector<ExprPtr>& shape) {
  if (!tile_view.stride.empty() || tile_view.start_offset || tile_view.pad != PadValue::null) {
    return false;
  }

  const std::vector<ExprPtr>& normalized_valid_shape =
      tile_view.valid_shape.empty() ? shape : tile_view.valid_shape;
  if (!ShapeExprListsEquivalent(normalized_valid_shape, shape)) {
    return false;
  }

  TileView default_view;
  return tile_view.blayout == default_view.blayout && tile_view.slayout == default_view.slayout &&
         tile_view.fractal == default_view.fractal;
}

/// Return whether TileView matches the semantics of omitted Python syntax.
inline bool IsImplicitPrintedTileView(const TileView& tile_view, const std::vector<ExprPtr>& shape,
                                      const std::optional<MemorySpace>& memory_space = std::nullopt) {
  if (!ShapeExprListsEquivalent(tile_view.valid_shape, shape)) {
    return false;
  }
  if (!tile_view.stride.empty() || tile_view.start_offset || tile_view.pad != PadValue::null) {
    return false;
  }

  TileView implicit_view = GetImplicitTileView(shape, memory_space);
  return tile_view.blayout == implicit_view.blayout && tile_view.slayout == implicit_view.slayout &&
         tile_view.fractal == implicit_view.fractal;
}

/// Normalize sparse/default TileView syntax to a comparable semantic form.
inline TileView NormalizeImplicitTileView(const std::optional<TileView>& tile_view,
                                          const std::vector<ExprPtr>& shape,
                                          const std::optional<MemorySpace>& memory_space = std::nullopt,
                                          bool fill_start_offset = false) {
  TileView normalized = tile_view.value_or(TileView{});
  if (normalized.valid_shape.empty()) {
    normalized.valid_shape = shape;
  }
  if (!tile_view.has_value() || IsDefaultPrintedTileView(normalized, shape)) {
    TileView implicit_view = GetImplicitTileView(shape, memory_space);
    normalized.blayout = implicit_view.blayout;
    normalized.slayout = implicit_view.slayout;
    normalized.fractal = implicit_view.fractal;
  }
  if (fill_start_offset && !normalized.start_offset) {
    normalized.start_offset = std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
  }
  return normalized;
}

/// Return whether explicit TileView() can be safely omitted in printed syntax.
inline bool CanOmitExplicitEmptyTileView(const std::vector<ExprPtr>& shape,
                                         const std::optional<MemorySpace>& memory_space = std::nullopt) {
  TileView default_view;
  TileView implicit_view = GetImplicitTileView(shape, memory_space);
  return implicit_view.blayout == default_view.blayout && implicit_view.slayout == default_view.slayout &&
         implicit_view.fractal == default_view.fractal;
}

/// Return the valid_shape the printer should materialize for tile operations.
inline std::vector<ExprPtr> GetPrintedValidShape(const std::optional<TileView>& tile_view,
                                                 const std::vector<ExprPtr>& shape) {
  if (tile_view.has_value() && !tile_view->valid_shape.empty()) {
    return tile_view->valid_shape;
  }
  return shape;
}

}  // namespace pypto::ir::tile_view_semantics

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TILE_VIEW_SEMANTICS_H_
