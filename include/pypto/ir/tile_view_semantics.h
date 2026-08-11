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

#ifndef PYPTO_IR_TILE_VIEW_SEMANTICS_H_
#define PYPTO_IR_TILE_VIEW_SEMANTICS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/type.h"

namespace pypto::ir::tile_view_semantics {

/// MX block-scale fractal size: one shared exponent per 32 elements (A5 ISA).
inline constexpr int kMXScaleFractal = 32;

/// Acc (L0C) fractal size: the accumulator is NZ-boxed at 1024 bytes.
inline constexpr uint64_t kAccFractal = 1024;

/// Static row/column granularity of one boxed tile storage unit.
struct BoxedTileAlignment {
  int64_t rows = 1;
  int64_t cols = 1;
};

/// Return the physical shape granularity imposed by a boxed TileView.
///
/// The result mirrors PTO's boxed ``tile_buf`` contract. Fractal-512 tiles
/// orient their 32-byte axis according to ``slayout``; accumulator
/// fractal-1024 tiles are 16x16, and MX-scale fractal-32 tiles are 16x2 or
/// 2x16 according to ``slayout``. Non-boxed or unsupported layouts return
/// ``nullopt`` so callers cannot silently apply a guessed alignment.
inline std::optional<BoxedTileAlignment> GetBoxedTileAlignment(const TileView& view, const DataType& dtype) {
  if (view.slayout == TileLayout::none_box) return std::nullopt;

  switch (view.fractal) {
    case kAccFractal:
      return BoxedTileAlignment{/*rows=*/16, /*cols=*/16};
    case kMXScaleFractal:
      if (view.slayout == TileLayout::row_major) {
        return BoxedTileAlignment{/*rows=*/16, /*cols=*/2};
      }
      if (view.slayout == TileLayout::col_major) {
        return BoxedTileAlignment{/*rows=*/2, /*cols=*/16};
      }
      return std::nullopt;
    case 512: {
      constexpr int64_t kAlignedBits = int64_t{32} * 8;
      const int64_t storage_bits = static_cast<int64_t>(storage_size::GetStorageBitWidth(dtype));
      if (storage_bits <= 0 || kAlignedBits % storage_bits != 0) return std::nullopt;
      const int64_t packed_extent = kAlignedBits / storage_bits;
      if (view.slayout == TileLayout::row_major) {
        return BoxedTileAlignment{/*rows=*/16, /*cols=*/packed_extent};
      }
      if (view.slayout == TileLayout::col_major) {
        return BoxedTileAlignment{/*rows=*/packed_extent, /*cols=*/16};
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

/// Return whether two shape-like expression lists are statically identical.
inline bool ShapeExprListsEquivalent(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    // ConstInt by value, binary composites structurally, others by pointer.
    if (!AreExprsEqual(lhs[i], rhs[i])) {
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

/// Boxing granularity implied by a memory space. Unlike blayout/slayout it does
/// not depend on the shape, so a caller that only needs the fractal (e.g. an op
/// deducing a result that lands in `memory_space`) can skip building a whole
/// TileView. GetImplicitTileView below delegates here so the two cannot drift.
inline uint64_t GetImplicitFractal(const std::optional<MemorySpace>& memory_space) {
  if (!memory_space.has_value()) {
    return TileView{}.fractal;
  }
  switch (*memory_space) {
    // MX scale tiles: one shared exponent per 32 elements.
    case MemorySpace::LeftScale:
    case MemorySpace::RightScale:
      return kMXScaleFractal;
    case MemorySpace::Acc:
      return kAccFractal;
    default:
      return TileView{}.fractal;
  }
}

/// The layout half of a TileView: the fields determined by (shape, memory
/// space) rather than by the data the tile holds. Kept separate from TileView so
/// a caller needing only the layout can skip building (and copying) a whole view.
struct TileLayoutSpec {
  TileLayout blayout = TileLayout::row_major;
  TileLayout slayout = TileLayout::none_box;
  uint64_t fractal = TileView{}.fractal;
};

inline bool operator==(const TileLayoutSpec& lhs, const TileLayoutSpec& rhs) {
  return lhs.blayout == rhs.blayout && lhs.slayout == rhs.slayout && lhs.fractal == rhs.fractal;
}
inline bool operator!=(const TileLayoutSpec& lhs, const TileLayoutSpec& rhs) { return !(lhs == rhs); }

/// The layout a tile of @p shape implicitly carries when it lives in
/// @p memory_space -- the single source of truth for the space->layout table.
/// An absent space yields the space-agnostic (flat) layout. Delegates the
/// fractal to GetImplicitFractal so the two cannot drift.
inline TileLayoutSpec GetImplicitTileLayout(const std::vector<ExprPtr>& shape,
                                            const std::optional<MemorySpace>& memory_space = std::nullopt) {
  TileLayoutSpec layout;
  layout.blayout = InferImplicitTileLayoutFromShape(shape);
  layout.fractal = GetImplicitFractal(memory_space);

  if (memory_space.has_value()) {
    switch (*memory_space) {
      case MemorySpace::Mat:
      case MemorySpace::Left:
        layout.blayout = TileLayout::col_major;
        layout.slayout = TileLayout::row_major;
        break;
      case MemorySpace::Right:
        layout.slayout = TileLayout::col_major;
        break;
      case MemorySpace::LeftScale:
        // ISA TileLeftScale: RowMajor / RowMajor.
        layout.blayout = TileLayout::row_major;
        layout.slayout = TileLayout::row_major;
        break;
      case MemorySpace::RightScale:
        // ISA TileRightScale: ColMajor / ColMajor.
        layout.blayout = TileLayout::col_major;
        layout.slayout = TileLayout::col_major;
        break;
      case MemorySpace::Acc:
        layout.blayout = TileLayout::col_major;
        layout.slayout = TileLayout::row_major;
        break;
      default:
        break;
    }
  }

  return layout;
}

/// Overwrite only @p view's layout fields, leaving valid_shape / stride /
/// start_offset / pad (which describe the data, not the memory) untouched.
inline void SetTileLayout(TileView& view, const TileLayoutSpec& layout) {
  view.blayout = layout.blayout;
  view.slayout = layout.slayout;
  view.fractal = layout.fractal;
}

/// Build the implicit TileView semantics represented by omitted Python syntax.
inline TileView GetImplicitTileView(const std::vector<ExprPtr>& shape,
                                    const std::optional<MemorySpace>& memory_space = std::nullopt) {
  TileView implicit_view;
  implicit_view.valid_shape = shape;
  SetTileLayout(implicit_view, GetImplicitTileLayout(shape, memory_space));
  return implicit_view;
}

/// Return whether TileView matches the printer's raw TileView() defaults.
inline bool IsDefaultPrintedTileView(const TileView& tile_view, const std::vector<ExprPtr>& shape) {
  if (!tile_view.stride.empty() || tile_view.start_offset || tile_view.pad != PadValue::null ||
      tile_view.compact != CompactMode::null) {
    return false;
  }

  const std::vector<ExprPtr>& normalized_valid_shape =
      tile_view.valid_shape.empty() ? shape : tile_view.valid_shape;
  if (!ShapeExprListsEquivalent(normalized_valid_shape, shape)) {
    return false;
  }

  TileView default_view;
  return tile_view.blayout == default_view.blayout && tile_view.slayout == default_view.slayout &&
         tile_view.fractal == default_view.fractal && tile_view.compact == default_view.compact;
}

/// Return whether TileView matches the semantics of omitted Python syntax.
inline bool IsImplicitPrintedTileView(const TileView& tile_view, const std::vector<ExprPtr>& shape,
                                      const std::optional<MemorySpace>& memory_space = std::nullopt) {
  // Empty valid_shape is semantically equivalent to shape (per the convention
  // in NormalizeImplicitTileView). Treat both forms as the same encoding so a
  // default-constructed TileView also collapses to nullopt and the canonical
  // encoding is unique.
  if (!tile_view.valid_shape.empty() && !ShapeExprListsEquivalent(tile_view.valid_shape, shape)) {
    return false;
  }
  if (!tile_view.stride.empty() || tile_view.start_offset || tile_view.pad != PadValue::null ||
      tile_view.compact != CompactMode::null) {
    return false;
  }

  TileView implicit_view = GetImplicitTileView(shape, memory_space);
  return tile_view.blayout == implicit_view.blayout && tile_view.slayout == implicit_view.slayout &&
         tile_view.fractal == implicit_view.fractal && tile_view.compact == implicit_view.compact;
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
         implicit_view.fractal == default_view.fractal && implicit_view.compact == default_view.compact;
}

/// Return the valid_shape the printer should materialize for tile operations.
inline std::vector<ExprPtr> GetPrintedValidShape(const std::optional<TileView>& tile_view,
                                                 const std::vector<ExprPtr>& shape) {
  if (tile_view.has_value() && !tile_view->valid_shape.empty()) {
    return tile_view->valid_shape;
  }
  return shape;
}

/// Return the effective TileView for a TileType. Empty valid_shape is expanded
/// to the physical shape, and an absent view receives the implicit layout for
/// (shape, memory_space). Callers that need semantic view fields should use
/// this rather than inspecting the canonical sparse storage directly.
inline TileView GetEffectiveTileView(const TileType& tile_type) {
  if (tile_type.tile_view_.has_value()) {
    TileView effective = *tile_type.tile_view_;
    if (effective.valid_shape.empty()) {
      effective.valid_shape = tile_type.shape_;
    }
    return effective;
  }
  return GetImplicitTileView(tile_type.shape_, tile_type.memory_space_);
}

/// TileType overload that first resolves implicit memory-space layout.
inline std::optional<BoxedTileAlignment> GetBoxedTileAlignment(const TileType& tile_type) {
  return GetBoxedTileAlignment(GetEffectiveTileView(tile_type), tile_type.dtype_);
}

}  // namespace pypto::ir::tile_view_semantics

#endif  // PYPTO_IR_TILE_VIEW_SEMANTICS_H_
