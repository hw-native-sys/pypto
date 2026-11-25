/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DIRECT_L1_DMA_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DIRECT_L1_DMA_H_

#include <cstdint>
#include <limits>

#include "pypto/core/logging.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir::direct_l1_dma {

inline constexpr int kBurstBytes = 32;
inline constexpr int kMaxBurstLength = std::numeric_limits<uint16_t>::max();
inline constexpr int kMaxBursts = 4095;
inline constexpr int kLaneOffset = 16;

enum class Layout : int { ND, NZ, DN, ZN };

struct TileGeometry {
  int rows;
  int cols;
  int element_bytes;
  Layout layout;

  [[nodiscard]] int RowBytes() const { return cols * element_bytes; }
  [[nodiscard]] int Bytes() const { return rows * RowBytes(); }
};

// The transport reads raw bytes. The descriptor must establish the physical
// layout and extent before the mixed kernel is outlined.
inline TileGeometry Describe(const TileTypePtr& tile, const Span& span) {
  CHECK_SPAN(tile && tile->shape_.size() == 2, span)
      << "A5 L1-to-UB transfer requires a two-dimensional Tile";
  const auto rows = transform_utils::EvalConstInt(tile->shape_[0]);
  const auto cols = transform_utils::EvalConstInt(tile->shape_[1]);
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  const int64_t bits = static_cast<int64_t>(tile->dtype_.GetBit());
  constexpr int64_t max_bytes = kMaxBurstLength * kBurstBytes;
  CHECK_SPAN(rows && cols && *rows > 0 && *cols > 0 && bits >= 8 && bits % 8 == 0 &&
                 *cols <= max_bytes / (bits / 8) && *rows <= max_bytes / (*cols * (bits / 8)),
             span)
      << "A5 L1-to-UB transfer needs static extents and a byte-addressable dtype within the DMA burst limit";
  CHECK_SPAN(view.valid_shape.size() == 2 && transform_utils::EvalConstInt(view.valid_shape[0]) == rows &&
                 transform_utils::EvalConstInt(view.valid_shape[1]) == cols,
             span)
      << "A5 L1-to-UB transfer requires full valid extents";
  CHECK_SPAN(view.compact == CompactMode::null &&
                 (!view.start_offset || transform_utils::EvalConstInt(view.start_offset) == 0),
             span)
      << "A5 L1-to-UB transfer requires a whole, non-compact Tile";
  const bool row_major = view.blayout == TileLayout::row_major;
  const bool flat = view.slayout == TileLayout::none_box;
  const Layout layout = flat ? (row_major ? Layout::ND : Layout::DN) : (row_major ? Layout::ZN : Layout::NZ);
  CHECK_SPAN(
      view.stride.empty() || (flat && view.stride.size() == 2 &&
                              transform_utils::EvalConstInt(view.stride[0]) == (row_major ? *cols : 1) &&
                              transform_utils::EvalConstInt(view.stride[1]) == (row_major ? 1 : *rows)),
      span)
      << "A5 L1-to-UB transfer requires packed Tile storage";
  CHECK_SPAN(flat || (view.fractal == 512 &&
                      view.slayout == (row_major ? TileLayout::col_major : TileLayout::row_major) &&
                      (row_major ? *cols : *rows) % 16 == 0),
             span)
      << "A5 L1-to-UB transfer requires standard 512-byte NZ or ZN boxes";
  CHECK_SPAN((row_major == flat ? *cols : *rows) * (bits / 8) % kBurstBytes == 0, span)
      << "A5 L1-to-UB transfer requires 32-byte-aligned physical row or column extents";
  return {static_cast<int>(*rows), static_cast<int>(*cols), static_cast<int>(bits / 8), layout};
}

}  // namespace pypto::ir::direct_l1_dma

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DIRECT_L1_DMA_H_
