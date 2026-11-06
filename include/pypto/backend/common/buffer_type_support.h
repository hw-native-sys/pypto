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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_
#define PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

/// Element types supported by the dense Vec descriptor and ordinary GM transfer
/// recipes. Arithmetic instructions have separate operand-type contracts.
inline bool IsDenseBufferTransferDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT32;
}

/// Logical descriptor shape shared by default selection and final lowering.
inline bool IsDenseVecBufferTile(const ir::TileTypePtr& tile, const ir::TileView& view) {
  return tile->shape_.size() == 2 &&
         (IsDenseBufferTransferDtype(tile->dtype_) || tile->dtype_ == DataType::UINT8 ||
          tile->dtype_ == DataType::UINT32) &&
         tile->GetMemorySpace() == ir::MemorySpace::Vec && view.blayout == ir::TileLayout::row_major &&
         view.slayout == ir::TileLayout::none_box && view.fractal == 512 && view.pad == ir::PadValue::null &&
         view.compact == ir::CompactMode::null;
}

/// Row reductions produce a contiguous column vector, not a row-major [R, 1]
/// tile. Keep this separate from ordinary dense Vec support and view recipes.
inline bool IsColumnVectorBufferTile(const ir::TileTypePtr& tile, const ir::TileView& view) {
  if (tile->shape_.size() != 2) return false;
  const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
  const auto cols = ir::As<ir::ConstInt>(tile->shape_[1]);
  return rows && cols && rows->value_ > 0 && rows->value_ % 8 == 0 && cols->value_ == 1 &&
         tile->dtype_ == DataType::FP32 && tile->GetMemorySpace() == ir::MemorySpace::Vec &&
         view.blayout == ir::TileLayout::col_major && view.slayout == ir::TileLayout::none_box &&
         view.fractal == 512 && view.pad == ir::PadValue::null && view.compact == ir::CompactMode::null;
}

inline bool IsColumnVectorBuffer(const ir::BufferTypePtr& type) {
  return type && type->shape_.size() == 2 && type->shape_[0] % 8 == 0 && type->shape_[1] == 1 &&
         type->dtype_ == DataType::FP32 && type->memory_space_ == ir::MemorySpace::Vec &&
         type->blayout_ == ir::TileLayout::col_major && type->slayout_ == ir::TileLayout::none_box &&
         type->fractal_ == 512 && type->pad_ == ir::PadValue::null &&
         type->compact_ == ir::CompactMode::null && type->valid_shape_[0] > 0 && type->valid_shape_[1] == 1;
}

/// Dense allocation descriptors shared by heterogeneous native recipes.
inline bool IsAlignedDenseVecBuffer(const ir::BufferTypePtr& type) {
  const auto element_bytes = type ? type->dtype_.GetBit() / 8 : 0;
  // Check alignment without multiplying a potentially large physical extent.
  return element_bytes > 0 && element_bytes <= 32 && 32 % element_bytes == 0 && type &&
         type->shape_.size() == 2 && type->memory_space_ == ir::MemorySpace::Vec &&
         type->blayout_ == ir::TileLayout::row_major && type->slayout_ == ir::TileLayout::none_box &&
         type->fractal_ == 512 && type->pad_ == ir::PadValue::null &&
         type->compact_ == ir::CompactMode::null && type->valid_shape_[0] > 0 && type->valid_shape_[1] > 0 &&
         type->shape_[1] % (32 / element_bytes) == 0;
}

/// Static positive-valid logical descriptors shared by heterogeneous recipes.
inline ir::BufferTypePtr StaticLogicalBufferDescriptor(const ir::TypePtr& type) {
  const auto tile = ir::As<ir::TileType>(type);
  if (!tile || tile->shape_.size() != 2) return nullptr;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  if (!IsDenseVecBufferTile(tile, view) && !IsColumnVectorBufferTile(tile, view)) return nullptr;
  const auto offset = ir::As<ir::ConstInt>(view.start_offset);
  if (view.start_offset && (!offset || offset->value_ != 0)) return nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> valid;
  const auto& extents = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (extents.size() != 2) return nullptr;
  for (size_t i = 0; i < 2; ++i) {
    const auto dimension = ir::As<ir::ConstInt>(tile->shape_[i]);
    const auto extent = ir::As<ir::ConstInt>(extents[i]);
    if (!dimension || dimension->value_ <= 0 || !extent || extent->value_ <= 0 ||
        extent->value_ > dimension->value_) {
      return nullptr;
    }
    shape.push_back(dimension->value_);
    valid.push_back(extent->value_);
  }
  if (!view.stride.empty()) {
    const std::vector<int64_t> expected = view.blayout == ir::TileLayout::col_major
                                              ? std::vector<int64_t>{1, shape[0]}
                                              : std::vector<int64_t>{shape[1], 1};
    if (view.stride.size() != 2) return nullptr;
    for (size_t i = 0; i < 2; ++i) {
      const auto stride = ir::As<ir::ConstInt>(view.stride[i]);
      if (!stride || stride->value_ != expected[i]) return nullptr;
    }
  }
  return std::make_shared<ir::BufferType>(shape, tile->dtype_, ir::MemorySpace::Vec, valid, view.blayout);
}

/// On-chip spaces whose descriptors carry a cube fractal layout. Their bytes
/// have no row-major byte-view form, so each storage window keeps its own
/// typed descriptor.
inline bool IsMatrixBufferSpace(ir::MemorySpace space) {
  return space == ir::MemorySpace::Mat || space == ir::MemorySpace::Left || space == ir::MemorySpace::Right ||
         space == ir::MemorySpace::Acc;
}

/// Element types a Mat/Left/Right cube operand may hold on every supported target.
inline bool IsCubeOperandDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT8;
}

/// Element types a matrix-space descriptor may hold: cube operands in
/// Mat/Left/Right and the FP32/INT32 accumulators in Acc.
inline bool IsMatrixBufferDtype(ir::MemorySpace space, DataType dtype) {
  if (space == ir::MemorySpace::Acc) return dtype == DataType::FP32 || dtype == DataType::INT32;
  return IsMatrixBufferSpace(space) && IsCubeOperandDtype(dtype);
}

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_
