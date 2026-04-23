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

#include "pypto/ir/type.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

namespace {

std::optional<MemorySpace> ValidateTileMemorySpaceConsistency(const std::optional<MemRefPtr>& memref,
                                                              std::optional<MemorySpace> memory_space) {
  if (!memref.has_value()) {
    return memory_space;
  }

  const auto& memref_ptr = memref.value();
  CHECK(memref_ptr != nullptr) << "TileType memref must not be null";
  CHECK(memory_space.has_value()) << "TileType with MemRef must have explicit memory_space";

  return memory_space;
}

}  // namespace

bool operator==(const TileView& lhs, const TileView& rhs) {
  return AreExprVectorsEqual(lhs.valid_shape, rhs.valid_shape) &&
         AreExprVectorsEqual(lhs.stride, rhs.stride) && AreExprsEqual(lhs.start_offset, rhs.start_offset) &&
         lhs.blayout == rhs.blayout && lhs.slayout == rhs.slayout && lhs.fractal == rhs.fractal &&
         lhs.pad == rhs.pad;
}

bool operator!=(const TileView& lhs, const TileView& rhs) { return !(lhs == rhs); }

ShapedType::ShapedType(DataType dtype, std::vector<ExprPtr> shape)
    : dtype_(dtype), shape_(std::move(shape)), memref_(std::nullopt) {}

std::string TensorLayoutToString(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::ND:
      return "ND";
    case TensorLayout::DN:
      return "DN";
    case TensorLayout::NZ:
      return "NZ";
    default:
      throw TypeError("Unknown TensorLayout value: " + std::to_string(static_cast<int>(layout)));
  }
}

TensorLayout StringToTensorLayout(const std::string& str) {
  if (str == "ND") {
    return TensorLayout::ND;
  } else if (str == "DN") {
    return TensorLayout::DN;
  } else if (str == "NZ") {
    return TensorLayout::NZ;
  }
  throw TypeError("Unknown TensorLayout string: " + str);
}

std::string TileLayoutToString(TileLayout layout) {
  switch (layout) {
    case TileLayout::none_box:
      return "none_box";
    case TileLayout::row_major:
      return "row_major";
    case TileLayout::col_major:
      return "col_major";
    default:
      throw TypeError("Unknown TileLayout value: " + std::to_string(static_cast<int>(layout)));
  }
}

TileLayout StringToTileLayout(const std::string& str) {
  if (str == "none_box") {
    return TileLayout::none_box;
  } else if (str == "row_major") {
    return TileLayout::row_major;
  } else if (str == "col_major") {
    return TileLayout::col_major;
  }
  throw TypeError("Unknown TileLayout string: " + str);
}

ShapedType::ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref)
    : dtype_(dtype), memref_(std::move(memref)) {
  for (int64_t dim : shape) {
    shape_.push_back(std::make_shared<ConstInt>(dim, DataType::INDEX, Span::unknown()));
  }
}

TensorView::TensorView(const std::vector<int64_t>& stride_ints, TensorLayout layout_,
                       const std::vector<int64_t>& valid_shape_ints, PadValue pad_)
    : layout(layout_), pad(pad_) {
  for (int64_t s : stride_ints) {
    stride.push_back(std::make_shared<ConstInt>(s, DataType::INDEX, Span::unknown()));
  }
  for (int64_t v : valid_shape_ints) {
    valid_shape.push_back(std::make_shared<ConstInt>(v, DataType::INDEX, Span::unknown()));
  }
}

TileView::TileView(const std::vector<int64_t>& valid_shape_ints, const std::vector<int64_t>& stride_ints,
                   ExprPtr start_offset_, TileLayout blayout_, TileLayout slayout_, uint64_t fractal_,
                   PadValue pad_)
    : start_offset(std::move(start_offset_)),
      blayout(blayout_),
      slayout(slayout_),
      fractal(fractal_),
      pad(pad_) {
  for (int64_t v : valid_shape_ints) {
    valid_shape.push_back(std::make_shared<ConstInt>(v, DataType::INDEX, Span::unknown()));
  }
  for (int64_t s : stride_ints) {
    stride.push_back(std::make_shared<ConstInt>(s, DataType::INDEX, Span::unknown()));
  }
}

ShapedType::ShapedType(DataType dtype, std::vector<ExprPtr> shape, MemRefPtr memref)
    : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref)) {}

ShapedType::ShapedType(DataType dtype, std::vector<ExprPtr> shape, std::optional<MemRefPtr> memref)
    : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref)) {}

TileType::TileType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
                   std::optional<TileView> tile_view, std::optional<MemorySpace> memory_space)
    : ShapedType(dtype, shape, std::move(memref)),
      tile_view_(std::move(tile_view)),
      memory_space_(ValidateTileMemorySpaceConsistency(memref_, memory_space)) {}

TileType::TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
                   std::optional<TileView> tile_view, std::optional<MemorySpace> memory_space)
    : ShapedType(dtype, std::move(shape), std::move(memref)),
      tile_view_(std::move(tile_view)),
      memory_space_(ValidateTileMemorySpaceConsistency(memref_, memory_space)) {}

std::optional<MemorySpace> TileType::GetMemorySpace() const { return memory_space_; }

std::optional<MemorySpace> TileType::ValidateMemorySpace(const std::optional<MemRefPtr>& memref,
                                                         std::optional<MemorySpace> memory_space) {
  return ValidateTileMemorySpaceConsistency(memref, memory_space);
}
}  // namespace ir
}  // namespace pypto
