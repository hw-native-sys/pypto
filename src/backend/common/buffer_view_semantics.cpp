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

#include "pypto/backend/common/buffer_view_semantics.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

std::optional<uint64_t> DenseBufferBytes(const ir::BufferTypePtr& type) {
  if (!type || type->blayout_ != ir::TileLayout::row_major || type->slayout_ != ir::TileLayout::none_box ||
      type->fractal_ != 512 || type->pad_ != ir::PadValue::null || type->compact_ != ir::CompactMode::null) {
    return std::nullopt;
  }
  return PhysicalBufferBytes(type);
}

std::optional<uint64_t> PhysicalBufferBytes(const ir::BufferTypePtr& type) {
  if (!type || (type->shape_.size() != 1 && type->shape_.size() != 2) || type->dtype_.GetBit() == 0 ||
      type->dtype_.GetBit() % 8 != 0) {
    return std::nullopt;
  }
  uint64_t bytes = type->dtype_.GetBit() / 8;
  for (const auto extent : type->shape_) {
    if (extent <= 0 || static_cast<uint64_t>(extent) > std::numeric_limits<uint64_t>::max() / bytes) {
      return std::nullopt;
    }
    bytes *= static_cast<uint64_t>(extent);
  }
  return bytes;
}

namespace {

ir::BufferTypePtr StaticViewDescriptor(const ir::TypePtr& type, const std::string& name) {
  auto buffer = ir::As<ir::BufferType>(type);
  const auto bytes = DenseBufferBytes(buffer);
  CHECK(buffer && buffer->shape_.size() == 2 && buffer->memory_space_ == ir::MemorySpace::Vec &&
        (IsDenseBufferTransferDtype(buffer->dtype_) || buffer->dtype_ == DataType::INT16 ||
         buffer->dtype_ == DataType::UINT8) &&
        bytes.has_value())
      << name << " requires dense rank-2 row-major Vec FP16/BF16/FP32/INT16/INT32/UINT8 descriptors";
  for (const auto extent : buffer->valid_shape_) {
    CHECK(extent >= 0) << name << " requires static valid extents";
  }
  const auto row_bytes = *bytes / static_cast<uint64_t>(buffer->shape_[0]);
  CHECK(row_bytes % 32 == 0) << name << " requires physical rows aligned to 32 bytes";
  return buffer;
}

// A fractal window is never sliced or re-pitched; the only static alias is a
// relabel of the whole window in the same space (for example NZ <-> ZN).
void ValidateMatrixReshape(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination) {
  CHECK(destination && destination->memory_space_ == source->memory_space_)
      << "buffer.reshape of a " << ir::MemorySpaceToString(source->memory_space_)
      << " buffer must stay in the same memory space";
  CHECK(source->shape_.size() == 2 && destination->shape_.size() == 2 &&
        destination->dtype_ == source->dtype_)
      << "buffer.reshape of a matrix-space buffer requires rank-2 descriptors with the same element type";
  const auto bytes = PhysicalBufferBytes(source);
  CHECK(bytes && bytes == PhysicalBufferBytes(destination))
      << "buffer.reshape requires equal physical byte sizes";
  for (const auto& type : {source, destination}) {
    for (const auto extent : type->valid_shape_) {
      CHECK(extent >= 0) << "buffer.reshape of a matrix-space buffer requires static valid extents";
    }
  }
}

}  // namespace

// A strided window of a dense row-major Vec buffer: the result keeps the
// source's row pitch, so any static-shape window fits, at static or runtime
// offsets. Static byte windows of the UINT8[N,32] storage root are the
// full-width special case. A result valid dimension is either static in the
// descriptor or dynamic (-1); the optional valid tuple states every dimension
// and is required when any is dynamic.
void ValidateBufferSubview(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK((args.size() == 2 || args.size() == 3) && args[0] && args[1])
      << "Internal error: buffer.subview requires a source, an offsets tuple and optional valid extents";
  auto source = ir::As<ir::BufferType>(args[0]->GetType());
  auto destination = ir::As<ir::BufferType>(result);
  for (const auto& [type, role] : {std::pair{source, "source"}, std::pair{destination, "result"}}) {
    CHECK(type && type->shape_.size() == 2 && type->memory_space_ == ir::MemorySpace::Vec &&
          DenseBufferBytes(type).has_value())
        << "buffer.subview " << role << " must be a dense rank-2 row-major Vec buffer";
  }
  CHECK(source->dtype_ == destination->dtype_)
      << "buffer.subview must keep the element type, got " << source->dtype_.ToString() << " -> "
      << destination->dtype_.ToString();
  auto offsets = ir::As<ir::MakeTuple>(args[1]);
  CHECK(offsets && offsets->elements_.size() == 2) << "buffer.subview offsets must be a rank-2 MakeTuple";
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto& offset = offsets->elements_[axis];
    auto scalar = offset ? ir::As<ir::ScalarType>(offset->GetType()) : nullptr;
    CHECK(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX))
        << "buffer.subview offset " << axis << " must be an integer or INDEX scalar";
    const auto extent = destination->shape_[axis];
    CHECK(extent <= source->shape_[axis]) << "buffer.subview window dimension " << axis << " (" << extent
                                          << ") exceeds the source (" << source->shape_[axis] << ")";
    if (auto constant = ir::As<ir::ConstInt>(offset)) {
      CHECK(constant->value_ >= 0 && constant->value_ <= source->shape_[axis] - extent)
          << "buffer.subview window exceeds source capacity on dimension " << axis;
    }
  }
  const bool dynamic = destination->valid_shape_[0] < 0 || destination->valid_shape_[1] < 0;
  CHECK(!dynamic || args.size() == 3)
      << "buffer.subview requires a valid-extents tuple when its result has a dynamic valid dimension";
  if (args.size() == 3) {
    auto valid = ir::As<ir::MakeTuple>(args[2]);
    CHECK(valid && valid->elements_.size() == 2) << "buffer.subview valid extents must be a rank-2 MakeTuple";
    for (size_t axis = 0; axis < 2; ++axis) {
      const auto& extent = valid->elements_[axis];
      auto scalar = extent ? ir::As<ir::ScalarType>(extent->GetType()) : nullptr;
      CHECK(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX))
          << "buffer.subview valid extent " << axis << " must be an integer or INDEX scalar";
      if (destination->valid_shape_[axis] >= 0) {
        auto constant = ir::As<ir::ConstInt>(extent);
        CHECK(constant && constant->value_ == destination->valid_shape_[axis])
            << "buffer.subview valid extent " << axis << " must equal the static descriptor dimension";
      }
    }
  }
}

void ValidateBufferReshape(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK(args.size() == 1 && args[0]) << "Internal error: buffer.reshape requires exactly one source";
  auto matrix_source = ir::As<ir::BufferType>(args[0]->GetType());
  if (matrix_source && IsMatrixBufferSpace(matrix_source->memory_space_)) {
    ValidateMatrixReshape(matrix_source, ir::As<ir::BufferType>(result));
    return;
  }
  auto source = StaticViewDescriptor(args[0]->GetType(), "buffer.reshape");
  auto destination = StaticViewDescriptor(result, "buffer.reshape");
  CHECK(DenseBufferBytes(source) == DenseBufferBytes(destination))
      << "buffer.reshape requires equal physical byte sizes";
}

}  // namespace pypto::backend
