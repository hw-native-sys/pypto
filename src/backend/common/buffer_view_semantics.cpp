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

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
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
  if (!type || (type->shape_.size() != 1 && type->shape_.size() != 2) ||
      type->blayout_ != ir::TileLayout::row_major || type->slayout_ != ir::TileLayout::none_box ||
      type->fractal_ != 512 || type->pad_ != ir::PadValue::null || type->compact_ != ir::CompactMode::null ||
      type->dtype_.GetBit() == 0 || type->dtype_.GetBit() % 8 != 0) {
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

}  // namespace

void ValidateBufferSubview(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK(args.size() == 2 && args[0] && args[1])
      << "Internal error: buffer.subview requires a source and offsets tuple";
  auto source = StaticViewDescriptor(args[0]->GetType(), "buffer.subview");
  auto destination = StaticViewDescriptor(result, "buffer.subview");
  CHECK(source->dtype_ == DataType::UINT8 && destination->dtype_ == DataType::UINT8 &&
        source->shape_[1] == 32 && destination->shape_[1] == 32 && source->valid_shape_ == source->shape_ &&
        destination->valid_shape_ == destination->shape_)
      << "buffer.subview currently requires full-valid UINT8[N,32] source and result descriptors";
  auto offsets = ir::As<ir::MakeTuple>(args[1]);
  CHECK(offsets && offsets->elements_.size() == 2) << "buffer.subview offsets must be a rank-2 MakeTuple";
  auto row = ir::As<ir::ConstInt>(offsets->elements_[0]);
  auto col = ir::As<ir::ConstInt>(offsets->elements_[1]);
  CHECK(row && col && row->dtype() == DataType::INDEX && col->dtype() == DataType::INDEX &&
        row->value_ >= 0 && col->value_ == 0)
      << "buffer.subview requires static INDEX offsets (nonnegative row, zero column)";
  CHECK(row->value_ <= source->shape_[0] && destination->shape_[0] <= source->shape_[0] - row->value_)
      << "buffer.subview window exceeds source capacity";
}

void ValidateBufferReshape(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK(args.size() == 1 && args[0]) << "Internal error: buffer.reshape requires exactly one source";
  auto source = StaticViewDescriptor(args[0]->GetType(), "buffer.reshape");
  auto destination = StaticViewDescriptor(result, "buffer.reshape");
  CHECK(DenseBufferBytes(source) == DenseBufferBytes(destination))
      << "buffer.reshape requires equal physical byte sizes";
}

}  // namespace pypto::backend
