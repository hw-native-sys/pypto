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

#include "pypto/codegen/cce/type_converter.h"

#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"

namespace pypto {

namespace codegen {

std::string TypeConverter::ConvertTileType(const ir::TileTypePtr& tile_type, int64_t rows,
                                           int64_t cols) const {
  std::ostringstream type_alias;
  if (!tile_type->memref_.has_value()) {
    type_alias << "Tile<TileType::Vec, " << tile_type->dtype_.ToCTypeString() << ", " << rows << ", " << cols
               << ", BLayout::RowMajor, -1, -1>;";
    LOG_ERROR << "TileType has no memref, using default TileType::Vec";
    return type_alias.str();
  }
  ir::MemorySpace space = tile_type->memref_.value()->memory_space_;
  std::string tile_type_str = ConvertMemorySpaceToTileType(space);
  type_alias << "Tile<" << tile_type_str << ", " << tile_type->dtype_.ToCTypeString() << ", " << rows << ", "
             << cols;
  if (space == ir::MemorySpace::L0C) {
    type_alias << ", BLayout::ColMajor, -1, -1, SLayout::RowMajor>;";
  } else {
    type_alias << ", BLayout::RowMajor, -1, -1>;";
  }
  return type_alias.str();
}

std::string TypeConverter::ConvertMemorySpaceToTileType(ir::MemorySpace space) const {
  switch (space) {
    case ir::MemorySpace::L0A:
      return "TileType::Left";
    case ir::MemorySpace::L0B:
      return "TileType::Right";
    case ir::MemorySpace::L0C:
      return "TileType::Acc";
    case ir::MemorySpace::L1:
      return "TileType::Mat";
    case ir::MemorySpace::UB:
      return "TileType::Vec";
    case ir::MemorySpace::DDR:
      // DDR is for GlobalTensor, not Tile - should not reach here
      throw pypto::ValueError("DDR is for GlobalTensor, not Tile");
    default:
      throw pypto::ValueError("Invalid MemorySpace value");
  }
}

std::string TypeConverter::ConvertPipeType(ir::PipeType pipe) const {
  switch (pipe) {
    case ir::PipeType::MTE1:
      return "PIPE_MTE1";
    case ir::PipeType::MTE2:
      return "PIPE_MTE2";
    case ir::PipeType::MTE3:
      return "PIPE_MTE3";
    case ir::PipeType::M:
      return "PIPE_M";
    case ir::PipeType::V:
      return "PIPE_V";
    case ir::PipeType::S:
      return "PIPE_S";
    case ir::PipeType::FIX:
      return "PIPE_FIX";
    case ir::PipeType::ALL:
      return "PIPE_ALL";
    default:
      throw pypto::ValueError("Invalid PipeType value");
  }
}

std::string TypeConverter::ConvertEventId(int event_id) const {
  CHECK(event_id >= 0 && event_id <= 7) << "Event ID must be in range [0, 7], got " << event_id;
  return "EVENT_ID" + std::to_string(event_id);
}

std::string TypeConverter::GenerateShapeType(const std::vector<int64_t>& dims) const {
  CHECK(!dims.empty()) << "Cannot generate Shape type for empty dimensions";

  std::ostringstream oss;
  oss << "Shape<";

  // Pad to 5 dimensions with leading 1s
  const size_t target_dims = 5;
  CHECK(dims.size() <= target_dims) << "Cannot generate Shape with more than " << target_dims
                                    << " dimensions, got " << dims.size();

  // Add leading 1s for padding
  for (size_t i = 0; i < target_dims - dims.size(); ++i) {
    oss << "1, ";
  }

  // Add actual dimensions
  for (size_t i = 0; i < dims.size(); ++i) {
    oss << dims[i];
    if (i < dims.size() - 1) {
      oss << ", ";
    }
  }

  oss << ">";
  return oss.str();
}

std::string TypeConverter::GenerateStrideType(const std::vector<int64_t>& shape) const {
  CHECK(!shape.empty()) << "Cannot generate Stride type for empty shape";

  auto strides = CalculateRowMajorStrides(shape);

  std::ostringstream oss;
  oss << "Stride<";

  // Pad to 5 dimensions with leading 1s
  const size_t target_dims = 5;
  CHECK(strides.size() <= target_dims)
      << "Cannot generate Stride with more than " << target_dims << " dimensions, got " << strides.size();

  // Add leading 1s for padding
  for (size_t i = 0; i < target_dims - strides.size(); ++i) {
    oss << "1, ";
  }

  // Add actual strides
  for (size_t i = 0; i < strides.size(); ++i) {
    oss << strides[i];
    if (i < strides.size() - 1) {
      oss << ", ";
    }
  }

  oss << ">";
  return oss.str();
}

std::vector<int64_t> TypeConverter::CalculateRowMajorStrides(const std::vector<int64_t>& shape) const {
  CHECK(!shape.empty()) << "Cannot calculate strides for empty shape";

  std::vector<int64_t> strides(shape.size());

  // Stride[i] = product of all dimensions after i
  // Last dimension has stride 1
  strides[shape.size() - 1] = 1;

  // Calculate remaining strides from right to left
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

}  // namespace codegen

}  // namespace pypto
