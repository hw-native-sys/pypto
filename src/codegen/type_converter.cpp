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

#include "pypto/codegen/type_converter.h"

#include <sstream>

#include "pypto/core/logging.h"

namespace pypto {

namespace codegen {

std::string TypeConverter::ConvertDataType(const DataType& dtype) const {
  if (dtype == DataType::FP32) {
    return "float";
  }
  if (dtype == DataType::FP16) {
    return "half";
  }
  if (dtype == DataType::INT32) {
    return "int32_t";
  }
  if (dtype == DataType::INT64) {
    return "int64_t";
  }
  if (dtype == DataType::BOOL) {
    return "bool";
  }
  if (dtype == DataType::BF16) {
    return "bfloat16";
  }
  if (dtype == DataType::UINT32) {
    return "uint32_t";
  }
  if (dtype == DataType::UINT64) {
    return "uint64_t";
  }

  // Unsupported type
  throw pypto::ValueError("Unsupported DataType for code generation: " + dtype.ToString());
}

std::string TypeConverter::ConvertMemorySpace(ir::MemorySpace space) const {
  switch (space) {
    case ir::MemorySpace::DDR:
      return "__gm__";
    case ir::MemorySpace::UB:
    case ir::MemorySpace::L1:
    case ir::MemorySpace::L0A:
    case ir::MemorySpace::L0B:
    case ir::MemorySpace::L0C:
      // No annotation needed for on-chip memory
      return "";
  }
  // Should never reach here with a valid enum
  return "";
}

std::string TypeConverter::GenerateShapeType(const std::vector<int64_t>& dims) const {
  CHECK(!dims.empty()) << "Cannot generate Shape type for empty dimensions";

  std::ostringstream oss;
  oss << "Shape<";

  // Pad to 5 dimensions with leading 1s
  const size_t target_dims = 5;
  CHECK(dims.size() <= target_dims)
      << "Cannot generate Shape with more than " << target_dims << " dimensions, got " << dims.size();

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
