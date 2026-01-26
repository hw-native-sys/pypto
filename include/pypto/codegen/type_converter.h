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

#ifndef PYPTO_CODEGEN_TYPE_CONVERTER_H_
#define PYPTO_CODEGEN_TYPE_CONVERTER_H_

#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/memref.h"

namespace pypto {
namespace codegen {

/**
 * @brief Utility for converting IR types to pto-isa C++ types
 *
 * TypeConverter handles the translation from PyPTO IR type representations
 * to corresponding pto-isa C++ type strings used in generated code.
 */
class TypeConverter {
 public:
  TypeConverter() = default;

  /**
   * @brief Convert DataType to C++ type string
   *
   * Maps PyPTO DataType to pto-isa C++ types:
   * - FP32 → "float"
   * - FP16 → "half"
   * - INT32 → "int32_t"
   * - INT64 → "int64_t"
   * - BOOL → "bool"
   *
   * @param dtype The PyPTO data type
   * @return C++ type string
   */
  [[nodiscard]] std::string ConvertDataType(const DataType& dtype) const;

  /**
   * @brief Convert MemorySpace to C++ memory space annotation
   *
   * Maps PyPTO MemorySpace to pto-isa annotations:
   * - DDR → "__gm__"
   * - UB → "" (no annotation needed for local tiles)
   * - L0A/L0B/L0C → "" (no annotation needed)
   *
   * @param space The memory space
   * @return C++ memory space annotation (empty string if none needed)
   */
  [[nodiscard]] std::string ConvertMemorySpace(ir::MemorySpace space) const;

  /**
   * @brief Generate Shape type instantiation
   *
   * Converts a shape vector to pto-isa Shape template instantiation.
   * Pads to 5 dimensions with leading 1s.
   *
   * Example: [128, 64] → "Shape<1, 1, 1, 128, 64>"
   *
   * @param dims The shape dimensions (must be constant values)
   * @return Shape type string
   */
  [[nodiscard]] std::string GenerateShapeType(const std::vector<int64_t>& dims) const;

  /**
   * @brief Generate Stride type instantiation for row-major layout
   *
   * Converts a shape vector to pto-isa Stride template instantiation.
   * Calculates row-major strides and pads to 5 dimensions.
   *
   * Example: [128, 64] → "Stride<1, 1, 1, 64, 1>"
   *
   * @param shape The shape dimensions (used to calculate strides)
   * @return Stride type string
   */
  [[nodiscard]] std::string GenerateStrideType(const std::vector<int64_t>& shape) const;

 private:
  /**
   * @brief Calculate row-major strides from shape
   *
   * Stride[i] = product of all dimensions after i
   *
   * @param shape The shape dimensions
   * @return Vector of stride values
   */
  [[nodiscard]] std::vector<int64_t> CalculateRowMajorStrides(const std::vector<int64_t>& shape) const;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_TYPE_CONVERTER_H_
