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

#ifndef SRC_IR_OP_SORT_UTILS_H_
#define SRC_IR_OP_SORT_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/// TSORT32 stores each value-index pair in 8 bytes. Expressed in source
/// elements, that consumes two FP32 slots or four FP16 slots per input value.
inline int64_t GetSort32OutputWidthFactor(DataType dtype) {
  CHECK(dtype == DataType::FP16 || dtype == DataType::FP32)
      << "sort32 output width requires FP16 or FP32 input, but got " << dtype.ToString();
  return dtype == DataType::FP16 ? 4 : 2;
}

/// Scale a shape's final dimension by sort32's dtype-dependent output factor.
/// Constants stay constants so capacity inference and PTO type rendering can
/// see the physical extent; symbolic valid extents remain runtime expressions.
inline std::vector<ExprPtr> GetSort32OutputShape(const std::vector<ExprPtr>& input_shape, DataType dtype,
                                                 const Span& span) {
  CHECK(!input_shape.empty()) << "sort32 requires a non-empty input shape";
  const int64_t factor = GetSort32OutputWidthFactor(dtype);
  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  if (auto constant = As<ConstInt>(input_shape.back())) {
    output_shape.push_back(std::make_shared<ConstInt>(constant->value_ * factor, DataType::INDEX, span));
  } else {
    auto factor_expr = std::make_shared<ConstInt>(factor, DataType::INDEX, span);
    output_shape.push_back(std::make_shared<Mul>(input_shape.back(), factor_expr, DataType::INDEX, span));
  }
  return output_shape;
}

}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_OP_SORT_UTILS_H_
