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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

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

ShapedType::ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref)
    : dtype_(dtype), memref_(std::move(memref)) {
  for (int64_t dim : shape) {
    shape_.push_back(std::make_shared<ConstInt>(dim, DataType::INDEX, Span::unknown()));
  }
}
}  // namespace ir
}  // namespace pypto
