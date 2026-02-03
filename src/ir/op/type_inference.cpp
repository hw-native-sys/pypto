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

#include "pypto/ir/type_inference.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2) {
  // Handle empty shapes
  if (shape1.empty() && shape2.empty()) {
    return BroadcastResult::Success({});
  }
  if (shape1.empty()) {
    return BroadcastResult::Success(shape2);
  }
  if (shape2.empty()) {
    return BroadcastResult::Success(shape1);
  }

  // Broadcast from right to left
  size_t max_ndim = std::max(shape1.size(), shape2.size());
  std::vector<ExprPtr> result_shape;
  result_shape.reserve(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    // Get dimensions from right to left
    int64_t idx1 = static_cast<int64_t>(shape1.size()) - 1 - i;  // NOLINT
    int64_t idx2 = static_cast<int64_t>(shape2.size()) - 1 - i;  // NOLINT

    ExprPtr dim1 = (idx1 >= 0) ? shape1[idx1] : nullptr;
    ExprPtr dim2 = (idx2 >= 0) ? shape2[idx2] : nullptr;

    // If one dimension is missing, use the other
    if (!dim1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (!dim2) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if dimensions are equal
    if (DimensionsEqual(dim1, dim2)) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if either dimension is 1 (broadcastable)
    auto const_dim1 = GetConstantDimension(dim1);
    auto const_dim2 = GetConstantDimension(dim2);

    if (const_dim1 && *const_dim1 == 1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (const_dim2 && *const_dim2 == 1) {
      result_shape.push_back(dim1);
      continue;
    }

    // Dimensions are incompatible for broadcasting
    std::ostringstream oss;
    oss << "Cannot broadcast shapes: dimension " << i << " mismatch";
    return BroadcastResult::Failure(oss.str());
  }

  // Reverse result since we built it from right to left
  std::reverse(result_shape.begin(), result_shape.end());
  return BroadcastResult::Success(std::move(result_shape));
}

std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2) {
  // If types are the same, return that type
  if (dtype1 == dtype2) {
    return dtype1;
  }

  // Float types take precedence
  bool is_float1 = dtype1.IsFloat();
  bool is_float2 = dtype2.IsFloat();

  if (is_float1 && !is_float2) {
    return dtype1;
  }
  if (is_float2 && !is_float1) {
    return dtype2;
  }

  // Both are floats or both are integers
  // Return the larger type
  size_t bits1 = dtype1.GetBit();
  size_t bits2 = dtype2.GetBit();

  if (bits1 > bits2) {
    return dtype1;
  }
  if (bits2 > bits1) {
    return dtype2;
  }

  // Same size - prefer signed over unsigned for integers
  if (!is_float1 && dtype1.IsSignedInt()) {
    return dtype1;
  }
  if (!is_float2 && dtype2.IsSignedInt()) {
    return dtype2;
  }

  // Default to first type
  return dtype1;
}

bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2) {
  // Check if both are scalar types
  auto scalar1 = As<ScalarType>(type1);
  auto scalar2 = As<ScalarType>(type2);
  if (scalar1 && scalar2) {
    return true;
  }

  // Check if both are tensor types
  auto tensor1 = As<TensorType>(type1);
  auto tensor2 = As<TensorType>(type2);
  if (tensor1 && tensor2) {
    return true;
  }

  // Check if both are tile types
  auto tile1 = As<TileType>(type1);
  auto tile2 = As<TileType>(type2);
  if (tile1 && tile2) {
    return true;
  }

  // Types are not compatible
  return false;
}

std::optional<DataType> ExtractDataType(const TypePtr& type) {
  // Try ScalarType
  if (auto scalar = As<ScalarType>(type)) {
    return scalar->dtype_;
  }

  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->dtype_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->dtype_;
  }

  return std::nullopt;
}

std::vector<ExprPtr> ExtractShape(const TypePtr& type) {
  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->shape_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->shape_;
  }

  // Not a shaped type
  return {};
}

std::optional<int64_t> GetConstantDimension(const ExprPtr& dim) {
  // Try to cast to ConstInt
  if (auto const_int = As<ConstInt>(dim)) {
    return const_int->value_;
  }

  // Not a constant
  return std::nullopt;
}

bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2) {
  // Pointer equality (same object)
  if (dim1 == dim2) {
    return true;
  }

  // Try constant comparison
  auto const1 = GetConstantDimension(dim1);
  auto const2 = GetConstantDimension(dim2);

  if (const1 && const2) {
    return *const1 == *const2;
  }

  // For symbolic dimensions, use pointer equality
  return false;
}

bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim) {
  // If dimensions are equal, they're broadcastable
  if (DimensionsEqual(source_dim, target_dim)) {
    return true;
  }

  // Check if source is constant 1
  auto const_source = GetConstantDimension(source_dim);
  if (const_source && *const_source == 1) {
    return true;
  }

  // Check if target is constant 1
  auto const_target = GetConstantDimension(target_dim);
  if (const_target && *const_target == 1) {
    return true;
  }

  return false;
}

std::string FormatShape(const std::vector<ExprPtr>& shape) {
  if (shape.empty()) {
    return "[]";
  }

  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    auto const_dim = GetConstantDimension(shape[i]);
    if (const_dim) {
      oss << *const_dim;
    } else {
      oss << "?";
    }
  }
  oss << "]";
  return oss.str();
}

}  // namespace ir
}  // namespace pypto
