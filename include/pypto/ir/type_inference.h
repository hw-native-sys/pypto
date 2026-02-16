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

/**
 * @file type_inference.h
 * @brief Type inference utilities for operator type deduction
 *
 * This file provides utilities for automatic type deduction in operator
 * registration, including broadcasting shape inference, data type promotion,
 * and type compatibility checking.
 */

#ifndef PYPTO_IR_TYPE_INFERENCE_H_
#define PYPTO_IR_TYPE_INFERENCE_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Result of shape broadcasting
 *
 * Contains the broadcast result shape or an error message if broadcasting fails.
 */
struct BroadcastResult {
  bool success;                // Whether broadcasting succeeded
  std::vector<ExprPtr> shape;  // Resulting broadcast shape (empty if failed)
  std::string error_message;   // Error message if broadcasting failed

  /**
   * @brief Create a successful broadcast result
   */
  static BroadcastResult Success(std::vector<ExprPtr> result_shape) {
    return BroadcastResult{true, std::move(result_shape), ""};
  }

  /**
   * @brief Create a failed broadcast result with error message
   */
  static BroadcastResult Failure(std::string message) {
    return BroadcastResult{false, {}, std::move(message)};
  }
};

/**
 * @brief Broadcast two shapes following NumPy-style broadcasting rules
 *
 * Broadcasting rules:
 * - Dimensions are aligned from right to left
 * - Size 1 dimensions are broadcast to match the other operand
 * - Missing dimensions are treated as size 1
 * - If dimensions don't match and neither is 1, broadcasting fails
 *
 * Examples:
 * - [4, 8] + [4, 8] -> [4, 8]
 * - [4, 8] + [8] -> [4, 8]
 * - [4, 1] + [8] -> [4, 8]
 * - [4, 8] + [5] -> Error (8 != 5)
 *
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return BroadcastResult with the resulting shape or error
 */
BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2);

/**
 * @brief Promote two data types to a common type
 *
 * Type promotion rules follow standard numeric promotion:
 * - If types are the same, return that type
 * - Float types take precedence over integer types
 * - Larger types take precedence over smaller types
 * - Signed types take precedence over unsigned types of the same size
 *
 * Examples:
 * - INT32 + INT32 -> INT32
 * - INT32 + FP32 -> FP32
 * - INT32 + INT64 -> INT64
 * - UINT32 + INT32 -> INT32
 *
 * @param dtype1 First data type
 * @param dtype2 Second data type
 * @return Promoted data type, or std::nullopt if types are incompatible
 */
std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2);

/**
 * @brief Check if two types are compatible for binary operations
 *
 * Types are compatible if:
 * - Both are scalar types
 * - Both are tensor types (shapes may differ for broadcasting)
 * - Both are tile types (shapes may differ for broadcasting)
 *
 * @param type1 First type
 * @param type2 Second type
 * @return true if types are compatible
 */
bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2);

/**
 * @brief Extract data type from a type pointer
 *
 * Works for ScalarType, TensorType, and TileType.
 *
 * @param type Type pointer
 * @return Data type, or std::nullopt if type is not a scalar/tensor/tile type
 */
std::optional<DataType> ExtractDataType(const TypePtr& type);

/**
 * @brief Extract shape from a tensor or tile type
 *
 * @param type Type pointer
 * @return Shape vector, or empty vector if type is not a tensor/tile type
 */
std::vector<ExprPtr> ExtractShape(const TypePtr& type);

/**
 * @brief Check if a dimension expression represents a constant value
 *
 * @param dim Dimension expression
 * @return std::optional with the constant value, or std::nullopt if not constant
 */
std::optional<int64_t> GetConstantDimension(const ExprPtr& dim);

/**
 * @brief Check if two dimension expressions are equal
 *
 * Handles both constant and symbolic dimensions.
 * For constant dimensions, compares values.
 * For symbolic dimensions, uses structural equality.
 *
 * @param dim1 First dimension
 * @param dim2 Second dimension
 * @return true if dimensions are equal
 */
bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2);

/**
 * @brief Check if a dimension is broadcastable to another
 *
 * A dimension is broadcastable if:
 * - It's equal to the target dimension
 * - It's a constant 1
 * - The target dimension is a constant 1
 *
 * @param source_dim Source dimension
 * @param target_dim Target dimension
 * @return true if source can be broadcast to target
 */
bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim);

/**
 * @brief Format a shape vector as a string for error messages
 *
 * Converts a shape (vector of ExprPtr) to a human-readable string.
 * Constant dimensions show their values, symbolic dimensions show as "?".
 *
 * Examples:
 * - [ConstInt(64), ConstInt(128)] -> "[64, 128]"
 * - [ConstInt(64), Var("N")] -> "[64, ?]"
 * - [] -> "[]"
 *
 * @param shape Shape vector to format
 * @return String representation of the shape
 */
std::string FormatShape(const std::vector<ExprPtr>& shape);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TYPE_INFERENCE_H_
