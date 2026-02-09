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

#ifndef PYPTO_IR_TRANSFORMS_VERIFICATION_ERROR_H_
#define PYPTO_IR_TRANSFORMS_VERIFICATION_ERROR_H_

#include <string>

#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/**
 * @brief SSA verification error types and utilities
 */
namespace ssa {

/**
 * @brief Error types for SSA verification
 */
enum class ErrorType : int {
  MULTIPLE_ASSIGNMENT = 1,  // Variable assigned more than once
  NAME_SHADOWING = 2,       // Variable name shadows outer scope variable
  MISSING_YIELD = 3         // ForStmt or IfStmt missing required YieldStmt
};

/**
 * @brief Convert SSA error type to string
 */
std::string ErrorTypeToString(ErrorType type);

}  // namespace ssa

/**
 * @brief Type checking error types and utilities
 */
namespace typecheck {

/**
 * @brief Error types for type checking
 */
enum class ErrorType : int {
  TYPE_KIND_MISMATCH = 101,           // Type kind mismatch (e.g., ScalarType vs TensorType)
  DTYPE_MISMATCH = 102,               // Data type mismatch
  SHAPE_DIMENSION_MISMATCH = 103,     // Shape dimension count mismatch
  SHAPE_VALUE_MISMATCH = 104,         // Shape dimension value mismatch
  SIZE_MISMATCH = 105,                // Vector size mismatch in control flow
  IF_CONDITION_MUST_BE_SCALAR = 106,  // IfStmt condition must be ScalarType
  FOR_RANGE_MUST_BE_SCALAR = 107      // ForStmt range must be ScalarType
};

/**
 * @brief Convert type check error type to string
 */
std::string ErrorTypeToString(ErrorType type);

}  // namespace typecheck

/**
 * @brief Nested call verification error types and utilities
 */
namespace nested_call {

/**
 * @brief Error types for nested call verification
 */
enum class ErrorType : int {
  CALL_IN_CALL_ARGS = 201,     // Call expression appears in call arguments
  CALL_IN_IF_CONDITION = 202,  // Call expression appears in if condition
  CALL_IN_FOR_RANGE = 203,     // Call expression appears in for range (start/stop/step)
  CALL_IN_BINARY_EXPR = 204,   // Call expression appears in binary expression operands
  CALL_IN_UNARY_EXPR = 205     // Call expression appears in unary expression operand
};

/**
 * @brief Convert nested call error type to string
 */
std::string ErrorTypeToString(ErrorType type);

}  // namespace nested_call

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_VERIFICATION_ERROR_H_
