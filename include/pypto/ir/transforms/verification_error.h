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

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {

/**
 * @brief Unified verification error structure
 *
 * This structure is used by all verification passes to report errors.
 * The error_code field contains the specific error type from different
 * error type enumerations (ssa::ErrorType, typecheck::ErrorType, etc.)
 */
struct VerificationError {
  int error_code;       // Error type code
  std::string message;  // Error message
  Span span;            // Source location
};

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
  TYPE_KIND_MISMATCH = 101,        // Type kind mismatch (e.g., ScalarType vs TensorType)
  DTYPE_MISMATCH = 102,            // Data type mismatch
  SHAPE_DIMENSION_MISMATCH = 103,  // Shape dimension count mismatch
  SHAPE_VALUE_MISMATCH = 104,      // Shape dimension value mismatch
  SIZE_MISMATCH = 105              // Vector size mismatch in control flow
};

/**
 * @brief Convert type check error type to string
 */
std::string ErrorTypeToString(ErrorType type);

}  // namespace typecheck

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_VERIFICATION_ERROR_H_
