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

#ifndef PYPTO_IR_TRANSFORM_PRINTER_H_
#define PYPTO_IR_TRANSFORM_PRINTER_H_

#include <sstream>
#include <string>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

/**
 * @brief Operator precedence levels
 *
 * Based on Python operator precedence.
 * Higher value = tighter binding (higher precedence).
 */
enum class Precedence : int {
  kOr = 1,          // or
  kXor = 2,         // xor
  kAnd = 3,         // and
  kNot = 4,         // not (unary)
  kComparison = 5,  // ==, !=, <, <=, >, >=
  kBitOr = 6,       // |
  kBitXor = 7,      // ^
  kBitAnd = 8,      // &
  kBitShift = 9,    // <<, >>
  kAddSub = 10,     // +, -
  kMulDivMod = 11,  // *, /, //, %
  kUnary = 12,      // -(unary), ~
  kPow = 13,        // ** (right-associative!)
  kCall = 14,       // function calls, min(), max(), abs()
  kAtom = 15        // variables, constants
};

/**
 * @brief Get operator precedence for an expression
 *
 * @param expr Expression to get precedence for
 * @return Precedence level
 */
Precedence GetPrecedence(const ExprPtr& expr);

/**
 * @brief Check if operator is right-associative
 *
 * @param expr Expression to check
 * @return true if right-associative, false if left-associative
 */
bool IsRightAssociative(const ExprPtr& expr);

/**
 * @brief Print an IR node in Python syntax
 *
 * @param node IR node to print (Expr, Stmt, Function, or Program)
 * @param prefix Module prefix to use (default: "pl", can be "ir" for legacy)
 * @return Python-style string representation
 */
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix = "pl");

/**
 * @brief Print a type in Python syntax
 *
 * @param type Type to print (ScalarType, TensorType, TupleType, etc.)
 * @param prefix Module prefix to use (default: "pl", can be "ir" for legacy)
 * @return Python-style string representation
 */
std::string PythonPrint(const TypePtr& type, const std::string& prefix = "pl");

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_PRINTER_H_
