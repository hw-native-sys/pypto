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

#include "pypto/ir/transform/visitor.h"

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
 * @brief Expression pretty printer
 *
 * Prints expressions with minimal parentheses based on operator precedence.
 * Inherits from ExprVisitor to traverse the expression tree.
 */
class ExprPrinter : public ExprVisitor {
 public:
  ExprPrinter() = default;
  ~ExprPrinter() override = default;

  /**
   * @brief Print an expression to a string
   *
   * @param expr Expression to print
   * @return String representation with minimal parentheses
   */
  std::string Print(const ExprPtr& expr);

 protected:
  // Leaf nodes
  void VisitExpr_(const VarPtr& op) override;
  void VisitExpr_(const ConstIntPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;

  // Binary operations
  void VisitExpr_(const AddPtr& op) override;
  void VisitExpr_(const SubPtr& op) override;
  void VisitExpr_(const MulPtr& op) override;
  void VisitExpr_(const FloorDivPtr& op) override;
  void VisitExpr_(const FloorModPtr& op) override;
  void VisitExpr_(const FloatDivPtr& op) override;
  void VisitExpr_(const MinPtr& op) override;
  void VisitExpr_(const MaxPtr& op) override;
  void VisitExpr_(const PowPtr& op) override;
  void VisitExpr_(const EqPtr& op) override;
  void VisitExpr_(const NePtr& op) override;
  void VisitExpr_(const LtPtr& op) override;
  void VisitExpr_(const LePtr& op) override;
  void VisitExpr_(const GtPtr& op) override;
  void VisitExpr_(const GePtr& op) override;
  void VisitExpr_(const AndPtr& op) override;
  void VisitExpr_(const OrPtr& op) override;
  void VisitExpr_(const XorPtr& op) override;
  void VisitExpr_(const BitAndPtr& op) override;
  void VisitExpr_(const BitOrPtr& op) override;
  void VisitExpr_(const BitXorPtr& op) override;
  void VisitExpr_(const BitShiftLeftPtr& op) override;
  void VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations
  void VisitExpr_(const AbsPtr& op) override;
  void VisitExpr_(const NegPtr& op) override;
  void VisitExpr_(const NotPtr& op) override;
  void VisitExpr_(const BitNotPtr& op) override;

 private:
  std::ostringstream stream_;

  /**
   * @brief Print a binary operator with minimal parentheses
   *
   * @param parent Parent binary expression
   * @param op_symbol Operator symbol (e.g., "+", "*", "and")
   */
  void PrintBinaryOp(const BinaryExprPtr& parent, const char* op_symbol);

  /**
   * @brief Print a function-style binary operator
   *
   * Examples: min(a, b), max(a, b)
   *
   * @param parent Parent binary expression
   * @param func_name Function name (e.g., "min", "max")
   */
  void PrintFunctionBinaryOp(const BinaryExprPtr& parent, const char* func_name);

  /**
   * @brief Print a child expression with parentheses if needed
   *
   * @param parent Parent expression
   * @param child Child expression to print
   * @param is_left True if child is left operand, false if right
   */
  void PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left);

  /**
   * @brief Determine if child needs parentheses
   *
   * @param parent Parent expression
   * @param child Child expression
   * @param is_left True if child is left operand, false if right
   * @return true if parentheses needed, false otherwise
   */
  bool NeedsParens(const ExprPtr& parent, const ExprPtr& child, bool is_left);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_PRINTER_H_
