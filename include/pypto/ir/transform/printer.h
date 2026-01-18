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
 * @brief IR pretty printer (DEPRECATED - use IRPythonPrinter for new code)
 *
 * Prints IR nodes (expressions and statements) with minimal parentheses based on operator precedence.
 * Inherits from IRVisitor to traverse the IR tree.
 *
 * @deprecated This printer is deprecated. Use IRPythonPrinter for Python-style IR syntax.
 */
class IRPrinter : public IRVisitor {
 public:
  IRPrinter() = default;
  ~IRPrinter() override = default;

  /**
   * @brief Print an expression to a string
   *
   * @param expr Expression to print
   * @return String representation with minimal parentheses
   */
  std::string Print(const ExprPtr& expr);

  /**
   * @brief Print a statement to a string
   *
   * @param stmt Statement to print
   * @return String representation
   */
  std::string Print(const StmtPtr& stmt);

  /**
   * @brief Print a function to a string
   *
   * @param func Function to print
   * @return String representation
   */
  std::string Print(const FunctionPtr& func);

  /**
   * @brief Print a program to a string
   *
   * @param program Program to print
   * @return String representation
   */
  std::string Print(const ProgramPtr& program);

 protected:
  // Leaf nodes
  void VisitExpr_(const VarPtr& op) override;
  void VisitExpr_(const IterArgPtr& op) override;
  void VisitExpr_(const ConstIntPtr& op) override;
  void VisitExpr_(const ConstFloatPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;
  void VisitExpr_(const TupleGetItemExprPtr& op) override;

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
  void VisitExpr_(const CastPtr& op) override;

  // Statement types
  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitStmt_(const YieldStmtPtr& op) override;
  void VisitStmt_(const ReturnStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const SeqStmtsPtr& op) override;
  void VisitStmt_(const OpStmtsPtr& op) override;
  void VisitStmt_(const StmtPtr& op) override;

  // Function type
  void VisitFunction(const FunctionPtr& func);

  // Program type
  void VisitProgram(const ProgramPtr& program);

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

/**
 * @brief Print an IR node in Python syntax
 *
 * @param node IR node to print (Expr, Stmt, Function, or Program)
 * @param prefix Module prefix to use (default: "pi", can be "ir" for legacy)
 * @return Python-style string representation
 */
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix = "pi");

/**
 * @brief Print a type in Python syntax
 *
 * @param type Type to print (ScalarType, TensorType, TupleType, etc.)
 * @param prefix Module prefix to use (default: "pi", can be "ir" for legacy)
 * @return Python-style string representation
 */
std::string PythonPrint(const TypePtr& type, const std::string& prefix = "pi");

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_PRINTER_H_
