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

#ifndef PYPTO_IR_SCALAR_EXPR_H_
#define PYPTO_IR_SCALAR_EXPR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {

// Forward declaration for visitor pattern
// Implementation in pypto/ir/transform/visitor.h
class ExprVisitor;

/**
 * @brief Base class for operations/functions
 *
 * Represents callable operations in the IR.
 */
class Op {
 public:
  std::string name_;

  explicit Op(std::string name) : name_(std::move(name)) {}
  virtual ~Op() = default;
};

using OpPtr = std::shared_ptr<const Op>;

/**
 * @brief Base class for all expressions in the IR
 *
 * Expressions represent computations that produce values.
 * All expressions are immutable.
 */
class Expr : public IRNode {
 public:
  ~Expr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "Add", "Var", "ConstInt")
   */
  [[nodiscard]] virtual const char* type_name() const { return "Expr"; }

 protected:
  explicit Expr(Span s) : IRNode(std::move(s)) {}
};

using ExprPtr = std::shared_ptr<const Expr>;

/**
 * @brief Variable reference expression
 *
 * Represents a reference to a named variable.
 */
class Var : public Expr {
 public:
  std::string name_;
  // TODO(siyuan): add dtype

  /**
   * @brief Create a variable reference
   *
   * @param name Variable name
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name, Span span) : Expr(std::move(span)), name_(std::move(name)) {}

  [[nodiscard]] const char* type_name() const override { return "Var"; }
};

using VarPtr = std::shared_ptr<const Var>;

/**
 * @brief Constant numeric expression
 *
 * Represents a constant numeric value.
 */
class ConstInt : public Expr {
 public:
  const int value;  // Numeric constant value (immutable)

  /**
   * @brief Create a constant expression
   *
   * @param value Numeric value
   * @param span Source location
   */
  ConstInt(int value, Span span) : Expr(std::move(span)), value(value) {}

  [[nodiscard]] const char* type_name() const override { return "ConstInt"; }
};

using ConstIntPtr = std::shared_ptr<const ConstInt>;

/**
 * @brief Function call expression
 *
 * Represents a function call with an operation and arguments.
 */
class Call : public Expr {
 public:
  OpPtr op_;                   // Operation/function
  std::vector<ExprPtr> args_;  // Arguments

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)) {}

  [[nodiscard]] const char* type_name() const override { return "Call"; }
};

using CallPtr = std::shared_ptr<const Call>;

/**
 * @brief Base class for binary expressions
 *
 * Abstract base for all operations with two operands.
 */
class BinaryExpr : public Expr {
 public:
  ExprPtr left_;   // Left operand
  ExprPtr right_;  // Right operand

  BinaryExpr(ExprPtr left, ExprPtr right, Span span)
      : Expr(std::move(span)), left_(std::move(left)), right_(std::move(right)) {}
};

using BinaryExprPtr = std::shared_ptr<const BinaryExpr>;

// Macro to define binary expression node classes
// Usage: DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)")
#define DEFINE_BINARY_EXPR_NODE(OpName, Description)                         \
  /* Description */                                                          \
  class OpName : public BinaryExpr {                                         \
   public:                                                                   \
    OpName(ExprPtr left, ExprPtr right, Span span)                           \
        : BinaryExpr(std::move(left), std::move(right), std::move(span)) {}  \
    [[nodiscard]] const char* type_name() const override { return #OpName; } \
  };                                                                         \
                                                                             \
  using OpName##Ptr = std::shared_ptr<const OpName>;

DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)");
DEFINE_BINARY_EXPR_NODE(Sub, "Subtraction expression (left - right)")
DEFINE_BINARY_EXPR_NODE(Mul, "Multiplication expression (left * right)")
DEFINE_BINARY_EXPR_NODE(FloorDiv, "Floor division expression (left // right)")
DEFINE_BINARY_EXPR_NODE(FloorMod, "Floor modulo expression (left % right)")
DEFINE_BINARY_EXPR_NODE(FloatDiv, "Float division expression (left / right)")
DEFINE_BINARY_EXPR_NODE(Min, "Minimum expression (min(left, right)")
DEFINE_BINARY_EXPR_NODE(Max, "Maximum expression (max(left, right)")
DEFINE_BINARY_EXPR_NODE(Pow, "Power expression (left ** right)")
DEFINE_BINARY_EXPR_NODE(Eq, "Equality expression (left == right)")
DEFINE_BINARY_EXPR_NODE(Ne, "Inequality expression (left != right)")
DEFINE_BINARY_EXPR_NODE(Lt, "Less than expression (left < right)")
DEFINE_BINARY_EXPR_NODE(Le, "Less than or equal to expression (left <= right)")
DEFINE_BINARY_EXPR_NODE(Gt, "Greater than expression (left > right)")
DEFINE_BINARY_EXPR_NODE(Ge, "Greater than or equal to expression (left >= right)")
DEFINE_BINARY_EXPR_NODE(And, "Logical and expression (left and right)")
DEFINE_BINARY_EXPR_NODE(Or, "Logical or expression (left or right)")
DEFINE_BINARY_EXPR_NODE(Xor, "Logical xor expression (left xor right)")
DEFINE_BINARY_EXPR_NODE(BitAnd, "Bitwise and expression (left & right)")
DEFINE_BINARY_EXPR_NODE(BitOr, "Bitwise or expression (left | right)")
DEFINE_BINARY_EXPR_NODE(BitXor, "Bitwise xor expression (left ^ right)")
DEFINE_BINARY_EXPR_NODE(BitShiftLeft, "Bitwise left shift expression (left << right)")
DEFINE_BINARY_EXPR_NODE(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef DEFINE_BINARY_EXPR_NODE

/**
 * @brief Base class for unary expressions
 *
 * Abstract base for all operations with one operand.
 */
class UnaryExpr : public Expr {
 public:
  ExprPtr operand_;  // Operand

  UnaryExpr(ExprPtr operand, Span span) : Expr(std::move(span)), operand_(std::move(operand)) {}
};

using UnaryExprPtr = std::shared_ptr<const UnaryExpr>;

// Macro to define unary expression node classes
// Usage: DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
#define DEFINE_UNARY_EXPR_NODE(OpName, Description)                                        \
  /* Description */                                                                        \
  class OpName : public UnaryExpr {                                                        \
   public:                                                                                 \
    OpName(ExprPtr operand, Span span) : UnaryExpr(std::move(operand), std::move(span)) {} \
    [[nodiscard]] const char* type_name() const override { return #OpName; }               \
  };                                                                                       \
                                                                                           \
  using OpName##Ptr = std::shared_ptr<const OpName>;

DEFINE_UNARY_EXPR_NODE(Abs, "Absolute value expression (abs(operand))")
DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
DEFINE_UNARY_EXPR_NODE(Not, "Logical not expression (not operand)")
DEFINE_UNARY_EXPR_NODE(BitNot, "Bitwise not expression (~operand)")

#undef DEFINE_UNARY_EXPR_NODE
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SCALAR_EXPR_H_
