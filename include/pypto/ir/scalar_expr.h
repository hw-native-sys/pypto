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
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declaration for visitor pattern
// Implementation in pypto/ir/transform/base/visitor.h
class ScalarExprVisitor;

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
 * @brief Base class for scalar expressions in the IR
 *
 * Scalar expressions represent computations that produce scalar values.
 * All expressions are immutable.
 */
class ScalarExpr : public Expr {
 public:
  DataType dtype_;

  /**
   * @brief Create a scalar expression
   *
   * @param span Source location
   * @param dtype Data type
   */
  ScalarExpr(Span s, DataType dtype) : Expr(std::move(s)), dtype_(dtype) {}
  ~ScalarExpr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "Add", "Var", "ConstInt")
   */
  [[nodiscard]] std::string TypeName() const override { return "ScalarExpr"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ScalarExpr::dtype_, "dtype")));
  }
};

using ScalarExprPtr = std::shared_ptr<const ScalarExpr>;

/**
 * @brief Variable reference expression
 *
 * Represents a reference to a named variable.
 */
class Var : public ScalarExpr {
 public:
  std::string name_;

  /**
   * @brief Create a variable reference
   *
   * @param name Variable name
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name, DataType dtype, Span span)
      : ScalarExpr(std::move(span), dtype), name_(std::move(name)) {}

  [[nodiscard]] std::string TypeName() const override { return "Var"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name_ as DEF field for auto-mapping)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScalarExpr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Var::name_, "name")));
  }
};

using VarPtr = std::shared_ptr<const Var>;

/**
 * @brief Constant numeric expression
 *
 * Represents a constant numeric value.
 */
class ConstInt : public ScalarExpr {
 public:
  const int value_;  // Numeric constant value (immutable)

  /**
   * @brief Create a constant expression
   *
   * @param value Numeric value
   * @param span Source location
   */
  ConstInt(int value, DataType dtype, Span span) : ScalarExpr(std::move(span), dtype), value_(value) {}

  [[nodiscard]] std::string TypeName() const override { return "ConstInt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScalarExpr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ConstInt::value_, "value")));
  }
};

using ConstIntPtr = std::shared_ptr<const ConstInt>;

/**
 * @brief Function call expression
 *
 * Represents a function call with an operation and arguments.
 */
class Call : public ScalarExpr {
 public:
  OpPtr op_;                         // Operation/function
  std::vector<ScalarExprPtr> args_;  // Arguments

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ScalarExprPtr> args, DataType dtype, Span span)
      : ScalarExpr(std::move(span), dtype), op_(std::move(op)), args_(std::move(args)) {}

  [[nodiscard]] std::string TypeName() const override { return "Call"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (op and args as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScalarExpr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Call::op_, "op"),
                                          reflection::UsualField(&Call::args_, "args")));
  }
};

using CallPtr = std::shared_ptr<const Call>;

/**
 * @brief Base class for binary expressions
 *
 * Abstract base for all operations with two operands.
 */
class BinaryExpr : public ScalarExpr {
 public:
  ScalarExprPtr left_;   // Left operand
  ScalarExprPtr right_;  // Right operand

  BinaryExpr(ScalarExprPtr left, ScalarExprPtr right, DataType dtype, Span span)
      : ScalarExpr(std::move(span), dtype), left_(std::move(left)), right_(std::move(right)) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (left and right as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScalarExpr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&BinaryExpr::left_, "left"),
                                          reflection::UsualField(&BinaryExpr::right_, "right")));
  }
};

using BinaryExprPtr = std::shared_ptr<const BinaryExpr>;

// Macro to define binary expression node classes
// Usage: DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)")
#define DEFINE_BINARY_EXPR_NODE(OpName, Description)                               \
  /* Description */                                                                \
  class OpName : public BinaryExpr {                                               \
   public:                                                                         \
    OpName(ScalarExprPtr left, ScalarExprPtr right, DataType dtype, Span span)     \
        : BinaryExpr(std::move(left), std::move(right), dtype, std::move(span)) {} \
    [[nodiscard]] std::string TypeName() const override { return #OpName; }        \
  };                                                                               \
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
class UnaryExpr : public ScalarExpr {
 public:
  ScalarExprPtr operand_;  // Operand

  UnaryExpr(ScalarExprPtr operand, DataType dtype, Span span)
      : ScalarExpr(std::move(span), dtype), operand_(std::move(operand)) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (operand_ as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::UsualField(&UnaryExpr::operand_, "operand"));
  }
};

using UnaryExprPtr = std::shared_ptr<const UnaryExpr>;

// Macro to define unary expression node classes
// Usage: DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
#define DEFINE_UNARY_EXPR_NODE(OpName, Description)                         \
  /* Description */                                                         \
  class OpName : public UnaryExpr {                                         \
   public:                                                                  \
    OpName(ScalarExprPtr operand, DataType dtype, Span span)                \
        : UnaryExpr(std::move(operand), dtype, std::move(span)) {}          \
    [[nodiscard]] std::string TypeName() const override { return #OpName; } \
  };                                                                        \
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
