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

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Forward declaration for visitor pattern
// Implementation in pypto/ir/transform/base/visitor.h
class IRVisitor;

// Forward declaration for Op (defined in expr.h)
class Op;
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
  ScalarExpr(Span s, DataType dtype)
      : Expr(std::move(s), std::make_shared<ScalarType>(dtype)), dtype_(dtype) {}
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
 * @brief Constant numeric expression
 *
 * Represents a constant numeric value.
 */
class ConstInt : public Expr {
 public:
  const int value_;  // Numeric constant value (immutable)

  /**
   * @brief Create a constant expression
   *
   * @param value Numeric value
   * @param span Source location
   */
  ConstInt(int value, DataType dtype, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), value_(value) {}

  [[nodiscard]] std::string TypeName() const override { return "ConstInt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ConstInt::value_, "value")));
  }

  [[nodiscard]] DataType dtype() const {
    auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(GetType());
    INTERNAL_CHECK(scalar_type) << "ConstInt is expected to have ScalarType type, but got " +
                                       GetType()->TypeName();
    return scalar_type->dtype_;
  }
};

using ConstIntPtr = std::shared_ptr<const ConstInt>;

/**
 * @brief Constant floating-point expression
 *
 * Represents a constant floating-point value.
 */
class ConstFloat : public Expr {
 public:
  const double value_;  // Floating-point constant value (immutable)

  /**
   * @brief Create a constant floating-point expression
   *
   * @param value Floating-point value
   * @param dtype Data type
   * @param span Source location
   */
  ConstFloat(double value, DataType dtype, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), value_(value) {}

  [[nodiscard]] std::string TypeName() const override { return "ConstFloat"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ConstFloat::value_, "value")));
  }

  [[nodiscard]] DataType dtype() const {
    auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(GetType());
    INTERNAL_CHECK(scalar_type) << "ConstFloat is expected to have ScalarType type, but got " +
                                       GetType()->TypeName();
    return scalar_type->dtype_;
  }
};

using ConstFloatPtr = std::shared_ptr<const ConstFloat>;

/**
 * @brief Base class for binary expressions
 *
 * Abstract base for all operations with two operands.
 */
class BinaryExpr : public Expr {
 public:
  ExprPtr left_;   // Left operand
  ExprPtr right_;  // Right operand

  BinaryExpr(ExprPtr left, ExprPtr right, DataType dtype, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(dtype)),
        left_(std::move(left)),
        right_(std::move(right)) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (left and right as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&BinaryExpr::left_, "left"),
                                          reflection::UsualField(&BinaryExpr::right_, "right")));
  }
};

using BinaryExprPtr = std::shared_ptr<const BinaryExpr>;

// Macro to define binary expression node classes
// Usage: DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)")
#define DEFINE_BINARY_EXPR_NODE(OpName, Description)                                          \
  /* Description */                                                                           \
  class OpName : public BinaryExpr {                                                          \
   public:                                                                                    \
    OpName(ExprPtr left, ExprPtr right, DataType dtype, Span span)                            \
        : BinaryExpr(std::move(left), std::move(right), std::move(dtype), std::move(span)) {} \
    [[nodiscard]] std::string TypeName() const override { return #OpName; }                   \
  };                                                                                          \
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

  UnaryExpr(ExprPtr operand, DataType dtype, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), operand_(std::move(operand)) {}

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&UnaryExpr::operand_, "operand")));
  }
};

using UnaryExprPtr = std::shared_ptr<const UnaryExpr>;

// Macro to define unary expression node classes
// Usage: DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
#define DEFINE_UNARY_EXPR_NODE(OpName, Description)                         \
  /* Description */                                                         \
  class OpName : public UnaryExpr {                                         \
   public:                                                                  \
    OpName(ExprPtr operand, DataType dtype, Span span)                      \
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
// ========== Helper Functions for Operator Construction ==========

/**
 * @brief Get the dtype from a scalar expression or scalar var
 *
 * @param expr Expression to extract dtype from
 * @return DataType of the expression
 * @throws std::runtime_error if expr is not a scalar expression or scalar var
 */
inline DataType GetScalarDtype(const ExprPtr& expr) {
  if (auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(expr->GetType())) {
    return scalar_type->dtype_;
  } else {
    throw TypeError("Expression must be ScalarExpr or Var with ScalarType, got " + expr->TypeName());
  }
}

/**
 * @brief Infer result dtype for binary operations
 *
 * @param left Left operand
 * @param right Right operand
 * @return DataType for the result (currently uses left dtype, TODO: type promotion)
 */
inline DataType InferBinaryDtype(const ExprPtr& left, const ExprPtr& right) {
  DataType left_dtype = GetScalarDtype(left);
  DataType right_dtype = GetScalarDtype(right);

  // For now, use left dtype
  // TODO(ir): Implement proper type promotion rules
  return left_dtype;
}

// ========== Binary Operator Construction Functions ==========

inline ExprPtr MakeAdd(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Add>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeSub(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Sub>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeMul(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Mul>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeFloatDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<FloatDiv>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeFloorDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<FloorDiv>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeFloorMod(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<FloorMod>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakePow(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Pow>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeEq(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Eq>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeNe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Ne>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeLt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Lt>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeLe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Le>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeGt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Gt>(left, right, InferBinaryDtype(left, right), span);
}

inline ExprPtr MakeGe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  return std::make_shared<Ge>(left, right, InferBinaryDtype(left, right), span);
}

// ========== Unary Operator Construction Functions ==========

inline ExprPtr MakeNeg(const ExprPtr& operand, const Span& span = Span::unknown()) {
  return std::make_shared<Neg>(operand, GetScalarDtype(operand), span);
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SCALAR_EXPR_H_
