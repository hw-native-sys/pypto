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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"
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
  const int64_t value_;  // Numeric constant value (immutable)

  /**
   * @brief Create a constant expression
   *
   * @param value Numeric value
   * @param span Source location
   */
  ConstInt(int64_t value, DataType dtype, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), value_(value) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstInt; }
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
    // Note: Must use dynamic_pointer_cast here because this header is included before
    // the TypePtr overload of As<> is defined in kind_traits.h
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

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstFloat; }
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
    // Note: Must use dynamic_pointer_cast here because this header is included before
    // the TypePtr overload of As<> is defined in kind_traits.h
    auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(GetType());
    INTERNAL_CHECK(scalar_type) << "ConstFloat is expected to have ScalarType type, but got " +
                                       GetType()->TypeName();
    return scalar_type->dtype_;
  }
};

using ConstFloatPtr = std::shared_ptr<const ConstFloat>;

/**
 * @brief Constant boolean expression
 *
 * Represents a constant boolean value.
 */
class ConstBool : public Expr {
 public:
  const bool value_;  // Boolean constant value (immutable)

  /**
   * @brief Create a constant boolean expression
   *
   * @param value Boolean value
   * @param span Source location
   */
  ConstBool(bool value, Span span)
      : Expr(std::move(span), std::make_shared<ScalarType>(DataType::BOOL)), value_(value) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstBool; }
  [[nodiscard]] std::string TypeName() const override { return "ConstBool"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ConstBool::value_, "value")));
  }

  [[nodiscard]] DataType dtype() const { return DataType::BOOL; }
};

using ConstBoolPtr = std::shared_ptr<const ConstBool>;

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
// NOLINTBEGIN(bugprone-macro-parentheses, readability/nolint)
#define DEFINE_BINARY_EXPR_NODE(OpName, Description)                                          \
  /* Description */                                                                           \
  class OpName : public BinaryExpr {                                                          \
   public:                                                                                    \
    OpName(ExprPtr left, ExprPtr right, DataType dtype, Span span)                            \
        : BinaryExpr(std::move(left), std::move(right), std::move(dtype), std::move(span)) {} \
    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::OpName; }          \
    [[nodiscard]] std::string TypeName() const override { return #OpName; }                   \
  };                                                                                          \
                                                                                              \
  using OpName##Ptr = std::shared_ptr<const OpName>;
// NOLINTEND(bugprone-macro-parentheses, readability/nolint)

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
// NOLINTBEGIN(bugprone-macro-parentheses, readability/nolint)
#define DEFINE_UNARY_EXPR_NODE(OpName, Description)                                  \
  /* Description */                                                                  \
  class OpName : public UnaryExpr {                                                  \
   public:                                                                           \
    OpName(ExprPtr operand, DataType dtype, Span span)                               \
        : UnaryExpr(std::move(operand), dtype, std::move(span)) {}                   \
    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::OpName; } \
    [[nodiscard]] std::string TypeName() const override { return #OpName; }          \
  };                                                                                 \
                                                                                     \
  using OpName##Ptr = std::shared_ptr<const OpName>;
// NOLINTEND(bugprone-macro-parentheses, readability/nolint)
DEFINE_UNARY_EXPR_NODE(Abs, "Absolute value expression (abs(operand))")
DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
DEFINE_UNARY_EXPR_NODE(Not, "Logical not expression (not operand)")
DEFINE_UNARY_EXPR_NODE(BitNot, "Bitwise not expression (~operand)")
DEFINE_UNARY_EXPR_NODE(Cast, "Cast expression (cast operand to dtype)")

#undef DEFINE_UNARY_EXPR_NODE
// ========== Helper Functions for Operator Construction ==========

/**
 * @brief Get the dtype from a scalar expression or scalar var
 *
 * @param expr Expression to extract dtype from
 * @return DataType of the expression
 * @throws pypto::TypeError if expr is not a scalar expression or scalar var
 */
inline DataType GetScalarDtype(const ExprPtr& expr) {
  // Note: Must use dynamic_pointer_cast here because this header is included before
  // the TypePtr overload of As<> is defined in kind_traits.h
  if (auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(expr->GetType())) {
    return scalar_type->dtype_;
  } else {
    throw TypeError("Expression must be ScalarExpr or Var with ScalarType, got " + expr->TypeName() +
                    " with type " + expr->GetType()->TypeName());
  }
}

inline bool IsBoolDtype(const DataType& dtype) { return dtype == DataType::BOOL; }

enum class ScalarCategory {
  kInt,
  kFloat,
};

inline ScalarCategory GetNumericCategory(const DataType& dtype, const std::string& op_name) {
  if (dtype.IsFloat()) {
    return ScalarCategory::kFloat;
  }
  if (dtype.IsInt()) {
    return ScalarCategory::kInt;
  }
  throw TypeError("Operator '" + op_name + "' requires numeric scalar dtype, got " + dtype.ToString());
}

inline DataType PromoteSameCategoryDtype(const DataType& left_dtype, const DataType& right_dtype,
                                         const std::string& op_name) {
  if (IsBoolDtype(left_dtype) || IsBoolDtype(right_dtype)) {
    throw TypeError("Operator '" + op_name + "' does not accept bool dtype");
  }
  auto left_category = GetNumericCategory(left_dtype, op_name);
  auto right_category = GetNumericCategory(right_dtype, op_name);
  if (left_category != right_category) {
    throw TypeError("Operator '" + op_name + "' requires same numeric dtype category, got " +
                    left_dtype.ToString() + " and " + right_dtype.ToString());
  }
  size_t left_bits = left_dtype.GetBit();
  size_t right_bits = right_dtype.GetBit();
  if (left_bits > right_bits) {
    return left_dtype;
  }
  if (right_bits > left_bits) {
    return right_dtype;
  }
  return left_dtype;
}

struct BinaryOperands {
  ExprPtr left;
  ExprPtr right;
  DataType dtype;
};

inline ExprPtr MaybeCast(const ExprPtr& expr, DataType target_dtype, const Span& span) {
  DataType dtype = GetScalarDtype(expr);
  if (dtype == target_dtype) {
    return expr;
  }
  return std::make_shared<Cast>(expr, target_dtype, span);
}

inline BinaryOperands PromoteBinaryOperands(const ExprPtr& left, const ExprPtr& right,
                                            const std::string& op_name, const Span& span) {
  DataType left_dtype = GetScalarDtype(left);
  DataType right_dtype = GetScalarDtype(right);
  DataType promoted_dtype = PromoteSameCategoryDtype(left_dtype, right_dtype, op_name);
  return {MaybeCast(left, promoted_dtype, span), MaybeCast(right, promoted_dtype, span), promoted_dtype};
}

inline BinaryOperands PromoteIntBinaryOperands(const ExprPtr& left, const ExprPtr& right,
                                               const std::string& op_name, const Span& span) {
  DataType left_dtype = GetScalarDtype(left);
  DataType right_dtype = GetScalarDtype(right);
  if (!left_dtype.IsInt() || !right_dtype.IsInt()) {
    throw TypeError("Operator '" + op_name + "' requires integer dtype, got " + left_dtype.ToString() +
                    " and " + right_dtype.ToString());
  }
  DataType promoted_dtype = PromoteSameCategoryDtype(left_dtype, right_dtype, op_name);
  return {MaybeCast(left, promoted_dtype, span), MaybeCast(right, promoted_dtype, span), promoted_dtype};
}

// ========== Binary Operator Construction Functions ==========

inline ExprPtr MakeCast(const ExprPtr& operand, DataType dtype, const Span& span = Span::unknown()) {
  return std::make_shared<Cast>(operand, dtype, span);
}

inline ExprPtr MakeAdd(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "add", span);
  return std::make_shared<Add>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeSub(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "sub", span);
  return std::make_shared<Sub>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeMul(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "mul", span);
  return std::make_shared<Mul>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloatDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "truediv", span);
  return std::make_shared<FloatDiv>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloorDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "floordiv", span);
  return std::make_shared<FloorDiv>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloorMod(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "mod", span);
  return std::make_shared<FloorMod>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakePow(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "pow", span);
  return std::make_shared<Pow>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeEq(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "eq", span);
  return std::make_shared<Eq>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeNe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "ne", span);
  return std::make_shared<Ne>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeLt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "lt", span);
  return std::make_shared<Lt>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeLe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "le", span);
  return std::make_shared<Le>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeGt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "gt", span);
  return std::make_shared<Gt>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeGe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteBinaryOperands(left, right, "ge", span);
  return std::make_shared<Ge>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeBitAnd(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteIntBinaryOperands(left, right, "bit_and", span);
  return std::make_shared<BitAnd>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitOr(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteIntBinaryOperands(left, right, "bit_or", span);
  return std::make_shared<BitOr>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitXor(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::unknown()) {
  auto operands = PromoteIntBinaryOperands(left, right, "bit_xor", span);
  return std::make_shared<BitXor>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitShiftLeft(const ExprPtr& left, const ExprPtr& right,
                                const Span& span = Span::unknown()) {
  auto operands = PromoteIntBinaryOperands(left, right, "bit_shift_left", span);
  return std::make_shared<BitShiftLeft>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitShiftRight(const ExprPtr& left, const ExprPtr& right,
                                 const Span& span = Span::unknown()) {
  auto operands = PromoteIntBinaryOperands(left, right, "bit_shift_right", span);
  return std::make_shared<BitShiftRight>(operands.left, operands.right, operands.dtype, span);
}

// ========== Unary Operator Construction Functions ==========

inline ExprPtr MakeNeg(const ExprPtr& operand, const Span& span = Span::unknown()) {
  return std::make_shared<Neg>(operand, GetScalarDtype(operand), span);
}

inline ExprPtr MakeBitNot(const ExprPtr& operand, const Span& span = Span::unknown()) {
  DataType dtype = GetScalarDtype(operand);
  if (!dtype.IsInt()) {
    throw TypeError("Operator 'bit_not' requires integer dtype, got " + dtype.ToString());
  }
  return std::make_shared<BitNot>(operand, dtype, span);
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SCALAR_EXPR_H_
