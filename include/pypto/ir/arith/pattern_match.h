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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

/**
 * @file pattern_match.h
 * @brief Expression-template based pattern matching for arithmetic simplification.
 *
 * Usage:
 * @code
 *   using namespace pypto::ir::arith;
 *   PVar<ExprPtr> x, y, z;
 *
 *   // max(x + z, y + z) => max(x, y) + z
 *   if (max(x + z, y + z).Match(expr)) {
 *     return (max(x, y) + z).Eval();
 *   }
 *
 *   PVar<ConstIntPtr> c;
 *   PVar<VarPtr> v;
 *   // v * c  matches  x * 3
 *   if ((v * c).Match(some_expr)) {
 *     int64_t val = c.Eval()->value_;
 *   }
 * @endcode
 *
 * PVar is not thread-safe. Do not reuse the same PVar across threads.
 * Filled values are valid until the next call to Match.
 */
#ifndef PYPTO_IR_ARITH_PATTERN_MATCH_H_
#define PYPTO_IR_ARITH_PATTERN_MATCH_H_

#include <algorithm>  // NOLINT(misc-include-cleaner) required by cpplint for hidden friend min/max
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {
namespace arith {

// Forward declarations for hidden friends in Pattern.
template <typename OpType, typename TA, typename TB>
class PBinaryExpr;

// ============================================================================
// Pattern base (CRTP)
// ============================================================================

/// CRTP base for all pattern types.
/// Provides Match() (top-level entry point) delegating to Derived::Match_().
template <typename Derived>
class Pattern {
  friend Derived;
  Pattern() = default;

 public:
  /// Nested storage type: value for intermediates, const& for PVars.
  using Nested = Derived;

  /// Hidden friends for min/max: non-template functions that beat
  /// std::min/std::max (found via ADL through shared_ptr in ExprPtr)
  /// in overload resolution. They handle same-type arguments only;
  /// different-type arguments use the template overloads which don't
  /// conflict with std::min (std::min requires both args same type).
  friend auto min(const Derived& a, const Derived& b) -> PBinaryExpr<Min, Derived, Derived> {
    return PBinaryExpr<Min, Derived, Derived>(a, b);
  }
  friend auto max(const Derived& a, const Derived& b) -> PBinaryExpr<Max, Derived, Derived> {
    return PBinaryExpr<Max, Derived, Derived>(a, b);
  }

  /// Match value against the pattern, populating any PVars.
  template <typename NodeType>
  [[nodiscard]] bool Match(const NodeType& value) const {
    return Match(value, []() { return true; });
  }

  /// Match with an additional condition checked after structural match.
  template <typename NodeType, typename Condition>
  [[nodiscard]] bool Match(const NodeType& value, Condition cond) const {
    derived().InitMatch_();
    return derived().Match_(value) && cond();
  }

  [[nodiscard]] const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

// ============================================================================
// PEqualChecker — default equality for pattern dedup
// ============================================================================

template <typename T>
class PEqualChecker {
 public:
  bool operator()(const T& lhs, const T& rhs) const { return lhs == rhs; }
};

/// ExprPtr equality: pointer identity (same IR node).
template <>
class PEqualChecker<ExprPtr> {
 public:
  bool operator()(const ExprPtr& lhs, const ExprPtr& rhs) const { return lhs.get() == rhs.get(); }
};

/// ConstIntPtr equality: value-based.
template <>
class PEqualChecker<ConstIntPtr> {
 public:
  bool operator()(const ConstIntPtr& lhs, const ConstIntPtr& rhs) const { return lhs->value_ == rhs->value_; }
};

/// VarPtr equality: pointer identity (same Var object).
template <>
class PEqualChecker<VarPtr> {
 public:
  bool operator()(const VarPtr& lhs, const VarPtr& rhs) const { return lhs.get() == rhs.get(); }
};

// ============================================================================
// PVar — pattern variable that captures a matched value
// ============================================================================

template <typename T>
class PVar;

/// PVar<ExprPtr>: matches any expression.
template <>
class PVar<ExprPtr> : public Pattern<PVar<ExprPtr>> {
 public:
  using Nested = const PVar<ExprPtr>&;

  void InitMatch_() const { filled_ = false; }

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    if (!filled_) {
      value_ = value;
      filled_ = true;
      return true;
    }
    return PEqualChecker<ExprPtr>()(value_, value);
  }

  [[nodiscard]] ExprPtr Eval() const {
    INTERNAL_CHECK(filled_) << "PVar<ExprPtr> evaluated before being matched";
    return value_;
  }

 protected:
  mutable ExprPtr value_;
  mutable bool filled_{false};
};

/// PVar<ConstIntPtr>: matches only ConstInt expressions.
template <>
class PVar<ConstIntPtr> : public Pattern<PVar<ConstIntPtr>> {
 public:
  using Nested = const PVar<ConstIntPtr>&;

  void InitMatch_() const { filled_ = false; }

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    auto ci = As<ConstInt>(value);
    if (!ci) return false;
    if (!filled_) {
      value_ = ci;
      filled_ = true;
      return true;
    }
    return PEqualChecker<ConstIntPtr>()(value_, ci);
  }

  [[nodiscard]] ConstIntPtr Eval() const {
    INTERNAL_CHECK(filled_) << "PVar<ConstIntPtr> evaluated before being matched";
    return value_;
  }

 protected:
  mutable ConstIntPtr value_;
  mutable bool filled_{false};
};

/// PVar<VarPtr>: matches only Var expressions.
template <>
class PVar<VarPtr> : public Pattern<PVar<VarPtr>> {
 public:
  using Nested = const PVar<VarPtr>&;

  void InitMatch_() const { filled_ = false; }

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    auto var = AsVarLike(value);
    if (!var) return false;
    if (!filled_) {
      value_ = var;
      filled_ = true;
      return true;
    }
    return PEqualChecker<VarPtr>()(value_, var);
  }

  [[nodiscard]] VarPtr Eval() const {
    INTERNAL_CHECK(filled_) << "PVar<VarPtr> evaluated before being matched";
    return value_;
  }

 protected:
  mutable VarPtr value_;
  mutable bool filled_{false};
};

// ============================================================================
// PConst — matches a fixed constant value
// ============================================================================

template <typename T>
class PConst : public Pattern<PConst<T>> {
 public:
  explicit PConst(T value) : value_(value) {}

  void InitMatch_() const {}
  [[nodiscard]] bool Match_(const T& value) const { return PEqualChecker<T>()(value_, value); }
  [[nodiscard]] T Eval() const { return value_; }

 private:
  const T value_;
};

// ============================================================================
// PExprLiteral — wraps a pre-built ExprPtr so it can appear in pattern
// result expressions. Match is identity; Eval returns the stored value.
// ============================================================================

class PExprLiteral : public Pattern<PExprLiteral> {
 public:
  explicit PExprLiteral(ExprPtr value) : value_(std::move(value)) {}

  void InitMatch_() const {}
  [[nodiscard]] bool Match_(const ExprPtr& value) const { return value_.get() == value.get(); }
  [[nodiscard]] ExprPtr Eval() const { return value_; }

 private:
  ExprPtr value_;
};

/// Convenience: wrap an ExprPtr as a pattern literal.
inline PExprLiteral pexpr(ExprPtr value) { return PExprLiteral(std::move(value)); }

// ============================================================================
// PConstWithTypeLike — matches ConstInt with a specific int64_t value,
// Eval produces a ConstInt with the dtype of a reference pattern.
// ============================================================================

template <typename TA>
class PConstWithTypeLike : public Pattern<PConstWithTypeLike<TA>> {
 public:
  PConstWithTypeLike(const TA& ref, int64_t value) : ref_(ref), value_(value) {}

  void InitMatch_() const {}

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    auto ci = As<ConstInt>(value);
    return ci && ci->value_ == value_;
  }

  [[nodiscard]] ExprPtr Eval() const {
    DataType dtype = GetScalarDtype(ref_.Eval());
    return MakeConstInt(value_, dtype);
  }

 private:
  typename TA::Nested ref_;
  int64_t value_;
};

// ============================================================================
// MakeBinaryExprHelper — traits mapping OpType -> Make* function
// ============================================================================

template <typename OpType>
struct MakeBinaryExprHelper;

// NOLINTBEGIN(bugprone-macro-parentheses, readability/nolint)
#define PYPTO_DEFINE_BINARY_MAKE_HELPER(OpType, MakeFunc)                              \
  template <>                                                                          \
  struct MakeBinaryExprHelper<OpType> {                                                \
    static ExprPtr Make(const ExprPtr& a, const ExprPtr& b) { return MakeFunc(a, b); } \
  };
// NOLINTEND(bugprone-macro-parentheses, readability/nolint)

PYPTO_DEFINE_BINARY_MAKE_HELPER(Add, MakeAdd)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Sub, MakeSub)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Mul, MakeMul)
PYPTO_DEFINE_BINARY_MAKE_HELPER(FloorDiv, MakeFloorDiv)
PYPTO_DEFINE_BINARY_MAKE_HELPER(FloorMod, MakeFloorMod)
PYPTO_DEFINE_BINARY_MAKE_HELPER(FloatDiv, MakeFloatDiv)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Min, MakeMin)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Max, MakeMax)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Pow, MakePow)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Eq, MakeEq)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Ne, MakeNe)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Lt, MakeLt)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Le, MakeLe)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Gt, MakeGt)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Ge, MakeGe)
PYPTO_DEFINE_BINARY_MAKE_HELPER(And, MakeAnd)
PYPTO_DEFINE_BINARY_MAKE_HELPER(Or, MakeOr)
PYPTO_DEFINE_BINARY_MAKE_HELPER(BitAnd, MakeBitAnd)
PYPTO_DEFINE_BINARY_MAKE_HELPER(BitOr, MakeBitOr)
PYPTO_DEFINE_BINARY_MAKE_HELPER(BitXor, MakeBitXor)
PYPTO_DEFINE_BINARY_MAKE_HELPER(BitShiftLeft, MakeBitShiftLeft)
PYPTO_DEFINE_BINARY_MAKE_HELPER(BitShiftRight, MakeBitShiftRight)

#undef PYPTO_DEFINE_BINARY_MAKE_HELPER

// Xor has no MakeXor in scalar_expr.h; construct directly.
template <>
struct MakeBinaryExprHelper<Xor> {
  static ExprPtr Make(const ExprPtr& a, const ExprPtr& b) {
    return std::make_shared<Xor>(a, b, DataType::BOOL, Span::unknown());
  }
};

// ============================================================================
// PBinaryExpr — matches a binary expression node and sub-patterns
// ============================================================================

template <typename OpType, typename TA, typename TB>
class PBinaryExpr : public Pattern<PBinaryExpr<OpType, TA, TB>> {
 public:
  PBinaryExpr(const TA& a, const TB& b) : a_(a), b_(b) {}

  void InitMatch_() const {
    a_.InitMatch_();
    b_.InitMatch_();
  }

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    auto node = As<OpType>(value);
    if (!node) return false;
    if (!a_.Match_(node->left_)) return false;
    if (!b_.Match_(node->right_)) return false;
    return true;
  }

  [[nodiscard]] ExprPtr Eval() const {
    ExprPtr lhs = a_.Eval();
    ExprPtr rhs = b_.Eval();
    auto folded = TryConstFoldBinary(KindTrait<OpType>::kind, lhs, rhs);
    if (folded) return folded;
    return MakeBinaryExprHelper<OpType>::Make(lhs, rhs);
  }

 private:
  typename TA::Nested a_;
  typename TB::Nested b_;
};

// ============================================================================
// MakeUnaryExprHelper — traits mapping OpType -> Make* function
// ============================================================================

template <typename OpType>
struct MakeUnaryExprHelper;

template <>
struct MakeUnaryExprHelper<Neg> {
  static ExprPtr Make(const ExprPtr& a) { return MakeNeg(a); }
};

template <>
struct MakeUnaryExprHelper<Not> {
  static ExprPtr Make(const ExprPtr& a) { return MakeNot(a); }
};

template <>
struct MakeUnaryExprHelper<BitNot> {
  static ExprPtr Make(const ExprPtr& a) { return MakeBitNot(a); }
};

template <>
struct MakeUnaryExprHelper<Abs> {
  static ExprPtr Make(const ExprPtr& a) {
    return std::make_shared<Abs>(a, GetScalarDtype(a), Span::unknown());
  }
};

// ============================================================================
// PUnaryExpr — matches a unary expression node
// ============================================================================

template <typename OpType, typename TA>
class PUnaryExpr : public Pattern<PUnaryExpr<OpType, TA>> {
 public:
  explicit PUnaryExpr(const TA& a) : a_(a) {}

  void InitMatch_() const { a_.InitMatch_(); }

  [[nodiscard]] bool Match_(const ExprPtr& value) const {
    auto node = As<OpType>(value);
    if (!node) return false;
    return a_.Match_(node->operand_);
  }

  [[nodiscard]] ExprPtr Eval() const {
    ExprPtr operand = a_.Eval();
    auto folded = TryConstFoldUnary(KindTrait<OpType>::kind, operand);
    if (folded) return folded;
    return MakeUnaryExprHelper<OpType>::Make(operand);
  }

 private:
  typename TA::Nested a_;
};

// ============================================================================
// PCastExpr — matches a Cast node with dtype and value sub-patterns
// ============================================================================

template <typename DType, typename TA>
class PCastExpr : public Pattern<PCastExpr<DType, TA>> {
 public:
  PCastExpr(const DType& dtype, const TA& value) : dtype_(dtype), value_(value) {}

  void InitMatch_() const {
    dtype_.InitMatch_();
    value_.InitMatch_();
  }

  [[nodiscard]] bool Match_(const ExprPtr& expr) const {
    auto node = As<Cast>(expr);
    if (!node) return false;
    DataType target_dtype = GetScalarDtype(node);
    if (!dtype_.Match_(target_dtype)) return false;
    if (!value_.Match_(node->operand_)) return false;
    return true;
  }

  [[nodiscard]] ExprPtr Eval() const { return MakeCast(value_.Eval(), dtype_.Eval()); }

 private:
  typename DType::Nested dtype_;
  typename TA::Nested value_;
};

template <typename DType, typename TA>
inline PCastExpr<DType, TA> cast(const Pattern<DType>& dtype, const Pattern<TA>& value) {
  return PCastExpr<DType, TA>(dtype.derived(), value.derived());
}

// ============================================================================
// Binary pattern operator overloads / named constructors
// ============================================================================

// Macro that generates three overloads for each binary pattern:
//   Pattern + Pattern
//   Pattern + int64_t  (wraps int in PConstWithTypeLike)
//   int64_t + Pattern
// NOLINTBEGIN(bugprone-macro-parentheses, readability/nolint)
#define PYPTO_PATTERN_BINARY_OP_EX(FuncName, NodeName, CheckStep)                                      \
  template <typename TA, typename TB>                                                                  \
  inline PBinaryExpr<NodeName, TA, TB> FuncName(const Pattern<TA>& a, const Pattern<TB>& b) {          \
    CheckStep;                                                                                         \
    return PBinaryExpr<NodeName, TA, TB>(a.derived(), b.derived());                                    \
  }                                                                                                    \
  template <typename TA>                                                                               \
  inline PBinaryExpr<NodeName, TA, PConstWithTypeLike<TA>> FuncName(const Pattern<TA>& a, int64_t b) { \
    CheckStep;                                                                                         \
    return FuncName(a, PConstWithTypeLike<TA>(a.derived(), b));                                        \
  }                                                                                                    \
  template <typename TA>                                                                               \
  inline PBinaryExpr<NodeName, PConstWithTypeLike<TA>, TA> FuncName(int64_t b, const Pattern<TA>& a) { \
    CheckStep;                                                                                         \
    return FuncName(PConstWithTypeLike<TA>(a.derived(), b), a);                                        \
  }                                                                                                    \
  template <typename TA>                                                                               \
  inline PBinaryExpr<NodeName, TA, PExprLiteral> FuncName(const Pattern<TA>& a, ExprPtr b) {           \
    CheckStep;                                                                                         \
    return FuncName(a, PExprLiteral(std::move(b)));                                                    \
  }                                                                                                    \
  template <typename TA>                                                                               \
  inline PBinaryExpr<NodeName, PExprLiteral, TA> FuncName(ExprPtr a, const Pattern<TA>& b) {           \
    CheckStep;                                                                                         \
    return FuncName(PExprLiteral(std::move(a)), b);                                                    \
  }

#define PYPTO_PATTERN_BINARY_OP(FuncName, NodeName) PYPTO_PATTERN_BINARY_OP_EX(FuncName, NodeName, )
// NOLINTEND(bugprone-macro-parentheses, readability/nolint)

// Arithmetic
PYPTO_PATTERN_BINARY_OP(operator+, Add)
PYPTO_PATTERN_BINARY_OP(operator-, Sub)
PYPTO_PATTERN_BINARY_OP(operator*, Mul)
PYPTO_PATTERN_BINARY_OP(min, Min)
PYPTO_PATTERN_BINARY_OP(max, Max)
PYPTO_PATTERN_BINARY_OP(pow, Pow)
PYPTO_PATTERN_BINARY_OP(floordiv, FloorDiv)
PYPTO_PATTERN_BINARY_OP(floormod, FloorMod)
PYPTO_PATTERN_BINARY_OP(floatdiv, FloatDiv)

// Comparisons
PYPTO_PATTERN_BINARY_OP(operator>, Gt)
PYPTO_PATTERN_BINARY_OP(operator>=, Ge)
PYPTO_PATTERN_BINARY_OP(operator<, Lt)
PYPTO_PATTERN_BINARY_OP(operator<=, Le)
PYPTO_PATTERN_BINARY_OP(operator==, Eq)
PYPTO_PATTERN_BINARY_OP(operator!=, Ne)

// Logical
PYPTO_PATTERN_BINARY_OP(operator&&, And)
PYPTO_PATTERN_BINARY_OP(operator||, Or)

// Bitwise
PYPTO_PATTERN_BINARY_OP(operator&, BitAnd)
PYPTO_PATTERN_BINARY_OP(operator|, BitOr)
PYPTO_PATTERN_BINARY_OP(operator^, BitXor)
PYPTO_PATTERN_BINARY_OP(operator<<, BitShiftLeft)
PYPTO_PATTERN_BINARY_OP(operator>>, BitShiftRight)

#undef PYPTO_PATTERN_BINARY_OP
#undef PYPTO_PATTERN_BINARY_OP_EX

// ============================================================================
// Unary pattern operator overloads
// ============================================================================

template <typename TA>
inline PUnaryExpr<Not, TA> operator!(const Pattern<TA>& a) {
  return PUnaryExpr<Not, TA>(a.derived());
}

template <typename TA>
inline PUnaryExpr<BitNot, TA> operator~(const Pattern<TA>& a) {
  return PUnaryExpr<BitNot, TA>(a.derived());
}

template <typename TA>
inline PUnaryExpr<Neg, TA> neg(const Pattern<TA>& a) {
  return PUnaryExpr<Neg, TA>(a.derived());
}

// ============================================================================
// PMatchesOneOf — try multiple source patterns for a single rewrite
// ============================================================================

template <typename... TPattern>
class PMatchesOneOf {
 public:
  explicit PMatchesOneOf(const TPattern&... patterns) : patterns_{patterns...} {}

  template <typename NodeType>
  bool Match(const NodeType& value) const {
    return Match(value, []() { return true; });
  }

  template <typename NodeType, typename Condition>
  bool Match(const NodeType& value, Condition cond) const {
    return MatchImpl(value, cond, std::make_index_sequence<sizeof...(TPattern)>());
  }

 private:
  template <typename NodeType, typename Condition>
  bool MatchImpl(const NodeType&, Condition, std::index_sequence<>) const {
    return false;
  }

  template <typename NodeType, typename Condition, size_t First, size_t... Rest>
  bool MatchImpl(const NodeType& value, Condition cond, std::index_sequence<First, Rest...>) const {
    return std::get<First>(patterns_).Match(value, cond) ||
           MatchImpl(value, cond, std::index_sequence<Rest...>());
  }

  // Store by value to avoid dangling references when patterns are temporaries.
  // PVar members inside the patterns are mutable, so Match() works on const copies.
  std::tuple<TPattern...> patterns_;
};

template <typename... TPattern>
inline std::enable_if_t<(std::is_base_of_v<Pattern<TPattern>, TPattern> && ... && true),
                        PMatchesOneOf<TPattern...>>
matches_one_of(const TPattern&... patterns) {
  return PMatchesOneOf<TPattern...>(patterns...);
}

// ============================================================================
// Rewrite macros — used by RewriteSimplifier (PR 3)
// ============================================================================

/// Evaluate a rewrite result: calls `.Eval()` on pattern expressions,
/// returns `ExprPtr` values directly.
template <typename T>
inline auto PatternEval(T&& val) -> decltype(val.Eval()) {
  return val.Eval();
}

inline ExprPtr PatternEval(const ExprPtr& val) { return val; }

// NOLINTBEGIN(cppcoreguidelines-macro-usage, readability/nolint)

/// Try to match SrcExpr against `ret`. On success, evaluate ResExpr and
/// return the result via RecursiveRewrite.
#define PYPTO_TRY_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {               \
    RecordAttemptedRewrite();               \
    auto r = PatternEval(ResExpr);          \
    RecordRewrite();                        \
    return RecursiveRewrite(r);             \
  }

/// Like PYPTO_TRY_REWRITE but with an additional guard condition.
#define PYPTO_TRY_REWRITE_IF(SrcExpr, ResExpr, Cond) \
  if ((SrcExpr).Match(ret)) {                        \
    RecordAttemptedRewrite();                        \
    if (Cond) {                                      \
      auto r = PatternEval(ResExpr);                 \
      RecordRewrite();                               \
      return RecursiveRewrite(r);                    \
    }                                                \
  }

// NOLINTEND(cppcoreguidelines-macro-usage, readability/nolint)

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_PATTERN_MATCH_H_
