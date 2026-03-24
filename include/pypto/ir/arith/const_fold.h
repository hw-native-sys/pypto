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

#ifndef PYPTO_IR_ARITH_CONST_FOLD_H_
#define PYPTO_IR_ARITH_CONST_FOLD_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {
namespace arith {

// ---------------------------------------------------------------------------
// Helper constructors for constant folding results
// ---------------------------------------------------------------------------

inline ExprPtr MakeConstInt(int64_t value, DataType dtype) {
  return std::make_shared<ConstInt>(value, dtype, Span::unknown());
}

inline ExprPtr MakeConstFloat(double value, DataType dtype) {
  return std::make_shared<ConstFloat>(value, dtype, Span::unknown());
}

inline ExprPtr MakeConstBool(bool value) { return std::make_shared<ConstBool>(value, Span::unknown()); }

// ---------------------------------------------------------------------------
// Overflow-safe integer arithmetic helpers
// ---------------------------------------------------------------------------

/// Check if a + b overflows int64_t.
inline bool AddWouldOverflow(int64_t a, int64_t b) {
  return (b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
         (b < 0 && a < std::numeric_limits<int64_t>::min() - b);
}

/// Check if a - b overflows int64_t.
inline bool SubWouldOverflow(int64_t a, int64_t b) {
  return (b > 0 && a < std::numeric_limits<int64_t>::min() + b) ||
         (b < 0 && a > std::numeric_limits<int64_t>::max() + b);
}

/// Check if a * b overflows int64_t.
inline bool MulWouldOverflow(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return false;
  if (b > 0) {
    if (a < std::numeric_limits<int64_t>::min() / b) return true;
    if (a > std::numeric_limits<int64_t>::max() / b) return true;
  } else {
    // b == -1 must be handled separately: INT64_MIN / (-1) is undefined behavior
    // (triggers SIGFPE on x86_64). When a == INT64_MIN, a * (-1) overflows;
    // otherwise -a is always representable.
    if (b == -1) return a == std::numeric_limits<int64_t>::min();
    if (a > std::numeric_limits<int64_t>::min() / b) return true;
    if (a < std::numeric_limits<int64_t>::max() / b) return true;
  }
  return false;
}

/// Check if -a overflows int64_t.
inline bool NegWouldOverflow(int64_t a) { return a == std::numeric_limits<int64_t>::min(); }

// ---------------------------------------------------------------------------
// detail — per-op fold helpers
// ---------------------------------------------------------------------------

namespace detail {

/// Bundled As<> casts for the two operands of a binary expression.
struct BinaryConstOperands {
  ExprPtr lhs, rhs;
  ConstIntPtr pa, pb;
  ConstFloatPtr fa, fb;
};

// ---- Generic fold patterns (merge int & float behavior) ------------------

/// Fold a comparison that works on both int and float, returning ConstBool.
template <typename Op>
inline ExprPtr FoldCompare(const BinaryConstOperands& c, Op op) {
  if (c.pa && c.pb) return MakeConstBool(op(c.pa->value_, c.pb->value_));
  if (c.fa && c.fb) return MakeConstBool(op(c.fa->value_, c.fb->value_));
  return nullptr;
}

/// Fold an int-only binary op (bitwise / shifts).
template <typename Op>
inline ExprPtr FoldIntOnly(const BinaryConstOperands& c, Op op) {
  if (c.pa && c.pb) return MakeConstInt(op(c.pa->value_, c.pb->value_), c.pa->dtype());
  return nullptr;
}

// ---- Identity helpers (one operand is a known constant) ------------------

/// True if lhs is a constant equal to v (int or float).
inline bool LhsIsConst(const BinaryConstOperands& c, int64_t v) {
  return (c.pa && c.pa->value_ == v) || (c.fa && c.fa->value_ == static_cast<double>(v));
}

/// True if rhs is a constant equal to v (int or float).
inline bool RhsIsConst(const BinaryConstOperands& c, int64_t v) {
  return (c.pb && c.pb->value_ == v) || (c.fb && c.fb->value_ == static_cast<double>(v));
}

// ---- Per-op fold functions -----------------------------------------------

inline ExprPtr FoldAdd(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    if (!AddWouldOverflow(c.pa->value_, c.pb->value_)) {
      return MakeConstInt(c.pa->value_ + c.pb->value_, c.pa->dtype());
    }
    return nullptr;  // overflow — skip folding
  }
  if (c.fa && c.fb) return MakeConstFloat(c.fa->value_ + c.fb->value_, c.fa->dtype());
  if (LhsIsConst(c, 0)) return c.rhs;  // 0 + x → x
  if (RhsIsConst(c, 0)) return c.lhs;  // x + 0 → x
  return nullptr;
}

inline ExprPtr FoldSub(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    if (!SubWouldOverflow(c.pa->value_, c.pb->value_)) {
      return MakeConstInt(c.pa->value_ - c.pb->value_, c.pa->dtype());
    }
    return nullptr;
  }
  if (c.fa && c.fb) return MakeConstFloat(c.fa->value_ - c.fb->value_, c.fa->dtype());
  if (RhsIsConst(c, 0)) return c.lhs;  // x - 0 → x
  return nullptr;
}

inline ExprPtr FoldMul(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    if (!MulWouldOverflow(c.pa->value_, c.pb->value_)) {
      return MakeConstInt(c.pa->value_ * c.pb->value_, c.pa->dtype());
    }
    return nullptr;
  }
  if (c.fa && c.fb) return MakeConstFloat(c.fa->value_ * c.fb->value_, c.fa->dtype());
  if (LhsIsConst(c, 1)) return c.rhs;  // 1 * x → x
  if (RhsIsConst(c, 1)) return c.lhs;  // x * 1 → x
  // Integer 0 * x → 0, x * 0 → 0 (well-defined for integers).
  // Float 0.0 * x is NOT folded because NaN * 0 = NaN, inf * 0 = NaN per IEEE 754.
  if (c.pa && c.pa->value_ == 0) return c.lhs;
  if (c.pb && c.pb->value_ == 0) return c.rhs;
  return nullptr;
}

inline ExprPtr FoldFloorDiv(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    CHECK(c.pb->value_ != 0) << "Floor division by zero";
    CHECK(!(c.pa->value_ == std::numeric_limits<int64_t>::min() && c.pb->value_ == -1))
        << "Floor division overflow: INT64_MIN // -1";
    return MakeConstInt(floordiv(c.pa->value_, c.pb->value_), c.pa->dtype());
  }
  // NOT folding 0 // x → 0: x could be 0 at runtime, must preserve the division.
  if (c.pb && c.pb->value_ == 1) return c.lhs;  // x // 1 → x
  if (c.pb) {
    CHECK(c.pb->value_ != 0) << "Floor division by zero";
  }
  return nullptr;
}

inline ExprPtr FoldFloorMod(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    CHECK(c.pb->value_ != 0) << "Floor modulo by zero";
    CHECK(!(c.pa->value_ == std::numeric_limits<int64_t>::min() && c.pb->value_ == -1))
        << "Floor modulo overflow: INT64_MIN % -1";
    return MakeConstInt(floormod(c.pa->value_, c.pb->value_), c.pa->dtype());
  }
  // NOT folding 0 % x → 0: x could be 0 at runtime, must preserve the modulo.
  if (c.pb && c.pb->value_ == 1) return MakeConstInt(0, c.pb->dtype());  // x % 1 → 0
  if (c.pb) {
    CHECK(c.pb->value_ != 0) << "Floor modulo by zero";
  }
  return nullptr;
}

inline ExprPtr FoldFloatDiv(const BinaryConstOperands& c) {
  if (c.fa && c.fb) {
    // IEEE 754: division by zero produces +/-inf, not an error.
    return MakeConstFloat(c.fa->value_ / c.fb->value_, c.fa->dtype());
  }
  // NOT folding 0.0 / x → 0.0: if x is 0.0 or NaN, result should be NaN per IEEE 754.
  if (c.fb && c.fb->value_ == 1) return c.lhs;  // x / 1.0 → x
  return nullptr;
}

/// Integer power via exponentiation by squaring, with overflow checking.
/// Returns nullptr if overflow occurs.
inline ExprPtr SafeIntPow(int64_t base, int64_t exp, DataType dtype) {
  CHECK(exp >= 0) << "Integer power requires non-negative exponent, got " << exp;
  int64_t result = 1;
  int64_t cur = base;
  while (exp > 0) {
    if (exp & 1) {
      if (MulWouldOverflow(result, cur)) return nullptr;
      result *= cur;
    }
    exp >>= 1;
    if (exp > 0) {
      if (MulWouldOverflow(cur, cur)) return nullptr;
      cur *= cur;
    }
  }
  return MakeConstInt(result, dtype);
}

inline ExprPtr FoldPow(const BinaryConstOperands& c) {
  if (c.pa && c.pb) return SafeIntPow(c.pa->value_, c.pb->value_, c.pa->dtype());
  if (c.fa && c.fb) return MakeConstFloat(std::pow(c.fa->value_, c.fb->value_), c.fa->dtype());
  if (LhsIsConst(c, 1)) return c.lhs;  // 1^x → 1
  if (c.pb && c.pb->value_ == 0) {     // x^0 → 1
    DataType dtype = c.pa ? c.pa->dtype() : GetScalarDtype(c.lhs);
    if (dtype.IsFloat()) return MakeConstFloat(1.0, dtype);
    return MakeConstInt(1, dtype);
  }
  if (RhsIsConst(c, 1)) return c.lhs;  // x^1 → x
  return nullptr;
}

inline ExprPtr FoldMin(const BinaryConstOperands& c) {
  if (c.pa && c.pb) return MakeConstInt(std::min(c.pa->value_, c.pb->value_), c.pa->dtype());
  if (c.fa && c.fb) return MakeConstFloat(std::min(c.fa->value_, c.fb->value_), c.fa->dtype());
  if (c.lhs.get() == c.rhs.get()) return c.lhs;  // min(x, x) → x
  return nullptr;
}

inline ExprPtr FoldMax(const BinaryConstOperands& c) {
  if (c.pa && c.pb) return MakeConstInt(std::max(c.pa->value_, c.pb->value_), c.pa->dtype());
  if (c.fa && c.fb) return MakeConstFloat(std::max(c.fa->value_, c.fb->value_), c.fa->dtype());
  if (c.lhs.get() == c.rhs.get()) return c.lhs;  // max(x, x) → x
  return nullptr;
}

// ---- Logical ops (use ConstBool, not ConstInt/ConstFloat) ----------------

inline ExprPtr FoldAnd(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto ba = As<ConstBool>(lhs);
  auto bb = As<ConstBool>(rhs);
  if (ba && ba->value_) return rhs;   // true  && x → x
  if (ba && !ba->value_) return lhs;  // false && x → false
  if (bb && bb->value_) return lhs;   // x && true  → x
  if (bb && !bb->value_) return rhs;  // x && false → false
  return nullptr;
}

inline ExprPtr FoldOr(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto ba = As<ConstBool>(lhs);
  auto bb = As<ConstBool>(rhs);
  if (ba && ba->value_) return lhs;   // true  || x → true
  if (ba && !ba->value_) return rhs;  // false || x → x
  if (bb && bb->value_) return rhs;   // x || true  → true
  if (bb && !bb->value_) return lhs;  // x || false → x
  return nullptr;
}

inline ExprPtr FoldXor(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto ba = As<ConstBool>(lhs);
  auto bb = As<ConstBool>(rhs);
  if (ba && bb) return MakeConstBool(ba->value_ != bb->value_);
  return nullptr;
}

/// Safe shift: skip folding for negative or out-of-range shift counts.
inline ExprPtr FoldShiftLeft(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    constexpr int64_t kBitWidth = static_cast<int64_t>(sizeof(int64_t) * 8);
    if (c.pb->value_ < 0 || c.pb->value_ >= kBitWidth) return nullptr;
    // Use unsigned to avoid UB on left-shift of negative values.
    auto result = static_cast<int64_t>(static_cast<uint64_t>(c.pa->value_) << c.pb->value_);
    return MakeConstInt(result, c.pa->dtype());
  }
  return nullptr;
}

inline ExprPtr FoldShiftRight(const BinaryConstOperands& c) {
  if (c.pa && c.pb) {
    constexpr int64_t kBitWidth = static_cast<int64_t>(sizeof(int64_t) * 8);
    if (c.pb->value_ < 0 || c.pb->value_ >= kBitWidth) return nullptr;
    return MakeConstInt(c.pa->value_ >> c.pb->value_, c.pa->dtype());
  }
  return nullptr;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// TryConstFoldBinary — thin dispatch to per-op helpers
// ---------------------------------------------------------------------------

/// Try to constant-fold a binary operation.
/// Returns nullptr if no fold applies.
inline ExprPtr TryConstFoldBinary(ObjectKind kind, const ExprPtr& lhs, const ExprPtr& rhs) {
  // Logical ops use ConstBool, not ConstInt/ConstFloat — handle separately.
  switch (kind) {
    case ObjectKind::And:
      return detail::FoldAnd(lhs, rhs);
    case ObjectKind::Or:
      return detail::FoldOr(lhs, rhs);
    case ObjectKind::Xor:
      return detail::FoldXor(lhs, rhs);
    default:
      break;
  }

  detail::BinaryConstOperands c{
      lhs, rhs, As<ConstInt>(lhs), As<ConstInt>(rhs), As<ConstFloat>(lhs), As<ConstFloat>(rhs)};

  switch (kind) {
    // Arithmetic
    case ObjectKind::Add:
      return detail::FoldAdd(c);
    case ObjectKind::Sub:
      return detail::FoldSub(c);
    case ObjectKind::Mul:
      return detail::FoldMul(c);
    case ObjectKind::FloorDiv:
      return detail::FoldFloorDiv(c);
    case ObjectKind::FloorMod:
      return detail::FoldFloorMod(c);
    case ObjectKind::FloatDiv:
      return detail::FoldFloatDiv(c);
    case ObjectKind::Pow:
      return detail::FoldPow(c);
    case ObjectKind::Min:
      return detail::FoldMin(c);
    case ObjectKind::Max:
      return detail::FoldMax(c);
    // Comparisons
    case ObjectKind::Eq:
      return detail::FoldCompare(c, std::equal_to<>{});
    case ObjectKind::Ne:
      return detail::FoldCompare(c, std::not_equal_to<>{});
    case ObjectKind::Lt:
      return detail::FoldCompare(c, std::less<>{});
    case ObjectKind::Le:
      return detail::FoldCompare(c, std::less_equal<>{});
    case ObjectKind::Gt:
      return detail::FoldCompare(c, std::greater<>{});
    case ObjectKind::Ge:
      return detail::FoldCompare(c, std::greater_equal<>{});
    // Bitwise
    case ObjectKind::BitAnd:
      return detail::FoldIntOnly(c, std::bit_and<int64_t>{});
    case ObjectKind::BitOr:
      return detail::FoldIntOnly(c, std::bit_or<int64_t>{});
    case ObjectKind::BitXor:
      return detail::FoldIntOnly(c, std::bit_xor<int64_t>{});
    case ObjectKind::BitShiftLeft:
      return detail::FoldShiftLeft(c);
    case ObjectKind::BitShiftRight:
      return detail::FoldShiftRight(c);
    default:
      return nullptr;
  }
}

// ---------------------------------------------------------------------------
// TryConstFoldUnary — fold unary operations on constant operands
// ---------------------------------------------------------------------------

/// Try to constant-fold a unary operation.
/// Returns nullptr if the operand is not constant or if no fold applies.
inline ExprPtr TryConstFoldUnary(ObjectKind kind, const ExprPtr& operand) {
  auto pi = As<ConstInt>(operand);
  auto pf = As<ConstFloat>(operand);
  auto pb = As<ConstBool>(operand);

  switch (kind) {
    case ObjectKind::Neg:
      if (pi) {
        if (NegWouldOverflow(pi->value_)) return nullptr;
        return MakeConstInt(-pi->value_, pi->dtype());
      }
      if (pf) return MakeConstFloat(-pf->value_, pf->dtype());
      return nullptr;
    case ObjectKind::Abs:
      if (pi) {
        if (NegWouldOverflow(pi->value_)) return nullptr;  // abs(INT64_MIN) overflows
        return MakeConstInt(pi->value_ >= 0 ? pi->value_ : -pi->value_, pi->dtype());
      }
      if (pf) return MakeConstFloat(std::fabs(pf->value_), pf->dtype());
      return nullptr;
    case ObjectKind::Not:
      if (pb) return MakeConstBool(!pb->value_);
      return nullptr;
    case ObjectKind::BitNot:
      if (pi) return MakeConstInt(~pi->value_, pi->dtype());
      return nullptr;
    default:
      return nullptr;
  }
}

// ---------------------------------------------------------------------------
// TryConstFold — unified entry point: inspect expression, dispatch to above
// ---------------------------------------------------------------------------

/// Try to constant-fold an expression node.
/// Accepts any BinaryExpr or UnaryExpr; returns nullptr for non-foldable nodes.
inline ExprPtr TryConstFold(const ExprPtr& expr) {
  if (auto bin = As<BinaryExpr>(expr)) {
    return TryConstFoldBinary(expr->GetKind(), bin->left_, bin->right_);
  }
  if (auto un = As<UnaryExpr>(expr)) {
    return TryConstFoldUnary(expr->GetKind(), un->operand_);
  }
  return nullptr;
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_CONST_FOLD_H_
