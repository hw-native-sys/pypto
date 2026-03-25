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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace arith {

using Bound = ConstIntBound;

// ============================================================================
// Overflow-safe arithmetic for bound propagation
// ============================================================================

// Note: InfAwareAdd(kPosInf, kNegInf) returns kPosInf. This is an indeterminate
// case (+inf + -inf), but in practice valid bounds always have min <= max, so
// this path is not hit in normal operation. This matches TVM's behavior.
static int64_t InfAwareAdd(int64_t a, int64_t b) {
  if (a == Bound::kPosInf || b == Bound::kPosInf) return Bound::kPosInf;
  if (a == Bound::kNegInf || b == Bound::kNegInf) return Bound::kNegInf;
  if (AddWouldOverflow(a, b)) {
    return (b > 0) ? Bound::kPosInf : Bound::kNegInf;
  }
  return a + b;
}

static int64_t InfAwareMul(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return 0;
  bool pos_result = (a > 0) == (b > 0);
  if (a == Bound::kPosInf || a == Bound::kNegInf || b == Bound::kPosInf || b == Bound::kNegInf) {
    return pos_result ? Bound::kPosInf : Bound::kNegInf;
  }
  if (MulWouldOverflow(a, b)) {
    return pos_result ? Bound::kPosInf : Bound::kNegInf;
  }
  return a * b;
}

static int64_t InfAwareNeg(int64_t a) {
  if (a == Bound::kPosInf) return Bound::kNegInf;
  if (a == Bound::kNegInf) return Bound::kPosInf;
  if (NegWouldOverflow(a)) return Bound::kPosInf;  // -INT64_MIN overflows
  return -a;
}

/// Overflow-safe exponentiation by squaring — O(log e).
static int64_t InfAwarePow(int64_t base, int64_t exp) {
  if (exp == 0) return 1;
  if (base == 0) return 0;
  if (base == 1) return 1;
  int64_t result = 1;
  int64_t b = base;
  int64_t e = exp;
  while (e > 0) {
    if (e & 1) {
      result = InfAwareMul(result, b);
      if (result == Bound::kPosInf || result == Bound::kNegInf) return result;
    }
    e >>= 1;
    if (e > 0) {
      b = InfAwareMul(b, b);
      if (b == Bound::kPosInf || b == Bound::kNegInf) {
        // Base overflowed; remaining result depends on remaining exponent parity
        return (result > 0) == (base > 0 || (exp & 1) == 0) ? Bound::kPosInf : Bound::kNegInf;
      }
    }
  }
  return result;
}

// ============================================================================
// Compound bound helpers
// ============================================================================

/// Return the "everything is possible" bound.
static Bound Everything() { return {Bound::kNegInf, Bound::kPosInf}; }

/// Four-corner multiplication of two bounds.
static Bound BoundMul(const Bound& a, const Bound& b) {
  int64_t v0 = InfAwareMul(a.min_value, b.min_value);
  int64_t v1 = InfAwareMul(a.min_value, b.max_value);
  int64_t v2 = InfAwareMul(a.max_value, b.min_value);
  int64_t v3 = InfAwareMul(a.max_value, b.max_value);
  return {std::min({v0, v1, v2, v3}), std::max({v0, v1, v2, v3})};
}

/// Floor-division bounds (positive divisor).
static Bound BoundFloorDivPositiveRHS(const Bound& a, const Bound& b) {
  // b is guaranteed > 0
  // floordiv(a, b) is non-decreasing in a.
  // For a >= 0: non-increasing in b.  For a < 0: non-decreasing in b.
  int64_t min_val =
      (a.min_value >= 0) ? floordiv(a.min_value, b.max_value) : floordiv(a.min_value, b.min_value);
  int64_t max_val =
      (a.max_value >= 0) ? floordiv(a.max_value, b.min_value) : floordiv(a.max_value, b.max_value);
  return {min_val, max_val};
}

/// Floor-division bounds (general).
static Bound BoundFloorDiv(const Bound& a, const Bound& b) {
  if (b.min_value > 0) {
    return BoundFloorDivPositiveRHS(a, b);
  }
  if (b.max_value < 0) {
    // Negative divisor: floordiv(a, -|b|) = floordiv(-a, |b|)
    Bound neg_a = {InfAwareNeg(a.max_value), InfAwareNeg(a.min_value)};
    Bound pos_b = {InfAwareNeg(b.max_value), InfAwareNeg(b.min_value)};
    return BoundFloorDivPositiveRHS(neg_a, pos_b);
  }
  // Divisor range includes zero — cannot bound.
  return Everything();
}

/// Floor-mod bounds (positive divisor).
static Bound BoundFloorMod(const Bound& a, const Bound& b) {
  if (b.min_value > 0) {
    if (a.is_non_negative()) {
      // 0 <= a % b <= min(a_max, b_max - 1)
      return {0, std::min(a.max_value, b.max_value - 1)};
    }
    // General: result in [0, b_max - 1] for floor mod with positive divisor
    // when a could be negative.
    // floormod always returns value in [0, b) for b > 0.
    return {0, b.max_value - 1};
  }
  return Everything();
}

// ============================================================================
// Implementation class — extends ExprFunctor<ConstIntBound>
// ============================================================================

class ConstIntBoundAnalyzer::Impl : public ExprFunctor<Bound> {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  void Bind(const VarPtr& var, const Bound& bound) { var_map_[var.get()] = bound; }

  void Unbind(const VarPtr& var) { var_map_.erase(var.get()); }

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

  Bound VisitExpr(const ExprPtr& expr) override { return ExprFunctor<Bound>::VisitExpr(expr); }

 protected:
  // --- Leaf nodes ---

  Bound VisitExpr_(const ConstIntPtr& op) override { return {op->value_, op->value_}; }

  Bound VisitExpr_(const ConstFloatPtr& op) override {
    double v = op->value_;
    if (std::isnan(v) || std::isinf(v)) return Everything();
    // Guard against finite values outside int64_t range (cast would be UB)
    constexpr auto kMax = static_cast<double>(Bound::kPosInf);
    constexpr auto kMin = static_cast<double>(Bound::kNegInf);
    double lo = std::floor(v);
    double hi = std::ceil(v);
    int64_t lo_i = (lo < kMin) ? Bound::kNegInf : static_cast<int64_t>(lo);
    int64_t hi_i = (hi > kMax) ? Bound::kPosInf : static_cast<int64_t>(hi);
    return {lo_i, hi_i};
  }

  Bound VisitExpr_(const ConstBoolPtr& op) override {
    int64_t v = op->value_ ? 1 : 0;
    return {v, v};
  }

  Bound VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return Everything();
  }

  Bound VisitExpr_(const IterArgPtr& op) override {
    // IterArg is a Var subclass — look up in var_map_
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return Everything();
  }

  Bound VisitExpr_(const MemRefPtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const CallPtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const MakeTuplePtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const TupleGetItemExprPtr& /*op*/) override { return Everything(); }

  // --- Binary arithmetic ---

  Bound VisitExpr_(const AddPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {InfAwareAdd(a.min_value, b.min_value), InfAwareAdd(a.max_value, b.max_value)};
  }

  Bound VisitExpr_(const SubPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {InfAwareAdd(a.min_value, InfAwareNeg(b.max_value)),
            InfAwareAdd(a.max_value, InfAwareNeg(b.min_value))};
  }

  Bound VisitExpr_(const MulPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return BoundMul(a, b);
  }

  Bound VisitExpr_(const FloorDivPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return BoundFloorDiv(a, b);
  }

  Bound VisitExpr_(const FloorModPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return BoundFloorMod(a, b);
  }

  Bound VisitExpr_(const FloatDivPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const MinPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {std::min(a.min_value, b.min_value), std::min(a.max_value, b.max_value)};
  }

  Bound VisitExpr_(const MaxPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {std::max(a.min_value, b.min_value), std::max(a.max_value, b.max_value)};
  }

  Bound VisitExpr_(const PowPtr& op) override {
    auto base = VisitExpr(op->left_);
    auto exp = VisitExpr(op->right_);
    // Only handle constant non-negative exponent
    if (!exp.is_const() || exp.min_value < 0) return Everything();
    int64_t e = exp.min_value;
    if (e == 0) return {1, 1};
    if (e == 1) return base;
    if (base.is_non_negative()) {
      // O(log e) exponentiation by squaring for bound propagation
      return {InfAwarePow(base.min_value, e), InfAwarePow(base.max_value, e)};
    }
    return Everything();
  }

  // --- Comparisons (always boolean result) ---

  Bound VisitExpr_(const EqPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const NePtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const LtPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const LePtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const GtPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const GePtr& /*op*/) override { return {0, 1}; }

  // --- Logical (boolean result) ---

  Bound VisitExpr_(const AndPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const OrPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const XorPtr& /*op*/) override { return {0, 1}; }

  // --- Bitwise (conservative) ---

  Bound VisitExpr_(const BitAndPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // If both non-negative, result is in [0, min(a_max, b_max)]
    if (a.is_non_negative() && b.is_non_negative()) {
      return {0, std::min(a.max_value, b.max_value)};
    }
    return Everything();
  }

  Bound VisitExpr_(const BitOrPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative()) {
      // Upper bound: next power-of-2 above max of both, minus 1
      // Conservative: use sum as loose upper bound
      return {std::max(a.min_value, b.min_value), InfAwareAdd(a.max_value, b.max_value)};
    }
    return Everything();
  }

  Bound VisitExpr_(const BitXorPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const BitShiftLeftPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative() && b.max_value < 63) {
      return {InfAwareMul(a.min_value, static_cast<int64_t>(1) << b.min_value),
              InfAwareMul(a.max_value, static_cast<int64_t>(1) << b.max_value)};
    }
    return Everything();
  }

  Bound VisitExpr_(const BitShiftRightPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative() && b.max_value < 63) {
      return {a.min_value >> b.max_value, a.max_value >> b.min_value};
    }
    return Everything();
  }

  // --- Unary ---

  Bound VisitExpr_(const NegPtr& op) override {
    auto a = VisitExpr(op->operand_);
    return {InfAwareNeg(a.max_value), InfAwareNeg(a.min_value)};
  }

  Bound VisitExpr_(const AbsPtr& op) override {
    auto a = VisitExpr(op->operand_);
    if (a.min_value >= 0) return a;
    if (a.max_value <= 0) return {InfAwareNeg(a.max_value), InfAwareNeg(a.min_value)};
    return {0, std::max(InfAwareNeg(a.min_value), a.max_value)};
  }

  Bound VisitExpr_(const NotPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const BitNotPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const CastPtr& op) override {
    auto a = VisitExpr(op->operand_);
    // For Cast to integer type, intersect with the target type's representable range
    auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(op->GetType());
    if (!scalar_type || !scalar_type->dtype_.IsInt()) return Everything();
    size_t bits = scalar_type->dtype_.GetBit();
    if (bits == 0 || bits >= 64) return a;  // Unknown or >= 64 bits: pass through
    int64_t type_min, type_max;
    if (scalar_type->dtype_.IsUnsignedInt()) {
      type_min = 0;
      type_max = (static_cast<int64_t>(1) << bits) - 1;
    } else {
      type_min = -(static_cast<int64_t>(1) << (bits - 1));
      type_max = (static_cast<int64_t>(1) << (bits - 1)) - 1;
    }
    return {std::max(a.min_value, type_min), std::min(a.max_value, type_max)};
  }

 private:
  Analyzer* parent_;
  std::unordered_map<const Expr*, Bound> var_map_;
};

// ============================================================================
// EnterConstraint — parse comparison expressions to tighten variable bounds
// ============================================================================

std::function<void()> ConstIntBoundAnalyzer::Impl::EnterConstraint(const ExprPtr& constraint) {
  std::vector<std::pair<const Expr*, Bound>> recovery;

  // Helper: try to tighten bound for a variable.
  auto TryTighten = [&](const Expr* var_ptr, const Bound& new_bound) {
    auto it = var_map_.find(var_ptr);
    Bound old = (it != var_map_.end()) ? it->second : Everything();
    recovery.emplace_back(var_ptr, old);
    var_map_[var_ptr] = {std::max(old.min_value, new_bound.min_value),
                         std::min(old.max_value, new_bound.max_value)};
  };

  // Try to parse simple comparisons: var >= const, var < const, etc.
  // Must use std::function for recursive calls (And case).
  std::function<void(const ExprPtr&)> TryParseConstraint = [&](const ExprPtr& expr) {
    // Ge: left >= right
    if (auto ge = As<Ge>(expr)) {
      if (auto var = As<Var>(ge->left_)) {
        auto rb = VisitExpr(ge->right_);
        if (rb.is_const()) TryTighten(var.get(), {rb.min_value, Bound::kPosInf});
      } else if (auto var = As<Var>(ge->right_)) {
        auto lb = VisitExpr(ge->left_);
        if (lb.is_const()) TryTighten(var.get(), {Bound::kNegInf, lb.max_value});
      }
      return;
    }
    // Gt: left > right
    if (auto gt = As<Gt>(expr)) {
      if (auto var = As<Var>(gt->left_)) {
        auto rb = VisitExpr(gt->right_);
        if (rb.is_const()) TryTighten(var.get(), {InfAwareAdd(rb.min_value, 1), Bound::kPosInf});
      } else if (auto var = As<Var>(gt->right_)) {
        auto lb = VisitExpr(gt->left_);
        if (lb.is_const()) TryTighten(var.get(), {Bound::kNegInf, InfAwareAdd(lb.max_value, -1)});
      }
      return;
    }
    // Le: left <= right
    if (auto le = As<Le>(expr)) {
      if (auto var_l = As<Var>(le->left_)) {
        auto rb = VisitExpr(le->right_);
        if (rb.is_const()) TryTighten(var_l.get(), {Bound::kNegInf, rb.max_value});
      } else if (auto var_r = As<Var>(le->right_)) {
        auto lb = VisitExpr(le->left_);
        if (lb.is_const()) TryTighten(var_r.get(), {lb.min_value, Bound::kPosInf});
      }
      return;
    }
    // Lt: left < right
    if (auto lt = As<Lt>(expr)) {
      if (auto var = As<Var>(lt->left_)) {
        auto rb = VisitExpr(lt->right_);
        if (rb.is_const()) TryTighten(var.get(), {Bound::kNegInf, InfAwareAdd(rb.max_value, -1)});
      } else if (auto var = As<Var>(lt->right_)) {
        auto lb = VisitExpr(lt->left_);
        if (lb.is_const()) TryTighten(var.get(), {InfAwareAdd(lb.min_value, 1), Bound::kPosInf});
      }
      return;
    }
    // Eq: left == right
    if (auto eq = As<Eq>(expr)) {
      if (auto var = As<Var>(eq->left_)) {
        auto rb = VisitExpr(eq->right_);
        if (rb.is_const()) TryTighten(var.get(), rb);
      } else if (auto var = As<Var>(eq->right_)) {
        auto lb = VisitExpr(eq->left_);
        if (lb.is_const()) TryTighten(var.get(), lb);
      }
      return;
    }
    // And: both constraints hold
    if (auto and_op = As<And>(expr)) {
      TryParseConstraint(and_op->left_);
      TryParseConstraint(and_op->right_);
    }
    // Note: Not(comparison) is handled by ConstraintContext, which normalizes
    // constraints via rewrite_simplify before dispatching to sub-analyzers.
    // E.g., Not(Lt(a,b)) is rewritten to Ge(a,b) before reaching this code.
  };

  TryParseConstraint(constraint);

  // Return recovery function — iterate in reverse so that when the same variable
  // was tightened multiple times (e.g., And(x >= 0, x < 8)), we restore the
  // original pre-scope bound, not the intermediate one.
  return [this, recovery = std::move(recovery)]() {
    for (auto it = recovery.rbegin(); it != recovery.rend(); ++it) {
      const auto& [ptr, bound] = *it;
      if (bound.is_everything()) {
        var_map_.erase(ptr);
      } else {
        var_map_[ptr] = bound;
      }
    }
  };
}

// ============================================================================
// ConstIntBoundAnalyzer — public interface delegation to Impl
// ============================================================================

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer() : impl_(std::make_unique<Impl>(nullptr)) {}

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

ConstIntBoundAnalyzer::~ConstIntBoundAnalyzer() = default;

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(ConstIntBoundAnalyzer&&) noexcept = default;
ConstIntBoundAnalyzer& ConstIntBoundAnalyzer::operator=(ConstIntBoundAnalyzer&&) noexcept = default;

Bound ConstIntBoundAnalyzer::operator()(const ExprPtr& expr) const { return impl_->VisitExpr(expr); }

void ConstIntBoundAnalyzer::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive) {
  CHECK(max_val_exclusive > min_val) << "Bind requires max_val_exclusive > min_val, got [" << min_val << ", "
                                     << max_val_exclusive << ")";
  impl_->Bind(var, {min_val, max_val_exclusive - 1});
}

void ConstIntBoundAnalyzer::Update(const VarPtr& var, const Bound& bound) {
  CHECK(bound.min_value <= bound.max_value)
      << "Update requires min_value <= max_value, got [" << bound.min_value << ", " << bound.max_value << "]";
  impl_->Bind(var, bound);
}

void ConstIntBoundAnalyzer::Unbind(const VarPtr& var) { impl_->Unbind(var); }

std::function<void()> ConstIntBoundAnalyzer::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
