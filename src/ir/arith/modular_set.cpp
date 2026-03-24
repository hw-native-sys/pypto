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

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <unordered_map>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// Internal Entry type with normalization
// ============================================================================

/// Internal representation: value = coeff * k + base.
/// coeff >= 0; when coeff > 0, base is normalized to [0, coeff).
struct Entry {
  int64_t coeff{1};
  int64_t base{0};

  Entry() = default;

  Entry(int64_t c, int64_t b) {
    if (c < 0) {
      c = -c;
      b = -b;
    }
    coeff = c;
    if (coeff != 0) {
      b = b % coeff;
      if (b < 0) b += coeff;
    }
    base = b;
  }

  [[nodiscard]] bool is_const() const { return coeff == 0; }
};

/// No useful modular info.
static Entry Everything() { return {1, 0}; }

/// Union of two modular sets: values that could come from either set.
static Entry Union(const Entry& a, const Entry& b) {
  // {a.coeff * x + a.base} ∪ {b.coeff * y + b.base}
  //   => {gcd(a.coeff, b.coeff) * z + base}
  int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
  if (coeff == 0) {
    if (a.base == b.base) return a;
    return Everything();
  }
  int64_t base0 = a.base % coeff;
  int64_t base1 = b.base % coeff;
  if (base0 == base1) {
    return {coeff, base0};
  }
  // Fallback: use gcd(coeff, |base0 - base1|) to capture patterns like
  // {4k+1} ∪ {4k+3} = {2k+1} (odd numbers).
  return {ZeroAwareGCD(coeff, std::abs(base0 - base1)), base0};
}

/// Intersection of two modular sets using the Chinese Remainder Theorem.
static Entry Intersect(const Entry& a, const Entry& b) {
  int64_t x, y;
  int64_t c1 = a.coeff, b1 = a.base, c2 = b.coeff, b2 = b.base;
  int64_t gcd = ExtendedEuclidean(c1, c2, &x, &y);
  int64_t v = b2 - b1;
  if (gcd != 0 && v % gcd == 0) {
    x = v / gcd * x;
    int64_t coeff = c1 / gcd * c2;
    return {coeff, x * c1 + b1};
  }
  // No solution via CRT — conservatively return everything.
  return Everything();
}

// ============================================================================
// Implementation class — extends ExprFunctor<Entry>
// ============================================================================

class ModularSetAnalyzer::Impl : public ExprFunctor<Entry> {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  void Update(const VarPtr& var, const Entry& entry) { var_map_[var.get()] = entry; }

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

  Entry VisitExpr(const ExprPtr& expr) override { return ExprFunctor<Entry>::VisitExpr(expr); }

 protected:
  // --- Leaf nodes ---

  Entry VisitExpr_(const ConstIntPtr& op) override { return {0, op->value_}; }

  Entry VisitExpr_(const ConstFloatPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const ConstBoolPtr& op) override { return {0, op->value_ ? 1 : 0}; }

  Entry VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return Everything();
  }

  Entry VisitExpr_(const IterArgPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return Everything();
  }

  Entry VisitExpr_(const MemRefPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const CallPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const MakeTuplePtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const TupleGetItemExprPtr& /*op*/) override { return Everything(); }

  // --- Binary arithmetic ---

  Entry VisitExpr_(const AddPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (AddWouldOverflow(a.base, b.base)) return Everything();
    return {ZeroAwareGCD(a.coeff, b.coeff), a.base + b.base};
  }

  Entry VisitExpr_(const SubPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (SubWouldOverflow(a.base, b.base)) return Everything();
    return {ZeroAwareGCD(a.coeff, b.coeff), a.base - b.base};
  }

  Entry VisitExpr_(const MulPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // (p*x + n) * (q*y + m) = pq*xy + pm*x + qn*y + nm
    // coeff = gcd(pq, pm, qn), base = nm
    if (MulWouldOverflow(a.base, b.base)) return Everything();
    if (MulWouldOverflow(a.coeff, b.coeff) || MulWouldOverflow(a.coeff, b.base) ||
        MulWouldOverflow(a.base, b.coeff)) {
      return Everything();
    }
    int64_t pq = a.coeff * b.coeff;
    int64_t pm = a.coeff * b.base;
    int64_t qn = a.base * b.coeff;
    return {ZeroAwareGCD(pq, ZeroAwareGCD(pm, qn)), a.base * b.base};
  }

  Entry VisitExpr_(const FloorDivPtr& op) override {
    auto b = VisitExpr(op->right_);
    if (b.is_const() && b.base != 0) {
      return FloorDivByConst(op->left_, b.base);
    }
    return Everything();
  }

  Entry VisitExpr_(const FloorModPtr& op) override {
    auto b = VisitExpr(op->right_);
    if (b.is_const() && b.base != 0) {
      return FloorModByConst(op->left_, b.base);
    }
    return Everything();
  }

  Entry VisitExpr_(const FloatDivPtr& /*op*/) override { return Everything(); }

  Entry VisitExpr_(const MinPtr& op) override { return Union(VisitExpr(op->left_), VisitExpr(op->right_)); }

  Entry VisitExpr_(const MaxPtr& op) override { return Union(VisitExpr(op->left_), VisitExpr(op->right_)); }

  Entry VisitExpr_(const PowPtr& /*op*/) override { return Everything(); }

  // --- Comparisons (boolean result — no useful modular info) ---

  Entry VisitExpr_(const EqPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const NePtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const LtPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const LePtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const GtPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const GePtr& /*op*/) override { return Everything(); }

  // --- Logical ---

  Entry VisitExpr_(const AndPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const OrPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const XorPtr& /*op*/) override { return Everything(); }

  // --- Bitwise ---

  Entry VisitExpr_(const BitAndPtr& op) override {
    // x & (2^k - 1) is equivalent to x % 2^k
    auto b = VisitExpr(op->right_);
    if (b.is_const() && b.base > 0) {
      int64_t mask = b.base;
      // Check if mask + 1 is a power of 2
      if ((mask & (mask + 1)) == 0) {
        return FloorModByConst(op->left_, mask + 1);
      }
    }
    return Everything();
  }

  Entry VisitExpr_(const BitOrPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const BitXorPtr& /*op*/) override { return Everything(); }

  Entry VisitExpr_(const BitShiftLeftPtr& op) override {
    // x << k is equivalent to x * 2^k
    auto b = VisitExpr(op->right_);
    if (b.is_const() && b.base >= 0 && b.base < 63) {
      auto a = VisitExpr(op->left_);
      int64_t multiplier = static_cast<int64_t>(1) << b.base;
      if (MulWouldOverflow(a.coeff, multiplier) || MulWouldOverflow(a.base, multiplier)) {
        return Everything();
      }
      return {ZeroAwareGCD(0, a.coeff * multiplier), a.base * multiplier};
    }
    return Everything();
  }

  Entry VisitExpr_(const BitShiftRightPtr& op) override {
    // x >> k is equivalent to x // 2^k
    auto b = VisitExpr(op->right_);
    if (b.is_const() && b.base >= 0 && b.base < 63) {
      return FloorDivByConst(op->left_, static_cast<int64_t>(1) << b.base);
    }
    return Everything();
  }

  // --- Unary ---

  Entry VisitExpr_(const NegPtr& op) override {
    auto a = VisitExpr(op->operand_);
    return {a.coeff, -a.base};
  }

  Entry VisitExpr_(const AbsPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const NotPtr& /*op*/) override { return Everything(); }
  Entry VisitExpr_(const BitNotPtr& /*op*/) override { return Everything(); }

  Entry VisitExpr_(const CastPtr& op) override { return VisitExpr(op->operand_); }

 private:
  Analyzer* parent_;
  std::unordered_map<const Expr*, Entry> var_map_;

  /// Floor division by a known constant.
  Entry FloorDivByConst(const ExprPtr& lhs, int64_t val) {
    Entry a = VisitExpr(lhs);
    INTERNAL_CHECK(val != 0) << "Internal error: division by zero in FloorDivByConst";
    if (a.coeff % val == 0) {
      if (a.base == 0) {
        return {std::abs(a.coeff / val), 0};
      }
      if (a.base > 0 && val > 0) {
        return {a.coeff / val, a.base / val};
      }
    }
    return Everything();
  }

  /// Floor mod by a known constant.
  Entry FloorModByConst(const ExprPtr& lhs, int64_t val) {
    Entry a = VisitExpr(lhs);
    INTERNAL_CHECK(val != 0) << "Internal error: modulo by zero in FloorModByConst";
    int64_t coeff = ZeroAwareGCD(a.coeff, val);
    if (a.base >= 0) {
      return {coeff, a.base % coeff};
    }
    return Everything();
  }

  /// Update a variable by intersecting with a new entry. Returns recovery function.
  std::function<void()> UpdateByIntersect(const Expr* var_ptr, const Entry& entry) {
    auto it = var_map_.find(var_ptr);
    Entry old = (it != var_map_.end()) ? it->second : Everything();
    var_map_[var_ptr] = Intersect(old, entry);
    return [this, old, var_ptr]() {
      if (old.coeff == 1 && old.base == 0) {
        var_map_.erase(var_ptr);
      } else {
        var_map_[var_ptr] = old;
      }
    };
  }
};

// ============================================================================
// EnterConstraint — parse floormod(var, c) == b patterns
// ============================================================================

std::function<void()> ModularSetAnalyzer::Impl::EnterConstraint(const ExprPtr& constraint) {
  // Pattern: floormod(var, c) == b
  if (auto eq = As<Eq>(constraint)) {
    if (auto fmod = As<FloorMod>(eq->left_)) {
      auto var = As<Var>(fmod->left_);
      auto coeff_ci = As<ConstInt>(fmod->right_);
      auto base_ci = As<ConstInt>(eq->right_);
      if (var && coeff_ci && base_ci) {
        return UpdateByIntersect(var.get(), {coeff_ci->value_, base_ci->value_});
      }
    }
    // Pattern: var == const
    auto var_l = As<Var>(eq->left_);
    auto ci_r = As<ConstInt>(eq->right_);
    if (var_l && ci_r) {
      return UpdateByIntersect(var_l.get(), {0, ci_r->value_});
    }
    auto ci_l = As<ConstInt>(eq->left_);
    auto var_r = As<Var>(eq->right_);
    if (ci_l && var_r) {
      return UpdateByIntersect(var_r.get(), {0, ci_l->value_});
    }
  }
  return nullptr;
}

// ============================================================================
// ModularSetAnalyzer — public interface delegation to Impl
// ============================================================================

ModularSetAnalyzer::ModularSetAnalyzer() : impl_(std::make_unique<Impl>(nullptr)) {}

ModularSetAnalyzer::ModularSetAnalyzer(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

ModularSetAnalyzer::~ModularSetAnalyzer() = default;

ModularSetAnalyzer::ModularSetAnalyzer(ModularSetAnalyzer&&) noexcept = default;
ModularSetAnalyzer& ModularSetAnalyzer::operator=(ModularSetAnalyzer&&) noexcept = default;

ModularSet ModularSetAnalyzer::operator()(const ExprPtr& expr) const {
  auto entry = impl_->VisitExpr(expr);
  return {entry.coeff, entry.base};
}

void ModularSetAnalyzer::Update(const VarPtr& var, const ModularSet& info) {
  CHECK(info.coeff >= 0) << "ModularSet coeff must be non-negative, got " << info.coeff;
  impl_->Update(var, {info.coeff, info.base});
}

std::function<void()> ModularSetAnalyzer::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
