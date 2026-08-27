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

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// IntSet static factory methods
// ============================================================================

IntSet IntSet::Everything() { return {nullptr, nullptr}; }

IntSet IntSet::SinglePoint(const ExprPtr& val) { return {val, val}; }

IntSet IntSet::Interval(const ExprPtr& min, const ExprPtr& max) { return {min, max}; }

bool IntSet::is_single_point() const {
  if (!min_value || !max_value) return false;
  return min_value.get() == max_value.get();
}

// ============================================================================
// Infinity-aware symbolic arithmetic helpers
// ============================================================================
//
// Convention: nullptr represents unbounded (negative infinity for min,
// positive infinity for max). The Sym* helpers propagate nullptr correctly
// according to the mathematical rules of interval arithmetic.

namespace {

/// Simplify an ExprPtr if an Analyzer is available.
ExprPtr MaybeSimplify(Analyzer* parent, const ExprPtr& expr) {
  if (parent) return parent->Simplify(expr);
  return expr;
}

/// Check if expr is provably non-negative using parent Analyzer.
bool IsNonNeg(const ExprPtr& e, Analyzer* parent) {
  if (!e) return false;
  if (auto ci = As<ConstInt>(e)) return ci->value_ >= 0;
  if (parent) return parent->CanProveGreaterEqual(e, 0);
  return false;
}

/// Check if expr is provably negative using parent Analyzer.
bool IsNeg(const ExprPtr& e, Analyzer* parent) {
  if (!e) return false;
  if (auto ci = As<ConstInt>(e)) return ci->value_ < 0;
  if (parent) return parent->CanProveLess(e, 0);
  return false;
}

/// Symbolic add: nullptr (unbounded) propagates.
ExprPtr SymAdd(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;
  return MaybeSimplify(parent, MakeAdd(a, b));
}

/// Symbolic sub: nullptr (unbounded) propagates.
ExprPtr SymSub(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;
  return MaybeSimplify(parent, MakeSub(a, b));
}

/// Symbolic negation: nullptr stays nullptr.
ExprPtr SymNeg(const ExprPtr& a, Analyzer* parent) {
  if (!a) return nullptr;
  return MaybeSimplify(parent, MakeNeg(a));
}

/// Symbolic multiply. Returns nullptr if either is nullptr.
ExprPtr SymMul(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;
  return MaybeSimplify(parent, MakeMul(a, b));
}

/// Symbolic min of two ExprPtrs.
/// nullptr for a lower bound means -inf, so min(-inf, x) = -inf = nullptr.
/// nullptr for an upper bound means +inf, so min(+inf, x) = x.
/// Since we use the same ExprPtr for both roles, callers must handle context.
/// Here we treat nullptr as "more negative" (i.e., -inf wins in min).
ExprPtr SymMinLower(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;  // -inf wins
  if (parent) {
    if (parent->CanProve(MakeLe(a, b))) return a;
    if (parent->CanProve(MakeGe(a, b))) return b;
  }
  return MaybeSimplify(parent, MakeMin(a, b));
}

/// Symbolic min for upper bounds: nullptr means +inf, so min(+inf, x) = x.
ExprPtr SymMinUpper(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a) return b;  // +inf, take the other
  if (!b) return a;
  if (parent) {
    if (parent->CanProve(MakeLe(a, b))) return a;
    if (parent->CanProve(MakeGe(a, b))) return b;
  }
  return MaybeSimplify(parent, MakeMin(a, b));
}

/// Symbolic max for lower bounds: nullptr means -inf, so max(-inf, x) = x.
ExprPtr SymMaxLower(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a) return b;  // -inf, take the other
  if (!b) return a;
  if (parent) {
    if (parent->CanProve(MakeGe(a, b))) return a;
    if (parent->CanProve(MakeLe(a, b))) return b;
  }
  return MaybeSimplify(parent, MakeMax(a, b));
}

/// Symbolic max for upper bounds: nullptr means +inf, so max(+inf, x) = +inf.
ExprPtr SymMaxUpper(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;  // +inf wins
  if (parent) {
    if (parent->CanProve(MakeGe(a, b))) return a;
    if (parent->CanProve(MakeLe(a, b))) return b;
  }
  return MaybeSimplify(parent, MakeMax(a, b));
}

/// Symbolic floor division. Returns nullptr if either is nullptr.
ExprPtr SymFloorDiv(const ExprPtr& a, const ExprPtr& b, Analyzer* parent) {
  if (!a || !b) return nullptr;
  return MaybeSimplify(parent, MakeFloorDiv(a, b));
}

// ============================================================================
// Interval combine helpers (one per binary operation)
// ============================================================================

using Interval = IntSet;

/// [a.min + b.min, a.max + b.max]
Interval CombineAdd(const Interval& a, const Interval& b, Analyzer* p) {
  return {SymAdd(a.min_value, b.min_value, p), SymAdd(a.max_value, b.max_value, p)};
}

/// [a.min - b.max, a.max - b.min]
Interval CombineSub(const Interval& a, const Interval& b, Analyzer* p) {
  return {SymSub(a.min_value, b.max_value, p), SymSub(a.max_value, b.min_value, p)};
}

/// Sign-dependent multiplication.
Interval CombineMul(const Interval& a, const Interval& b, Analyzer* p) {
  // If both intervals are single-point constants, just multiply.
  if (a.is_single_point() && b.is_single_point()) {
    ExprPtr val = SymMul(a.min_value, b.min_value, p);
    return {val, val};
  }

  // Normalize: ensure singleton factor is on the RHS for symmetric handling.
  if (a.is_single_point() && !b.is_single_point()) {
    return CombineMul(b, a, p);
  }

  // When b is a single-point non-negative constant: [a.min * b, a.max * b]
  if (b.is_single_point() && IsNonNeg(b.min_value, p)) {
    return {SymMul(a.min_value, b.min_value, p), SymMul(a.max_value, b.min_value, p)};
  }
  // When b is a single-point negative constant: [a.max * b, a.min * b] (flip)
  if (b.is_single_point() && IsNeg(b.min_value, p)) {
    return {SymMul(a.max_value, b.min_value, p), SymMul(a.min_value, b.min_value, p)};
  }

  // When both intervals are non-negative: straightforward
  if (IsNonNeg(a.min_value, p) && IsNonNeg(b.min_value, p)) {
    return {SymMul(a.min_value, b.min_value, p), SymMul(a.max_value, b.max_value, p)};
  }

  // Conservative fallback
  return IntSet::Everything();
}

/// FloorDiv with single-point positive divisor.
/// Floor division by a positive constant is monotonic, so [a.min // d, a.max // d]
/// regardless of the sign of the dividend.
Interval CombineFloorDiv(const Interval& a, const Interval& b, Analyzer* p) {
  if (!b.is_single_point()) return IntSet::Everything();
  auto ci = As<ConstInt>(b.min_value);
  if (!ci || ci->value_ <= 0) return IntSet::Everything();

  return {SymFloorDiv(a.min_value, b.min_value, p), SymFloorDiv(a.max_value, b.min_value, p)};
}

/// FloorMod with single-point positive divisor: conservatively returns [0, d - 1].
Interval CombineFloorMod(const Interval& /*a*/, const Interval& b, Analyzer* /*p*/) {
  if (!b.is_single_point()) return IntSet::Everything();
  auto ci = As<ConstInt>(b.min_value);
  if (!ci || ci->value_ <= 0) return IntSet::Everything();

  // Conservative: [0, d - 1]. Could be tightened when the dividend range
  // fits within one modular period (a.max - a.min < d).
  ExprPtr zero = MakeConstInt(0, DataType::INT64);
  ExprPtr denom_minus_one = MakeConstInt(ci->value_ - 1, DataType::INT64);
  return {zero, denom_minus_one};
}

/// min(a, b) = [min(a.min, b.min), min(a.max, b.max)]
Interval CombineMin(const Interval& a, const Interval& b, Analyzer* p) {
  return {SymMinLower(a.min_value, b.min_value, p), SymMinUpper(a.max_value, b.max_value, p)};
}

/// max(a, b) = [max(a.min, b.min), max(a.max, b.max)]
Interval CombineMax(const Interval& a, const Interval& b, Analyzer* p) {
  return {SymMaxLower(a.min_value, b.min_value, p), SymMaxUpper(a.max_value, b.max_value, p)};
}

/// Boolean result interval: [0, 1].
Interval BoolInterval() { return {MakeConstInt(0, DataType::INT64), MakeConstInt(1, DataType::INT64)}; }

}  // namespace

// ============================================================================
// Implementation class — extends ExprFunctor<IntSet>
// ============================================================================

class IntSetAnalyzer::Impl : public ExprFunctor<IntSet> {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  void Update(const VarPtr& var, const IntSet& set) { var_map_[var.get()] = set; }

  void Bind(const VarPtr& var, const ExprPtr& min_val, const ExprPtr& max_val_exclusive) {
    // Convert half-open [min, max_exclusive) to inclusive [min, max_exclusive - 1]
    ExprPtr one = MakeConstInt(1, DataType::INT64);
    ExprPtr sub_expr = MakeSub(max_val_exclusive, one);
    // Try constant folding first (works even without parent Analyzer),
    // then fall back to simplification if parent is available.
    ExprPtr max_inclusive = TryConstFold(sub_expr);
    if (!max_inclusive) max_inclusive = MaybeSimplify(parent_, sub_expr);
    var_map_[var.get()] = {min_val, max_inclusive};
  }

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

 protected:
  // --- Leaf nodes ---

  IntSet VisitExpr_(const ConstIntPtr& op) override {
    return IntSet::SinglePoint(std::static_pointer_cast<const Expr>(op));
  }

  IntSet VisitExpr_(const ConstFloatPtr& /*op*/) override { return IntSet::Everything(); }

  IntSet VisitExpr_(const ConstBoolPtr& op) override {
    return IntSet::SinglePoint(std::static_pointer_cast<const Expr>(op));
  }

  IntSet VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    // Unknown var: single point [var, var] — the var IS its own symbolic bound.
    return IntSet::SinglePoint(std::static_pointer_cast<const Expr>(op));
  }

  IntSet VisitExpr_(const IterArgPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return IntSet::SinglePoint(std::static_pointer_cast<const Expr>(op));
  }

  IntSet VisitExpr_(const MemRefPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const WindowBufferPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const CallPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const SubmitPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const MakeTuplePtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const TupleGetItemExprPtr& /*op*/) override { return IntSet::Everything(); }

  // --- Binary arithmetic ---

  IntSet VisitExpr_(const AddPtr& op) override {
    return CombineAdd(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const SubPtr& op) override {
    return CombineSub(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const MulPtr& op) override {
    return CombineMul(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const FloorDivPtr& op) override {
    return CombineFloorDiv(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const FloorModPtr& op) override {
    return CombineFloorMod(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const FloatDivPtr& /*op*/) override { return IntSet::Everything(); }

  IntSet VisitExpr_(const MinPtr& op) override {
    return CombineMin(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const MaxPtr& op) override {
    return CombineMax(VisitExpr(op->left_), VisitExpr(op->right_), parent_);
  }

  IntSet VisitExpr_(const PowPtr& /*op*/) override { return IntSet::Everything(); }

  // --- Comparisons (boolean result) ---
  IntSet VisitExpr_(const EqPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const NePtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const LtPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const LePtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const GtPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const GePtr& /*op*/) override { return BoolInterval(); }

  // --- Logical ---
  IntSet VisitExpr_(const AndPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const OrPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const XorPtr& /*op*/) override { return BoolInterval(); }

  // --- Bitwise (conservative) ---
  IntSet VisitExpr_(const BitAndPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const BitOrPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const BitXorPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const BitShiftLeftPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const BitShiftRightPtr& /*op*/) override { return IntSet::Everything(); }

  // --- Unary ---

  IntSet VisitExpr_(const NegPtr& op) override {
    auto a = VisitExpr(op->operand_);
    // neg([min, max]) = [-max, -min]
    return {SymNeg(a.max_value, parent_), SymNeg(a.min_value, parent_)};
  }

  IntSet VisitExpr_(const AbsPtr& /*op*/) override { return IntSet::Everything(); }
  IntSet VisitExpr_(const NotPtr& /*op*/) override { return BoolInterval(); }
  IntSet VisitExpr_(const BitNotPtr& /*op*/) override { return IntSet::Everything(); }

  IntSet VisitExpr_(const CastPtr& op) override {
    // Only propagate bounds through widening casts (target >= source bits).
    // Narrowing casts can wrap/truncate, making interval propagation unsound.
    DataType src_dtype = GetScalarDtype(op->operand_);
    DataType dst_dtype = GetScalarDtype(std::static_pointer_cast<const Expr>(op));
    if (dst_dtype.GetBit() >= src_dtype.GetBit() && dst_dtype.IsInt() == src_dtype.IsInt()) {
      return VisitExpr(op->operand_);
    }
    return IntSet::Everything();
  }

 private:
  Analyzer* parent_;
  std::unordered_map<const Expr*, IntSet> var_map_;
};

// ============================================================================
// EnterConstraint — parse comparison constraints to tighten variable intervals
// ============================================================================

std::function<void()> IntSetAnalyzer::Impl::EnterConstraint(const ExprPtr& constraint) {
  std::vector<std::pair<const Expr*, IntSet>> recovery;

  // SymMaxLower / SymMinUpper merge a new bound with the existing one by building
  // Ge / Min / Max over the two, and those builders reject operands that are not
  // scalars of the same numeric category. So a bound is recorded only when it is a
  // numeric scalar matching the category of the variable it bounds; anything else
  // (non-scalar, bool, mismatched category) is dropped instead.
  auto NumericScalarType = [](const ExprPtr& expr) -> ScalarTypePtr {
    auto scalar_type = As<ScalarType>(expr->GetType());
    if (scalar_type && (scalar_type->dtype_.IsInt() || scalar_type->dtype_.IsFloat())) return scalar_type;
    return nullptr;
  };

  auto TryTighten = [&](const VarPtr& var, const IntSet& new_set) {
    if (new_set.is_everything()) return;  // No bound survived — nothing to record.
    auto var_type = NumericScalarType(var);
    if (!var_type) return;
    auto MatchesVarCategory = [&](const ExprPtr& bound) {
      auto bound_type = NumericScalarType(bound);
      return bound_type && bound_type->dtype_.IsInt() == var_type->dtype_.IsInt();
    };
    if (new_set.min_value && !MatchesVarCategory(new_set.min_value)) return;
    if (new_set.max_value && !MatchesVarCategory(new_set.max_value)) return;

    const Expr* var_ptr = var.get();
    auto it = var_map_.find(var_ptr);
    IntSet old = (it != var_map_.end()) ? it->second : IntSet::Everything();
    // Update() / Bind() are not routed through this guard, so an existing bound may
    // be of the other category — merging it here is what would build the mixed Min/Max.
    if (old.min_value && !MatchesVarCategory(old.min_value)) return;
    if (old.max_value && !MatchesVarCategory(old.max_value)) return;

    recovery.emplace_back(var_ptr, old);
    // Tighten: new_min = max(old.min, new.min), new_max = min(old.max, new.max)
    var_map_[var_ptr] = {SymMaxLower(old.min_value, new_set.min_value, parent_),
                         SymMinUpper(old.max_value, new_set.max_value, parent_)};
  };

  ExprPtr one = MakeConstInt(1, DataType::INT64);

  // A strict comparison is stored as the equivalent closed bound. Integers have a
  // predecessor, so `left > right` is *exactly* `left >= right + 1`. No other dtype
  // does — `f > 1.0` does not imply `f >= 2.0` — so there the strictness is simply
  // forgotten and the non-strict bound `left >= right` is recorded instead: weaker,
  // but sound, and it keeps float comparisons contributing bounds at all.
  auto StrictBound = [&](const ExprPtr& bound, bool increment) -> ExprPtr {
    auto bound_type = NumericScalarType(bound);
    if (!bound_type || !bound_type->dtype_.IsInt()) return bound;
    return MaybeSimplify(parent_, increment ? MakeAdd(bound, one) : MakeSub(bound, one));
  };

  std::function<void(const ExprPtr&)> TryParse = [&](const ExprPtr& expr) {
    // Ge: left >= right  =>  left.min = right
    if (auto ge = As<Ge>(expr)) {
      if (auto var = As<Var>(ge->left_)) {
        TryTighten(var, {ge->right_, nullptr});
      }
      if (auto var = As<Var>(ge->right_)) {
        TryTighten(var, {nullptr, ge->left_});
      }
      return;
    }
    // Gt: left > right  =>  left.min = right + 1
    if (auto gt = As<Gt>(expr)) {
      if (auto var = As<Var>(gt->left_)) {
        TryTighten(var, {StrictBound(gt->right_, /*increment=*/true), nullptr});
      }
      if (auto var = As<Var>(gt->right_)) {
        TryTighten(var, {nullptr, StrictBound(gt->left_, /*increment=*/false)});
      }
      return;
    }
    // Le: left <= right  =>  left.max = right
    if (auto le = As<Le>(expr)) {
      if (auto var = As<Var>(le->left_)) {
        TryTighten(var, {nullptr, le->right_});
      }
      if (auto var = As<Var>(le->right_)) {
        TryTighten(var, {le->left_, nullptr});
      }
      return;
    }
    // Lt: left < right  =>  left.max = right - 1
    if (auto lt = As<Lt>(expr)) {
      if (auto var = As<Var>(lt->left_)) {
        TryTighten(var, {nullptr, StrictBound(lt->right_, /*increment=*/false)});
      }
      if (auto var = As<Var>(lt->right_)) {
        TryTighten(var, {StrictBound(lt->left_, /*increment=*/true), nullptr});
      }
      return;
    }
    // Eq: left == right  =>  left = [right, right]
    if (auto eq = As<Eq>(expr)) {
      if (auto var = As<Var>(eq->left_)) {
        TryTighten(var, IntSet::SinglePoint(eq->right_));
      }
      if (auto var = As<Var>(eq->right_)) {
        TryTighten(var, IntSet::SinglePoint(eq->left_));
      }
      return;
    }
    // And: recurse into both sides
    if (auto and_op = As<And>(expr)) {
      TryParse(and_op->left_);
      TryParse(and_op->right_);
    }
  };

  TryParse(constraint);

  if (recovery.empty()) return nullptr;

  return [this, recovery = std::move(recovery)]() {
    for (auto it = recovery.rbegin(); it != recovery.rend(); ++it) {
      const auto& [ptr, set] = *it;
      if (set.is_everything()) {
        var_map_.erase(ptr);
      } else {
        var_map_[ptr] = set;
      }
    }
  };
}

// ============================================================================
// IntSetAnalyzer — public interface delegation to Impl
// ============================================================================

IntSetAnalyzer::IntSetAnalyzer() : impl_(std::make_unique<Impl>(nullptr)) {}

IntSetAnalyzer::IntSetAnalyzer(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

IntSetAnalyzer::~IntSetAnalyzer() = default;

IntSetAnalyzer::IntSetAnalyzer(IntSetAnalyzer&&) noexcept = default;

IntSetAnalyzer& IntSetAnalyzer::operator=(IntSetAnalyzer&&) noexcept = default;

IntSet IntSetAnalyzer::operator()(const ExprPtr& expr) const { return impl_->VisitExpr(expr); }

void IntSetAnalyzer::Update(const VarPtr& var, const IntSet& set) { impl_->Update(var, set); }

void IntSetAnalyzer::Bind(const VarPtr& var, const ExprPtr& min_val, const ExprPtr& max_val_exclusive) {
  impl_->Bind(var, min_val, max_val_exclusive);
}

std::function<void()> IntSetAnalyzer::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
