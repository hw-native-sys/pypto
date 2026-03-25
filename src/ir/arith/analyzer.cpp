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

#include "pypto/ir/arith/analyzer.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// Analyzer
// ============================================================================

Analyzer::Analyzer()
    : const_int_bound(this), modular_set(this), rewrite_simplify(this), transitive_cmp(this), int_set(this) {}

Analyzer::~Analyzer() = default;

void Analyzer::Bind(const VarPtr& var, const ExprPtr& expr, bool allow_override) {
  ExprPtr simplified = rewrite_simplify(expr);

  // Propagate to all sub-analyzers.
  const_int_bound.Update(var, const_int_bound(simplified));
  modular_set.Update(var, modular_set(simplified));
  rewrite_simplify.Update(var, simplified);
  transitive_cmp.Bind(var, simplified, allow_override);
  int_set.Update(var, IntSet::SinglePoint(simplified));
}

void Analyzer::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive, bool allow_override) {
  CHECK(max_val_exclusive > min_val) << "Bind requires max_val_exclusive > min_val, got [" << min_val << ", "
                                     << max_val_exclusive << ")";
  const_int_bound.Bind(var, min_val, max_val_exclusive);
  transitive_cmp.Bind(var, min_val, max_val_exclusive, allow_override);
  // Propagate concrete range to int_set as well.
  DataType dtype = GetScalarDtype(var);
  int_set.Bind(var, MakeConstInt(min_val, dtype), MakeConstInt(max_val_exclusive, dtype));
  // If the range is a single value, propagate exact value to all sub-analyzers.
  if (max_val_exclusive - min_val == 1) {
    ExprPtr bound_value = MakeConstInt(min_val, dtype);
    rewrite_simplify.Update(var, bound_value);
    modular_set.Update(var, modular_set(bound_value));
  }
}

void Analyzer::Unbind(const VarPtr& var) {
  const_int_bound.Unbind(var);
  modular_set.Unbind(var);
  rewrite_simplify.Update(var, nullptr);
  transitive_cmp.Unbind(var);
}

ExprPtr Analyzer::Simplify(const ExprPtr& expr, int steps) {
  CHECK(steps >= 0) << "Simplify requires non-negative steps, got " << steps;
  ExprPtr result = expr;
  for (int i = 0; i < steps; ++i) {
    result = rewrite_simplify(result);
  }
  return result;
}

bool Analyzer::CanProveGreaterEqual(const ExprPtr& expr, int64_t lower_bound) {
  auto bound = const_int_bound(Simplify(expr));
  return bound.min_value >= lower_bound;
}

bool Analyzer::CanProveLess(const ExprPtr& expr, int64_t upper_bound) {
  auto bound = const_int_bound(Simplify(expr));
  return bound.max_value < upper_bound;
}

bool Analyzer::CanProveEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  // First: pointer identity check.
  if (lhs.get() == rhs.get()) return true;
  // Simplify (lhs - rhs) and check if it's provably zero.
  ExprPtr diff = Simplify(MakeSub(lhs, rhs));
  auto bound = const_int_bound(diff);
  return bound.is_const(0);
}

/// Check if a CompareResult satisfies a required comparison kind.
static bool ResultImplies(CompareResult result, CompareResult required) {
  // Using the bitwise encoding: result must be a subset of required.
  return (result & required) == result && result != CompareResult::kUnknown &&
         result != CompareResult::kInconsistent;
}

bool Analyzer::CanProve(const ExprPtr& cond) {
  ExprPtr simplified = Simplify(cond);

  if (auto cb = As<ConstBool>(simplified)) return cb->value_;
  if (auto ci = As<ConstInt>(simplified)) return ci->value_ != 0;

  // Recursively handle logical And: both sides must be provable.
  if (auto op = As<And>(simplified)) {
    return CanProve(op->left_) && CanProve(op->right_);
  }

  // Symbolic fallback strategy: when const_int_bound and transitive_cmp are
  // inconclusive, use int_set to get symbolic bounds, then check those via
  // const_int_bound. Uses the original operand (not int_set result) on the
  // "other" side to preserve symbolic relationships. Guarded by
  // in_int_set_eval_ to prevent CanProve -> int_set -> SymMin -> CanProve.

  // RAII guard for in_int_set_eval_ flag to ensure exception safety.
  struct IntSetEvalGuard {
    bool& flag;
    explicit IntSetEvalGuard(bool& f) : flag(f) { flag = true; }
    ~IntSetEvalGuard() { flag = false; }
  };

  // Symbolic fallback helper: evaluates int_set on lhs/rhs, then checks if a
  // symbolic bound satisfies the comparison threshold via const_int_bound.
  // Constructs CanonicalSimplifier lazily (only when this fallback is actually reached).
  //   check_upper=true:  prove max(lhs - rhs) </<= threshold (Lt/Le)
  //   check_upper=false: prove min(lhs - rhs) >/>= threshold (Gt/Ge)
  auto TrySymbolicFallback = [this](const ExprPtr& lhs, const ExprPtr& rhs, bool check_upper,
                                    bool strict) -> bool {
    if (in_int_set_eval_) return false;
    IntSetEvalGuard guard(in_int_set_eval_);
    // Standalone CanonicalSimplifier (no parent) for algebraic cancellations like
    // (n-1)-n -> -1. Intentionally parent-less to avoid recursion back into Analyzer.
    CanonicalSimplifier canon_simplify;
    auto DeepSimplify = [&](const ExprPtr& e) -> ExprPtr { return canon_simplify(Simplify(e)); };
    auto lhs_set = int_set(lhs);
    auto rhs_set = int_set(rhs);
    if (check_upper) {
      if (lhs_set.max_value) {
        auto b = const_int_bound(DeepSimplify(MakeSub(lhs_set.max_value, rhs)));
        if (strict ? b.max_value < 0 : b.max_value <= 0) return true;
      }
      if (rhs_set.min_value) {
        auto b = const_int_bound(DeepSimplify(MakeSub(lhs, rhs_set.min_value)));
        if (strict ? b.max_value < 0 : b.max_value <= 0) return true;
      }
    } else {
      if (lhs_set.min_value) {
        auto b = const_int_bound(DeepSimplify(MakeSub(lhs_set.min_value, rhs)));
        if (strict ? b.min_value > 0 : b.min_value >= 0) return true;
      }
      if (rhs_set.max_value) {
        auto b = const_int_bound(DeepSimplify(MakeSub(lhs, rhs_set.max_value)));
        if (strict ? b.min_value > 0 : b.min_value >= 0) return true;
      }
    }
    return false;
  };

  // Decompose comparison expressions and check via bounds analysis.
  // Fallback chain: const_int_bound -> transitive_cmp -> int_set symbolic.
  if (auto op = As<Lt>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    if (const_int_bound(diff).max_value < 0) return true;
    if (ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kLT)) return true;
    return TrySymbolicFallback(op->left_, op->right_, /*check_upper=*/true, /*strict=*/true);
  }
  if (auto op = As<Le>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    if (const_int_bound(diff).max_value <= 0) return true;
    if (ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kLE)) return true;
    return TrySymbolicFallback(op->left_, op->right_, /*check_upper=*/true, /*strict=*/false);
  }
  if (auto op = As<Gt>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    if (const_int_bound(diff).min_value > 0) return true;
    if (ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kGT)) return true;
    return TrySymbolicFallback(op->left_, op->right_, /*check_upper=*/false, /*strict=*/true);
  }
  if (auto op = As<Ge>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    if (const_int_bound(diff).min_value >= 0) return true;
    if (ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kGE)) return true;
    return TrySymbolicFallback(op->left_, op->right_, /*check_upper=*/false, /*strict=*/false);
  }
  // For a == b: prove a - b == 0
  if (auto op = As<Eq>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    if (const_int_bound(diff).is_const(0)) return true;
    return ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kEQ);
  }
  // For a != b: prove a - b never zero
  if (auto op = As<Ne>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    auto bound = const_int_bound(diff);
    if (bound.min_value > 0 || bound.max_value < 0) return true;
    return ResultImplies(transitive_cmp.TryCompare(op->left_, op->right_), CompareResult::kNE);
  }

  return false;
}

ConstraintContext Analyzer::GetConstraintContext(const ExprPtr& constraint) {
  return ConstraintContext(shared_from_this(), constraint);
}

// ============================================================================
// ConstraintContext
// ============================================================================

ConstraintContext::ConstraintContext(AnalyzerPtr analyzer, const ExprPtr& constraint)
    : analyzer_(std::move(analyzer)) {
  // Normalize the constraint via rewrite_simplify before dispatching.
  // This decomposes Not(Lt(a,b)) → Ge(a,b), etc., so all sub-analyzers
  // receive comparison expressions they can directly interpret.
  ExprPtr normalized = analyzer_->rewrite_simplify(constraint);

  // Enter constraint on each sub-analyzer and collect recovery functions.
  if (auto fn = analyzer_->const_int_bound.EnterConstraint(normalized)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->modular_set.EnterConstraint(normalized)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->rewrite_simplify.EnterConstraint(normalized)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->transitive_cmp.EnterConstraint(normalized)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->int_set.EnterConstraint(normalized)) {  // Use normalized, not raw constraint
    recovery_functions_.push_back(std::move(fn));
  }
}

ConstraintContext::ConstraintContext(ConstraintContext&& other) noexcept
    : analyzer_(std::move(other.analyzer_)),
      exited_(other.exited_),
      recovery_functions_(std::move(other.recovery_functions_)) {
  other.exited_ = true;  // Moved-from object should not call recovery.
}

void ConstraintContext::ExitScope() {
  if (exited_) return;
  exited_ = true;
  // Restore in reverse order.
  for (auto it = recovery_functions_.rbegin(); it != recovery_functions_.rend(); ++it) {
    (*it)();
  }
}

ConstraintContext::~ConstraintContext() { ExitScope(); }

}  // namespace arith
}  // namespace ir
}  // namespace pypto
