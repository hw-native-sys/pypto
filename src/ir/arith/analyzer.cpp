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

Analyzer::Analyzer() : const_int_bound(this), modular_set(this), rewrite_simplify(this) {}

Analyzer::~Analyzer() = default;

void Analyzer::Bind(const VarPtr& var, const ExprPtr& expr, bool /*allow_override*/) {
  ExprPtr simplified = rewrite_simplify(expr);

  // Propagate to all sub-analyzers.
  const_int_bound.Update(var, const_int_bound(simplified));
  modular_set.Update(var, modular_set(simplified));
  rewrite_simplify.Update(var, simplified);
}

void Analyzer::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive, bool /*allow_override*/) {
  CHECK(max_val_exclusive > min_val) << "Bind requires max_val_exclusive > min_val, got [" << min_val << ", "
                                     << max_val_exclusive << ")";
  const_int_bound.Bind(var, min_val, max_val_exclusive);
  // If the range is a single value, propagate exact value to all sub-analyzers.
  if (max_val_exclusive - min_val == 1) {
    DataType dtype = GetScalarDtype(var);
    ExprPtr bound_value = MakeConstInt(min_val, dtype);
    rewrite_simplify.Update(var, bound_value);
    modular_set.Update(var, modular_set(bound_value));
  }
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

bool Analyzer::CanProve(const ExprPtr& cond) {
  ExprPtr simplified = Simplify(cond);

  if (auto cb = As<ConstBool>(simplified)) return cb->value_;
  if (auto ci = As<ConstInt>(simplified)) return ci->value_ != 0;

  // Recursively handle logical And: both sides must be provable.
  if (auto op = As<And>(simplified)) {
    return CanProve(op->left_) && CanProve(op->right_);
  }

  // Decompose comparison expressions and check via bounds analysis.
  // For a < b: prove max(a - b) < 0
  if (auto op = As<Lt>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    return const_int_bound(diff).max_value < 0;
  }
  // For a <= b: prove max(a - b) <= 0
  if (auto op = As<Le>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    return const_int_bound(diff).max_value <= 0;
  }
  // For a > b: prove min(a - b) > 0
  if (auto op = As<Gt>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    return const_int_bound(diff).min_value > 0;
  }
  // For a >= b: prove min(a - b) >= 0
  if (auto op = As<Ge>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    return const_int_bound(diff).min_value >= 0;
  }
  // For a == b: prove a - b == 0
  if (auto op = As<Eq>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    return const_int_bound(diff).is_const(0);
  }
  // For a != b: prove a - b never zero
  if (auto op = As<Ne>(simplified)) {
    ExprPtr diff = Simplify(MakeSub(op->left_, op->right_));
    auto bound = const_int_bound(diff);
    return bound.min_value > 0 || bound.max_value < 0;
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
  // Enter constraint on each sub-analyzer and collect recovery functions.
  if (auto fn = analyzer_->const_int_bound.EnterConstraint(constraint)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->modular_set.EnterConstraint(constraint)) {
    recovery_functions_.push_back(std::move(fn));
  }
  if (auto fn = analyzer_->rewrite_simplify.EnterConstraint(constraint)) {
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
