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

#ifndef PYPTO_IR_ARITH_ANALYZER_H_
#define PYPTO_IR_ARITH_ANALYZER_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace arith {

/// Inclusive integer bounds [min_value, max_value] for an expression.
struct ConstIntBound {
  int64_t min_value;
  int64_t max_value;

  /// Sentinel for positive infinity. Uses INT64_MAX so that -kPosInf avoids overflow.
  static constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
  /// Sentinel for negative infinity (= -kPosInf, NOT INT64_MIN).
  static constexpr int64_t kNegInf = -kPosInf;

  [[nodiscard]] bool is_const() const { return min_value == max_value; }
  [[nodiscard]] bool is_const(int64_t v) const { return min_value == v && max_value == v; }
  [[nodiscard]] bool is_non_negative() const { return min_value >= 0; }
  [[nodiscard]] bool is_positive() const { return min_value > 0; }
  [[nodiscard]] bool is_everything() const { return min_value == kNegInf && max_value == kPosInf; }
};

// Forward declarations.
class Analyzer;
class ConstraintContext;

/// Propagates constant integer bounds through expression trees.
///
/// Given variable ranges (e.g., x in [0, 7]), computes [min, max] for
/// any expression involving those variables.
class ConstIntBoundAnalyzer {
 public:
  /// Construct a standalone analyzer (no parent Analyzer).
  ConstIntBoundAnalyzer();

  ~ConstIntBoundAnalyzer();

  ConstIntBoundAnalyzer(const ConstIntBoundAnalyzer&) = delete;
  ConstIntBoundAnalyzer& operator=(const ConstIntBoundAnalyzer&) = delete;
  ConstIntBoundAnalyzer(ConstIntBoundAnalyzer&&) noexcept;
  ConstIntBoundAnalyzer& operator=(ConstIntBoundAnalyzer&&) noexcept;

  /// Compute bounds for an expression.
  ConstIntBound operator()(const ExprPtr& expr) const;

  /// Bind a variable to the half-open range [min_val, max_val_exclusive).
  void Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive);

  /// Update a variable's bound (inclusive on both ends).
  void Update(const VarPtr& var, const ConstIntBound& bound);

  /// Enter a constraint scope (e.g., inside an if-branch where expr is known true).
  /// Returns a recovery function that restores original bounds.
  std::function<void()> EnterConstraint(const ExprPtr& constraint);

 private:
  friend class Analyzer;
  explicit ConstIntBoundAnalyzer(Analyzer* parent);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Modular arithmetic properties: value = coeff * k + base for some integer k.
///
/// When coeff == 0, the value is exactly `base` (known constant).
/// When coeff == 1 && base == 0, no useful modular information is known.
struct ModularSet {
  int64_t coeff;  ///< Always >= 0. 0 means exact value known.
  int64_t base;   ///< Normalized: 0 <= base < coeff (when coeff > 0).

  [[nodiscard]] bool is_exact() const { return coeff == 0; }
  [[nodiscard]] bool is_everything() const { return coeff == 1 && base == 0; }
};

/// Tracks modular arithmetic properties through expression trees.
///
/// Given an expression, computes {coeff, base} such that the expression
/// is always of the form coeff * k + base. Enables simplifications like
/// (2*x) % 2 → 0.
class ModularSetAnalyzer {
 public:
  /// Construct a standalone analyzer (no parent Analyzer).
  ModularSetAnalyzer();

  ~ModularSetAnalyzer();

  ModularSetAnalyzer(const ModularSetAnalyzer&) = delete;
  ModularSetAnalyzer& operator=(const ModularSetAnalyzer&) = delete;
  ModularSetAnalyzer(ModularSetAnalyzer&&) noexcept;
  ModularSetAnalyzer& operator=(ModularSetAnalyzer&&) noexcept;

  /// Compute modular set for an expression.
  ModularSet operator()(const ExprPtr& expr) const;

  /// Update a variable's modular set information.
  void Update(const VarPtr& var, const ModularSet& info);

  /// Enter a constraint scope. Returns a recovery function.
  std::function<void()> EnterConstraint(const ExprPtr& constraint);

 private:
  friend class Analyzer;
  explicit ModularSetAnalyzer(Analyzer* parent);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Pattern-matching rewrite engine for algebraic simplification of integer/index expressions.
///
/// Designed primarily for index arithmetic and integer shape calculations.
/// Applies ~180 algebraic identities (e.g., x + 0 -> x, x - x -> 0) using
/// pattern matching. Float-typed expressions are returned unchanged since
/// integer-style identities are not semantics-preserving for floating-point
/// (rounding errors, NaN). Supports variable substitution and constraint scoping.
class RewriteSimplifier {
 public:
  /// Construct a standalone simplifier (no parent Analyzer).
  RewriteSimplifier();

  ~RewriteSimplifier();

  RewriteSimplifier(const RewriteSimplifier&) = delete;
  RewriteSimplifier& operator=(const RewriteSimplifier&) = delete;
  RewriteSimplifier(RewriteSimplifier&&) noexcept;
  RewriteSimplifier& operator=(RewriteSimplifier&&) noexcept;

  /// Simplify an expression by applying rewrite rules.
  ExprPtr operator()(const ExprPtr& expr) const;

  /// Register a variable substitution: replace var with new_expr during simplification.
  /// Pass nullptr to remove a previous substitution.
  void Update(const VarPtr& var, const ExprPtr& new_expr);

  /// Enter a constraint scope (e.g., inside an if-branch where constraint is known true).
  /// Returns a recovery function that restores original state.
  std::function<void()> EnterConstraint(const ExprPtr& constraint);

 private:
  friend class Analyzer;
  explicit RewriteSimplifier(Analyzer* parent);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Coordinates all sub-analyzers for arithmetic expression analysis and simplification.
///
/// Provides a unified interface for binding variable ranges, simplifying expressions,
/// and proving arithmetic properties. Each sub-analyzer can also be used standalone,
/// but the coordinator enables cross-analyzer queries (e.g., rewrite rules that
/// use bound information to determine applicability).
class Analyzer : public std::enable_shared_from_this<Analyzer> {
 public:
  Analyzer();
  ~Analyzer();

  Analyzer(const Analyzer&) = delete;
  Analyzer& operator=(const Analyzer&) = delete;

  /// Sub-analyzers — public for direct access when needed.
  ConstIntBoundAnalyzer const_int_bound;
  ModularSetAnalyzer modular_set;
  RewriteSimplifier rewrite_simplify;

  /// Bind a variable to an expression: propagates information to all sub-analyzers.
  /// \note allow_override is reserved for future use and currently has no effect.
  void Bind(const VarPtr& var, const ExprPtr& expr, bool allow_override = false);

  /// Bind a variable to the half-open range [min_val, max_val_exclusive).
  /// \note allow_override is reserved for future use and currently has no effect.
  void Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive, bool allow_override = false);

  /// Simplify an expression by iterative rewrite simplification.
  /// \param steps Number of simplification rounds (default 2).
  ExprPtr Simplify(const ExprPtr& expr, int steps = 2);

  /// Prove that expr >= lower_bound for all possible variable values.
  bool CanProveGreaterEqual(const ExprPtr& expr, int64_t lower_bound);

  /// Prove that expr < upper_bound for all possible variable values.
  bool CanProveLess(const ExprPtr& expr, int64_t upper_bound);

  /// Prove that lhs and rhs are always equal.
  bool CanProveEqual(const ExprPtr& lhs, const ExprPtr& rhs);

  /// Prove that a boolean condition is always true.
  bool CanProve(const ExprPtr& cond);

  /// Create a constraint scope (RAII guard).
  /// Within the returned scope, the constraint is assumed true.
  /// \note Analyzer must be managed by shared_ptr (use std::make_shared<Analyzer>()).
  ConstraintContext GetConstraintContext(const ExprPtr& constraint);
};

using AnalyzerPtr = std::shared_ptr<Analyzer>;

/// RAII guard that enters a constraint scope on all sub-analyzers.
///
/// Within the scope, the constraint expression is assumed to be true,
/// which tightens variable bounds and enables additional simplifications.
/// On destruction, all sub-analyzers are restored to their pre-constraint state.
///
/// Usage:
///   {
///     auto ctx = analyzer->GetConstraintContext(x >= 0);
///     // Within this scope, analyzer knows x >= 0
///     auto simplified = analyzer->Simplify(expr);
///   }  // Bounds restored here
class ConstraintContext {
 public:
  ConstraintContext(AnalyzerPtr analyzer, const ExprPtr& constraint);
  ~ConstraintContext();

  ConstraintContext(ConstraintContext&& other) noexcept;
  ConstraintContext& operator=(ConstraintContext&&) = delete;
  ConstraintContext(const ConstraintContext&) = delete;
  ConstraintContext& operator=(const ConstraintContext&) = delete;

  /// Explicitly exit the constraint scope (idempotent).
  /// Called by the destructor, but can also be called earlier (e.g., from Python __exit__).
  void ExitScope();

 private:
  AnalyzerPtr analyzer_;
  bool exited_{false};
  std::vector<std::function<void()>> recovery_functions_;
};

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_ANALYZER_H_
