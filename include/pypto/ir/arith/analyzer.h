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

/// Forward declaration — completed in PR 6.
class Analyzer;

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

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_ANALYZER_H_
