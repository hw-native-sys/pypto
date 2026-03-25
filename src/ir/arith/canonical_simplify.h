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

#ifndef SRC_IR_ARITH_CANONICAL_SIMPLIFY_H_
#define SRC_IR_ARITH_CANONICAL_SIMPLIFY_H_

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// Internal representations: SplitExpr and SumExpr
// ============================================================================

/// Represents: ((index % upper_factor) / lower_factor) * scale
///
/// Special cases:
/// - upper_factor == kPosInf && lower_factor == 1: index * scale
/// - lower_factor == 1: (index % upper_factor) * scale
/// - scale == 0: term contributes nothing (can be pruned)
struct SplitExpr {
  ExprPtr index;
  int64_t lower_factor{1};
  int64_t upper_factor{ConstIntBound::kPosInf};
  int64_t scale{1};

  /// Check if this is a simple scaled index (no div/mod).
  [[nodiscard]] bool IsSimpleIndex() const {
    return lower_factor == 1 && upper_factor == ConstIntBound::kPosInf;
  }
};

/// Represents: base + sum(split_expr_i)
///
/// The canonical form for an integer expression. All integer expressions
/// are converted to this form during visitation, then converted back to
/// ExprPtr via Normalize().
struct SumExpr {
  int64_t base{0};
  std::vector<SplitExpr> args;
  DataType dtype;
};

// ============================================================================
// CanonicalSimplifier::Impl
// ============================================================================

class CanonicalSimplifier::Impl : public ExprFunctor<ExprPtr> {
 public:
  explicit Impl(Analyzer* parent);

  ExprPtr VisitExpr(const ExprPtr& expr) override;

  void Update(const VarPtr& var, const ExprPtr& info);

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

  /// Clear the per-call sum cache (called from operator() before each top-level visit).
  void ClearSumCache() { sum_cache_.clear(); }

 protected:
  // --- Leaf nodes ---
  ExprPtr VisitExpr_(const VarPtr& op) override;
  ExprPtr VisitExpr_(const IterArgPtr& op) override;
  ExprPtr VisitExpr_(const ConstIntPtr& op) override;
  ExprPtr VisitExpr_(const ConstFloatPtr& op) override;
  ExprPtr VisitExpr_(const ConstBoolPtr& op) override;
  ExprPtr VisitExpr_(const MemRefPtr& op) override;
  ExprPtr VisitExpr_(const CallPtr& op) override;
  ExprPtr VisitExpr_(const MakeTuplePtr& op) override;
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override;

  // --- Binary arithmetic ---
  ExprPtr VisitExpr_(const AddPtr& op) override;
  ExprPtr VisitExpr_(const SubPtr& op) override;
  ExprPtr VisitExpr_(const MulPtr& op) override;
  ExprPtr VisitExpr_(const FloorDivPtr& op) override;
  ExprPtr VisitExpr_(const FloorModPtr& op) override;
  ExprPtr VisitExpr_(const FloatDivPtr& op) override;
  ExprPtr VisitExpr_(const MinPtr& op) override;
  ExprPtr VisitExpr_(const MaxPtr& op) override;
  ExprPtr VisitExpr_(const PowPtr& op) override;

  // --- Comparisons ---
  ExprPtr VisitExpr_(const EqPtr& op) override;
  ExprPtr VisitExpr_(const NePtr& op) override;
  ExprPtr VisitExpr_(const LtPtr& op) override;
  ExprPtr VisitExpr_(const LePtr& op) override;
  ExprPtr VisitExpr_(const GtPtr& op) override;
  ExprPtr VisitExpr_(const GePtr& op) override;

  // --- Logical ---
  ExprPtr VisitExpr_(const AndPtr& op) override;
  ExprPtr VisitExpr_(const OrPtr& op) override;
  ExprPtr VisitExpr_(const XorPtr& op) override;

  // --- Bitwise ---
  ExprPtr VisitExpr_(const BitAndPtr& op) override;
  ExprPtr VisitExpr_(const BitOrPtr& op) override;
  ExprPtr VisitExpr_(const BitXorPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftRightPtr& op) override;

  // --- Unary ---
  ExprPtr VisitExpr_(const NegPtr& op) override;
  ExprPtr VisitExpr_(const AbsPtr& op) override;
  ExprPtr VisitExpr_(const NotPtr& op) override;
  ExprPtr VisitExpr_(const BitNotPtr& op) override;
  ExprPtr VisitExpr_(const CastPtr& op) override;

 private:
  // ---- Core canonical form operations ----

  /// Get the SumExpr for an already-visited expression.
  /// Uses the cache if available, otherwise creates a basic SumExpr.
  SumExpr GetOrCreateSumExpr(const ExprPtr& expr);

  /// Convert a SumExpr back to a normal ExprPtr, and cache it.
  ExprPtr NormalizeAndCache(const SumExpr& sum);

  /// Convert a SumExpr back to a normal ExprPtr.
  ExprPtr Normalize(const SumExpr& sum);

  /// Convert a single SplitExpr back to a normal ExprPtr.
  ExprPtr Normalize(const SplitExpr& split, DataType dtype);

  // ---- SumExpr manipulation ----

  /// Merge two SumExprs (addition). Combines like terms.
  SumExpr SumAdd(const SumExpr& lhs, const SumExpr& rhs);

  /// Negate a SumExpr (for subtraction: lhs - rhs = lhs + neg(rhs)).
  SumExpr SumNegate(const SumExpr& sum);

  /// Multiply all terms in a SumExpr by a constant.
  SumExpr SumMulConst(const SumExpr& sum, int64_t c);

  /// Try to perform FloorDiv on a SumExpr by a positive constant.
  bool TrySumFloorDiv(const SumExpr& sum, int64_t divisor, SumExpr* result);

  /// Try to perform FloorMod on a SumExpr by a positive constant.
  bool TrySumFloorMod(const SumExpr& sum, int64_t divisor, SumExpr* result);

  // ---- SplitExpr merging ----

  /// Try to merge two SplitExprs with same index and same factors (combine scales).
  static bool TryMergeSameFactors(const SplitExpr& a, const SplitExpr& b, SplitExpr* merged);

  /// Try to merge a div-mod complementary pair:
  /// SplitExpr{x, L, kPosInf, L*s} + SplitExpr{x, 1, L, s} → SplitExpr{x, 1, kPosInf, s}
  static bool TryMergeDivModPair(const SplitExpr& a, const SplitExpr& b, SplitExpr* merged);

  /// Try all pairwise merges in a SumExpr's args list.
  static void MergeTerms(std::vector<SplitExpr>& args);

  // ---- Bounds helpers ----

  CompareResult TryCompare(const ExprPtr& expr, int64_t val);

  bool CanProveGE(const ExprPtr& expr, int64_t val) {
    auto r = TryCompare(expr, val);
    return r == CompareResult::kGE || r == CompareResult::kGT || r == CompareResult::kEQ;
  }

  [[maybe_unused]] Analyzer* parent_;
  std::unordered_map<const Expr*, ExprPtr> var_map_;

  /// Cache mapping ExprPtr (by raw pointer) to its SumExpr representation.
  /// Populated by arithmetic visitors so parent expressions can decompose children.
  /// Cleared at the start of each top-level operator() call.
  std::unordered_map<const Expr*, SumExpr> sum_cache_;
};

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_ARITH_CANONICAL_SIMPLIFY_H_
