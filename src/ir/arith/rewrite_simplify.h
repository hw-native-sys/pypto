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

#ifndef SRC_IR_ARITH_REWRITE_SIMPLIFY_H_
#define SRC_IR_ARITH_REWRITE_SIMPLIFY_H_

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"

namespace pypto {
namespace ir {
namespace arith {

class RewriteSimplifier::Impl : public ExprFunctor<ExprPtr> {
 public:
  explicit Impl(Analyzer* parent);

  ExprPtr VisitExpr(const ExprPtr& expr) override;

  void Update(const VarPtr& var, const ExprPtr& info);

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

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
  /// Re-invoke VisitExpr up to kMaxRecursiveDepth.
  ExprPtr RecursiveRewrite(const ExprPtr& expr);

  /// Try to compare expr with val using bounds analysis.
  /// Returns kUnknown in standalone mode (no parent Analyzer).
  CompareResult TryCompare(const ExprPtr& expr, int64_t val);

  bool CanProveGE(const ExprPtr& expr, int64_t val) {
    auto r = TryCompare(expr, val);
    return r == CompareResult::kGE || r == CompareResult::kGT || r == CompareResult::kEQ;
  }

  bool CanProveGT(const ExprPtr& expr, int64_t val) { return TryCompare(expr, val) == CompareResult::kGT; }

  bool CanProveEqual(const ExprPtr& expr, int64_t val) { return TryCompare(expr, val) == CompareResult::kEQ; }

  bool CanProveLess(const ExprPtr& expr, int64_t val) { return TryCompare(expr, val) == CompareResult::kLT; }

  bool CanProveLE(const ExprPtr& expr, int64_t val) {
    auto r = TryCompare(expr, val);
    return r == CompareResult::kLE || r == CompareResult::kLT || r == CompareResult::kEQ;
  }

  void RecordAttemptedRewrite() { ++num_attempted_rewrites_; }
  void RecordRewrite() { ++num_rewrites_; }

  /// Try to match a literal constraint and return simplified boolean.
  ExprPtr TryMatchLiteralConstraint(const ExprPtr& expr) const;

  static constexpr int kMaxRecursiveDepth = 5;

  Analyzer* parent_;
  int recursive_depth_{0};
  int64_t num_attempted_rewrites_{0};
  int64_t num_rewrites_{0};

  std::unordered_map<const Expr*, ExprPtr> var_map_;

  std::vector<ExprPtr> literal_constraints_;
};

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_ARITH_REWRITE_SIMPLIFY_H_
