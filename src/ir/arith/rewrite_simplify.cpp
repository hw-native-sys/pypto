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

#include "src/ir/arith/rewrite_simplify.h"

#include <algorithm>  // NOLINT(misc-include-cleaner) required by cpplint for pattern-match max()
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/arith/pattern_match.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// Constraint extraction helpers (inlined from constraint_extract.h/cc)
// ============================================================================

/// Decompose an AND-chain into individual constraints.
/// For `a && b`, returns `[a && b, a, b]`.
static std::vector<ExprPtr> ExtractConstraints(const ExprPtr& expr) {
  std::vector<ExprPtr> result;
  result.push_back(expr);
  auto and_node = As<And>(expr);
  if (and_node) {
    auto left_parts = ExtractConstraints(and_node->left_);
    auto right_parts = ExtractConstraints(and_node->right_);
    result.insert(result.end(), left_parts.begin(), left_parts.end());
    result.insert(result.end(), right_parts.begin(), right_parts.end());
  }
  return result;
}

// ============================================================================
// Constructor and common methods
// ============================================================================

RewriteSimplifier::Impl::Impl(Analyzer* parent) : parent_(parent) {}

ExprPtr RewriteSimplifier::Impl::VisitExpr(const ExprPtr& expr) {
  // Skip algebraic rewrite rules for float-typed expressions.
  // Integer-style identities (cancellation, associativity) are not
  // semantics-preserving for floating-point (rounding, NaN).
  if (auto stype = std::dynamic_pointer_cast<const ScalarType>(expr->GetType());
      stype && stype->dtype_.IsFloat()) {
    return expr;
  }
  return ExprFunctor<ExprPtr>::VisitExpr(expr);
}

ExprPtr RewriteSimplifier::Impl::RecursiveRewrite(const ExprPtr& expr) {
  if (recursive_depth_ >= kMaxRecursiveDepth) return expr;
  ++recursive_depth_;
  ExprPtr result = VisitExpr(expr);
  --recursive_depth_;
  return result;
}

CompareResult RewriteSimplifier::Impl::TryCompare(const ExprPtr& x, int64_t val) {
  if (!parent_) return CompareResult::kUnknown;

  ConstIntBound bound = parent_->const_int_bound(x);
  if (bound.min_value == val && bound.max_value == val) return CompareResult::kEQ;
  if (bound.min_value > val) return CompareResult::kGT;
  if (bound.min_value == val) return CompareResult::kGE;
  if (bound.max_value < val) return CompareResult::kLT;
  if (bound.max_value == val) return CompareResult::kLE;
  return CompareResult::kUnknown;
}

void RewriteSimplifier::Impl::Update(const VarPtr& var, const ExprPtr& info) {
  if (info) {
    var_map_[var.get()] = info;
  } else {
    var_map_.erase(var.get());
  }
}

std::function<void()> RewriteSimplifier::Impl::EnterConstraint(const ExprPtr& constraint) {
  size_t old_size = literal_constraints_.size();

  // Simplify the constraint first
  ExprPtr simplified = VisitExpr(constraint);
  auto parts = ExtractConstraints(simplified);

  for (auto& part : parts) {
    literal_constraints_.push_back(part);
  }

  return [this, old_size]() { literal_constraints_.resize(old_size); };
}

ExprPtr RewriteSimplifier::Impl::TryMatchLiteralConstraint(const ExprPtr& expr) const {
  for (const auto& constraint : literal_constraints_) {
    if (constraint.get() == expr.get()) {
      return MakeConstBool(true);
    }
    // Check if the negation matches
    if (auto not_expr = As<Not>(expr)) {
      if (constraint.get() == not_expr->operand_.get()) {
        // !(known_true) => false
        return MakeConstBool(false);
      }
    }
    if (auto not_constraint = As<Not>(constraint)) {
      if (not_constraint->operand_.get() == expr.get()) {
        return MakeConstBool(false);
      }
    }
  }
  return nullptr;
}

// ============================================================================
// Leaf nodes
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const VarPtr& op) {
  auto it = var_map_.find(op.get());
  if (it != var_map_.end()) return it->second;
  return op;
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const IterArgPtr& op) {
  auto it = var_map_.find(op.get());
  if (it != var_map_.end()) return it->second;
  return op;
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const ConstIntPtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const ConstFloatPtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const ConstBoolPtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const MemRefPtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const CallPtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const MakeTuplePtr& op) { return op; }
ExprPtr RewriteSimplifier::Impl::VisitExpr_(const TupleGetItemExprPtr& op) { return op; }

// ============================================================================
// Helper: reconstruct binary expr only if children changed
// ============================================================================

template <typename NodeT>
static ExprPtr MutateBinary(const std::shared_ptr<const NodeT>& op, const ExprPtr& a, const ExprPtr& b,
                            ExprPtr (*make_fn)(const ExprPtr&, const ExprPtr&, const Span&)) {
  if (a.get() == op->left_.get() && b.get() == op->right_.get()) return op;
  return make_fn(a, b, op->span_);
}

template <typename NodeT>
static ExprPtr MutateUnary(const std::shared_ptr<const NodeT>& op, const ExprPtr& a,
                           ExprPtr (*make_fn)(const ExprPtr&, const Span&)) {
  if (a.get() == op->operand_.get()) return op;
  return make_fn(a, op->span_);
}

// ============================================================================
// Add rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const AddPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Add, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeAdd);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // Constant reassociation
  // (x + c1) + c2 => x + (c1 + c2)
  PYPTO_TRY_REWRITE((x + c1) + c2, x + (c1 + c2));
  // (c1 + x) + c2 => x + (c1 + c2)
  PYPTO_TRY_REWRITE((c1 + x) + c2, x + (c1 + c2));

  // Cancellation
  // (x - y) + y => x
  PYPTO_TRY_REWRITE((x - y) + y, x);
  // x + (y - x) => y
  PYPTO_TRY_REWRITE(x + (y - x), y);
  // (x - y) + (y - z) => x - z
  PYPTO_TRY_REWRITE((x - y) + (y - z), x - z);
  // (x - y) + (z - x) => z - y
  PYPTO_TRY_REWRITE((x - y) + (z - x), z - y);

  // Coefficient folding
  // x + x => x * 2
  PYPTO_TRY_REWRITE(x + x, x * 2);
  // x*y + x => (y + 1) * x
  PYPTO_TRY_REWRITE(x * y + x, (y + 1) * x);
  // x + x*y => (y + 1) * x
  PYPTO_TRY_REWRITE(x + x * y, (y + 1) * x);
  // x*y + x*z => (y + z) * x
  PYPTO_TRY_REWRITE(x * y + x * z, (y + z) * x);
  // y*x + x*z => (y + z) * x
  PYPTO_TRY_REWRITE(y * x + x * z, (y + z) * x);
  // x*y + z*x => (y + z) * x
  PYPTO_TRY_REWRITE(x * y + z * x, (y + z) * x);
  // y*x + z*x => (y + z) * x
  PYPTO_TRY_REWRITE(y * x + z * x, (y + z) * x);

  // FloorDiv/FloorMod reconstruction
  // floordiv(x, y) * y + floormod(x, y) => x
  PYPTO_TRY_REWRITE(floordiv(x, y) * y + floormod(x, y), x);
  // floormod(x, y) + floordiv(x, y) * y => x
  PYPTO_TRY_REWRITE(floormod(x, y) + floordiv(x, y) * y, x);
  // y * floordiv(x, y) + floormod(x, y) => x
  PYPTO_TRY_REWRITE(y * floordiv(x, y) + floormod(x, y), x);
  // floormod(x, y) + y * floordiv(x, y) => x
  PYPTO_TRY_REWRITE(floormod(x, y) + y * floordiv(x, y), x);

  // Min/Max interactions
  // min(x, y - z) + z => min(x + z, y)
  PYPTO_TRY_REWRITE(min(x, y - z) + z, min(x + z, y));
  // min(x - z, y) + z => min(x, y + z)
  PYPTO_TRY_REWRITE(min(x - z, y) + z, min(x, y + z));
  // z + min(x, y - z) => min(x + z, y)
  PYPTO_TRY_REWRITE(z + min(x, y - z), min(x + z, y));
  // max(x, y - z) + z => max(x + z, y)
  PYPTO_TRY_REWRITE(max(x, y - z) + z, max(x + z, y));
  // max(x - z, y) + z => max(x, y + z)
  PYPTO_TRY_REWRITE(max(x - z, y) + z, max(x, y + z));
  // z + max(x, y - z) => max(x + z, y)
  PYPTO_TRY_REWRITE(z + max(x, y - z), max(x + z, y));

  // max(x, y) + min(x, y) => x + y (and commutation variants)
  PYPTO_TRY_REWRITE(max(x, y) + min(x, y), x + y);
  PYPTO_TRY_REWRITE(min(x, y) + max(x, y), x + y);
  PYPTO_TRY_REWRITE(max(x, y) + min(y, x), x + y);
  PYPTO_TRY_REWRITE(min(x, y) + max(y, x), x + y);

  // Canonicalization: move constants to the right
  // c1 + x => x + c1
  PYPTO_TRY_REWRITE(c1 + x, x + c1);
  // x + (c1 - y) => (x - y) + c1
  PYPTO_TRY_REWRITE(x + (c1 - y), (x - y) + c1);
  // (c1 - y) + x => (x - y) + c1
  PYPTO_TRY_REWRITE((c1 - y) + x, (x - y) + c1);
  // (x + c1) + y => (x + y) + c1
  PYPTO_TRY_REWRITE((x + c1) + y, (x + y) + c1);

  return ret;
}

// ============================================================================
// Sub rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const SubPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Sub, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeSub);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // Self-subtraction
  // x - x => 0
  PYPTO_TRY_REWRITE(x - x, MakeConstInt(0, GetScalarDtype(ret)));

  // Cancellation
  // (x + y) - y => x
  PYPTO_TRY_REWRITE((x + y) - y, x);
  // (x + y) - x => y
  PYPTO_TRY_REWRITE((x + y) - x, y);
  // (y + x) - x => y
  PYPTO_TRY_REWRITE((y + x) - x, y);
  // x - (x + y) => neg(y)
  PYPTO_TRY_REWRITE(x - (x + y), neg(y));
  // x - (y + x) => neg(y)
  PYPTO_TRY_REWRITE(x - (y + x), neg(y));

  // Constant reassociation
  // (x + c1) - c2 => x + (c1 - c2)
  PYPTO_TRY_REWRITE((x + c1) - c2, x + (c1 - c2));
  // (c1 + x) - c2 => x + (c1 - c2)
  PYPTO_TRY_REWRITE((c1 + x) - c2, x + (c1 - c2));
  // c1 - (c2 - x) => x + (c1 - c2)
  PYPTO_TRY_REWRITE(c1 - (c2 - x), x + (c1 - c2));
  // c1 - (x + c2) => (c1 - c2) - x
  PYPTO_TRY_REWRITE(c1 - (x + c2), (c1 - c2) - x);
  // c1 - (c2 + x) => (c1 - c2) - x
  PYPTO_TRY_REWRITE(c1 - (c2 + x), (c1 - c2) - x);
  // (c1 - x) - (c2 - y) => (y - x) + (c1 - c2)
  PYPTO_TRY_REWRITE((c1 - x) - (c2 - y), (y - x) + (c1 - c2));

  // Cross-subtraction cancellation
  // (x - y) - (x - z) => z - y
  PYPTO_TRY_REWRITE((x - y) - (x - z), z - y);
  // (x + y) - (x + z) => y - z
  PYPTO_TRY_REWRITE((x + y) - (x + z), y - z);
  // (y + x) - (x + z) => y - z
  PYPTO_TRY_REWRITE((y + x) - (x + z), y - z);
  // (x + y) - (z + x) => y - z
  PYPTO_TRY_REWRITE((x + y) - (z + x), y - z);
  // (y + x) - (z + x) => y - z
  PYPTO_TRY_REWRITE((y + x) - (z + x), y - z);

  // Coefficient folding
  // x*y - x => (y - 1) * x
  PYPTO_TRY_REWRITE(x * y - x, (y - 1) * x);
  // x - x*y => (1 - y) * x
  PYPTO_TRY_REWRITE(x - x * y, (1 - y) * x);
  // x*y - x*z => (y - z) * x
  PYPTO_TRY_REWRITE(x * y - x * z, (y - z) * x);
  // y*x - x*z => (y - z) * x
  PYPTO_TRY_REWRITE(y * x - x * z, (y - z) * x);
  // x*y - z*x => (y - z) * x
  PYPTO_TRY_REWRITE(x * y - z * x, (y - z) * x);
  // y*x - z*x => (y - z) * x
  PYPTO_TRY_REWRITE(y * x - z * x, (y - z) * x);

  // FloorDiv/FloorMod extraction
  // x - floordiv(x, c1) * c1 => floormod(x, c1) when c1 != 0
  PYPTO_TRY_REWRITE_IF(x - floordiv(x, c1) * c1, floormod(x, c1), c1.Eval()->value_ != 0);
  // x - c1 * floordiv(x, c1) => floormod(x, c1) when c1 != 0
  PYPTO_TRY_REWRITE_IF(x - c1 * floordiv(x, c1), floormod(x, c1), c1.Eval()->value_ != 0);
  // floordiv(x, c1) * c1 - x => 0 - floormod(x, c1) when c1 != 0
  PYPTO_TRY_REWRITE_IF(floordiv(x, c1) * c1 - x, MakeConstInt(0, GetScalarDtype(ret)) - floormod(x, c1),
                       c1.Eval()->value_ != 0);
  // x - floordiv(x+y, c1) * c1 => floormod(x+y, c1) - y when c1 != 0
  PYPTO_TRY_REWRITE_IF(x - floordiv(x + y, c1) * c1, floormod(x + y, c1) - y, c1.Eval()->value_ != 0);
  // floordiv(x+y, c1) * c1 - x => y - floormod(x+y, c1) when c1 != 0
  PYPTO_TRY_REWRITE_IF(floordiv(x + y, c1) * c1 - x, y - floormod(x + y, c1), c1.Eval()->value_ != 0);
  // x - floordiv(x-y, c1) * c1 => floormod(x-y, c1) + y when c1 != 0
  PYPTO_TRY_REWRITE_IF(x - floordiv(x - y, c1) * c1, floormod(x - y, c1) + y, c1.Eval()->value_ != 0);
  // floordiv(x-y, c1) * c1 - x => 0 - floormod(x-y, c1) - y when c1 != 0
  PYPTO_TRY_REWRITE_IF(floordiv(x - y, c1) * c1 - x,
                       MakeConstInt(0, GetScalarDtype(ret)) - floormod(x - y, c1) - y,
                       c1.Eval()->value_ != 0);

  // Min/Max interactions
  // min(x, y) - x => min(0, y - x)
  PYPTO_TRY_REWRITE(min(x, y) - x, min(MakeConstInt(0, GetScalarDtype(ret)), y - x));
  // min(x, y) - y => min(x - y, 0)
  PYPTO_TRY_REWRITE(min(x, y) - y, min(x - y, MakeConstInt(0, GetScalarDtype(ret))));
  // max(x, y) - x => max(0, y - x)
  PYPTO_TRY_REWRITE(max(x, y) - x, max(MakeConstInt(0, GetScalarDtype(ret)), y - x));
  // max(x, y) - y => max(x - y, 0)
  PYPTO_TRY_REWRITE(max(x, y) - y, max(x - y, MakeConstInt(0, GetScalarDtype(ret))));
  // x - min(x, y) => max(0, x - y)
  PYPTO_TRY_REWRITE(x - min(x, y), max(MakeConstInt(0, GetScalarDtype(ret)), x - y));
  // x - max(x, y) => min(0, x - y)
  PYPTO_TRY_REWRITE(x - max(x, y), min(MakeConstInt(0, GetScalarDtype(ret)), x - y));

  // min(x+y, z) - x => min(y, z-x)
  PYPTO_TRY_REWRITE(min(x + y, z) - x, min(y, z - x));
  PYPTO_TRY_REWRITE(min(y + x, z) - x, min(y, z - x));
  // min(z, x+y) - x => min(z-x, y)
  PYPTO_TRY_REWRITE(min(z, x + y) - x, min(z - x, y));
  PYPTO_TRY_REWRITE(min(z, y + x) - x, min(z - x, y));
  // max(x+y, z) - x => max(y, z-x)
  PYPTO_TRY_REWRITE(max(x + y, z) - x, max(y, z - x));
  PYPTO_TRY_REWRITE(max(y + x, z) - x, max(y, z - x));
  // max(z, x+y) - x => max(z-x, y)
  PYPTO_TRY_REWRITE(max(z, x + y) - x, max(z - x, y));
  PYPTO_TRY_REWRITE(max(z, y + x) - x, max(z - x, y));

  // x - min(x+y, z) => max(0-y, x-z)
  PYPTO_TRY_REWRITE(x - min(x + y, z), max(MakeConstInt(0, GetScalarDtype(ret)) - y, x - z));
  PYPTO_TRY_REWRITE(x - min(y + x, z), max(MakeConstInt(0, GetScalarDtype(ret)) - y, x - z));
  // x - min(z, x+y) => max(x-z, 0-y)
  PYPTO_TRY_REWRITE(x - min(z, x + y), max(x - z, MakeConstInt(0, GetScalarDtype(ret)) - y));
  PYPTO_TRY_REWRITE(x - min(z, y + x), max(x - z, MakeConstInt(0, GetScalarDtype(ret)) - y));
  // x - max(x+y, z) => min(0-y, x-z)
  PYPTO_TRY_REWRITE(x - max(x + y, z), min(MakeConstInt(0, GetScalarDtype(ret)) - y, x - z));
  PYPTO_TRY_REWRITE(x - max(y + x, z), min(MakeConstInt(0, GetScalarDtype(ret)) - y, x - z));
  // x - max(z, x+y) => min(x-z, 0-y)
  PYPTO_TRY_REWRITE(x - max(z, x + y), min(x - z, MakeConstInt(0, GetScalarDtype(ret)) - y));
  PYPTO_TRY_REWRITE(x - max(z, y + x), min(x - z, MakeConstInt(0, GetScalarDtype(ret)) - y));

  // min(x, y) - min(y, x) => 0
  PYPTO_TRY_REWRITE(min(x, y) - min(y, x), MakeConstInt(0, GetScalarDtype(ret)));
  // max(x, y) - max(y, x) => 0
  PYPTO_TRY_REWRITE(max(x, y) - max(y, x), MakeConstInt(0, GetScalarDtype(ret)));

  // Canonicalization
  // x - (y + c1) => (x - y) + (0 - c1)
  PYPTO_TRY_REWRITE(x - (y + c1), (x - y) + (0 - c1));
  // (x + c1) - y => (x - y) + c1
  PYPTO_TRY_REWRITE((x + c1) - y, (x - y) + c1);
  // x - (y - z) => (x + z) - y
  PYPTO_TRY_REWRITE(x - (y - z), (x + z) - y);

  return ret;
}

// ============================================================================
// Mul rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const MulPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Mul, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeMul);

  PVar<ExprPtr> x, y;
  PVar<ConstIntPtr> c1, c2;

  // Associativity with constants
  // (x * c1) * c2 => x * (c1 * c2)
  PYPTO_TRY_REWRITE((x * c1) * c2, x * (c1 * c2));
  // (c1 * x) * c2 => x * (c1 * c2)
  PYPTO_TRY_REWRITE((c1 * x) * c2, x * (c1 * c2));

  // min(x,y) * max(x,y) => x * y
  PYPTO_TRY_REWRITE(min(x, y) * max(x, y), x * y);
  PYPTO_TRY_REWRITE(max(x, y) * min(x, y), x * y);

  // Canonicalize: const to right
  // c1 * x => x * c1
  PYPTO_TRY_REWRITE(c1 * x, x * c1);
  // x * (c1 * y) => (x * y) * c1
  PYPTO_TRY_REWRITE(x * (c1 * y), (x * y) * c1);

  // Distributive law (recursive)
  // (x + c1) * c2 => x * c2 + c1 * c2
  PYPTO_TRY_REWRITE((x + c1) * c2, x * c2 + c1 * c2);

  // Flip negative constant: (x - y) * c1 => (y - x) * (-c1) when c1 < 0
  PYPTO_TRY_REWRITE_IF((x - y) * c1, (y - x) * (0 - c1), c1.Eval()->value_ < 0);

  return ret;
}

// ============================================================================
// FloorDiv rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const FloorDivPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::FloorDiv, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeFloorDiv);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x * c1 / c2 => x * (c1/c2) when c1 % c2 == 0 and c2 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x * c1, c2), x * floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // c1 * x / c2 => x * (c1/c2) when c1 % c2 == 0 and c2 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(c1 * x, c2), x * floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // x * c1 / c2 => floordiv(x, c2/c1) when c2 % c1 == 0 and c1 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x * c1, c2), floordiv(x, floordiv(c2, c1)),
                       c1.Eval()->value_ > 0 && c2.Eval()->value_ % c1.Eval()->value_ == 0 &&
                           c2.Eval()->value_ / c1.Eval()->value_ > 0);

  // floordiv(floordiv(x, c1), c2) => floordiv(x, c1 * c2) when c1 > 0 and c2 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(floordiv(x, c1), c2), floordiv(x, c1 * c2),
                       c1.Eval()->value_ > 0 && c2.Eval()->value_ > 0);

  // floordiv(floordiv(x, c1) + c2, c3) => floordiv(x + c1*c2, c1*c3) when c1 > 0 and c3 > 0
  {
    PVar<ConstIntPtr> c3;
    PYPTO_TRY_REWRITE_IF(floordiv(floordiv(x, c1) + c2, c3), floordiv(x + c1 * c2, c1 * c3),
                         c1.Eval()->value_ > 0 && c3.Eval()->value_ > 0);
  }

  // floordiv(x, x) => 1  (requires x != 0 to preserve semantics)
  PYPTO_TRY_REWRITE_IF(floordiv(x, x), MakeConstInt(1, GetScalarDtype(ret)),
                       CanProveGE(x.Eval(), 1) || CanProveLE(x.Eval(), -1));
  // floordiv(x * c1, x) => c1  (requires x != 0)
  PYPTO_TRY_REWRITE_IF(floordiv(x * c1, x), c1, CanProveGE(x.Eval(), 1) || CanProveLE(x.Eval(), -1));
  // floordiv(c1 * x, x) => c1  (requires x != 0)
  PYPTO_TRY_REWRITE_IF(floordiv(c1 * x, x), c1, CanProveGE(x.Eval(), 1) || CanProveLE(x.Eval(), -1));

  // floordiv(x + c1, c2) => floordiv(x, c2) + c1/c2 when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floordiv(x + c1, c2), floordiv(x, c2) + floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(c1 + x, c2) => floordiv(x, c2) + c1/c2 when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floordiv(c1 + x, c2), floordiv(x, c2) + floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // floordiv(x * c1 + y, c2) => x * (c1/c2) + floordiv(y, c2)
  // when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floordiv(x * c1 + y, c2), x * floordiv(c1, c2) + floordiv(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(c1 * x + y, c2) => x * (c1/c2) + floordiv(y, c2)
  PYPTO_TRY_REWRITE_IF(floordiv(c1 * x + y, c2), x * floordiv(c1, c2) + floordiv(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(y + x * c1, c2) => floordiv(y, c2) + x * (c1/c2)
  PYPTO_TRY_REWRITE_IF(floordiv(y + x * c1, c2), floordiv(y, c2) + x * floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(y + c1 * x, c2) => floordiv(y, c2) + x * (c1/c2)
  PYPTO_TRY_REWRITE_IF(floordiv(y + c1 * x, c2), floordiv(y, c2) + x * floordiv(c1, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // Min/Max distribution
  // floordiv(min(x*c1, y), c2) => min(x*floordiv(c1,c2), floordiv(y,c2))
  PYPTO_TRY_REWRITE_IF(floordiv(min(x * c1, y), c2), min(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(max(x*c1, y), c2) => max(x*floordiv(c1,c2), floordiv(y,c2))
  PYPTO_TRY_REWRITE_IF(floordiv(max(x * c1, y), c2), max(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(min(y, x*c1), c2) => min(floordiv(y,c2), x*floordiv(c1,c2))
  PYPTO_TRY_REWRITE_IF(floordiv(min(y, x * c1), c2), min(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floordiv(max(y, x*c1), c2) => max(floordiv(y,c2), x*floordiv(c1,c2))
  PYPTO_TRY_REWRITE_IF(floordiv(max(y, x * c1), c2), max(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // floordiv(x - floormod(x, c1), c1) => floordiv(x, c1) when c1 != 0
  PYPTO_TRY_REWRITE_IF(floordiv(x - floormod(x, c1), c1), floordiv(x, c1), c1.Eval()->value_ != 0);

  // floordiv(x*y, y) => x when y != 0 (dormant in PR 3)
  PYPTO_TRY_REWRITE_IF(floordiv(x * y, y), x, CanProveGE(y.Eval(), 1) || CanProveLE(y.Eval(), -1));
  PYPTO_TRY_REWRITE_IF(floordiv(y * x, y), x, CanProveGE(y.Eval(), 1) || CanProveLE(y.Eval(), -1));

  // floordiv(x*z + y, z) => x + floordiv(y, z) when z >= 0 (dormant)
  PYPTO_TRY_REWRITE_IF(floordiv(x * z + y, z), x + floordiv(y, z), CanProveGE(z.Eval(), 0));
  PYPTO_TRY_REWRITE_IF(floordiv(z * x + y, z), x + floordiv(y, z), CanProveGE(z.Eval(), 0));
  PYPTO_TRY_REWRITE_IF(floordiv(y + x * z, z), floordiv(y, z) + x, CanProveGE(z.Eval(), 0));
  PYPTO_TRY_REWRITE_IF(floordiv(y + z * x, z), floordiv(y, z) + x, CanProveGE(z.Eval(), 0));

  // Bound-dependent: floordiv(x, c) => 0 when 0 <= x < c (dormant in PR 3)
  PYPTO_TRY_REWRITE_IF(
      floordiv(x, c1), MakeConstInt(0, GetScalarDtype(ret)),
      c1.Eval()->value_ > 0 && CanProveGE(x.Eval(), 0) && CanProveLess(x.Eval(), c1.Eval()->value_));

  return ret;
}

// ============================================================================
// FloorMod rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const FloorModPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::FloorMod, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeFloorMod);

  PVar<ExprPtr> x, y;
  PVar<ConstIntPtr> c1, c2;

  // floormod(x * c1, c2) => 0 when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(x * c1, c2), MakeConstInt(0, GetScalarDtype(ret)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floormod(c1 * x, c2) => 0 when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(c1 * x, c2), MakeConstInt(0, GetScalarDtype(ret)),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // floormod(x + c1, c2) => floormod(x, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(x + c1, c2), floormod(x, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floormod(c1 + x, c2) => floormod(x, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(c1 + x, c2), floormod(x, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // floormod(x * c1 + y, c2) => floormod(y, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(x * c1 + y, c2), floormod(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floormod(c1 * x + y, c2) => floormod(y, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(c1 * x + y, c2), floormod(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floormod(y + x * c1, c2) => floormod(y, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(y + x * c1, c2), floormod(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);
  // floormod(y + c1 * x, c2) => floormod(y, c2) when c2 > 0 and c1 % c2 == 0
  PYPTO_TRY_REWRITE_IF(floormod(y + c1 * x, c2), floormod(y, c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ == 0);

  // floormod(x * c1, c2) => floormod(x * floormod(c1, c2), c2)
  // Guard: c2 != 0 and c1 is actually reduced (avoid no-op rewrite)
  PYPTO_TRY_REWRITE_IF(
      floormod(x * c1, c2), floormod(x * floormod(c1, c2), c2),
      c2.Eval()->value_ != 0 && arith::floormod(c1.Eval()->value_, c2.Eval()->value_) != c1.Eval()->value_);

  // floormod(x * y, y) => 0  (requires y != 0 to preserve semantics)
  PYPTO_TRY_REWRITE_IF(floormod(x * y, y), MakeConstInt(0, GetScalarDtype(ret)),
                       CanProveGE(y.Eval(), 1) || CanProveLE(y.Eval(), -1));
  PYPTO_TRY_REWRITE_IF(floormod(y * x, y), MakeConstInt(0, GetScalarDtype(ret)),
                       CanProveGE(y.Eval(), 1) || CanProveLE(y.Eval(), -1));

  // floormod(x + y * c1, c2) => floormod(x + y * floormod(c1, c2), c2)
  // Guard: c2 > 0 and c1 is actually reduced
  PYPTO_TRY_REWRITE_IF(
      floormod(x + y * c1, c2), floormod(x + y * floormod(c1, c2), c2),
      c2.Eval()->value_ > 0 && arith::floormod(c1.Eval()->value_, c2.Eval()->value_) != c1.Eval()->value_);

  // floormod(x * c1, x * c2) => x * floormod(c1, c2) when c2 != 0 and x != 0
  PYPTO_TRY_REWRITE_IF(floormod(x * c1, x * c2), x * floormod(c1, c2),
                       c2.Eval()->value_ != 0 && (CanProveGE(x.Eval(), 1) || CanProveLE(x.Eval(), -1)));

  // floormod(x + c1, c2) => floormod(x + floormod(c1, c2), c2)
  // when c2 > 0 and c1 % c2 != 0 and |c1| >= c2
  PYPTO_TRY_REWRITE_IF(floormod(x + c1, c2), floormod(x + floormod(c1, c2), c2),
                       c2.Eval()->value_ > 0 && c1.Eval()->value_ % c2.Eval()->value_ != 0 &&
                           (c1.Eval()->value_ >= c2.Eval()->value_ || c1.Eval()->value_ < 0));

  // Bound-dependent: floormod(x, c) => x when 0 <= x < c (dormant in PR 3)
  PYPTO_TRY_REWRITE_IF(
      floormod(x, c1), x,
      c1.Eval()->value_ > 0 && CanProveGE(x.Eval(), 0) && CanProveLess(x.Eval(), c1.Eval()->value_));

  return ret;
}

// ============================================================================
// Min rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const MinPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Min, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeMin);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // min(x, x) => x
  PYPTO_TRY_REWRITE(min(x, x), x);

  // Constant reassociation
  // min(min(x, c1), c2) => min(x, min(c1, c2))
  PYPTO_TRY_REWRITE(min(min(x, c1), c2), min(x, min(c1, c2)));
  // min(c1, min(x, c2)) => min(x, min(c1, c2))
  PYPTO_TRY_REWRITE(min(c1, min(x, c2)), min(x, min(c1, c2)));

  // Factor out common subexpressions
  // min(y - x, z - x) => min(y, z) - x
  PYPTO_TRY_REWRITE(min(y - x, z - x), min(y, z) - x);
  // min(x - y, x - z) => x - max(y, z)
  PYPTO_TRY_REWRITE(min(x - y, x - z), x - max(y, z));
  // min(x + y, x + z) => x + min(y, z)
  PYPTO_TRY_REWRITE(min(x + y, x + z), x + min(y, z));
  // min(y + x, x + z) => x + min(y, z)
  PYPTO_TRY_REWRITE(min(y + x, x + z), x + min(y, z));
  // min(x + y, z + x) => x + min(y, z)
  PYPTO_TRY_REWRITE(min(x + y, z + x), x + min(y, z));
  // min(y + x, z + x) => x + min(y, z)
  PYPTO_TRY_REWRITE(min(y + x, z + x), x + min(y, z));

  // Constant comparison for add offsets
  // min(x + c1, x + c2) => x + min(c1, c2) (resolves to x + smaller)
  if ((min(x + c1, x + c2)).Match(ret)) {
    if (c1.Eval()->value_ < c2.Eval()->value_) {
      return RecursiveRewrite((x + c1).Eval());
    } else {
      return RecursiveRewrite((x + c2).Eval());
    }
  }
  // min(x + c1, x) and min(x, x + c1)
  if ((min(x + c1, x)).Match(ret) || (min(x, x + c1)).Match(ret)) {
    if (c1.Eval()->value_ < 0) {
      return RecursiveRewrite((x + c1).Eval());
    } else {
      return x.Eval();
    }
  }
  // min(c1 - x, c2 - x) => min(c1, c2) - x
  if ((min(c1 - x, c2 - x)).Match(ret)) {
    if (c1.Eval()->value_ < c2.Eval()->value_) {
      return RecursiveRewrite((c1 - x).Eval());
    } else {
      return RecursiveRewrite((c2 - x).Eval());
    }
  }

  // Nested collapse
  // min(min(x, y), x) => min(x, y)
  PYPTO_TRY_REWRITE(min(min(x, y), x), min(x, y));
  // min(min(x, y), y) => min(x, y)
  PYPTO_TRY_REWRITE(min(min(x, y), y), min(x, y));
  // min(x, min(x, y)) => min(x, y)
  PYPTO_TRY_REWRITE(min(x, min(x, y)), min(x, y));
  // min(y, min(x, y)) => min(x, y)
  PYPTO_TRY_REWRITE(min(y, min(x, y)), min(x, y));

  // Absorption
  // min(max(x, y), y) => y
  PYPTO_TRY_REWRITE(min(max(x, y), y), y);
  // min(max(y, x), x) => x
  PYPTO_TRY_REWRITE(min(max(y, x), x), x);
  // min(y, max(x, y)) => y
  PYPTO_TRY_REWRITE(min(y, max(x, y)), y);
  // min(max(x, y), x) => x
  PYPTO_TRY_REWRITE(min(max(x, y), x), x);
  // min(x, max(x, y)) => x
  PYPTO_TRY_REWRITE(min(x, max(x, y)), x);
  // min(x, max(y, x)) => x
  PYPTO_TRY_REWRITE(min(x, max(y, x)), x);

  // Cross min/max distribution
  // min(max(x,y), max(x,z)) => max(min(y,z), x)
  PYPTO_TRY_REWRITE(min(max(x, y), max(x, z)), max(min(y, z), x));
  PYPTO_TRY_REWRITE(min(max(x, y), max(z, x)), max(min(y, z), x));
  PYPTO_TRY_REWRITE(min(max(y, x), max(x, z)), max(min(y, z), x));
  PYPTO_TRY_REWRITE(min(max(y, x), max(z, x)), max(min(y, z), x));
  // min(min(x,y), min(x,z)) => min(min(y,z), x)
  PYPTO_TRY_REWRITE(min(min(x, y), min(x, z)), min(min(y, z), x));
  PYPTO_TRY_REWRITE(min(min(x, y), min(z, x)), min(min(y, z), x));
  PYPTO_TRY_REWRITE(min(min(y, x), min(x, z)), min(min(y, z), x));
  PYPTO_TRY_REWRITE(min(min(y, x), min(z, x)), min(min(y, z), x));

  // Scaling
  // min(floordiv(x, c1), floordiv(y, c1)) => floordiv(min(x,y), c1) when c1 > 0
  if ((min(floordiv(x, c1), floordiv(y, c1))).Match(ret)) {
    if (c1.Eval()->value_ > 0) {
      return RecursiveRewrite(floordiv(min(x, y), c1).Eval());
    } else if (c1.Eval()->value_ < 0) {
      return RecursiveRewrite(floordiv(max(x, y), c1).Eval());
    }
  }
  // min(x * c1, y * c1) => min(x,y) * c1 when c1 > 0
  if ((min(x * c1, y * c1)).Match(ret)) {
    if (c1.Eval()->value_ > 0) {
      return RecursiveRewrite((min(x, y) * c1).Eval());
    } else if (c1.Eval()->value_ < 0) {
      return RecursiveRewrite((max(x, y) * c1).Eval());
    }
  }

  // Bound-dependent (dormant in PR 3)
  PYPTO_TRY_REWRITE_IF(min(x, y), x, CanProveLE(MakeSub(x.Eval(), y.Eval()), 0));
  PYPTO_TRY_REWRITE_IF(min(x, y), y, CanProveGE(MakeSub(x.Eval(), y.Eval()), 0));

  // Canonicalization: constants to the right
  // min(c1, x) => min(x, c1)
  PYPTO_TRY_REWRITE(min(c1, x), min(x, c1));
  // min(min(x, c1), y) => min(min(x, y), c1) (move const outward)
  PYPTO_TRY_REWRITE(min(min(x, c1), y), min(min(x, y), c1));

  return ret;
}

// ============================================================================
// Max rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const MaxPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Max, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeMax);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // max(x, x) => x
  PYPTO_TRY_REWRITE(max(x, x), x);

  // Constant reassociation
  // max(max(x, c1), c2) => max(x, max(c1, c2))
  PYPTO_TRY_REWRITE(max(max(x, c1), c2), max(x, max(c1, c2)));
  // max(c1, max(x, c2)) => max(x, max(c1, c2))
  PYPTO_TRY_REWRITE(max(c1, max(x, c2)), max(x, max(c1, c2)));

  // Factor out common subexpressions
  // max(y - x, z - x) => max(y, z) - x
  PYPTO_TRY_REWRITE(max(y - x, z - x), max(y, z) - x);
  // max(x - y, x - z) => x - min(y, z)
  PYPTO_TRY_REWRITE(max(x - y, x - z), x - min(y, z));
  // max(x + y, x + z) => x + max(y, z)
  PYPTO_TRY_REWRITE(max(x + y, x + z), x + max(y, z));
  // max(y + x, x + z) => x + max(y, z)
  PYPTO_TRY_REWRITE(max(y + x, x + z), x + max(y, z));
  // max(x + y, z + x) => x + max(y, z)
  PYPTO_TRY_REWRITE(max(x + y, z + x), x + max(y, z));
  // max(y + x, z + x) => x + max(y, z)
  PYPTO_TRY_REWRITE(max(y + x, z + x), x + max(y, z));

  // Constant comparison for add offsets
  // max(x + c1, x + c2) => x + max(c1, c2)
  if ((max(x + c1, x + c2)).Match(ret)) {
    if (c1.Eval()->value_ > c2.Eval()->value_) {
      return RecursiveRewrite((x + c1).Eval());
    } else {
      return RecursiveRewrite((x + c2).Eval());
    }
  }
  // max(x + c1, x) and max(x, x + c1)
  if ((max(x + c1, x)).Match(ret) || (max(x, x + c1)).Match(ret)) {
    if (c1.Eval()->value_ > 0) {
      return RecursiveRewrite((x + c1).Eval());
    } else {
      return x.Eval();
    }
  }
  // max(c1 - x, c2 - x) => max(c1, c2) - x
  if ((max(c1 - x, c2 - x)).Match(ret)) {
    if (c1.Eval()->value_ > c2.Eval()->value_) {
      return RecursiveRewrite((c1 - x).Eval());
    } else {
      return RecursiveRewrite((c2 - x).Eval());
    }
  }

  // Nested collapse
  // max(max(x, y), x) => max(x, y)
  PYPTO_TRY_REWRITE(max(max(x, y), x), max(x, y));
  // max(max(x, y), y) => max(x, y)
  PYPTO_TRY_REWRITE(max(max(x, y), y), max(x, y));
  // max(x, max(x, y)) => max(x, y)
  PYPTO_TRY_REWRITE(max(x, max(x, y)), max(x, y));
  // max(y, max(x, y)) => max(x, y)
  PYPTO_TRY_REWRITE(max(y, max(x, y)), max(x, y));

  // Absorption
  // max(min(x, y), y) => y
  PYPTO_TRY_REWRITE(max(min(x, y), y), y);
  // max(min(y, x), x) => x
  PYPTO_TRY_REWRITE(max(min(y, x), x), x);
  // max(y, min(x, y)) => y
  PYPTO_TRY_REWRITE(max(y, min(x, y)), y);
  // max(min(x, y), x) => x
  PYPTO_TRY_REWRITE(max(min(x, y), x), x);
  // max(x, min(x, y)) => x
  PYPTO_TRY_REWRITE(max(x, min(x, y)), x);
  // max(x, min(y, x)) => x
  PYPTO_TRY_REWRITE(max(x, min(y, x)), x);

  // Cross max/min distribution
  // max(min(x,y), min(x,z)) => min(max(y,z), x)
  PYPTO_TRY_REWRITE(max(min(x, y), min(x, z)), min(max(y, z), x));
  PYPTO_TRY_REWRITE(max(min(x, y), min(z, x)), min(max(y, z), x));
  PYPTO_TRY_REWRITE(max(min(y, x), min(x, z)), min(max(y, z), x));
  PYPTO_TRY_REWRITE(max(min(y, x), min(z, x)), min(max(y, z), x));
  // max(max(x,y), max(x,z)) => max(max(y,z), x)
  PYPTO_TRY_REWRITE(max(max(x, y), max(x, z)), max(max(y, z), x));
  PYPTO_TRY_REWRITE(max(max(x, y), max(z, x)), max(max(y, z), x));
  PYPTO_TRY_REWRITE(max(max(y, x), max(x, z)), max(max(y, z), x));
  PYPTO_TRY_REWRITE(max(max(y, x), max(z, x)), max(max(y, z), x));

  // Scaling
  // max(floordiv(x, c1), floordiv(y, c1)) => floordiv(max(x,y), c1) when c1 > 0
  if ((max(floordiv(x, c1), floordiv(y, c1))).Match(ret)) {
    if (c1.Eval()->value_ > 0) {
      return RecursiveRewrite(floordiv(max(x, y), c1).Eval());
    } else if (c1.Eval()->value_ < 0) {
      return RecursiveRewrite(floordiv(min(x, y), c1).Eval());
    }
  }
  // max(x * c1, y * c1) => max(x,y) * c1 when c1 > 0
  if ((max(x * c1, y * c1)).Match(ret)) {
    if (c1.Eval()->value_ > 0) {
      return RecursiveRewrite((max(x, y) * c1).Eval());
    } else if (c1.Eval()->value_ < 0) {
      return RecursiveRewrite((min(x, y) * c1).Eval());
    }
  }

  // Bound-dependent (dormant in PR 3)
  PYPTO_TRY_REWRITE_IF(max(x, y), x, CanProveGE(MakeSub(x.Eval(), y.Eval()), 0));
  PYPTO_TRY_REWRITE_IF(max(x, y), y, CanProveLE(MakeSub(x.Eval(), y.Eval()), 0));

  // Canonicalization
  // max(c1, x) => max(x, c1)
  PYPTO_TRY_REWRITE(max(c1, x), max(x, c1));
  // max(max(x, c1), y) => max(max(x, y), c1) (move const outward)
  PYPTO_TRY_REWRITE(max(max(x, c1), y), max(max(x, y), c1));

  return ret;
}

// ============================================================================
// Comparison rules: Eq
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const EqPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Eq, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeEq);

  // Skip rewrite rules for float operands (NaN breaks reflexivity/trichotomy)
  if (GetScalarDtype(a).IsFloat()) return ret;

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x == x => true
  PYPTO_TRY_REWRITE(x == x, MakeConstBool(true));

  // Cancellation
  // x + y == x + z => y == z
  PYPTO_TRY_REWRITE(x + y == x + z, y == z);
  // y + x == x + z => y == z
  PYPTO_TRY_REWRITE(y + x == x + z, y == z);
  // x + y == z + x => y == z
  PYPTO_TRY_REWRITE(x + y == z + x, y == z);
  // y + x == z + x => y == z
  PYPTO_TRY_REWRITE(y + x == z + x, y == z);

  // Constant rearrangement
  // x - c1 == c2 => x == c1 + c2
  PYPTO_TRY_REWRITE(x - c1 == c2, x == c1 + c2);
  // x + c1 == c2 => x == c2 - c1
  PYPTO_TRY_REWRITE(x + c1 == c2, x == c2 - c1);
  // c1 - x == c2 => x == c1 - c2
  PYPTO_TRY_REWRITE(c1 - x == c2, x == c1 - c2);
  // c1 == x => x == c1  (canonicalize)
  PYPTO_TRY_REWRITE(c1 == x, x == c1);

  return ret;
}

// ============================================================================
// Comparison rules: Ne
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const NePtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Ne, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeNe);

  if (GetScalarDtype(a).IsFloat()) return ret;

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x != x => false
  PYPTO_TRY_REWRITE(x != x, MakeConstBool(false));

  // Cancellation
  PYPTO_TRY_REWRITE(x + y != x + z, y != z);
  PYPTO_TRY_REWRITE(y + x != x + z, y != z);
  PYPTO_TRY_REWRITE(x + y != z + x, y != z);
  PYPTO_TRY_REWRITE(y + x != z + x, y != z);

  // Constant rearrangement
  PYPTO_TRY_REWRITE(x - c1 != c2, x != c1 + c2);
  PYPTO_TRY_REWRITE(x + c1 != c2, x != c2 - c1);
  PYPTO_TRY_REWRITE(c1 != x, x != c1);

  return ret;
}

// ============================================================================
// Comparison rules: Lt
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const LtPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Lt, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeLt);

  if (GetScalarDtype(a).IsFloat()) return ret;

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x < x => false
  PYPTO_TRY_REWRITE(x < x, MakeConstBool(false));

  // Cancellation
  PYPTO_TRY_REWRITE(x + y < x + z, y < z);
  PYPTO_TRY_REWRITE(y + x < x + z, y < z);
  PYPTO_TRY_REWRITE(x + y < z + x, y < z);
  PYPTO_TRY_REWRITE(y + x < z + x, y < z);
  PYPTO_TRY_REWRITE(y - x < z - x, y < z);
  PYPTO_TRY_REWRITE(x - y < x - z, z < y);

  // x < x + z => 0 < z
  PYPTO_TRY_REWRITE(x < x + z, MakeConstInt(0, GetScalarDtype(a)) < z);
  PYPTO_TRY_REWRITE(x < z + x, MakeConstInt(0, GetScalarDtype(a)) < z);
  // x < x - z => z < 0
  PYPTO_TRY_REWRITE(x < x - z, z < MakeConstInt(0, GetScalarDtype(a)));

  // Constant rearrangement
  PYPTO_TRY_REWRITE(x + c1 < c2, x < c2 - c1);
  PYPTO_TRY_REWRITE(x - c1 < c2, x < c2 + c1);
  PYPTO_TRY_REWRITE(c1 - x < c2, c1 - c2 < x);
  PYPTO_TRY_REWRITE(c1 < x + c2, c1 - c2 < x);
  PYPTO_TRY_REWRITE(c1 < x - c2, c1 + c2 < x);

  // Multiply by positive/negative constant
  PYPTO_TRY_REWRITE_IF(x * c1 < y * c1, x < y, c1.Eval()->value_ > 0);
  PYPTO_TRY_REWRITE_IF(x * c1 < y * c1, y < x, c1.Eval()->value_ < 0);

  // FloorDiv comparisons
  // floordiv(x, c1) < c2 => x < c1 * c2 when c1 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x, c1) < c2, x < c1 * c2, c1.Eval()->value_ > 0);
  // c1 < floordiv(x, c2) => (c1+1)*c2 - 1 < x when c2 > 0
  PYPTO_TRY_REWRITE_IF(c1 < floordiv(x, c2), (c1 + 1) * c2 - 1 < x, c2.Eval()->value_ > 0);
  // floordiv(x, c1) * c1 < x => 0 < floormod(x, c1) when c1 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x, MakeConstInt(0, GetScalarDtype(a)) < floormod(x, c1),
                       c1.Eval()->value_ > 0);
  // floordiv(x, c1) * c1 < x + y => 0 < floormod(x, c1) + y when c1 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x + y, MakeConstInt(0, GetScalarDtype(a)) < floormod(x, c1) + y,
                       c1.Eval()->value_ > 0);
  // floordiv(x, c1) * c1 < x - y => y < floormod(x, c1) when c1 > 0
  PYPTO_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x - y, y < floormod(x, c1), c1.Eval()->value_ > 0);

  // Min/Max decomposition into boolean
  // min(x, y) < z => x < z || y < z
  PYPTO_TRY_REWRITE(min(x, y) < z, (x < z) || (y < z));
  // max(x, y) < z => x < z && y < z
  PYPTO_TRY_REWRITE(max(x, y) < z, (x < z) && (y < z));
  // z < min(x, y) => z < x && z < y
  PYPTO_TRY_REWRITE(z < min(x, y), (z < x) && (z < y));
  // z < max(x, y) => z < x || z < y
  PYPTO_TRY_REWRITE(z < max(x, y), (z < x) || (z < y));

  return ret;
}

// ============================================================================
// Comparison rules: Le
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const LePtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Le, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeLe);

  if (GetScalarDtype(a).IsFloat()) return ret;

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x <= x => true
  PYPTO_TRY_REWRITE(x <= x, MakeConstBool(true));

  // Cancellation
  PYPTO_TRY_REWRITE(x + y <= x + z, y <= z);
  PYPTO_TRY_REWRITE(y + x <= x + z, y <= z);
  PYPTO_TRY_REWRITE(x + y <= z + x, y <= z);
  PYPTO_TRY_REWRITE(y + x <= z + x, y <= z);
  PYPTO_TRY_REWRITE(y - x <= z - x, y <= z);
  // x - y <= x - z => z <= y
  PYPTO_TRY_REWRITE(x - y <= x - z, z <= y);

  // Constant rearrangement
  PYPTO_TRY_REWRITE(x + c1 <= c2, x <= c2 - c1);
  PYPTO_TRY_REWRITE(x - c1 <= c2, x <= c2 + c1);
  PYPTO_TRY_REWRITE(c1 <= x + c2, c1 - c2 <= x);
  PYPTO_TRY_REWRITE(c1 <= x - c2, c1 + c2 <= x);
  PYPTO_TRY_REWRITE(c1 - x <= c2, c1 - c2 <= x);

  // Multiply by positive/negative constant
  PYPTO_TRY_REWRITE_IF(x * c1 <= y * c1, x <= y, c1.Eval()->value_ > 0);
  PYPTO_TRY_REWRITE_IF(x * c1 <= y * c1, y <= x, c1.Eval()->value_ < 0);

  return ret;
}

// ============================================================================
// Comparison rules: Gt — delegate to Lt(b, a)
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const GtPtr& op) {
  // a > b  ⟺  b < a
  return VisitExpr(MakeLt(op->right_, op->left_));
}

// ============================================================================
// Comparison rules: Ge — delegate to Le(b, a)
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const GePtr& op) {
  // a >= b  ⟺  b <= a
  return VisitExpr(MakeLe(op->right_, op->left_));
}

// ============================================================================
// Not rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const NotPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);

  if (auto folded = TryConstFoldUnary(ObjectKind::Not, a)) return folded;

  ExprPtr ret = MutateUnary(op, a, MakeNot);

  PVar<ExprPtr> x, y;

  // Double negation: !!x => x
  PYPTO_TRY_REWRITE(!(!x), x);

  // Comparison negation
  // !(x < y) => y <= x
  PYPTO_TRY_REWRITE(!(x < y), y <= x);
  // !(x <= y) => y < x
  PYPTO_TRY_REWRITE(!(x <= y), y < x);
  // !(x > y) => y >= x
  PYPTO_TRY_REWRITE(!(x > y), y >= x);
  // !(x >= y) => y > x
  PYPTO_TRY_REWRITE(!(x >= y), y > x);
  // !(x == y) => x != y
  PYPTO_TRY_REWRITE(!(x == y), x != y);
  // !(x != y) => x == y
  PYPTO_TRY_REWRITE(!(x != y), x == y);

  // De Morgan's laws
  // !(x || y) => !x && !y
  PYPTO_TRY_REWRITE(!(x || y), (!x) && (!y));
  // !(x && y) => !x || !y
  PYPTO_TRY_REWRITE(!(x && y), (!x) || (!y));

  return ret;
}

// ============================================================================
// And rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const AndPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::And, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeAnd);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x && x => x
  PYPTO_TRY_REWRITE(x && x, x);

  // Contradiction
  // x && !x => false
  PYPTO_TRY_REWRITE(x && (!x), MakeConstBool(false));
  // !x && x => false
  PYPTO_TRY_REWRITE((!x) && x, MakeConstBool(false));

  // x == y && x != y => false
  PYPTO_TRY_REWRITE((x == y) && (x != y), MakeConstBool(false));
  PYPTO_TRY_REWRITE((x != y) && (x == y), MakeConstBool(false));

  // x <= y && y < x => false
  PYPTO_TRY_REWRITE((x <= y) && (y < x), MakeConstBool(false));
  PYPTO_TRY_REWRITE((y < x) && (x <= y), MakeConstBool(false));

  // Range contradictions
  // x < c1 && c2 < x => false when c2 + 1 >= c1
  PYPTO_TRY_REWRITE_IF((x < c1) && (c2 < x), MakeConstBool(false),
                       c2.Eval()->value_ + 1 >= c1.Eval()->value_);
  PYPTO_TRY_REWRITE_IF((c2 < x) && (x < c1), MakeConstBool(false),
                       c2.Eval()->value_ + 1 >= c1.Eval()->value_);

  // x <= c1 && c2 <= x => false when c2 > c1
  PYPTO_TRY_REWRITE_IF((x <= c1) && (c2 <= x), MakeConstBool(false), c2.Eval()->value_ > c1.Eval()->value_);
  PYPTO_TRY_REWRITE_IF((c2 <= x) && (x <= c1), MakeConstBool(false), c2.Eval()->value_ > c1.Eval()->value_);

  // Associativity
  PYPTO_TRY_REWRITE(x && (y && z), (x && y) && z);

  return ret;
}

// ============================================================================
// Or rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const OrPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);

  if (auto folded = TryConstFoldBinary(ObjectKind::Or, a, b)) return folded;

  ExprPtr ret = MutateBinary(op, a, b, MakeOr);

  PVar<ExprPtr> x, y, z;
  PVar<ConstIntPtr> c1, c2;

  // x || x => x
  PYPTO_TRY_REWRITE(x || x, x);

  // Tautology
  // x || !x => true
  PYPTO_TRY_REWRITE(x || (!x), MakeConstBool(true));
  // !x || x => true
  PYPTO_TRY_REWRITE((!x) || x, MakeConstBool(true));

  // x == y || x != y => true
  PYPTO_TRY_REWRITE((x == y) || (x != y), MakeConstBool(true));
  PYPTO_TRY_REWRITE((x != y) || (x == y), MakeConstBool(true));

  // x <= y || y < x => true
  PYPTO_TRY_REWRITE((x <= y) || (y < x), MakeConstBool(true));
  PYPTO_TRY_REWRITE((y < x) || (x <= y), MakeConstBool(true));

  // x < y || y < x => x != y
  PYPTO_TRY_REWRITE((x < y) || (y < x), x != y);

  // Range tautologies
  // x < c1 || c2 < x => true when c2 < c1
  PYPTO_TRY_REWRITE_IF((x < c1) || (c2 < x), MakeConstBool(true), c2.Eval()->value_ < c1.Eval()->value_);
  PYPTO_TRY_REWRITE_IF((c2 < x) || (x < c1), MakeConstBool(true), c2.Eval()->value_ < c1.Eval()->value_);

  // x <= c1 || c2 <= x => true when c2 <= c1 + 1
  PYPTO_TRY_REWRITE_IF((x <= c1) || (c2 <= x), MakeConstBool(true),
                       c2.Eval()->value_ <= c1.Eval()->value_ + 1);
  PYPTO_TRY_REWRITE_IF((c2 <= x) || (x <= c1), MakeConstBool(true),
                       c2.Eval()->value_ <= c1.Eval()->value_ + 1);

  // x < y || x == y => x <= y
  PYPTO_TRY_REWRITE((x < y) || (x == y), x <= y);
  PYPTO_TRY_REWRITE((x < y) || (y == x), x <= y);
  PYPTO_TRY_REWRITE((x == y) || (x < y), x <= y);
  PYPTO_TRY_REWRITE((y == x) || (x < y), x <= y);

  // Associativity
  PYPTO_TRY_REWRITE(x || (y || z), (x || y) || z);

  return ret;
}

// ============================================================================
// Neg rules
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const NegPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);

  if (auto folded = TryConstFoldUnary(ObjectKind::Neg, a)) return folded;

  ExprPtr ret = MutateUnary(op, a, MakeNeg);

  PVar<ExprPtr> x, y;

  // neg(neg(x)) => x
  PYPTO_TRY_REWRITE(neg(neg(x)), x);
  // neg(x - y) => y - x
  PYPTO_TRY_REWRITE(neg(x - y), y - x);

  return ret;
}

// ============================================================================
// Passthrough visitors (simplify children + const fold only)
// ============================================================================

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const FloatDivPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::FloatDiv, a, b)) return folded;
  return MutateBinary(op, a, b, MakeFloatDiv);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const PowPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Pow, a, b)) return folded;
  return MutateBinary(op, a, b, MakePow);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const XorPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Xor, a, b)) return folded;
  if (a.get() == op->left_.get() && b.get() == op->right_.get()) return op;
  return std::make_shared<Xor>(a, b, DataType::BOOL, op->span_);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitAndPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitAnd, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitAnd);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitOrPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitOr, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitOr);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitXorPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitXor, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitXor);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitShiftLeftPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitShiftLeft, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitShiftLeft);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitShiftRightPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitShiftRight, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitShiftRight);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const AbsPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::Abs, a)) return folded;
  if (a.get() == op->operand_.get()) return op;
  return std::make_shared<Abs>(a, GetScalarDtype(a), op->span_);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const BitNotPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::BitNot, a)) return folded;
  return MutateUnary(op, a, MakeBitNot);
}

ExprPtr RewriteSimplifier::Impl::VisitExpr_(const CastPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (a.get() == op->operand_.get()) return op;
  return MakeCast(a, GetScalarDtype(op));
}

// ============================================================================
// RewriteSimplifier — public interface delegation to Impl
// ============================================================================

RewriteSimplifier::RewriteSimplifier() : impl_(std::make_unique<Impl>(nullptr)) {}

RewriteSimplifier::RewriteSimplifier(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

RewriteSimplifier::~RewriteSimplifier() = default;

RewriteSimplifier::RewriteSimplifier(RewriteSimplifier&&) noexcept = default;
RewriteSimplifier& RewriteSimplifier::operator=(RewriteSimplifier&&) noexcept = default;

ExprPtr RewriteSimplifier::operator()(const ExprPtr& expr) const { return impl_->VisitExpr(expr); }

void RewriteSimplifier::Update(const VarPtr& var, const ExprPtr& new_expr) { impl_->Update(var, new_expr); }

std::function<void()> RewriteSimplifier::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
