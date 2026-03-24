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

#include "src/ir/arith/canonical_simplify.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"
#include "src/ir/arith/rewrite_simplify.h"  // CompareResult

namespace pypto {
namespace ir {
namespace arith {

// ============================================================================
// Helper: reconstruct binary/unary expr only if children changed
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
// Constructor and common methods
// ============================================================================

CanonicalSimplifier::Impl::Impl(Analyzer* parent) : parent_(parent) {}

ExprPtr CanonicalSimplifier::Impl::VisitExpr(const ExprPtr& expr) {
  // Skip algebraic simplification for float-typed expressions.
  if (auto stype = std::dynamic_pointer_cast<const ScalarType>(expr->GetType());
      stype && stype->dtype_.IsFloat()) {
    return expr;
  }
  return ExprFunctor<ExprPtr>::VisitExpr(expr);
}

CompareResult CanonicalSimplifier::Impl::TryCompare(const ExprPtr& /*x*/, int64_t /*val*/) {
  // Full implementation using parent_->const_int_bound() comes in PR 6.
  // In standalone mode (no parent), all comparisons are unknown.
  return CompareResult::kUnknown;
}

void CanonicalSimplifier::Impl::Update(const VarPtr& var, const ExprPtr& info) {
  if (info) {
    var_map_[var.get()] = info;
  } else {
    var_map_.erase(var.get());
  }
}

std::function<void()> CanonicalSimplifier::Impl::EnterConstraint(const ExprPtr& /*constraint*/) {
  // Canonical simplifier's constraint handling works via the parent Analyzer's
  // bounds (used in TrySumFloorDiv and similar). In standalone mode, this is a no-op.
  return nullptr;
}

// ============================================================================
// Leaf nodes
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const VarPtr& op) {
  auto it = var_map_.find(op.get());
  if (it != var_map_.end()) return it->second;
  return op;
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const IterArgPtr& op) {
  auto it = var_map_.find(op.get());
  if (it != var_map_.end()) return it->second;
  return op;
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const ConstIntPtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const ConstFloatPtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const ConstBoolPtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const MemRefPtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const CallPtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const MakeTuplePtr& op) { return op; }
ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const TupleGetItemExprPtr& op) { return op; }

// ============================================================================
// Core canonical form operations
// ============================================================================

SumExpr CanonicalSimplifier::Impl::GetOrCreateSumExpr(const ExprPtr& expr) {
  // Check cache first — arithmetic visitors populate this
  auto it = sum_cache_.find(expr.get());
  if (it != sum_cache_.end()) return it->second;

  // Basic conversion: ConstInt → base, everything else → single term
  SumExpr result;
  result.dtype = GetScalarDtype(expr);

  if (auto ci = As<ConstInt>(expr)) {
    result.base = ci->value_;
    return result;
  }

  SplitExpr split;
  split.index = expr;
  split.lower_factor = 1;
  split.upper_factor = ConstIntBound::kPosInf;
  split.scale = 1;
  result.args.push_back(std::move(split));
  return result;
}

ExprPtr CanonicalSimplifier::Impl::NormalizeAndCache(const SumExpr& sum) {
  ExprPtr result = Normalize(sum);
  sum_cache_[result.get()] = sum;
  return result;
}

ExprPtr CanonicalSimplifier::Impl::Normalize(const SplitExpr& split, DataType dtype) {
  ExprPtr index = split.index;

  // Apply modulo: (index % upper_factor)
  if (split.upper_factor != ConstIntBound::kPosInf) {
    index = MakeFloorMod(index, MakeConstInt(split.upper_factor, dtype));
  }

  // Apply division: ... / lower_factor
  if (split.lower_factor != 1) {
    index = MakeFloorDiv(index, MakeConstInt(split.lower_factor, dtype));
  }

  // Apply scale: ... * scale
  if (split.scale == 1) return index;
  if (split.scale == -1) return MakeNeg(index);
  return MakeMul(index, MakeConstInt(split.scale, dtype));
}

ExprPtr CanonicalSimplifier::Impl::Normalize(const SumExpr& sum) {
  // Collect non-zero terms
  std::vector<ExprPtr> terms;
  for (const auto& arg : sum.args) {
    if (arg.scale == 0) continue;
    terms.push_back(Normalize(arg, sum.dtype));
  }

  if (terms.empty()) {
    return MakeConstInt(sum.base, sum.dtype);
  }

  // Build sum: term0 + term1 + ... + base
  ExprPtr result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = MakeAdd(result, terms[i]);
  }

  if (sum.base != 0) {
    result = MakeAdd(result, MakeConstInt(sum.base, sum.dtype));
  }

  return result;
}

// ============================================================================
// SumExpr manipulation
// ============================================================================

bool CanonicalSimplifier::Impl::TryMergeSameFactors(const SplitExpr& a, const SplitExpr& b,
                                                    SplitExpr* merged) {
  // Same index (pointer identity), same lower_factor, same upper_factor → add scales
  if (a.index.get() != b.index.get()) return false;
  if (a.lower_factor != b.lower_factor) return false;
  if (a.upper_factor != b.upper_factor) return false;
  if (AddWouldOverflow(a.scale, b.scale)) return false;
  *merged = a;
  merged->scale = a.scale + b.scale;
  return true;
}

bool CanonicalSimplifier::Impl::TryMergeDivModPair(const SplitExpr& a, const SplitExpr& b,
                                                   SplitExpr* merged) {
  // Detect complementary div/mod splits and recombine:
  //   SplitExpr{x, L, U_a, S_a} + SplitExpr{x, 1, L, S_b}
  //   where S_a == L * S_b and U_a == kPosInf (or compatible)
  // → SplitExpr{x, 1, U_a, S_b}
  //
  // Example: (x // 4) * 4 + x % 4
  //   = SplitExpr{x, 4, kPosInf, 4} + SplitExpr{x, 1, 4, 1}
  //   → SplitExpr{x, 1, kPosInf, 1} = x

  if (a.index.get() != b.index.get()) return false;

  // Try a as the "div" part and b as the "mod" part
  // div part: lower_factor = L, scale = L * base_scale
  // mod part: lower_factor = 1, upper_factor = L, scale = base_scale
  auto try_merge = [](const SplitExpr& div_part, const SplitExpr& mod_part, SplitExpr* out) -> bool {
    // mod part must have lower_factor == 1 (or more generally, mod_part.lower_factor < div_part.lower_factor)
    if (mod_part.lower_factor != 1) return false;
    // The boundary must match: div_part.lower_factor == mod_part.upper_factor
    if (mod_part.upper_factor == ConstIntBound::kPosInf) return false;
    if (div_part.lower_factor != mod_part.upper_factor) return false;
    // Scale relationship: div_part.scale == div_part.lower_factor * mod_part.scale
    if (MulWouldOverflow(div_part.lower_factor, mod_part.scale)) return false;
    if (div_part.scale != div_part.lower_factor * mod_part.scale) return false;

    // Merge: take the mod part's lower_factor (1) and the div part's upper_factor
    *out = mod_part;
    out->upper_factor = div_part.upper_factor;
    out->scale = mod_part.scale;
    return true;
  };

  // Try both orderings
  if (try_merge(a, b, merged)) return true;
  if (try_merge(b, a, merged)) return true;
  return false;
}

void CanonicalSimplifier::Impl::MergeTerms(std::vector<SplitExpr>& args) {
  // Try all pairwise merges (same-factor merge and div-mod recombination).
  // Repeat until no more merges found. Each iteration zeros at least one term's
  // scale, so the loop terminates in at most K iterations (O(K^3) total where
  // K is the number of terms in a single expression — not the IR size).
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].scale == 0) continue;
      for (size_t j = i + 1; j < args.size(); ++j) {
        if (args[j].scale == 0) continue;
        SplitExpr merged;
        if (TryMergeSameFactors(args[i], args[j], &merged) || TryMergeDivModPair(args[i], args[j], &merged)) {
          args[i] = merged;
          args[j].scale = 0;  // Mark for removal
          changed = true;
        }
      }
    }
  }
  // Remove zero-scale terms
  args.erase(std::remove_if(args.begin(), args.end(), [](const SplitExpr& s) { return s.scale == 0; }),
             args.end());
}

SumExpr CanonicalSimplifier::Impl::SumAdd(const SumExpr& lhs, const SumExpr& rhs) {
  SumExpr result;
  result.dtype = lhs.dtype;

  // Overflow-safe base addition
  if (AddWouldOverflow(lhs.base, rhs.base)) {
    // Fall back: normalize both sides and wrap
    result.base = 0;
    SplitExpr s;
    s.index = MakeAdd(Normalize(lhs), Normalize(rhs));
    s.scale = 1;
    result.args.push_back(std::move(s));
    return result;
  }
  result.base = lhs.base + rhs.base;

  // Collect all args
  result.args = lhs.args;
  result.args.insert(result.args.end(), rhs.args.begin(), rhs.args.end());

  // Merge like terms and complementary div/mod pairs
  MergeTerms(result.args);

  return result;
}

SumExpr CanonicalSimplifier::Impl::SumNegate(const SumExpr& sum) {
  SumExpr result;
  result.dtype = sum.dtype;
  if (NegWouldOverflow(sum.base)) {
    // Extremely rare edge case (base == INT64_MIN)
    result.base = 0;
    SplitExpr split;
    split.index = Normalize(sum);
    split.scale = -1;
    result.args.push_back(std::move(split));
    return result;
  }
  result.base = -sum.base;
  result.args = sum.args;
  for (auto& arg : result.args) {
    if (NegWouldOverflow(arg.scale)) {
      // Extremely rare — normalize this term and wrap
      arg.index = Normalize(arg, sum.dtype);
      arg.lower_factor = 1;
      arg.upper_factor = ConstIntBound::kPosInf;
      arg.scale = -1;
    } else {
      arg.scale = -arg.scale;
    }
  }
  return result;
}

SumExpr CanonicalSimplifier::Impl::SumMulConst(const SumExpr& sum, int64_t c) {
  if (c == 0) {
    SumExpr result;
    result.dtype = sum.dtype;
    return result;
  }
  if (c == 1) return sum;

  SumExpr result;
  result.dtype = sum.dtype;
  if (MulWouldOverflow(sum.base, c)) {
    result.base = 0;
    SplitExpr split;
    split.index = Normalize(sum);
    split.scale = c;
    result.args.push_back(std::move(split));
    return result;
  }
  result.base = sum.base * c;
  result.args = sum.args;
  for (auto& arg : result.args) {
    if (MulWouldOverflow(arg.scale, c)) {
      arg.index = Normalize(arg, sum.dtype);
      arg.lower_factor = 1;
      arg.upper_factor = ConstIntBound::kPosInf;
      arg.scale = c;
    } else {
      arg.scale *= c;
    }
  }
  return result;
}

// ============================================================================
// FloorDiv/FloorMod on SumExpr
// ============================================================================

bool CanonicalSimplifier::Impl::TrySumFloorDiv(const SumExpr& sum, int64_t divisor, SumExpr* result) {
  INTERNAL_CHECK(divisor > 0) << "TrySumFloorDiv: divisor must be positive";

  // Case 1: All scales and base are divisible by divisor.
  // e.g., (4*x + 6*y + 8) // 2 = 2*x + 3*y + 4
  {
    bool all_divisible = (floormod(sum.base, divisor) == 0);
    if (all_divisible) {
      for (const auto& arg : sum.args) {
        if (arg.scale % divisor != 0) {
          all_divisible = false;
          break;
        }
      }
    }
    if (all_divisible) {
      *result = sum;
      result->base = floordiv(sum.base, divisor);
      for (auto& arg : result->args) {
        arg.scale /= divisor;
      }
      return true;
    }
  }

  // Case 2: Single SplitExpr with base == 0.
  if (sum.args.size() == 1 && sum.base == 0) {
    const auto& split = sum.args[0];

    // Sub-case 2a: scale is divisible by divisor.
    // e.g., (x * 4) // 2 = x * 2
    // Note: C++17 guarantees truncation toward zero for %, so this works for negative scales too.
    if (split.scale % divisor == 0) {
      *result = sum;
      result->args[0].scale /= divisor;
      return true;
    }

    // Sub-case 2b: divisor is divisible by scale (positive scale only).
    // ((index % U) / L) * S  //  (S * K) = (index % U) / (L * K)
    // Only valid for positive scale because floor_div(-S*x, S*K) != -floor_div(x, K).
    if (split.scale > 0 && divisor % split.scale == 0) {
      int64_t k = divisor / split.scale;
      if (MulWouldOverflow(split.lower_factor, k)) return false;
      int64_t new_lower = split.lower_factor * k;
      // Validity: new_lower must divide upper_factor (or upper == kPosInf)
      if (split.upper_factor != ConstIntBound::kPosInf && split.upper_factor % new_lower != 0) {
        return false;
      }
      *result = sum;
      result->args[0].lower_factor = new_lower;
      result->args[0].scale = 1;
      return true;
    }
  }

  return false;
}

bool CanonicalSimplifier::Impl::TrySumFloorMod(const SumExpr& sum, int64_t divisor, SumExpr* result) {
  INTERNAL_CHECK(divisor > 0) << "TrySumFloorMod: divisor must be positive";

  // Case 1: All scales are divisible by divisor.
  // (4*x + 6*y + base) % 2 = base % 2
  {
    bool all_divisible = true;
    for (const auto& arg : sum.args) {
      if (arg.scale % divisor != 0) {
        all_divisible = false;
        break;
      }
    }
    if (all_divisible) {
      result->dtype = sum.dtype;
      result->base = floormod(sum.base, divisor);
      result->args.clear();
      return true;
    }
  }

  // Case 2: Single SplitExpr with base == 0.
  if (sum.args.size() == 1 && sum.base == 0) {
    const auto& split = sum.args[0];

    // Sub-case 2a: scale divisible by divisor → result is 0.
    // e.g., (x * 4) % 2 = 0
    // Note: C++17 guarantees truncation toward zero for %, so this works for negative scales too.
    if (split.scale != 0 && split.scale % divisor == 0) {
      result->dtype = sum.dtype;
      result->base = 0;
      result->args.clear();
      return true;
    }

    // Sub-case 2b: scale == 1 && lower_factor == 1, adjust upper_factor.
    // (index % U) % divisor → index % divisor when U % divisor == 0
    if (split.scale == 1 && split.lower_factor == 1 && split.upper_factor != ConstIntBound::kPosInf &&
        split.upper_factor % divisor == 0) {
      *result = sum;
      result->args[0].upper_factor = divisor;
      return true;
    }

    // Sub-case 2c: simple index (no mod/div).
    // index % divisor would just reconstruct the same FloorMod node.
    // Return false to let the caller preserve the original node via MutateBinary,
    // which avoids unnecessary allocation and preserves pointer identity.
    // The FloorMod visitor will still cache the SplitExpr representation.
  }

  return false;
}

// ============================================================================
// Binary arithmetic visitors
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const AddPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Add, a, b)) return folded;

  SumExpr sa = GetOrCreateSumExpr(a);
  SumExpr sb = GetOrCreateSumExpr(b);
  SumExpr merged = SumAdd(sa, sb);
  return NormalizeAndCache(merged);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const SubPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Sub, a, b)) return folded;

  SumExpr sa = GetOrCreateSumExpr(a);
  SumExpr sb = GetOrCreateSumExpr(b);
  SumExpr merged = SumAdd(sa, SumNegate(sb));
  return NormalizeAndCache(merged);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const MulPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Mul, a, b)) return folded;

  // If one side is constant, distribute over the sum
  if (auto ci = As<ConstInt>(b)) {
    SumExpr sa = GetOrCreateSumExpr(a);
    return NormalizeAndCache(SumMulConst(sa, ci->value_));
  }
  if (auto ci = As<ConstInt>(a)) {
    SumExpr sb = GetOrCreateSumExpr(b);
    return NormalizeAndCache(SumMulConst(sb, ci->value_));
  }

  // Both non-constant: can't represent product of sums in canonical form
  return MutateBinary(op, a, b, MakeMul);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const FloorDivPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::FloorDiv, a, b)) return folded;

  auto ci = As<ConstInt>(b);
  if (ci && ci->value_ > 0) {
    SumExpr sa = GetOrCreateSumExpr(a);
    SumExpr result;
    result.dtype = sa.dtype;
    if (TrySumFloorDiv(sa, ci->value_, &result)) {
      return NormalizeAndCache(result);
    }
  }

  ExprPtr ret = MutateBinary(op, a, b, MakeFloorDiv);
  // Cache the result as a SplitExpr so parent Add can do div-mod recombination
  if (ci && ci->value_ > 0) {
    SumExpr div_sum;
    div_sum.dtype = GetScalarDtype(ret);
    SplitExpr split;
    split.index = a;
    split.lower_factor = ci->value_;
    split.scale = 1;
    div_sum.args.push_back(std::move(split));
    sum_cache_[ret.get()] = std::move(div_sum);
  }
  return ret;
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const FloorModPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::FloorMod, a, b)) return folded;

  auto ci = As<ConstInt>(b);
  if (ci && ci->value_ > 0) {
    SumExpr sa = GetOrCreateSumExpr(a);
    SumExpr result;
    result.dtype = sa.dtype;
    if (TrySumFloorMod(sa, ci->value_, &result)) {
      return NormalizeAndCache(result);
    }
  }

  ExprPtr ret = MutateBinary(op, a, b, MakeFloorMod);
  // Cache the result as a SplitExpr so parent Add can do div-mod recombination
  if (ci && ci->value_ > 0) {
    SumExpr mod_sum;
    mod_sum.dtype = GetScalarDtype(ret);
    SplitExpr split;
    split.index = a;
    split.upper_factor = ci->value_;
    split.scale = 1;
    mod_sum.args.push_back(std::move(split));
    sum_cache_[ret.get()] = std::move(mod_sum);
  }
  return ret;
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const NegPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::Neg, a)) return folded;

  SumExpr sa = GetOrCreateSumExpr(a);
  return NormalizeAndCache(SumNegate(sa));
}

// ============================================================================
// Passthrough binary visitors (simplify children, const fold, reconstruct)
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const FloatDivPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::FloatDiv, a, b)) return folded;
  return MutateBinary(op, a, b, MakeFloatDiv);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const MinPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Min, a, b)) return folded;
  return MutateBinary(op, a, b, MakeMin);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const MaxPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Max, a, b)) return folded;
  return MutateBinary(op, a, b, MakeMax);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const PowPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Pow, a, b)) return folded;
  return MutateBinary(op, a, b, MakePow);
}

// ============================================================================
// Comparison visitors
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const EqPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Eq, a, b)) return folded;
  return MutateBinary(op, a, b, MakeEq);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const NePtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Ne, a, b)) return folded;
  return MutateBinary(op, a, b, MakeNe);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const LtPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Lt, a, b)) return folded;
  return MutateBinary(op, a, b, MakeLt);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const LePtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Le, a, b)) return folded;
  return MutateBinary(op, a, b, MakeLe);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const GtPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Gt, a, b)) return folded;
  return MutateBinary(op, a, b, MakeGt);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const GePtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Ge, a, b)) return folded;
  return MutateBinary(op, a, b, MakeGe);
}

// ============================================================================
// Logical visitors
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const AndPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::And, a, b)) return folded;
  return MutateBinary(op, a, b, MakeAnd);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const OrPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Or, a, b)) return folded;
  return MutateBinary(op, a, b, MakeOr);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const XorPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::Xor, a, b)) return folded;
  if (a.get() == op->left_.get() && b.get() == op->right_.get()) return op;
  return std::make_shared<Xor>(a, b, DataType::BOOL, op->span_);
}

// ============================================================================
// Bitwise visitors
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitAndPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitAnd, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitAnd);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitOrPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitOr, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitOr);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitXorPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitXor, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitXor);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitShiftLeftPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitShiftLeft, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitShiftLeft);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitShiftRightPtr& op) {
  ExprPtr a = VisitExpr(op->left_);
  ExprPtr b = VisitExpr(op->right_);
  if (auto folded = TryConstFoldBinary(ObjectKind::BitShiftRight, a, b)) return folded;
  return MutateBinary(op, a, b, MakeBitShiftRight);
}

// ============================================================================
// Unary visitors (passthrough)
// ============================================================================

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const AbsPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::Abs, a)) return folded;
  if (a.get() == op->operand_.get()) return op;
  return std::make_shared<Abs>(a, GetScalarDtype(a), op->span_);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const NotPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::Not, a)) return folded;
  return MutateUnary(op, a, MakeNot);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const BitNotPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (auto folded = TryConstFoldUnary(ObjectKind::BitNot, a)) return folded;
  return MutateUnary(op, a, MakeBitNot);
}

ExprPtr CanonicalSimplifier::Impl::VisitExpr_(const CastPtr& op) {
  ExprPtr a = VisitExpr(op->operand_);
  if (a.get() == op->operand_.get()) return op;
  return MakeCast(a, GetScalarDtype(op));
}

// ============================================================================
// CanonicalSimplifier — public interface delegation to Impl
// ============================================================================

CanonicalSimplifier::CanonicalSimplifier() : impl_(std::make_unique<Impl>(nullptr)) {}

CanonicalSimplifier::CanonicalSimplifier(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

CanonicalSimplifier::~CanonicalSimplifier() = default;

CanonicalSimplifier::CanonicalSimplifier(CanonicalSimplifier&&) noexcept = default;
CanonicalSimplifier& CanonicalSimplifier::operator=(CanonicalSimplifier&&) noexcept = default;

ExprPtr CanonicalSimplifier::operator()(const ExprPtr& expr) const {
  // Clear per-call cache to prevent stale entries and unbounded memory growth.
  impl_->ClearSumCache();
  return impl_->VisitExpr(expr);
}

void CanonicalSimplifier::Update(const VarPtr& var, const ExprPtr& new_expr) { impl_->Update(var, new_expr); }

std::function<void()> CanonicalSimplifier::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
