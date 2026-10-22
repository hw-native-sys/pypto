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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace arith {

using Bound = ConstIntBound;

// ============================================================================
// Overflow-safe arithmetic for bound propagation
// ============================================================================

/// One interval endpoint, plus whether it is the exact mathematical result.
///
/// `exact == false` means the true value lies outside the int64 domain and
/// `value` has been saturated. A saturated endpoint must never be reported as a
/// bound on its own: the hardware wraps rather than saturating, so the only
/// sound answer is the whole range the node's dtype can hold. See
/// NarrowToNodeWidth, which every arithmetic visitor routes its result through.
struct Endpoint {
  int64_t value;
  bool exact;
};

/// Which extreme an inexact endpoint saturates to.
///
/// The direction is the caller's and is never derived from the operands: a
/// lower endpoint must move down and an upper endpoint must move up, whatever
/// the sign of the overflow. The previous InfAwareNeg returned kPosInf for
/// both, which is how `x == INT64_MIN` bounded `x - INT64_MIN` as
/// [+inf, +inf] -- an interval excluding 0 -- and folded a true predicate to
/// False.
enum class Side { kLower, kUpper };

static constexpr int64_t SaturatedTo(Side side) {
  return side == Side::kLower ? Bound::kNegInf : Bound::kPosInf;
}

static bool IsInfinite(int64_t v) { return v == Bound::kPosInf || v == Bound::kNegInf; }

/// Resolve a *lower*-endpoint sentinel to the extreme it denotes.
///
/// kNegInf is stored as -INT64_MAX (= INT64_MIN + 1) so that negating it cannot
/// overflow, but as a lower bound it means "the interval reaches INT64_MIN".
/// Reading the stored value is one step off the true extreme: `kNegInf + c`
/// lands one above the reachable `INT64_MIN + c`, and `0 - kNegInf` misses the
/// wrap `0 - INT64_MIN == INT64_MIN`. As an *upper* bound the sentinel is the
/// exact stored value INT64_MIN + 1 -- a scalar pinned to it -- so callers apply
/// this only to lower-endpoint operands. kPosInf needs no such step: it is
/// already INT64_MAX, the true maximum.
static int64_t DemoteNegInf(int64_t v) {
  return v == Bound::kNegInf ? std::numeric_limits<int64_t>::min() : v;
}

// Exact checked endpoint arithmetic over the int64 domain.
//
// The only question asked is whether the exact result leaves int64 -- in which
// case the emitted instruction wraps, the endpoint is reported inexact, and the
// caller widens to the node's dtype range. kNegInf is the one exception: it is
// stored as -INT64_MAX (= INT64_MIN + 1) so negating it cannot overflow, but as
// a lower endpoint it means "reaches INT64_MIN", so Add/Subtract demote that
// lower endpoint to the true extreme first (DemoteNegInf) instead of reading the
// stored value.
//
// Earlier revisions tried to decide this per operator instead, exempting a sum,
// or a product with one finite operand, on the grounds that overflow needed a
// value near the domain edge. Both were wrong, and in the same way: the
// exemption was written in terms of the operator while the counterexamples live
// in the magnitudes. `x + z` wraps once both reach 2^62, and `x * c` wraps at
// x == 3 for a large enough c. Checking the arithmetic answers all of them
// without a rule to tune, and keeps the cases where no wrap is possible --
// `12 - x` for x in [0, +inf] cannot overflow, which is what lets
// `LowerCompositeOps` prove a ring collective's segment write in range.
//
// Identities are kept explicitly because they compute nothing and must survive
// even an unbounded operand: adding or subtracting zero, multiplying by zero
// or +-1.
static Endpoint SatAdd(int64_t a, int64_t b, Side side) {
  // Adding zero keeps an unbounded endpoint unbounded, where the checked path
  // below would report `kPosInf + 0` as an ordinary sum. `x + 0` / `x - 0`
  // appear in rewritten index expressions, and `max(n, 0)` folds only while
  // `n - 0` keeps `n`'s own bound.
  if (b == 0) return {a, true};
  if (a == 0) return {b, true};
  // Only a lower endpoint is "unbounded below"; an upper endpoint equal to the
  // sentinel is the exact stored value INT64_MIN + 1 and must not be shifted.
  if (side == Side::kLower) {
    a = DemoteNegInf(a);
    b = DemoteNegInf(b);
  }
  if (AddWouldOverflow(a, b)) return {SaturatedTo(side), false};
  return {a + b, true};
}

static Endpoint SatNeg(int64_t a, Side side) {
  // Unbounded *above* negates exactly. The interval stops at INT64_MAX, whose
  // negation is kNegInf, so the mirrored interval stops there too; INT64_MIN is
  // never produced, because the only value that would produce it is -INT64_MIN,
  // which is not an int64 and so was never in the interval.
  if (a == Bound::kPosInf) return {Bound::kNegInf, true};
  // Unbounded *below* does not. kNegInf means "no lower bound", so the interval
  // contains INT64_MIN, and `arith.subi 0, x : i64` wraps INT64_MIN back onto
  // itself. Mirroring the endpoints would hand back a non-negative lower bound
  // for an interval whose negation is still negative at that one input -- which
  // is exactly how `-x` under `x <= 0` folded `-x < 0` to False. Report it
  // inexact so the caller widens to the node's dtype range. Abs and
  // multiplication by -1 reach the same wrap through this function.
  //
  // A bound pinned to exactly -INT64_MAX is indistinguishable from the sentinel
  // and widens here too. That costs precision only in a corner no real index
  // reaches, and the alternative -- trusting the sentinel -- is unsound.
  if (a == Bound::kNegInf) return {SaturatedTo(side), false};
  if (NegWouldOverflow(a)) return {SaturatedTo(side), false};
  return {-a, true};
}

static Endpoint SatSub(int64_t a, int64_t b, Side side) {
  // Subtract directly rather than adding a negated endpoint. Routing through a
  // negation destroys INT64_MIN before the subtraction can cancel it:
  // `INT64_MIN - INT64_MIN` is 0, but -INT64_MIN alone is unrepresentable.
  if (b == 0) return {a, true};  // identity; see SatAdd
  // `a` is a lower endpoint only when computing a lower bound; `b` is a lower
  // endpoint only when computing an upper bound (a - b's maximum is reached at
  // b's minimum). Demote each operand only in its lower-endpoint role.
  if (side == Side::kLower) {
    a = DemoteNegInf(a);
  } else {
    b = DemoteNegInf(b);
  }
  if (SubWouldOverflow(a, b)) return {SaturatedTo(side), false};
  return {a - b, true};
}

static Endpoint SatMul(int64_t a, int64_t b, Side side) {
  // Zero annihilates even an unbounded operand: `x * 0` is 0 for every x, and
  // no multiply can wrap away from it.
  if (a == 0 || b == 0) return {0, true};
  // Multiplying by 1 is the identity and cannot wrap. By -1 it is a negation,
  // which can: SatNeg decides, and widens for an unbounded-below endpoint.
  if (a == 1) return {b, true};
  if (b == 1) return {a, true};
  if (a == -1) return SatNeg(b, side);
  if (b == -1) return SatNeg(a, side);
  // Everything else is decided by the checked multiply. Where a product of a
  // genuinely bounded quantity has to stay exact, the bound is tightened at its
  // source rather than exempted here: BoundFromOpSemantics caps the hardware and
  // distributed identity queries, which is what keeps a ragged shard extent
  // `min(max(rows - aiv_id * 8, 0), 8)` folding. A user INDEX parameter has no
  // such cap and must not be given one (issue #2500), so a launch count derived
  // from one legitimately stops folding.
  if (MulWouldOverflow(a, b)) return {SaturatedTo(side), false};
  return {a * b, true};
}

/// Overflow-safe exponentiation by squaring — O(log e).
///
/// Stops at the first inexact multiply: once the running value has left the
/// int64 domain, no later step can bring it back to a value the node holds.
static Endpoint SatPow(int64_t base, int64_t exp, Side side) {
  if (exp == 0) return {1, true};
  if (base == 0) return {0, true};
  if (base == 1) return {1, true};
  int64_t result = 1;
  int64_t b = base;
  int64_t e = exp;
  while (e > 0) {
    if (e & 1) {
      auto next = SatMul(result, b, side);
      if (!next.exact) return next;
      result = next.value;
    }
    e >>= 1;
    if (e > 0) {
      auto squared = SatMul(b, b, side);
      if (!squared.exact) return squared;
      b = squared.value;
    }
  }
  return {result, true};
}

// ============================================================================
// Compound bound helpers
// ============================================================================

/// Return the "everything is possible" bound.
static Bound Everything() { return {Bound::kNegInf, Bound::kPosInf}; }

/// The int64 range `dtype` can hold, when it has one.
///
/// nullopt for anything whose run-time value is not a bounded integer
/// expressible in int64: floats, BOOL, unknown dtype codes, and UINT64, whose
/// 2^64-1 ceiling does not fit. INDEX needs no special case -- its code sits in
/// the signed-integer range, so IsInt() holds and GetBit() is 64, which lands
/// on the whole [kNegInf, kPosInf] domain and makes the width check a no-op for
/// every index expression.
static std::optional<Bound> IntDtypeRange(DataType dtype) {
  if (!dtype.IsInt()) return std::nullopt;
  const size_t bits = dtype.GetBit();
  if (bits == 0 || bits > 64) return std::nullopt;
  if (dtype.IsUnsignedInt()) {
    if (bits >= 64) return std::nullopt;  // 2^64-1 is not representable in int64
    return Bound{0, (static_cast<int64_t>(1) << bits) - 1};
  }
  if (bits >= 64) return Bound{Bound::kNegInf, Bound::kPosInf};
  return Bound{-(static_cast<int64_t>(1) << (bits - 1)), (static_cast<int64_t>(1) << (bits - 1)) - 1};
}

/// Reconcile an exact mathematical interval with the width the node is
/// evaluated at.
///
/// Codegen emits one `arith.*i` at the node's own dtype and coerces the
/// operands into it, so the operation wraps at that width. For `i` in
/// [120, 127], `cast(i, INT8) + const(1, INT8)` is `arith.addi : i8`, which
/// yields 121..127 and -128 -- not the mathematical [121, 128]. A result the
/// dtype cannot hold says nothing about *which* value lands, so the dtype's
/// whole range is all that is left to claim. An inexact endpoint is the same
/// escape one level up: it left the int64 domain, so it left every dtype range
/// too.
///
/// An interval that fits is returned untouched, which is why INDEX and INT64
/// bounds -- every loop-derived index in practice -- keep exactly the precision
/// they had: for those dtypes the range is the entire domain and this is a
/// no-op.
static Bound NarrowToNodeWidth(const ExprPtr& node, Endpoint lo, Endpoint hi) {
  const std::optional<Bound> range = IntDtypeRange(GetScalarDtype(node));
  if (!range) return (lo.exact && hi.exact) ? Bound{lo.value, hi.value} : Everything();
  if (!lo.exact || !hi.exact) return *range;
  if (lo.value < range->min_value || hi.value > range->max_value) return *range;
  return {lo.value, hi.value};
}

/// A finite interval is usable only if lowering preserves both its endpoints.
/// Unknown intervals carry no sign/range facts to invalidate. Reject empty
/// intervals as well instead of deriving signs from inconsistent endpoints.
static bool BoundPreservesDtype(const Bound& bound, const ExprPtr& operand, DataType dtype) {
  if (bound.min_value > bound.max_value) return false;
  if (bound.is_everything()) return true;
  const DataType operand_dtype = GetScalarDtype(operand);
  return detail::ConstIntFitsDtype(bound.min_value, operand_dtype) &&
         detail::ConstIntFitsDtype(bound.max_value, operand_dtype) &&
         detail::ConstIntFitsDtype(bound.min_value, dtype) &&
         detail::ConstIntFitsDtype(bound.max_value, dtype);
}

/// Four-corner multiplication of two bounds.
///
/// Returns an inexact pair as soon as any corner overflows: the hull is only
/// meaningful when every corner is a real product, since a saturated corner
/// could be the one that decides min or max. `x in [2, 3]` times
/// 4611686018427387903 used to report [INT64_MAX-1, INT64_MAX] -- "definitely
/// positive" -- where the emitted `arith.muli : i64` wraps negative at x == 3.
static std::pair<Endpoint, Endpoint> BoundMulEndpoints(const Bound& a, const Bound& b) {
  const Endpoint corners[4] = {
      SatMul(a.min_value, b.min_value, Side::kLower), SatMul(a.min_value, b.max_value, Side::kLower),
      SatMul(a.max_value, b.min_value, Side::kLower), SatMul(a.max_value, b.max_value, Side::kLower)};
  int64_t lo = corners[0].value;
  int64_t hi = corners[0].value;
  bool exact = true;
  for (const auto& corner : corners) {
    exact = exact && corner.exact;
    lo = std::min(lo, corner.value);
    hi = std::max(hi, corner.value);
  }
  return {Endpoint{lo, exact}, Endpoint{hi, exact}};
}

/// Floor-division bounds (positive divisor).
static Bound BoundFloorDivPositiveRHS(const Bound& a, const Bound& b) {
  // b is guaranteed > 0
  // floordiv(a, b) is non-decreasing in a.
  // For a >= 0: non-increasing in b.  For a < 0: non-decreasing in b.
  //
  // A lower endpoint equal to kNegInf means "reaches INT64_MIN", so
  // floor-dividing the stored value INT64_MIN + 1 records a minimum one above
  // the reachable floordiv(INT64_MIN, b) -- the same off-by-one SatAdd/SatSub
  // demote, which for `x // 7` folded `x // 7 == -1317624576693539402` to False
  // although `x == INT64_MIN` produces it. Demote the lower endpoint first.
  const int64_t a_min = DemoteNegInf(a.min_value);
  int64_t min_val = (a_min >= 0) ? floordiv(a_min, b.max_value) : floordiv(a_min, b.min_value);
  int64_t max_val =
      (a.max_value >= 0) ? floordiv(a.max_value, b.min_value) : floordiv(a.max_value, b.max_value);
  return {min_val, max_val};
}

/// Floor-division bounds (general).
static Bound BoundFloorDiv(const Bound& a, const Bound& b) {
  if (b.min_value > 0) {
    return BoundFloorDivPositiveRHS(a, b);
  }
  if (b.max_value < 0) {
    // Negative divisor: floordiv(a, -|b|) = floordiv(-a, |b|).
    //
    // Negating an endpoint can leave the domain (-INT64_MIN does), and the
    // negated operands feed floordiv(), which rejects the INT64_MIN / -1 pair
    // outright. Bail to "unknown" rather than divide a saturated endpoint.
    const Endpoint neg_a_lo = SatNeg(a.max_value, Side::kLower);
    const Endpoint neg_a_hi = SatNeg(a.min_value, Side::kUpper);
    const Endpoint pos_b_lo = SatNeg(b.max_value, Side::kLower);
    const Endpoint pos_b_hi = SatNeg(b.min_value, Side::kUpper);
    if (!neg_a_lo.exact || !neg_a_hi.exact || !pos_b_lo.exact || !pos_b_hi.exact) {
      return Everything();
    }
    Bound neg_a = {neg_a_lo.value, neg_a_hi.value};
    Bound pos_b = {pos_b_lo.value, pos_b_hi.value};
    return BoundFloorDivPositiveRHS(neg_a, pos_b);
  }
  // Divisor range includes zero — cannot bound.
  return Everything();
}

/// Floor-mod bounds (positive divisor).
static Bound BoundFloorMod(const Bound& a, const Bound& b) {
  if (b.min_value > 0) {
    if (a.is_non_negative()) {
      // 0 <= a % b <= min(a_max, b_max - 1)
      return {0, std::min(a.max_value, b.max_value - 1)};
    }
    // General: result in [0, b_max - 1] for floor mod with positive divisor
    // when a could be negative.
    // floormod always returns value in [0, b) for b > 0.
    return {0, b.max_value - 1};
  }
  return Everything();
}

// ============================================================================
// Implementation class — extends ExprFunctor<ConstIntBound>
// ============================================================================

class ConstIntBoundAnalyzer::Impl : public ExprFunctor<Bound> {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  void Bind(const VarPtr& var, const Bound& bound) { var_map_[var.get()] = bound; }

  void Unbind(const VarPtr& var) { var_map_.erase(var.get()); }

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

  Bound VisitExpr(const ExprPtr& expr) override { return ExprFunctor<Bound>::VisitExpr(expr); }

 protected:
  // --- Leaf nodes ---

  Bound VisitExpr_(const ConstIntPtr& op) override {
    // Stored literals need not fit their dtype. Their raw value must not leak
    // a false sign through enclosing arithmetic or a widening Cast.
    if (!detail::ConstIntFitsDtype(op->value_, op->dtype())) return Everything();
    // A literal equal to a sentinel cannot be told apart from an unbounded
    // operand once it is stored in a Bound, and every consumer would read it as
    // an infinity: `x + INT64_MAX` for `x` in [-5, -1] became [+inf, +inf],
    // an interval excluding 0 that "proved" the sum non-zero. Refuse to bound
    // it instead. INT64_MIN is deliberately not a sentinel (see analyzer.h), so
    // `x - INT64_MIN` keeps cancelling exactly.
    if (IsInfinite(op->value_)) return Everything();
    return {op->value_, op->value_};
  }

  Bound VisitExpr_(const ConstFloatPtr& op) override {
    double v = op->value_;
    if (std::isnan(v) || std::isinf(v)) return Everything();
    // Guard against finite values outside int64_t range (cast would be UB)
    constexpr auto kMax = static_cast<double>(Bound::kPosInf);
    constexpr auto kMin = static_cast<double>(Bound::kNegInf);
    double lo = std::floor(v);
    double hi = std::ceil(v);
    int64_t lo_i = (lo < kMin) ? Bound::kNegInf : static_cast<int64_t>(lo);
    int64_t hi_i = (hi > kMax) ? Bound::kPosInf : static_cast<int64_t>(hi);
    return {lo_i, hi_i};
  }

  Bound VisitExpr_(const ConstBoolPtr& op) override {
    int64_t v = op->value_ ? 1 : 0;
    return {v, v};
  }

  Bound VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return DefaultBound();
  }

  Bound VisitExpr_(const IterArgPtr& op) override {
    // IterArg is a Var subclass — look up in var_map_
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) return it->second;
    return DefaultBound();
  }

  Bound VisitExpr_(const MemRefPtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const WindowBufferPtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const CallPtr& op) override { return BoundFromOpSemantics(op->op_); }
  Bound VisitExpr_(const SubmitPtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const MakeTuplePtr& /*op*/) override { return Everything(); }
  Bound VisitExpr_(const TupleGetItemExprPtr& /*op*/) override { return Everything(); }

  // --- Binary arithmetic ---

  Bound VisitExpr_(const AddPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return NarrowToNodeWidth(op, SatAdd(a.min_value, b.min_value, Side::kLower),
                             SatAdd(a.max_value, b.max_value, Side::kUpper));
  }

  Bound VisitExpr_(const SubPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // Subtract directly instead of adding a negated endpoint. Routing through a
    // negation destroys INT64_MIN before the subtraction can cancel it:
    // `INT64_MIN - INT64_MIN` is 0, but -INT64_MIN alone is unrepresentable and
    // used to saturate to +inf, making `x == INT64_MIN` provably false.
    return NarrowToNodeWidth(op, SatSub(a.min_value, b.max_value, Side::kLower),
                             SatSub(a.max_value, b.min_value, Side::kUpper));
  }

  Bound VisitExpr_(const MulPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    auto [lo, hi] = BoundMulEndpoints(a, b);
    return NarrowToNodeWidth(op, lo, hi);
  }

  Bound VisitExpr_(const FloorDivPtr& op) override {
    const DataType dtype = GetScalarDtype(op);
    if (!detail::FloorDivModOperandsFitDtype(op->left_, op->right_, dtype)) return Everything();
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // A wide operand can wrap even when it is not a literal. Bounds computed
    // before that conversion cannot establish the result's sign.
    if (!BoundPreservesDtype(a, op->left_, dtype) || !BoundPreservesDtype(b, op->right_, dtype)) {
      return Everything();
    }
    return BoundFloorDiv(a, b);
  }

  Bound VisitExpr_(const FloorModPtr& op) override {
    const DataType dtype = GetScalarDtype(op);
    if (!detail::FloorDivModOperandsFitDtype(op->left_, op->right_, dtype)) return Everything();
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // Enclosing min/max/comparison rewrites also consume these bounds, so the
    // conservative literal policy must apply beyond direct constant folding.
    if (!BoundPreservesDtype(a, op->left_, dtype) || !BoundPreservesDtype(b, op->right_, dtype)) {
      return Everything();
    }
    return BoundFloorMod(a, b);
  }

  Bound VisitExpr_(const FloatDivPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const MinPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {std::min(a.min_value, b.min_value), std::min(a.max_value, b.max_value)};
  }

  Bound VisitExpr_(const MaxPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    return {std::max(a.min_value, b.min_value), std::max(a.max_value, b.max_value)};
  }

  Bound VisitExpr_(const PowPtr& op) override {
    auto base = VisitExpr(op->left_);
    auto exp = VisitExpr(op->right_);
    // Only handle constant non-negative exponent
    if (!exp.is_const() || exp.min_value < 0) return Everything();
    int64_t e = exp.min_value;
    if (e == 0) return {1, 1};
    if (e == 1) return base;
    if (base.is_non_negative()) {
      // O(log e) exponentiation by squaring for bound propagation
      return NarrowToNodeWidth(op, SatPow(base.min_value, e, Side::kLower),
                               SatPow(base.max_value, e, Side::kUpper));
    }
    return Everything();
  }

  // --- Comparisons (always boolean result) ---

  Bound VisitExpr_(const EqPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const NePtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const LtPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const LePtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const GtPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const GePtr& /*op*/) override { return {0, 1}; }

  // --- Logical (boolean result) ---

  Bound VisitExpr_(const AndPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const OrPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const XorPtr& /*op*/) override { return {0, 1}; }

  // --- Bitwise (conservative) ---

  Bound VisitExpr_(const BitAndPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    // If both non-negative, result is in [0, min(a_max, b_max)]
    if (a.is_non_negative() && b.is_non_negative()) {
      return {0, std::min(a.max_value, b.max_value)};
    }
    return Everything();
  }

  Bound VisitExpr_(const BitOrPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative()) {
      // Upper bound: next power-of-2 above max of both, minus 1
      // Conservative: use sum as loose upper bound
      return NarrowToNodeWidth(op, Endpoint{std::max(a.min_value, b.min_value), true},
                               SatAdd(a.max_value, b.max_value, Side::kUpper));
    }
    return Everything();
  }

  Bound VisitExpr_(const BitXorPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const BitShiftLeftPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative() && b.max_value < 63) {
      return NarrowToNodeWidth(op, SatMul(a.min_value, static_cast<int64_t>(1) << b.min_value, Side::kLower),
                               SatMul(a.max_value, static_cast<int64_t>(1) << b.max_value, Side::kUpper));
    }
    return Everything();
  }

  Bound VisitExpr_(const BitShiftRightPtr& op) override {
    auto a = VisitExpr(op->left_);
    auto b = VisitExpr(op->right_);
    if (a.is_non_negative() && b.is_non_negative() && b.max_value < 63) {
      return {a.min_value >> b.max_value, a.max_value >> b.min_value};
    }
    return Everything();
  }

  // --- Unary ---

  Bound VisitExpr_(const NegPtr& op) override {
    auto a = VisitExpr(op->operand_);
    return NarrowToNodeWidth(op, SatNeg(a.max_value, Side::kLower), SatNeg(a.min_value, Side::kUpper));
  }

  Bound VisitExpr_(const AbsPtr& op) override {
    auto a = VisitExpr(op->operand_);
    if (a.min_value >= 0) return a;
    if (a.max_value <= 0) {
      return NarrowToNodeWidth(op, SatNeg(a.max_value, Side::kLower), SatNeg(a.min_value, Side::kUpper));
    }
    // Straddles zero. The upper endpoint is max(-a_min, a_max); a non-negative
    // *lower* endpoint of 0 is only justified once that negation is exact --
    // two's-complement abs(INT64_MIN) is INT64_MIN, i.e. still negative, so
    // claiming [0, ...] there would be a fresh unsoundness.
    const Endpoint neg_min = SatNeg(a.min_value, Side::kUpper);
    if (!neg_min.exact) {
      return NarrowToNodeWidth(op, Endpoint{Bound::kNegInf, false}, Endpoint{Bound::kPosInf, false});
    }
    return NarrowToNodeWidth(op, Endpoint{0, true}, Endpoint{std::max(neg_min.value, a.max_value), true});
  }

  Bound VisitExpr_(const NotPtr& /*op*/) override { return {0, 1}; }
  Bound VisitExpr_(const BitNotPtr& /*op*/) override { return Everything(); }

  Bound VisitExpr_(const CastPtr& op) override {
    auto a = VisitExpr(op->operand_);
    // Narrowing casts wrap instead of clipping. A scalar integer cast truncates
    // (codegen emits `arith.index_cast` / `arith.trunci`), it does not saturate:
    // a source range that fits the target is unchanged, but one that overflows
    // it can land anywhere in the target range -- `[120, 139]` cast to INT8 is
    // 120..127 and -128..-117. Intersecting that down to `[120, 127]` would
    // prove `cast(x, INT8) >= 0` and hide the values the wrap produces.
    //
    // UINT64 is the one target with no int64 range to clamp to, so it widens to
    // unknown rather than passing the source interval through: a negative
    // source bound describes none of the values a UINT64 actually holds.
    const DataType target = GetScalarDtype(op);
    // A signed 64-bit target holds every int64 value, so the cast is the identity on
    // the interval. Do not test it against IntDtypeRange(): that range ends at the
    // kNegInf sentinel (INT64_MIN + 1), so a genuine INT64_MIN lower endpoint would
    // read as out of range and be widened to unknown. Passing it through matches the
    // behavior before the narrowing-cast fix; UINT64 stays below with no int64 range.
    if (target.IsInt() && !target.IsUnsignedInt() && target.GetBit() >= 64) return a;
    const std::optional<Bound> range = IntDtypeRange(target);
    if (!range) return Everything();
    // An inverted interval (min > max) describes no value at all, so widen it
    // here too rather than deriving a sign from inconsistent endpoints.
    if (a.min_value < range->min_value || a.max_value > range->max_value || a.min_value > a.max_value) {
      return *range;
    }
    return a;
  }

 private:
  /// The range a builtin's own semantics establish for its result.
  ///
  /// These are the hardware identity queries: which block / subblock this
  /// instance is, and how many there are. A block index is never negative and a
  /// block count is never zero -- facts about what the op returns, not about
  /// `INDEX`, which is signed and proves nothing (issue #2500). Keeping them
  /// here ties each range to the op that guarantees it, instead of to a dtype
  /// default that would also cover unrelated user scalars.
  ///
  /// Without this, a shard extent such as `min(max(rows - aiv_id * 8, 0), 8)`
  /// no longer folds to `max(rows - aiv_id * 8, 0)`, because an unbounded
  /// `aiv_id` lets `rows - aiv_id * 8` exceed the tile.
  ///
  /// The *upper* bound carries as much weight as the lower one, and for the same
  /// reason. An endpoint left at kPosInf means "unbounded", and arithmetic on an
  /// unbounded operand has to widen to unknown because it may wrap -- which
  /// would take `aiv_id * 8` with it. A launch grid is not unbounded, so saying
  /// so costs the fold for nothing. kMaxLaunchExtent is deliberately far above
  /// any real block or subblock count rather than a per-architecture figure:
  /// this is the IR's arithmetic layer, which has no backend to ask, and a
  /// generous architecture-independent cap is enough to keep the products these
  /// indices appear in exact (2^31 * a tile dimension stays well inside int64).
  /// Kept inside the INT32 range: the distributed queries below return i32, and
  /// a bound wider than the result's own type would describe values it cannot
  /// hold.
  static constexpr int64_t kMaxLaunchExtent = (int64_t{1} << 31) - 1;

  static Bound BoundFromOpSemantics(const OpPtr& op) {
    if (!op) return Everything();
    if (IsOp(op, "tile.get_block_idx") || IsOp(op, "tile.get_subblock_idx") ||
        IsOp(op, "tensor.get_block_idx") || IsOp(op, "tensor.get_subblock_idx")) {
      return {0, kMaxLaunchExtent};
    }
    if (IsOp(op, "tile.get_block_num") || IsOp(op, "tensor.get_block_num")) {
      return {1, kMaxLaunchExtent};
    }
    // A rank and a world size are the distributed form of the same identity
    // query, and carry the same obligation: a ring collective derives its
    // segment offsets from them, and `LowerCompositeOps` has to prove those
    // offsets land inside the valid extent before it will emit the write.
    // Leaving them unbounded does not merely cost a fold there -- it rejects a
    // legal program with "not provably at or before the target valid extent".
    if (IsOp(op, "pld.system.rank")) {
      return {0, kMaxLaunchExtent};
    }
    if (IsOp(op, "pld.system.nranks")) {
      return {1, kMaxLaunchExtent};
    }
    return Everything();
  }

  /// The bound a Var carries when nothing has bound it.
  ///
  /// Always unbounded. No dtype implies a sign: `INDEX` is signed, and a Var of
  /// it -- a derived scalar or a caller-supplied parameter alike -- can be
  /// negative at runtime (issue #2500). A caller that knows better binds the
  /// variable; nothing is assumed on its behalf here.
  static Bound DefaultBound() { return Everything(); }

  Analyzer* parent_;
  std::unordered_map<const Expr*, Bound> var_map_;
};

// ============================================================================
// EnterConstraint — parse comparison expressions to tighten variable bounds
// ============================================================================

std::function<void()> ConstIntBoundAnalyzer::Impl::EnterConstraint(const ExprPtr& constraint) {
  // Per tightening: the variable, the bound it had before, and the bound this
  // constraint applied (see the recovery function below).
  struct Recovery {
    const Expr* var;
    Bound old;
    Bound applied;
  };
  std::vector<Recovery> recovery;

  // A strict comparison is recorded as the equivalent closed bound. Only integers
  // have a predecessor, so `x > c` is `x >= c + 1` for them and merely `x >= c` for
  // everything else: applying the unit step to a float variable records `f >= 2`
  // for `f > 1.0`, which excludes the legal value 1.5 and lets CanProve() fold a
  // nested `f >= 2.0` to always-true. The relaxed bound stays sound — a float
  // variable keeps contributing bounds, just without the strictness.
  auto IsIntegerScalar = [](const VarPtr& var) {
    auto scalar_type = As<ScalarType>(var->GetType());
    return scalar_type && scalar_type->dtype_.IsInt();
  };
  auto StrictBound = [&](const VarPtr& var, int64_t bound, int64_t unit) {
    if (!IsIntegerScalar(var)) return bound;
    // `unit` is +-1, so this can only overflow at the domain's own extremes --
    // `x > kPosInf` and the like, which describe an empty constraint. Keep the
    // un-stepped endpoint there rather than wrapping past it.
    const Endpoint stepped = SatAdd(bound, unit, unit > 0 ? Side::kUpper : Side::kLower);
    return stepped.exact ? stepped.value : bound;
  };

  // Helper: try to tighten bound for a variable.
  auto TryTighten = [&](const VarPtr& var, const Bound& new_bound) {
    const Expr* var_ptr = var.get();
    auto it = var_map_.find(var_ptr);
    Bound old = (it != var_map_.end()) ? it->second : DefaultBound();
    Bound applied{std::max(old.min_value, new_bound.min_value), std::min(old.max_value, new_bound.max_value)};
    // Refuse an endpoint pinned onto the *opposite* sentinel. Callers pass
    // kNegInf/kPosInf as "no constraint on this side", but a constraint can also
    // land on one: `x > INT64_MIN` steps to INT64_MIN + 1, which *is* kNegInf,
    // so `x > INT64_MIN and x < -9223372036854775806` pins x to
    // [kNegInf, kNegInf]. Endpoint arithmetic would then read that as unbounded
    // while the fold reads it as a value, and `x * 2` would prove negative
    // although `arith.muli : i64` wraps positive. A refused constraint only
    // leaves the bound wider, so dropping it is always sound. (The literal
    // itself is already refused by VisitExpr_(ConstIntPtr); this is the same
    // ambiguity arriving through a constraint rather than through a literal.)
    if (applied.max_value == Bound::kNegInf || applied.min_value == Bound::kPosInf) return;
    recovery.push_back({var_ptr, old, applied});
    var_map_[var_ptr] = applied;
  };

  // Try to parse simple comparisons: var >= const, var < const, etc.
  // Must use std::function for recursive calls (And case).
  std::function<void(const ExprPtr&)> TryParseConstraint = [&](const ExprPtr& expr) {
    // Ge: left >= right
    if (auto ge = As<Ge>(expr)) {
      if (auto var = As<Var>(ge->left_)) {
        auto rb = VisitExpr(ge->right_);
        if (rb.is_const()) TryTighten(var, {rb.min_value, Bound::kPosInf});
      } else if (auto var = As<Var>(ge->right_)) {
        auto lb = VisitExpr(ge->left_);
        if (lb.is_const()) TryTighten(var, {Bound::kNegInf, lb.max_value});
      }
      return;
    }
    // Gt: left > right
    if (auto gt = As<Gt>(expr)) {
      if (auto var = As<Var>(gt->left_)) {
        auto rb = VisitExpr(gt->right_);
        if (rb.is_const()) TryTighten(var, {StrictBound(var, rb.min_value, 1), Bound::kPosInf});
      } else if (auto var = As<Var>(gt->right_)) {
        auto lb = VisitExpr(gt->left_);
        if (lb.is_const()) TryTighten(var, {Bound::kNegInf, StrictBound(var, lb.max_value, -1)});
      }
      return;
    }
    // Le: left <= right
    if (auto le = As<Le>(expr)) {
      if (auto var_l = As<Var>(le->left_)) {
        auto rb = VisitExpr(le->right_);
        if (rb.is_const()) TryTighten(var_l, {Bound::kNegInf, rb.max_value});
      } else if (auto var_r = As<Var>(le->right_)) {
        auto lb = VisitExpr(le->left_);
        if (lb.is_const()) TryTighten(var_r, {lb.min_value, Bound::kPosInf});
      }
      return;
    }
    // Lt: left < right
    if (auto lt = As<Lt>(expr)) {
      if (auto var = As<Var>(lt->left_)) {
        auto rb = VisitExpr(lt->right_);
        if (rb.is_const()) TryTighten(var, {Bound::kNegInf, StrictBound(var, rb.max_value, -1)});
      } else if (auto var = As<Var>(lt->right_)) {
        auto lb = VisitExpr(lt->left_);
        if (lb.is_const()) TryTighten(var, {StrictBound(var, lb.min_value, 1), Bound::kPosInf});
      }
      return;
    }
    // Eq: left == right
    if (auto eq = As<Eq>(expr)) {
      if (auto var = As<Var>(eq->left_)) {
        auto rb = VisitExpr(eq->right_);
        if (rb.is_const()) TryTighten(var, rb);
      } else if (auto var = As<Var>(eq->right_)) {
        auto lb = VisitExpr(eq->left_);
        if (lb.is_const()) TryTighten(var, lb);
      }
      return;
    }
    // And: both constraints hold
    if (auto and_op = As<And>(expr)) {
      TryParseConstraint(and_op->left_);
      TryParseConstraint(and_op->right_);
    }
    // Note: Not(comparison) is handled by ConstraintContext, which normalizes
    // constraints via rewrite_simplify before dispatching to sub-analyzers.
    // E.g., Not(Lt(a,b)) is rewritten to Ge(a,b) before reaching this code.
  };

  TryParseConstraint(constraint);

  // Return recovery function — iterate in reverse so that when the same variable
  // was tightened multiple times (e.g., And(x >= 0, x < 8)), we restore the
  // original pre-scope bound, not the intermediate one.
  //
  // A bound is restored only while the entry still holds what this constraint
  // applied. If the scope rebound or unbound the variable -- a pre-SSA
  // reassignment inside `if i < 4:` that the caller reset and then erased on
  // leaving an inner region -- the old bound describes a value the variable no
  // longer has; restoring it would bring back e.g. a loop header's `[0, 8)`
  // after `i = i + 100`. The entry is left as it is instead. Restoring on a
  // coincidental match is still sound: `applied` is a subset of `old`.
  return [this, recovery = std::move(recovery)]() {
    for (auto it = recovery.rbegin(); it != recovery.rend(); ++it) {
      auto cur = var_map_.find(it->var);
      if (cur == var_map_.end() || cur->second.min_value != it->applied.min_value ||
          cur->second.max_value != it->applied.max_value) {
        continue;
      }
      if (it->old.is_everything()) {
        var_map_.erase(cur);
      } else {
        cur->second = it->old;
      }
    }
  };
}

// ============================================================================
// ConstIntBoundAnalyzer — public interface delegation to Impl
// ============================================================================

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer() : impl_(std::make_unique<Impl>(nullptr)) {}

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(Analyzer* parent) : impl_(std::make_unique<Impl>(parent)) {}

ConstIntBoundAnalyzer::~ConstIntBoundAnalyzer() = default;

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(ConstIntBoundAnalyzer&&) noexcept = default;
ConstIntBoundAnalyzer& ConstIntBoundAnalyzer::operator=(ConstIntBoundAnalyzer&&) noexcept = default;

Bound ConstIntBoundAnalyzer::operator()(const ExprPtr& expr) const { return impl_->VisitExpr(expr); }

void ConstIntBoundAnalyzer::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive) {
  CHECK(max_val_exclusive > min_val) << "Bind requires max_val_exclusive > min_val, got [" << min_val << ", "
                                     << max_val_exclusive << ")";
  impl_->Bind(var, {min_val, max_val_exclusive - 1});
}

void ConstIntBoundAnalyzer::Update(const VarPtr& var, const Bound& bound) {
  CHECK(bound.min_value <= bound.max_value)
      << "Update requires min_value <= max_value, got [" << bound.min_value << ", " << bound.max_value << "]";
  impl_->Bind(var, bound);
}

void ConstIntBoundAnalyzer::Unbind(const VarPtr& var) { impl_->Unbind(var); }

std::function<void()> ConstIntBoundAnalyzer::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
