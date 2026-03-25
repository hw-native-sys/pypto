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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "src/ir/arith/constraint_extract.h"

namespace pypto {
namespace ir {
namespace arith {

// Bring IR utilities from enclosing namespace into scope.
using ir::As;
using ir::GetScalarDtype;
using ir::MakeEq;
using ir::MakeGe;
using ir::MakeLt;

// ============================================================================
// Utility functions
// ============================================================================

namespace {

/// Reverse a comparison result (swap LHS and RHS).
CompareResult Reverse(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kEQ:
      return CompareResult::kEQ;
    case CompareResult::kLT:
      return CompareResult::kGT;
    case CompareResult::kLE:
      return CompareResult::kGE;
    case CompareResult::kGT:
      return CompareResult::kLT;
    case CompareResult::kGE:
      return CompareResult::kLE;
    case CompareResult::kNE:
      return CompareResult::kNE;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
  }
  INTERNAL_CHECK(false) << "Internal error: invalid CompareResult " << static_cast<int>(res);
  return CompareResult::kUnknown;  // unreachable
}

/// Extract a constant additive offset from an expression.
/// Returns (inner_expr, offset) such that expr == inner_expr + offset.
/// For pure constants, inner_expr is nullptr.
std::pair<ExprPtr, int64_t> ExtractOffset(const ExprPtr& expr) {
  if (auto add = As<Add>(expr)) {
    if (auto ci = As<ConstInt>(add->right_)) return {add->left_, ci->value_};
    if (auto ci = As<ConstInt>(add->left_)) return {add->right_, ci->value_};
  }
  if (auto sub = As<Sub>(expr)) {
    if (auto ci = As<ConstInt>(sub->right_)) return {sub->left_, -ci->value_};
  }
  if (auto ci = As<ConstInt>(expr)) return {nullptr, ci->value_};
  return {expr, 0};
}

/// Extract constant offsets from both sides of a comparison.
/// Returns (lhs_inner, rhs_inner, offset) such that
/// (lhs OP rhs) <==> (lhs_inner OP rhs_inner + offset).
std::tuple<ExprPtr, ExprPtr, int64_t> ExtractOffsets(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto [lhs_inner, lhs_off] = ExtractOffset(lhs);
  auto [rhs_inner, rhs_off] = ExtractOffset(rhs);
  return {lhs_inner, rhs_inner, rhs_off - lhs_off};
}

}  // namespace

// ============================================================================
// TransitiveComparisonAnalyzer::Impl
// ============================================================================

class TransitiveComparisonAnalyzer::Impl {
 public:
  explicit Impl(Analyzer* /*parent*/) {}

  [[nodiscard]] CompareResult TryCompare(const ExprPtr& lhs, const ExprPtr& rhs,
                                         bool propagate_inequalities) const;

  void Bind(const VarPtr& var, const ExprPtr& expr, bool allow_override);
  void Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive, bool allow_override);
  void Unbind(const VarPtr& var);

  std::function<void()> EnterConstraint(const ExprPtr& constraint);

 private:
  /// Type-safe key for expression identity.
  enum class Key : size_t {};

  struct KeyHash {
    size_t operator()(Key k) const { return std::hash<size_t>{}(static_cast<size_t>(k)); }
  };

  /// Internal representation of a comparison: lhs OP rhs + offset.
  struct Comparison {
    Comparison(Key lhs, Key rhs, int64_t offset, CompareResult result);

    [[nodiscard]] bool IsNormalized() const;
    [[nodiscard]] std::optional<Comparison> WithLHS(Key new_lhs) const;
    [[nodiscard]] bool Implies(const Comparison& other) const;

    Key lhs_;
    Key rhs_;
    int64_t offset_{0};
    CompareResult result_{CompareResult::kInconsistent};
  };

  /// Get or create a Key for an expression.
  Key ExprToKey(const ExprPtr& expr);

  /// Look up an existing Key (returns nullopt if not seen before).
  [[nodiscard]] std::optional<Key> ExprToPreviousKey(const ExprPtr& expr) const;

  /// Parse a comparison expression into internal representation.
  std::optional<Comparison> FromExpr(const ExprPtr& expr);

  /// Add known comparisons from a constraint expression.
  void AddKnown(const ExprPtr& expr, std::vector<Comparison>* vec);

  /// Collect direct (non-transitive) comparisons between lhs and rhs.
  [[nodiscard]] std::vector<Comparison> CollectDirectComparisons(Key lhs_key, Key rhs_key) const;

  /// Collect all comparisons including transitive ones.
  [[nodiscard]] std::vector<Comparison> CollectIndirectComparisons(Key lhs_key, Key rhs_key) const;

  /// DFS from lhs_key to find all reachable comparisons to rhs_key.
  [[nodiscard]] std::vector<Comparison> DFSFromLHS(Key lhs_key, Key rhs_key) const;

  /// Merge a set of comparisons sharing the same LHS/RHS into a single result.
  [[nodiscard]] static CompareResult MergeComparisons(const std::vector<Comparison>& lhs_to_rhs,
                                                      int64_t offset);

  /// Get a canonical "zero" expression for constant offset handling.
  ExprPtr GetZeroExpr() const;

  /// Map from expression pointer to Key.
  std::unordered_map<const Expr*, Key> expr_to_key_;

  /// Owned expression pointers to ensure keys in expr_to_key_ remain valid.
  std::vector<ExprPtr> key_exprs_;

  /// Known comparisons from Bind calls (always true by definition).
  std::vector<Comparison> knowns_;

  /// Scoped comparisons from EnterConstraint (true only within scope).
  std::vector<Comparison> scoped_knowns_;

  /// Shared zero expression for constant comparisons (lazily initialized, even in const methods).
  mutable ExprPtr zero_expr_;
};

// ============================================================================
// Comparison
// ============================================================================

TransitiveComparisonAnalyzer::Impl::Comparison::Comparison(Key lhs, Key rhs, int64_t offset,
                                                           CompareResult result)
    : lhs_(lhs), rhs_(rhs), offset_(offset), result_(result) {
  // Normalize LT/GT to LE/GE with adjusted offset.
  // For integers: (i < j + c) <==> (i <= j + c - 1).
  if (result_ == CompareResult::kLT) {
    result_ = CompareResult::kLE;
    offset_ -= 1;
  }
  if (result_ == CompareResult::kGT) {
    result_ = CompareResult::kGE;
    offset_ += 1;
  }
}

bool TransitiveComparisonAnalyzer::Impl::Comparison::IsNormalized() const {
  return result_ != CompareResult::kLT && result_ != CompareResult::kGT;
}

std::optional<TransitiveComparisonAnalyzer::Impl::Comparison>
TransitiveComparisonAnalyzer::Impl::Comparison::WithLHS(Key new_lhs) const {
  if (new_lhs == lhs_) {
    return *this;
  }
  if (new_lhs == rhs_) {
    return Comparison(rhs_, lhs_, -offset_, Reverse(result_));
  }
  return std::nullopt;
}

bool TransitiveComparisonAnalyzer::Impl::Comparison::Implies(const Comparison& other) const {
  INTERNAL_CHECK(lhs_ == other.lhs_) << "Internal error: Implies requires same LHS";
  INTERNAL_CHECK(rhs_ == other.rhs_) << "Internal error: Implies requires same RHS";
  INTERNAL_CHECK(IsNormalized()) << "Internal error: Implies requires normalized comparison";
  INTERNAL_CHECK(other.IsNormalized()) << "Internal error: Implies requires normalized comparison";

  // Same relation and offset: trivially implied.
  if (result_ == other.result_ && offset_ == other.offset_) {
    return true;
  }

  // x <= y + c1 implies x <= y + c2 if c1 <= c2
  // x == y + c1 implies x <= y + c2 if c1 <= c2
  if (other.result_ == CompareResult::kLE && offset_ <= other.offset_) {
    if (result_ == CompareResult::kEQ || result_ == CompareResult::kLE) {
      return true;
    }
  }

  // x >= y + c1 implies x >= y + c2 if c1 >= c2
  // x == y + c1 implies x >= y + c2 if c1 >= c2
  if (other.result_ == CompareResult::kGE && offset_ >= other.offset_) {
    if (result_ == CompareResult::kEQ || result_ == CompareResult::kGE) {
      return true;
    }
  }

  if (other.result_ == CompareResult::kNE) {
    // x == y + c1 implies x != y + c2 if c1 != c2
    if (result_ == CompareResult::kEQ && offset_ != other.offset_) {
      return true;
    }
    // x <= y + c1 implies x != y + c2 if c1 < c2
    if (result_ == CompareResult::kLE && offset_ < other.offset_) {
      return true;
    }
    // x >= y + c1 implies x != y + c2 if c1 > c2
    if (result_ == CompareResult::kGE && offset_ > other.offset_) {
      return true;
    }
  }

  return false;
}

// ============================================================================
// Key management
// ============================================================================

TransitiveComparisonAnalyzer::Impl::Key TransitiveComparisonAnalyzer::Impl::ExprToKey(const ExprPtr& expr) {
  const Expr* raw = expr.get();
  auto it = expr_to_key_.find(raw);
  if (it != expr_to_key_.end()) {
    return it->second;
  }
  Key new_key = Key(expr_to_key_.size());
  expr_to_key_[raw] = new_key;
  key_exprs_.push_back(expr);  // Keep ExprPtr alive for the lifetime of the analyzer.
  return new_key;
}

std::optional<TransitiveComparisonAnalyzer::Impl::Key> TransitiveComparisonAnalyzer::Impl::ExprToPreviousKey(
    const ExprPtr& expr) const {
  auto it = expr_to_key_.find(expr.get());
  if (it != expr_to_key_.end()) {
    return it->second;
  }
  return std::nullopt;
}

ExprPtr TransitiveComparisonAnalyzer::Impl::GetZeroExpr() const {
  if (!zero_expr_) {
    zero_expr_ = MakeConstInt(0, DataType::INT64);
  }
  return zero_expr_;
}

// ============================================================================
// Expression parsing
// ============================================================================

std::optional<TransitiveComparisonAnalyzer::Impl::Comparison> TransitiveComparisonAnalyzer::Impl::FromExpr(
    const ExprPtr& expr) {
  CompareResult res;
  ExprPtr lhs_expr;
  ExprPtr rhs_expr;

  if (auto op = As<Le>(expr)) {
    res = CompareResult::kLE;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else if (auto op = As<Ge>(expr)) {
    res = CompareResult::kGE;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else if (auto op = As<Lt>(expr)) {
    res = CompareResult::kLT;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else if (auto op = As<Gt>(expr)) {
    res = CompareResult::kGT;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else if (auto op = As<Eq>(expr)) {
    res = CompareResult::kEQ;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else if (auto op = As<Ne>(expr)) {
    res = CompareResult::kNE;
    lhs_expr = op->left_;
    rhs_expr = op->right_;
  } else {
    return std::nullopt;
  }

  // Both sides constant — should have been constant-folded already.
  if (As<ConstInt>(lhs_expr) && As<ConstInt>(rhs_expr)) {
    return std::nullopt;
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);

  // Use zero sentinel for pure-constant sides.
  ExprPtr lhs_key_expr = lhs ? lhs : GetZeroExpr();
  ExprPtr rhs_key_expr = rhs ? rhs : GetZeroExpr();

  Key lhs_key = ExprToKey(lhs_key_expr);
  Key rhs_key = ExprToKey(rhs_key_expr);

  return Comparison(lhs_key, rhs_key, offset, res);
}

void TransitiveComparisonAnalyzer::Impl::AddKnown(const ExprPtr& expr, std::vector<Comparison>* vec) {
  for (const auto& subexpr : ExtractConstraints(expr)) {
    if (auto cmp = FromExpr(subexpr)) {
      vec->push_back(cmp.value());
    }
  }
}

// ============================================================================
// Comparison collection
// ============================================================================

std::vector<TransitiveComparisonAnalyzer::Impl::Comparison>
TransitiveComparisonAnalyzer::Impl::CollectDirectComparisons(Key lhs_key, Key rhs_key) const {
  std::vector<Comparison> output;

  auto append_known = [&](const Comparison& cmp) {
    if (auto normalized = cmp.WithLHS(lhs_key)) {
      if (normalized.value().rhs_ == rhs_key) {
        output.push_back(normalized.value());
      }
    }
  };

  for (const auto& known : knowns_) {
    append_known(known);
  }
  for (const auto& known : scoped_knowns_) {
    append_known(known);
  }

  return output;
}

std::vector<TransitiveComparisonAnalyzer::Impl::Comparison>
TransitiveComparisonAnalyzer::Impl::CollectIndirectComparisons(Key lhs_key, Key rhs_key) const {
  // Search from LHS→RHS and RHS→LHS, then normalize.
  auto output = DFSFromLHS(lhs_key, rhs_key);
  for (Comparison cmp : DFSFromLHS(rhs_key, lhs_key)) {
    auto opt_normalized = cmp.WithLHS(lhs_key);
    INTERNAL_CHECK(opt_normalized.has_value()) << "Internal error: DFS result should be normalizable";
    output.push_back(opt_normalized.value());
  }
  return output;
}

std::vector<TransitiveComparisonAnalyzer::Impl::Comparison> TransitiveComparisonAnalyzer::Impl::DFSFromLHS(
    Key lhs_key, Key rhs_key) const {
  std::unordered_set<Key, KeyHash> to_visit;
  std::unordered_map<Key, std::vector<Comparison>, KeyHash> compared_to_lhs;

  // Utility to add a new known comparison, managing redundancy.
  // Re-enqueues cmp.rhs_ for visitation whenever the derived fact set changes.
  auto declare_known = [&](Comparison cmp) {
    std::vector<Comparison>& knowns = compared_to_lhs[cmp.rhs_];

    // Check if existing knowledge already implies this.
    for (const auto& prev_known : knowns) {
      if (prev_known.Implies(cmp)) {
        return;
      }
    }

    // Check if this replaces a weaker existing comparison.
    for (auto& prev_known : knowns) {
      if (cmp.Implies(prev_known)) {
        prev_known = cmp;
        // Strengthened — re-enqueue to recompute outgoing edges.
        if (cmp.rhs_ != rhs_key) {
          to_visit.insert(cmp.rhs_);
        }
        return;
      }
    }

    // Independent knowledge — track separately and enqueue.
    knowns.push_back(cmp);
    if (cmp.rhs_ != rhs_key) {
      to_visit.insert(cmp.rhs_);
    }
  };

  // Seed with direct comparisons involving the LHS.
  for (const auto& known : knowns_) {
    if (auto normalized = known.WithLHS(lhs_key)) {
      declare_known(normalized.value());
    }
  }
  for (const auto& known : scoped_knowns_) {
    if (auto normalized = known.WithLHS(lhs_key)) {
      declare_known(normalized.value());
    }
  }

  // DFS: explore reachable comparisons via transitive chains.
  while (!to_visit.empty()) {
    Key middle_key = *to_visit.begin();
    to_visit.erase(to_visit.begin());

    const std::vector<Comparison>& prev_knowns_using_middle = compared_to_lhs.at(middle_key);
    std::vector<Comparison> new_knowns;

    auto attempt_transitive = [&](const Comparison& cmp) {
      INTERNAL_CHECK(cmp.IsNormalized()) << "Internal error: comparison should be normalized";

      Key right_key = cmp.rhs_;
      if (right_key == lhs_key) {
        return;
      }

      for (const auto& prev : prev_knowns_using_middle) {
        CompareResult new_result = CompareResult::kUnknown;
        int64_t new_offset = prev.offset_ + cmp.offset_;

        if (prev.result_ == CompareResult::kEQ || cmp.result_ == CompareResult::kEQ) {
          // x == y + c1 && y OP z + c2 ==> x OP z + (c1 + c2)
          // x OP y + c1 && y == z + c2 ==> x OP z + (c1 + c2)
          // When one side is EQ, the other side's operator propagates.
          new_result = (prev.result_ == CompareResult::kEQ) ? cmp.result_ : prev.result_;
        } else if (prev.result_ == cmp.result_ &&
                   (prev.result_ == CompareResult::kLE || prev.result_ == CompareResult::kGE)) {
          // x <= y + c1 && y <= z + c2 ==> x <= z + (c1 + c2)
          // x >= y + c1 && y >= z + c2 ==> x >= z + (c1 + c2)
          new_result = prev.result_;
        }

        if (new_result != CompareResult::kUnknown) {
          new_knowns.emplace_back(lhs_key, right_key, new_offset, new_result);
        }
      }
    };

    // Try combining with each original known comparison.
    for (const auto& known : knowns_) {
      if (auto cmp = known.WithLHS(middle_key)) {
        attempt_transitive(cmp.value());
      }
    }
    for (const auto& known : scoped_knowns_) {
      if (auto cmp = known.WithLHS(middle_key)) {
        attempt_transitive(cmp.value());
      }
    }

    for (const auto& new_known : new_knowns) {
      declare_known(new_known);
    }
  }

  auto it = compared_to_lhs.find(rhs_key);
  if (it != compared_to_lhs.end()) {
    return it->second;
  }
  return {};
}

CompareResult TransitiveComparisonAnalyzer::Impl::MergeComparisons(const std::vector<Comparison>& lhs_to_rhs,
                                                                   int64_t offset) {
  CompareResult result = CompareResult::kUnknown;
  for (const auto& cmp : lhs_to_rhs) {
    switch (cmp.result_) {
      case CompareResult::kInconsistent:
        result = CompareResult::kInconsistent;
        break;

      case CompareResult::kEQ:
        if (offset == cmp.offset_) {
          result = result & CompareResult::kEQ;
        } else {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kLE:
        if (cmp.offset_ < offset) {
          result = result & CompareResult::kLT;
        } else if (cmp.offset_ <= offset) {
          result = result & CompareResult::kLE;
        }
        break;

      case CompareResult::kGE:
        if (cmp.offset_ > offset) {
          result = result & CompareResult::kGT;
        } else if (cmp.offset_ >= offset) {
          result = result & CompareResult::kGE;
        }
        break;

      case CompareResult::kNE:
        if (offset == cmp.offset_) {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kUnknown:
        break;

      case CompareResult::kGT:
      case CompareResult::kLT:
        INTERNAL_CHECK(false) << "Internal error: normalized comparisons should only include LE and GE";
        break;
    }
  }
  return result;
}

// ============================================================================
// Public API
// ============================================================================

CompareResult TransitiveComparisonAnalyzer::Impl::TryCompare(const ExprPtr& lhs_expr, const ExprPtr& rhs_expr,
                                                             bool propagate_inequalities) const {
  // Only supports integer expressions.
  auto lhs_stype = std::dynamic_pointer_cast<const ScalarType>(lhs_expr->GetType());
  auto rhs_stype = std::dynamic_pointer_cast<const ScalarType>(rhs_expr->GetType());
  if (!lhs_stype || !rhs_stype || !lhs_stype->dtype_.IsInt() || !rhs_stype->dtype_.IsInt()) {
    return CompareResult::kUnknown;
  }

  // Early return for two constants.
  auto lhs_ci = As<ConstInt>(lhs_expr);
  auto rhs_ci = As<ConstInt>(rhs_expr);
  if (lhs_ci && rhs_ci) {
    if (lhs_ci->value_ < rhs_ci->value_) return CompareResult::kLT;
    if (lhs_ci->value_ > rhs_ci->value_) return CompareResult::kGT;
    return CompareResult::kEQ;
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);

  // Tautological case: same base expression (e.g., x+1 vs x+2).
  if (lhs && rhs && lhs.get() == rhs.get()) {
    if (offset < 0) return CompareResult::kGT;
    if (offset > 0) return CompareResult::kLT;
    return CompareResult::kEQ;
  }

  // Use zero sentinel for pure-constant sides.
  auto lhs_key = lhs ? ExprToPreviousKey(lhs) : ExprToPreviousKey(GetZeroExpr());
  auto rhs_key = rhs ? ExprToPreviousKey(rhs) : ExprToPreviousKey(GetZeroExpr());

  if (!lhs_key.has_value() || !rhs_key.has_value()) {
    return CompareResult::kUnknown;
  }

  auto lhs_to_rhs = propagate_inequalities ? CollectIndirectComparisons(lhs_key.value(), rhs_key.value())
                                           : CollectDirectComparisons(lhs_key.value(), rhs_key.value());
  return MergeComparisons(lhs_to_rhs, offset);
}

void TransitiveComparisonAnalyzer::Impl::Bind(const VarPtr& var, const ExprPtr& expr,
                                              bool /*allow_override*/) {
  ExprPtr var_expr = std::static_pointer_cast<const Expr>(var);
  AddKnown(MakeEq(var_expr, expr), &knowns_);
}

void TransitiveComparisonAnalyzer::Impl::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive,
                                              bool /*allow_override*/) {
  DataType dtype = GetScalarDtype(var);
  ExprPtr var_expr = std::static_pointer_cast<const Expr>(var);

  if (max_val_exclusive - min_val == 1) {
    // Exact value.
    AddKnown(MakeEq(var_expr, MakeConstInt(min_val, dtype)), &knowns_);
  } else {
    AddKnown(MakeGe(var_expr, MakeConstInt(min_val, dtype)), &knowns_);
    AddKnown(MakeLt(var_expr, MakeConstInt(max_val_exclusive, dtype)), &knowns_);
  }
}

void TransitiveComparisonAnalyzer::Impl::Unbind(const VarPtr& var) {
  auto key = ExprToPreviousKey(std::static_pointer_cast<const Expr>(var));
  if (!key.has_value()) return;

  Key var_key = key.value();
  auto involves_var = [var_key](const Comparison& cmp) { return cmp.lhs_ == var_key || cmp.rhs_ == var_key; };
  knowns_.erase(std::remove_if(knowns_.begin(), knowns_.end(), involves_var), knowns_.end());
}

std::function<void()> TransitiveComparisonAnalyzer::Impl::EnterConstraint(const ExprPtr& constraint) {
  size_t old_size = scoped_knowns_.size();
  AddKnown(constraint, &scoped_knowns_);
  size_t new_size = scoped_knowns_.size();

  if (old_size == new_size) {
    return []() {};  // No constraints extracted — return no-op for consistent callable return.
  }

  return [old_size, new_size, this]() {
    INTERNAL_CHECK(scoped_knowns_.size() == new_size)
        << "Internal error: scoped_knowns_ size changed unexpectedly";
    scoped_knowns_.erase(scoped_knowns_.begin() + static_cast<ptrdiff_t>(old_size), scoped_knowns_.end());
  };
}

// ============================================================================
// TransitiveComparisonAnalyzer (forwarding to Impl)
// ============================================================================

TransitiveComparisonAnalyzer::TransitiveComparisonAnalyzer() : impl_(std::make_unique<Impl>(nullptr)) {}

TransitiveComparisonAnalyzer::TransitiveComparisonAnalyzer(Analyzer* parent)
    : impl_(std::make_unique<Impl>(parent)) {}

TransitiveComparisonAnalyzer::~TransitiveComparisonAnalyzer() = default;

TransitiveComparisonAnalyzer::TransitiveComparisonAnalyzer(TransitiveComparisonAnalyzer&&) noexcept = default;
TransitiveComparisonAnalyzer& TransitiveComparisonAnalyzer::operator=(
    TransitiveComparisonAnalyzer&&) noexcept = default;

CompareResult TransitiveComparisonAnalyzer::TryCompare(const ExprPtr& lhs, const ExprPtr& rhs,
                                                       bool propagate_inequalities) const {
  return impl_->TryCompare(lhs, rhs, propagate_inequalities);
}

void TransitiveComparisonAnalyzer::Bind(const VarPtr& var, const ExprPtr& expr, bool allow_override) {
  impl_->Bind(var, expr, allow_override);
}

void TransitiveComparisonAnalyzer::Bind(const VarPtr& var, int64_t min_val, int64_t max_val_exclusive,
                                        bool allow_override) {
  impl_->Bind(var, min_val, max_val_exclusive, allow_override);
}

void TransitiveComparisonAnalyzer::Unbind(const VarPtr& var) { impl_->Unbind(var); }

std::function<void()> TransitiveComparisonAnalyzer::EnterConstraint(const ExprPtr& constraint) {
  return impl_->EnterConstraint(constraint);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
