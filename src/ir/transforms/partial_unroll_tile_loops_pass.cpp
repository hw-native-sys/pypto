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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using Attrs = std::vector<std::pair<std::string, std::any>>;

namespace {

constexpr const char* kUnrollFactorAttr = "unroll_factor";
constexpr const char* kUnrollReplicatedAttr = "unroll_replicated";

/// Extract a compile-time integer from a ConstInt or Neg(ConstInt) expression.
int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto neg = std::dynamic_pointer_cast<const Neg>(expr)) {
    if (auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_)) {
      return -inner->value_;
    }
  }
  throw pypto::ValueError("PartialUnrollTileLoops: " + what +
                          " must be a compile-time integer constant, got " + expr->TypeName());
}

/// Non-throwing variant — returns nullopt if `expr` is not a compile-time integer.
std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto neg = std::dynamic_pointer_cast<const Neg>(expr)) {
    if (auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_)) {
      return -inner->value_;
    }
  }
  return std::nullopt;
}

/// Trip count for a static for-loop range.
int64_t ComputeStaticTripCount(int64_t start, int64_t stop, int64_t step) {
  if (step > 0 && start < stop) return (stop - start + step - 1) / step;
  if (step < 0 && start > stop) return (start - stop + (-step) - 1) / (-step);
  return 0;
}

ExprPtr MakeConstIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

/// `base + offset_val`, with constant-folding when `base` is a ConstInt.
/// Emitting the unfolded form trips the round-trip verifier because the
/// reparser folds `8 + 1` back to `9`.
ExprPtr OffsetIndex(const ExprPtr& base, int64_t offset_val, const Span& span) {
  if (offset_val == 0) return base;
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(base)) {
    return MakeConstIndex(ci->value_ + offset_val, span);
  }
  return MakeAdd(base, MakeConstIndex(offset_val, span), span);
}

/// Copy `original` while removing `kUnrollFactorAttr` and (optionally) adding
/// `kUnrollReplicatedAttr` with `factor` as an int.
Attrs RewriteAttrs(const Attrs& original, std::optional<int64_t> replicated_factor) {
  Attrs out;
  out.reserve(original.size() + 1);
  for (const auto& [k, v] : original) {
    if (k == kUnrollFactorAttr) continue;
    if (k == kUnrollReplicatedAttr) continue;  // never preserve a stale marker
    out.emplace_back(k, v);
  }
  if (replicated_factor.has_value()) {
    out.emplace_back(kUnrollReplicatedAttr, static_cast<int>(*replicated_factor));
  }
  return out;
}

/// Build a fresh outer loop variable mirroring `original` (same name, same type, same span).
VarPtr CloneLoopVar(const VarPtr& original) {
  return std::make_shared<Var>(original->name_hint_, original->GetType(), original->span_);
}

/**
 * @brief Mutator that lowers ForStmt nodes carrying `attrs_["unroll_factor"]`
 *        into a replicated main loop plus a modulo-dispatch remainder.
 *
 * Static bounds → single-branch tail with exactly rem_iters clones.
 * Dynamic bounds (start and/or stop are runtime Exprs) → a cascaded
 *   `if rem == k` dispatch for k in [1, factor), each branch containing
 *   k cloned bodies tagged `unroll_replicated = k`. Step must always be a
 *   compile-time constant; iter_args are rejected (matches full-unroll).
 */
class PartialUnrollMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (!op->HasAttr(kUnrollFactorAttr)) {
      return IRMutator::VisitStmt_(op);
    }
    int64_t factor = static_cast<int64_t>(op->GetAttr<int>(kUnrollFactorAttr, 0));
    CHECK(factor >= 1) << "PartialUnrollTileLoops: unroll_factor must be >= 1, got " << factor;
    CHECK(op->iter_args_.empty())
        << "PartialUnrollTileLoops: loops with iter_args/init_values are not supported. "
           "Drop init_values= or remove unroll= from the loop.";

    // Recurse into the body first so nested unroll-marked loops are lowered too.
    auto inner_body = VisitStmt(op->body_);

    // Step must always be static — the main loop's stride and per-clone offsets
    // both depend on `factor * step` being a compile-time integer.
    int64_t step = GetConstIntValue(op->step_, "step");
    CHECK(step != 0) << "PartialUnrollTileLoops: step cannot be zero";

    // factor == 1: drop the attr and keep the loop otherwise unchanged.
    if (factor == 1) {
      return CleanupUnrollAttr(op, inner_body);
    }

    auto start_const = TryGetConstInt(op->start_);
    auto stop_const = TryGetConstInt(op->stop_);
    if (start_const.has_value() && stop_const.has_value()) {
      return LowerStatic(op, inner_body, factor, *start_const, *stop_const, step);
    }
    return LowerDynamic(op, inner_body, factor, step);
  }

 private:
  /// No replication needed — drop the attr and return the loop unchanged.
  StmtPtr CleanupUnrollAttr(const ForStmtPtr& op, const StmtPtr& inner_body) {
    auto cleaned = MutableCopy(op);
    cleaned->body_ = inner_body;
    cleaned->attrs_ = RewriteAttrs(op->attrs_, std::nullopt);
    return cleaned;
  }

  /**
   * @brief Build the replicated main loop: body is a SeqStmts of `factor` clones,
   *        each with the original loop var substituted by `(new_var + k * step)`.
   */
  StmtPtr BuildMainLoop(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t step,
                        const ExprPtr& main_start, const ExprPtr& main_stop) {
    Span sp = op->span_;
    VarPtr new_loop_var = CloneLoopVar(op->loop_var_);

    std::vector<StmtPtr> clones;
    clones.reserve(static_cast<size_t>(factor));
    for (int64_t k = 0; k < factor; ++k) {
      ExprPtr substitute = OffsetIndex(new_loop_var, k * step, sp);
      std::unordered_map<const Var*, ExprPtr> sub_map = {{op->loop_var_.get(), substitute}};
      auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);
      clones.push_back(cloned.cloned_body);
    }
    auto new_body = SeqStmts::Flatten(std::move(clones), sp);

    ExprPtr new_step = MakeConstIndex(factor * step, sp);
    Attrs new_attrs = RewriteAttrs(op->attrs_, factor);
    return std::make_shared<ForStmt>(new_loop_var, main_start, main_stop, new_step,
                                     /*iter_args=*/std::vector<IterArgPtr>{}, new_body,
                                     /*return_vars=*/std::vector<VarPtr>{}, sp, op->kind_,
                                     /*chunk_config=*/std::nullopt, new_attrs, op->leading_comments_);
  }

  /**
   * @brief Build a trip-1 ForStmt wrapping `k_clones` cloned bodies at
   *        offsets `base_index + j*step` (j in [0, k_clones)), tagged with
   *        `unroll_replicated = k_clones` so ReorderUnrolledIO processes it.
   *
   * This is the attrs-bearing container for a remainder branch; SeqStmts has
   * no attrs_, so we wrap in a degenerate ForStmt purely for the marker.
   */
  StmtPtr BuildTailBranch(const ForStmtPtr& op, const StmtPtr& body, int64_t k_clones, int64_t step,
                          const ExprPtr& base_index) {
    Span sp = op->span_;
    std::vector<StmtPtr> clones;
    clones.reserve(static_cast<size_t>(k_clones));
    for (int64_t j = 0; j < k_clones; ++j) {
      ExprPtr substitute = OffsetIndex(base_index, j * step, sp);
      std::unordered_map<const Var*, ExprPtr> sub_map = {{op->loop_var_.get(), substitute}};
      auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);
      clones.push_back(cloned.cloned_body);
    }
    auto seq_body = SeqStmts::Flatten(std::move(clones), sp);

    auto dummy_var = std::make_shared<Var>("_tail_iter_" + std::to_string(k_clones),
                                           std::make_shared<ScalarType>(DataType::INDEX), sp);
    Attrs attrs = {{kUnrollReplicatedAttr, static_cast<int>(k_clones)}};
    return std::make_shared<ForStmt>(dummy_var, MakeConstIndex(0, sp), MakeConstIndex(1, sp),
                                     MakeConstIndex(1, sp),
                                     /*iter_args=*/std::vector<IterArgPtr>{}, seq_body,
                                     /*return_vars=*/std::vector<VarPtr>{}, sp, ForKind::Sequential,
                                     /*chunk_config=*/std::nullopt, attrs,
                                     /*leading_comments=*/std::vector<std::string>{});
  }

  /**
   * @brief Static lowering: compile-time trip count → main loop + (optional)
   *        single-branch tail with exactly rem_iters clones. No dispatch needed
   *        because the remainder count is known.
   */
  StmtPtr LowerStatic(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t start, int64_t stop,
                      int64_t step) {
    int64_t trip = ComputeStaticTripCount(start, stop, step);
    if (trip == 0) {
      return CleanupUnrollAttr(op, body);
    }
    int64_t main_iters = trip / factor;
    int64_t rem_iters = trip % factor;

    std::vector<StmtPtr> result;
    if (main_iters > 0) {
      ExprPtr main_start = op->start_;
      ExprPtr main_stop = MakeConstIndex(start + main_iters * factor * step, op->span_);
      result.push_back(BuildMainLoop(op, body, factor, step, main_start, main_stop));
    }
    if (rem_iters > 0) {
      int64_t tail_base = start + main_iters * factor * step;
      ExprPtr base_index = MakeConstIndex(tail_base, op->span_);
      result.push_back(BuildTailBranch(op, body, rem_iters, step, base_index));
    }
    return SeqStmts::Flatten(std::move(result), op->span_);
  }

  /**
   * @brief Dynamic lowering: start and/or stop are runtime Exprs. Emits:
   *
   *   trip_iters    = ceil_div(stop - start, step)
   *   main_iters    = trip_iters / factor                       (compile-time: `/ factor`)
   *   main_end      = start + main_iters * (factor * step)      (SSA-bound to `unroll_main_end`)
   *   for i in range(start, main_end, F*step) [unroll_replicated=F]: <F clones>
   *   rem_iters     = trip_iters - main_iters * factor          (SSA-bound to `unroll_rem`)
   *   if rem_iters == 1: <1 clone>      # outermost
   *   else if rem_iters == 2: <2 clones>
   *   else ...
   *   else if rem_iters == F-1: <F-1 clones>
   *   # rem_iters == 0 matches no branch → tail is skipped.
   *
   * Dynamic bounds require step > 0; negative-step dynamic ranges are not in
   * the first-cut scope (static bounds handle negative step via
   * ComputeStaticTripCount).
   */
  StmtPtr LowerDynamic(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t step) {
    Span sp = op->span_;
    CHECK(step > 0) << "PartialUnrollTileLoops: dynamic bounds require a positive step, got " << step
                    << ". Use static bounds for negative-step loops.";

    // trip_iters = ceil_div(stop - start, step). For step == 1 the ceil_div
    // collapses to (stop - start), so skip the `+ (step-1)` / `// step` wrapping
    // to keep the emitted IR minimal.
    ExprPtr span_expr = MakeSub(op->stop_, op->start_, sp);
    ExprPtr trip_expr;
    if (step == 1) {
      trip_expr = span_expr;
    } else {
      ExprPtr step_expr = MakeConstIndex(step, sp);
      ExprPtr adjusted = MakeAdd(span_expr, MakeConstIndex(step - 1, sp), sp);
      trip_expr = MakeFloorDiv(adjusted, step_expr, sp);
    }

    ExprPtr factor_expr = MakeConstIndex(factor, sp);
    ExprPtr main_iters_expr = MakeFloorDiv(trip_expr, factor_expr, sp);

    ExprPtr chunk = MakeConstIndex(factor * step, sp);
    ExprPtr scaled = MakeMul(main_iters_expr, chunk, sp);
    ExprPtr main_end_value = MakeAdd(op->start_, scaled, sp);

    VarPtr main_end_var =
        std::make_shared<Var>("unroll_main_end", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto main_end_assign = std::make_shared<AssignStmt>(main_end_var, main_end_value, sp);

    // Main loop — stop is the fresh SSA var `main_end_var`.
    StmtPtr main_loop = BuildMainLoop(op, body, factor, step,
                                      /*main_start=*/op->start_,
                                      /*main_stop=*/main_end_var);

    // rem_iters = trip_iters - main_iters * factor. For step == 1 this equals
    // stop - main_end (since trip == stop - start and main_iters*factor*step ==
    // main_end - start collapse), which keeps the emitted IR simple for the
    // common case.
    VarPtr rem_var = std::make_shared<Var>("unroll_rem", std::make_shared<ScalarType>(DataType::INDEX), sp);
    ExprPtr rem_value = (step == 1) ? MakeSub(op->stop_, main_end_var, sp)
                                    : MakeSub(trip_expr, MakeMul(main_iters_expr, factor_expr, sp), sp);
    auto rem_assign = std::make_shared<AssignStmt>(rem_var, rem_value, sp);

    // Build the cascade from innermost (k = factor-1) outward so that each outer
    // IfStmt can point at the previously-built IfStmt as its else branch.
    std::optional<StmtPtr> inner;
    for (int64_t k = factor - 1; k >= 1; --k) {
      ExprPtr cond = MakeEq(rem_var, MakeConstIndex(k, sp), sp);
      StmtPtr branch_body = BuildTailBranch(op, body, k, step, main_end_var);
      auto if_stmt = std::make_shared<IfStmt>(cond, branch_body, inner,
                                              /*return_vars=*/std::vector<VarPtr>{}, sp);
      inner = StmtPtr(if_stmt);
    }

    std::vector<StmtPtr> result;
    result.push_back(main_end_assign);
    result.push_back(main_loop);
    if (inner.has_value()) {
      result.push_back(rem_assign);
      result.push_back(*inner);
    }
    return SeqStmts::Flatten(std::move(result), sp);
  }
};

FunctionPtr TransformPartialUnrollTileLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "PartialUnrollTileLoops cannot run on null function";
  PartialUnrollMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass PartialUnrollTileLoops() {
  return CreateFunctionPass(TransformPartialUnrollTileLoops, "PartialUnrollTileLoops",
                            kPartialUnrollTileLoopsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
