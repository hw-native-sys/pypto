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

/// Fresh IterArg mirroring `original`, with `init_value` as the initial value.
IterArgPtr MakeFreshIterArg(const IterArgPtr& original, const ExprPtr& init_value) {
  return std::make_shared<IterArg>(original->name_hint_, original->GetType(), init_value, original->span_);
}

/// Fresh Var mirroring `original` with a suffixed name (for intermediate return_vars).
VarPtr MakeFreshVar(const VarPtr& original, const std::string& suffix) {
  return std::make_shared<Var>(original->name_hint_ + suffix, original->GetType(), original->span_);
}

/// Split a body into (stmts_before_yield, yield_values). If the body ends with a
/// terminal `YieldStmt` (either standalone or as the final stmt of a top-level
/// `SeqStmts`), strip it and return its values. Otherwise return the body unchanged
/// and an empty value list. Always pass through — callers that have no iter_args
/// simply see an empty yield vector and treat `stmts` as the whole body.
std::pair<StmtPtr, std::vector<ExprPtr>> SplitBodyYield(const StmtPtr& body) {
  if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(body)) {
    return {std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, body->span_), yield->value_};
  }
  auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
  if (!seq || seq->stmts_.empty()) {
    return {body, {}};
  }
  auto yield = std::dynamic_pointer_cast<const YieldStmt>(seq->stmts_.back());
  if (!yield) {
    return {body, {}};
  }
  std::vector<StmtPtr> without(seq->stmts_.begin(), seq->stmts_.end() - 1);
  return {std::make_shared<SeqStmts>(std::move(without), seq->span_), yield->value_};
}

/**
 * @brief Mutator that lowers ForStmt nodes carrying `attrs_["unroll_factor"]`
 *        into a replicated main loop plus a modulo-dispatch remainder.
 *
 * Static bounds → single-branch tail with exactly rem_iters clones.
 * Dynamic bounds (start and/or stop are runtime Exprs) → a cascaded
 *   `if rem == k` dispatch for k in [1, factor), each branch containing
 *   k cloned bodies tagged `unroll_replicated = k`. Step must always be a
 *   compile-time constant.
 *
 * `iter_args` are supported: loop-carried state threads sequentially through the
 * F replicated clones in the main loop (each clone consumes the previous clone's
 * yielded expressions), and through the tail clones starting from the main
 * loop's return_vars. In the dynamic case, each IfStmt in the cascade carries
 * `return_vars` matching the iter_args types; the innermost else yields the
 * main-loop return_vars so the `rem == 0` fall-through is a no-op.
 */
class PartialUnrollMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (!op->HasAttr(kUnrollFactorAttr)) {
      return IRMutator::VisitStmt_(op);
    }
    int64_t factor = static_cast<int64_t>(op->GetAttr<int>(kUnrollFactorAttr, 0));
    CHECK(factor >= 1) << "PartialUnrollTileLoops: unroll_factor must be >= 1, got " << factor;

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
   * @brief Clone `body` `n` times with loop-var / iter-arg substitutions,
   *        threading loop-carried state through the clones.
   *
   * Each clone k:
   *  - substitutes `loop_var → base + k * step` (via OffsetIndex)
   *  - substitutes original iter_args with `initial_iter_substitutes` (when k == 0)
   *    or with the previous clone's yielded expressions (when k > 0)
   *  - is DeepCloned with `clone_def_vars=true` so nested definitions get fresh SSA vars
   *  - has its trailing `YieldStmt` (if any) stripped into the next clone's substitution
   *
   * Returns the concatenated body (a `SeqStmts` of the stripped clones) paired with
   * the last clone's yielded expressions. For loops without iter_args, the yield
   * vector is empty and each cloned body is appended verbatim.
   */
  struct ReplicatedRegion {
    StmtPtr body;                       // SeqStmts of cloned bodies (yields stripped)
    std::vector<ExprPtr> final_yields;  // last clone's yielded expressions
  };

  ReplicatedRegion ReplicateBody(const ForStmtPtr& op, const StmtPtr& body, int64_t n_clones, int64_t step,
                                 const ExprPtr& base, const std::vector<ExprPtr>& initial_iter_substitutes) {
    Span sp = op->span_;
    INTERNAL_CHECK(initial_iter_substitutes.size() == op->iter_args_.size())
        << "Internal error: iter substitute count mismatch";

    std::vector<StmtPtr> clones;
    clones.reserve(static_cast<size_t>(n_clones));
    std::vector<ExprPtr> prev_yields;
    for (int64_t k = 0; k < n_clones; ++k) {
      std::unordered_map<const Var*, ExprPtr> sub_map;
      sub_map[op->loop_var_.get()] = OffsetIndex(base, k * step, sp);
      for (size_t j = 0; j < op->iter_args_.size(); ++j) {
        sub_map[op->iter_args_[j].get()] = (k == 0) ? initial_iter_substitutes[j] : prev_yields[j];
      }
      auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);
      auto [cloned_stmts, cloned_yields] = SplitBodyYield(cloned.cloned_body);
      INTERNAL_CHECK(cloned_yields.size() == op->iter_args_.size())
          << "Internal error: loop body must yield " << op->iter_args_.size() << " values for iter_args, got "
          << cloned_yields.size();
      clones.push_back(cloned_stmts);
      prev_yields = std::move(cloned_yields);
    }
    return {SeqStmts::Flatten(std::move(clones), sp), std::move(prev_yields)};
  }

  /// Convert a vector of return_vars into a vector of ExprPtrs (for YieldStmt values
  /// or iter_arg init-value forwarding).
  std::vector<ExprPtr> ReturnVarsAsExprs(const std::vector<VarPtr>& vars) {
    std::vector<ExprPtr> result;
    result.reserve(vars.size());
    for (const auto& v : vars) result.push_back(v);
    return result;
  }

  /// Collect `initValue_` expressions from a vector of IterArgs — used when the
  /// tail runs without a preceding main loop, so its iter_args seed directly
  /// from the source loop's init values rather than a main-loop return_var.
  std::vector<ExprPtr> InitValueExprs(const std::vector<IterArgPtr>& iter_args) {
    std::vector<ExprPtr> result;
    result.reserve(iter_args.size());
    for (const auto& ia : iter_args) result.push_back(ia->initValue_);
    return result;
  }

  /// Fresh iter_args for a wrapper ForStmt, taking initial values from `init_values`.
  std::vector<IterArgPtr> MakeFreshIterArgs(const std::vector<IterArgPtr>& originals,
                                            const std::vector<ExprPtr>& init_values) {
    INTERNAL_CHECK(originals.size() == init_values.size())
        << "Internal error: iter_arg/init_value count mismatch";
    std::vector<IterArgPtr> result;
    result.reserve(originals.size());
    for (size_t j = 0; j < originals.size(); ++j) {
      result.push_back(MakeFreshIterArg(originals[j], init_values[j]));
    }
    return result;
  }

  /// Fresh return_vars matching the originals' types, with a suffix applied to names.
  std::vector<VarPtr> MakeFreshReturnVars(const std::vector<VarPtr>& originals, const std::string& suffix) {
    std::vector<VarPtr> result;
    result.reserve(originals.size());
    for (const auto& v : originals) result.push_back(MakeFreshVar(v, suffix));
    return result;
  }

  /**
   * @brief Build the replicated main loop.
   *
   * Body is a SeqStmts of `factor` clones threading iter_args sequentially. When
   * the original loop has iter_args, the main loop gets fresh iter_args seeded
   * from the originals' init values; each clone consumes the previous clone's
   * yield, and the body ends with a YieldStmt of the last clone's yields to feed
   * the next outer iteration. `main_return_vars` controls the ForStmt's
   * return_vars (may be the original return_vars or fresh ones, depending on
   * whether a tail follows).
   */
  StmtPtr BuildMainLoop(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t step,
                        const ExprPtr& main_start, const ExprPtr& main_stop,
                        const std::vector<VarPtr>& main_return_vars) {
    Span sp = op->span_;
    VarPtr new_loop_var = CloneLoopVar(op->loop_var_);

    // Fresh iter_args mirroring the originals (same init values as the source loop).
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(op->iter_args_.size());
    std::vector<ExprPtr> initial_substitutes;
    initial_substitutes.reserve(op->iter_args_.size());
    for (const auto& orig : op->iter_args_) {
      auto fresh = MakeFreshIterArg(orig, orig->initValue_);
      new_iter_args.push_back(fresh);
      initial_substitutes.push_back(fresh);
    }

    auto region = ReplicateBody(op, body, factor, step, new_loop_var, initial_substitutes);

    // Body = replicated clones, followed by YieldStmt(last_yields) when iter_args exist.
    std::vector<StmtPtr> body_parts = {region.body};
    if (!op->iter_args_.empty()) {
      body_parts.push_back(std::make_shared<YieldStmt>(region.final_yields, sp));
    }
    auto new_body = SeqStmts::Flatten(std::move(body_parts), sp);

    ExprPtr new_step = MakeConstIndex(factor * step, sp);
    Attrs new_attrs = RewriteAttrs(op->attrs_, factor);
    return std::make_shared<ForStmt>(new_loop_var, main_start, main_stop, new_step, new_iter_args, new_body,
                                     main_return_vars, sp, op->kind_,
                                     /*chunk_config=*/std::nullopt, new_attrs, op->leading_comments_);
  }

  /**
   * @brief Build a trip-1 ForStmt wrapping `k_clones` cloned bodies at offsets
   *        `base_index + j*step` (j in [0, k_clones)), tagged with
   *        `unroll_replicated = k_clones` so ReorderUnrolledIO processes it.
   *
   * SeqStmts has no attrs_, so the wrapper exists purely to carry the marker.
   * When the source loop has iter_args, the wrapper also threads loop-carried
   * state: its fresh iter_args are seeded from `iter_init_values`, its clones
   * are chained via the previous clone's yields, and a trailing YieldStmt feeds
   * the wrapper's `return_vars`.
   */
  StmtPtr BuildTailBranch(const ForStmtPtr& op, const StmtPtr& body, int64_t k_clones, int64_t step,
                          const ExprPtr& base_index, const std::vector<ExprPtr>& iter_init_values,
                          const std::vector<VarPtr>& return_vars) {
    Span sp = op->span_;
    auto fresh_iter_args = MakeFreshIterArgs(op->iter_args_, iter_init_values);
    std::vector<ExprPtr> initial_substitutes;
    initial_substitutes.reserve(fresh_iter_args.size());
    for (const auto& ia : fresh_iter_args) initial_substitutes.push_back(ia);

    auto region = ReplicateBody(op, body, k_clones, step, base_index, initial_substitutes);

    std::vector<StmtPtr> body_parts = {region.body};
    if (!op->iter_args_.empty()) {
      body_parts.push_back(std::make_shared<YieldStmt>(region.final_yields, sp));
    }
    auto seq_body = SeqStmts::Flatten(std::move(body_parts), sp);

    auto dummy_var = std::make_shared<Var>("_tail_iter_" + std::to_string(k_clones),
                                           std::make_shared<ScalarType>(DataType::INDEX), sp);
    Attrs attrs = {{kUnrollReplicatedAttr, static_cast<int>(k_clones)}};
    return std::make_shared<ForStmt>(dummy_var, MakeConstIndex(0, sp), MakeConstIndex(1, sp),
                                     MakeConstIndex(1, sp), fresh_iter_args, seq_body, return_vars, sp,
                                     ForKind::Sequential,
                                     /*chunk_config=*/std::nullopt, attrs,
                                     /*leading_comments=*/std::vector<std::string>{});
  }

  /**
   * @brief Static lowering: compile-time trip count → main loop + (optional)
   *        single-branch tail with exactly rem_iters clones. No dispatch needed
   *        because the remainder count is known.
   *
   * When iter_args are present, the main loop's return_vars forward loop-carried
   * state to the tail branch's iter_args; the tail branch inherits the original
   * outer loop's return_vars so downstream references stay valid.
   */
  StmtPtr LowerStatic(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t start, int64_t stop,
                      int64_t step) {
    int64_t trip = ComputeStaticTripCount(start, stop, step);
    if (trip == 0) {
      return CleanupUnrollAttr(op, body);
    }
    int64_t main_iters = trip / factor;
    int64_t rem_iters = trip % factor;
    bool has_tail = rem_iters > 0;

    // Main loop's return_vars forward to the tail via fresh names when a tail
    // follows; otherwise they terminate at the original names. When there are
    // no return_vars, fresh-rename is a no-op (both empty), so guard that.
    std::vector<VarPtr> main_return_vars = op->return_vars_;
    if (has_tail && !main_return_vars.empty()) {
      main_return_vars = MakeFreshReturnVars(op->return_vars_, "_main");
    }

    std::vector<StmtPtr> result;
    if (main_iters > 0) {
      ExprPtr main_start = op->start_;
      ExprPtr main_stop = MakeConstIndex(start + main_iters * factor * step, op->span_);
      result.push_back(BuildMainLoop(op, body, factor, step, main_start, main_stop, main_return_vars));
    }
    if (has_tail) {
      int64_t tail_base = start + main_iters * factor * step;
      ExprPtr base_index = MakeConstIndex(tail_base, op->span_);
      // Tail iter_args seed from main_return_vars when a main loop precedes
      // the tail; otherwise (trip < factor) they seed from the original loop's
      // init_values.
      std::vector<ExprPtr> tail_init_values =
          (main_iters > 0) ? ReturnVarsAsExprs(main_return_vars) : InitValueExprs(op->iter_args_);
      result.push_back(
          BuildTailBranch(op, body, rem_iters, step, base_index, tail_init_values, op->return_vars_));
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

    // The cascade always lives after the main loop, so the main loop's
    // return_vars forward state to the IfStmt cascade and need fresh names.
    std::vector<VarPtr> main_return_vars =
        op->return_vars_.empty() ? op->return_vars_ : MakeFreshReturnVars(op->return_vars_, "_main");

    // Main loop — stop is the fresh SSA var `main_end_var`.
    StmtPtr main_loop = BuildMainLoop(op, body, factor, step, /*main_start=*/op->start_,
                                      /*main_stop=*/main_end_var, main_return_vars);

    // rem_iters = trip_iters - main_iters * factor. For step == 1 this equals
    // stop - main_end (since trip == stop - start and main_iters*factor*step ==
    // main_end - start collapse), which keeps the emitted IR simple for the
    // common case.
    VarPtr rem_var = std::make_shared<Var>("unroll_rem", std::make_shared<ScalarType>(DataType::INDEX), sp);
    ExprPtr rem_value = (step == 1) ? MakeSub(op->stop_, main_end_var, sp)
                                    : MakeSub(trip_expr, MakeMul(main_iters_expr, factor_expr, sp), sp);
    auto rem_assign = std::make_shared<AssignStmt>(rem_var, rem_value, sp);

    // Fall-through (rem == 0) state expressions — the main loop's return_vars
    // passed through unchanged. Used as the innermost else's YieldStmt and as
    // the seed for each branch's tail iter_args.
    std::vector<ExprPtr> main_return_exprs = ReturnVarsAsExprs(main_return_vars);
    bool has_iter_args = !op->iter_args_.empty();

    // Build the cascade from innermost (k = factor-1) outward so each outer
    // IfStmt's else points at the previously-built IfStmt. With iter_args,
    // every IfStmt carries return_vars (fresh at inner levels, the original
    // outer return_vars at the outermost level) and every branch ends with a
    // YieldStmt — including the innermost else, which yields main_return_exprs
    // for the rem == 0 case.
    std::optional<StmtPtr> inner;
    std::vector<VarPtr> inner_return_vars;
    for (int64_t k = factor - 1; k >= 1; --k) {
      // Each branch's tail seeds its iter_args from the main-loop return_vars:
      // the cascade is a dispatch on `rem`, so every live branch starts from
      // the same post-main state.
      auto tail_branch_return_vars = op->return_vars_.empty()
                                         ? op->return_vars_
                                         : MakeFreshReturnVars(op->return_vars_, "_tail" + std::to_string(k));
      StmtPtr tail =
          BuildTailBranch(op, body, k, step, main_end_var, main_return_exprs, tail_branch_return_vars);

      std::vector<StmtPtr> then_parts = {tail};
      if (has_iter_args) {
        then_parts.push_back(std::make_shared<YieldStmt>(ReturnVarsAsExprs(tail_branch_return_vars), sp));
      }
      auto then_body = SeqStmts::Flatten(std::move(then_parts), sp);

      std::optional<StmtPtr> else_body;
      if (k == factor - 1) {
        // Innermost: rem == 0 fall-through yields the main-loop state.
        if (has_iter_args) else_body = std::make_shared<YieldStmt>(main_return_exprs, sp);
      } else {
        INTERNAL_CHECK(inner.has_value())
            << "Internal error: inner IfStmt must be built by the previous iteration";
        std::vector<StmtPtr> else_parts = {*inner};
        if (has_iter_args) {
          else_parts.push_back(std::make_shared<YieldStmt>(ReturnVarsAsExprs(inner_return_vars), sp));
        }
        else_body = SeqStmts::Flatten(std::move(else_parts), sp);
      }

      // return_vars: original names at the outermost level (k == 1); fresh at inner levels.
      std::vector<VarPtr> my_return_vars;
      if (has_iter_args) {
        my_return_vars =
            (k == 1) ? op->return_vars_ : MakeFreshReturnVars(op->return_vars_, "_rem" + std::to_string(k));
      }

      ExprPtr cond = MakeEq(rem_var, MakeConstIndex(k, sp), sp);
      auto if_stmt = std::make_shared<IfStmt>(cond, then_body, else_body, my_return_vars, sp);
      inner = StmtPtr(if_stmt);
      inner_return_vars = std::move(my_return_vars);
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
