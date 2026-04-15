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

/// Trip count for a static for-loop range.
int64_t ComputeStaticTripCount(int64_t start, int64_t stop, int64_t step) {
  if (step > 0 && start < stop) return (stop - start + step - 1) / step;
  if (step < 0 && start > stop) return (start - stop + (-step) - 1) / (-step);
  return 0;
}

ExprPtr MakeConstIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
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
 *        into a replicated outer loop (+ optional remainder loop).
 *
 * Triggers only on loops with a compile-time constant trip count. Loops with
 * iter_args / loop-carried state are rejected; this matches the existing
 * full-unroll pass restriction and keeps the first cut focused on the
 * ping-pong-buffering use case.
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

    int64_t start = GetConstIntValue(op->start_, "start");
    int64_t stop = GetConstIntValue(op->stop_, "stop");
    int64_t step = GetConstIntValue(op->step_, "step");
    CHECK(step != 0) << "PartialUnrollTileLoops: step cannot be zero";

    int64_t trip = ComputeStaticTripCount(start, stop, step);
    if (factor == 1 || trip == 0) {
      // Drop the unroll_factor attr and return otherwise unchanged.
      auto cleaned = MutableCopy(op);
      cleaned->body_ = inner_body;
      cleaned->attrs_ = RewriteAttrs(op->attrs_, std::nullopt);
      return cleaned;
    }

    int64_t main_iters = trip / factor;
    int64_t rem_iters = trip % factor;

    std::vector<StmtPtr> result;
    if (main_iters > 0) {
      result.push_back(BuildMainLoop(op, inner_body, factor, start, step, main_iters));
    }
    if (rem_iters > 0) {
      result.push_back(BuildRemainderLoop(op, inner_body, factor, start, step, main_iters));
    }
    return SeqStmts::Flatten(std::move(result), op->span_);
  }

 private:
  /**
   * @brief Build the replicated main loop: body is a SeqStmts of `factor` clones,
   *        each with the original loop var substituted by `(new_var + k * step)`.
   */
  StmtPtr BuildMainLoop(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t start,
                        int64_t step, int64_t main_iters) {
    Span sp = op->span_;
    VarPtr new_loop_var = CloneLoopVar(op->loop_var_);

    std::vector<StmtPtr> clones;
    clones.reserve(static_cast<size_t>(factor));
    for (int64_t k = 0; k < factor; ++k) {
      ExprPtr substitute;
      if (k == 0) {
        substitute = new_loop_var;
      } else {
        substitute = MakeAdd(new_loop_var, MakeConstIndex(k * step, sp), sp);
      }
      std::unordered_map<const Var*, ExprPtr> sub_map = {{op->loop_var_.get(), substitute}};
      auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);
      clones.push_back(cloned.cloned_body);
    }
    auto new_body = SeqStmts::Flatten(std::move(clones), sp);

    // Outer step becomes factor * step; outer stop is the largest multiple of
    // (factor * step) that fits within the original range.
    ExprPtr new_start = op->start_;
    ExprPtr new_stop = MakeConstIndex(start + main_iters * factor * step, sp);
    ExprPtr new_step = MakeConstIndex(factor * step, sp);

    Attrs new_attrs = RewriteAttrs(op->attrs_, factor);
    return std::make_shared<ForStmt>(new_loop_var, new_start, new_stop, new_step,
                                     /*iter_args=*/std::vector<IterArgPtr>{}, new_body,
                                     /*return_vars=*/std::vector<VarPtr>{}, sp, op->kind_,
                                     /*chunk_config=*/std::nullopt, new_attrs, op->leading_comments_);
  }

  /**
   * @brief Build the trailing remainder loop covering the iterations that don't
   *        fit a full replicated stride. No replication, no marker attr.
   */
  StmtPtr BuildRemainderLoop(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t start,
                             int64_t step, int64_t main_iters) {
    Span sp = op->span_;
    VarPtr new_loop_var = CloneLoopVar(op->loop_var_);

    std::unordered_map<const Var*, ExprPtr> sub_map = {{op->loop_var_.get(), new_loop_var}};
    auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);

    ExprPtr rem_start = MakeConstIndex(start + main_iters * factor * step, sp);
    ExprPtr rem_stop = op->stop_;
    ExprPtr rem_step = MakeConstIndex(step, sp);

    Attrs new_attrs = RewriteAttrs(op->attrs_, std::nullopt);
    return std::make_shared<ForStmt>(new_loop_var, rem_start, rem_stop, rem_step,
                                     /*iter_args=*/std::vector<IterArgPtr>{}, cloned.cloned_body,
                                     /*return_vars=*/std::vector<VarPtr>{}, sp, op->kind_,
                                     /*chunk_config=*/std::nullopt, new_attrs,
                                     /*leading_comments=*/std::vector<std::string>{});
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
