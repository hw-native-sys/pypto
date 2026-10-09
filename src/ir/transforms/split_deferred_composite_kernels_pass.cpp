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

/**
 * @file split_deferred_composite_kernels_pass.cpp
 * @brief Split InCore kernels that mix payload with ``defer_wait`` into push / wait / epi.
 *
 * OutlineIncoreScopes stamps ``deferred_completion_waiter`` on ``defer=True``
 * composites before LowerCompositeOps expands them. After expansion the body
 * contains puts + notify + ``defer_wait`` + epilogue notifies, which fails
 * DeferredWaitContractValidator. This Program pass runs immediately before
 * ExpandMixedKernel and rewrites those mixed kernels into up to three functions,
 * updating Orchestration Submit call sites so the public TaskId is the epilogue
 * (or wait when there is no epilogue).
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/deferred_wait_contract.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

[[nodiscard]] bool ContainsOpNamed(const StmtPtr& body, const char* op_name) {
  class Finder : public IRVisitor {
   public:
    explicit Finder(const char* name) : name_(name) {}
    bool found = false;

   protected:
    void VisitExpr_(const CallPtr& call) override {
      if (IsOp(call, name_)) found = true;
      IRVisitor::VisitExpr_(call);
    }
    const char* name_;
  };
  Finder finder(op_name);
  finder.VisitStmt(body);
  return finder.found;
}

[[nodiscard]] bool IsScalarSetupStmt(const StmtPtr& stmt) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign || !assign->var_ || AsTensorTypeLike(assign->var_->GetType())) return false;
  if (auto call = As<Call>(assign->value_)) {
    return IsOp(call, "pld.system.get_comm_ctx") || IsOp(call, "pld.system.nranks") ||
           IsOp(call, "pld.system.rank");
  }
  // Cast of nranks → INDEX, etc.
  return As<Cast>(assign->value_) != nullptr;
}

[[nodiscard]] std::vector<StmtPtr> FlattenStmts(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  if (body) return {body};
  return {};
}

[[nodiscard]] StmtPtr MakeBody(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.empty()) {
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  if (stmts.size() == 1) return stmts.front();
  return std::make_shared<SeqStmts>(std::move(stmts), span);
}

struct SplitPlan {
  std::string original_name;
  FunctionPtr original;
  FunctionPtr push_func;
  FunctionPtr wait_func;
  FunctionPtr epi_func;  // null when no epilogue stmts
  std::vector<TypePtr> original_return_types;
};

[[nodiscard]] bool NeedsDeferredCompositeSplit(const FunctionPtr& func) {
  if (!func) return false;
  if (func->func_type_ != FunctionType::InCore && func->func_type_ != FunctionType::AIV) return false;
  if (!func->GetAttr<bool>(kAttrDeferredCompletionWaiter, false)) return false;
  if (!outline_utils::ContainsDeferredWait(func->body_)) return false;
  // Pure registration waiters already satisfy ExpandMixedKernel — leave them.
  // Mixed bodies carry payload puts and/or barrier/epilogue notifies.
  return ContainsOpNamed(func->body_, "pld.tile.put") || ContainsOpNamed(func->body_, "pld.system.notify");
}

[[nodiscard]] std::vector<std::pair<std::string, std::any>> CopyAttrsWithoutDeferredMarker(
    const FunctionPtr& func) {
  std::vector<std::pair<std::string, std::any>> attrs;
  attrs.reserve(func->attrs_.size());
  for (const auto& [k, v] : func->attrs_) {
    if (k == kAttrDeferredCompletionWaiter) continue;
    attrs.emplace_back(k, v);
  }
  return attrs;
}

[[nodiscard]] std::vector<std::pair<std::string, std::any>> CopyAttrsKeepDeferredMarker(
    const FunctionPtr& func) {
  return func->attrs_;
}

[[nodiscard]] std::string ClaimUniqueFunctionName(const std::string& preferred,
                                                  std::unordered_set<std::string>* reserved) {
  INTERNAL_CHECK(reserved != nullptr) << "Internal error: null reserved name set";
  if (reserved->insert(preferred).second) return preferred;
  for (size_t suffix = 1;; ++suffix) {
    const std::string candidate = preferred + "_" + std::to_string(suffix);
    if (reserved->insert(candidate).second) return candidate;
  }
}

SplitPlan BuildSplitPlan(const FunctionPtr& func, std::unordered_set<std::string>* reserved_names) {
  SplitPlan plan;
  plan.original_name = func->name_;
  plan.original = func;
  plan.original_return_types = func->return_types_;

  auto stmts = FlattenStmts(func->body_);
  CHECK_SPAN(!stmts.empty(), func->span_)
      << "SplitDeferredCompositeKernels: empty body for deferred composite '" << func->name_ << "'";

  // Strip trailing ReturnStmt for partitioning; re-attach per phase.
  std::vector<ExprPtr> return_values;
  if (auto ret = As<ReturnStmt>(stmts.back())) {
    return_values = ret->value_;
    stmts.pop_back();
  }

  size_t wait_begin = stmts.size();
  size_t wait_end = stmts.size();
  for (size_t i = 0; i < stmts.size(); ++i) {
    if (outline_utils::ContainsDeferredWait(stmts[i])) {
      if (wait_begin == stmts.size()) wait_begin = i;
      wait_end = i + 1;
    }
  }
  CHECK_SPAN(wait_begin < stmts.size(), func->span_)
      << "SplitDeferredCompositeKernels: deferred_completion_waiter function '" << func->name_
      << "' has no defer_wait region to split";

  std::vector<StmtPtr> setup;
  size_t setup_end = 0;
  while (setup_end < wait_begin && IsScalarSetupStmt(stmts[setup_end])) {
    setup.push_back(stmts[setup_end]);
    ++setup_end;
  }

  std::vector<StmtPtr> push_stmts(stmts.begin(), stmts.begin() + static_cast<std::ptrdiff_t>(wait_begin));
  std::vector<StmtPtr> wait_region(stmts.begin() + static_cast<std::ptrdiff_t>(wait_begin),
                                   stmts.begin() + static_cast<std::ptrdiff_t>(wait_end));
  std::vector<StmtPtr> epi_stmts(stmts.begin() + static_cast<std::ptrdiff_t>(wait_end), stmts.end());

  // A wait region that still carries puts/notifies is almost always an if/loop
  // that nested the whole deferred composite. Placement verify rejects that at
  // source; keep a span-anchored CHECK here as defense in depth.
  for (const auto& s : wait_region) {
    CHECK_SPAN(!(ContainsOpNamed(s, "pld.tile.put") || ContainsOpNamed(s, "pld.system.notify")), func->span_)
        << "SplitDeferredCompositeKernels: defer=True nested under control flow in '" << func->name_
        << "' is unsupported; keep the deferred composite as the straight-line body of its own "
           "pl.at(CORE_GROUP) task";
  }

  // Each sibling Function must own fresh parameter Vars. Sharing ``func->params_``
  // across push/wait/epi breaks print→parse RoundtripInstrument: structural_equal
  // sees one Var identity in multiple functions and reports an inconsistent mapping.
  auto make_fresh_params = [&]() {
    std::unordered_map<const Var*, ExprPtr> param_map;
    std::vector<VarPtr> fresh_params;
    fresh_params.reserve(func->params_.size());
    for (const auto& var : func->params_) {
      auto fresh = std::make_shared<Var>(var->name_hint_, var->GetType(), func->span_);
      fresh_params.push_back(fresh);
      param_map[var.get()] = fresh;
    }
    return std::make_pair(std::move(fresh_params), std::move(param_map));
  };

  auto clone_body = [&](std::vector<StmtPtr> stmts, const std::unordered_map<const Var*, ExprPtr>& param_map,
                        const std::vector<ExprPtr>& ret_values, bool empty_return) -> StmtPtr {
    if (empty_return) {
      stmts.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, func->span_));
    } else {
      stmts.push_back(std::make_shared<ReturnStmt>(ret_values, func->span_));
    }
    return DeepClone(MakeBody(std::move(stmts), func->span_), param_map).cloned_body;
  };

  const std::string push_name = ClaimUniqueFunctionName(func->name_ + "_push", reserved_names);
  const std::string wait_name = ClaimUniqueFunctionName(func->name_ + "_wait", reserved_names);

  {
    auto [fresh_params, param_map] = make_fresh_params();
    auto push_body = clone_body(std::move(push_stmts), param_map, {}, /*empty_return=*/true);
    plan.push_func = std::make_shared<Function>(
        push_name, std::move(fresh_params), func->param_directions_, std::vector<TypePtr>{}, push_body,
        func->span_, func->func_type_, func->level_, func->role_, CopyAttrsWithoutDeferredMarker(func));
  }

  {
    auto [fresh_params, param_map] = make_fresh_params();
    std::vector<StmtPtr> wait_stmts = setup;
    wait_stmts.insert(wait_stmts.end(), wait_region.begin(), wait_region.end());
    auto wait_body = clone_body(std::move(wait_stmts), param_map, {}, /*empty_return=*/true);
    plan.wait_func = std::make_shared<Function>(
        wait_name, std::move(fresh_params), func->param_directions_, std::vector<TypePtr>{}, wait_body,
        func->span_, func->func_type_, func->level_, func->role_, CopyAttrsKeepDeferredMarker(func));
  }

  if (!epi_stmts.empty() || !return_values.empty()) {
    auto [fresh_params, param_map] = make_fresh_params();
    std::vector<StmtPtr> epi_combined = setup;
    epi_combined.insert(epi_combined.end(), epi_stmts.begin(), epi_stmts.end());
    auto epi_body = clone_body(std::move(epi_combined), param_map, return_values,
                               /*empty_return=*/return_values.empty());
    const std::string epi_name = ClaimUniqueFunctionName(func->name_ + "_epi", reserved_names);
    plan.epi_func = std::make_shared<Function>(
        epi_name, std::move(fresh_params), func->param_directions_, func->return_types_, epi_body,
        func->span_, func->func_type_, func->level_, func->role_, CopyAttrsWithoutDeferredMarker(func));
  }

  // ExpandMixedKernel requires registration-only waiters.
  auto contract = outline_utils::DeferredWaitContractValidator::Validate(plan.wait_func->body_, func->span_);
  INTERNAL_CHECK_SPAN(contract.has_deferred_wait, func->span_)
      << "Internal error: split wait function for '" << func->name_ << "' lost defer_wait";

  return plan;
}

[[nodiscard]] TypePtr MakeSubmitReturnType(const std::vector<TypePtr>& callee_returns) {
  std::vector<TypePtr> elems = callee_returns;
  elems.push_back(std::make_shared<ScalarType>(DataType::TASK_ID));
  return std::make_shared<TupleType>(elems);
}

[[nodiscard]] std::shared_ptr<Submit> CloneSubmitWithCallee(const SubmitPtr& src, const GlobalVarPtr& callee,
                                                            std::vector<ExprPtr> deps,
                                                            const TypePtr& return_type) {
  return std::make_shared<Submit>(callee, src->args_, std::move(deps), src->kwargs_, src->attrs_, return_type,
                                  src->span_, src->core_num_, src->sync_start_, src->allow_early_resolve_,
                                  src->predicate_);
}

/// Promote a plain Call into a Submit so push→wait→epi can carry TaskId deps.
/// Outline now forces Submit for deferred waiters; this remains as defense for
/// any Call site that still reaches the rewriter.
[[nodiscard]] SubmitPtr PromoteCallToSubmit(const CallPtr& call, const TypePtr& return_type) {
  return std::make_shared<Submit>(call->op_, call->args_, /*deps=*/std::vector<ExprPtr>{}, call->kwargs_,
                                  call->attrs_, return_type, call->span_);
}

class OrchestrationCallerRewriter : public IRMutator {
 public:
  explicit OrchestrationCallerRewriter(const std::unordered_map<std::string, SplitPlan>& plans)
      : plans_(plans) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Common outline shape:
    //   ret = submit(name, args..., deps=[...])
    //   out_i = ret[i]
    //   tid = ret[N]
    // Rewrite into push → wait → epi submits with chained deps; bind outputs /
    // public tid from the epilogue (or wait) return.
    SubmitPtr submit = As<Submit>(op->value_);
    if (!submit) {
      auto call = As<Call>(op->value_);
      if (!call) return IRMutator::VisitStmt_(op);
      auto gv = As<GlobalVar>(call->op_);
      if (!gv) return IRMutator::VisitStmt_(op);
      auto it = plans_.find(gv->name_);
      if (it == plans_.end()) return IRMutator::VisitStmt_(op);
      submit = PromoteCallToSubmit(call, MakeSubmitReturnType(it->second.original_return_types));
      return RewriteBoundSubmit(op->var_, submit, it->second, op->span_);
    }
    auto gv = As<GlobalVar>(submit->op_);
    if (!gv) return IRMutator::VisitStmt_(op);
    auto it = plans_.find(gv->name_);
    if (it == plans_.end()) return IRMutator::VisitStmt_(op);
    return RewriteBoundSubmit(op->var_, submit, it->second, op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    SubmitPtr submit = As<Submit>(op->expr_);
    if (!submit) {
      auto call = As<Call>(op->expr_);
      if (!call) return IRMutator::VisitStmt_(op);
      auto gv = As<GlobalVar>(call->op_);
      if (!gv || plans_.count(gv->name_) == 0) return IRMutator::VisitStmt_(op);
      // Plain Call (no TaskId): promote to Submit and chain push→wait→epi deps.
      const SplitPlan& plan = plans_.at(gv->name_);
      submit = PromoteCallToSubmit(call, MakeSubmitReturnType(plan.original_return_types));
      return RewriteUnboundSubmit(submit, plan, op->span_);
    }

    auto gv = As<GlobalVar>(submit->op_);
    if (!gv || plans_.count(gv->name_) == 0) return IRMutator::VisitStmt_(op);
    return RewriteUnboundSubmit(submit, plans_.at(gv->name_), op->span_);
  }

 private:
  StmtPtr RewriteBoundSubmit(const VarPtr& ret_var, const SubmitPtr& submit, const SplitPlan& plan,
                             const Span& span) {
    auto push_gv = std::make_shared<GlobalVar>(plan.push_func->name_);
    auto wait_gv = std::make_shared<GlobalVar>(plan.wait_func->name_);
    GlobalVarPtr public_gv = wait_gv;
    if (plan.epi_func) {
      public_gv = std::make_shared<GlobalVar>(plan.epi_func->name_);
    }

    auto push_ret_ty = MakeSubmitReturnType({});
    auto wait_ret_ty = MakeSubmitReturnType({});
    auto public_ret_ty = MakeSubmitReturnType(plan.original_return_types);

    auto push_submit = CloneSubmitWithCallee(submit, push_gv, submit->deps_, push_ret_ty);
    auto push_ret = std::make_shared<Var>(ret_var->name_hint_ + "_push", push_ret_ty, span);
    auto push_tid = std::make_shared<Var>(ret_var->name_hint_ + "_push_tid",
                                          std::make_shared<ScalarType>(DataType::TASK_ID), span);

    auto wait_submit = CloneSubmitWithCallee(submit, wait_gv, std::vector<ExprPtr>{push_tid}, wait_ret_ty);
    auto wait_ret = std::make_shared<Var>(ret_var->name_hint_ + "_wait", wait_ret_ty, span);
    auto wait_tid = std::make_shared<Var>(ret_var->name_hint_ + "_wait_tid",
                                          std::make_shared<ScalarType>(DataType::TASK_ID), span);

    std::vector<StmtPtr> out;
    out.push_back(std::make_shared<AssignStmt>(push_ret, push_submit, span));
    out.push_back(
        std::make_shared<AssignStmt>(push_tid, std::make_shared<TupleGetItemExpr>(push_ret, 0, span), span));
    out.push_back(std::make_shared<AssignStmt>(wait_ret, wait_submit, span));
    out.push_back(
        std::make_shared<AssignStmt>(wait_tid, std::make_shared<TupleGetItemExpr>(wait_ret, 0, span), span));

    if (plan.epi_func) {
      auto epi_submit =
          CloneSubmitWithCallee(submit, public_gv, std::vector<ExprPtr>{wait_tid}, public_ret_ty);
      // Keep the original ret Var so subsequent TupleGetItem unpacking still binds.
      out.push_back(std::make_shared<AssignStmt>(ret_var, epi_submit, span));
    } else {
      // No epilogue: public tid is the wait tid. If the original returned only
      // TASK_ID, rebind the original Var to a Tuple{wait_tid}.
      if (plan.original_return_types.empty()) {
        auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{wait_tid}, span);
        out.push_back(std::make_shared<AssignStmt>(ret_var, tup, span));
      } else {
        // Unexpected: outputs without an epilogue region. Keep wait as the
        // public submit (outputs would have been produced in push — rare).
        auto wait_as_public =
            CloneSubmitWithCallee(submit, wait_gv, std::vector<ExprPtr>{push_tid}, public_ret_ty);
        out.push_back(std::make_shared<AssignStmt>(ret_var, wait_as_public, span));
      }
    }

    return MakeBody(std::move(out), span);
  }

  StmtPtr RewriteUnboundSubmit(const SubmitPtr& submit, const SplitPlan& plan, const Span& span) {
    // EvalStmt Submit without binding — synthesize tid temps for chaining.
    auto push_gv = std::make_shared<GlobalVar>(plan.push_func->name_);
    auto wait_gv = std::make_shared<GlobalVar>(plan.wait_func->name_);
    auto push_ret_ty = MakeSubmitReturnType({});
    auto wait_ret_ty = MakeSubmitReturnType({});
    auto push_ret = std::make_shared<Var>("split_push_ret", push_ret_ty, span);
    auto push_tid =
        std::make_shared<Var>("split_push_tid", std::make_shared<ScalarType>(DataType::TASK_ID), span);
    auto wait_ret = std::make_shared<Var>("split_wait_ret", wait_ret_ty, span);
    auto wait_tid =
        std::make_shared<Var>("split_wait_tid", std::make_shared<ScalarType>(DataType::TASK_ID), span);

    std::vector<StmtPtr> out;
    out.push_back(std::make_shared<AssignStmt>(
        push_ret, CloneSubmitWithCallee(submit, push_gv, submit->deps_, push_ret_ty), span));
    out.push_back(
        std::make_shared<AssignStmt>(push_tid, std::make_shared<TupleGetItemExpr>(push_ret, 0, span), span));
    out.push_back(std::make_shared<AssignStmt>(
        wait_ret, CloneSubmitWithCallee(submit, wait_gv, std::vector<ExprPtr>{push_tid}, wait_ret_ty), span));
    out.push_back(
        std::make_shared<AssignStmt>(wait_tid, std::make_shared<TupleGetItemExpr>(wait_ret, 0, span), span));
    if (plan.epi_func) {
      auto epi_gv = std::make_shared<GlobalVar>(plan.epi_func->name_);
      auto epi_ret_ty = MakeSubmitReturnType(plan.original_return_types);
      out.push_back(std::make_shared<EvalStmt>(
          CloneSubmitWithCallee(submit, epi_gv, std::vector<ExprPtr>{wait_tid}, epi_ret_ty), span));
    }
    return MakeBody(std::move(out), span);
  }

  const std::unordered_map<std::string, SplitPlan>& plans_;
};

ProgramPtr TransformSplitDeferredCompositeKernels(const ProgramPtr& program) {
  std::unordered_set<std::string> reserved_names;
  reserved_names.reserve(program->functions_.size() * 2);
  for (const auto& [gvar, func] : program->functions_) {
    if (func) reserved_names.insert(func->name_);
  }

  std::unordered_map<std::string, SplitPlan> plans;
  for (const auto& [gvar, func] : program->functions_) {
    if (!NeedsDeferredCompositeSplit(func)) continue;
    // The original name is replaced by phase siblings; free it from the
    // reserved set so a collision with an unrelated pre-existing `_push` still
    // forces a fresh suffix, while the removed original does not block reuse.
    reserved_names.erase(func->name_);
    auto plan = BuildSplitPlan(func, &reserved_names);
    plans.emplace(plan.original_name, std::move(plan));
  }
  if (plans.empty()) return program;

  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size() + plans.size() * 2);

  for (const auto& [gvar, func] : program->functions_) {
    auto it = plans.find(func->name_);
    if (it != plans.end()) {
      functions.push_back(it->second.push_func);
      functions.push_back(it->second.wait_func);
      if (it->second.epi_func) functions.push_back(it->second.epi_func);
      continue;
    }
    if (IsOrchestrationLike(func->func_type_) ||
        (func->role_.has_value() && *func->role_ == Role::Orchestrator)) {
      OrchestrationCallerRewriter rewriter(plans);
      functions.push_back(rewriter.VisitFunction(func));
    } else {
      functions.push_back(func);
    }
  }

  return std::make_shared<Program>(std::move(functions), program->name_, program->span_);
}

}  // namespace

Pass SplitDeferredCompositeKernels() {
  return CreateProgramPass(TransformSplitDeferredCompositeKernels, "SplitDeferredCompositeKernels",
                           kSplitDeferredCompositeKernelsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
