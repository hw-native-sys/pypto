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

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

struct FunctionCtxPlan {
  std::vector<size_t> dist_param_indices;
  std::unordered_map<const Var*, VarPtr> param_to_ctx;
  std::vector<std::optional<size_t>> returned_param_indices;
};

[[nodiscard]] bool IsDistTensor(const ExprPtr& expr) {
  return expr && As<DistributedTensorType>(expr->GetType());
}

[[nodiscard]] bool HasDistTensorParam(const FunctionPtr& func) {
  if (!func) return false;
  for (const auto& param : func->params_) {
    if (As<DistributedTensorType>(param->GetType())) return true;
  }
  return false;
}

[[nodiscard]] std::string MakeCtxParamBaseName(const VarPtr& dist_param) {
  return auto_name::GetBaseName(dist_param->name_hint_) + "_ctx";
}

class LocalNameCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> names;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      names.insert(op->var_->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarPtr& op) override {
    if (op) {
      names.insert(op->name_hint_);
    }
    IRVisitor::VisitExpr_(op);
  }
};

[[nodiscard]] std::string MakeUniqueName(const std::string& base_name,
                                         std::unordered_set<std::string>* used_names) {
  std::string name = base_name;
  size_t suffix = 1;
  while (used_names->count(name) != 0) {
    name = base_name + "_" + std::to_string(suffix);
    ++suffix;
  }
  used_names->insert(name);
  return name;
}

// Return the yield that terminates a structured control-flow branch.  The
// branch body may be wrapped in a SeqStmts or in one of the transparent scope
// statements inserted by earlier passes.  This intentionally does not change
// IfStmt lowering: it only lets this pass recover an already-existing context
// when both branches yield the same DistributedTensor backing/context.
[[nodiscard]] YieldStmtPtr FindBranchYield(const StmtPtr& body) {
  if (!body) return nullptr;
  if (auto yield = As<YieldStmt>(body)) return yield;
  if (auto seq = As<SeqStmts>(body)) {
    for (auto it = seq->stmts_.rbegin(); it != seq->stmts_.rend(); ++it) {
      if (auto yield = FindBranchYield(*it)) return yield;
    }
  }
  if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(body)) {
    return FindBranchYield(scope->body_);
  }
  return nullptr;
}

/// Host orchestration is the one place where get_comm_ctx is a real runtime
/// query: host codegen resolves it from the window's per-rank device context.
/// Chip orchestration and all lower/device functions must have the query
/// eliminated here, after the matching explicit CommCtx parameter is known.
[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] FunctionCtxPlan BuildFunctionCtxPlan(const FunctionPtr& func) {
  FunctionCtxPlan plan;
  if (!func) return plan;
  // SIMT helpers borrow only the rank-local GM pointer. Their verified scalar
  // subset cannot query communication contexts or perform distributed ops,
  // and SimtLaunchStmt does not carry the runtime kernel's CommCtx ABI.
  if (func->func_type_ == FunctionType::SIMT) return plan;
  plan.returned_param_indices = return_lineage::ExplicitReturnedParamIndices(func);
  std::unordered_set<std::string> used_names;
  for (const auto& param : func->params_) {
    used_names.insert(param->name_hint_);
  }
  LocalNameCollector collector;
  collector.VisitStmt(func->body_);
  used_names.insert(collector.names.begin(), collector.names.end());
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    if (!As<DistributedTensorType>(param->GetType())) continue;
    auto ctx = std::make_shared<Var>(MakeUniqueName(MakeCtxParamBaseName(param), &used_names),
                                     GetCommCtxType(), param->span_);
    plan.dist_param_indices.push_back(i);
    plan.param_to_ctx[param.get()] = ctx;
  }
  return plan;
}

[[nodiscard]] std::vector<ArgDirection> AppendCtxArgDirections(const std::vector<ArgDirection>& old_dirs,
                                                               size_t ctx_count) {
  auto dirs = old_dirs;
  dirs.reserve(dirs.size() + ctx_count);
  for (size_t i = 0; i < ctx_count; ++i) {
    dirs.push_back(ArgDirection::Scalar);
  }
  return dirs;
}

/// Index of the argument whose communication context @p call inherits.
///
/// Two builtin families bind a fresh SSA var to a DistributedTensor that
/// already exists, so the result keeps the original's window — and with it its
/// context:
///   - output-side writebacks (`tile.store`, `tensor.assemble`, ...), which
///     write into an existing tensor and return it;
///   - zero-copy buffer-aliasing views (`tensor.view`, `tile.slice`,
///     `tensor.reshape`, ...), whose deduced result type propagates
///     `DistributedTensorType::window_buffer_` straight from `args[0]`.
///
/// Neither reaches the callee-return machinery below: their op is an `Op`, not
/// a `GlobalVar`, so without this the context would look unresolvable.
[[nodiscard]] std::optional<size_t> CtxInheritingArgIndex(const CallPtr& call) {
  if (!call || !call->op_) return std::nullopt;
  if (auto writeback = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) {
    return writeback;
  }
  if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
    return 0;
  }
  return std::nullopt;
}

/// Describe a DistributedTensor value for a user-facing diagnostic.
[[nodiscard]] std::string DescribeDistValue(const ExprPtr& expr) {
  if (auto var = AsVarLike(expr)) return "'" + var->name_hint_ + "'";
  return "a DistributedTensor value";
}

/// Map each result position of a `Call` / `Submit` to the caller-side CommCtx
/// of the callee parameter that result writes back.
///
/// `Call` and `Submit` share one result-position map: a `Submit`'s type is
/// `TupleType([*<callee returns>, Scalar[TASK_ID]])`, so the trailing TASK_ID
/// simply falls off the end of the returned vector.  Positions that write back
/// a *runtime-allocated* Out param — one the caller never passed, per the
/// `Submit` args-prefix invariant (`pass-submit-awareness.md` rule 5) — have no
/// caller-side argument and resolve to null.
///
/// @param lookup_arg_ctx resolves the ctx of one caller-side argument
/// @return one entry per callee return position (null = unresolved)
template <typename LookupArgCtx>
[[nodiscard]] std::vector<VarPtr> ResolveResultCtxs(
    const ExprPtr& expr, const ProgramPtr& program,
    const std::unordered_map<std::string, FunctionCtxPlan>& plans, const LookupArgCtx& lookup_arg_ctx) {
  OpPtr op;
  const std::vector<ExprPtr>* args = nullptr;
  if (auto call = As<Call>(expr)) {
    op = call->op_;
    args = &call->args_;
  } else if (auto submit = As<Submit>(expr)) {
    op = submit->op_;
    args = &submit->args_;
  } else {
    return {};
  }

  auto gvar = As<GlobalVar>(op);
  if (!gvar || !program) return {};
  auto callee = program->GetFunction(gvar->name_);
  if (!callee) return {};
  auto plan_it = plans.find(callee->name_);
  if (plan_it == plans.end()) return {};

  std::vector<VarPtr> result(plan_it->second.returned_param_indices.size());
  for (size_t result_idx = 0; result_idx < result.size(); ++result_idx) {
    const auto& param_idx = plan_it->second.returned_param_indices[result_idx];
    if (!param_idx || *param_idx >= args->size() || *param_idx >= callee->params_.size()) continue;
    if (!As<DistributedTensorType>(callee->params_[*param_idx]->GetType())) continue;
    result[result_idx] = lookup_arg_ctx((*args)[*param_idx]);
  }
  return result;
}

class DistParamAliasCollector : public IRVisitor {
 public:
  DistParamAliasCollector(ProgramPtr program, const std::unordered_map<std::string, FunctionCtxPlan>* plans,
                          const FunctionCtxPlan* plan)
      : program_(std::move(program)), plans_(plans), plan_(plan) {}
  std::unordered_map<const Var*, VarPtr> alias_to_ctx;
  std::unordered_map<const Var*, std::vector<VarPtr>> tuple_to_ctx;

  VarPtr ResolveCtx(const ExprPtr& expr) const { return LookupCtx(expr); }

 protected:
  void VisitStmt_(const SimtLaunchStmtPtr& op) override {
    // A writable GM launch result is a new SSA name for the same allocation.
    // Follow the explicit result-to-argument mapping so later context queries
    // and dispatches retain the window's context, including successive launches.
    for (size_t i = 0; i < op->results_.size(); ++i) {
      const auto& result = op->results_[i];
      if (!As<DistributedTensorType>(result->GetType())) continue;
      if (auto ctx = LookupCtx(op->args_[op->result_param_indices_[i]])) {
        alias_to_ctx[result.get()] = std::move(ctx);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_ && As<DistributedTensorType>(op->var_->GetType())) {
      if (auto ctx = LookupCtx(op->value_)) {
        alias_to_ctx[op->var_.get()] = ctx;
      }
    } else if (op && op->var_ && As<TupleType>(op->var_->GetType())) {
      auto returned_ctxs = ReturnedCtxs(op->value_);
      if (!returned_ctxs.empty()) tuple_to_ctx[op->var_.get()] = std::move(returned_ctxs);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit the branches first so aliases created inside each branch are
    // available when the merged return var is inspected below.
    IRVisitor::VisitStmt_(op);
    if (!op || !op->else_body_.has_value()) return;
    auto then_yield = FindBranchYield(op->then_body_);
    auto else_yield = FindBranchYield(*op->else_body_);
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      if (!return_var || !As<DistributedTensorType>(return_var->GetType())) continue;
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) continue;

      auto then_ctx = LookupCtx(then_yield->value_[i]);
      auto else_ctx = LookupCtx(else_yield->value_[i]);
      if (!then_ctx || !else_ctx) continue;
      CHECK_SPAN(then_ctx.get() == else_ctx.get(), op->span_)
          << "Assigning a different DistributedTensor in each branch of an `if` is not supported: '"
          << return_var->name_hint_
          << "' would take its data pointer from one allocation and its communication context from "
             "another. Assign a single DistributedTensor before the `if`, and branch on the data "
             "read from it instead (see GitHub issue #2027).";
      alias_to_ctx[return_var.get()] = std::move(then_ctx);
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    RecordLoopCarries(op->iter_args_, op->return_vars_);
    IRVisitor::VisitStmt_(op);
    ValidateLoopCarries(op->iter_args_, op->body_, op->span_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    RecordLoopCarries(op->iter_args_, op->return_vars_);
    IRVisitor::VisitStmt_(op);
    ValidateLoopCarries(op->iter_args_, op->body_, op->span_);
  }

 private:
  VarPtr LookupVarCtx(const Var* var) const {
    if (!plan_) return nullptr;
    auto param_it = plan_->param_to_ctx.find(var);
    if (param_it != plan_->param_to_ctx.end()) return param_it->second;
    auto alias_it = alias_to_ctx.find(var);
    if (alias_it != alias_to_ctx.end()) return alias_it->second;
    return nullptr;
  }

  VarPtr LookupCtx(const ExprPtr& expr) const {
    if (auto var = AsVarLike(expr)) return LookupVarCtx(var.get());

    if (auto get_item = As<TupleGetItemExpr>(expr); get_item && get_item->index_ >= 0) {
      auto tuple_var = AsVarLike(get_item->tuple_);
      const auto index = static_cast<size_t>(get_item->index_);
      if (tuple_var) {
        auto tuple_it = tuple_to_ctx.find(tuple_var.get());
        if (tuple_it == tuple_to_ctx.end() || index >= tuple_it->second.size()) return nullptr;
        return tuple_it->second[index];
      }
      auto returned_ctxs = ReturnedCtxs(get_item->tuple_);
      return index < returned_ctxs.size() ? returned_ctxs[index] : nullptr;
    }

    if (auto call = As<Call>(expr)) {
      if (auto inherited = CtxInheritingArgIndex(call)) {
        return LookupCtx(call->args_[*inherited]);
      }
    }

    auto returned_ctxs = ReturnedCtxs(expr);
    return returned_ctxs.size() == 1 ? returned_ctxs[0] : nullptr;
  }

  std::vector<VarPtr> ReturnedCtxs(const ExprPtr& expr) const {
    if (auto tuple_var = AsVarLike(expr)) {
      auto it = tuple_to_ctx.find(tuple_var.get());
      return it == tuple_to_ctx.end() ? std::vector<VarPtr>{} : it->second;
    }
    if (!plans_) return {};
    return ResolveResultCtxs(expr, program_, *plans_, [this](const ExprPtr& arg) { return LookupCtx(arg); });
  }

  /// A loop carry is seeded from its init value *before* the body is visited so
  /// a self-carry (`data = self.comm(data)`) can resolve at all.  That seed is
  /// only sound while the value yielded back into the carry still names the
  /// same context: a loop that rebinds the carry to a different
  /// DistributedTensor would take its data pointer from one allocation and its
  /// communication context from another — the same unsupported program the
  /// `IfStmt` merge above diagnoses.
  ///
  /// A yield this pass cannot trace at all leaves the seed in place; that value
  /// came from an op with no modelled ctx lineage, and rejecting it here would
  /// turn programs that compile today into hard errors.
  void ValidateLoopCarries(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body, const Span& span) {
    auto yield = FindBranchYield(body);
    if (!yield) return;
    for (size_t i = 0; i < iter_args.size() && i < yield->value_.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      if (!iter_arg) continue;
      auto seeded = alias_to_ctx.find(iter_arg.get());
      if (seeded == alias_to_ctx.end()) continue;
      auto yield_ctx = LookupCtx(yield->value_[i]);
      if (!yield_ctx) continue;
      CHECK_SPAN(yield_ctx.get() == seeded->second.get(), span)
          << "Rebinding loop-carried DistributedTensor '" << iter_arg->name_hint_
          << "' to a different DistributedTensor inside the loop is not supported: it would enter the "
             "loop with the communication context of its initial value and leave with another. Carry a "
             "single DistributedTensor through the loop, and vary the data read from it instead (see "
             "GitHub issue #2027).";
    }
  }

  void RecordLoopCarries(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      if (!iter_arg || !As<DistributedTensorType>(iter_arg->GetType())) continue;
      auto ctx = LookupCtx(iter_arg->initValue_);
      if (!ctx) continue;
      alias_to_ctx[iter_arg.get()] = ctx;
      if (i < return_vars.size() && As<DistributedTensorType>(return_vars[i]->GetType())) {
        alias_to_ctx[return_vars[i].get()] = std::move(ctx);
      }
    }
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, FunctionCtxPlan>* plans_;
  const FunctionCtxPlan* plan_;
};

class MaterializeDistTensorCtxMutator : public IRMutator {
 public:
  MaterializeDistTensorCtxMutator(ProgramPtr program,
                                  const std::unordered_map<std::string, FunctionCtxPlan>& plans)
      : program_(std::move(program)), plans_(plans) {}

  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    current_func_ = func;
    current_plan_ = nullptr;
    current_alias_to_ctx_.clear();
    current_tuple_to_ctx_.clear();
    pending_prefix_.clear();
    can_emit_prefix_ = false;
    local_ctx_names_.clear();
    for (const auto& param : func->params_) {
      local_ctx_names_.insert(param->name_hint_);
    }
    LocalNameCollector collector;
    collector.VisitStmt(func->body_);
    local_ctx_names_.insert(collector.names.begin(), collector.names.end());
    auto it = plans_.find(func->name_);
    if (it != plans_.end()) {
      current_plan_ = &it->second;
      DistParamAliasCollector alias_collector(program_, &plans_, current_plan_);
      alias_collector.VisitStmt(func->body_);
      current_alias_to_ctx_ = std::move(alias_collector.alias_to_ctx);
      current_tuple_to_ctx_ = std::move(alias_collector.tuple_to_ctx);
    }
    auto new_body = VisitStmt(func->body_);
    INTERNAL_CHECK_SPAN(pending_prefix_.empty(), func->span_)
        << "MaterializeDistTensorCtx: generated get_comm_ctx prefix was not attached to a statement";
    auto out = func;
    if (new_body.get() != func->body_.get()) {
      out =
          std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                     new_body, func->span_, func->func_type_, func->level_, func->role_,
                                     func->attrs_, func->requires_runtime_binding_, func->ir_stage_);
    }
    current_func_ = nullptr;
    current_plan_ = nullptr;
    current_alias_to_ctx_.clear();
    current_tuple_to_ctx_.clear();
    return out;
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    pending_prefix_.clear();
    auto value = VisitExprWithPrefixSynthesis(op->value_);
    if (pending_prefix_.empty() && value.get() == op->value_.get()) return op;

    auto assign = op;
    if (value.get() != op->value_.get()) {
      assign = std::make_shared<AssignStmt>(op->var_, value, op->span_, op->leading_comments_);
    }
    if (pending_prefix_.empty()) return assign;

    std::vector<StmtPtr> stmts = std::move(pending_prefix_);
    pending_prefix_.clear();
    stmts.push_back(assign);
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    pending_prefix_.clear();
    auto expr = VisitExprWithPrefixSynthesis(op->expr_);
    if (pending_prefix_.empty() && expr.get() == op->expr_.get()) return op;

    auto eval = op;
    if (expr.get() != op->expr_.get()) {
      eval = std::make_shared<EvalStmt>(expr, op->span_, op->leading_comments_);
    }
    if (pending_prefix_.empty()) return eval;

    std::vector<StmtPtr> stmts = std::move(pending_prefix_);
    pending_prefix_.clear();
    stmts.push_back(eval);
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    pending_prefix_.clear();
    bool changed = false;
    std::vector<ExprPtr> values;
    values.reserve(op->value_.size());
    for (const auto& value : op->value_) {
      auto new_value = VisitExprWithPrefixSynthesis(value);
      changed = changed || new_value.get() != value.get();
      values.push_back(std::move(new_value));
    }
    if (pending_prefix_.empty() && !changed) return op;

    auto ret =
        changed ? std::make_shared<ReturnStmt>(std::move(values), op->span_, op->leading_comments_) : op;
    if (pending_prefix_.empty()) return ret;

    std::vector<StmtPtr> stmts = std::move(pending_prefix_);
    pending_prefix_.clear();
    stmts.push_back(ret);
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;

    if (IsOp(call, "pld.system.get_comm_ctx")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
          << "MaterializeDistTensorCtx: pld.system.get_comm_ctx expects exactly one argument";
      if (IsHostOrch(current_func_)) return call;

      auto ctx = LookupExistingCtx(call->args_[0]);
      CHECK_SPAN(ctx, call->span_)
          << "Cannot resolve pld.system.get_comm_ctx(" << DescribeDistValue(call->args_[0]) << ") in '"
          << (current_func_ ? current_func_->name_ : std::string("<unknown>"))
          << "'. Outside host orchestration the query has no runtime representation, so the context must "
             "be reachable from a parameter of this function. Take "
          << DescribeDistValue(call->args_[0])
          << " as a parameter (or return it from a callee that does) instead of producing it locally.";
      // get_comm_ctx is an IR-level query, but it has no device-side runtime
      // representation.  Replace it with the explicit context SSA value so
      // PTO/InCore codegen never has to rediscover a tensor-to-context edge.
      return ctx;
    }

    auto callee = ResolveCallee(call->op_);
    if (!callee) return call;
    auto plan_it = plans_.find(callee->name_);
    if (plan_it == plans_.end()) return call;

    std::vector<ExprPtr> new_args = call->args_;
    AppendCtxArgs(plan_it->second, call->args_, call->span_, &new_args);

    auto attrs = call->attrs_;
    if (call->HasArgDirections()) {
      attrs = WithArgDirectionsAttr(
          std::move(attrs),
          AppendCtxArgDirections(call->GetArgDirections(), plan_it->second.dist_param_indices.size()));
    }
    return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, std::move(attrs),
                                  call->GetType(), call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    if (!submit) return base;
    auto callee = ResolveCallee(submit->op_);
    if (!callee) return submit;
    auto plan_it = plans_.find(callee->name_);
    if (plan_it == plans_.end()) return submit;

    std::vector<ExprPtr> new_args = submit->args_;
    AppendCtxArgs(plan_it->second, submit->args_, submit->span_, &new_args);

    auto attrs = submit->attrs_;
    if (submit->HasArgDirections()) {
      attrs = WithArgDirectionsAttr(
          std::move(attrs),
          AppendCtxArgDirections(submit->GetArgDirections(), plan_it->second.dist_param_indices.size()));
    }
    return std::make_shared<Submit>(submit->op_, std::move(new_args), submit->deps_, submit->kwargs_,
                                    std::move(attrs), submit->GetType(), submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

 private:
  class PrefixSynthesisScope {
   public:
    explicit PrefixSynthesisScope(MaterializeDistTensorCtxMutator* owner)
        : owner_(owner), previous_(owner->can_emit_prefix_) {
      owner_->can_emit_prefix_ = true;
    }
    ~PrefixSynthesisScope() { owner_->can_emit_prefix_ = previous_; }

    PrefixSynthesisScope(const PrefixSynthesisScope&) = delete;
    PrefixSynthesisScope& operator=(const PrefixSynthesisScope&) = delete;

   private:
    MaterializeDistTensorCtxMutator* owner_;
    bool previous_;
  };

  ExprPtr VisitExprWithPrefixSynthesis(const ExprPtr& expr) {
    PrefixSynthesisScope scope(this);
    return VisitExpr(expr);
  }

  FunctionPtr ResolveCallee(const OpPtr& op) const {
    auto gvar = As<GlobalVar>(op);
    if (!gvar || !program_) return nullptr;
    return program_->GetFunction(gvar->name_);
  }

  std::vector<VarPtr> LookupReturnedCtxs(const ExprPtr& call_like) const {
    return ResolveResultCtxs(call_like, program_, plans_,
                             [this](const ExprPtr& arg) { return LookupExistingCtx(arg); });
  }

  VarPtr LookupExistingCtx(const ExprPtr& arg) const {
    if (!IsDistTensor(arg) || !current_plan_) return nullptr;
    if (auto var = AsVarLike(arg)) {
      auto param_it = current_plan_->param_to_ctx.find(var.get());
      if (param_it != current_plan_->param_to_ctx.end()) return param_it->second;
      auto alias_it = current_alias_to_ctx_.find(var.get());
      if (alias_it != current_alias_to_ctx_.end()) return alias_it->second;
      return nullptr;
    }

    if (auto get_item = As<TupleGetItemExpr>(arg); get_item && get_item->index_ >= 0) {
      const auto index = static_cast<size_t>(get_item->index_);
      if (auto tuple_var = AsVarLike(get_item->tuple_)) {
        auto tuple_it = current_tuple_to_ctx_.find(tuple_var.get());
        if (tuple_it == current_tuple_to_ctx_.end()) return nullptr;
        return index < tuple_it->second.size() ? tuple_it->second[index] : nullptr;
      }
      auto returned_ctxs = LookupReturnedCtxs(get_item->tuple_);
      return index < returned_ctxs.size() ? returned_ctxs[index] : nullptr;
    }

    if (auto call = As<Call>(arg)) {
      if (auto inherited = CtxInheritingArgIndex(call)) {
        return LookupExistingCtx(call->args_[*inherited]);
      }
    }

    auto returned_ctxs = LookupReturnedCtxs(arg);
    return returned_ctxs.size() == 1 ? returned_ctxs[0] : nullptr;
  }

  ExprPtr GetCtxForArg(const ExprPtr& arg, const Span& span) {
    if (!IsDistTensor(arg)) return nullptr;
    if (auto ctx = LookupExistingCtx(arg)) return ctx;
    // Only host orchestration can recover a context through the runtime query.
    // Chip orchestration and device functions must carry an explicit context
    // SSA value; leaving a synthesized get_comm_ctx in those functions would
    // give device codegen an operation with no runtime representation.
    CHECK_SPAN(IsHostOrch(current_func_), span)
        << "Cannot determine the communication context of DistributedTensor " << DescribeDistValue(arg)
        << " passed from '" << (current_func_ ? current_func_->name_ : std::string("<unknown>"))
        << "'. Only host orchestration can query a context at runtime; a chip-orchestration or device "
           "function must reach its DistributedTensor from one of its own parameters. Take "
        << DescribeDistValue(arg)
        << " as a parameter of this function (or return it from a callee that does) instead of producing "
           "it locally.";
    INTERNAL_CHECK_SPAN(can_emit_prefix_, span)
        << "MaterializeDistTensorCtx: cannot synthesize get_comm_ctx prefix in this expression context";
    std::string base_name = "dist";
    if (auto var = AsVarLike(arg)) {
      base_name = auto_name::GetBaseName(var->name_hint_);
    }
    auto ctx =
        std::make_shared<Var>(MakeUniqueName(base_name + "_ctx", &local_ctx_names_), GetCommCtxType(), span);
    auto call = OpRegistry::GetInstance().Create("pld.system.get_comm_ctx", {arg}, {}, span);
    pending_prefix_.push_back(std::make_shared<AssignStmt>(ctx, call, span));
    return ctx;
  }

  void AppendCtxArgs(const FunctionCtxPlan& plan, const std::vector<ExprPtr>& old_args, const Span& span,
                     std::vector<ExprPtr>* new_args) {
    for (auto param_idx : plan.dist_param_indices) {
      INTERNAL_CHECK_SPAN(param_idx < old_args.size(), span)
          << "MaterializeDistTensorCtx: call-like expression does not provide DistributedTensor arg at "
          << "callee param index " << param_idx;
      auto ctx = GetCtxForArg(old_args[param_idx], span);
      INTERNAL_CHECK_SPAN(ctx, span)
          << "MaterializeDistTensorCtx: expected DistributedTensor arg at callee param index " << param_idx;
      new_args->push_back(ctx);
    }
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, FunctionCtxPlan>& plans_;
  FunctionPtr current_func_;
  const FunctionCtxPlan* current_plan_ = nullptr;
  std::unordered_map<const Var*, VarPtr> current_alias_to_ctx_;
  std::unordered_map<const Var*, std::vector<VarPtr>> current_tuple_to_ctx_;
  std::vector<StmtPtr> pending_prefix_;
  std::unordered_set<std::string> local_ctx_names_;
  bool can_emit_prefix_ = false;
};

[[nodiscard]] FunctionPtr ExtendFunctionSignature(const FunctionPtr& func, const FunctionCtxPlan& plan) {
  if (plan.dist_param_indices.empty()) return func;

  std::vector<VarPtr> params = func->params_;
  std::vector<ParamDirection> dirs = func->param_directions_;
  params.reserve(params.size() + plan.dist_param_indices.size());
  dirs.reserve(dirs.size() + plan.dist_param_indices.size());
  for (auto param_idx : plan.dist_param_indices) {
    auto it = plan.param_to_ctx.find(func->params_[param_idx].get());
    INTERNAL_CHECK(it != plan.param_to_ctx.end())
        << "MaterializeDistTensorCtx: missing ctx param for " << func->params_[param_idx]->name_hint_;
    params.push_back(it->second);
    dirs.push_back(ParamDirection::In);
  }
  return std::make_shared<Function>(func->name_, std::move(params), std::move(dirs), func->return_types_,
                                    func->body_, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_, func->ir_stage_);
}

// Trace Host provenance along context forwarding edges. Each node/edge is
// visited once; shared device functions can accept both Host and Device windows.
// Reject only parameters used by peer-MTE operations, not unrelated HBM args.
class HostPeerAccessChecker : public IRVisitor {
 public:
  HostPeerAccessChecker(ProgramPtr program, const std::unordered_map<std::string, FunctionCtxPlan>& plans)
      : program_(std::move(program)), plans_(plans) {}

  void Check() {
    for (const auto& [gvar, func] : program_->functions_) {
      auto it = plans_.find(func->name_);
      DistParamAliasCollector aliases(program_, &plans_, it == plans_.end() ? nullptr : &it->second);
      aliases.VisitStmt(func->body_);
      aliases_ = &aliases;
      VisitStmt(func->body_);
    }
    aliases_ = nullptr;
    std::unordered_set<const Var*> visited;
    while (!host_nodes_.empty()) {
      const Var* node = host_nodes_.back();
      host_nodes_.pop_back();
      if (!visited.insert(node).second) continue;
      auto bad = peer_uses_.find(node);
      CHECK_SPAN(bad == peer_uses_.end(), node->span_)
          << "Host communication windows require URMA for peer access; " << bad->second
          << " uses peer MTE and cannot access memory_source='host'.";
      auto edge = edges_.find(node);
      if (edge != edges_.end()) {
        host_nodes_.insert(host_nodes_.end(), edge->second.begin(), edge->second.end());
      }
    }
  }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    CheckCall(call->op_, call->args_);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    CheckCall(submit->op_, submit->args_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  const Var* Resolve(const ExprPtr& expr) {
    if (auto type = As<DistributedTensorType>(expr->GetType()); type && type->window_buffer_) {
      const auto& window = *type->window_buffer_;
      if (window->memory_source_ == "host") host_nodes_.push_back(window.get());
      return window.get();
    }
    auto ctx = aliases_->ResolveCtx(expr);
    return ctx.get();
  }

  void CheckCall(const OpPtr& op, const std::vector<ExprPtr>& args) {
    if (auto global = As<GlobalVar>(op)) {
      auto callee = program_->GetFunction(global->name_);
      auto plan = plans_.find(global->name_);
      if (!callee || plan == plans_.end()) return;
      for (size_t index : plan->second.dist_param_indices) {
        if (index >= args.size()) continue;
        if (auto source = Resolve(args[index])) {
          edges_[source].push_back(plan->second.param_to_ctx.at(callee->params_[index].get()).get());
        }
      }
      return;
    }
    if (!op || op->name_.rfind("pld.", 0) != 0) return;
    if (IsOp(op, "pld.tensor.window") || IsOp(op, "pld.tensor.alloc_window_buffer") ||
        IsOp(op, "pld.system.get_comm_ctx") || IsOp(op, "pld.system.rank") || IsOp(op, "pld.system.nranks") ||
        IsOp(op, "pld.system.world_size") || IsOp(op, "pld.system.wait") ||
        IsOp(op, "pld.system.defer_wait") || IsOp(op, "pld.tensor.put_async") ||
        IsOp(op, "pld.tensor.get_async"))
      return;
    std::optional<size_t> remote_arg;
    if (IsOp(op, "pld.tensor.get") || IsOp(op, "pld.tile.get")) remote_arg = 2;
    if (IsOp(op, "pld.tensor.put") || IsOp(op, "pld.tile.put") || IsOp(op, "pld.tile.remote_load") ||
        IsOp(op, "pld.system.notify"))
      remote_arg = 0;
    if (IsOp(op, "pld.tensor.remote_store") || IsOp(op, "pld.tile.remote_store")) remote_arg = 1;
    for (size_t i = 0; i < args.size(); ++i) {
      if (remote_arg && i != *remote_arg) continue;
      if (auto source = Resolve(args[i])) peer_uses_.emplace(source, op->name_);
    }
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, FunctionCtxPlan>& plans_;
  const DistParamAliasCollector* aliases_ = nullptr;
  std::unordered_map<const Var*, std::vector<const Var*>> edges_;
  std::unordered_map<const Var*, std::string> peer_uses_;
  std::vector<const Var*> host_nodes_;
};

[[nodiscard]] ProgramPtr TransformProgram(const ProgramPtr& program) {
  std::unordered_map<std::string, FunctionCtxPlan> plans;
  for (const auto& [gvar, func] : program->functions_) {
    if (!HasDistTensorParam(func)) continue;
    auto plan = BuildFunctionCtxPlan(func);
    if (!plan.dist_param_indices.empty()) plans.emplace(func->name_, std::move(plan));
  }
  HostPeerAccessChecker(program, plans).Check();
  if (plans.empty()) return program;

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> extended;
  for (const auto& [gvar, func] : program->functions_) {
    auto it = plans.find(func->name_);
    extended[gvar] = (it == plans.end()) ? func : ExtendFunctionSignature(func, it->second);
  }
  auto with_signatures = std::make_shared<Program>(std::move(extended), program->name_, program->span_);

  MaterializeDistTensorCtxMutator mutator(with_signatures, plans);
  return mutator.VisitProgram(with_signatures);
}

// ============================================================================
// DistTensorCtxMaterialized property verifier
// ============================================================================

/// Flags every `pld.system.get_comm_ctx` left outside host orchestration.
///
/// The pass eliminates the query by construction, but only for functions it
/// visits: a Program in which no function declares a DistributedTensor
/// parameter is returned untouched, and a context that never traces back to a
/// parameter cannot be resolved at all. Either way the query would reach device
/// codegen, which has no runtime representation for it. Checking the invariant
/// independently turns that into a diagnostic instead of unusable generated
/// code.
class GetCommCtxCallChecker : public IRVisitor {
 public:
  GetCommCtxCallChecker(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (!op || !op->op_ || !IsOp(op, "pld.system.get_comm_ctx")) return;
    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "DistTensorCtxMaterialized", 0,
        "Function '" + func_name_ +
            "' still calls pld.system.get_comm_ctx. Only host orchestration may query a communication "
            "context at runtime; everywhere else MaterializeDistTensorCtx must have replaced it with the "
            "explicit CommCtx parameter, and device codegen cannot lower the query.",
        op->span_);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

class DistTensorCtxMaterializedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "DistTensorCtxMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_ || IsHostOrch(func)) continue;
      GetCommCtxCallChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateDistTensorCtxMaterializedPropertyVerifier() {
  return std::make_shared<DistTensorCtxMaterializedPropertyVerifierImpl>();
}

namespace pass {

Pass MaterializeDistTensorCtx() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };
  return CreateProgramPass(pass_func, "MaterializeDistTensorCtx", kMaterializeDistTensorCtxProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
