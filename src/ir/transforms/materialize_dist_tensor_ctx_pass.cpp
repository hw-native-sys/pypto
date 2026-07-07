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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

struct FunctionCtxPlan {
  std::vector<size_t> dist_param_indices;
  std::unordered_map<const Var*, VarPtr> param_to_ctx;
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

[[nodiscard]] FunctionCtxPlan BuildFunctionCtxPlan(const FunctionPtr& func) {
  FunctionCtxPlan plan;
  if (!func) return plan;
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

class DistParamAliasCollector : public IRVisitor {
 public:
  explicit DistParamAliasCollector(const FunctionCtxPlan* plan) : plan_(plan) {}
  std::unordered_map<const Var*, VarPtr> alias_to_ctx;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_ && As<DistributedTensorType>(op->var_->GetType())) {
      if (auto src = AsVarLike(op->value_)) {
        if (auto ctx = LookupCtx(src.get())) {
          alias_to_ctx[op->var_.get()] = ctx;
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  VarPtr LookupCtx(const Var* var) const {
    if (!plan_) return nullptr;
    auto param_it = plan_->param_to_ctx.find(var);
    if (param_it != plan_->param_to_ctx.end()) return param_it->second;
    auto alias_it = alias_to_ctx.find(var);
    if (alias_it != alias_to_ctx.end()) return alias_it->second;
    return nullptr;
  }

  const FunctionCtxPlan* plan_;
};

class MaterializeDistTensorCtxMutator : public IRMutator {
 public:
  MaterializeDistTensorCtxMutator(ProgramPtr program,
                                  const std::unordered_map<std::string, FunctionCtxPlan>& plans)
      : program_(std::move(program)), plans_(plans) {}

  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    current_plan_ = nullptr;
    current_alias_to_ctx_.clear();
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
      DistParamAliasCollector alias_collector(current_plan_);
      alias_collector.VisitStmt(func->body_);
      current_alias_to_ctx_ = std::move(alias_collector.alias_to_ctx);
    }
    auto new_body = VisitStmt(func->body_);
    INTERNAL_CHECK_SPAN(pending_prefix_.empty(), func->span_)
        << "MaterializeDistTensorCtx: generated get_comm_ctx prefix was not attached to a statement";
    auto out = func;
    if (new_body.get() != func->body_.get()) {
      out = std::make_shared<Function>(
          func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
          func->func_type_, func->level_, func->role_, func->attrs_, func->requires_runtime_binding_);
    }
    current_plan_ = nullptr;
    current_alias_to_ctx_.clear();
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
                                    submit->sync_start_, submit->allow_early_resolve_);
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

  ExprPtr GetCtxForArg(const ExprPtr& arg, const Span& span) {
    if (!IsDistTensor(arg)) return nullptr;
    if (current_plan_) {
      auto var = AsVarLike(arg);
      if (var) {
        auto it = current_plan_->param_to_ctx.find(var.get());
        if (it != current_plan_->param_to_ctx.end()) return it->second;
        auto alias_it = current_alias_to_ctx_.find(var.get());
        if (alias_it != current_alias_to_ctx_.end()) return alias_it->second;
      }
    }
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
  const FunctionCtxPlan* current_plan_ = nullptr;
  std::unordered_map<const Var*, VarPtr> current_alias_to_ctx_;
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
                                    func->attrs_, func->requires_runtime_binding_);
}

[[nodiscard]] ProgramPtr TransformProgram(const ProgramPtr& program) {
  std::unordered_map<std::string, FunctionCtxPlan> plans;
  for (const auto& [gvar, func] : program->functions_) {
    if (!HasDistTensorParam(func)) continue;
    auto plan = BuildFunctionCtxPlan(func);
    if (!plan.dist_param_indices.empty()) plans.emplace(func->name_, std::move(plan));
  }
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

}  // namespace

namespace pass {

Pass MaterializeDistTensorCtx() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };
  return CreateProgramPass(pass_func, "MaterializeDistTensorCtx", kMaterializeDistTensorCtxProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
