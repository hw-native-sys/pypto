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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DIST_TENSOR_ALIAS_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DIST_TENSOR_ALIAS_UTILS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dist_tensor_alias {

// Shared lineage for engine analysis before domain grouping and CommCtx
// materialization afterwards. Identity values are supplied by the caller:
// allocation/parameter Vars for analysis, or context SSA Vars for lowering.
struct FunctionCtxPlan {
  std::vector<size_t> dist_param_indices;
  std::unordered_map<const Var*, VarPtr> param_to_ctx;
  std::vector<std::optional<size_t>> returned_param_indices;
};

// Return the yield that terminates a structured control-flow branch.  The
// branch body may be wrapped in a SeqStmts or in one of the transparent scope
// statements inserted by earlier passes.  This intentionally does not change
// IfStmt lowering: it only lets this pass recover an already-existing context
// when both branches yield the same DistributedTensor backing/context.
[[nodiscard]] inline YieldStmtPtr FindBranchYield(const StmtPtr& body) {
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
[[nodiscard]] inline std::optional<size_t> CtxInheritingArgIndex(const CallPtr& call) {
  if (!call || !call->op_) return std::nullopt;
  if (auto writeback = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) {
    return writeback;
  }
  if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
    return 0;
  }
  return std::nullopt;
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
    if (plan_) {
      auto param_it = plan_->param_to_ctx.find(var);
      if (param_it != plan_->param_to_ctx.end()) return param_it->second;
    }
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

}  // namespace dist_tensor_alias
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DIST_TENSOR_ALIAS_UTILS_H_
