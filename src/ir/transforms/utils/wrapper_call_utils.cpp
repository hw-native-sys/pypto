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

#include "pypto/ir/transforms/utils/wrapper_call_utils.h"

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {

namespace {

/// Shared scaffold: visit every Call in the body, resolve its op via
/// `GlobalVar` lookup, invoke @p on_match for each resolved (call, callee)
/// pair. Returning `true` from @p on_match terminates the walk early.
class CallVisitor : public IRVisitor {
 public:
  using OnMatchFn = std::function<bool(const CallPtr&, const FunctionPtr&)>;

  CallVisitor(const ProgramPtr& program, OnMatchFn on_match)
      : program_(program), on_match_(std::move(on_match)) {}

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (stop_) return;
    if (auto gv = As<GlobalVar>(call->op_)) {
      if (auto callee = program_->GetFunction(gv->name_)) {
        if (on_match_(call, callee)) {
          stop_ = true;
          return;
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  const ProgramPtr& program_;
  OnMatchFn on_match_;
  bool stop_ = false;
};

}  // namespace

WrapperCallInfo FindFirstInnerCall(const FunctionPtr& wrapper, const ProgramPtr& program) {
  WrapperCallInfo info;
  if (!wrapper || !wrapper->body_ || !program) return info;
  CallVisitor visitor(program, [&](const CallPtr& call, const FunctionPtr& callee) {
    info.inner_call = call;
    info.inner_callee = callee;
    return true;  // first match wins; stop the walk
  });
  visitor.VisitStmt(wrapper->body_);
  return info;
}

GroupCalleeInfo FindGroupCallees(const FunctionPtr& group_func, const ProgramPtr& program) {
  GroupCalleeInfo info;
  if (!group_func || !group_func->body_ || !program) return info;
  // `aic_name` / `aiv_name` are first-match-per-type. `inner_call` is
  // first-match in source order regardless of type — this matches the
  // behavior of the original CalleeFinder in orchestration_codegen.cpp
  // and is what BuildWrapperReorderedParams expects (the call whose arg
  // order it reorders against). Group bodies emitted by ExpandMixedKernel
  // place AIC before AIV in source order, so the AIC call wins in practice.
  CallVisitor visitor(program, [&](const CallPtr& call, const FunctionPtr& callee) {
    if (callee->func_type_ == FunctionType::AIC && info.aic_name.empty()) {
      info.aic_name = callee->name_;
      if (!info.inner_call) {
        info.inner_call = call;
        info.inner_callee = callee;
      }
    } else if (callee->func_type_ == FunctionType::AIV && info.aiv_name.empty()) {
      info.aiv_name = callee->name_;
      if (!info.inner_call) {
        info.inner_call = call;
        info.inner_callee = callee;
      }
    } else if (callee->func_type_ == FunctionType::InCore && !info.inner_call) {
      info.inner_call = call;
      info.inner_callee = callee;
    }
    return false;  // collect all matches
  });
  visitor.VisitStmt(group_func->body_);
  return info;
}

std::vector<WrapperCallInfo> CollectInnerCalls(const FunctionPtr& wrapper, const ProgramPtr& program) {
  std::vector<WrapperCallInfo> result;
  if (!wrapper || !wrapper->body_ || !program) return result;
  CallVisitor visitor(program, [&](const CallPtr& call, const FunctionPtr& callee) {
    if (callee->func_type_ != FunctionType::Orchestration && callee->func_type_ != FunctionType::Opaque) {
      result.push_back({call, callee});
    }
    return false;
  });
  visitor.VisitStmt(wrapper->body_);
  return result;
}

namespace {

/// The function's own `param_directions_`, padded to `params_.size()` so the
/// result is always positionally indexable.
std::vector<ParamDirection> DeclaredDirections(const FunctionPtr& func) {
  std::vector<ParamDirection> declared = func->param_directions_;
  declared.resize(func->params_.size(), ParamDirection::In);
  return declared;
}

}  // namespace

std::unordered_map<const Function*, std::vector<ParamDirection>> ComputeWrapperEffectiveDirections(
    const ProgramPtr& program) {
  std::unordered_map<const Function*, std::vector<ParamDirection>> memo;
  if (!program) return memo;

  std::unordered_set<const Function*> visiting;

  std::function<std::vector<ParamDirection>(const FunctionPtr&)> compute_effective =
      [&](const FunctionPtr& func) -> std::vector<ParamDirection> {
    if (!func) return {};
    auto memo_it = memo.find(func.get());
    if (memo_it != memo.end()) return memo_it->second;

    // Cycle guard: fall back to declared directions if a recursion cycle exists.
    // Not memoized — the value is only valid for this in-progress recursion.
    if (!visiting.insert(func.get()).second) {
      return DeclaredDirections(func);
    }

    // Seed from the declaration so the merge below is monotone: an inner call
    // can reveal that a param is written (In → Out → InOut), never that a
    // declared writer is read-only. Without this a wrapper that writes a param
    // through a builtin rather than an inner call — or that has no body / no
    // inner calls at all — would infer a bogus all-In vector, and
    // DeriveCallDirections would write that loss back into the signature.
    std::vector<ParamDirection> directions = DeclaredDirections(func);

    std::unordered_map<const Var*, size_t> param_to_index;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      param_to_index[func->params_[i].get()] = i;
    }

    for (const auto& [inner_call, inner_callee] : CollectInnerCalls(func, program)) {
      const auto& inner_args = inner_call->args_;
      std::vector<ParamDirection> inner_dirs = IsWrapperType(inner_callee->func_type_)
                                                   ? compute_effective(inner_callee)
                                                   : inner_callee->param_directions_;
      for (size_t arg_idx = 0; arg_idx < inner_args.size() && arg_idx < inner_dirs.size(); ++arg_idx) {
        auto var = AsVarLike(inner_args[arg_idx]);
        if (!var) continue;
        auto it = param_to_index.find(var.get());
        if (it == param_to_index.end()) continue;
        ParamDirection d = inner_dirs[arg_idx];
        ParamDirection& merged = directions[it->second];
        if (d == ParamDirection::InOut || (d == ParamDirection::Out && merged == ParamDirection::In)) {
          merged = d;
        }
      }
    }

    visiting.erase(func.get());
    memo.emplace(func.get(), directions);
    return directions;
  };

  for (const auto& [gvar, func] : program->functions_) {
    if (func && IsWrapperType(func->func_type_)) compute_effective(func);
  }
  // compute_effective only ever runs on wrappers — seeded here, or reached via
  // the nested Group/Spmd recursion — so every memo key is a Group/Spmd func.
  return memo;
}

}  // namespace ir
}  // namespace pypto
