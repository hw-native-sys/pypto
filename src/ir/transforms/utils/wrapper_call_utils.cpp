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
#include <deque>
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
  std::unordered_map<const Function*, std::vector<ParamDirection>> effective;
  if (!program) return effective;

  // Index every wrapper and walk each body exactly once.
  std::vector<FunctionPtr> wrappers;
  std::unordered_map<const Function*, FunctionPtr> by_ptr;
  std::unordered_map<const Function*, std::vector<WrapperCallInfo>> inner_calls;
  for (const auto& [gvar, func] : program->functions_) {
    if (!func || !IsWrapperType(func->func_type_)) continue;
    wrappers.push_back(func);
    by_ptr.emplace(func.get(), func);
    // Seed at the declaration: the transfer below only ever promotes, so the
    // solution is the least fixed point *above* what each signature declares.
    // A wrapper that writes a param through a builtin rather than an inner
    // call — or that has no body, or no inner calls at all — therefore keeps
    // what it declares instead of collapsing to a bogus all-In vector.
    effective.emplace(func.get(), DeclaredDirections(func));
    inner_calls.emplace(func.get(), CollectInnerCalls(func, program));
  }
  if (wrappers.empty()) return effective;

  // Reverse edges: wrapper callee -> the wrappers that call it. When a callee's
  // directions grow, only its callers can grow as a result.
  std::unordered_map<const Function*, std::vector<const Function*>> callers;
  for (const auto& wrapper : wrappers) {
    for (const auto& info : inner_calls.at(wrapper.get())) {
      const Function* callee = info.inner_callee.get();
      if (by_ptr.count(callee) != 0) callers[callee].push_back(wrapper.get());
    }
  }

  // Merge every inner call's directions onto @p wrapper's params. Returns true
  // when at least one param was promoted.
  auto merge_once = [&](const FunctionPtr& wrapper) -> bool {
    std::vector<ParamDirection>& directions = effective.at(wrapper.get());
    bool promoted = false;
    for (const auto& [inner_call, inner_callee] : inner_calls.at(wrapper.get())) {
      // Copy: for a self-recursive wrapper this would otherwise alias
      // `directions` while we mutate it.
      const std::vector<ParamDirection> inner_dirs = IsWrapperType(inner_callee->func_type_)
                                                         ? effective.at(inner_callee.get())
                                                         : inner_callee->param_directions_;
      const auto& inner_args = inner_call->args_;
      for (size_t arg_idx = 0; arg_idx < inner_args.size() && arg_idx < inner_dirs.size(); ++arg_idx) {
        auto var = AsVarLike(inner_args[arg_idx]);
        if (!var) continue;
        // Params are few (single digits); a linear scan beats hashing here.
        for (size_t p = 0; p < wrapper->params_.size(); ++p) {
          if (wrapper->params_[p].get() != var.get()) continue;
          const ParamDirection d = inner_dirs[arg_idx];
          ParamDirection& merged = directions[p];
          if (d == ParamDirection::InOut && merged != ParamDirection::InOut) {
            merged = d;
            promoted = true;
          } else if (d == ParamDirection::Out && merged == ParamDirection::In) {
            merged = d;
            promoted = true;
          }
          break;
        }
      }
    }
    return promoted;
  };

  // Monotone worklist to the least fixed point. Unlike a recursive walk with a
  // cycle guard, this is independent of the order wrappers are visited in: a
  // mutually recursive pair (A -> B -> A) converges to the same directions
  // whichever one is seeded first. Each promotion moves one param one step up
  // the In < Out < InOut lattice, so the loop runs at most
  // 2 * (total wrapper params) times.
  std::deque<const Function*> work;
  std::unordered_set<const Function*> queued;
  for (const auto& wrapper : wrappers) {
    work.push_back(wrapper.get());
    queued.insert(wrapper.get());
  }
  while (!work.empty()) {
    const Function* func = work.front();
    work.pop_front();
    queued.erase(func);
    if (!merge_once(by_ptr.at(func))) continue;
    auto it = callers.find(func);
    if (it == callers.end()) continue;
    for (const Function* caller : it->second) {
      if (queued.insert(caller).second) work.push_back(caller);
    }
  }
  return effective;
}

}  // namespace ir
}  // namespace pypto
