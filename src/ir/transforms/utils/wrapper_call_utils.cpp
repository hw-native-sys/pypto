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

#include <functional>
#include <utility>
#include <vector>

#include "pypto/ir/kind_traits.h"
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

}  // namespace ir
}  // namespace pypto
