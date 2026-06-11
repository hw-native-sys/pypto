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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] bool IsTensorAllReduce(const CallPtr& call) {
  return call && call->op_ && call->op_->name_ == "pld.tensor.allreduce";
}

[[nodiscard]] WindowBufferPtr GetWindowBuffer(const ExprPtr& expr) {
  auto dist_type = As<DistributedTensorType>(expr->GetType());
  INTERNAL_CHECK_SPAN(dist_type, expr->span_)
      << "LowerHostTensorCollectives: allreduce args must be DistributedTensorType";
  INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), expr->span_)
      << "LowerHostTensorCollectives: allreduce args must have materialized WindowBuffer back-references";
  return *dist_type->window_buffer_;
}

[[nodiscard]] bool ScopeContainsSlot(const CommDomainScopeStmtPtr& scope, const WindowBufferPtr& wb) {
  for (const auto& slot : scope->slots_) {
    if (slot.get() == wb.get()) return true;
  }
  return false;
}

[[nodiscard]] CommDomainScopeStmtPtr FindScopeForAllReduce(
    const std::vector<CommDomainScopeStmtPtr>& scope_stack, const WindowBufferPtr& data_wb,
    const WindowBufferPtr& signal_wb) {
  for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
    const auto& scope = *it;
    if (ScopeContainsSlot(scope, data_wb) && ScopeContainsSlot(scope, signal_wb)) {
      return scope;
    }
  }
  return nullptr;
}

void CheckStaticSignalCapacity(const CallPtr& call, size_t required_slots) {
  auto signal_type = As<DistributedTensorType>(call->args_[1]->GetType());
  INTERNAL_CHECK_SPAN(signal_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce signal must be DistributedTensorType";
  CHECK_SPAN(signal_type->shape_.size() == 1, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce signal must be rank-1";
  if (signal_type->shape_.empty()) return;
  auto extent = As<ConstInt>(signal_type->shape_[0]);
  if (!extent) return;
  CHECK_SPAN(extent->value_ >= static_cast<int64_t>(required_slots), call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce signal shape[0] (" << extent->value_
      << ") must be at least the participating device count (" << required_slots << ")";
}

[[nodiscard]] CallPtr MakeBuiltinCall(const CallPtr& call, const ExprPtr& device) {
  auto src_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(src_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce src must be DistributedTensorType";
  auto op_value = call->GetKwarg<int>("op");
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  auto builtin =
      OpRegistry::GetInstance().CreateInternal("builtin.tensor.allreduce", call->args_, kwargs, call->span_);
  std::vector<std::pair<std::string, std::any>> attrs = {
      {kAttrDevice, device},
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  attrs = WithArgDirectionsAttr(std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
  return std::make_shared<Call>(builtin->op_, builtin->args_, builtin->kwargs_, std::move(attrs),
                                builtin->GetType(), builtin->span_);
}

class LowerHostTensorCollectivesMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    scope_stack_.push_back(op);
    auto new_body = VisitStmt(op->body_);
    scope_stack_.pop_back();
    if (new_body.get() == op->body_.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = new_body;
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (IsTensorAllReduce(call)) {
      return LowerAllReduce(call, op->span_, op->leading_comments_);
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsTensorAllReduce(call)) {
      return IRMutator::VisitStmt_(op);
    }
    std::vector<StmtPtr> stmts;
    stmts.push_back(LowerAllReduce(call, op->span_, op->leading_comments_));
    stmts.push_back(std::make_shared<AssignStmt>(op->var_, call->args_[0], op->span_));
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  StmtPtr LowerAllReduce(const CallPtr& call, const Span& span,
                         const std::vector<std::string>& leading_comments) {
    INTERNAL_CHECK_SPAN(!scope_stack_.empty(), call->span_)
        << "LowerHostTensorCollectives: pld.tensor.allreduce must appear inside a CommDomainScopeStmt";
    auto data_wb = GetWindowBuffer(call->args_[0]);
    auto signal_wb = GetWindowBuffer(call->args_[1]);
    auto scope = FindScopeForAllReduce(scope_stack_, data_wb, signal_wb);
    INTERNAL_CHECK_SPAN(scope, call->span_)
        << "LowerHostTensorCollectives: allreduce data and signal must resolve to the same comm-domain scope";

    if (!scope->devices_.empty()) {
      CheckStaticSignalCapacity(call, scope->devices_.size());
      std::vector<StmtPtr> stmts;
      stmts.reserve(scope->devices_.size());
      for (auto device : scope->devices_) {
        auto device_expr = std::make_shared<ConstInt>(device, DataType::INT64, call->span_);
        stmts.push_back(std::make_shared<EvalStmt>(MakeBuiltinCall(call, device_expr), call->span_));
      }
      return std::make_shared<SeqStmts>(std::move(stmts), span, leading_comments);
    }

    auto loop_var = std::make_shared<Var>("r", std::make_shared<ScalarType>(DataType::INT64), call->span_);
    auto zero = std::make_shared<ConstInt>(0, DataType::INT64, call->span_);
    auto one = std::make_shared<ConstInt>(1, DataType::INT64, call->span_);
    auto stop = OpRegistry::GetInstance().Create("pld.system.world_size", {}, call->span_);
    auto body = std::make_shared<EvalStmt>(MakeBuiltinCall(call, loop_var), call->span_);
    return std::make_shared<ForStmt>(loop_var, zero, stop, one, std::vector<IterArgPtr>{}, body,
                                     std::vector<VarPtr>{}, span, ForKind::Sequential, std::nullopt,
                                     std::vector<std::pair<std::string, std::any>>{}, leading_comments);
  }

  std::vector<CommDomainScopeStmtPtr> scope_stack_;
};

FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsHostOrch(func)) return func;
  LowerHostTensorCollectivesMutator mutator;
  return mutator.VisitFunction(func);
}

ProgramPtr TransformProgram(const ProgramPtr& program) {
  bool modified = false;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  for (const auto& [gvar, func] : program->functions_) {
    auto new_func = TransformFunction(func);
    new_functions[gvar] = new_func;
    if (new_func.get() != func.get()) modified = true;
  }
  if (!modified) return program;
  return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass LowerHostTensorCollectives() {
  return CreateProgramPass(TransformProgram, "LowerHostTensorCollectives",
                           kLowerHostTensorCollectivesProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
