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
#include <map>
#include <memory>
#include <set>
#include <string>
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
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsTensorAllReduce(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.allreduce");
}

[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

class NameCollector : public IRVisitor {
 public:
  std::set<std::string> names;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op && !op->name_hint_.empty()) names.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "pld.tensor.alloc_window_buffer")) {
      auto name = op->GetKwarg<std::string>("name");
      if (!name.empty()) names.insert(name);
    }
    IRVisitor::VisitExpr_(op);
  }
};

class AllReduceSignalSynthesizer : public IRMutator {
 public:
  AllReduceSignalSynthesizer(std::set<std::string>* used_names, int64_t* next_id)
      : used_names_(used_names), next_id_(next_id) {}

  [[nodiscard]] bool modified() const { return modified_; }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    if (IsTensorAllReduce(op)) {
      CheckAllReduceCall(op);
      CHECK_SPAN(false, op->span_)
          << "pld.tensor.allreduce must be a direct assignment, expression statement, or return value before "
             "allreduce signal synthesis.";
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsTensorAllReduce(call)) return IRMutator::VisitStmt_(op);
    CheckAllReduceCall(call);
    if (call->args_.size() == 2) {
      for (const auto& arg : call->args_) VisitExpr(arg);
      return op;
    }

    auto [prefix, signal] = MakeSignalBinding(call->span_);
    auto target = VisitExpr(call->args_[0]);
    auto rewritten_call = MakeAllReduceCall(call, target, signal);
    auto result = MutableCopy(op);
    result->value_ = rewritten_call;

    prefix.push_back(result);
    modified_ = true;
    return SeqStmts::Flatten(std::move(prefix), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!IsTensorAllReduce(call)) return IRMutator::VisitStmt_(op);
    CheckAllReduceCall(call);
    if (call->args_.size() == 2) {
      for (const auto& arg : call->args_) VisitExpr(arg);
      return op;
    }

    auto [prefix, signal] = MakeSignalBinding(call->span_);
    auto target = VisitExpr(call->args_[0]);
    auto rewritten_call = MakeAllReduceCall(call, target, signal);
    std::vector<StmtPtr> stmts = std::move(prefix);
    stmts.push_back(std::make_shared<EvalStmt>(rewritten_call, op->span_, op->leading_comments_));
    modified_ = true;
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<StmtPtr> prelude;
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool changed = false;

    for (std::size_t i = 0; i < op->value_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
      auto call = As<Call>(op->value_[i]);
      if (!IsTensorAllReduce(call)) {
        auto new_value = VisitExpr(op->value_[i]);
        new_values.push_back(new_value);
        if (new_value.get() != op->value_[i].get()) changed = true;
        continue;
      }

      CheckAllReduceCall(call);
      auto target = VisitExpr(call->args_[0]);
      ExprPtr signal;
      if (call->args_.size() == 1) {
        auto [prefix, synthesized_signal] = MakeSignalBinding(call->span_);
        for (auto& stmt : prefix) prelude.push_back(std::move(stmt));
        signal = synthesized_signal;
      } else {
        signal = VisitExpr(call->args_[1]);
      }

      auto rewritten_call = MakeAllReduceCall(call, target, signal);
      auto result_var = std::make_shared<Var>(FreshGeneratedName("__allreduce_result_"),
                                              rewritten_call->GetType(), call->span_);
      prelude.push_back(std::make_shared<AssignStmt>(result_var, rewritten_call, call->span_));
      new_values.push_back(result_var);
      changed = true;
    }

    if (!changed) return op;
    auto new_return = MutableCopy(op);
    new_return->value_ = std::move(new_values);
    prelude.push_back(new_return);
    modified_ = true;
    return SeqStmts::Flatten(std::move(prelude), op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    ++repeating_scope_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --repeating_scope_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    ++repeating_scope_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --repeating_scope_depth_;
    return result;
  }

 private:
  void CheckAllReduceCall(const CallPtr& call) const {
    CHECK_SPAN(call->args_.size() == 1 || call->args_.size() == 2, call->span_)
        << "pld.tensor.allreduce expects target[, signal], got " << call->args_.size()
        << " positional arguments";
    CHECK_SPAN(repeating_scope_depth_ == 0, call->span_)
        << "pld.tensor.allreduce is not supported inside a for/while loop. "
           "The signal protocol is single-use and cannot reuse a signal across dynamic invocations.";
  }

  struct SignalNames {
    std::string world_size_name;
    std::string buf_name;
    std::string signal_name;
  };

  [[nodiscard]] SignalNames FreshSignalNames() {
    while (true) {
      auto suffix = std::to_string((*next_id_)++);
      auto world_size_name = "__allreduce_signal_world_size_" + suffix;
      auto buf_name = "__allreduce_signal_buf_" + suffix;
      auto signal_name = "__allreduce_signal_" + suffix;
      if (used_names_->count(world_size_name) != 0 || used_names_->count(buf_name) != 0 ||
          used_names_->count(signal_name) != 0) {
        continue;
      }
      used_names_->insert(world_size_name);
      used_names_->insert(buf_name);
      used_names_->insert(signal_name);
      return {world_size_name, buf_name, signal_name};
    }
  }

  [[nodiscard]] std::string FreshGeneratedName(const std::string& prefix) {
    while (true) {
      auto name = prefix + std::to_string((*next_id_)++);
      if (used_names_->count(name) != 0) continue;
      used_names_->insert(name);
      return name;
    }
  }

  [[nodiscard]] std::pair<std::vector<StmtPtr>, VarPtr> MakeSignalBinding(const Span& span) {
    auto names = FreshSignalNames();

    auto world_size_call = OpRegistry::GetInstance().Create("pld.system.world_size", {}, span);
    auto world_size_var = std::make_shared<Var>(names.world_size_name, world_size_call->GetType(), span);
    auto world_size_assign = std::make_shared<AssignStmt>(world_size_var, world_size_call, span);

    auto four = std::make_shared<ConstInt>(4, DataType::INT64, span);
    auto size_bytes = MakeMul(world_size_var, four, span);

    std::vector<std::pair<std::string, std::any>> alloc_kwargs = {{"name", names.buf_name}};
    auto alloc_call =
        OpRegistry::GetInstance().Create("pld.tensor.alloc_window_buffer", {size_bytes}, alloc_kwargs, span);
    auto buf_var = std::make_shared<Var>(names.buf_name, alloc_call->GetType(), span);
    auto buf_assign = std::make_shared<AssignStmt>(buf_var, alloc_call, span);

    auto one = std::make_shared<ConstInt>(1, DataType::INT64, span);
    auto signal_shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{world_size_var, one}, span);
    std::vector<std::pair<std::string, std::any>> window_kwargs = {{"dtype", DataType::INT32}};
    auto window_call =
        OpRegistry::GetInstance().Create("pld.tensor.window", {buf_var, signal_shape}, window_kwargs, span);
    auto signal_var = std::make_shared<Var>(names.signal_name, window_call->GetType(), span);
    auto signal_assign = std::make_shared<AssignStmt>(signal_var, window_call, span);

    return {{world_size_assign, buf_assign, signal_assign}, signal_var};
  }

  [[nodiscard]] CallPtr MakeAllReduceCall(const CallPtr& call, const ExprPtr& target, const ExprPtr& signal) {
    return OpRegistry::GetInstance().Create("pld.tensor.allreduce", {target, signal}, call->kwargs_,
                                            call->span_);
  }

  std::set<std::string>* used_names_;
  int64_t* next_id_;
  int repeating_scope_depth_ = 0;
  bool modified_ = false;
};

}  // namespace

namespace pass {

Pass SynthesizeAllReduceSignals() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    NameCollector name_collector;
    name_collector.VisitProgram(program);
    int64_t next_signal_id = 0;

    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool modified = false;
    for (const auto& [gvar, func] : program->functions_) {
      if (!IsHostOrch(func)) {
        new_functions[gvar] = func;
        continue;
      }

      AllReduceSignalSynthesizer synthesizer(&name_collector.names, &next_signal_id);
      auto new_body = synthesizer.VisitStmt(func->body_);
      if (!synthesizer.modified()) {
        new_functions[gvar] = func;
        continue;
      }

      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      new_functions[gvar] = new_func;
      modified = true;
    }

    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "SynthesizeAllReduceSignals", kSynthesizeAllReduceSignalsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
