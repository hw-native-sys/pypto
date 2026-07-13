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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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

[[nodiscard]] WindowBufferPtr GetWindowBuffer(const ExprPtr& expr, const char* context) {
  auto dist_type = As<DistributedTensorType>(expr->GetType());
  INTERNAL_CHECK_SPAN(dist_type, expr->span_)
      << "LowerHostTensorCollectives: " << context << " must be DistributedTensorType";
  INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), expr->span_)
      << "LowerHostTensorCollectives: " << context << " must have materialized WindowBuffer back-references";
  return *dist_type->window_buffer_;
}

[[nodiscard]] VarPtr MintDistributedResultVar(const VarPtr& old_var, const ExprPtr& src) {
  auto lhs_type = As<DistributedTensorType>(old_var->GetType());
  INTERNAL_CHECK_SPAN(lhs_type, old_var->span_)
      << "LowerHostTensorCollectives: collective result Var should have DistributedTensorType";
  auto src_type = As<DistributedTensorType>(src->GetType());
  INTERNAL_CHECK_SPAN(src_type && src_type->window_buffer_.has_value(), old_var->span_)
      << "LowerHostTensorCollectives: collective alias source must carry a materialized WindowBuffer";
  auto new_type =
      std::make_shared<const DistributedTensorType>(lhs_type->shape_, lhs_type->dtype_, lhs_type->memref_,
                                                    lhs_type->tensor_view_, src_type->window_buffer_);
  return std::make_shared<Var>(old_var->name_hint_, new_type, old_var->span_);
}

[[nodiscard]] bool ScopeContainsSlot(const CommDomainScopeStmtPtr& scope, const WindowBufferPtr& wb) {
  for (const auto& slot : scope->slots_) {
    if (slot.get() == wb.get()) return true;
  }
  return false;
}

[[nodiscard]] CommDomainScopeStmtPtr FindScopeForBuffers(
    const std::vector<CommDomainScopeStmtPtr>& scope_stack, const std::vector<WindowBufferPtr>& buffers) {
  INTERNAL_CHECK(!buffers.empty()) << "LowerHostTensorCollectives: scope lookup needs at least one buffer";
  for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
    const auto& scope = *it;
    bool all_present = true;
    for (const auto& wb : buffers) {
      if (!ScopeContainsSlot(scope, wb)) {
        all_present = false;
        break;
      }
    }
    if (all_present) return scope;
  }
  return nullptr;
}

void CheckStaticSignalCapacity(const CallPtr& call, const ExprPtr& signal_expr, size_t required_slots) {
  auto signal_type = As<DistributedTensorType>(signal_expr->GetType());
  INTERNAL_CHECK_SPAN(signal_type, call->span_)
      << "LowerHostTensorCollectives: collective signal must be DistributedTensorType";
  CHECK_SPAN(signal_type->shape_.size() == 1 || signal_type->shape_.size() == 2, call->span_)
      << "LowerHostTensorCollectives: collective signal must be rank-1 [world_size] "
         "or rank-2 [world_size, 1]";
  if (signal_type->shape_.empty()) return;
  if (signal_type->shape_.size() == 2) {
    auto second_extent = As<ConstInt>(signal_type->shape_[1]);
    CHECK_SPAN(second_extent, call->span_)
        << "LowerHostTensorCollectives: collective rank-2 signal shape[1] must be the constant 1";
    CHECK_SPAN(second_extent->value_ == 1, call->span_)
        << "LowerHostTensorCollectives: collective rank-2 signal shape[1] must be 1, got "
        << second_extent->value_;
  }
  auto extent = As<ConstInt>(signal_type->shape_[0]);
  if (!extent) return;
  CHECK_SPAN(extent->value_ >= static_cast<int64_t>(required_slots), call->span_)
      << "LowerHostTensorCollectives: collective signal shape[0] (" << extent->value_
      << ") must be at least the participating device count (" << required_slots << ")";
}

[[nodiscard]] CallPtr MakeBuiltinCallWithAttrs(const std::string& builtin_name, const CallPtr& call,
                                               const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                                               const ExprPtr& device,
                                               std::vector<std::pair<std::string, std::any>> attrs,
                                               std::vector<ArgDirection> arg_directions) {
  auto builtin = OpRegistry::GetInstance().CreateInternal(builtin_name, args, kwargs, call->span_);
  attrs.emplace_back(kAttrDevice, device);
  attrs = WithArgDirectionsAttr(std::move(attrs), std::move(arg_directions));
  return std::make_shared<Call>(builtin->op_, builtin->args_, builtin->kwargs_, std::move(attrs),
                                builtin->GetType(), builtin->span_);
}

[[nodiscard]] CallPtr MakeBuiltinAllReduce(const CallPtr& call, const ExprPtr& device) {
  auto src_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(src_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce src must be DistributedTensorType";
  auto op_value = call->GetKwarg<int>("op");
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  return MakeBuiltinCallWithAttrs("builtin.tensor.allreduce", call, call->args_, kwargs, device,
                                  std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinBarrier(const CallPtr& call, const ExprPtr& device) {
  return MakeBuiltinCallWithAttrs("builtin.tensor.barrier", call, call->args_, {}, device, {},
                                  {ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinBroadcast(const CallPtr& call, const ExprPtr& device) {
  auto target_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(target_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.broadcast target must be DistributedTensorType";
  auto root_value = call->GetKwarg<int>("root");
  std::vector<std::pair<std::string, std::any>> kwargs = {{"root", root_value},
                                                          {"dtype", target_type->dtype_}};
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"root", root_value},
      {"dtype", target_type->dtype_},
  };
  return MakeBuiltinCallWithAttrs("builtin.tensor.broadcast", call, call->args_, kwargs, device,
                                  std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinReduceScatter(const CallPtr& call, const ExprPtr& device) {
  auto target_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(target_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.reduce_scatter target must be DistributedTensorType";
  auto op_value = call->GetKwarg<int>("op");
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"op", op_value},
      {"dtype", target_type->dtype_},
  };
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"op", op_value},
      {"dtype", target_type->dtype_},
  };
  return MakeBuiltinCallWithAttrs("builtin.tensor.reduce_scatter", call, call->args_, kwargs, device,
                                  std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinAllGather(const CallPtr& call, const ExprPtr& device) {
  // Staged allgather: each rank's chunk is pre-staged in the shared window by
  // publish_step.  The allgather AIV kernel requires concurrent cross-chip
  // dispatch (TNOTIFY / TWAIT / TLOAD), but LowerHostTensorCollectives emits
  // per-device builtins sequentially in a world_size loop.  A barrier on the
  // signal suffices to synchronise visibility of the pre-staged data;
  // consume_step uses pld.tile.remote_load to gather from all peers.
  auto signal = call->args_[1];
  return MakeBuiltinCallWithAttrs("builtin.tensor.barrier", call, {signal}, {}, device, {},
                                  {ArgDirection::InOut});
}

struct HostCollectiveRule {
  const char* pld_name;
  using MakeBuiltinFn = std::function<CallPtr(const CallPtr&, const ExprPtr&)>;
  using ScopeBuffersFn = std::function<std::vector<WindowBufferPtr>(const CallPtr&)>;
  using SignalExprFn = std::function<ExprPtr(const CallPtr&)>;
  using AliasSourceFn = std::function<std::optional<ExprPtr>(const CallPtr&)>;
  MakeBuiltinFn make_builtin;
  ScopeBuffersFn scope_buffers;
  SignalExprFn signal_expr;
  AliasSourceFn alias_source;
};

[[nodiscard]] const HostCollectiveRule* LookupHostCollectiveRule(const std::string& op_name) {
  static const HostCollectiveRule kRules[] = {
      {
          "pld.tensor.allreduce",
          &MakeBuiltinAllReduce,
          [](const CallPtr& call) {
            return std::vector<WindowBufferPtr>{GetWindowBuffer(call->args_[0], "allreduce src"),
                                                GetWindowBuffer(call->args_[1], "allreduce signal")};
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.barrier",
          &MakeBuiltinBarrier,
          [](const CallPtr& call) {
            return std::vector<WindowBufferPtr>{GetWindowBuffer(call->args_[0], "barrier signal")};
          },
          [](const CallPtr& call) { return call->args_[0]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.broadcast",
          &MakeBuiltinBroadcast,
          [](const CallPtr& call) {
            return std::vector<WindowBufferPtr>{GetWindowBuffer(call->args_[0], "broadcast target"),
                                                GetWindowBuffer(call->args_[1], "broadcast signal")};
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.reduce_scatter",
          &MakeBuiltinReduceScatter,
          [](const CallPtr& call) {
            return std::vector<WindowBufferPtr>{GetWindowBuffer(call->args_[0], "reduce_scatter target"),
                                                GetWindowBuffer(call->args_[1], "reduce_scatter signal")};
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.allgather",
          &MakeBuiltinAllGather,
          [](const CallPtr& call) {
            return std::vector<WindowBufferPtr>{
                GetWindowBuffer(call->args_[0], "allgather target"),
                GetWindowBuffer(call->args_[1], "allgather signal"),
            };
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
  };
  for (const auto& rule : kRules) {
    if (op_name == rule.pld_name) return &rule;
  }
  return nullptr;
}

[[nodiscard]] bool IsHostTensorCollective(const CallPtr& call) {
  return call && call->op_ && LookupHostCollectiveRule(call->op_->name_) != nullptr;
}

StmtPtr EmitPerDeviceBuiltinCalls(const CallPtr& call, const HostCollectiveRule& rule,
                                  const CommDomainScopeStmtPtr& scope, const Span& span,
                                  const std::vector<std::string>& leading_comments) {
  if (!scope->devices_.empty()) {
    CheckStaticSignalCapacity(call, rule.signal_expr(call), scope->devices_.size());
    std::vector<StmtPtr> stmts;
    stmts.reserve(scope->devices_.size());
    for (auto device : scope->devices_) {
      auto device_expr = std::make_shared<ConstInt>(device, DataType::INT64, call->span_);
      stmts.push_back(std::make_shared<EvalStmt>(rule.make_builtin(call, device_expr), call->span_));
    }
    return std::make_shared<SeqStmts>(std::move(stmts), span, leading_comments);
  }

  auto loop_var = std::make_shared<Var>("r", std::make_shared<ScalarType>(DataType::INT64), call->span_);
  auto zero = std::make_shared<ConstInt>(0, DataType::INT64, call->span_);
  auto one = std::make_shared<ConstInt>(1, DataType::INT64, call->span_);
  auto stop = OpRegistry::GetInstance().Create("pld.system.world_size", {}, call->span_);
  auto body = std::make_shared<EvalStmt>(rule.make_builtin(call, loop_var), call->span_);
  return std::make_shared<ForStmt>(loop_var, zero, stop, one, std::vector<IterArgPtr>{}, body,
                                   std::vector<VarPtr>{}, span, ForKind::Sequential,
                                   std::vector<std::pair<std::string, std::any>>{}, leading_comments);
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
    if (IsHostTensorCollective(call)) {
      auto visited_call = As<Call>(VisitExpr(op->expr_));
      INTERNAL_CHECK_SPAN(IsHostTensorCollective(visited_call), op->span_)
          << "LowerHostTensorCollectives: collective EvalStmt rewrote to a non-collective expression";
      return LowerCollective(visited_call, op->span_, op->leading_comments_);
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsHostTensorCollective(call)) {
      return IRMutator::VisitStmt_(op);
    }
    auto visited_call = As<Call>(VisitExpr(op->value_));
    INTERNAL_CHECK_SPAN(IsHostTensorCollective(visited_call), op->span_)
        << "LowerHostTensorCollectives: collective AssignStmt rewrote to a non-collective expression";
    std::vector<StmtPtr> stmts;
    stmts.push_back(LowerCollective(visited_call, op->span_, op->leading_comments_));
    const auto* rule = LookupHostCollectiveRule(visited_call->op_->name_);
    INTERNAL_CHECK(rule) << "LowerHostTensorCollectives: missing rule for " << visited_call->op_->name_;
    if (auto alias_src = rule->alias_source(visited_call)) {
      auto result_var = MintDistributedResultVar(op->var_, *alias_src);
      var_remap_[op->var_.get()] = result_var;
      stmts.push_back(std::make_shared<AssignStmt>(result_var, *alias_src, op->span_));
    }
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  StmtPtr LowerCollective(const CallPtr& call, const Span& span,
                          const std::vector<std::string>& leading_comments) {
    const auto* rule = LookupHostCollectiveRule(call->op_->name_);
    INTERNAL_CHECK(rule) << "LowerHostTensorCollectives: missing rule for " << call->op_->name_;
    INTERNAL_CHECK_SPAN(!scope_stack_.empty(), call->span_)
        << "LowerHostTensorCollectives: " << call->op_->name_ << " must appear inside a CommDomainScopeStmt";
    auto buffers = rule->scope_buffers(call);
    auto scope = FindScopeForBuffers(scope_stack_, buffers);
    INTERNAL_CHECK_SPAN(scope, call->span_) << "LowerHostTensorCollectives: " << call->op_->name_
                                            << " window buffers must resolve to the same comm-domain scope";
    return EmitPerDeviceBuiltinCalls(call, *rule, scope, span, leading_comments);
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
