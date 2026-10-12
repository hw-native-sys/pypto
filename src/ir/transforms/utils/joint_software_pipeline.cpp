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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/software_pipeline.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

// One indexed scan per function, followed by O(1) work per Group. A generated
// family identifies the same source loop on both endpoints. Hand-written pairs,
// feedback and multiple communication sites retain the established lowering.
class Endpoint : public IRVisitor {
 public:
  ForStmtPtr loop;
  CallPtr transfer;
  CallPtr init;
  CallPtr reserve;
  CallPtr import;
  std::map<const Var*, CallPtr> definitions;
  bool valid = true;
  bool writes_global = false;

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!depth_) {
      if (loop || op->kind_ != ForKind::Pipeline || !op->iter_args_.empty() || !op->return_vars_.empty()) {
        valid = false;
      }
      loop = op;
    }
    // Local child scopes are analyzed transactionally by the common planner.
    // A transfer inside a child still fails the depth == 1 check below.
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }
  void VisitStmt_(const WhileStmtPtr&) override { valid = false; }
  void VisitStmt(const StmtPtr& stmt) override {
    if (As<ScopeStmt>(stmt)) {
      valid = false;
    } else {
      IRVisitor::VisitStmt(stmt);
    }
  }
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto tile = As<TileType>(op->var_->GetType()); tile && tile->memref_.has_value()) {
      valid = false;
    }
    if (auto call = As<Call>(op->value_)) definitions.emplace(op->var_.get(), call);
    IRVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const SubmitPtr&) override { valid = false; }
  void VisitExpr_(const CallPtr& op) override {
    if (OpRegistry::GetInstance().IsRegistered(op->op_->name_)) {
      const auto& entry = OpRegistry::GetInstance().GetEntry(op->op_->name_);
      for (size_t i = 0; i < op->args_.size(); ++i) {
        if (As<TensorType>(op->args_[i]->GetType()) &&
            ((!entry.HasDeclaredArgEffects() &&
              entry.GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::Functional &&
              entry.GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::NoAccess) ||
             entry.GetArgEffect(i, op->kwargs_) != ArgEffect::Read)) {
          writes_global = true;
        }
      }
    }
    if (IsOp(op, "tile.tpush_to_aiv") || IsOp(op, "tile.tpop_from_aic")) {
      if (transfer || depth_ != 1) valid = false;
      transfer = op;
    } else if (IsOp(op, "system.aic_initialize_pipe") || IsOp(op, "system.aiv_initialize_pipe")) {
      if (init || depth_) valid = false;
      init = op;
    } else if (IsOp(op, "system.reserve_buffer")) {
      if (reserve || depth_) valid = false;
      reserve = op;
    } else if (IsOp(op, "system.import_peer_buffer")) {
      if (import || depth_) valid = false;
      import = op;
    } else if (!IsOp(op, "system.tfree_to_aic")) {
      if (!OpRegistry::GetInstance().IsRegistered(op->op_->name_) ||
          OpRegistry::GetInstance().GetEntry(op->op_->name_).GetCrossCoreRole()) {
        valid = false;
      }
    }
    IRVisitor::VisitExpr_(op);
  }
  [[nodiscard]] bool Ready() const {
    return valid && loop && transfer && init && loop->GetAttr<int>("software_pipeline_family", 0) > 0 &&
           transform_utils::EvalConstInt(loop->start_) == 0 &&
           transform_utils::EvalConstInt(loop->step_) == 1 && init->GetKwarg<int>("dir_mask", 0) == 1 &&
           init->GetKwarg<int>("slot_num", 0) >= 2 && init->GetKwarg<int>("slot_num", 0) <= 4;
  }
  [[nodiscard]] bool HasBuffer(const CallPtr& definition) const {
    if (init->args_.empty()) return false;
    auto var = AsVarLike(init->args_[0]);
    if (!var) return false;
    auto it = definitions.find(var.get());
    return it != definitions.end() && it->second == definition;
  }

 private:
  int depth_ = 0;
};

bool Matches(const Endpoint& aic, const Endpoint& aiv, const FunctionPtr& aiv_func) {
  if (!aic.Ready() || !aiv.Ready() || aic.writes_global || !aic.import || !aiv.reserve || aic.reserve ||
      aiv.import || !IsOp(aic.transfer, "tile.tpush_to_aiv") || !IsOp(aiv.transfer, "tile.tpop_from_aic") ||
      !aic.HasBuffer(aic.import) || !aiv.HasBuffer(aiv.reserve)) {
    return false;
  }
  const int split = aic.transfer->GetKwarg<int>("split", 0);
  if (split < 0 || split > 2 || aic.transfer->args_.size() != 1) return false;
  auto source = As<TileType>(aic.transfer->args_[0]->GetType());
  auto destination = As<TileType>(aiv.transfer->GetType());
  if (!source || !destination || source->shape_.size() != 2 || destination->shape_.size() != 2 ||
      source->dtype_ != destination->dtype_) {
    return false;
  }
  const auto view = tile_view_semantics::GetEffectiveTileView(*source);
  for (size_t i = 0; i < 2; ++i) {
    auto extent = transform_utils::EvalConstInt(source->shape_[i]);
    auto received = transform_utils::EvalConstInt(destination->shape_[i]);
    if (!extent || !received || *extent <= 0 ||
        (view.valid_shape.size() > i && transform_utils::EvalConstInt(view.valid_shape[i]) != extent)) {
      return false;
    }
    const int partition = (split == 1 && i == 0) || (split == 2 && i == 1) ? 2 : 1;
    if (*extent % partition || *extent / partition != *received) return false;
  }
  if (aic.loop->GetAttr<int>("software_pipeline_family", 0) !=
      aiv.loop->GetAttr<int>("software_pipeline_family", 0)) {
    return false;
  }
  for (const auto* key : {"slot_size", "slot_num"}) {
    if (aic.init->GetKwarg<int>(key, 0) != aiv.init->GetKwarg<int>(key, 0)) return false;
  }
  for (const auto* key : {"split", "id"}) {
    if (aic.transfer->GetKwarg<int>(key, 0) != aiv.transfer->GetKwarg<int>(key, 0)) return false;
  }
  return aic.import->GetKwarg<std::string>("peer_func", "") == aiv_func->name_ &&
         aic.import->GetKwarg<std::string>("name", "") == aiv.reserve->GetKwarg<std::string>("name", "");
}

}  // namespace

ProgramPtr LowerJointSoftwarePipeline(const ProgramPtr& program) {
  auto* ctx = PassContext::Current();
  if (!ctx || !ctx->GetEnableSoftwarePipeline() || ctx->GetMemoryPlanner() != MemoryPlanner::PyPTO ||
      !ctx->GetBackendHandler()->RequiresGMPipeBuffer()) {
    return program;
  }
  std::map<const Function*, Endpoint> endpoints;
  std::map<const Function*, size_t> uses;
  std::map<const Function*, std::vector<WrapperCallInfo>> groups;
  for (const auto& [name, func] : program->functions_) {
    if (func->func_type_ == FunctionType::AIC || func->func_type_ == FunctionType::AIV) {
      endpoints[func.get()].VisitStmt(func->body_);
    }
    auto calls = CollectInnerCalls(func, program);
    for (const auto& call : calls) ++uses[call.inner_callee.get()];
    if (func->func_type_ == FunctionType::Group) groups.emplace(func.get(), std::move(calls));
  }
  std::map<const Function*, FunctionPtr> replacements;
  for (const auto& [group, calls] : groups) {
    if (calls.size() != 2) continue;
    auto body = As<SeqStmts>(group->body_);
    if (!body) continue;
    bool direct_calls = true;
    for (const auto& stmt : body->stmts_) {
      if (As<ReturnStmt>(stmt)) continue;
      auto assign = As<AssignStmt>(stmt);
      auto eval = As<EvalStmt>(stmt);
      auto call = As<Call>(assign ? assign->value_ : eval ? eval->expr_ : nullptr);
      if (!call || !As<GlobalVar>(call->op_)) direct_calls = false;
    }
    if (!direct_calls) continue;
    const auto& aic = calls[0].inner_callee;
    const auto& aiv = calls[1].inner_callee;
    if (aic->func_type_ != FunctionType::AIC || aiv->func_type_ != FunctionType::AIV ||
        uses[aic.get()] != 1 || uses[aiv.get()] != 1) {
      continue;
    }
    bool same_abi = true;
    for (const auto& call : calls) {
      if (call.inner_call->args_.size() != group->params_.size() ||
          call.inner_callee->params_.size() != group->params_.size()) {
        same_abi = false;
        break;
      }
      for (size_t i = 0; i < group->params_.size(); ++i) {
        if (call.inner_call->args_[i] != group->params_[i]) same_abi = false;
      }
    }
    const auto& c = endpoints.at(aic.get());
    const auto& v = endpoints.at(aiv.get());
    if (!same_abi || !Matches(c, v, aiv)) {
      LOG_DEBUG << "LowerJointSoftwarePipeline: pair declined: ABI=" << same_abi << " AIC=" << c.Ready()
                << " AIV=" << v.Ready() << " producer GM writes=" << c.writes_global;
      continue;
    }
    const auto bytes = v.reserve->GetKwarg<int>("size", 0);
    const auto slots = v.init->GetKwarg<int>("slot_num", 0);
    const auto stride = v.init->GetKwarg<int>("slot_size", 0);
    if (bytes <= 0 || stride <= 0 || static_cast<int64_t>(slots) * stride > bytes) continue;
    auto consumer = LowerFifoSoftwarePipeline(aiv, v.loop, slots);
    if (consumer == aiv) continue;
    // GM-entry pipes do not own a second local FIFO reservation. Keep the
    // scalar bindings for SSA users; the codegen derives entry descriptors from
    // the owned pop and its matching push.
    class RemoveLocalReservation : public IRMutator {
      ExprPtr VisitExpr_(const CallPtr& call) override {
        if (IsOp(call, "system.reserve_buffer") || IsOp(call, "system.import_peer_buffer")) {
          return std::make_shared<ConstInt>(0, DataType::INT32, call->span_);
        }
        return IRMutator::VisitExpr_(call);
      }
    };
    RemoveLocalReservation remove;
    auto new_consumer = MutableCopy(consumer);
    new_consumer->body_ = remove.VisitStmt(consumer->body_);
    auto new_producer = MutableCopy(aic);
    new_producer->body_ = remove.VisitStmt(aic->body_);
    replacements[aiv.get()] = new_consumer;
    replacements[aic.get()] = new_producer;
  }
  if (replacements.empty()) return program;
  std::vector<FunctionPtr> functions;
  for (const auto& [name, func] : program->functions_) {
    auto it = replacements.find(func.get());
    functions.push_back(it == replacements.end() ? func : it->second);
  }
  return std::make_shared<Program>(functions, program->name_, program->span_);
}
}  // namespace pypto::ir
