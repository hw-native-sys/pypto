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

#include "pypto/ir/transforms/utils/op_predicates.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace op_predicates {

using core_affinity::CrossCoreRole;

std::string DistributedOpContextError(const OpPtr& op, std::optional<Level> level, std::optional<Role> role,
                                      bool in_device_scope) {
  if (!op) return "";
  const bool count = IsOp(op, "pld.system.num_workers");
  const bool world = IsOp(op, "pld.system.world_size");
  const bool alloc = IsOp(op, "pld.tensor.alloc_window_buffer");
  if (count || world || alloc) {
    const bool allowed_level = level == Level::HOST || (!world && level == Level::CLUSTER_0);
    // An unspecified role uses the orchestration level's default. Explicit
    // SubWorkers never own a child worker pool, even when their level is HOST.
    if (!allowed_level || role == Role::SubWorker || in_device_scope) {
      return op->name_ + " can only be called in " + (world ? "HOST" : "HOST or CLUSTER_0") +
             " orchestration context (role=Orchestrator, not inside InCore / SPMD scopes)";
    }
  }
  // These namespaces contain communication, not window declarations or topology
  // queries. Match families by namespace and named operations through IsOp; an
  // alias or a separately constructed Op must not bypass the L4 restriction.
  if (level == Level::CLUSTER_0 &&
      (IsManagedTensorCollective(op) || op->name_.rfind("builtin.tensor.", 0) == 0 ||
       op->name_.rfind("pld.tile.", 0) == 0 || IsOp(op, "pld.tensor.remote_store") ||
       IsOp(op, "pld.tensor.put") || IsOp(op, "pld.tensor.get") || IsOp(op, "pld.system.notify") ||
       IsOp(op, "pld.system.wait") || IsOp(op, "pld.system.defer_wait"))) {
    return op->name_ +
           " is not supported in CLUSTER_0; place communication in HOST or device "
           "orchestration and dispatch it from CLUSTER_0";
  }
  return "";
}

std::vector<std::pair<std::string, Span>> DistributedOpContextErrors(const FunctionPtr& func) {
  class ContextVisitor : public IRVisitor {
   public:
    std::optional<Level> level;
    std::optional<Role> role;
    int device_depth = 0;
    std::vector<std::pair<std::string, Span>> errors;

    void Check(const OpPtr& op, const Span& span) {
      auto message = DistributedOpContextError(op, level, role, device_depth != 0);
      if (!message.empty()) errors.emplace_back(std::move(message), span);
    }

    void VisitExpr_(const CallPtr& op) override {
      Check(op->op_, op->span_);
      IRVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const SubmitPtr& op) override {
      Check(op->op_, op->span_);
      IRVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const HierarchyScopeStmtPtr& op) override {
      // Scope headers are evaluated by the enclosing context, not the body.
      // Visiting only the body after changing context also avoids double-walking
      // attrs via the base handler (and incorrectly legalizing cluster headers).
      VisitScopeAttrs(op);
      const auto saved_level = level;
      const auto saved_role = role;
      level = op->level_;
      role = op->role_;
      VisitStmt(op->body_);
      level = saved_level;
      role = saved_role;
    }
    void VisitStmt_(const InCoreScopeStmtPtr& op) override {
      VisitScopeAttrs(op);
      ++device_depth;
      VisitStmt(op->body_);
      --device_depth;
    }
    void VisitStmt_(const SpmdScopeStmtPtr& op) override {
      VisitExpr(op->core_num_);
      VisitScopeAttrs(op);
      ++device_depth;
      VisitStmt(op->body_);
      --device_depth;
    }
  } visitor;
  if (!func) return {};
  visitor.level = func->level_;
  visitor.role = func->role_;
  const auto type = func->func_type_;
  visitor.device_depth = type == FunctionType::InCore || type == FunctionType::AIC ||
                         type == FunctionType::AIV || type == FunctionType::Group ||
                         type == FunctionType::Spmd;
  visitor.VisitStmt(func->body_);
  return std::move(visitor.errors);
}

bool CastRequiresDisjointSource(const CallPtr& call, bool is_a2a3) {
  if (!IsOp(call, "tile.cast") || call->args_.empty()) return false;
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto result = As<TileType>(call->GetType());
  if (!source || !result) return false;
  const auto source_bits = source->dtype_.GetBit();
  const auto result_bits = result->dtype_.GetBit();
  return result_bits > source_bits || (is_a2a3 && result_bits < source_bits);
}

namespace {

bool HasRole(const CallPtr& call, CrossCoreRole expected) {
  if (!call || !call->op_) return false;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return false;
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op->name_)) return false;
  auto role = registry.GetEntry(op->name_).GetCrossCoreRole();
  return role.has_value() && *role == expected;
}

}  // namespace

bool IsTPop(const CallPtr& call) { return HasRole(call, CrossCoreRole::TPop); }
bool IsTPush(const CallPtr& call) { return HasRole(call, CrossCoreRole::TPush); }
bool IsTFree(const CallPtr& call) { return HasRole(call, CrossCoreRole::TFree); }
bool IsInitializePipe(const CallPtr& call) { return HasRole(call, CrossCoreRole::InitializePipe); }

bool IsBufferAliasingViewOp(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;
  const auto& entry = registry.GetEntry(op_name);
  // An inherit-input op aliases its input buffer only if it is also in-place
  // safe. tile.transpose inherits input memory but PERMUTES data into a fresh
  // buffer (pto.ttrans, registered not_inplace_safe()), so its output does NOT
  // alias the input — IsInplaceSafe() excludes it without hardcoding op names.
  return entry.OutputMemoryInheritsInput() && entry.IsInplaceSafe();
}

bool OutputInheritsSourceBuffer(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;
  const auto& entry = registry.GetEntry(op_name);
  // Inheriting the input's memory *space* is not the same as landing in its
  // buffer: tile.transpose inherits the space but permutes into a fresh one
  // (pto.ttrans, registered not_inplace_safe()). IsBufferAliasingViewOp already
  // draws that line, so reuse it rather than testing OutputMemoryInheritsInput()
  // directly — otherwise a transpose output, which does own its buffer, could
  // not be bound to a declared allocation.
  return IsBufferAliasingViewOp(op_name) || entry.GetOutputReusesInputArg().has_value();
}

std::optional<size_t> BuiltinWritebackArgIndex(const OpPtr& op, size_t arg_count) {
  if (!op) return std::nullopt;
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op->name_)) return std::nullopt;
  auto aliased_idx = registry.GetEntry(op->name_).GetOutputReusesInputArg();
  if (!aliased_idx || *aliased_idx >= arg_count) return std::nullopt;
  return aliased_idx;
}

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.rfind("tile.", 0) == 0 || op_name.rfind("tensor.", 0) == 0 ||
         op_name.rfind("system.", 0) == 0 || op_name.rfind("array.", 0) == 0 ||
         op_name == "pld.system.get_comm_ctx";
}

bool IsManagedTensorCollective(const OpPtr& op) {
  if (!op) return false;
  return IsOp(op, "pld.tensor.allgather") || IsOp(op, "pld.tensor.allreduce") ||
         IsOp(op, "pld.tensor.barrier") || IsOp(op, "pld.tensor.broadcast") ||
         IsOp(op, "pld.tensor.reduce_scatter") || IsOp(op, "pld.tensor.all_to_all") ||
         IsOp(op, "pld.tensor.all_to_all_v");
}

bool IsPublishingWrite(const CallPtr& call) {
  if (!call || !call->op_) return false;
  // Remote writes always publish into a peer's window.
  if (IsOp(call, "pld.tile.remote_store") || IsOp(call, "pld.tile.put") || IsOp(call, "pld.tensor.put")) {
    return true;
  }
  // A local store whose destination tensor is window-bound: a peer can
  // remote_load it, so it carries the same data-before-signal obligation.
  if (IsOp(call, "tile.store") && call->args_.size() > 2 && call->args_[2] &&
      As<DistributedTensorType>(call->args_[2]->GetType())) {
    return true;
  }
  // A scalar write into a window-bound tensor publishes for the same reason.
  // ConvertTensorToTileOps deliberately leaves `tensor.write` unconverted when
  // its destination is a DistributedTensor (PTO codegen lowers it to
  // `pto.store`), so it reaches this pass as a registered builtin and
  // would otherwise fall through to the `IsRegistered` guard below as false.
  if (IsOp(call, "tensor.write") && !call->args_.empty() && call->args_[0] &&
      As<DistributedTensorType>(call->args_[0]->GetType())) {
    return true;
  }
  // A get into a window-bound local destination publishes for the same reason.
  if ((IsOp(call, "pld.tile.get") || IsOp(call, "pld.tensor.get")) && !call->args_.empty() &&
      call->args_[0] && As<DistributedTensorType>(call->args_[0]->GetType())) {
    return true;
  }
  // Conservative: an unregistered op name is a call to a user function whose body
  // is not analysed interprocedurally — assume it may publish. Registered builtin
  // and distributed ops fall through to false (handled above or non-publishing).
  return !OpRegistry::GetInstance().IsRegistered(call->op_->name_);
}

}  // namespace op_predicates
}  // namespace ir
}  // namespace pypto
