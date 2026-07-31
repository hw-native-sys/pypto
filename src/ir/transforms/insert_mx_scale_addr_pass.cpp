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

/**
 * @file insert_mx_scale_addr_pass.cpp
 * @brief Insert tile.tget_scale_addr bindings before MX matmul consumers.
 *
 * Runs after InferTileMemorySpace so Left/LeftScale and Right/RightScale pairs
 * are already resolved. Because tget_scale_addr mutates shared physical scale
 * buffers in place, bindings are never reused across MX matmul consumers.
 */

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

bool IsResolvedMxScaleDataPair(const VarPtr& scale, const VarPtr& data) {
  auto scale_type = As<TileType>(scale->GetType());
  auto data_type = As<TileType>(data->GetType());
  if (!scale_type || !data_type) return false;
  auto scale_space = scale_type->GetMemorySpace();
  auto data_space = data_type->GetMemorySpace();
  if (!scale_space.has_value() || !data_space.has_value()) return false;
  const bool is_left = *scale_space == MemorySpace::LeftScale && *data_space == MemorySpace::Left;
  const bool is_right = *scale_space == MemorySpace::RightScale && *data_space == MemorySpace::Right;
  return is_left || is_right;
}

class MxScaleAddrInserter : public IRMutator {
 public:
  // Entry / control-flow body helper: NormalizedStmtStructure unwraps single-child
  // SeqStmts, so if/for/while (and even the function body) may be a bare AssignStmt.
  // SeqStmts bodies use VisitStmt_(SeqStmts); bare MX matmul assigns are wrapped
  // into a SeqStmts with the inserted bindings (same pattern as InsertCommFence).
  StmtPtr RewriteBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);

    auto visited = VisitStmt(body);
    auto assign = As<AssignStmt>(visited);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !call || GetDataScalePairs(call).empty()) return visited;

    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    auto rewritten = InsertBindingsForMxMatmul(new_stmts, assign, call, changed);
    new_stmts.push_back(rewritten);
    return SeqStmts::Flatten(std::move(new_stmts), body->span_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());

    for (const auto& stmt : op->stmts_) {
      auto rewritten = IRMutator::VisitStmt(stmt);
      if (rewritten.get() != stmt.get()) changed = true;

      auto assign = As<AssignStmt>(rewritten);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && call) {
        rewritten = InsertBindingsForMxMatmul(new_stmts, assign, call, changed);
      }
      new_stmts.push_back(rewritten);
    }

    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_then = RewriteBody(op->then_body_);
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) new_else = RewriteBody(op->else_body_.value());
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op, op->body_); }

 private:
  static std::vector<std::pair<size_t, size_t>> GetDataScalePairs(const CallPtr& call) {
    if (IsOp(call, "tile.matmul_mx") || IsOp(call, "tile.matmul_mx_bias")) {
      return {{0, 1}, {2, 3}};
    }
    if (IsOp(call, "tile.matmul_mx_acc")) {
      return {{1, 2}, {3, 4}};
    }
    return {};
  }

  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    auto new_body = RewriteBody(body);
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }

  StmtPtr InsertBindingsForMxMatmul(std::vector<StmtPtr>& new_stmts, const AssignStmtPtr& assign,
                                    const CallPtr& call, bool& changed) {
    const auto pairs = GetDataScalePairs(call);
    if (pairs.empty()) return assign;

    if (backend::BackendConfig::IsConfigured()) {
      const auto* pass_context = PassContext::Current();
      const auto* handler = pass_context ? pass_context->GetBackendHandler()
                                         : backend::BackendConfig::GetBackend()->GetHandler();
      CHECK_SPAN(handler->GetPtoTargetArch() == "a5", call->span_)
          << call->op_->name_ << " is only supported on the Ascend950 ('a5') backend, but got '"
          << handler->GetPtoTargetArch() << "'";
    }

    std::vector<ExprPtr> new_args = call->args_;
    auto& registry = OpRegistry::GetInstance();
    bool call_changed = false;

    for (const auto& [data_index, scale_index] : pairs) {
      INTERNAL_CHECK_SPAN(data_index < new_args.size() && scale_index < new_args.size(), call->span_)
          << "Internal error: malformed " << call->op_->name_ << " operand list";
      auto data = AsVarLike(new_args[data_index]);
      auto scale = AsVarLike(new_args[scale_index]);
      INTERNAL_CHECK_SPAN(data && scale, call->span_)
          << "Internal error: " << call->op_->name_
          << " MX data/scale operands must be Var-like (Var or IterArg)";
      INTERNAL_CHECK_SPAN(IsResolvedMxScaleDataPair(scale, data), call->span_)
          << "Internal error: InsertMxScaleAddr requires resolved LeftScale↔Left or "
             "RightScale↔Right pairing before inserting tile.tget_scale_addr (run "
             "InferTileMemorySpace first)";

      auto binding_call = registry.Create("tile.tget_scale_addr", {scale, data}, {}, call->span_);
      auto bound_scale =
          std::make_shared<Var>(scale->name_hint_ + "_bound", binding_call->GetType(), call->span_);
      new_stmts.push_back(std::make_shared<AssignStmt>(bound_scale, binding_call, call->span_));
      new_args[scale_index] = bound_scale;
      call_changed = true;
      changed = true;
    }

    if (!call_changed) return assign;
    auto deduced = registry.Create(call->op_->name_, new_args, call->kwargs_, call->span_);
    auto rebound_call = std::make_shared<Call>(deduced->op_, deduced->args_, deduced->kwargs_, call->attrs_,
                                               deduced->GetType(), deduced->span_);
    return std::make_shared<AssignStmt>(assign->var_, rebound_call, assign->span_);
  }
};

FunctionPtr TransformInsertMxScaleAddr(const FunctionPtr& func) {
  MxScaleAddrInserter inserter;
  auto new_body = inserter.RewriteBody(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto updated = MutableCopy(func);
  updated->body_ = new_body;
  return updated;
}

}  // namespace

namespace pass {

Pass InsertMxScaleAddr() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        new_functions[gvar] = TransformInsertMxScaleAddr(func);
      } else {
        new_functions[gvar] = func;
      }
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "InsertMxScaleAddr", kInsertMxScaleAddrProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
