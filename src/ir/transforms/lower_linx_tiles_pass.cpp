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

#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "src/ir/transforms/tensor_to_tile/type_propagation.h"

namespace pypto::ir {
namespace {

// Tensor conversion already owns GM handles, windows and output aliases. Place
// its unresolved tile producers here and propagate the new SSA types to uses.
class LinxTileLowering : public tensor_to_tile::TypePropagatingMutator {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& stmt) override {
    return HandlePassThroughAssign(stmt, VisitExpr(stmt->value_));
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    const auto call = As<Call>(IRMutator::VisitExpr_(op));
    if (As<GlobalVar>(call->op_) || call->op_->name_.rfind("tile.", 0) != 0) return call;

    auto kwargs = call->kwargs_;
    if (IsOp(call, "tile.load") || IsOp(call, "tile.create") || IsOp(call, "tile.move")) {
      CHECK_SPAN(call->GetKwarg("target_memory", MemorySpace::Local) == MemorySpace::Local, call->span_)
          << "Linx " << call->op_->name_ << " requires Local storage";
      if (!call->HasKwarg("target_memory")) kwargs.emplace_back("target_memory", MemorySpace::Local);
    } else {
      CHECK_SPAN(IsOp(call, "tile.add") || IsOp(call, "tile.store") || IsOp(call, "tile.publish_shared") ||
                     IsOp(call, "tile.broadcast_shared"),
                 call->span_)
          << "Linx tile lowering does not support " << call->op_->name_;
    }

    // Broadcast consumes a published handle. Compute, GM stores and publication
    // consume private CELL tiles. Operand roles never imply Ascend storage.
    const auto expected = IsOp(call, "tile.broadcast_shared") ? MemorySpace::Shared : MemorySpace::Local;
    for (const auto& arg : call->args_) {
      if (const auto tile = As<TileType>(arg->GetType())) {
        CHECK_SPAN(tile->memory_space_ == expected, call->span_)
            << "Linx " << call->op_->name_ << " requires " << MemorySpaceToString(expected)
            << " tile operands";
      }
    }
    // Use the operation's registered type rule to preserve geometry and valid
    // extents. In particular a GM store returns its destination tensor type.
    const auto result = OpRegistry::GetInstance().Create(call->op_->name_, call->args_, kwargs, call->span_);
    if (call->attrs_.empty()) return result;
    auto copy = MutableCopy(result);
    copy->attrs_ = call->attrs_;
    return copy;
  }
};
}  // namespace

namespace pass {
Pass LowerLinxTiles() {
  return CreateFunctionPass(
      [](const FunctionPtr& func) -> FunctionPtr {
        LinxTileLowering mutator;
        auto body = mutator.VisitStmt(func->body_);
        if (body == func->body_) return func;
        auto copy = MutableCopy(func);
        copy->body_ = body;
        return copy;
      },
      "LowerLinxTiles", PassProperties{.required = {IRProperty::SSAForm}});
}
}  // namespace pass
}  // namespace pypto::ir
