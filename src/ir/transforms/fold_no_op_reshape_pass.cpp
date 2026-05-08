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

#include <memory>
#include <string>

#include "pypto/codegen/pto/tile_buf_signature.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

/// Returns true if @p assign is `lhs = tile.reshape(src, shape)` where the
/// LHS and the source share the same MemRef root and produce identical
/// `TileBufSignature`s. In that case the reshape is a pure no-op at the PTO
/// level (the per-var alloc model already pre-declared LHS with the same
/// shape and addr) and we can replace the call with a Var-to-Var assignment.
bool IsNoOpReshape(const AssignStmtPtr& assign) {
  if (!assign || !assign->var_) return false;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || call->op_->name_ != "tile.reshape") return false;
  // Canonical tile.reshape arity is exactly 2 (tile, shape). Anything else
  // is malformed IR and should remain visible to the verifier rather than
  // being silently folded.
  if (call->args_.size() != 2) return false;

  auto src_var = AsVarLike(call->args_[0]);
  if (!src_var) return false;

  auto lhs_tile = As<TileType>(assign->var_->GetType());
  auto rhs_tile = As<TileType>(src_var->GetType());
  if (!lhs_tile || !rhs_tile) return false;

  // Both sides must be backed by the same MemRef. LegalizePTOBufferReuse
  // makes this decision; if it didn't, the reshape is a real shape change
  // and PTO must materialize it via pto.treshape.
  if (!lhs_tile->memref_.has_value() || !rhs_tile->memref_.has_value()) return false;
  const auto& lhs_memref = *lhs_tile->memref_;
  const auto& rhs_memref = *rhs_tile->memref_;
  if (!lhs_memref || !rhs_memref || !lhs_memref->base_ || !rhs_memref->base_) return false;
  if (lhs_memref->base_.get() != rhs_memref->base_.get()) return false;

  auto lhs_sig = codegen::TileBufSignature::FromTileType(*lhs_tile);
  auto rhs_sig = codegen::TileBufSignature::FromTileType(*rhs_tile);
  return lhs_sig == rhs_sig;
}

class FoldNoOpReshapeMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto base_assign = As<AssignStmt>(base);
    if (!base_assign || !IsNoOpReshape(base_assign)) return base;
    auto src_var = AsVarLike(As<Call>(base_assign->value_)->args_[0]);
    return std::make_shared<AssignStmt>(base_assign->var_, src_var, base_assign->span_);
  }
};

}  // namespace

Pass FoldNoOpReshape() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    if (!IsInCoreType(func->func_type_)) return func;
    FoldNoOpReshapeMutator mutator;
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) return func;
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "FoldNoOpReshape", kFoldNoOpReshapeProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
