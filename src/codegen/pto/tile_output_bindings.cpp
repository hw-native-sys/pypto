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

#include "pypto/codegen/pto/tile_output_bindings.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/memref_utils.h"

namespace pypto::codegen {
namespace {

class TileOutputPlanner : public ir::IRVisitor {
 public:
  explicit TileOutputPlanner(const backend::Backend& backend) : backend_(backend) {}

  TileOutputBindings Prepare(const ir::StmtPtr& body) {
    if (body) VisitStmt(body);
    // Validate in producer order once all projections have been visited.
    // Hardware writes every slot, even when the output has no later reads.
    for (const auto& [tuple, call] : producers_) {
      const auto& outputs = bindings_.at(tuple);
      for (size_t i = 0; i < outputs.size(); ++i) {
        INTERNAL_CHECK_SPAN(outputs[i].var, call->span_)
            << "Internal error: '" << call->op_->name_ << "' output " << i << " has no `<var> = tuple[" << i
            << "]` binding to name its destination";
      }
    }
    return std::move(bindings_);
  }

 protected:
  // Only assignments define tuple producers and projections. Bounds, shape
  // metadata and other expression trees cannot introduce a storage binding.
  void VisitExpr(const ir::ExprPtr&) override {}

  void VisitStmt_(const ir::AssignStmtPtr& stmt) override {
    if (const auto call = ir::As<ir::Call>(stmt->value_)) {
      const auto* entry = ir::LookupOpEntry(call->op_);
      if (entry && entry->GetOutputArity() > 1 && entry->GetOpCategory() == "TileOp" &&
          backend_.GetOpInfo(call->op_->name_)) {
        const bool inserted = bindings_.try_emplace(stmt->var_.get(), entry->GetOutputArity()).second;
        INTERNAL_CHECK_SPAN(inserted, stmt->span_)
            << "Internal error: duplicate Tile tuple producer for '" << stmt->var_->name_hint_ << "'";
        producers_.emplace_back(stmt->var_.get(), call);
      }
      return;
    }

    const auto item = ir::As<ir::TupleGetItemExpr>(stmt->value_);
    if (!item) return;
    const auto tuple = ir::AsVarLike(item->tuple_);
    const auto found = bindings_.find(tuple.get());
    // VF calls and GM tuple projections borrow storage. Only SSA results of
    // native Tile producers registered above own hardware destinations.
    if (found == bindings_.end()) return;
    INTERNAL_CHECK_SPAN(item->index_ >= 0 && static_cast<size_t>(item->index_) < found->second.size(),
                        stmt->span_)
        << "Internal error: output index " << item->index_ << " is outside the arity of tuple '"
        << tuple->name_hint_ << "'";
    auto& binding = found->second[item->index_];
    INTERNAL_CHECK_SPAN(!binding.var, stmt->span_)
        << "Internal error: element " << item->index_ << " of tuple '" << tuple->name_hint_
        << "' is bound twice, by '" << binding.var->name_hint_ << "' and '" << stmt->var_->name_hint_ << "'";
    const auto type = ir::GetTileTypeWithMemRef(stmt->var_->GetType());
    INTERNAL_CHECK_SPAN(type, stmt->span_)
        << "Internal error: output " << item->index_ << " of tuple '" << tuple->name_hint_
        << "' must have TileType with MemRef set by InitMemRef";
    binding = {stmt->var_, type};
  }

 private:
  const backend::Backend& backend_;
  TileOutputBindings bindings_;
  std::vector<std::pair<const ir::Var*, ir::CallPtr>> producers_;
};

}  // namespace

TileOutputBindings PrepareTileOutputBindings(const ir::StmtPtr& body, const backend::Backend& backend) {
  return TileOutputPlanner(backend).Prepare(body);
}

}  // namespace pypto::codegen
