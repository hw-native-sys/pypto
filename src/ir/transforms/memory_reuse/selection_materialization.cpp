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
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tile_buf_signature.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace memory_reuse {
namespace {

// An optional update normally keeps every earlier descriptor candidate live.
// Under capacity pressure, copy the unchanged arm into the updated arm's own
// allocation. The two arms then produce one fixed value, and ordinary liveness
// can recycle earlier allocations. Neither producer is retargeted to an input.
// One indexed traversal plus the selection analysis is O(N log N); each yield
// operand is inspected once, without walking its transitive candidate set.
class ForwardedVecMaterializer : public StorageBoundaryMutator {
  struct YieldCopy {
    size_t index;
    VarPtr value;
    StmtPtr transfer;
  };

 public:
  ForwardedVecMaterializer(const FunctionPtr& func, const std::set<const Var*>& pinned_bases)
      : pinned_bases_(pinned_bases), selections_(AnalyzeTileSelections(func->body_).values) {
    names_.Collect(func->body_, func->params_);
    for (const auto& parameter : func->params_) Record(parameter);
  }

 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override {
    const auto found = var_remap_.find(argument.get());
    if (found != var_remap_.end()) return found->second;
    // Cache unchanged carries too. Repeated body reads must not recursively
    // revisit a nested loop's initializer chain on this legacy Tile path.
    const auto result = StorageBoundaryMutator::VisitExpr_(argument);
    var_remap_[argument.get()] = result;
    return result;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = As<AssignStmt>(IRMutator::VisitStmt_(op));
    Record(result->var_);
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto condition = VisitExpr(op->condition_);
    const auto then_start = order_;
    auto then_body = VisitStmt(op->then_body_);
    const auto then_end = order_;
    const auto else_start = order_;
    auto else_body = op->else_body_ ? VisitStmt(*op->else_body_) : nullptr;
    const auto else_end = order_;
    auto result = MutableCopy(op);
    result->condition_ = condition;
    result->then_body_ = then_body;
    if (else_body) result->else_body_ = else_body;
    const auto lhs = transform_utils::GetLastYieldStmt(then_body);
    const auto rhs = else_body ? transform_utils::GetLastYieldStmt(else_body) : nullptr;
    std::vector<YieldCopy> then_moves;
    std::vector<YieldCopy> else_moves;
    // Different result slots must never write the same destination. A branch
    // can yield one producer at several slots; copying distinct forwarded
    // values into that producer would silently collapse simultaneous results.
    std::set<const Var*> destinations;
    if (lhs && rhs) {
      for (size_t i = 0; i < op->return_vars_.size() && i < lhs->value_.size() && i < rhs->value_.size();
           ++i) {
        const auto a = AsVarLike(lhs->value_[i]);
        const auto b = AsVarLike(rhs->value_[i]);
        if (!Compatible(a, b)) continue;
        const bool local_a = Local(a, then_start, then_end);
        const bool local_b = Local(b, else_start, else_end);
        if (local_a == local_b) continue;
        const auto target = local_a ? a : b;
        const auto source = local_a ? b : a;
        // Only fixed, compiler-owned, whole allocations are valid output
        // destinations. In particular a nested selected handle's MemRef is
        // merely a hint, and author-pinned storage has fixed membership.
        const auto memory = GetDefinedMemRef(GetTileTypeWithMemRef(target->GetType()));
        if (selections_.count(target.get()) || pinned_bases_.count(memory->base_.get()) ||
            !destinations.insert(memory->base_.get()).second) {
          continue;
        }
        auto [copied, move] = CreateStorageMove(
            source, memory, MemorySpace::Vec, [this](const ExprPtr& value) { return VisitExpr(value); },
            names_.Fresh("vec_selection_copy_"));
        (local_a ? else_moves : then_moves).push_back({i, copied, move});
        auto replacement = std::make_shared<Var>(
            op->return_vars_[i]->name_hint_,
            CloneTypeWithMemRef(op->return_vars_[i]->GetType(), memory, MemorySpace::Vec),
            op->return_vars_[i]->span_);
        result->return_vars_[i] = replacement;
        var_remap_[op->return_vars_[i].get()] = replacement;
      }
    }
    result->then_body_ = InsertMoves(then_body, then_moves);
    if (else_body) result->else_body_ = InsertMoves(else_body, else_moves);
    for (const auto& value : result->return_vars_) Record(value);
    if (then_moves.empty() && else_moves.empty() && then_body == op->then_body_ &&
        (!op->else_body_ || else_body == *op->else_body_) && condition == op->condition_) {
      return op;
    }
    return result;
  }

 private:
  void Record(const VarPtr& value) {
    if (const auto tile = GetTileTypeWithMemRef(value->GetType())) {
      const auto point = order_++;
      auto [entry, inserted] =
          definitions_.try_emplace(GetDefinedMemRef(tile)->base_.get(), std::make_pair(point, point));
      if (!inserted) entry->second.second = point;
    }
  }

  bool Local(const VarPtr& value, size_t start, size_t end) const {
    const auto tile = GetTileTypeWithMemRef(value->GetType());
    const auto found = definitions_.find(GetDefinedMemRef(tile)->base_.get());
    // A base already shared by both mutually exclusive arms is not private
    // output storage. Copying into it could overwrite another returned value
    // from the forwarding arm. All its bindings must lie in the producer arm.
    return found != definitions_.end() && found->second.first >= start && found->second.second < end;
  }

  static bool Compatible(const VarPtr& a, const VarPtr& b) {
    const auto lhs = a ? GetTileTypeWithMemRef(a->GetType()) : nullptr;
    const auto rhs = b ? GetTileTypeWithMemRef(b->GetType()) : nullptr;
    if (!lhs || !rhs || lhs->GetMemorySpace() != MemorySpace::Vec ||
        rhs->GetMemorySpace() != MemorySpace::Vec || lhs->shape_.size() != 2 || rhs->shape_.size() != 2) {
      return false;
    }
    // Physical shape remains static, but matching valid extents may be partial
    // or dynamic. Pointer identity proves equality of runtime expressions;
    // do not compare arbitrary expression DAGs once per conditional result.
    // Strided/offset views and multi-slot allocations keep native selection.
    for (const auto& tile : {lhs, rhs}) {
      const auto memory = GetDefinedMemRef(tile);
      const auto offset = As<ConstInt>(memory->byte_offset_);
      const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
      const auto start = As<ConstInt>(view.start_offset);
      if (!offset || offset->value_ != 0 || memory->slot_count_ != 1 || memory->slot_index_ ||
          (view.start_offset && (!start || start->value_ != 0)) || view.valid_shape.size() != 2) {
        return false;
      }
      for (size_t i = 0; i < 2; ++i) {
        const auto shape = As<ConstInt>(tile->shape_[i]);
        const auto valid = As<ConstInt>(view.valid_shape[i]);
        if (!shape || !view.valid_shape[i] ||
            (valid && (valid->value_ < 0 || valid->value_ > shape->value_))) {
          return false;
        }
      }
      if (!view.stride.empty()) {
        if (view.stride.size() != 2) return false;
        const auto rows = As<ConstInt>(tile->shape_[0])->value_;
        const auto columns = As<ConstInt>(tile->shape_[1])->value_;
        const auto row_stride = As<ConstInt>(view.stride[0]);
        const auto column_stride = As<ConstInt>(view.stride[1]);
        const bool row_major = view.blayout == TileLayout::row_major;
        if (!row_stride || !column_stride || row_stride->value_ != (row_major ? columns : 1) ||
            column_stride->value_ != (row_major ? 1 : rows)) {
          return false;
        }
      }
    }
    const auto lhs_view = tile_view_semantics::GetEffectiveTileView(*lhs);
    const auto rhs_view = tile_view_semantics::GetEffectiveTileView(*rhs);
    for (size_t i = 0; i < 2; ++i) {
      if (lhs_view.valid_shape[i] == rhs_view.valid_shape[i]) continue;
      const auto a_extent = As<ConstInt>(lhs_view.valid_shape[i]);
      const auto b_extent = As<ConstInt>(rhs_view.valid_shape[i]);
      if (!a_extent || !b_extent || a_extent->value_ != b_extent->value_) return false;
    }
    return GetDefinedMemRef(lhs)->size_ == GetDefinedMemRef(rhs)->size_ &&
           TileBufSignature::FromTileType(*lhs) == TileBufSignature::FromTileType(*rhs);
  }

  static StmtPtr InsertMoves(const StmtPtr& body, const std::vector<YieldCopy>& replacements) {
    if (replacements.empty()) return body;
    return transform_utils::RewriteLastYieldStmt(body, [&](const YieldStmtPtr& yield) -> StmtPtr {
      auto rewritten = MutableCopy(yield);
      std::vector<StmtPtr> statements;
      for (const auto& copy : replacements) {
        statements.push_back(copy.transfer);
        rewritten->value_[copy.index] = copy.value;
      }
      statements.push_back(rewritten);
      return SeqStmts::Flatten(std::move(statements), yield->span_);
    });
  }

  const std::set<const Var*>& pinned_bases_;
  std::set<const Var*> selections_;
  StorageNameSupply names_;
  size_t order_ = 0;
  std::map<const Var*, std::pair<size_t, size_t>> definitions_;
};

}  // namespace

StmtPtr MaterializeForwardedVecSelections(const FunctionPtr& func, const std::set<const Var*>& pinned_bases) {
  return ForwardedVecMaterializer(func, pinned_bases).VisitStmt(func->body_);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto
