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

#include "pypto/codegen/pto/tile_storage_plan.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/dist_tensor_alias_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tile_buf_signature.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {

// Full-MemRef-identity key used by PTOAS memory-planner codegen to decide when
// two tile variables denote the *same* buffer (and must share one tile_buf
// handle so the op writes in place). Same base + byte_offset + size = same
// buffer (loop-carried accumulator, in-place op result). A view shares the
// base but differs in offset and/or size, so it gets a distinct key.
std::string MemRefIdentityKey(const ir::MemRefPtr& memref) {
  std::ostringstream key;
  key << static_cast<const void*>(memref->base_.get()) << '|';
  if (auto off = ir::As<ir::ConstInt>(memref->byte_offset_)) {
    key << "off" << off->value_;
  } else {
    key << "off@" << static_cast<const void*>(memref->byte_offset_.get());
  }
  key << "|sz" << memref->size_;
  return key.str();
}

// These target instructions update input zero, including partially written windows.
bool IsInPlaceInput0DpsOp(const ir::OpPtr& op) {
  return ir::IsOp(op, "tile.scatter") || ir::IsOp(op, "tile.scatter_mask") || ir::IsOp(op, "tile.assemble") ||
         ir::IsOp(op, "tile.tget_scale_addr") || ir::IsOp(op, "tile.write");
}

bool ShareOneMemRefWindow(const std::shared_ptr<const ir::TileType>& lhs,
                          const std::shared_ptr<const ir::TileType>& rhs) {
  auto lhs_memref = ir::GetDefinedMemRef(lhs);
  auto rhs_memref = ir::GetDefinedMemRef(rhs);
  if (!lhs_memref || !rhs_memref) return false;
  if (lhs_memref->base_.get() != rhs_memref->base_.get()) return false;
  // Same base is not enough: two *different* windows of one allocation share it,
  // and aliasing those would silently redirect the write. Require the same byte
  // offset and extent too. The offset is an expression (a slice of a loop-carried
  // tile carries a loop-dependent one), so compare it structurally.
  if (lhs_memref->size_ != rhs_memref->size_) return false;
  const auto& lhs_offset = lhs_memref->byte_offset_;
  const auto& rhs_offset = rhs_memref->byte_offset_;
  if (!lhs_offset || !rhs_offset) return lhs_offset == rhs_offset;
  return ir::structural_equal(lhs_offset, rhs_offset);
}

ir::ExprPtr WritebackInput(const ir::CallPtr& call, const ir::TileTypePtr& result) {
  if (!call || !call->op_ || call->args_.empty()) return nullptr;
  // These instructions update input zero through its current descriptor, even
  // for a partial window. A common base preserves that input's untouched data.
  // Other writebacks may share a descriptor only for an identical typed window.
  if (IsInPlaceInput0DpsOp(call->op_)) {
    const auto input = ir::GetTileTypeWithMemRef(call->args_.front()->GetType());
    if (input && ir::GetDefinedMemRef(result)->base_ == ir::GetDefinedMemRef(input)->base_) {
      return call->args_.front();
    }
    return nullptr;
  }
  // A declared writeback of a differently typed window (e.g. fillpad) still
  // requires its own descriptor. A tensor argument has no Tile destination.
  const auto index = ir::op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
  if (!index) return nullptr;
  const auto& arg = call->args_[*index];
  const auto input = ir::GetTileTypeWithMemRef(arg->GetType());
  if (input && ShareOneMemRefWindow(result, input) &&
      ir::TileBufSignature::FromTileType(*result) == ir::TileBufSignature::FromTileType(*input)) {
    return arg;
  }
  return nullptr;
}

TileAssignmentBinding PrepareAssignment(const ir::CallPtr& call, const ir::TileTypePtr& tile,
                                        bool emit_tile_addr) {
  TileAssignmentBinding result;
  result.writeback_input = WritebackInput(call, tile);
  const bool changes_validity = ir::IsOp(call, "tile.set_validshape");
  result.allocate = !result.writeback_input && !changes_validity;
  result.prepare_destination = !emit_tile_addr && call && call->op_ && !changes_validity &&
                               !ir::IsOp(call, "tile.extract") &&
                               !ir::op_predicates::IsBufferAliasingViewOp(call->op_->name_);
  if (result.prepare_destination && ir::IsOp(call, "tile.move") && call->args_.size() == 1) {
    const auto source = ir::GetTileTypeWithMemRef(call->args_.front()->GetType());
    result.same_window_move =
        source && ShareOneMemRefWindow(source, tile) &&
        ir::structural_equal(ir::CloneTypeWithoutMemRef(source), ir::CloneTypeWithoutMemRef(tile));
  }
  return result;
}

TileExtents GetExtents(const ir::TileType& type, const std::vector<ir::ExprPtr>& shape) {
  if (shape.empty()) return {};
  auto dims = GetPtoTileCoordinates(type, shape);
  if (dims.size() == 1) return {ir::MakeIndexConst(1), dims[0]};
  return {dims[0], dims[1]};
}

class TileStoragePlanner : public ir::IRVisitor {
 public:
  explicit TileStoragePlanner(bool emit_tile_addr) : emit_tile_addr_(emit_tile_addr) {}

  TileStoragePlan Prepare(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params) {
    for (const auto& param : params) {
      auto type = ir::As<ir::TileType>(param->GetType());
      if (!type || type->memory_space_ != ir::MemorySpace::Vec) continue;
      RecordOperands(type);
      if (auto memref = type->memref_.value_or(nullptr)) plan_.parameter_bases.insert(memref->base_.get());
    }
    if (body) VisitStmt(body);
    for (const auto* var : region_producers_) plan_.scope_end[var] = TileStoragePlan::kFunctionEnd;
    for (auto& [stmt, branch] : plan_.branches) {
      for (size_t i = 0; i < branch.results.size(); ++i) {
        branch.results[i] =
            JoinBinding(stmt->return_vars_[i], branch.then_arm.transfers[i], branch.else_arm.transfers[i]);
      }
    }
    return std::move(plan_);
  }

 private:
  void RecordOperands(const ir::TileTypePtr& type) {
    auto [it, inserted] = plan_.operands.try_emplace(type.get());
    if (!inserted) return;
    auto& operands = it->second;
    operands.physical = GetExtents(*type, type->shape_);
    operands.valid = operands.physical;
    if (type->tile_view_ && !type->tile_view_->valid_shape.empty()) {
      const auto& valid = type->tile_view_->valid_shape;
      operands.valid = GetExtents(*type, valid);
      if (std::any_of(valid.begin(), valid.end(),
                      [](const auto& dim) { return dim && !ir::As<ir::ConstInt>(dim); })) {
        operands.restore_valid_column = valid.back();
      }
    }
    if (auto memref = type->memref_.value_or(nullptr); emit_tile_addr_ && memref) {
      operands.address = memref->byte_offset_;
    }
  }

  void VisitStmt_(const ir::AssignStmtPtr& stmt) override {
    auto tile = ir::GetTileTypeWithMemRef(stmt->var_->GetType());
    if (tile || ir::As<ir::TupleType>(stmt->var_->GetType())) region_producers_.push_back(stmt->var_.get());
    if (tile) {
      RecordOperands(tile);
      auto memref = ir::GetDefinedMemRef(tile);
      plan_.allocations.emplace_back(stmt->var_, tile);
      plan_.var_to_memref[stmt->var_.get()] = memref->base_.get();
      if (!emit_tile_addr_) {
        const auto key = MemRefIdentityKey(memref);
        auto [owner, first] = owners_.try_emplace(key, stmt->var_.get());
        plan_.allocation_owner.emplace(stmt->var_.get(), owner->second);
        if (!first && owner->second->GetType() != tile &&
            AllocationType(ir::As<ir::TileType>(owner->second->GetType())) != AllocationType(tile)) {
          mixed_.insert(owner->second);
        }
      }
      const auto call = ir::As<ir::Call>(stmt->value_);
      plan_.assignments.emplace(stmt.get(), PrepareAssignment(call, tile, emit_tile_addr_));
      const auto* entry = call ? ir::LookupOpEntry(call->op_) : nullptr;
      if (call && call->op_ && (!entry || !entry->OutputMemoryInheritsInput())) {
        definitions_[stmt->var_.get()] = ++position_;
      }
      if (memref->slot_count_ > 1 && call && call->GetAttr<bool>(ir::kSoftwarePipelineSlotsAttr, false) &&
          (ir::IsOp(call, "tile.load") || ir::IsOp(call, "tile.create"))) {
        plan_.has_software_pipeline_slots = true;
      }
    }
    IRVisitor::VisitStmt_(stmt);
  }

  const std::string& AllocationType(const ir::TileTypePtr& type) {
    auto [it, inserted] = types_.try_emplace(type.get());
    if (inserted) it->second = FormatAllocatedTileType(*type, DataTypeToMLIR(type->dtype_));
    return it->second;
  }

  void VisitExpr_(const ir::VarPtr& var) override {
    auto type = ir::GetTileTypeWithMemRef(var->GetType());
    if (!type) return;
    auto memref = ir::GetDefinedMemRef(type);
    auto [it, first] = plan_.memref_to_tile_type.try_emplace(memref->base_.get(), type);
    if (first || !type->tile_view_ || type->tile_view_->pad == ir::PadValue::null) return;
    // Preserve the first allocation's validity and the fillpad result's padding.
    const auto& original = it->second;
    auto view = original->tile_view_.value_or(ir::TileView{});
    view.pad = type->tile_view_->pad;
    it->second = std::make_shared<ir::TileType>(original->shape_, original->dtype_, original->memref_, view,
                                                original->memory_space_);
  }

  void VisitExpr_(const ir::IterArgPtr& arg) override { VisitExpr(arg->initValue_); }

  template <class Loop>
  void RecordLoop(const Loop& loop) {
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& arg = loop->iter_args_[i];
      if (auto tile = ir::GetTileTypeWithMemRef(arg->GetType())) {
        const auto* base = ir::GetDefinedMemRef(tile)->base_.get();
        plan_.var_to_memref[arg.get()] = base;
        definitions_[arg.get()] = ++position_;
        if (i < loop->return_vars_.size()) {
          const auto* result = loop->return_vars_[i].get();
          plan_.var_to_memref[result] = base;
          definitions_[result] = position_;
        }
      }
    }
  }

  template <class VisitBody>
  void VisitRegion(VisitBody visit_body) {
    std::vector<const ir::Var*> outer;
    outer.swap(region_producers_);
    visit_body();
    const auto end = ++position_;
    for (const auto* var : region_producers_) plan_.scope_end[var] = end;
    region_producers_ = std::move(outer);
  }

  void VisitStmt_(const ir::RegScopeStmtPtr& scope) override {
    VisitRegion([&] { IRVisitor::VisitStmt_(scope); });
  }

  void VisitStmt_(const ir::ForStmtPtr& loop) override {
    RecordLoop(loop);
    VisitRegion([&] { IRVisitor::VisitStmt_(loop); });
  }
  void VisitStmt_(const ir::WhileStmtPtr& loop) override {
    RecordLoop(loop);
    VisitRegion([&] { IRVisitor::VisitStmt_(loop); });
  }

  TileBranchArm PrepareArm(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& results) {
    const size_t before = position_;
    if (body) VisitRegion([&] { VisitStmt(body); });
    TileBranchArm arm;
    arm.yield = ir::dist_tensor_alias::FindBranchYield(body);
    arm.transfers.resize(results.size());
    for (size_t i = 0; arm.yield && i < results.size() && i < arm.yield->value_.size(); ++i) {
      if (!ir::As<ir::TileType>(results[i]->GetType())) continue;
      const auto& value = arm.yield->value_[i];
      auto& transfer = arm.transfers[i];
      transfer.source = ir::AsVarLike(value);
      if (!ir::GetTileTypeWithMemRef(results[i]->GetType())) continue;  // Borrowed results carry SSA.
      auto tile = ir::As<ir::TileType>(value->GetType());
      transfer.borrowed = tile && !tile->memref_;
      transfer.copy_if_distinct = !emit_tile_addr_ || transfer.borrowed;
      auto def = definitions_.find(transfer.source.get());
      transfer.redirect_producer =
          !emit_tile_addr_ && !transfer.borrowed && def != definitions_.end() && def->second > before;
    }
    return arm;
  }

  void VisitStmt_(const ir::IfStmtPtr& stmt) override {
    VisitExpr(stmt->condition_);
    const bool needs_bindings =
        std::any_of(stmt->return_vars_.begin(), stmt->return_vars_.end(), [](const auto& var) {
          return ir::As<ir::TileType>(var->GetType()) || ir::As<ir::DistributedTensorType>(var->GetType());
        });
    if (!needs_bindings) {
      VisitRegion([&] { VisitStmt(stmt->then_body_); });
      if (stmt->else_body_) VisitRegion([&] { VisitStmt(*stmt->else_body_); });
      return;
    }
    auto& branch = plan_.branches[stmt.get()];
    branch.position = ++position_;
    branch.then_arm = PrepareArm(stmt->then_body_, stmt->return_vars_);
    branch.else_arm = PrepareArm(stmt->else_body_.value_or(nullptr), stmt->return_vars_);
    branch.results.resize(stmt->return_vars_.size());
    for (const auto& result : stmt->return_vars_) {
      VisitExpr(result);
      if (auto tile = ir::GetTileTypeWithMemRef(result->GetType())) {
        RecordOperands(tile);
        definitions_[result.get()] = ++position_;
      }
    }
  }

  [[nodiscard]] TileJoinBinding JoinBinding(const ir::VarPtr& result, const BranchTileTransfer& lhs,
                                            const BranchTileTransfer& rhs) const {
    const auto type = ir::As<ir::TileType>(result->GetType());
    if (!type) return {};
    const auto memref = type->memref_.value_or(nullptr);
    if (!memref) return {nullptr, lhs.source && rhs.source};
    if (emit_tile_addr_) return {};
    const auto key = MemRefIdentityKey(memref);
    const auto owner = owners_.find(key);
    if (owner == owners_.end()) return {};
    TileJoinBinding binding;
    if (!mixed_.count(owner->second)) binding.allocation_owner = owner->second;
    for (const auto& source : {lhs.source, rhs.source}) {
      if (!source) return binding;
      const auto tile = ir::GetTileTypeWithMemRef(source->GetType());
      if (!tile || !type->GetMemorySpace() || tile->GetMemorySpace() != type->GetMemorySpace() ||
          MemRefIdentityKey(ir::GetDefinedMemRef(tile)) != key) {
        return binding;
      }
    }
    binding.use_common_source = true;
    return binding;
  }

  bool emit_tile_addr_;
  size_t position_ = 0;
  std::vector<const ir::Var*> region_producers_;
  std::map<const ir::Var*, size_t> definitions_;
  std::map<std::string, const ir::Var*> owners_;
  std::map<const ir::TileType*, std::string> types_;
  std::set<const ir::Var*> mixed_;
  TileStoragePlan plan_;
};

}  // namespace

TileStoragePlan PrepareTileStorage(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params,
                                   bool emit_tile_addr) {
  return TileStoragePlanner(emit_tile_addr).Prepare(body, params);
}

}  // namespace pypto::codegen
