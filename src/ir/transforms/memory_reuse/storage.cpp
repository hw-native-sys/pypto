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

#include "src/ir/transforms/memory_reuse/storage.h"

#include <any>
#include <cctype>
#include <climits>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

/// True when two MemRefs denote the same codegen-visible byte window.
/// Allocation identity alone is insufficient: one allocation can contain
/// multiple disjoint tile windows and pipeline slots.
bool SamePhysicalWindow(const MemRefPtr& lhs, const MemRefPtr& rhs) {
  if (!lhs || !rhs || !MemRef::SameAllocation(lhs, rhs) || lhs->size_ != rhs->size_ ||
      lhs->slot_count_ != rhs->slot_count_ || !structural_equal(lhs->byte_offset_, rhs->byte_offset_)) {
    return false;
  }

  const bool lhs_has_slot = lhs->slot_index_.has_value() && *lhs->slot_index_;
  const bool rhs_has_slot = rhs->slot_index_.has_value() && *rhs->slot_index_;
  if (lhs_has_slot != rhs_has_slot) return false;
  return !lhs_has_slot || structural_equal(*lhs->slot_index_, *rhs->slot_index_);
}

// Equal unresolved annotations cannot prove that two tiles occupy the same
// hardware memory space, even when their MemRef windows match.
bool SameKnownTileMemorySpace(const TileTypePtr& lhs, const TileTypePtr& rhs) {
  return lhs && rhs && lhs->GetMemorySpace().has_value() && lhs->GetMemorySpace() == rhs->GetMemorySpace();
}

bool SameTransferDescriptor(const TileTypePtr& source, const TileTypePtr& target) {
  return SameKnownTileMemorySpace(source, target) &&
         structural_equal(CloneTypeWithoutMemRef(source), CloneTypeWithoutMemRef(target));
}

// Storage reuse needs the same physical box and addressing layout. Logical
// validity and padding may change when an in-place operation produces a value.
bool SameStorageDescriptor(const TileTypePtr& source, const TileTypePtr& target) {
  if (!SameKnownTileMemorySpace(source, target)) return false;
  const auto physical_type = [](const TileTypePtr& tile) {
    auto type = MutableCopy(tile);
    type->memref_ = std::nullopt;
    auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    view.valid_shape.clear();
    view.pad = PadValue::null;
    type->tile_view_ = std::move(view);
    return type;
  };
  return structural_equal(physical_type(source), physical_type(target));
}

size_t FindSet(std::vector<size_t>& parent, size_t value) {
  while (parent[value] != value) {
    parent[value] = parent[parent[value]];
    value = parent[value];
  }
  return value;
}

void UnionSets(std::vector<size_t>& parent, size_t lhs, size_t rhs) {
  lhs = FindSet(parent, lhs);
  rhs = FindSet(parent, rhs);
  if (lhs != rhs) parent[rhs] = lhs;
}

std::vector<ExprPtr> BufferWindowGeometry::RootOffsets(const BufferViewAnalysis::View& view,
                                                       const Span& span) const {
  if (view.offsets) {
    return {std::make_shared<ConstInt>((*view.offsets)[0], DataType::INDEX, span),
            std::make_shared<ConstInt>((*view.offsets)[1], DataType::INDEX, span)};
  }
  const auto found = root_offsets_.find(view.slice.get());
  INTERNAL_CHECK_SPAN(found != root_offsets_.end(), span)
      << "Internal error: runtime window offsets must dominate their transfer";
  return found->second;
}

StmtPtr BufferWindowGeometry::Index(const AssignStmtPtr& assign, const CallPtr& call) {
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto result = As<TileType>(assign->var_->GetType());
  const auto offsets = As<MakeTuple>(call->args_[2]);
  if (!source || !result || source->shape_.size() != 2 || result->shape_.size() != 2 || !offsets ||
      offsets->elements_.size() != 2) {
    return assign;
  }
  const auto view = views_.Get(assign->var_);
  if (view.offsets || !view.root) return assign;
  const auto root = As<TileType>(view.root->GetType());
  if (!root || root->shape_.size() != 2) return assign;
  const auto parent = RootOffsets(views_.Get(call->args_[0]), call->span_);
  std::vector<StmtPtr> statements;
  std::vector<ExprPtr> cumulative;
  for (size_t axis = 0; axis < 2; ++axis) {
    auto offset = offsets->elements_[axis];
    if (GetScalarDtype(offset) != DataType::INDEX) offset = MakeCast(offset, DataType::INDEX);
    auto sum = AddByteOffsets(parent[axis], offset);
    if (!As<ConstInt>(sum) && !AsVarLike(sum)) {
      auto var = std::make_shared<Var>(names_.Fresh("window_offset_"), sum->GetType(), call->span_);
      statements.push_back(std::make_shared<AssignStmt>(var, sum, call->span_));
      sum = var;
    }
    cumulative.push_back(sum);
  }
  root_offsets_.emplace(call.get(), std::move(cumulative));
  statements.push_back(assign);
  return std::make_shared<SeqStmts>(statements, assign->span_);
}

std::pair<VarPtr, StmtPtr> CreateStorageMove(const VarPtr& source, const MemRefPtr& target_memref,
                                             std::optional<MemorySpace> target_memory,
                                             const std::function<ExprPtr(const ExprPtr&)>& visit_expr,
                                             const std::string& name) {
  INTERNAL_CHECK_SPAN(target_memory.has_value(), source->span_)
      << "Internal error: target TileType must have memory_space for tile.move";
  auto source_tile = GetTileTypeWithMemRef(source->GetType());
  INTERNAL_CHECK_SPAN(source_tile, source->span_)
      << "Internal error: YieldFixup tile.move source must be a TileType with MemRef";
  INTERNAL_CHECK_SPAN(
      !(source_tile->GetMemorySpace() == MemorySpace::Acc && target_memory.value() == MemorySpace::Acc),
      source->span_)
      << "Internal error: MemoryReuse cannot reconcile divergent L0C accumulator buffers with "
         "tile.move; accumulator control-flow values must be coalesced before YieldFixup.";
  auto& op_reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", std::any(target_memory.value())}};
  auto move_call = op_reg.Create("tile.move", {source}, kwargs, source->span_);
  auto moved_type =
      CloneTypeWithMemRefAndRemapExprs(source->GetType(), target_memref, visit_expr, target_memory);
  auto moved_var = std::make_shared<Var>(name, moved_type, source->span_);
  auto move_stmt = std::make_shared<AssignStmt>(moved_var, move_call, source->span_);
  return {moved_var, move_stmt};
}

bool IsA5Target() {
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  return ctx != nullptr && ctx->GetBackendHandler()->GetPtoTargetArch() == "a5";
}

bool IsA5Prelu(const CallPtr& call) { return IsOp(call, "tile.prelu") && IsA5Target(); }

// Resolve a tile Var to its MemRef base pointer (nullptr if it is not a tile or
// has no MemRef yet).  View ops share their source's base, so a view and its
// source resolve to the same pointer here.
const Var* TileMemRefBase(const VarPtr& v) {
  auto t = As<TileType>(v->GetType());
  if (t && t->memref_.has_value() && t->memref_.value()->base_) return t->memref_.value()->base_.get();
  return nullptr;
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto
