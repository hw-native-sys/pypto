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

#include <algorithm>
#include <any>
#include <array>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_ir_selection.h"
#include "pypto/ir/transforms/utils/buffer_valid_analysis.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/reserve_buffer_utils.h"
#include "pypto/ir/transforms/utils/storage_access_index.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitVarLike_(const VarPtr& var) override { used_vars.insert(var); }
  void VisitExpr_(const IterArgPtr& var) override { VisitVarLike_(var); }
};

std::optional<size_t> DataWritebackArgIndex(const OpPtr& op, size_t argument_count) {
  if (const auto index = op_predicates::BuiltinWritebackArgIndex(op, argument_count)) {
    return index;
  }
  const auto* entry = LookupOpEntry(op);
  if (argument_count != 0 && entry && entry->GetIRStage() == OpIRStage::Functional &&
      op_predicates::IsBufferAliasingViewOp(op->name_) && entry->MayWriteArg(0)) {
    return 0;
  }
  return std::nullopt;
}

bool WritesTileResult(const CallPtr& call) {
  return call && call->op_ &&
         (DataWritebackArgIndex(call->op_, call->args_.size()) ||
          !op_predicates::IsBufferAliasingViewOp(call->op_->name_));
}

/// Visit every operand through which a functional operation may write data.
/// Explicit writeback bindings also cover operators without argument effects.
template <typename Callback>
void ForEachWrittenArgument(const OpPtr& op, const std::vector<ExprPtr>& arguments,
                            const Callback& callback) {
  const auto writeback = op_predicates::BuiltinWritebackArgIndex(op, arguments.size());
  const auto* entry = LookupOpEntry(op);
  for (size_t i = 0; i < arguments.size(); ++i) {
    if ((writeback && *writeback == i) ||
        (entry && entry->GetIRStage() == OpIRStage::Functional && entry->MayWriteArg(i))) {
      callback(i);
    }
  }
}

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

// Path-compressed equivalence classes for physical windows and parallel copies.
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

// ============================================================================
// Top-down target retargeting
//
// Walks nested control flow (ForStmt -> IfStmt -> branches) and, for each
// ForStmt's iter_arg / return_var chain, pushes the canonical target MemRef
// (= initValue's MemRef, established by InitMemRef) down through the yield
// graph. Producers along the chain are rewritten to land on the target:
//
//   - Unconstrained producers (plain matmul, move, etc.): retyped to write
//     directly to the target MemRef.
//   - Pinned producers (set_output_reuses_input, e.g. matmul_acc): we cannot
//     change their output MemRef, so we recurse onto the pinned-input var
//     and try to place that upstream.
//   - IfStmt return_vars: recurse into both branches' yield values with the
//     same target, and retype the return_var itself.
//
// Liveness check: one analysis walk numbers statements twice, once with each
// IfStmt's then arm first and once with its else arm first. A statement can
// execute after a definition iff it is later in both orderings; sibling arms
// are later in exactly one and are therefore excluded. Frozen read/write
// indexes and an incremental byte/slot observer index avoid rescanning storage
// families. Old-value reads also retain their logical definition version, so
// the complete retargeting analysis remains O(N log N).
// ============================================================================

// One collection and a monotonically increasing suffix avoid repeated name
// scans when fanout transfers share a source or reconciliation runs after reuse.
class StorageNameSupply : public IRVisitor {
 public:
  void Collect(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    for (const auto& parameter : parameters) VisitVarLike_(parameter);
    VisitStmt(body);
  }
  std::string Fresh(const std::string& prefix) {
    std::string result;
    do {
      result = prefix + std::to_string(counter_++);
    } while (!names_.insert(result).second);
    return result;
  }

 protected:
  void VisitVarLike_(const VarPtr& value) override {
    names_.insert(value->name_hint_);
    if (auto memref = GetTypeMemRef(value->GetType()); memref && *memref && (*memref)->base_) {
      names_.insert((*memref)->base_->name_hint_);
    }
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  std::set<std::string> names_;
  size_t counter_ = 0;
};

// Cache each carry at its binding, including unchanged carries. The general
// mutator removes these mappings on loop exit; repeated body uses then remain
// constant-time instead of expanding nested initializer chains.
class StorageBoundaryMutator : public IRMutator {
 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override {
    const auto* context = PassContext::Current();
    if (context == nullptr || !context->GetEnableBufferIR()) return IRMutator::VisitExpr_(argument);
    auto found = var_remap_.find(argument.get());
    if (found != var_remap_.end()) return found->second;
    auto result = IRMutator::VisitExpr_(argument);
    var_remap_[argument.get()] = result;
    return result;
  }
};

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

// Capture cumulative runtime coordinates once at each slice definition. Both
// snapshots and storage transfers then address nested windows without expanding
// their ancestor expressions at every use.
class BufferWindowGeometry {
 public:
  BufferWindowGeometry(const BufferViewAnalysis& views, StorageNameSupply& names)
      : views_(views), names_(names) {}

  std::vector<ExprPtr> RootOffsets(const BufferViewAnalysis::View& view, const Span& span) const {
    if (view.offsets) {
      return {std::make_shared<ConstInt>((*view.offsets)[0], DataType::INDEX, span),
              std::make_shared<ConstInt>((*view.offsets)[1], DataType::INDEX, span)};
    }
    const auto found = root_offsets_.find(view.slice.get());
    INTERNAL_CHECK_SPAN(found != root_offsets_.end(), span)
        << "Internal error: runtime window offsets must dominate their transfer";
    return found->second;
  }

  StmtPtr Index(const AssignStmtPtr& assign, const CallPtr& call) {
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
        // Bind once at the slice's lexical definition. Descendants refer to
        // this SSA scalar instead of repeatedly expanding their ancestor tree.
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

 private:
  const BufferViewAnalysis& views_;
  StorageNameSupply& names_;
  std::unordered_map<const Call*, std::vector<ExprPtr>> root_offsets_;
};

// Preserve an assemble source before its destination can overwrite the same
// storage family. The copy is explicit before planning, so every planner sees
// its lifetime, capacity and no-alias constraints. Exact self-writes stay views.
class SnapshotAssembleSources : public IRMutator {
 public:
  explicit SnapshotAssembleSources(const FunctionPtr& function)
      : views_(function->body_), geometry_(views_, names_) {
    names_.Collect(function->body_, function->params_);
  }

  std::vector<StmtPtr> allocations;

 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override { return argument; }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    const auto call = As<Call>(assign->value_);
    if (call && IsOp(call, "tile.slice")) return geometry_.Index(assign, call);
    if (!call || !IsOp(call, "tile.assemble")) return assign;
    const auto target = GetTileTypeWithMemRef(call->args_[0]->GetType());
    const auto source = AsVarLike(call->args_[1]);
    const auto source_type = source ? GetTileTypeWithMemRef(source->GetType()) : nullptr;
    if (!target || !source_type || source_type->GetMemorySpace() != MemorySpace::Vec ||
        target->GetMemorySpace() != MemorySpace::Vec ||
        GetDefinedMemRef(target)->base_ != GetDefinedMemRef(source_type)->base_) {
      return assign;
    }
    const auto slice = views_.Get(source).slice;
    if (slice && slice->args_[0] == call->args_[0] && structural_equal(slice->args_[2], call->args_[2])) {
      return assign;
    }
    const auto view = views_.Get(source);
    if (HasDisjointWindows(call, source_type, view)) return assign;
    const auto copied_source = view.window ? view.root : source;
    CHECK_SPAN(copied_source && (!view.window || view.slice), call->span_)
        << "Buffer IR overlapping assemble requires a window of a typed allocation";
    const auto copied_type = GetTileTypeWithMemRef(copied_source->GetType());
    const auto size = utils::StaticPhysicalAllocationBytes(copied_type, MemorySpace::Vec, nullptr);
    CHECK_SPAN(size, call->span_) << "Buffer IR assemble snapshots require a static source allocation";
    auto base = std::make_shared<Var>(names_.Fresh("mem_assemble_source_"), GetPtrType(), call->span_);
    auto memory = std::make_shared<MemRef>(base, int64_t{0}, *size, call->span_);
    allocations.push_back(CreateAllocStatement(memory, MemorySpace::Vec));
    auto [snapshot, copy] = CreateStorageMove(
        copied_source, memory, MemorySpace::Vec, [](const ExprPtr& value) { return value; },
        names_.Fresh("assemble_source_"));
    std::vector<StmtPtr> statements{copy};
    if (view.window) {
      const auto offsets = geometry_.RootOffsets(view, call->span_);
      const auto valid = tile_view_semantics::GetEffectiveTileView(*source_type).valid_shape;
      auto window_call = OpRegistry::GetInstance().Create(
          "tile.slice",
          {snapshot, std::make_shared<MakeTuple>(source_type->shape_, call->span_),
           std::make_shared<MakeTuple>(offsets, call->span_),
           std::make_shared<MakeTuple>(valid, call->span_)},
          {}, call->span_);
      const auto old_memory = GetDefinedMemRef(source_type);
      // Rebase element coordinates onto the new allocation, never carry an
      // addressed parent's absolute byte origin into its snapshot.
      INTERNAL_CHECK_SPAN(view.row_bytes, call->span_) << "Internal error: missing window pitch";
      const auto pitch = std::make_shared<ConstInt>(*view.row_bytes, DataType::INDEX, call->span_);
      const auto element =
          std::make_shared<ConstInt>(source_type->dtype_.GetBit() / 8, DataType::INDEX, call->span_);
      const auto relative =
          AddByteOffsets(MulByteOffsets(offsets[0], pitch), MulByteOffsets(offsets[1], element));
      auto window_memory = std::make_shared<MemRef>(base, relative, old_memory->size_, call->span_);
      auto window = std::make_shared<Var>(names_.Fresh("assemble_window_"),
                                          CloneTypeWithMemRef(source_type, window_memory), call->span_);
      statements.push_back(std::make_shared<AssignStmt>(window, window_call, call->span_));
      snapshot = window;
    }
    auto args = call->args_;
    args[1] = snapshot;
    auto rewritten = std::make_shared<Call>(*call);
    rewritten->args_ = std::move(args);
    statements.push_back(std::make_shared<AssignStmt>(assign->var_, rewritten, assign->span_));
    return std::make_shared<SeqStmts>(statements, assign->span_);
  }

 private:
  bool HasDisjointWindows(const CallPtr& call, const TileTypePtr& source_type,
                          const BufferViewAnalysis::View& source) const {
    const auto target = views_.Get(call->args_[0]);
    if (!source.window || !source.root || target.window || !source.offsets || !source.row_bytes ||
        source.row_bytes != target.row_bytes || source_type->shape_.size() != 2) {
      return false;
    }
    // Assemble results retain the destination allocation under a new SSA name.
    // Compare physical origins without changing the logical root used by snapshots.
    const auto root_type = GetTileTypeWithMemRef(source.root->GetType());
    const auto target_type = GetTileTypeWithMemRef(call->args_[0]->GetType());
    if (!root_type || !target_type) return false;
    const auto root_memory = GetDefinedMemRef(root_type);
    const auto target_memory = GetDefinedMemRef(target_type);
    if (root_memory->base_ != target_memory->base_) return false;
    const auto root_origin = As<ConstInt>(root_memory->byte_offset_);
    const auto target_origin = As<ConstInt>(target_memory->byte_offset_);
    if (!root_origin || !target_origin || root_origin->value_ < 0 || target_origin->value_ < 0) return false;
    const auto offsets = As<MakeTuple>(call->args_[2]);
    if (!offsets || offsets->elements_.size() != 2) return false;
    const auto row = As<ConstInt>(offsets->elements_[0]);
    const auto col = As<ConstInt>(offsets->elements_[1]);
    const auto rows = As<ConstInt>(source_type->shape_[0]);
    const auto cols = As<ConstInt>(source_type->shape_[1]);
    if (!row || !col || !rows || !cols || row->value_ < 0 || col->value_ < 0 || rows->value_ <= 0 ||
        cols->value_ <= 0) {
      return false;
    }
    // Match the verifier's conservative first-to-last-byte envelope. Separate
    // columns across multiple rows may still require a snapshot under this
    // contract; do not replace it with a weaker rectangle-overlap check.
    const __int128 pitch = *source.row_bytes;
    const __int128 element = source_type->dtype_.GetBit() / 8;
    const __int128 source_start =
        root_origin->value_ + (*source.offsets)[0] * pitch + (*source.offsets)[1] * element;
    const __int128 target_start = target_origin->value_ + row->value_ * pitch + col->value_ * element;
    const __int128 bytes = (rows->value_ - 1) * pitch + cols->value_ * element;
    return source_start <= target_start ? target_start - source_start >= bytes
                                        : source_start - target_start >= bytes;
  }

  BufferViewAnalysis views_;
  StorageNameSupply names_;
  BufferWindowGeometry geometry_;
};

/// Describes where a TileType Var is defined.
struct VarDef {
  enum Kind { kAssign, kIfReturn, kLoopReturn, kIterArg, kUnknown };
  Kind kind = kUnknown;
  StmtPtr assign_stmt;   // AssignStmt (for kAssign)
  StmtPtr control_stmt;  // IfStmt/ForStmt/WhileStmt (for kIfReturn/kLoopReturn)
  StmtPtr definition_stmt;
  StmtPtr nearest_loop;
  size_t return_idx = 0;  // index into return_vars_ (for kIfReturn/kLoopReturn)
  IterArgPtr iter_arg;    // for kIterArg
  size_t order = 0;       // lexical statement order of the definition
};

/// Data-flow facts for one immutable IfStmt snapshot. Slot identity is
/// (owner, result index), never a MemRef: equal windows can contain different
/// SSA values, and one incoming value can feed several result slots.
struct IfResultInfo {
  VarPtr result;
  std::array<VarPtr, 2> incoming;  // Then, Else. IterArgs keep their own identity.
};

struct IfInfo {
  IfStmtPtr owner;
  std::array<StmtPtr, 2> bodies;
  std::array<YieldStmtPtr, 2> exits;
  std::vector<IfResultInfo> results;
};

/// Walks the IR once to build definitions, lexical statement intervals and
/// per-statement VarLike accesses. Definitions keep only their nearest loop;
/// no per-definition ancestor-chain copies are required.
class DefMapVisitor : public IRVisitor {
 public:
  std::map<VarPtr, VarDef> defs;
  std::map<const Stmt*, size_t> stmt_order;
  std::map<const Stmt*, size_t> stmt_end_order;
  std::map<const Stmt*, std::set<VarPtr>> reads;
  std::map<const Stmt*, std::set<VarPtr>> writes;
  std::map<const Stmt*, std::set<VarPtr>> data_writes;
  std::map<const IfStmt*, IfInfo> if_results;
  std::set<VarPtr> mutable_inputs;
  std::map<VarPtr, size_t> value_last_use;
  std::map<VarPtr, std::set<const Stmt*>> recurring_reads;
  struct WritebackUse {
    VarPtr target;
    std::set<VarPtr> other_inputs;
    const Stmt* statement;
  };
  std::vector<WritebackUse> writeback_uses;
  StmtPtr root;

  void Run(const StmtPtr& body) {
    root = body;
    VisitStmt(body);
    for (auto& [var, def] : defs) {
      (void)var;
      if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kLoopReturn) {
        def.order = stmt_end_order.at(def.control_stmt.get());
      }
    }
    for (const auto& [value, loops] : recurring_reads) {
      for (const auto* loop : loops) {
        value_last_use[value] = std::max(value_last_use[value], stmt_end_order.at(loop));
      }
    }
  }

 protected:
  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    auto previous = current_stmt_;
    current_stmt_ = stmt;
    stmt_order.emplace(stmt.get(), next_order_++);
    IRVisitor::VisitStmt(stmt);
    stmt_end_order.emplace(stmt.get(), next_order_ - 1);
    current_stmt_ = previous;
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) VisitStmt(s);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (As<TileType>(op->var_->GetType())) {
      VarDef d;
      d.kind = VarDef::kAssign;
      d.assign_stmt = op;
      d.definition_stmt = op;
      d.nearest_loop = loop_stack_.empty() ? nullptr : loop_stack_.back();
      d.order = stmt_order.at(op.get());
      defs[op->var_] = d;
      writes[op.get()].insert(op->var_);
      if (WritesTileResult(As<Call>(op->value_))) data_writes[op.get()].insert(op->var_);
    }
    if (op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    const auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
    const auto else_yield = op->else_body_ ? transform_utils::GetLastYieldStmt(*op->else_body_) : nullptr;
    IfInfo info{op, {op->then_body_, op->else_body_.value_or(nullptr)}, {then_yield, else_yield}, {}};
    info.results.reserve(op->return_vars_.size());
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      info.results.push_back(
          {rv,
           {then_yield && i < then_yield->value_.size() ? AsVarLike(then_yield->value_[i]) : nullptr,
            else_yield && i < else_yield->value_.size() ? AsVarLike(else_yield->value_[i]) : nullptr}});
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIfReturn;
      d.control_stmt = op;
      d.definition_stmt = op;
      d.nearest_loop = loop_stack_.empty() ? nullptr : loop_stack_.back();
      d.return_idx = i;
      d.order = stmt_order.at(op.get());
      defs[rv] = d;
    }
    if_results.emplace(op.get(), std::move(info));
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(op->else_body_.value());
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    VisitExpr(op->loop_var_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    VisitLoop(op);
  }

  void VisitExpr_(const IterArgPtr& argument) override { VisitVarLike_(argument); }

  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& loop) {
    for (const auto& argument : loop->iter_args_) {
      VisitExpr(argument->initValue_);
    }
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (!As<TileType>(argument->GetType())) continue;
      VarDef definition;
      definition.kind = VarDef::kIterArg;
      definition.control_stmt = loop;
      definition.definition_stmt = loop;
      definition.nearest_loop = loop;
      definition.return_idx = i;
      definition.iter_arg = argument;
      definition.order = stmt_order.at(loop.get());
      defs[argument] = definition;
    }
    for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
      const auto& result = loop->return_vars_[i];
      if (!As<TileType>(result->GetType())) continue;
      VarDef definition;
      definition.kind = VarDef::kLoopReturn;
      definition.control_stmt = loop;
      definition.definition_stmt = loop;
      definition.nearest_loop = loop;
      definition.return_idx = i;
      definition.order = stmt_order.at(loop.get());
      defs[result] = definition;
    }
    loop_stack_.push_back(loop);
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) VisitExpr(loop->condition_);
    VisitStmt(loop->body_);
    loop_stack_.pop_back();
  }

  void VisitExpr_(const CallPtr& call) override {
    const auto assignment = As<AssignStmt>(current_stmt_);
    RecordInputWrites(call->op_, call->args_, assignment && assignment->value_ == call);
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    RecordInputWrites(submit->op_, submit->args_);
    IRVisitor::VisitExpr_(submit);
  }

  void RecordInputWrites(const OpPtr& op, const std::vector<ExprPtr>& arguments,
                         bool direct_assignment = false) {
    const auto mark = [&](size_t index) {
      if (index >= arguments.size()) return;
      if (const auto value = AsVarLike(arguments[index])) {
        mutable_inputs.insert(value);
        if (current_stmt_) {
          writes[current_stmt_.get()].insert(value);
          data_writes[current_stmt_.get()].insert(value);
        }
      }
    };
    ForEachWrittenArgument(op, arguments, mark);
    if (const auto index = DataWritebackArgIndex(op, arguments.size())) {
      if (direct_assignment && *index < arguments.size()) {
        if (const auto target = AsVarLike(arguments[*index])) {
          VarUseCollector others;
          for (size_t i = 0; i < arguments.size(); ++i) {
            if (i != *index) others.VisitExpr(arguments[i]);
          }
          writeback_uses.push_back({target, std::move(others.used_vars), current_stmt_.get()});
        }
      }
    }
  }

  void VisitStmt_(const WhileStmtPtr& loop) override { VisitLoop(loop); }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  void VisitVarLike_(const VarPtr& op) override {
    if (current_stmt_) {
      reads[current_stmt_.get()].insert(op);
      value_last_use[op] = std::max(value_last_use[op], stmt_order.at(current_stmt_.get()));
      const auto definition = defs.find(op);
      const auto loop =
          definition == defs.end()
              ? loop_stack_.begin()
              : std::upper_bound(loop_stack_.begin(), loop_stack_.end(), definition->second.order,
                                 [&](size_t order, const StmtPtr& candidate) {
                                   return order < stmt_order.at(candidate.get());
                                 });
      if (loop != loop_stack_.end()) recurring_reads[op].insert(loop->get());
    }
    IRVisitor::VisitVarLike_(op);
  }

 private:
  StmtPtr current_stmt_;
  std::vector<StmtPtr> loop_stack_;
  size_t next_order_ = 0;
};

/// A second lexical numbering with every IfStmt's arms visited in reverse.
/// In structured control flow, statement A can execute before statement B iff
/// A precedes B in both this ordering and DefMapVisitor's then-first ordering.
class ReverseBranchOrderVisitor : public IRVisitor {
 public:
  std::map<const Stmt*, size_t> stmt_order;
  std::map<const Stmt*, size_t> stmt_end_order;

  void Run(const StmtPtr& body) { VisitStmt(body); }

 protected:
  // Expressions contain no lexical statements; in particular, do not expand
  // an IterArg's initializer tree while computing statement-only numbering.
  void VisitExpr(const ExprPtr&) override {}

  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    stmt_order.emplace(stmt.get(), next_order_++);
    IRVisitor::VisitStmt(stmt);
    stmt_end_order.emplace(stmt.get(), next_order_ - 1);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
    VisitStmt(op->then_body_);
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

 private:
  size_t next_order_ = 0;
};

/// Range-maximum index over events (primary lexical order, reverse-branch
/// lexical order). A query succeeds when an event lies in the primary interval
/// and is also after the lower bound in the reverse ordering.
class ReachableEventIndex {
 public:
  struct Event {
    size_t primary;
    size_t reverse;
    const Var* excluded_slot = nullptr;
  };

  void Build(const std::vector<std::pair<size_t, size_t>>& events, bool active = true) {
    std::vector<Event> tagged;
    tagged.reserve(events.size());
    for (const auto& event : events) tagged.push_back({event.first, event.second});
    Build(std::move(tagged), active);
  }

  void Build(std::vector<Event> events, bool active = true) {
    primary_.clear();
    summaries_.clear();
    size_ = 0;
    if (events.empty()) return;
    std::sort(events.begin(), events.end(),
              [](const Event& lhs, const Event& rhs) { return lhs.primary < rhs.primary; });
    for (const auto& event : events) {
      if (primary_.empty() || primary_.back() != event.primary) primary_.push_back(event.primary);
    }
    size_ = 1;
    while (size_ < primary_.size()) size_ *= 2;
    summaries_.assign(size_ * 2, Summary{});
    if (active) {
      size_t position = 0;
      for (const auto& event : events) {
        while (primary_[position] < event.primary) ++position;
        auto& leaf = summaries_[size_ + position];
        leaf = Merge(leaf, {event.reverse + 1, event.excluded_slot, true});
      }
    }
    for (size_t i = size_; i-- > 1;) {
      summaries_[i] = Merge(summaries_[i * 2], summaries_[i * 2 + 1]);
    }
  }

  // Activate observations in logical-definition order for loop-entry queries.
  void Activate(size_t primary, size_t reverse, const Var* excluded_slot = nullptr) {
    size_t node = size_ + static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary) -
                                              primary_.begin());
    summaries_[node] = Merge(summaries_[node], {reverse + 1, excluded_slot, true});
    while (node > 1) {
      node /= 2;
      summaries_[node] = Merge(summaries_[node * 2], summaries_[node * 2 + 1]);
    }
  }

  [[nodiscard]] bool HasAfter(size_t primary_begin, size_t primary_end, size_t reverse_lower,
                              const Var* excluded_slot = nullptr) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    size_t left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                      primary_.begin());
    size_t right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                       primary_.begin());
    left += size_;
    right += size_;
    Summary summary;
    while (left < right) {
      if (left & 1U) summary = Merge(summary, summaries_[left++]);
      if (right & 1U) summary = Merge(summary, summaries_[--right]);
      left /= 2;
      right /= 2;
    }
    return summary.maximum > reverse_lower + 1 &&
           !(excluded_slot && summary.uniform && summary.excluded_slot == excluded_slot);
  }

 private:
  struct Summary {
    size_t maximum = 0;
    const Var* excluded_slot = nullptr;
    bool uniform = true;
  };

  static Summary Merge(const Summary& lhs, const Summary& rhs) {
    if (lhs.maximum == 0) return rhs;
    if (rhs.maximum == 0) return lhs;
    return {std::max(lhs.maximum, rhs.maximum), lhs.excluded_slot,
            lhs.uniform && rhs.uniform && lhs.excluded_slot == rhs.excluded_slot};
  }

  std::vector<size_t> primary_;
  std::vector<Summary> summaries_;
  size_t size_ = 0;
};

/// Spatially filtered reads with correlated primary/reverse reachability.
/// Fixed-width byte routing visits O(65) payloads per interval, each containing
/// the existing tagged temporal index. Build and definition-ordered activation
/// cost O(N log N); queries cost O(log N), including allocation/slot lookup.
class WindowedReachableEventIndex {
 public:
  struct Event {
    MemRefPtr window;
    size_t primary;
    size_t reverse;
    const Var* excluded_slot = nullptr;
  };

  void Build(const std::vector<Event>& events, bool active = true) {
    windows_.Clear();
    for (const auto& event : events) {
      windows_.UpdateWindow(event.window, [&](Payload& payload) {
        payload.events.push_back({event.primary, event.reverse, event.excluded_slot});
      });
    }
    windows_.ForEachPayload(
        [&](Payload& payload) { payload.index.Build(std::move(payload.events), active); });
  }

  void Activate(const Event& event) {
    // Every activation uses a window supplied to Build, so spatial routing is
    // frozen while the temporal indexes advance through definition versions.
    windows_.UpdateWindow(event.window, [&](Payload& payload) {
      payload.index.Activate(event.primary, event.reverse, event.excluded_slot);
    });
  }

  [[nodiscard]] bool HasAfter(const MemRefPtr& window, size_t primary_begin, size_t primary_end,
                              size_t reverse_lower, const Var* excluded_slot = nullptr) const {
    return windows_.Any(window, [&](const Payload& payload) {
      return payload.index.HasAfter(primary_begin, primary_end, reverse_lower, excluded_slot);
    });
  }

 private:
  struct Payload {
    std::vector<ReachableEventIndex::Event> events;
    ReachableEventIndex index;
  };
  StorageWindowIndex<Payload> windows_;
};

/// Reachable read events filtered by their logical definition versions.
/// Aliases can share a physical allocation while naming different values over
/// time (notably an IterArg's initializer and the corresponding ForStmt
/// return). A post-If read is an independent observation of the old value only
/// when the handle being read was defined before that IfStmt.
///
/// A forward suffix can include an enclosing if's mutually exclusive sibling.
/// Each immutable definition version therefore indexes primary positions with
/// their maximum reverse position. Build uses O(N log N) time/storage and each
/// query costs O(log N), without scanning the handles in a storage family.
class VersionedReachableEventIndex {
 public:
  struct Event {
    size_t primary;
    size_t definition;
    size_t reverse;
  };

  void Build(std::vector<Event> events) {
    primary_.clear();
    versions_.clear();
    nodes_.assign(1, Node{});
    if (events.empty()) return;
    primary_.reserve(events.size());
    for (const auto& event : events) primary_.push_back(event.primary);
    std::sort(primary_.begin(), primary_.end());
    primary_.erase(std::unique(primary_.begin(), primary_.end()), primary_.end());
    std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
      return std::tie(lhs.definition, lhs.primary) < std::tie(rhs.definition, rhs.primary);
    });
    size_t root = 0;
    for (size_t i = 0; i < events.size();) {
      const auto definition = events[i].definition;
      do {
        const auto& event = events[i];
        const auto position = static_cast<size_t>(
            std::lower_bound(primary_.begin(), primary_.end(), event.primary) - primary_.begin());
        root = Update(root, 0, primary_.size(), position, event.reverse + 1);
        ++i;
      } while (i < events.size() && events[i].definition == definition);
      versions_.push_back({definition, root});
    }
  }

  [[nodiscard]] bool HasAfterFromHandleDefinedBefore(size_t primary_begin, size_t primary_end,
                                                     size_t definition_upper, size_t reverse_lower) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    auto version =
        std::lower_bound(versions_.begin(), versions_.end(), definition_upper,
                         [](const Version& entry, size_t upper) { return entry.definition < upper; });
    if (version == versions_.begin()) return false;
    --version;  // Only handles strictly older than this region are active.
    const auto left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                          primary_.begin());
    const auto right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                           primary_.begin());
    return Query(version->root, 0, primary_.size(), left, right) > reverse_lower + 1;
  }

 private:
  struct Node {
    size_t left = 0;
    size_t right = 0;
    size_t maximum = 0;
  };
  struct Version {
    size_t definition;
    size_t root;
  };

  size_t Update(size_t previous, size_t begin, size_t end, size_t position, size_t reverse) {
    const auto node = nodes_.size();
    nodes_.push_back(nodes_[previous]);
    nodes_[node].maximum = std::max(nodes_[node].maximum, reverse);
    if (end - begin > 1) {
      const auto middle = begin + (end - begin) / 2;
      if (position < middle) {
        const auto child = Update(nodes_[node].left, begin, middle, position, reverse);
        nodes_[node].left = child;
      } else {
        const auto child = Update(nodes_[node].right, middle, end, position, reverse);
        nodes_[node].right = child;
      }
    }
    return node;
  }

  [[nodiscard]] size_t Query(size_t node, size_t begin, size_t end, size_t left, size_t right) const {
    if (node == 0 || right <= begin || end <= left) return 0;
    if (left <= begin && end <= right) return nodes_[node].maximum;
    const auto middle = begin + (end - begin) / 2;
    return std::max(Query(nodes_[node].left, begin, middle, left, right),
                    Query(nodes_[node].right, middle, end, left, right));
  }

  std::vector<size_t> primary_;
  std::vector<Version> versions_;
  std::vector<Node> nodes_;
};

bool IsA5Target() {
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  return ctx != nullptr && ctx->GetBackendHandler()->GetPtoTargetArch() == "a5";
}

bool IsA5Prelu(const CallPtr& call) { return IsOp(call, "tile.prelu") && IsA5Target(); }

// Pipeline membership prevents moving an allocation between stages. Hardware
// hazards additionally forbid creating a mutable alias of its existing bytes.
// Both roles are collected once from the function snapshot before planning.
struct BranchStorageProtection {
  std::set<const Var*> immovable_bases;
  std::set<const Var*> no_mutable_alias_bases;
};

BranchStorageProtection CollectBranchStorageProtection(const FunctionPtr& function, const StmtPtr& body);

/// Index old-value observations before selecting private loop-entry storage.
/// An observation through a handle defined before the loop requires isolation.
/// An observation can recur only in an enclosing loop entered after its
/// logical definition. Index those loop intervals without walking ancestors.
/// The fixed walk and indexed version queries cost O(N log N), including fanout.
class LoopInputIsolation : public IRVisitor {
 public:
  std::map<VarPtr, TypePtr> rewrites;
  std::map<const IterArg*, VarPtr> initializers;
  std::map<const Stmt*, std::vector<StmtPtr>> entries;
  std::vector<StmtPtr> allocations;

  /// Collect one immutable snapshot of actual writes and possible carry
  /// writebacks. The same proof drives isolation and result mutability; merely
  /// entering a loop does not write its initializer. Use a fresh instance for
  /// each snapshot, since projected windows and value versions belong to it.
  void AnalyzeEffects(const StmtPtr& body) {
    views_ = std::make_unique<BufferViewAnalysis>(body);
    const auto collect_pinned = [&](const StmtPtr& statement) {
      if (const auto base = GetPinnedAllocBase(statement)) pinned_bases_.insert(base.get());
    };
    if (const auto sequence = As<SeqStmts>(body)) {
      for (const auto& statement : sequence->stmts_) collect_pinned(statement);
    } else {
      collect_pinned(body);
    }
    ReverseBranchOrderVisitor reverse;
    reverse.Run(body);
    reverse_order_ = std::move(reverse.stmt_order);
    VisitStmt(body);
    writes_index_.Build(write_events_);
  }

  [[nodiscard]] std::set<VarPtr> WrittenLoopInitializers() const {
    std::set<VarPtr> written;
    const auto collect = [&](const auto& loop) {
      for (const auto& argument : loop->iter_args_) {
        const auto initial = AsVarLike(argument->initValue_);
        if (!initial) continue;
        ForEachStorageWindow(initial, [&](const MemRefPtr& window) {
          if (MayWriteInLoop(loop, window)) written.insert(initial);
        });
      }
    };
    for (const auto& statement : loops_) {
      if (const auto loop = As<ForStmt>(statement)) collect(loop);
      if (const auto loop = As<WhileStmt>(statement)) collect(loop);
    }
    return written;
  }

  void Plan(const StmtPtr& body, const std::vector<VarPtr>& parameters,
            const std::vector<ParamDirection>& directions) {
    names_.Collect(body, parameters);
    AnalyzeEffects(body);
    std::vector<StorageAccessIndex::Event> incoming_windows;
    for (size_t i = 0; i < parameters.size(); ++i) {
      // Read-only inputs belong to the caller. Out/InOut storage may be a
      // deliberate continuation; independent old-value uses still isolate it.
      if (directions.at(i) != ParamDirection::In) continue;
      if (const auto window = GetTypeMemRef(parameters[i]->GetType()); window && *window) {
        incoming_windows.push_back({*window, 1, 1});
      } else if (As<TileType>(parameters[i]->GetType())) {
        // An opaque incoming tile cannot prove separation from a written carry.
        incoming_windows.push_back({nullptr, 1, 1});
      }
    }
    incoming_storage_.Build(incoming_windows);
    struct Observation {
      WindowedReachableEventIndex::Event event;
      size_t definition;
    };
    std::vector<Observation> observations;
    std::vector<WindowedReachableEventIndex::Event> reachable_events;
    std::vector<WindowedReachableEventIndex::Event> recurring_events;
    for (const auto& read : recurring_reads_) {
      const auto begin = positions_.at(read.loop);
      const auto end = loop_ends_.at(read.loop);
      ForEachStorageWindow(read.value, [&](const MemRefPtr& window) {
        recurring_events.push_back({window, begin, end, read.excluded_slot});
      });
    }
    for (const auto& read : reads_) {
      auto definition = definitions_.find(read.value.get());
      const size_t version = definition == definitions_.end() ? 0 : definition->second;
      const size_t reverse = read_reverse_.at(read.position);
      ForEachStorageWindow(read.value, [&](const MemRefPtr& window) {
        WindowedReachableEventIndex::Event event{window, read.position, reverse, read.excluded_slot};
        reachable_events.push_back(event);
        observations.push_back({std::move(event), version});
      });
    }
    recurring_index_.Build(recurring_events);
    reachable_index_.Build(reachable_events, false);
    std::sort(observations.begin(), observations.end(),
              [](const auto& a, const auto& b) { return a.definition < b.definition; });
    size_t next = 0;
    for (const auto& loop : loops_) {
      // The preorder loop sweep activates only handles naming older values.
      // A range maximum then tests both branch orderings in O(log N).
      while (next < observations.size() && observations[next].definition < positions_.at(loop.get())) {
        const auto& event = observations[next++];
        reachable_index_.Activate(event.event);
      }
      if (auto for_loop = As<ForStmt>(loop)) PlanLoop(for_loop);
      if (auto while_loop = As<WhileStmt>(loop)) PlanLoop(while_loop);
    }
  }

 protected:
  void VisitStmt(const StmtPtr& statement) override {
    auto saved = current_;
    const auto saved_reverse = current_reverse_;
    current_reverse_ = reverse_order_.at(statement.get()) + 1;
    current_ = ++order_;
    positions_[statement.get()] = current_;
    IRVisitor::VisitStmt(statement);
    current_ = saved;
    current_reverse_ = saved_reverse;
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    auto source = AsVarLike(statement->value_);
    const auto call = As<Call>(statement->value_);
    const auto writeback =
        call ? op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size()) : std::nullopt;
    const bool view = call && call->op_ && !writeback && !call->args_.empty() &&
                      op_predicates::IsBufferAliasingViewOp(call->op_->name_);
    if (view) source = AsVarLike(call->args_[0]);
    if (writeback) {
      if (const auto input = AsVarLike(call->args_[*writeback])) {
        storage_windows_[statement->var_.get()] = StorageWindow(input);
      }
    }
    if (source && GetTileTypeWithMemRef(source->GetType())) {
      const bool metadata_only = !WritesTileResult(call);
      if (metadata_only) {
        // Only data-preserving aliases inherit their source's value version.
        // A writing alias retains storage but defines a new data value.
        auto origin = definitions_.find(source.get());
        definitions_[statement->var_.get()] = origin == definitions_.end() ? 0 : origin->second;
      }
      const auto output_tile = GetTileTypeWithMemRef(statement->var_->GetType());
      const auto source_tile = GetTileTypeWithMemRef(source->GetType());
      if (SameKnownTileMemorySpace(output_tile, source_tile)) {
        const auto input_window = StorageWindow(source);
        const auto output_window =
            view ? MakeAliasingMemRef(input_window, ComputeViewByteOffset(call, source_tile),
                                      GetDefinedMemRef(output_tile)->size_, statement->var_->span_)
                 : input_window;
        // Partial views retain their projected byte/slot window even though
        // they do not denote the same complete value as their source.
        storage_windows_[statement->var_.get()] = output_window;
        if (metadata_only && SamePhysicalWindow(output_window, input_window)) {
          value_origins_[statement->var_.get()] = ValueOrigin(source);
        }
      }
      if (metadata_only) {
        // The first operand supplies metadata. Other arguments can still have
        // data reads or nested Call/Submit effects and must be visited.
        if (call) {
          RecordInputWrites(call->op_, call->args_);
          for (size_t i = 1; i < call->args_.size(); ++i) VisitExpr(call->args_[i]);
        }
        return;
      }
    }
    definitions_[statement->var_.get()] = current_;
    if (WritesTileResult(call) || As<Submit>(statement->value_)) RecordWrite(statement->var_);
    VisitExpr(statement->value_);
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    const auto then_yield = transform_utils::GetLastYieldStmt(branch->then_body_);
    const auto else_yield =
        branch->else_body_ ? transform_utils::GetLastYieldStmt(*branch->else_body_) : nullptr;
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      definitions_[result.get()] = ++order_;
      const auto tile = GetTileTypeWithMemRef(result->GetType());
      if (!tile) continue;
      const auto target = GetDefinedMemRef(tile);
      const bool fixed = target->is_pinned_ || pinned_bases_.count(target->base_.get());
      const auto lhs =
          then_yield && i < then_yield->value_.size() ? AsVarLike(then_yield->value_[i]) : nullptr;
      const auto rhs =
          else_yield && i < else_yield->value_.size() ? AsVarLike(else_yield->value_[i]) : nullptr;
      const auto lhs_tile = lhs ? GetTileTypeWithMemRef(lhs->GetType()) : nullptr;
      const auto rhs_tile = rhs ? GetTileTypeWithMemRef(rhs->GetType()) : nullptr;
      const bool common = lhs_tile && rhs_tile && ValueOrigin(lhs) == ValueOrigin(rhs) &&
                          SameKnownTileMemorySpace(tile, lhs_tile) &&
                          SameKnownTileMemorySpace(tile, rhs_tile) &&
                          SamePhysicalWindow(StorageWindow(lhs), StorageWindow(rhs));
      if (common && (!fixed || SamePhysicalWindow(target, StorageWindow(lhs)))) {
        storage_windows_[result.get()] = StorageWindow(lhs);
        value_origins_[result.get()] = ValueOrigin(lhs);
      }
      if (fixed) {
        for (const auto& yield : {then_yield, else_yield}) {
          const auto input = yield && i < yield->value_.size() ? AsVarLike(yield->value_[i]) : nullptr;
          const auto input_tile = input ? GetTileTypeWithMemRef(input->GetType()) : nullptr;
          if (!input_tile || !SamePhysicalWindow(target, StorageWindow(input))) {
            if (yield) RecordWrite(result, positions_.at(yield.get()), reverse_order_.at(yield.get()) + 1);
          }
        }
      }
    }
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    VisitLoop(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override { VisitLoop(loop); }
  void VisitExpr_(const CallPtr& call) override {
    RecordInputWrites(call->op_, call->args_);
    const auto excluded_slot = AssembleOverwrittenCarry(call);
    if (!excluded_slot) {
      IRVisitor::VisitExpr_(call);
      return;
    }
    // Assemble reads its old target only outside the inserted source window.
    // Keep that read for every other carry and retain every source/attr read.
    RecordRead(AsVarLike(call->args_[0]), excluded_slot);
    for (size_t i = 1; i < call->args_.size(); ++i) VisitExpr(call->args_[i]);
    for (const auto& [key, value] : call->attrs_) {
      (void)key;
      ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpr(expression); });
    }
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    RecordInputWrites(submit->op_, submit->args_);
    IRVisitor::VisitExpr_(submit);
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitVarLike_(const VarPtr& value) override { RecordRead(value, nullptr); }

 private:
  [[nodiscard]] bool MayWriteInLoop(const StmtPtr& loop, const MemRefPtr& window) const {
    return writes_index_.MayOverlapAfter(window, positions_.at(loop.get()), reverse_order_.at(loop.get()),
                                         loop_ends_.at(loop.get()));
  }

  void RecordRead(const VarPtr& value, const Var* excluded_slot) {
    if (GetTileTypeWithMemRef(value->GetType())) {
      reads_.push_back({current_, value, excluded_slot});
      read_reverse_[current_] = current_reverse_;
      const auto definition = definitions_.find(value.get());
      const size_t version = definition == definitions_.end() ? 0 : definition->second;
      // The stack is ordered by header position. The first header after the
      // definition is the outermost loop that can repeat this same value's read.
      const auto loop =
          std::upper_bound(loop_stack_.begin(), loop_stack_.end(), version,
                           [](size_t position, const auto& entry) { return position < entry.first; });
      if (loop != loop_stack_.end()) recurring_reads_.push_back({value, loop->second, excluded_slot});
    }
  }

  /// A whole carry result replaces precisely its original slice of the target.
  /// Only that producing slot can ignore the target's preserved outside bytes;
  /// this is not a declaration that the whole target has stopped being read.
  /// A slot may reserve extra capacity: the proof concerns its typed value,
  /// not a promise that every reserved byte was written. Independent reads of
  /// padding, other slices or the whole old target retain their ordinary tag.
  /// The complete typed value must also be valid: Vec insertion writes only
  /// valid elements, whereas a redirected loop producer can write its full box.
  [[nodiscard]] const Var* AssembleOverwrittenCarry(const CallPtr& call) const {
    if (!call || !IsOp(call, "tile.assemble") || call->args_.size() != 3) return nullptr;
    const auto target = AsVarLike(call->args_[0]);
    const auto source = AsVarLike(call->args_[1]);
    if (!target || !source) return nullptr;
    const auto found = returned_slots_.find(ValueOrigin(source).get());
    if (found == returned_slots_.end()) return nullptr;
    const auto& slot = found->second;
    const auto source_tile = GetTileTypeWithMemRef(source->GetType());
    const auto target_tile = GetTileTypeWithMemRef(target->GetType());
    const auto slot_tile = GetTileTypeWithMemRef(slot->GetType());
    const auto source_window = StorageWindow(source);
    if (!source_tile || !target_tile || !slot_tile || !SameStorageDescriptor(source_tile, slot_tile) ||
        !SamePhysicalWindow(source_window, StorageWindow(slot))) {
      return nullptr;
    }
    const auto full_valid = [](const std::shared_ptr<const TileType>& tile) {
      const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
      if (view.valid_shape.size() != tile->shape_.size()) return false;
      for (size_t i = 0; i < tile->shape_.size(); ++i) {
        if (!structural_equal(view.valid_shape[i], tile->shape_[i])) return false;
      }
      return true;
    };
    if (!full_valid(source_tile) || !full_valid(slot_tile)) return nullptr;
    const auto slice = views_->Get(source).slice;
    if (!slice || slice->args_.size() < 3 || !structural_equal(call->args_[2], slice->args_[2])) {
      return nullptr;
    }
    const auto parent = AsVarLike(slice->args_[0]);
    const auto parent_tile = parent ? GetTileTypeWithMemRef(parent->GetType()) : nullptr;
    const auto slice_tile = As<TileType>(slice->GetType());
    // A narrowed type alone does not retain its root's row pitch. Coordinate
    // equality needs the same value/view lineage, not just coincident bytes.
    if (!parent_tile || !slice_tile || ValueOrigin(parent) != ValueOrigin(target) ||
        !SameStorageDescriptor(parent_tile, target_tile) || !SameStorageDescriptor(slice_tile, source_tile) ||
        !SamePhysicalWindow(StorageWindow(parent), StorageWindow(target))) {
      return nullptr;
    }
    const auto inserted = MakeAliasingMemRef(StorageWindow(target), ComputeViewByteOffset(slice, parent_tile),
                                             source_window->size_, call->span_);
    return SamePhysicalWindow(inserted, source_window) ? slot.get() : nullptr;
  }

  [[nodiscard]] MemRefPtr StorageWindow(const VarPtr& value) const {
    const auto found = storage_windows_.find(value.get());
    if (found != storage_windows_.end() && found->second) return found->second;
    const auto tile = GetTileTypeWithMemRef(value->GetType());
    return tile ? GetDefinedMemRef(tile) : nullptr;
  }
  [[nodiscard]] VarPtr ValueOrigin(const VarPtr& value) const {
    const auto found = value_origins_.find(value.get());
    return found == value_origins_.end() ? value : found->second;
  }
  template <typename Callback>
  void ForEachStorageWindow(const ExprPtr& value, Callback&& callback) const {
    const auto tile = GetTileTypeWithMemRef(value->GetType());
    if (!tile) return;
    const auto original = GetDefinedMemRef(tile);
    callback(original);
    if (const auto variable = AsVarLike(value)) {
      const auto projected = StorageWindow(variable);
      if (projected && !SamePhysicalWindow(original, projected)) callback(projected);
    }
  }
  void RecordWrite(const VarPtr& value, size_t primary, size_t reverse) {
    ForEachStorageWindow(value, [&](const MemRefPtr& window) {
      write_events_.push_back({window, primary, reverse});
      ++write_generations_[window->base_.get()];
    });
  }
  void RecordWrite(const VarPtr& value) { RecordWrite(value, current_, current_reverse_); }
  void RecordInputWrites(const OpPtr& op, const std::vector<ExprPtr>& arguments) {
    ForEachWrittenArgument(op, arguments, [&](size_t index) {
      const auto value = AsVarLike(arguments[index]);
      if (!value) return;
      if (GetTileTypeWithMemRef(value->GetType())) {
        RecordWrite(value);
      } else if (const auto window = GetTypeMemRef(value->GetType()); window && *window) {
        // Known GM writes retain their window, including typed DDR aliases.
        write_events_.push_back({*window, current_, current_reverse_});
        ++write_generations_[(*window)->base_.get()];
      } else if (As<PtrType>(value->GetType())) {
        // An unbound written pointer may alias any carried allocation. Unknown
        // GM/scalar values do not imply such an on-chip pointer write.
        write_events_.push_back({nullptr, current_, current_reverse_});
        ++opaque_write_generation_;
      }
    });
  }

  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& loop) {
    loops_.push_back(loop);
    const size_t opaque_generation = opaque_write_generation_;
    std::vector<size_t> generations;
    for (const auto& argument : loop->iter_args_) {
      VisitExpr(argument->initValue_);
      definitions_[argument.get()] = current_;
      const auto initial = AsVarLike(argument->initValue_);
      if (initial) {
        storage_windows_[argument.get()] = StorageWindow(initial);
        value_origins_[argument.get()] = ValueOrigin(initial);
      }
      const auto window = initial ? StorageWindow(initial) : nullptr;
      generations.push_back(window ? write_generations_[window->base_.get()] : 0);
    }
    // Initializers execute once at entry, outside this loop's recurrence.
    loop_stack_.emplace_back(positions_.at(loop.get()), loop.get());
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      // Separate condition reads from initializer reads at the loop header.
      current_ = ++order_;
      VisitExpr(loop->condition_);
    }
    VisitStmt(loop->body_);
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    std::vector<bool> unchanged_values;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      const auto source = yield && i < yield->value_.size() ? AsVarLike(yield->value_[i]) : nullptr;
      const bool unchanged = source && ValueOrigin(source) == ValueOrigin(argument);
      unchanged_values.push_back(unchanged);
      if (!unchanged && yield) {
        RecordWrite(argument, positions_.at(yield.get()), reverse_order_.at(yield.get()) + 1);
      }
    }
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto initial = AsVarLike(loop->iter_args_[i]->initValue_);
      if (initial && i < loop->return_vars_.size()) {
        const auto& result = loop->return_vars_[i];
        const auto window = StorageWindow(initial);
        storage_windows_[result.get()] = window;
        returned_slots_[result.get()] = loop->iter_args_[i];
        // A changing nested loop still names its entry storage for effect
        // queries, but its produced value is not a pass-through initializer.
        if (unchanged_values[i] && window && opaque_generation == opaque_write_generation_ &&
            generations[i] == write_generations_[window->base_.get()]) {
          value_origins_[result.get()] = ValueOrigin(initial);
        }
      }
    }
    loop_stack_.pop_back();
    loop_ends_[loop.get()] = order_;
    for (const auto& result : loop->return_vars_) definitions_[result.get()] = ++order_;
  }

  template <typename LoopPtr>
  void PlanLoop(const LoopPtr& loop) {
    const auto position = positions_.at(loop.get());
    std::set<size_t> isolate;
    std::vector<StorageAccessIndex::Event> carry_windows;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      ForEachStorageWindow(loop->iter_args_[i]->initValue_,
                           [&](const MemRefPtr& window) { carry_windows.push_back({window, i + 1, 1}); });
    }
    StorageAccessIndex other_carries;
    other_carries.Build(carry_windows);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& initial = loop->iter_args_[i]->initValue_;
      auto tile = GetTileTypeWithMemRef(initial->GetType());
      if (!tile) continue;
      // Before binding closure, a phi/view can still name a provisional
      // allocation while denoting its source's storage. Reads, writes and
      // queries must use the same two windows, including for nested loops.
      ForEachStorageWindow(initial, [&](const MemRefPtr& window) {
        // Loop slots are independent mutable bindings, even when the present
        // body passes them through. A later consumer may update just one result.
        if (other_carries.MayOverlapAfter(window, 0, 0, i) ||
            other_carries.MayOverlapAfter(window, i + 1, 0)) {
          isolate.insert(i);
        }
        // Old observers matter only when an iteration can change this window.
        // Native writes and implicit changed-yield copies use the same index;
        // read-only nested loops and metadata aliases require no entry copy.
        if (!MayWriteInLoop(loop, window)) {
          return;
        }
        // Recurrence intervals use (loop begin, loop end) in the range-maximum
        // index: some begin <= position must have end >= position. A read outside
        // all such loops cannot observe this carry on a later iteration.
        const auto* slot = loop->iter_args_[i].get();
        const bool nested_observation = recurring_index_.HasAfter(window, 0, position, position - 1, slot);
        if (incoming_storage_.MayOverlapAfter(window, 0, 0) || nested_observation ||
            reachable_index_.HasAfter(window, position + 1, order_, reverse_order_.at(loop.get()), slot)) {
          isolate.insert(i);
        }
      });
    }
    for (const auto index : isolate) {
      const auto& argument = loop->iter_args_[index];
      auto source = AsVarLike(argument->initValue_);
      CHECK_SPAN(source, loop->span_) << "Buffer IR loop initializers must be storage variables";
      const auto tile = GetTileTypeWithMemRef(source->GetType());
      const auto memory = tile->GetMemorySpace();
      CHECK_SPAN(memory && IsTileMoveEverSupported(*memory, *memory), loop->span_)
          << "Buffer IR cannot preserve an independently live loop input without a same-space copy; "
             "produce an independent initial tile for this carry";
      auto base = std::make_shared<Var>(names_.Fresh("mem_carry_input_"), GetPtrType(), source->span_);
      auto memref = std::make_shared<MemRef>(base, int64_t{0}, GetDefinedMemRef(tile)->size_, source->span_);
      auto [initial, transfer] = CreateStorageMove(
          source, memref, memory, [](const ExprPtr& e) { return e; }, names_.Fresh("carry_input_"));
      allocations.push_back(CreateAllocStatement(memref, *memory));
      entries[loop.get()].push_back(transfer);
      initializers[argument.get()] = initial;
      rewrites[argument] = CloneTypeWithMemRef(argument->GetType(), memref, memory);
      if (index < loop->return_vars_.size()) {
        const auto& result = loop->return_vars_[index];
        rewrites[result] = CloneTypeWithMemRef(result->GetType(), memref, memory);
      }
    }
  }

  size_t order_ = 0;
  size_t current_ = 0;
  size_t current_reverse_ = 0;
  std::map<const Stmt*, size_t> reverse_order_;
  std::map<size_t, size_t> read_reverse_;
  WindowedReachableEventIndex reachable_index_;
  std::vector<std::pair<size_t, const Stmt*>> loop_stack_;
  std::map<const Stmt*, size_t> loop_ends_;
  struct RecurringRead {
    VarPtr value;
    const Stmt* loop;
    const Var* excluded_slot;
  };
  std::vector<RecurringRead> recurring_reads_;
  WindowedReachableEventIndex recurring_index_;
  StorageNameSupply names_;
  std::map<const Stmt*, size_t> positions_;
  std::map<const Var*, size_t> definitions_;
  struct Read {
    size_t position;
    VarPtr value;
    const Var* excluded_slot;
  };
  std::vector<Read> reads_;
  std::vector<StmtPtr> loops_;
  StorageAccessIndex incoming_storage_;
  std::set<const Var*> pinned_bases_;
  // Effect proofs only: storage windows project pending bindings, while value
  // tokens distinguish changed results from aliases. Neither relation rewrites
  // SSA identity or determines an if result's eventual destination.
  std::map<const Var*, MemRefPtr> storage_windows_;
  std::map<const Var*, VarPtr> value_origins_;
  std::map<const Var*, IterArgPtr> returned_slots_;
  std::unique_ptr<BufferViewAnalysis> views_;
  // A base-wide generation conservatively prevents forwarding a changed loop
  // result as its initializer. Actual isolation uses byte/slot-sensitive events.
  std::map<const Var*, size_t> write_generations_;
  std::vector<StorageAccessIndex::Event> write_events_;
  size_t opaque_write_generation_ = 0;
  StorageAccessIndex writes_index_;
};

/// Plans top-down retypes. Produces (old Var -> new Type) map.
class TopDownRetargeter {
 public:
  /// Plan the complete function against one immutable SSA snapshot. Loop and
  /// accumulator requirements are collected before optional if-result choices;
  /// no IR is rewritten until the caller applies the returned map. All memory
  /// planners consume the same settled result/edge contract.
  std::map<VarPtr, TypePtr> PlanStorage(const FunctionPtr& function, const StmtPtr& func_body,
                                        std::vector<StmtPtr>* allocations) {
    BuildAnalysis(func_body, /*index_branch_storage=*/true);
    auto protection = CollectBranchStorageProtection(function, func_body);
    immovable_branch_bases_ = std::move(protection.immovable_bases);
    no_mutable_alias_branch_bases_ = std::move(protection.no_mutable_alias_bases);
    std::vector<StorageAccessIndex::Event> readonly_caller_windows;
    for (size_t i = 0; i < function->params_.size(); ++i) {
      const auto& parameter = function->params_[i];
      if (const auto memory = GetTypeMemRef(parameter->GetType());
          function->param_directions_[i] == ParamDirection::In && memory && *memory) {
        readonly_caller_windows.push_back({*memory, 1, 1});
      }
      // A Ptr ABI parameter can name a directly supplied result window. Its
      // binding is caller-owned, just like the storage of a Tile parameter.
      if (As<PtrType>(parameter->GetType())) pinned_branch_bases_.insert(parameter.get());
      if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
        const auto* base = GetDefinedMemRef(tile)->base_.get();
        pinned_branch_bases_.insert(base);
      }
    }
    readonly_caller_storage_.Build(readonly_caller_windows);
    // InitMemRef clears MemRef::is_pinned_; the hoisted alloc then carries the
    // authoritative declaration contract. Preserve those output bindings.
    if (auto body = As<SeqStmts>(func_body)) {
      for (const auto& stmt : body->stmts_) {
        if (auto base = GetPinnedAllocBase(stmt)) {
          pinned_branch_bases_.insert(base.get());
          fixed_branch_bases_.insert(base.get());
        }
        if (auto assign = As<AssignStmt>(stmt); assign && IsOp(As<Call>(assign->value_), "tile.alloc")) {
          owned_branch_bases_.insert(assign->var_.get());
        }
      }
    }
    for (const auto& [node, info] : if_results_) {
      (void)node;
      for (const auto& slot : info.results) {
        if (const auto memory = GetTypeMemRef(slot.result->GetType());
            memory && *memory && (*memory)->is_pinned_) {
          pinned_branch_bases_.insert((*memory)->base_.get());
        }
      }
    }
    VisitLoopConstraints(func_body);
    VisitIfPhisForAccumulator(func_body);
    RememberRequiredTargets();
    VisitBranchStorage(func_body, nullptr, /*select_results=*/false);
    IndexFutureWrites();
    // Backward loop propagation is a candidate optimization, not permission
    // to destroy a logical result. Validate against all pending writes before
    // committing it. A bounded retry can only reuse the first candidate's
    // producer destinations (or their original storage), never invent new ones.
    for (const auto& [value, type] : rewrites_) {
      const auto definition = defs_.find(value);
      if (definition != defs_.end() && definition->second.kind == VarDef::kIfReturn &&
          MayClobberResult(As<IfStmt>(definition->second.control_stmt), value,
                           GetDefinedMemRef(GetTileTypeWithMemRef(type)))) {
        blocked_branch_results_.insert(value.get());
      }
    }
    if (!blocked_branch_results_.empty()) {
      for (const auto& [value, type] : rewrites_) {
        if (const auto tile = GetTileTypeWithMemRef(type)) {
          replay_targets_[value.get()] = GetDefinedMemRef(tile);
        }
      }
      RollbackRewrites(0);
      BuildAnalysis(func_body, /*index_branch_storage=*/true);
      replaying_ = true;
      VisitLoopConstraints(func_body);
      VisitIfPhisForAccumulator(func_body);
      RememberRequiredTargets();
      VisitBranchStorage(func_body, nullptr, /*select_results=*/false);
      replaying_ = false;
      IndexFutureWrites();
      for (const auto& [value, type] : rewrites_) {
        const auto definition = defs_.find(value);
        if (definition != defs_.end() && definition->second.kind == VarDef::kIfReturn) {
          CHECK_SPAN(!MayClobberResult(As<IfStmt>(definition->second.control_stmt), value,
                                       GetDefinedMemRef(GetTileTypeWithMemRef(type))),
                     value->span_)
              << "Cannot preserve an if result across conflicting required storage writes";
        }
      }
    }
    VisitBranchStorage(func_body, allocations);
    for (auto& [var, type] : rewrites_) {
      (void)var;
      auto tile = GetTileTypeWithMemRef(type);
      auto original = GetDefinedMemRef(tile);
      auto destination = ResolveBranchTarget(original);
      if (destination != original) type = CloneTypeWithMemRef(type, destination, tile->GetMemorySpace());
    }
    return std::move(rewrites_);
  }

 private:
  void RememberRequiredTargets() {
    required_branch_targets_.clear();
    for (const auto& [var, type] : rewrites_) {
      const auto definition = defs_.find(var);
      if (definition != defs_.end() && definition->second.kind == VarDef::kIfReturn) {
        const auto window = GetDefinedMemRef(GetTileTypeWithMemRef(type));
        const auto original = GetTypeMemRef(var->GetType()).value_or(nullptr);
        const bool fixed =
            original && (original->is_pinned_ || fixed_branch_bases_.count(original->base_.get()));
        const auto owner = required_target_owners_.find(var.get());
        required_branch_targets_.emplace(
            var.get(),
            RequiredBranchTarget{fixed || owner == required_target_owners_.end() ? nullptr : owner->second,
                                 window});
      }
    }
  }

  void IndexFutureWrites() {
    std::vector<StorageAccessIndex::Event> future_writes;
    for (const auto& [statement, values] : data_writes_) {
      for (const auto& value : values) {
        const bool tile = As<TileType>(value->GetType()) != nullptr;
        auto memory = tile ? CurrentMemRef(value) : GetTypeMemRef(value->GetType()).value_or(nullptr);
        // Keep known GM writes: typed DDR tiles can share their allocation.
        // Unknown GM/scalar storage is not an opaque on-chip write, whereas
        // an untyped pointer effect can still reach a tile allocation.
        if (!tile && !memory && !As<PtrType>(value->GetType())) continue;
        // An unplaced fresh result cannot clobber an existing allocation.
        // Opaque externally supplied written operands remain conservative.
        const auto definition = defs_.find(value);
        if (!memory && definition != defs_.end() && definition->second.kind == VarDef::kAssign) continue;
        future_writes.push_back({memory, stmt_order_.at(statement), reverse_stmt_order_.at(statement)});
      }
    }
    for (const auto& [node, info] : if_results_) {
      (void)node;
      for (const auto& slot : info.results) {
        const auto tile = CurrentTileType(slot.result);
        if (!tile) continue;
        const auto original = GetTypeMemRef(slot.result->GetType()).value_or(GetDefinedMemRef(tile));
        const auto required = required_branch_targets_.find(slot.result.get());
        if (required == required_branch_targets_.end() && !original->is_pinned_ &&
            !fixed_branch_bases_.count(original->base_.get())) {
          continue;
        }
        const auto destination =
            required == required_branch_targets_.end() ? original : ResolveRequiredTarget(required->second);
        for (size_t arm = 0; arm < slot.incoming.size(); ++arm) {
          const auto source = slot.incoming[arm];
          if (info.exits[arm] && (!source || !SamePhysicalWindow(CurrentMemRef(source), destination))) {
            future_writes.push_back({destination, stmt_order_.at(info.exits[arm].get()),
                                     reverse_stmt_order_.at(info.exits[arm].get())});
            if (source) {
              const auto producer = defs_.find(ResolveAliasRoot(source));
              if (producer != defs_.end() && IsInside(producer->second.definition_stmt, info.bodies[arm])) {
                const auto* statement = producer->second.definition_stmt.get();
                future_writes.push_back(
                    {destination, stmt_order_.at(statement), reverse_stmt_order_.at(statement)});
              }
            }
          }
        }
      }
    }
    for (const auto& [value, definition] : defs_) {
      if (definition.kind != VarDef::kIterArg) continue;
      const auto loop_for = As<ForStmt>(definition.control_stmt);
      const auto loop_while = As<WhileStmt>(definition.control_stmt);
      const auto body = loop_for ? loop_for->body_ : loop_while ? loop_while->body_ : nullptr;
      const auto yield = body ? transform_utils::GetLastYieldStmt(body) : nullptr;
      if (!yield || definition.return_idx >= yield->value_.size()) continue;
      const auto source = AsVarLike(yield->value_[definition.return_idx]);
      const auto target = CurrentMemRef(value);
      if (target && (!source || !SamePhysicalWindow(CurrentMemRef(source), target))) {
        future_writes.push_back({target, stmt_order_.at(yield.get()), reverse_stmt_order_.at(yield.get())});
      }
    }
    future_writes_.Build(future_writes);
    for (auto& event : future_writes) std::swap(event.primary, event.reverse);
    reverse_future_writes_.Build(future_writes);
    std::vector<StorageAccessIndex::Event> accesses;
    for (const auto& [statement, values] : accesses_) {
      for (const auto& value : values) {
        if (const auto memory = CurrentMemRef(value)) {
          accesses.push_back({memory, stmt_order_.at(statement), reverse_stmt_order_.at(statement)});
        }
      }
    }
    future_accesses_.Build(accesses);
    access_index_ready_ = true;
  }

  [[nodiscard]] bool CanReplayTarget(const VarPtr& value, const MemRefPtr& target) const {
    if (!replaying_ || SamePhysicalWindow(GetTypeMemRef(value->GetType()).value_or(nullptr), target)) {
      return true;
    }
    const auto accepted = replay_targets_.find(value.get());
    return accepted != replay_targets_.end() && SamePhysicalWindow(accepted->second, target);
  }

  struct RewriteJournalEntry {
    VarPtr var;
    std::optional<TypePtr> previous;
    std::optional<VarPtr> previous_required_owner;
    const Var* added_mutable_base = nullptr;
  };

  // A loop requirement follows its IterArg; an accumulator requirement follows
  // its native reused-input value. The owner names the complete actual window,
  // including a view's relative byte origin and declared slot. Explicit result
  // bindings use only the captured window and never follow another owner.
  struct RequiredBranchTarget {
    VarPtr owner;
    MemRefPtr captured_window;
  };

  class RequiredTargetScope {
   public:
    RequiredTargetScope(TopDownRetargeter& planner, VarPtr owner)
        : planner_(planner), previous_(std::move(planner.required_target_owner_)) {
      planner_.required_target_owner_ = std::move(owner);
    }
    ~RequiredTargetScope() { planner_.required_target_owner_ = std::move(previous_); }
    RequiredTargetScope(const RequiredTargetScope&) = delete;
    RequiredTargetScope& operator=(const RequiredTargetScope&) = delete;

   private:
    TopDownRetargeter& planner_;
    VarPtr previous_;
  };

  class RewriteTransaction {
   public:
    explicit RewriteTransaction(TopDownRetargeter& owner)
        : owner_(owner), checkpoint_(owner.rewrite_journal_.size()) {}
    RewriteTransaction(const RewriteTransaction&) = delete;
    RewriteTransaction& operator=(const RewriteTransaction&) = delete;
    ~RewriteTransaction() {
      if (!committed_) owner_.RollbackRewrites(checkpoint_);
    }

    void Commit() { committed_ = true; }

   private:
    TopDownRetargeter& owner_;
    size_t checkpoint_;
    bool committed_ = false;
  };

  struct IncomingValues {
    std::array<const Var*, 2> values;
    TypePtr descriptor;  // Result requirements, without its provisional address.
  };
  struct IncomingHash {
    size_t operator()(const IncomingValues& key) const {
      size_t hash = structural_hash(key.descriptor);
      for (const auto* value : key.values) {
        hash ^= std::hash<const Var*>{}(value) + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };
  struct IncomingEqual {
    bool operator()(const IncomingValues& lhs, const IncomingValues& rhs) const {
      return lhs.values == rhs.values && structural_equal(lhs.descriptor, rhs.descriptor);
    }
  };

  std::map<VarPtr, VarDef> defs_;
  std::map<const IfStmt*, IfInfo> if_results_;
  std::map<const Var*, RequiredBranchTarget> required_branch_targets_;
  std::map<const Var*, VarPtr> required_target_owners_;
  VarPtr required_target_owner_;
  std::map<const Var*, MemRefPtr> replay_targets_;
  std::set<const Var*> blocked_branch_results_;
  bool replaying_ = false;
  StorageAccessIndex future_writes_;
  StorageAccessIndex reverse_future_writes_;
  StorageAccessIndex future_accesses_;
  bool access_index_ready_ = false;
  std::map<const Stmt*, std::set<VarPtr>> accesses_;
  std::map<const Stmt*, std::set<VarPtr>> data_writes_;
  std::map<VarPtr, VarPtr> identity_roots_;
  std::map<VarPtr, VarPtr> value_origins_;
  std::map<const Var*, size_t> result_last_use_;
  std::map<const Var*, size_t> result_reverse_last_use_;
  std::set<const Var*> last_use_is_writeback_;
  std::set<const Var*> reverse_last_use_is_writeback_;
  std::set<const Var*> mutable_branch_bases_;
  // Logical result mutability is separate from a physical allocation's flags:
  // moving another value onto its bytes must not change this SSA obligation.
  std::set<const Var*> mutable_branch_results_;
  std::map<const Stmt*, size_t> stmt_order_;
  std::map<const Stmt*, size_t> stmt_end_order_;
  std::map<const Stmt*, size_t> reverse_stmt_order_;
  std::map<const Stmt*, size_t> reverse_stmt_end_order_;
  StmtPtr analysis_root_;
  std::map<VarPtr, VarPtr> alias_source_;
  std::map<VarPtr, VarPtr> alias_root_cache_;
  std::map<VarPtr, VarPtr> inplace_source_;
  std::map<VarPtr, VarPtr> inplace_root_cache_;
  std::map<VarPtr, VarPtr> accumulator_input_cache_;
  std::set<VarPtr> non_accumulator_producers_;
  std::map<const MemRef*, size_t> memref_component_;
  std::map<size_t, ReachableEventIndex> physical_access_index_;
  StorageAccessIndex physical_read_index_;
  StorageAccessIndex reverse_physical_read_index_;
  std::map<const Var*, VersionedReachableEventIndex> alias_read_index_;
  std::map<VarPtr, TypePtr> rewrites_;
  std::vector<RewriteJournalEntry> rewrite_journal_;
  std::set<VarPtr> visiting_;  // cycle guard
  std::map<const Var*, std::vector<VarPtr>> storage_handles_;
  std::map<const Var*, std::set<VarPtr>> phi_storage_handles_;
  std::map<const Var*, const Var*> phi_handle_bases_;
  std::map<const Var*, const Var*> result_request_owners_;
  std::set<const Var*> ambiguous_result_requests_;
  std::map<const Var*, std::pair<IfStmtPtr, MemRefPtr>> canonical_branch_destinations_;
  std::map<const Var*, MemRefPtr> branch_target_redirects_;
  std::set<const Var*> nonuniform_storage_bases_;
  std::map<const Var*, const Var*> branch_storage_handle_bases_;
  std::map<const Var*, std::pair<size_t, size_t>> storage_definition_bounds_;
  std::set<const Var*> changed_storage_bases_;
  std::set<const Var*> changed_observer_bases_;
  // Relocated definitions supplement the immutable physical indexes with a
  // conservative read envelope in both orders. Unmoved handles use the index.
  StorageObserverIndex planned_storage_observers_;
  std::set<const Var*> pinned_branch_bases_;
  StorageAccessIndex readonly_caller_storage_;
  std::set<const Var*> fixed_branch_bases_;
  std::set<const Var*> owned_branch_bases_;
  std::set<const Var*> immovable_branch_bases_;
  std::set<const Var*> no_mutable_alias_branch_bases_;
  std::vector<size_t> branch_loop_headers_;
  std::unique_ptr<BufferViewAnalysis> input_views_;
  uint64_t branch_storage_counter_ = 0;

  void BuildAnalysis(const StmtPtr& func_body, bool index_branch_storage = false) {
    required_branch_targets_.clear();
    required_target_owners_.clear();
    required_target_owner_.reset();
    access_index_ready_ = false;
    identity_roots_.clear();
    value_origins_.clear();
    result_last_use_.clear();
    result_reverse_last_use_.clear();
    last_use_is_writeback_.clear();
    reverse_last_use_is_writeback_.clear();
    mutable_branch_bases_.clear();
    mutable_branch_results_.clear();
    storage_handles_.clear();
    phi_storage_handles_.clear();
    phi_handle_bases_.clear();
    result_request_owners_.clear();
    ambiguous_result_requests_.clear();
    canonical_branch_destinations_.clear();
    branch_target_redirects_.clear();
    nonuniform_storage_bases_.clear();
    branch_storage_handle_bases_.clear();
    storage_definition_bounds_.clear();
    changed_storage_bases_.clear();
    changed_observer_bases_.clear();
    planned_storage_observers_.Clear();
    branch_loop_headers_.clear();
    DefMapVisitor def_v;
    def_v.Run(func_body);
    defs_ = std::move(def_v.defs);
    if_results_ = std::move(def_v.if_results);
    stmt_order_ = std::move(def_v.stmt_order);
    stmt_end_order_ = std::move(def_v.stmt_end_order);
    analysis_root_ = def_v.root;
    input_views_ = std::make_unique<BufferViewAnalysis>(func_body);

    ReverseBranchOrderVisitor reverse_v;
    reverse_v.Run(func_body);
    reverse_stmt_order_ = std::move(reverse_v.stmt_order);
    reverse_stmt_end_order_ = std::move(reverse_v.stmt_end_order);
    BuildAliasFamilies();
    LoopInputIsolation loop_effects;
    loop_effects.AnalyzeEffects(func_body);
    const auto written_initializers = loop_effects.WrittenLoopInitializers();
    def_v.mutable_inputs.insert(written_initializers.begin(), written_initializers.end());
    for (const auto& value : def_v.mutable_inputs) {
      // A later write also makes a pass-through loop's initializer mutable.
      // Follow mandatory storage aliases here, not logical value versions:
      // LoopReturn/IterArg versions remain distinct for all old-value reads.
      // An actual isolation copy owns separate storage and stops this walk.
      const auto root = ResolveAliasRoot(value);
      const auto definition = defs_.find(root);
      if (definition != defs_.end() && definition->second.kind == VarDef::kIfReturn) {
        if (auto tile = GetTileTypeWithMemRef(root->GetType())) {
          mutable_branch_bases_.insert(GetDefinedMemRef(tile)->base_.get());
          mutable_branch_results_.insert(root.get());
        }
      }
    }
    for (const auto& [value, last] : def_v.value_last_use) {
      const auto root = ValueOrigin(value);
      result_last_use_[root.get()] = std::max(result_last_use_[root.get()], last);
    }
    for (const auto& [statement, values] : def_v.reads) {
      const auto point = reverse_stmt_order_.at(statement);
      for (const auto& value : values) {
        const auto root = ValueOrigin(value);
        result_reverse_last_use_[root.get()] = std::max(result_reverse_last_use_[root.get()], point);
      }
    }
    for (const auto& [value, loops] : def_v.recurring_reads) {
      const auto root = ValueOrigin(value);
      for (const auto* loop : loops) {
        result_reverse_last_use_[root.get()] =
            std::max(result_reverse_last_use_[root.get()], reverse_stmt_end_order_.at(loop));
      }
    }
    for (const auto& use : def_v.writeback_uses) {
      const auto root = ValueOrigin(use.target);
      const bool other_read = std::any_of(use.other_inputs.begin(), use.other_inputs.end(),
                                          [&](const VarPtr& input) { return ValueOrigin(input) == root; });
      if (other_read) continue;
      if (result_last_use_[root.get()] == stmt_order_.at(use.statement)) {
        last_use_is_writeback_.insert(root.get());
      }
      if (result_reverse_last_use_[root.get()] == reverse_stmt_order_.at(use.statement)) {
        reverse_last_use_is_writeback_.insert(root.get());
      }
    }
    data_writes_ = std::move(def_v.data_writes);
    accesses_ = def_v.reads;
    for (const auto& [statement, values] : def_v.writes) {
      accesses_[statement].insert(values.begin(), values.end());
    }
    BuildAccessIndexes(def_v.reads, def_v.writes, def_v.recurring_reads);
    if (!index_branch_storage) return;

    // Index each allocation's handles and definition bounds once. If results
    // consume branch storage; every independently observable alias must remain
    // in the index. Whole-window local families can then be retargeted together
    // without per-result subtree scans.
    std::set<VarPtr> counted;
    auto count_storage = [&](const VarPtr& var) {
      if (!counted.insert(var).second) return;
      auto def = defs_.find(var);
      auto tile = GetTileTypeWithMemRef(var->GetType());
      if (tile) {
        const auto* base = GetDefinedMemRef(tile)->base_.get();
        // Uses through a logical if result observe that result, not its
        // provisional input allocation. Keep these handles for propagation,
        // while excluding them from proof of independent source ownership.
        auto root = defs_.find(ResolveAliasRoot(var));
        if (root != defs_.end() && root->second.kind == VarDef::kIfReturn) {
          phi_storage_handles_[base].insert(var);
          phi_handle_bases_[var.get()] = base;
          const auto [owner, inserted] = result_request_owners_.emplace(base, root->first.get());
          if (!inserted && owner->second != root->first.get()) ambiguous_result_requests_.insert(base);
          return;
        }
        auto& handles = storage_handles_[base];
        if (!handles.empty() &&
            !SamePhysicalWindow(GetTypeMemRef(handles.front()->GetType()).value(), GetDefinedMemRef(tile))) {
          nonuniform_storage_bases_.insert(base);
        }
        handles.push_back(var);
        branch_storage_handle_bases_[var.get()] = base;
        auto [bounds, inserted] =
            storage_definition_bounds_.try_emplace(base, std::numeric_limits<size_t>::max(), size_t{0});
        (void)inserted;
        if (def == defs_.end() || !def->second.definition_stmt) {
          bounds->second = {0, std::numeric_limits<size_t>::max()};
        } else {
          bounds->second.first = std::min(bounds->second.first, def->second.order);
          bounds->second.second = std::max(bounds->second.second, def->second.order);
        }
      }
    };
    for (const auto& [var, def] : defs_) {
      (void)def;
      count_storage(var);
    }
    for (const auto& [stmt, vars] : def_v.reads) {
      (void)stmt;
      for (const auto& var : vars) count_storage(var);
    }
  }

  bool RetargetBranchProducer(const VarPtr& var, const StmtPtr& branch, const MemRefPtr& target,
                              std::optional<MemorySpace> memory, const Stmt* live_target_boundary = nullptr) {
    auto source_tile = CurrentTileType(var);
    auto window = source_tile ? GetDefinedMemRef(source_tile) : nullptr;
    if (window && source_tile->GetMemorySpace() == memory && SamePhysicalWindow(window, target)) return true;
    // Protect both ends before the canonical-result shortcut as well as the
    // ordinary producer path. Redirecting an inner result moves its entire
    // alias family, including hidden workspace writes and pipeline storage.
    if (immovable_branch_bases_.count(target->base_.get()) ||
        (window && immovable_branch_bases_.count(window->base_.get()))) {
      return false;
    }
    if (window && source_tile->GetMemorySpace() == memory) {
      auto owner = canonical_branch_destinations_.find(window->base_.get());
      if (owner != canonical_branch_destinations_.end() && IsInside(owner->second.first, branch) &&
          SamePhysicalWindow(window, owner->second.second) && window->size_ == target->size_ &&
          (!live_target_boundary ||
           !future_accesses_.MayOverlapAfter(target, stmt_order_.at(owner->second.first.get()),
                                             reverse_stmt_order_.at(owner->second.first.get()),
                                             stmt_end_order_.at(live_target_boundary)))) {
        // An inner result already owns an independent destination. Push the
        // outer result's requirement through that destination in one step; a
        // final indexed rewrite updates its producers without rescanning them.
        branch_target_redirects_[window->base_.get()] = target;
        if (mutable_branch_bases_.count(window->base_.get())) {
          mutable_branch_bases_.insert(target->base_.get());
        }
        return true;
      }
    }
    auto root = ResolveAliasRoot(var);
    auto def = defs_.find(root);
    auto tile = CurrentTileType(root);
    if (def == defs_.end() || def->second.kind != VarDef::kAssign || !tile ||
        !SamePhysicalWindow(CurrentMemRef(var), GetDefinedMemRef(tile))) {
      return false;
    }
    const auto* base = GetDefinedMemRef(tile)->base_.get();
    auto bounds = storage_definition_bounds_.find(base);
    if (pinned_branch_bases_.count(base) || immovable_branch_bases_.count(base) ||
        changed_storage_bases_.count(base) || nonuniform_storage_bases_.count(base) ||
        bounds == storage_definition_bounds_.end() || bounds->second.first < stmt_order_.at(branch.get()) ||
        bounds->second.second > stmt_end_order_.at(branch.get())) {
      return false;
    }
    if (live_target_boundary && (immovable_branch_bases_.count(base) ||
                                 !IsTargetDeadAfterDefinition(def->second, target, live_target_boundary))) {
      return false;
    }
    // All handles belong to this arm and denote the complete same window. Move
    // the producer and its required aliases together; no external observer can
    // retain the old allocation. A borrowed target additionally passed the
    // branch-scoped old-value check above.
    RewriteTransaction transaction(*this);
    if (!RetargetAssign(root, def->second, target, memory, /*check_liveness=*/false)) return false;
    std::vector<VarPtr> phi_handles;
    if (auto phis = phi_storage_handles_.find(base); phis != phi_storage_handles_.end()) {
      for (const auto& handle : phis->second) {
        if (SamePhysicalWindow(CurrentMemRef(handle), GetDefinedMemRef(tile)) &&
            SamePhysicalWindow(CurrentMemRef(ResolveAliasRoot(handle)), GetDefinedMemRef(tile))) {
          phi_handles.push_back(handle);
        }
      }
    }
    for (const auto& handle : storage_handles_.at(base)) {
      if (handle != root) PlanRewrite(handle, target, memory);
    }
    // Existing inner results still on this window follow the family. A result
    // already assigned its own destination is an independent value and stays put.
    for (const auto& handle : phi_handles) PlanRewrite(handle, target, memory);
    transaction.Commit();
    return true;
  }

  /// Original and already relocated old values must not observe an overwrite.
  [[nodiscard]] bool HasIndependentOldStorageObserver(const IfStmtPtr& branch,
                                                      const MemRefPtr& target) const {
    const auto end = stmt_end_order_.at(branch.get());
    const auto reverse_end = reverse_stmt_end_order_.at(branch.get());
    if (planned_storage_observers_.MayObserveAfter(target, end, reverse_end)) return true;
    // Reversing definition coordinates makes a strict suffix contain exactly
    // handles older than this If. Each read keeps its actual byte/slot window;
    // an unused whole-allocation handle cannot bridge independent observations.
    const auto before_definition =
        stmt_end_order_.at(analysis_root_.get()) + 1 - stmt_order_.at(branch.get());
    return physical_read_index_.MayOverlapAfter(target, before_definition, end) &&
           reverse_physical_read_index_.MayOverlapAfter(target, before_definition, reverse_end);
  }

  void RecordPlannedStorageObserver(const VarPtr& value) {
    const auto memory = CurrentMemRef(value);
    const auto original = GetTypeMemRef(value->GetType()).value_or(nullptr);
    const auto root = ValueOrigin(value).get();
    const auto last = result_last_use_.find(root);
    const auto reverse_last = result_reverse_last_use_.find(root);
    if (memory && !SamePhysicalWindow(memory, original) && last != result_last_use_.end()) {
      planned_storage_observers_.Observe(memory, last->second,
                                         reverse_last == result_reverse_last_use_.end()
                                             ? reverse_stmt_end_order_.at(analysis_root_.get())
                                             : reverse_last->second);
    }
  }

  /// Prefer an incoming live-in window when no older value can observe its
  /// overwrite. The other arm can produce directly into it or use a legal
  /// boundary copy. Failed proofs retain the independent result request.
  bool TryBorrowBranchStorage(const IfInfo& info, const IfResultInfo& slot, size_t arm,
                              const TileTypePtr& result_tile, const std::set<const Var*>& reserved,
                              const StorageAccessIndex& peer_writes, MemRefPtr* destination) {
    const auto& branch = info.owner;
    const auto& incoming = slot.incoming[arm];
    const auto& other = slot.incoming[1 - arm];
    const auto& other_body = info.bodies[1 - arm];
    auto incoming_tile = CurrentTileType(incoming);
    auto other_tile = CurrentTileType(other);
    auto root = ResolveAliasRoot(incoming);
    auto definition = defs_.find(root);
    auto root_tile = CurrentTileType(root);
    if (!incoming_tile || !other_tile || !root_tile || definition == defs_.end() ||
        definition->second.order >= stmt_order_.at(branch.get())) {
      return false;
    }
    const auto target_memory = incoming_tile->GetMemorySpace();
    const auto source_memory = other_tile->GetMemorySpace();
    if (!target_memory) return false;
    const auto original = GetTypeMemRef(slot.result->GetType()).value_or(GetDefinedMemRef(result_tile));
    if (no_mutable_alias_branch_bases_.count(original->base_.get())) return false;
    const auto target = GetDefinedMemRef(incoming_tile);
    const auto* base = target->base_.get();
    const auto handles = storage_handles_.find(base);
    if (handles == storage_handles_.end() || handles->second.empty()) return false;
    const auto owner_tile = CurrentTileType(ResolveAliasRoot(handles->second.front()));
    if (!owner_tile || !SamePhysicalWindow(target, GetDefinedMemRef(owner_tile)) ||
        !SameStorageDescriptor(incoming_tile, owner_tile) || input_views_->IsWindow(incoming)) {
      return false;
    }
    // An allocation produced before an enclosing loop can be observed again
    // on its next iteration, including through a phi created inside the loop.
    const auto bounds = storage_definition_bounds_.find(base);
    if (!branch_loop_headers_.empty() &&
        (bounds == storage_definition_bounds_.end() || bounds->second.first <= branch_loop_headers_.back())) {
      return false;
    }
    if (!SameKnownTileMemorySpace(incoming_tile, result_tile) ||
        !SamePhysicalWindow(target, GetDefinedMemRef(root_tile)) ||
        target->size_ < std::max(GetDefinedMemRef(result_tile)->size_, GetDefinedMemRef(other_tile)->size_) ||
        reserved.count(base) || pinned_branch_bases_.count(base) || changed_storage_bases_.count(base) ||
        changed_observer_bases_.count(base) || nonuniform_storage_bases_.count(base) ||
        immovable_branch_bases_.count(base) || mutable_branch_bases_.count(base) ||
        MayClobberResult(branch, slot.result, target) || peer_writes.MayOverlapAfter(target, 0, 0) ||
        !SameStorageDescriptor(other_tile, result_tile)) {
      return false;
    }
    if (HasIndependentOldStorageObserver(branch, target)) return false;
    if (!RetargetBranchProducer(other, other_body, target, target_memory, branch.get()) &&
        (!source_memory || !IsTileMoveEverSupported(*source_memory, *target_memory))) {
      return false;
    }
    *destination = target;
    return true;
  }

  /// Keep required aliases current between successive phi decisions. Waiting
  /// until RetypeApplier would let a later phi choose an abandoned view window.
  void FollowBranchStorageAlias(const VarPtr& result, const VarPtr& source, const CallPtr& call = nullptr) {
    auto output = CurrentTileType(result);
    auto input = source ? CurrentTileType(source) : nullptr;
    if (!output || !input) return;
    auto target = GetDefinedMemRef(input);
    if (call && !op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) {
      target = MakeAliasingMemRef(target, ComputeViewByteOffset(call, input), GetDefinedMemRef(output)->size_,
                                  result->span_);
    }
    if (!SamePhysicalWindow(GetDefinedMemRef(output), target) ||
        output->GetMemorySpace() != input->GetMemorySpace()) {
      PlanRewrite(result, target, input->GetMemorySpace());
    }
  }

  template <typename LoopPtr>
  void VisitBranchStorageLoop(const LoopPtr& loop, std::vector<StmtPtr>* allocations, bool select_results) {
    for (const auto& argument : loop->iter_args_) {
      FollowBranchStorageAlias(argument, AsVarLike(argument->initValue_));
      // An IterArg is a distinct value version. Its initializer's last read
      // ends at the header and cannot stand in for reads inside the body.
      if (select_results) RecordPlannedStorageObserver(argument);
    }
    branch_loop_headers_.push_back(stmt_order_.at(loop.get()));
    VisitBranchStorage(loop->body_, allocations, select_results);
    branch_loop_headers_.pop_back();
    for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
      // A zero-trip loop returns its initializer, not its body's yield storage.
      FollowBranchStorageAlias(loop->return_vars_[i], loop->iter_args_[i]);
      if (select_results) RecordPlannedStorageObserver(loop->return_vars_[i]);
    }
  }

  void VisitBranchStorage(const StmtPtr& stmt, std::vector<StmtPtr>* allocations,
                          bool select_results = true) {
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) VisitBranchStorage(child, allocations, select_results);
    } else if (auto assign = As<AssignStmt>(stmt)) {
      auto source = alias_source_.find(assign->var_);
      if (source != alias_source_.end()) {
        FollowBranchStorageAlias(assign->var_, source->second, As<Call>(assign->value_));
      }
      if (select_results) RecordPlannedStorageObserver(assign->var_);
    } else if (auto loop = As<ForStmt>(stmt)) {
      VisitBranchStorageLoop(loop, allocations, select_results);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      VisitBranchStorageLoop(loop, allocations, select_results);
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitBranchStorage(scope->body_, allocations, select_results);
    } else if (auto branch = As<IfStmt>(stmt)) {
      VisitBranchStorage(branch->then_body_, allocations, select_results);
      const auto else_body = branch->else_body_;
      if (else_body) {
        VisitBranchStorage(*else_body, allocations, select_results);
      }
      if (select_results) PlanBranchResults(if_results_.at(branch.get()), allocations);
    }
  }

  /// Resolve only pure SSA renames for optional duplicate-result sharing.
  /// IterArgs, views and writeback results remain distinct logical values.
  VarPtr IdentityValue(const VarPtr& value) { return ResolveValueOrigin(value, false); }
  VarPtr ValueOrigin(const VarPtr& value) { return ResolveValueOrigin(value, true); }

  VarPtr ResolveValueOrigin(const VarPtr& value, bool include_views) {
    auto& cache = include_views ? value_origins_ : identity_roots_;
    if (const auto found = cache.find(value); found != cache.end()) {
      return found->second;
    }
    std::vector<VarPtr> path;
    std::set<const Var*> visited;
    auto current = value;
    while (true) {
      INTERNAL_CHECK_SPAN(visited.insert(current.get()).second, value->span_)
          << "Internal error: cycle in SSA value aliases";
      if (const auto found = cache.find(current); found != cache.end()) {
        current = found->second;
        break;
      }
      const auto definition = defs_.find(current);
      if (definition == defs_.end() || definition->second.kind != VarDef::kAssign) break;
      const auto assignment = As<AssignStmt>(definition->second.assign_stmt);
      auto source = assignment ? AsVarLike(assignment->value_) : nullptr;
      if (!source && assignment && include_views) {
        const auto call = As<Call>(assignment->value_);
        if (call && call->op_ && !call->args_.empty() &&
            op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !WritesTileResult(call)) {
          source = AsVarLike(call->args_[0]);
        }
      }
      if (!source) break;
      path.push_back(current);
      current = source;
    }
    for (const auto& member : path) cache[member] = current;
    cache[value] = current;
    return current;
  }

  [[nodiscard]] bool MayClobberResult(const IfStmtPtr& branch, const VarPtr& result,
                                      const MemRefPtr& target) const {
    const auto last = result_last_use_.find(result.get());
    const auto reverse_last = result_reverse_last_use_.find(result.get());
    const auto end = stmt_end_order_.at(branch.get());
    const auto reverse_end = reverse_stmt_end_order_.at(branch.get());
    auto last_read = last == result_last_use_.end() ? end : std::max(end, last->second);
    auto reverse_last_read = reverse_last == result_reverse_last_use_.end()
                                 ? reverse_end
                                 : std::max(reverse_end, reverse_last->second);
    // An actual writeback target is consumed before its final in-place write.
    // Each lexical order has its own final read; an opposite-arm pass-through
    // yield can be last in one order while the consuming update is last in the
    // other. Other operands/workspaces retain the inclusive boundary.
    if (last_read > end && last_use_is_writeback_.count(result.get())) --last_read;
    if (reverse_last_read > reverse_end && reverse_last_use_is_writeback_.count(result.get())) {
      --reverse_last_read;
    }
    // A reachable clobber must precede some read in both sibling orderings.
    // Either negative query proves safety; independent conservative envelopes
    // may still decline reuse, but cannot hide an actual clobber.
    return future_writes_.MayOverlapAfter(target, end, reverse_end, last_read) &&
           reverse_future_writes_.MayOverlapAfter(target, reverse_end, end, reverse_last_read);
  }

  MemRefPtr RequiredResultTarget(const IfResultInfo& slot) {
    if (const auto required = required_branch_targets_.find(slot.result.get());
        required != required_branch_targets_.end()) {
      return ResolveRequiredTarget(required->second);
    }
    const auto original = GetTypeMemRef(slot.result->GetType()).value_or(nullptr);
    return original && (original->is_pinned_ || fixed_branch_bases_.count(original->base_.get())) ? original
                                                                                                  : nullptr;
  }

  MemRefPtr ResolveRequiredTarget(const RequiredBranchTarget& required) {
    if (!required.owner) return required.captured_window;
    auto current = CurrentMemRef(required.owner);
    INTERNAL_CHECK_SPAN(current, required.owner->span_)
        << "Internal error: required storage owner lost its window";
    return current;
  }

  IncomingValues IncomingKey(const IfResultInfo& slot, const TileTypePtr& result,
                             const std::array<TileTypePtr, 2>& inputs) {
    return {{IdentityValue(slot.incoming[0]).get(), IdentityValue(slot.incoming[1]).get()},
            std::make_shared<TupleType>(std::vector<TypePtr>{CloneTypeWithoutMemRef(result),
                                                             CloneTypeWithoutMemRef(inputs[0]),
                                                             CloneTypeWithoutMemRef(inputs[1])})};
  }

  void PlanBranchResults(const IfInfo& info, std::vector<StmtPtr>* allocations) {
    std::set<const Var*> reserved;
    std::map<const Var*, MemRefPtr> preferred;
    std::unordered_map<IncomingValues, MemRefPtr, IncomingHash, IncomingEqual> equivalent;
    std::vector<StorageAccessIndex::Event> peer_events;
    std::vector<StorageAccessIndex::Event> live_peer_windows;
    std::vector<StorageAccessIndex::Event> mutable_peer_windows;
    for (size_t i = 0; i < info.results.size(); ++i) {
      const auto& slot = info.results[i];
      const auto target = RequiredResultTarget(slot);
      const auto tile = CurrentTileType(slot.result);
      if (!tile) continue;
      std::array<TileTypePtr, 2> inputs;
      for (size_t arm = 0; arm < slot.incoming.size(); ++arm) {
        inputs[arm] = slot.incoming[arm] ? CurrentTileType(slot.incoming[arm]) : nullptr;
      }
      const auto last = result_last_use_.find(slot.result.get());
      if (last != result_last_use_.end() && last->second > stmt_end_order_.at(info.owner.get())) {
        auto peer = target;
        if (!peer && inputs[0] && inputs[1] && SameKnownTileMemorySpace(tile, inputs[0]) &&
            SameKnownTileMemorySpace(tile, inputs[1]) &&
            SamePhysicalWindow(GetDefinedMemRef(inputs[0]), GetDefinedMemRef(inputs[1]))) {
          peer = GetDefinedMemRef(inputs[0]);
        }
        if (peer) {
          live_peer_windows.push_back({peer, i + 1, 1});
          if (mutable_branch_results_.count(slot.result.get())) {
            mutable_peer_windows.push_back({peer, i + 1, 1});
          }
        }
      }
      if (!target) continue;
      preferred.emplace(slot.result.get(), target);
      reserved.insert(target->base_.get());
      const bool written = !inputs[0] || !inputs[1] ||
                           !SamePhysicalWindow(GetDefinedMemRef(inputs[0]), target) ||
                           !SamePhysicalWindow(GetDefinedMemRef(inputs[1]), target);
      if (written) peer_events.push_back({target, 1, 1});
      if (inputs[0] && inputs[1] && !mutable_branch_results_.count(slot.result.get()) &&
          !mutable_branch_bases_.count(target->base_.get())) {
        equivalent.emplace(IncomingKey(slot, tile, inputs), target);
      }
    }
    // Yield slots are simultaneous. A free result cannot borrow a peer's
    // incompatible fixed write destination; equal incoming-value pairs may
    // intentionally share its one transfer instead.
    StorageAccessIndex peer_writes;
    peer_writes.Build(peer_events);
    StorageAccessIndex live_peers;
    live_peers.Build(live_peer_windows);
    StorageAccessIndex mutable_peers;
    mutable_peers.Build(mutable_peer_windows);
    // Reservations cover known future slots. These incremental indexes also
    // cover destinations created while earlier slots settle their producers.
    // A result of this If cannot later redirect as an inner owner of its own
    // arm, and changed producer families cannot move again, so publication is
    // monotone. No later slot may undo a live peer's isolation proof.
    StorageObserverIndex settled_live_peers;
    StorageObserverIndex settled_mutable_peers;
    const auto overlaps_peer = [&](size_t index, const MemRefPtr& candidate, bool mutable_result) {
      const auto& reserved_peers = mutable_result ? live_peers : mutable_peers;
      const auto& settled_peers = mutable_result ? settled_live_peers : settled_mutable_peers;
      return reserved_peers.MayOverlapAfter(candidate, 0, 0, index) ||
             reserved_peers.MayOverlapAfter(candidate, index + 1, 0) ||
             settled_peers.MayObserveAfter(candidate, 0, 0);
    };
    auto common_destination = [&](size_t index, const IfResultInfo& slot, const TileTypePtr& result_tile,
                                  const std::array<TileTypePtr, 2>& inputs) -> MemRefPtr {
      if (!inputs[0] || !inputs[1]) return nullptr;
      auto source = GetDefinedMemRef(inputs[0]);
      if (!SameKnownTileMemorySpace(inputs[0], result_tile) ||
          !SameKnownTileMemorySpace(inputs[1], result_tile) ||
          !SamePhysicalWindow(source, GetDefinedMemRef(inputs[1]))) {
        return nullptr;
      }
      const auto original = GetTypeMemRef(slot.result->GetType()).value_or(GetDefinedMemRef(result_tile));
      const bool mutable_result = mutable_branch_results_.count(slot.result.get());
      const bool mutable_common = mutable_result || mutable_branch_bases_.count(source->base_.get());
      // Independent mutable outputs must preserve every live peer. Readonly
      // duplicate values can share a window, including after a prior slot has
      // retargeted both incoming producer families onto one destination.
      // Retaining these exact incoming bytes moves no pipeline allocation.
      // Hardware hazards still prohibit making their original bytes mutable.
      if (overlaps_peer(index, source, mutable_result) ||
          (mutable_common && (readonly_caller_storage_.MayOverlapAfter(source, 0, 0) ||
                              no_mutable_alias_branch_bases_.count(source->base_.get()) ||
                              no_mutable_alias_branch_bases_.count(original->base_.get()) ||
                              HasIndependentOldStorageObserver(info.owner, source))) ||
          peer_writes.MayOverlapAfter(source, 0, 0) || MayClobberResult(info.owner, slot.result, source)) {
        return nullptr;
      }
      return source;
    };
    // Establish every legal common window before considering optional reuse.
    // In (a, a)/(b, a), the second result still needs a on both paths. Borrowing
    // a for the first result before seeing that slot would destroy its value.
    // Reserving all preferred windows makes this priority independent of slot
    // order, while readonly overlapping views can retain their own descriptors.
    for (size_t i = 0; i < info.results.size(); ++i) {
      const auto& slot = info.results[i];
      const auto result_tile = CurrentTileType(slot.result);
      if (!result_tile || preferred.count(slot.result.get())) continue;
      std::array<TileTypePtr, 2> inputs;
      for (size_t arm = 0; arm < slot.incoming.size(); ++arm) {
        inputs[arm] = slot.incoming[arm] ? CurrentTileType(slot.incoming[arm]) : nullptr;
      }
      if (const auto source = common_destination(i, slot, result_tile, inputs)) {
        preferred.emplace(slot.result.get(), source);
        reserved.insert(source->base_.get());
        if (!mutable_branch_results_.count(slot.result.get()) &&
            !mutable_branch_bases_.count(source->base_.get())) {
          equivalent.emplace(IncomingKey(slot, result_tile, inputs), source);
        }
      }
    }
    for (size_t i = 0; i < info.results.size(); ++i) {
      const auto& slot = info.results[i];
      auto result_tile = CurrentTileType(slot.result);
      if (!result_tile) continue;
      std::array<TileTypePtr, 2> inputs;
      for (size_t arm = 0; arm < slot.incoming.size(); ++arm) {
        CHECK_SPAN(info.bodies[arm] && slot.incoming[arm], info.owner->span_)
            << "If storage planning requires both arms to yield every tile result";
        inputs[arm] = CurrentTileType(slot.incoming[arm]);
      }
      // Incomplete external storage is not evidence for borrowing or identity.
      // Keep it for the existing unknown-storage legalization path.
      if (!inputs[0] || !inputs[1]) continue;
      std::array<MemRefPtr, 2> sources{GetDefinedMemRef(inputs[0]), GetDefinedMemRef(inputs[1])};
      const auto memory = result_tile->GetMemorySpace();
      CHECK_SPAN(memory.has_value(), info.owner->span_)
          << "If storage planning requires a known result memory space";
      const auto key = IncomingKey(slot, result_tile, inputs);
      const auto required = RequiredResultTarget(slot);
      const bool fixed = required != nullptr;
      // Equal values are only a sharing opportunity. A later mutable consumer
      // keeps its own destination so sharing cannot undo entry isolation.
      const bool read_only = !mutable_branch_results_.count(slot.result.get());
      MemRefPtr target;
      if (const auto found = preferred.find(slot.result.get()); found != preferred.end()) {
        const auto candidate = required ? required : found->second;
        const bool peer_conflict = overlaps_peer(i, candidate, !read_only);
        if (required) {
          CHECK_SPAN(!peer_conflict, info.owner->span_)
              << "Required if-result storage overlaps an independently live mutable result";
        }
        if (!peer_conflict) target = candidate;
      } else {
        // Earlier slots may have moved their complete alias families. Recheck
        // actual incoming windows with the same proof used for reservation;
        // sharing their new common window does not merge distinct SSA values.
        target = common_destination(i, slot, result_tile, inputs);
      }
      if (!target && read_only) {
        if (const auto found = equivalent.find(key);
            found != equivalent.end() && !mutable_branch_bases_.count(found->second->base_.get()) &&
            !overlaps_peer(i, found->second, false) &&
            !MayClobberResult(info.owner, slot.result, found->second)) {
          target = found->second;
        }
      }
      for (size_t arm = 0; !target && arm < slot.incoming.size(); ++arm) {
        // Borrowing can retarget a producer, so reject peer conflicts before
        // entering its transaction, while this incoming window is unchanged.
        if (overlaps_peer(i, GetDefinedMemRef(inputs[arm]), !read_only)) continue;
        TryBorrowBranchStorage(info, slot, arm, result_tile, reserved, peer_writes, &target);
      }
      if (!target) {
        // InitMemRef creates an independent, correctly sized request per r.
        // Raw IR can still supply an inherited window: use it as a request
        // only when this function owns it and neither source occupies it.
        auto request = GetDefinedMemRef(result_tile);
        if (owned_branch_bases_.count(request->base_.get()) && !reserved.count(request->base_.get()) &&
            !overlaps_peer(i, request, !read_only) &&
            !ambiguous_result_requests_.count(request->base_.get()) &&
            result_request_owners_.count(request->base_.get()) &&
            result_request_owners_.at(request->base_.get()) == slot.result.get() &&
            !storage_handles_.count(request->base_.get()) && !MemRef::SameAllocation(request, sources[0]) &&
            !MemRef::SameAllocation(request, sources[1])) {
          target = request;
        } else {
          auto base = std::make_shared<Var>("mem_branch_result_" + std::to_string(branch_storage_counter_++),
                                            GetPtrType(), slot.result->span_);
          target = std::make_shared<MemRef>(base, int64_t{0}, request->size_, slot.result->span_);
          allocations->push_back(CreateAllocStatement(target, *memory));
        }
        canonical_branch_destinations_[target->base_.get()] = {info.owner, target};
      }
      PlanRewrite(slot.result, target, memory);
      reserved.insert(target->base_.get());
      for (size_t arm = 0; arm < slot.incoming.size(); ++arm) {
        if (SameStorageDescriptor(inputs[arm], result_tile)) {
          RetargetBranchProducer(slot.incoming[arm], info.bodies[arm], target, memory, info.owner.get());
        }
      }
      if (!fixed && read_only && !mutable_branch_bases_.count(target->base_.get())) {
        equivalent.emplace(key, target);
      }
      RecordPlannedStorageObserver(slot.result);
      const auto last = result_last_use_.find(slot.result.get());
      if (last != result_last_use_.end() && last->second > stmt_end_order_.at(info.owner.get())) {
        // Both orders are outside this If for a live result. Unit timestamps
        // express presence only; geometry remains byte/slot precise.
        settled_live_peers.Observe(target, 1, 1);
        if (!read_only) settled_mutable_peers.Observe(target, 1, 1);
      }
    }
  }

  void BuildAccessIndexes(const std::map<const Stmt*, std::set<VarPtr>>& reads,
                          const std::map<const Stmt*, std::set<VarPtr>>& writes,
                          const std::map<VarPtr, std::set<const Stmt*>>& recurring_reads) {
    memref_component_.clear();
    physical_access_index_.clear();
    alias_read_index_.clear();

    std::map<const MemRef*, MemRefPtr> unique_memrefs;
    auto register_var = [&](const VarPtr& var) {
      if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
        auto memref = GetDefinedMemRef(tile);
        unique_memrefs.emplace(memref.get(), memref);
      }
    };
    for (const auto& [var, def] : defs_) {
      (void)def;
      register_var(var);
    }
    for (const auto& [stmt, vars] : reads) {
      (void)stmt;
      for (const auto& var : vars) register_var(var);
    }
    for (const auto& [stmt, vars] : writes) {
      (void)stmt;
      for (const auto& var : vars) register_var(var);
    }

    std::vector<MemRefPtr> memrefs;
    memrefs.reserve(unique_memrefs.size());
    for (const auto& [ptr, memref] : unique_memrefs) {
      (void)ptr;
      memrefs.push_back(memref);
    }
    std::vector<size_t> parent(memrefs.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = i;

    std::map<const Var*, std::vector<size_t>> by_base;
    for (size_t i = 0; i < memrefs.size(); ++i) by_base[memrefs[i]->base_.get()].push_back(i);
    for (const auto& [base, members] : by_base) {
      (void)base;
      struct StaticInterval {
        int64_t begin;
        __int128 end;
        size_t index;
      };
      std::vector<StaticInterval> intervals;
      std::vector<size_t> dynamic;
      for (size_t index : members) {
        if (auto offset = As<ConstInt>(memrefs[index]->byte_offset_)) {
          intervals.push_back(StaticInterval{
              offset->value_, static_cast<__int128>(offset->value_) + memrefs[index]->size_, index});
        } else {
          dynamic.push_back(index);
        }
      }
      std::sort(intervals.begin(), intervals.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.begin, lhs.end) < std::tie(rhs.begin, rhs.end);
      });
      if (!intervals.empty()) {
        size_t representative = intervals.front().index;
        __int128 component_end = intervals.front().end;
        for (size_t i = 1; i < intervals.size(); ++i) {
          if (static_cast<__int128>(intervals[i].begin) < component_end) {
            UnionSets(parent, representative, intervals[i].index);
            component_end = std::max(component_end, intervals[i].end);
          } else {
            representative = intervals[i].index;
            component_end = intervals[i].end;
          }
        }
      }
      // Symbolic offsets may overlap every window in their allocation.
      if (!dynamic.empty() && !members.empty()) {
        for (size_t index : members) UnionSets(parent, dynamic.front(), index);
      }
    }

    std::map<size_t, size_t> dense_component;
    for (size_t i = 0; i < memrefs.size(); ++i) {
      const size_t root = FindSet(parent, i);
      auto [it, inserted] = dense_component.emplace(root, dense_component.size());
      (void)inserted;
      memref_component_[memrefs[i].get()] = it->second;
    }

    std::map<size_t, std::vector<std::pair<size_t, size_t>>> physical_events;
    std::vector<StorageAccessIndex::Event> physical_reads;
    std::vector<StorageAccessIndex::Event> reverse_physical_reads;
    std::map<const Var*, std::vector<VersionedReachableEventIndex::Event>> alias_events;
    auto record = [&](const Stmt* stmt, const VarPtr& var, bool is_read, bool recurrence = false) {
      const auto& primary_orders = recurrence ? stmt_end_order_ : stmt_order_;
      const auto& reverse_orders = recurrence ? reverse_stmt_end_order_ : reverse_stmt_order_;
      auto primary = primary_orders.find(stmt);
      auto reverse = reverse_orders.find(stmt);
      if (primary == primary_orders.end() || reverse == reverse_orders.end()) return;
      if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
        auto component = memref_component_.find(GetDefinedMemRef(tile).get());
        if (component != memref_component_.end()) {
          physical_events[component->second].emplace_back(primary->second, reverse->second);
        }
      }
      if (is_read) {
        auto def = defs_.find(var);
        // Function inputs and other externally defined handles predate every
        // statement in the analysed body.
        const size_t definition = def == defs_.end() ? 0 : def->second.order;
        if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
          const auto before_definition = stmt_end_order_.at(analysis_root_.get()) + 1 - definition;
          const auto window = GetDefinedMemRef(tile);
          physical_reads.push_back({window, before_definition, primary->second});
          reverse_physical_reads.push_back({window, before_definition, reverse->second});
        }
        alias_events[ResolveAliasRoot(var).get()].push_back(
            VersionedReachableEventIndex::Event{primary->second, definition, reverse->second});
      }
    };
    for (const auto& [stmt, vars] : reads) {
      for (const auto& var : vars) record(stmt, var, true);
    }
    for (const auto& [stmt, vars] : writes) {
      for (const auto& var : vars) record(stmt, var, false);
    }
    // A value defined before a repeated loop can be observed again on its
    // backedge. Preserve its handle version while placing that virtual read at
    // the loop end in both orders, including in the physical access index.
    for (const auto& [value, loops] : recurring_reads) {
      for (const auto* loop : loops) record(loop, value, true, true);
    }
    for (auto& [component, events] : physical_events) {
      physical_access_index_[component].Build(events);
    }
    physical_read_index_.Build(physical_reads);
    reverse_physical_read_index_.Build(reverse_physical_reads);
    for (auto& [root, events] : alias_events) alias_read_index_[root].Build(std::move(events));
  }

  [[nodiscard]] bool IsInside(const StmtPtr& inner, const StmtPtr& outer) const {
    if (!inner || !outer) return false;
    auto inner_order = stmt_order_.find(inner.get());
    auto outer_begin = stmt_order_.find(outer.get());
    auto outer_end = stmt_end_order_.find(outer.get());
    return inner_order != stmt_order_.end() && outer_begin != stmt_order_.end() &&
           outer_end != stmt_end_order_.end() && outer_begin->second <= inner_order->second &&
           inner_order->second <= outer_end->second;
  }

  void BuildAliasFamilies() {
    alias_source_.clear();
    alias_root_cache_.clear();
    inplace_source_.clear();
    inplace_root_cache_.clear();
    accumulator_input_cache_.clear();
    non_accumulator_producers_.clear();

    const auto& registry = OpRegistry::GetInstance();
    for (const auto& [var, def] : defs_) {
      if (def.kind == VarDef::kIterArg) {
        if (def.iter_arg) {
          if (auto init = AsVarLike(def.iter_arg->initValue_)) alias_source_[var] = init;
        }
        continue;
      }
      if (def.kind == VarDef::kLoopReturn) {
        if (const auto loop = As<ForStmt>(def.control_stmt);
            loop && def.return_idx < loop->iter_args_.size()) {
          alias_source_[var] = loop->iter_args_[def.return_idx];
        } else if (const auto loop = As<WhileStmt>(def.control_stmt);
                   loop && def.return_idx < loop->iter_args_.size()) {
          alias_source_[var] = loop->iter_args_[def.return_idx];
        }
        continue;
      }
      if (def.kind != VarDef::kAssign) continue;
      auto assign = As<AssignStmt>(def.assign_stmt);
      if (!assign) continue;

      if (auto source = AsVarLike(assign->value_)) {
        // A bare SSA assignment is semantic identity even when lowering has
        // temporarily left stale MemRef metadata on one side.
        alias_source_[var] = source;
        inplace_source_[var] = source;
        continue;
      }

      auto call = As<Call>(assign->value_);
      if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_)) continue;
      if (!op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) continue;
      if (auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) {
        if (auto source = AsVarLike(call->args_[*reuse_idx])) {
          alias_source_[var] = source;
          inplace_source_[var] = source;
        }
        continue;
      }
      if (!call->args_.empty()) {
        if (auto source = AsVarLike(call->args_[0])) alias_source_[var] = source;
      }
    }
  }

  VarPtr ResolveAliasRoot(const VarPtr& var) {
    auto cached = alias_root_cache_.find(var);
    if (cached != alias_root_cache_.end()) return cached->second;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    while (seen.insert(current.get()).second) {
      auto source = alias_source_.find(current);
      if (source == alias_source_.end()) break;
      path.push_back(current);
      current = source->second;
      auto root_cached = alias_root_cache_.find(current);
      if (root_cached != alias_root_cache_.end()) {
        current = root_cached->second;
        break;
      }
    }
    for (const auto& member : path) alias_root_cache_[member] = current;
    alias_root_cache_[var] = current;
    return current;
  }

  VarPtr ResolveInplaceRoot(const VarPtr& var) {
    auto cached = inplace_root_cache_.find(var);
    if (cached != inplace_root_cache_.end()) return cached->second;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    while (seen.insert(current.get()).second) {
      auto source = inplace_source_.find(current);
      if (source == inplace_source_.end()) break;
      path.push_back(current);
      current = source->second;
      auto root_cached = inplace_root_cache_.find(current);
      if (root_cached != inplace_root_cache_.end()) {
        current = root_cached->second;
        break;
      }
    }
    for (const auto& member : path) inplace_root_cache_[member] = current;
    inplace_root_cache_[var] = current;
    return current;
  }

  // Walk IR, calling Propagate for each ForStmt we encounter.
  void VisitLoopConstraints(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) VisitLoopConstraints(s);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      PropagateFromLoop(for_stmt);
      VisitLoopConstraints(for_stmt->body_);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      PropagateFromLoop(loop);
      VisitLoopConstraints(loop->body_);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      VisitLoopConstraints(if_stmt->then_body_);
      if (if_stmt->else_body_.has_value()) VisitLoopConstraints(if_stmt->else_body_.value());
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitLoopConstraints(scope->body_);
    }
  }

  template <typename LoopPtr>
  void PropagateFromLoop(const LoopPtr& for_stmt) {
    if (for_stmt->iter_args_.empty()) return;
    auto yield = FindYieldStmt(for_stmt->body_);
    if (!yield) return;

    for (size_t i = 0; i < for_stmt->iter_args_.size() && i < yield->value_.size(); ++i) {
      auto ia = for_stmt->iter_args_[i];
      // If a prior PropagateFromLoop (an enclosing ForStmt) already retyped
      // this iter_arg, use the planned new TileType so we don't push a stale
      // target down the chain.
      auto ia_var_key = std::static_pointer_cast<const Var>(ia);
      auto rit = rewrites_.find(ia_var_key);
      TypePtr ia_type = (rit != rewrites_.end()) ? rit->second : ia->GetType();
      auto ia_tile = GetTileTypeWithMemRef(ia_type);
      if (!ia_tile) continue;
      auto target_memref = GetDefinedMemRef(ia_tile);
      auto target_memory = ia_tile->GetMemorySpace();

      auto yield_var = AsVarLike(yield->value_[i]);
      if (!yield_var) continue;

      RequiredTargetScope requirement(*this, ia_var_key);
      TryRetargetVar(yield_var, target_memref, target_memory);
    }
  }

  // Walk the IR; coalesce every accumulator if-phi we encounter.
  void VisitIfPhisForAccumulator(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) VisitIfPhisForAccumulator(s);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      VisitIfPhisForAccumulator(for_stmt->body_);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      VisitIfPhisForAccumulator(loop->body_);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      // Coalesce nested accumulator phis before their enclosing K-window phi.
      // Otherwise a later child rewrite can overwrite the outer target and
      // reintroduce a divergent L0C allocation.
      VisitIfPhisForAccumulator(if_stmt->then_body_);
      if (if_stmt->else_body_.has_value()) VisitIfPhisForAccumulator(if_stmt->else_body_.value());
      TryCoalesceAccIfPhi(if_stmt);
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitIfPhisForAccumulator(scope->body_);
    }
  }

  struct AccumulatorTarget {
    MemRefPtr memref;
    std::optional<MemorySpace> memory_space;
    // Canonical reused input whose pre-existing aliases must remain observable.
    // This is populated for both direct in-place accumulator producers and
    // branch-local accumulator loops.
    VarPtr reused_input;
  };

  // Resolve the canonical allocation of an in-place accumulator producer. A
  // bare SSA alias may still carry a stale type after a nested child rewrite,
  // so return the underlying producer's target rather than a boolean that would
  // make the caller accidentally select the alias's stale MemRef.
  std::optional<AccumulatorTarget> FindInplaceAccumulatorProducer(const VarPtr& var) {
    auto in_var = FindAccumulatorInput(var);
    if (!in_var) return std::nullopt;
    auto in_tile = CurrentTileType(in_var);
    if (!in_tile) return std::nullopt;
    // The registered reuse input is authoritative. Lowering may have left the
    // Call result typed on an older allocation; TryRetargetVar below repairs the
    // result onto this canonical input allocation before the phi is rewritten.
    return AccumulatorTarget{GetDefinedMemRef(in_tile), in_tile->GetMemorySpace(), in_var};
  }

  VarPtr FindAccumulatorInput(const VarPtr& var) {
    auto cached = accumulator_input_cache_.find(var);
    if (cached != accumulator_input_cache_.end()) return cached->second;
    if (non_accumulator_producers_.count(var) != 0) return nullptr;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    VarPtr result;
    while (seen.insert(current.get()).second) {
      auto known = accumulator_input_cache_.find(current);
      if (known != accumulator_input_cache_.end()) {
        result = known->second;
        break;
      }
      if (non_accumulator_producers_.count(current) != 0) break;
      path.push_back(current);

      auto def = defs_.find(current);
      if (def == defs_.end() || def->second.kind != VarDef::kAssign) break;
      auto assign = As<AssignStmt>(def->second.assign_stmt);
      if (!assign) break;
      if (auto alias = AsVarLike(assign->value_)) {
        current = alias;
        continue;
      }
      auto call = As<Call>(assign->value_);
      if (!call || !call->op_) break;
      auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
      if (!reuse_idx.has_value()) break;
      result = AsVarLike(call->args_[*reuse_idx]);
      break;
    }
    for (const auto& member : path) {
      if (result) {
        accumulator_input_cache_[member] = result;
      } else {
        non_accumulator_producers_.insert(member);
      }
    }
    return result;
  }

  /// True when ``var`` is an alias/in-place producer chain whose reused input
  /// ultimately reaches ``root``. This proves accumulator semantics without
  /// relying on the MemRefs already being coalesced: MemoryReuse may have just
  /// packed one member of the family onto another physical window, which is the
  /// mismatch this analysis exists to repair.
  bool IsInplaceChainRootedAt(const VarPtr& var, const VarPtr& root) {
    return ResolveInplaceRoot(var).get() == root.get();
  }

  /// Resolve a branch-local accumulator loop seeded by a value defined outside
  /// ``branch_scope``. Both locality halves are required: the loop itself must
  /// be inside the branch and its seed must be outside it.
  std::optional<AccumulatorTarget> FindExternallySeededAccumulatorLoop(const VarPtr& var,
                                                                       const IfStmtPtr& branch_scope) {
    auto it = defs_.find(var);
    if (it == defs_.end() || it->second.kind != VarDef::kLoopReturn) return std::nullopt;
    if (const auto loop = As<ForStmt>(it->second.control_stmt)) {
      return FindSeededAccumulatorLoop(var, it->second, loop, branch_scope);
    }
    if (const auto loop = As<WhileStmt>(it->second.control_stmt)) {
      return FindSeededAccumulatorLoop(var, it->second, loop, branch_scope);
    }
    return std::nullopt;
  }

  template <typename LoopPtr>
  std::optional<AccumulatorTarget> FindSeededAccumulatorLoop(const VarPtr& var, const VarDef& definition,
                                                             const LoopPtr& loop,
                                                             const IfStmtPtr& branch_scope) {
    if (definition.return_idx >= loop->iter_args_.size()) return std::nullopt;
    if (!IsInside(loop, branch_scope)) return std::nullopt;

    auto init = AsVarLike(loop->iter_args_[definition.return_idx]->initValue_);
    auto init_tile = init ? CurrentTileType(init) : nullptr;
    if (!init || !CurrentTileType(var) || !init_tile) return std::nullopt;

    auto init_def = defs_.find(init);
    if (init_def != defs_.end() && IsInside(init_def->second.definition_stmt, branch_scope)) {
      return std::nullopt;
    }

    auto body_yield = FindYieldStmt(loop->body_);
    if (!body_yield || definition.return_idx >= body_yield->value_.size()) return std::nullopt;
    auto yielded = AsVarLike(body_yield->value_[definition.return_idx]);
    auto iter_arg = std::static_pointer_cast<const Var>(loop->iter_args_[definition.return_idx]);
    if (!yielded || !IsInplaceChainRootedAt(yielded, iter_arg)) return std::nullopt;
    return AccumulatorTarget{GetDefinedMemRef(init_tile), init_tile->GetMemorySpace(), init};
  }

  // For an IfStmt whose branches yield an in-place accumulator on one side and a
  // fresh seed on the other (a different L0C buffer), retarget the seed onto the
  // accumulator buffer.  Scoped to Acc — the ISA case with no legal reconciling
  // move.  A declined retarget is a hard error (see TryCoalesceAccIfPhi).
  void TryCoalesceAccIfPhi(const IfStmtPtr& if_stmt) {
    if (!if_stmt->else_body_.has_value() || if_stmt->return_vars_.empty()) return;
    auto then_yield = FindYieldStmt(if_stmt->then_body_);
    auto else_yield = FindYieldStmt(if_stmt->else_body_.value());
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) continue;
      auto then_var = AsVarLike(then_yield->value_[i]);
      auto else_var = AsVarLike(else_yield->value_[i]);
      if (!then_var || !else_var) continue;

      auto then_target = FindInplaceAccumulatorProducer(then_var);
      if (!then_target) then_target = FindExternallySeededAccumulatorLoop(then_var, if_stmt);
      auto else_target = FindInplaceAccumulatorProducer(else_var);
      if (!else_target) else_target = FindExternallySeededAccumulatorLoop(else_var, if_stmt);
      if (then_target.has_value() == else_target.has_value()) continue;  // need exactly one accumulator

      const bool then_acc = then_target.has_value();
      const VarPtr& acc_var = then_acc ? then_var : else_var;
      const VarPtr& seed_var = then_acc ? else_var : then_var;
      const AccumulatorTarget& target = then_acc ? *then_target : *else_target;
      if (blocked_branch_results_.count(if_stmt->return_vars_[i].get()) ||
          !CanReplayTarget(if_stmt->return_vars_[i], target.memref)) {
        continue;
      }

      const auto result_memory = GetTypeMemRef(if_stmt->return_vars_[i]->GetType());
      if (result_memory && *result_memory &&
          ((*result_memory)->is_pinned_ || fixed_branch_bases_.count((*result_memory)->base_.get())) &&
          !SamePhysicalWindow(*result_memory, target.memref)) {
        continue;  // An accumulator requirement cannot override an explicit binding.
      }

      auto seed_tile = CurrentTileType(seed_var);
      if (!seed_tile) continue;
      if (target.memory_space != MemorySpace::Acc) continue;  // Acc-only (no legal move)

      const bool seed_already_shared = SamePhysicalWindow(target.memref, GetDefinedMemRef(seed_tile));

      auto seed_def = defs_.find(seed_var);
      if (!seed_already_shared && seed_def == defs_.end()) continue;

      // The `check_liveness=false` bypass below is only sound when branch
      // exclusivity actually applies, which requires BOTH:
      //  (a) the seed producer is lexically *inside* this IfStmt's branch — a
      //      pre-if value yielded through the branch runs unconditionally and
      //      would clobber the accumulator the sibling in-place branch reads; and
      //  (b) the accumulator buffer is dead *within the branch* after the seed
      //      (exclusivity covers only cross-branch and post-if reads, not a
      //      same-branch tail read between the seed producer and the yield).
      // When either fails, leave the phi untouched here. YieldFixup will fail
      // loudly rather than emit an unsupported Acc->Acc move.
      if (!seed_already_shared) {
        if (!IsInside(seed_def->second.definition_stmt, if_stmt)) continue;
        if (!IsTargetDeadAfterDefinition(seed_def->second, target.memref,
                                         /*stop_at=*/if_stmt.get())) {
          continue;
        }
      }

      // A branch-local loop can reuse a seed defined outside the IfStmt only
      // when that external value is not observed independently after the phi.
      // Otherwise retargeting the sibling seed would clobber the external value
      // on the path where the loop does not execute.
      if (target.reused_input &&
          HasIndependentExternalUseAfterIf(if_stmt, if_stmt->return_vars_[i], target)) {
        continue;
      }

      // Coalescing one phi is atomic. Retargeting the accumulator side may
      // recursively plan several rewrites before the seed side is inspected;
      // a later decline must not leak that partial family into the pass-wide
      // rewrite map.
      RewriteTransaction transaction(*this);
      RequiredTargetScope requirement(*this, target.reused_input);

      // Align any alias/ForReturn wrapper on the accumulator side with the
      // canonical producer target before using it as the phi's allocation.
      INTERNAL_CHECK_SPAN(TryRetargetVar(acc_var, target.memref, target.memory_space), acc_var->span_)
          << "Internal error: cannot align the accumulator branch with its canonical L0C allocation";

      // Now safe: (a)+(b) plus exclusivity cover every read of the target allocation, so we
      // bypass the global liveness (which would false-decline on the legitimate
      // post-if phi consumer). A remaining decline is a genuine "cannot coalesce
      // this Acc phi" — fail loud, since no legal Acc->Acc move exists.
      bool ok = seed_already_shared;
      if (!seed_already_shared) {
        if (seed_def->second.kind == VarDef::kAssign) {
          auto seed_assign = As<AssignStmt>(seed_def->second.assign_stmt);
          if (!seed_assign) continue;
          if (As<Call>(seed_assign->value_)) {
            ok = RetargetAssign(seed_var, seed_def->second, target.memref, target.memory_space,
                                /*check_liveness=*/false);
          } else if (AsVarLike(seed_assign->value_)) {
            ok = TryRetargetVar(seed_var, target.memref, target.memory_space);
          } else {
            continue;
          }
        } else if (seed_def->second.kind == VarDef::kLoopReturn) {
          ok = TryRetargetVar(seed_var, target.memref, target.memory_space);
        } else {
          continue;
        }
      }
      INTERNAL_CHECK_SPAN(ok, seed_var->span_)
          << "Internal error: cannot coalesce L0C accumulator across a peeled if-phi — seed producer '"
          << seed_var->name_hint_
          << "' refused retarget onto the accumulator buffer, which would force an illegal "
             "Acc->Acc tile.move.";

      // The phi itself is part of the semantic accumulator family. Pinning it
      // here is required by the PTOAS planner, which skips legacy MemoryReuse
      // and otherwise declares a separate head handle for the merged value.
      PlanRewrite(if_stmt->return_vars_[i], target.memref, target.memory_space);
      transaction.Commit();
    }
  }

  // Canonical destinations are created at byte offset zero. Redirecting an
  // inner result to an enclosing result preserves every member's own window.
  MemRefPtr ResolveBranchTarget(const MemRefPtr& memory) {
    auto redirect = branch_target_redirects_.find(memory->base_.get());
    if (redirect == branch_target_redirects_.end()) return memory;
    std::vector<const Var*> path;
    MemRefPtr target;
    while (redirect != branch_target_redirects_.end()) {
      path.push_back(redirect->first);
      target = redirect->second;
      redirect = branch_target_redirects_.find(target->base_.get());
    }
    for (const auto* base : path) branch_target_redirects_[base] = target;
    auto offset = As<ConstInt>(memory->byte_offset_);
    if (memory->size_ == target->size_ && offset && offset->value_ == 0) return target;
    return MakeAliasingMemRef(target, memory->byte_offset_, memory->size_, memory->span_);
  }

  TileTypePtr CurrentTileType(const VarPtr& var) {
    auto it = rewrites_.find(var);
    auto type = (it != rewrites_.end()) ? it->second : var->GetType();
    auto tile = GetTileTypeWithMemRef(type);
    if (!tile) return nullptr;
    auto original = GetDefinedMemRef(tile);
    auto destination = ResolveBranchTarget(original);
    return destination == original ? tile : As<TileType>(CloneTypeWithMemRef(type, destination));
  }

  MemRefPtr CurrentMemRef(const VarPtr& var) {
    auto tile = CurrentTileType(var);
    return tile ? GetDefinedMemRef(tile) : nullptr;
  }

  /// Current (possibly-rewritten) MemRef base of `var`.
  const Var* CurrentBase(const VarPtr& var) {
    auto tile = CurrentTileType(var);
    if (!tile) return nullptr;
    return GetDefinedMemRef(tile)->base_.get();
  }

  /// Attempts to rewrite `var`'s MemRef to `target` by walking its producer chain.
  /// Returns true if var already has target MemRef or a rewrite was planned.
  bool TryRetargetVar(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    if (blocked_branch_results_.count(var.get()) || !CanReplayTarget(var, target)) return false;
    if (SamePhysicalWindow(CurrentMemRef(var), target)) return true;  // already aligned
    if (auto source = CurrentMemRef(var);
        source && (source->is_pinned_ || pinned_branch_bases_.count(source->base_.get()))) {
      return false;
    }
    if (!visiting_.insert(var).second) return false;  // cycle
    struct Guard {
      std::set<VarPtr>* s;
      VarPtr v;
      ~Guard() { s->erase(v); }
    } g{&visiting_, var};

    RewriteTransaction transaction(*this);

    auto it = defs_.find(var);
    if (it == defs_.end()) return false;
    const auto& def = it->second;

    bool retargeted = false;
    if (def.kind == VarDef::kAssign) {
      retargeted = RetargetAssign(var, def, target, target_memory);
    } else if (def.kind == VarDef::kIfReturn) {
      retargeted = RetargetIfReturn(var, def, target, target_memory);
    } else if (def.kind == VarDef::kLoopReturn) {
      retargeted = RetargetLoopReturn(var, def, target, target_memory);
    } else if (def.kind == VarDef::kIterArg) {
      retargeted = RetargetIterArg(var, def, target, target_memory);
    }
    if (retargeted) transaction.Commit();
    return retargeted;
  }

  [[nodiscard]] static bool FitsTargetStorage(const TileTypePtr& tile, const MemRefPtr& target,
                                              std::optional<MemorySpace> memory) {
    if (!tile || !target || !memory) return false;
    const auto* context = PassContext::Current();
    const auto* handler = backend::BackendConfig::IsConfigured()
                              ? (context ? context->GetBackendHandler() : backend::GetBackend()->GetHandler())
                              : nullptr;
    const auto required = utils::StaticPhysicalAllocationBytes(tile, *memory, handler);
    return required && *required <= target->size_;
  }

  /// Retype a Var defined by an AssignStmt.
  ///
  /// The normal loop-carry path checks target liveness. Branch storage may
  /// bypass that check only after proving the whole source family is local and
  /// the destination has no observable prior contents on that path. Accumulator
  /// coalescing supplies its separate branch-scoped safety proof. Operation
  /// legality checks always apply.
  bool RetargetAssign(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                      std::optional<MemorySpace> target_memory, bool check_liveness = true) {
    if (!CanReplayTarget(var, target)) return false;
    const auto tile = CurrentTileType(var);
    if (!FitsTargetStorage(tile, target, target_memory)) return false;
    if (auto source = CurrentMemRef(var);
        source && !SamePhysicalWindow(source, target) &&
        (source->is_pinned_ || pinned_branch_bases_.count(source->base_.get()))) {
      return false;
    }
    auto assign = As<AssignStmt>(def.assign_stmt);
    INTERNAL_CHECK_SPAN(assign, var->span_) << "Internal error: kAssign VarDef must carry an AssignStmt";
    auto call = As<Call>(assign->value_);
    if (!call) {
      auto alias = AsVarLike(assign->value_);
      if (!alias) return false;
      auto alias_tile = GetTileTypeWithMemRef(alias->GetType());
      auto var_tile = GetTileTypeWithMemRef(var->GetType());
      if (!alias_tile || !var_tile ||
          !SamePhysicalWindow(GetDefinedMemRef(alias_tile), GetDefinedMemRef(var_tile))) {
        return false;
      }
      if (!TryRetargetVar(alias, target, target_memory)) return false;
      PlanRewrite(var, target, target_memory);
      return true;
    }
    const auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(call->op_->name_)) return false;
    const auto& entry = reg.GetEntry(call->op_->name_);

    auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
    if (reuse_idx.has_value()) {
      // Pinned output: can't change this stmt's LHS MemRef; recurse onto pinned input.
      auto input_var = AsVarLike(call->args_[*reuse_idx]);
      if (!input_var) return false;
      if (!TryRetargetVar(input_var, target, target_memory)) return false;
      // Also record that `var`'s MemRef should follow the pinned input to target.
      PlanRewrite(var, target, target_memory);
      return true;
    }

    // Decline retargeting for ops whose output memory is not fully captured
    // by the LHS type alone:
    //   1. Buffer-aliasing view ops — output MemRef inherits the input's view
    //      byte_offset_ / size_, which rewriting with target's full MemRef
    //      would silently drop.  `tile.transpose` is declined here too, even
    //      though it permutes into a fresh buffer and so has no inherited view
    //      offset to lose: `set_output_memory_inherit_input()` also pins its
    //      output's memory SPACE to its input's, and nothing below enforces
    //      that — case 4 reads a null `deduce_output_memory` for an
    //      inherit-input op and never fires, and case 3 compares MemRef bases,
    //      not spaces.  Retyping its LHS onto a different-space target would
    //      contradict the relation InferTileMemorySpace established.  Making
    //      transpose retargetable needs that space guard first.
    //   2. Ops that encode output memory in a `target_memory` kwarg (e.g.
    //      tile.create / tile.move / tile.load) — retyping the LHS to a
    //      different memory_space would require rewriting the kwarg too.
    //      Same-memory_space retypes are safe: only the MemRef base changes,
    //      while the op's declared output memory stays consistent.
    //   3. Ops registered `not_inplace_safe()` (e.g. tile.mrgsort_format1)
    //      whose implementation requires src buffer != dst buffer.  If any
    //      input of the call already lives on `target_base`, retyping the
    //      output onto the same buffer creates an in-place execution that
    //      the op cannot handle and fails at runtime.
    //   4. Ops with a registered *fixed* output memory space (e.g. every
    //      `tile.add`-style elementwise op is `set_output_memory(Vec)`) when
    //      the target sits in a different space.  The op physically cannot
    //      write there, so retyping the LHS would only mint IR that claims a
    //      Vec-only op produces DDR.  Declining leaves the carry yielding a
    //      different buffer than its init, which YieldFixupMutator reconciles
    //      with a real move (see RetypeApplier's scoped loop-binding contract).
    //      Same-space targets are unaffected: only the MemRef base moves.
    if (entry.OutputMemoryInheritsInput()) {
      // Most view ops can carry a sub-region offset, so retargeting their LHS
      // directly would lose view-relative addressing. tile.set_validshape is
      // the narrow exception: it is a zero-offset, shape-preserving alias of
      // its first argument. Retarget that storage producer first, then record
      // the view result on the same target. This lets padded loop-carried Acc
      // initializers coalesce with their accumulator instead of leaving a
      // divergent Acc carry for YieldFixup to reject.
      if (!IsOp(call, "tile.set_validshape") || call->args_.empty()) return false;
      auto input_var = AsVarLike(call->args_[0]);
      if (!input_var) return false;
      auto input_tile = GetTileTypeWithMemRef(input_var->GetType());
      auto output_tile = GetTileTypeWithMemRef(var->GetType());
      if (!input_tile || !output_tile ||
          !SamePhysicalWindow(GetDefinedMemRef(input_tile), GetDefinedMemRef(output_tile))) {
        return false;
      }
      if (!TryRetargetVar(input_var, target, target_memory)) return false;
      PlanRewrite(var, target, target_memory);
      return true;
    }
    if (HasKwarg(*call, "target_memory") && !TargetMemoryKwargMatches(*call, target_memory)) {
      return false;
    }
    if (FixedOutputMemoryConflicts(entry, *call, target_memory)) return false;
    const auto& forbidden_args = entry.ForbidOutputAliasArgs();
    const std::set<size_t> forbidden(forbidden_args.begin(), forbidden_args.end());
    const bool disjoint_cast = op_predicates::CastRequiresDisjointSource(call, !IsA5Target());
    const size_t read_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
    for (size_t i = 0; i < std::min(read_arg_count, call->args_.size()); ++i) {
      VarUseCollector reads;
      reads.VisitExpr(call->args_[i]);
      for (const auto& input : reads.used_vars) {
        auto input_tile = CurrentTileType(input);
        if (!input_tile) continue;
        auto input_memref = GetDefinedMemRef(input_tile);
        if (input_memref->base_ != target->base_) continue;
        if (!entry.IsInplaceSafe() || forbidden.count(i) || (disjoint_cast && i == 0) ||
            !SamePhysicalWindow(input_memref, target) || input_views_->IsWindow(input)) {
          return false;
        }
      }
    }

    // Unconstrained: check liveness, then plan retype.  (Skipped for if-phi
    // branch coalescing, where branch exclusivity is a stronger guarantee.)
    if (check_liveness && !IsTargetDeadAtAssign(def, target)) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  static bool HasKwarg(const Call& call, const std::string& key) {
    for (const auto& [k, _] : call.kwargs_) {
      if (k == key) return true;
    }
    return false;
  }

  /// True when the call has a `target_memory` kwarg whose MemorySpace value
  /// equals `target_memory`. Used to allow retypes that change the MemRef
  /// base but keep the memory space the same — the kwarg stays consistent
  /// with the new LHS type without rewriting.  Caller must have verified
  /// `HasKwarg(call, "target_memory")` first; AnyCast errors propagate as
  /// pypto exceptions if the kwarg holds an unexpected type (which would be
  /// an internal IR consistency bug rather than a retype-decline signal).
  static bool TargetMemoryKwargMatches(const Call& call, std::optional<MemorySpace> target_memory) {
    if (!target_memory.has_value()) return false;
    for (const auto& [k, v] : call.kwargs_) {
      if (k != "target_memory") continue;
      return AnyCast<MemorySpace>(v, "kwarg key: target_memory") == *target_memory;
    }
    return false;
  }

  /// True when the op resolves to a concrete output memory space that differs
  /// from `target_memory` — i.e. retyping the LHS would claim the op writes
  /// somewhere it physically cannot.  Ops that defer the choice (the resolver
  /// returns nullopt: inherit-input and the retargetable `target_memory`-kwarg
  /// ops) are not constrained here and are handled by their own guards above.
  static bool FixedOutputMemoryConflicts(const OpRegistryEntry& entry, const Call& call,
                                         std::optional<MemorySpace> target_memory) {
    if (!target_memory.has_value()) return false;
    const auto& spec = entry.GetMemorySpec();
    if (!spec.has_value() || !spec->deduce_output_memory) return false;
    // Resolve against the call's own kwargs so a kwarg-driven op reports the
    // space it actually produces rather than its registered default.
    auto fixed = spec->deduce_output_memory(call.kwargs_);
    return fixed.has_value() && *fixed != *target_memory;
  }

  /// Bind the result independently of whether each producer can write directly.
  /// A legal edge copy can meet the same destination after the branch's old
  /// reads complete. Unsupported-copy spaces still require producer retargeting.
  bool RetargetIfReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                        std::optional<MemorySpace> target_memory) {
    auto if_stmt = As<IfStmt>(def.control_stmt);
    if (!if_stmt) return false;
    size_t idx = def.return_idx;
    const auto result_tile = CurrentTileType(var);
    // Producer-local old reads can finish before its yield, allowing a late
    // transfer. Old target reads after the merge still forbid that transfer:
    // the enclosing loop may not update its carry until its own final yield.
    const bool allow_edge_copy = result_tile && result_tile->GetMemorySpace() == target_memory &&
                                 FitsTargetStorage(result_tile, target, target_memory) &&
                                 IsTargetDeadAfterDefinition(def, target);

    auto visit_branch = [&](const StmtPtr& body) -> bool {
      auto y = FindYieldStmt(body);
      if (!y || idx >= y->value_.size()) return false;
      auto yv = AsVarLike(y->value_[idx]);
      if (!yv) return false;
      if (TryRetargetVar(yv, target, target_memory)) return true;
      const auto source_tile = CurrentTileType(yv);
      const auto source_memory = source_tile ? source_tile->GetMemorySpace() : std::nullopt;
      if (!allow_edge_copy || !source_memory || !target_memory) return false;
      return IsTileMoveEverSupported(*source_memory, *target_memory);
    };

    bool then_ok = visit_branch(if_stmt->then_body_);
    bool else_ok = if_stmt->else_body_.has_value() ? visit_branch(if_stmt->else_body_.value()) : true;
    if (!then_ok || !else_ok) return false;

    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// Retype a ForStmt return_var: recurse into the loop's body-yield value.
  /// Pushing the target onto the body-yield-value also pushes it transitively
  /// onto the iter_arg (via the matmul_acc-style pinned chain) and onto the
  /// init.  After all recursive PlanRewrites, the iter_arg's MemRef will
  /// match the target, and YieldFixupMutator's PatchIterArgsAndReturnVars
  /// finalises iter_arg/return_var TileType updates.
  bool RetargetLoopReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                          std::optional<MemorySpace> target_memory) {
    if (const auto loop = As<ForStmt>(def.control_stmt)) {
      return RetargetLoopResult(var, def, loop, target, target_memory);
    }
    if (const auto loop = As<WhileStmt>(def.control_stmt)) {
      return RetargetLoopResult(var, def, loop, target, target_memory);
    }
    return false;
  }

  template <typename LoopPtr>
  bool RetargetLoopResult(const VarPtr& var, const VarDef& def, const LoopPtr& for_stmt,
                          const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    size_t idx = def.return_idx;

    auto y = FindYieldStmt(for_stmt->body_);
    if (!y || idx >= y->value_.size()) return false;
    auto yv = AsVarLike(y->value_[idx]);
    if (!yv) return false;
    if (!TryRetargetVar(yv, target, target_memory)) return false;

    // Also retype the corresponding iter_arg's init so the next iteration's
    // loop-carried value reads from the same buffer the body just wrote.
    if (idx < for_stmt->iter_args_.size()) {
      auto ia = for_stmt->iter_args_[idx];
      auto init_var = AsVarLike(ia->initValue_);
      if (init_var && !TryRetargetVar(init_var, target, target_memory)) return false;
      // Plan rewrite for the IterArg itself (its TileType will be updated).
      PlanRewrite(std::static_pointer_cast<const Var>(ia), target, target_memory);
    }

    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// Retype an IterArg: push the target onto its initValue. The iter_arg
  /// itself records a rewrite so RetypeApplier substitutes its TileType in
  /// body references.
  bool RetargetIterArg(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                       std::optional<MemorySpace> target_memory) {
    auto iter_arg = def.iter_arg;
    if (!iter_arg) return false;
    auto init_var = AsVarLike(iter_arg->initValue_);
    if (!init_var) return false;
    if (!TryRetargetVar(init_var, target, target_memory)) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// Keep optional branch-producer eligibility conservative as aliases move.
  /// Each handle contributes once, even when its original descriptor was stale.
  void UpdateBranchStorageHandle(const VarPtr& var, const TypePtr& type) {
    if (auto phi = phi_handle_bases_.find(var.get()); phi != phi_handle_bases_.end()) {
      const auto* base = GetDefinedMemRef(GetTileTypeWithMemRef(type))->base_.get();
      if (base != phi->second) {
        // Physical read indexes describe the original IR. Moving a logical
        // result can introduce an independent observer on its new allocation;
        // keep later optional borrowing conservative without rescanning users.
        changed_observer_bases_.insert(phi->second);
        changed_observer_bases_.insert(base);
        phi_storage_handles_[phi->second].erase(var);
        phi_storage_handles_[base].insert(var);
        phi->second = base;
      }
      return;
    }
    auto tracked = branch_storage_handle_bases_.find(var.get());
    if (tracked == branch_storage_handle_bases_.end()) return;
    auto tile = GetTileTypeWithMemRef(type);
    if (!tile) return;
    const auto* base = GetDefinedMemRef(tile)->base_.get();
    if (base == tracked->second) return;
    changed_storage_bases_.insert(tracked->second);
    changed_storage_bases_.insert(base);
    tracked->second = base;
  }

  /// IterArg keys are stored under their `Var` base via static_pointer_cast;
  /// `rewrites_` keying compares shared_ptr control blocks, not static types,
  /// so `RetypeApplier`'s ForStmt handler can find the IterArg's planned
  /// retype using the same VarPtr-keyed lookup as ordinary AssignStmt LHS
  /// rewrites. Every recursive retarget attempt is transactional: if any arm
  /// declines, RollbackRewrites restores all entries written since that
  /// attempt's checkpoint, including successful nested attempts.
  void PlanRewrite(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    const auto previous = rewrites_.find(var);
    const auto previous_owner = required_target_owners_.find(var.get());
    RewriteJournalEntry entry{
        var, previous == rewrites_.end() ? std::nullopt : std::optional<TypePtr>(previous->second),
        previous_owner == required_target_owners_.end() ? std::nullopt
                                                        : std::optional<VarPtr>(previous_owner->second)};
    const auto definition = defs_.find(var);
    if (required_target_owner_ && definition != defs_.end() && definition->second.kind == VarDef::kIfReturn) {
      // Alias-root equality would make a result depend on a carry initialized
      // from itself. Keep that captured constraint instead of creating a cycle.
      if (ResolveAliasRoot(required_target_owner_) != var &&
          SamePhysicalWindow(CurrentMemRef(required_target_owner_), target)) {
        required_target_owners_[var.get()] = required_target_owner_;
      } else {
        required_target_owners_.erase(var.get());
      }
    }
    if (const auto tile = CurrentTileType(var);
        tile && mutable_branch_bases_.count(GetDefinedMemRef(tile)->base_.get()) &&
        mutable_branch_bases_.insert(target->base_.get()).second) {
      entry.added_mutable_base = target->base_.get();
    }
    auto new_type = CloneTypeWithMemRef(var->GetType(), target, target_memory);
    UpdateBranchStorageHandle(var, new_type);
    rewrite_journal_.push_back(std::move(entry));
    rewrites_[var] = std::move(new_type);
  }

  void RollbackRewrites(size_t checkpoint) {
    INTERNAL_CHECK(checkpoint <= rewrite_journal_.size()) << "Invalid rewrite transaction checkpoint";
    while (rewrite_journal_.size() > checkpoint) {
      auto entry = std::move(rewrite_journal_.back());
      rewrite_journal_.pop_back();
      UpdateBranchStorageHandle(entry.var, entry.previous.value_or(entry.var->GetType()));
      if (entry.previous_required_owner) {
        required_target_owners_[entry.var.get()] = *entry.previous_required_owner;
      } else {
        required_target_owners_.erase(entry.var.get());
      }
      if (entry.added_mutable_base) mutable_branch_bases_.erase(entry.added_mutable_base);
      if (entry.previous.has_value()) {
        rewrites_[entry.var] = *entry.previous;
      } else {
        rewrites_.erase(entry.var);
      }
    }
  }

  /// True when a branch-local accumulator loop's external seed is read after
  /// the enclosing IfStmt through a path other than the IfStmt return value.
  /// Such a read observes the pre-loop value on the sibling path, so placing
  /// that sibling's producer in the seed's window would be a clobber.
  bool HasIndependentExternalUseAfterIf(const IfStmtPtr& if_stmt, const VarPtr& if_return,
                                        const AccumulatorTarget& target) {
    if (!target.reused_input) return false;
    auto return_def = defs_.find(if_return);
    if (return_def == defs_.end() || return_def->second.kind != VarDef::kIfReturn) return true;
    auto primary_end = stmt_end_order_.find(if_stmt.get());
    auto root_end = stmt_end_order_.find(analysis_root_.get());
    if (primary_end == stmt_end_order_.end() || root_end == stmt_end_order_.end()) {
      return true;
    }
    auto reads = alias_read_index_.find(ResolveAliasRoot(target.reused_input).get());
    if (reads == alias_read_index_.end()) return false;
    auto if_begin = stmt_order_.find(if_stmt.get());
    if (if_begin == stmt_order_.end()) return true;
    return reads->second.HasAfterFromHandleDefinedBefore(primary_end->second + 1, root_end->second,
                                                         if_begin->second,
                                                         reverse_stmt_end_order_.at(if_stmt.get()));
  }

  /// Is target's physical window untouched after the definition and before the
  /// nearest loop (or caller-supplied branch boundary) completes? The dual
  /// lexical ordering excludes mutually exclusive sibling arms without walking
  /// ancestor chains or rescanning subtrees.
  bool IsTargetDeadAfterDefinition(const VarDef& def, const MemRefPtr& target,
                                   const Stmt* stop_at = nullptr) {
    if (!target || !def.definition_stmt) return false;
    if (access_index_ready_) {
      const Stmt* boundary = stop_at            ? stop_at
                             : def.nearest_loop ? def.nearest_loop.get()
                                                : analysis_root_.get();
      const bool region_result = def.kind == VarDef::kIfReturn || def.kind == VarDef::kLoopReturn;
      const auto primary = region_result ? stmt_end_order_.at(def.definition_stmt.get()) : def.order;
      const auto reverse = region_result ? reverse_stmt_end_order_.at(def.definition_stmt.get())
                                         : reverse_stmt_order_.at(def.definition_stmt.get());
      return !future_accesses_.MayOverlapAfter(target, primary, reverse, stmt_end_order_.at(boundary));
    }
    auto component = memref_component_.find(target.get());
    if (component == memref_component_.end()) return false;
    auto accesses = physical_access_index_.find(component->second);
    if (accesses == physical_access_index_.end()) return true;

    const Stmt* boundary = stop_at;
    if (!boundary && def.nearest_loop) boundary = def.nearest_loop.get();
    if (!boundary && analysis_root_) boundary = analysis_root_.get();
    auto boundary_end = stmt_end_order_.find(boundary);
    auto reverse_definition = reverse_stmt_order_.find(def.definition_stmt.get());
    if (boundary_end == stmt_end_order_.end() || reverse_definition == reverse_stmt_order_.end()) {
      return false;
    }
    if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kLoopReturn) {
      auto reverse_end = reverse_stmt_end_order_.find(def.definition_stmt.get());
      if (reverse_end == reverse_stmt_end_order_.end()) return false;
      return !accesses->second.HasAfter(def.order + 1, boundary_end->second, reverse_end->second);
    }
    return !accesses->second.HasAfter(def.order + 1, boundary_end->second, reverse_definition->second);
  }

  bool IsTargetDeadAtAssign(const VarDef& def, const MemRefPtr& target, const Stmt* stop_at = nullptr) {
    INTERNAL_CHECK(def.kind == VarDef::kAssign)
        << "Internal error: IsTargetDeadAtAssign requires an assignment definition";
    return IsTargetDeadAfterDefinition(def, target, stop_at);
  }
};

/// Applies planned retypes to the IR.
class RetypeApplier : public IRMutator {
 public:
  explicit RetypeApplier(std::map<VarPtr, TypePtr> rewrites,
                         std::map<const IterArg*, VarPtr> initializers = {},
                         std::map<const Stmt*, std::vector<StmtPtr>> entries = {})
      : rewrites_(std::move(rewrites)),
        initializers_(std::move(initializers)),
        entries_(std::move(entries)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_substitution_.find(op);
    if (it != var_substitution_.end()) return it->second;
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Check var_remap_ first.  IRMutator::VisitStmt_(ForStmtPtr) registers
    // OLD → NEW for each iter_arg whose pointer changed when we visited the
    // For's iter_args_ list, then visits the body.  Without this lookup, every
    // body reference to OLD would build its own fresh IterArg (each
    // make_shared returns a distinct pointer), leaving the For's iter_args_
    // and body referencing two different IterArg objects with the same name —
    // a malformed-IR pattern that downstream substitution passes (e.g.,
    // Simplify's Fold B) cannot rewrite consistently.
    auto remap_it = var_remap_.find(op.get());
    if (remap_it != var_remap_.end()) {
      return remap_it->second;
    }

    INTERNAL_CHECK_SPAN(op->initValue_, op->span_) << "IterArg has null initValue";
    // Always recurse into initValue_ first so prior substitutions applied to
    // the AssignStmt that defines initValue (e.g., a tile.create whose LHS
    // we retyped) propagate into the IterArg's initValue field.
    auto new_init_value = VisitExpr(op->initValue_);
    INTERNAL_CHECK_SPAN(new_init_value, op->span_) << "IterArg initValue mutated to null";

    // Look up whether the IterArg itself is being retyped (its TileType
    // changed by a planned PlanRewrite). If so, return a new IterArg with the
    // retyped TileType and the (possibly substituted) initValue.
    auto var_key = std::static_pointer_cast<const Var>(op);
    auto it = var_substitution_.find(var_key);
    if (it != var_substitution_.end()) {
      auto sub_iter_arg = std::dynamic_pointer_cast<const IterArg>(it->second);
      if (sub_iter_arg) {
        return std::make_shared<IterArg>(sub_iter_arg->name_hint_, sub_iter_arg->GetType(), new_init_value,
                                         sub_iter_arg->span_);
      }
      return it->second;
    }
    if (new_init_value.get() != op->initValue_.get()) {
      return std::make_shared<IterArg>(op->name_hint_, op->GetType(), new_init_value, op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rit = rewrites_.find(op->var_);
    if (rit != rewrites_.end()) {
      auto new_var = std::make_shared<Var>(op->var_->name_hint_, rit->second, op->var_->span_);
      var_substitution_[op->var_] = new_var;
      return IRMutator::VisitStmt_(op);
    }
    if (auto followed = FollowRetargetedViewInput(op)) return followed;
    return IRMutator::VisitStmt_(op);
  }

  /// Re-anchor an inherit-input view (tile.transpose_view / reshape / slice /
  /// set_validshape / tensor.view, ...) onto its input's reused buffer when
  /// the input was retargeted.  The planner never retargets a view's output
  /// directly (RetargetAssign declines OutputMemoryInheritsInput ops), so when its
  /// input (e.g. a tile.load) is retargeted onto a reused buffer, the view's
  /// declared MemRef is left pointing at the now-orphaned original buffer.  Codegen
  /// then emits the view's alloc_tile at the stale address while the load writes the
  /// reused one — the consumer reads an uninitialised buffer (qwen3 b_trans
  /// regression, issue #1776; the sub-region tile.slice case has the same shape).
  /// Re-derive the view's MemRef from its (retargeted) input, mirroring InitMemRef's
  /// ShareMemRefFrom.
  ///
  /// Returns the rewritten AssignStmt, or nullptr when `op` is not an inherit-input
  /// view whose input was retargeted (caller then falls back to the default visit).
  StmtPtr FollowRetargetedViewInput(const AssignStmtPtr& op) {
    auto orig_call = As<Call>(op->value_);
    if (!orig_call || !orig_call->op_ || orig_call->args_.empty()) return nullptr;
    if (!op_predicates::IsBufferAliasingViewOp(orig_call->op_->name_)) return nullptr;
    auto in_var = AsVarLike(orig_call->args_[0]);
    if (!in_var) return nullptr;
    auto sub = var_substitution_.find(in_var);
    if (sub == var_substitution_.end()) return nullptr;  // input not retargeted
    auto out_mr_opt = GetTypeMemRef(op->var_->GetType());
    auto in_new_opt = GetTypeMemRef(sub->second->GetType());
    if (!out_mr_opt || !in_new_opt) return nullptr;
    const auto& out_mr = *out_mr_opt;
    const auto& in_new = *in_new_opt;
    // The registered alias contract also holds when only the source's offset,
    // extent or slot changed, or the output already carried a stale descriptor.
    auto additional = ComputeViewByteOffset(orig_call, sub->second->GetType());
    auto new_mr = MakeAliasingMemRef(in_new, additional, out_mr->size_, out_mr->span_);
    // OutputMemoryInheritsInput also requires the view's memory space to follow its
    // input: if the retarget moved across memory spaces, carry the new space too.
    std::optional<MemorySpace> new_memory;
    if (auto in_new_tile = GetTileTypeWithMemRef(sub->second->GetType())) {
      new_memory = in_new_tile->GetMemorySpace();
    }
    auto output_tile = As<TileType>(op->var_->GetType());
    if (SamePhysicalWindow(out_mr, new_mr) && (!output_tile || output_tile->GetMemorySpace() == new_memory)) {
      return nullptr;
    }
    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), new_mr, new_memory);
    auto follow_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_substitution_[op->var_] = follow_var;
    auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(recursed, op->span_) << "Internal error: AssignStmt visit must yield an AssignStmt";
    return std::make_shared<AssignStmt>(follow_var, recursed->value_, recursed->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Pre-register substitutions for any retargeted return_var; the default
    // IRMutator IfStmt handler will then pick them up through VisitExpr_(Var).
    for (const auto& rv : op->return_vars_) {
      auto rit = rewrites_.find(rv);
      if (rit != rewrites_.end()) {
        var_substitution_[rv] = std::make_shared<Var>(rv->name_hint_, rit->second, rv->span_);
      }
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return RetypeLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return RetypeLoop(op); }

  template <typename LoopPtr>
  StmtPtr RetypeLoop(const LoopPtr& op) {
    std::vector<StmtPtr> prefix;
    if (auto entry = entries_.find(op.get()); entry != entries_.end()) {
      for (const auto& transfer : entry->second) prefix.push_back(VisitStmt(transfer));
    }
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const auto& argument = op->iter_args_[i];
      auto replacement = initializers_.find(argument.get());
      auto initial =
          VisitExpr(replacement == initializers_.end() ? argument->initValue_ : replacement->second);
      auto argument_type = argument->GetType();
      if (auto found = rewrites_.find(argument); found != rewrites_.end()) {
        argument_type = found->second;
      }
      auto initial_shaped = As<ShapedType>(initial->GetType());
      auto argument_shaped = As<ShapedType>(argument_type);
      if (initial_shaped && argument_shaped && initial_shaped->memref_ && argument_shaped->memref_ &&
          (!SamePhysicalWindow(*initial_shaped->memref_, *argument_shaped->memref_) ||
           initial_shaped->GetMemorySpace() != argument_shaped->GetMemorySpace())) {
        argument_type =
            CloneTypeWithMemRef(argument_type, initial_shaped->memref_, initial_shaped->GetMemorySpace());
      }
      if (argument_type != argument->GetType() || initial != argument->initValue_) {
        auto updated =
            std::make_shared<IterArg>(argument->name_hint_, argument_type, initial, argument->span_);
        var_substitution_[argument] = updated;
        var_remap_[argument.get()] = updated;
      } else {
        // Cache unchanged bindings too: body uses must not recursively expand
        // the initializer chain of every enclosing loop. The base loop visitor
        // removes these entries when leaving the scope.
        var_remap_[argument.get()] = argument;
      }
      if (i >= op->return_vars_.size()) continue;
      const auto& result = op->return_vars_[i];
      auto result_type = result->GetType();
      if (auto found = rewrites_.find(result); found != rewrites_.end()) {
        result_type = found->second;
      }
      auto result_shaped = As<ShapedType>(result_type);
      if (initial_shaped && result_shaped && initial_shaped->memref_ && result_shaped->memref_ &&
          (!SamePhysicalWindow(*initial_shaped->memref_, *result_shaped->memref_) ||
           initial_shaped->GetMemorySpace() != result_shaped->GetMemorySpace())) {
        result_type =
            CloneTypeWithMemRef(result_type, initial_shaped->memref_, initial_shaped->GetMemorySpace());
      }
      if (result_type != result->GetType()) {
        var_substitution_[result] = std::make_shared<Var>(result->name_hint_, result_type, result->span_);
      }
    }
    auto result = IRMutator::VisitStmt_(op);
    if (prefix.empty()) return result;
    prefix.push_back(result);
    return std::make_shared<SeqStmts>(std::move(prefix), op->span_);
  }

 private:
  std::map<VarPtr, TypePtr> rewrites_;
  std::map<VarPtr, VarPtr> var_substitution_;
  std::map<const IterArg*, VarPtr> initializers_;
  std::map<const Stmt*, std::vector<StmtPtr>> entries_;
};

/// Collect logical SSA lifetimes before choosing physical reuse. An if result
/// is defined at its merge; its inputs are read only at their own branch yields.
/// A second statement numbering reverses sibling arms, not execution within an
/// arm. Both orders produce conservative intervals, including loop recurrence.
class LifetimeAnalyzer : public IRVisitor {
 public:
  struct Result {
    std::vector<VarPtr> ordered_defs;
    std::map<VarPtr, int> var_def_order;
    std::map<VarPtr, int> var_last_use;
    std::map<VarPtr, StmtPtr> var_def_stmt;
    std::map<VarPtr, std::pair<int, int>> var_reverse_liveness;
  };

  Result Analyze(const StmtPtr& func_body, const std::vector<VarPtr>& entry_vars = {}) {
    ReverseBranchOrderVisitor reverse;
    reverse.Run(func_body);
    reverse_order_ = std::move(reverse.stmt_order);
    reverse_end_order_ = std::move(reverse.stmt_end_order);
    for (const auto& var : entry_vars) {
      if (GetTileTypeWithMemRef(var->GetType())) RecordDefinition(var, nullptr, 0, 0);
    }
    if (func_body) VisitStmt(func_body);
    auto last_use = ComputeEffectiveLastUse();
    return {std::move(ordered_defs_), std::move(var_def_order_), std::move(last_use),
            std::move(var_def_stmt_), std::move(var_reverse_liveness_)};
  }

 protected:
  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    const auto saved = current_reverse_;
    current_reverse_ = reverse_order_.at(stmt.get()) + 1;
    IRVisitor::VisitStmt(stmt);
    current_reverse_ = saved;
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) VisitStmt(stmt);
  }
  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    const int order = current_order_++;
    if (As<TileType>(op->var_->GetType())) {
      RecordDefinition(op->var_, op, order, current_reverse_);
    }
    CollectUsesFromExpr(op->value_, order);
  }
  void VisitStmt_(const EvalStmtPtr& op) override { CollectUsesFromExpr(op->expr_, current_order_++); }
  void VisitStmt_(const YieldStmtPtr& op) override { CollectUsesFromValues(op->value_); }
  void VisitStmt_(const ReturnStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  void VisitStmt_(const IfStmtPtr& op) override {
    CollectUsesFromExpr(op->condition_, current_order_);
    VisitStmt(op->then_body_);
    if (op->else_body_) VisitStmt(*op->else_body_);
    const int merge = std::max(0, current_order_ - 1);
    const size_t reverse_merge = reverse_end_order_.at(op.get()) + 1;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& result = op->return_vars_[i];
      if (!GetTileTypeWithMemRef(result->GetType())) continue;
      RecordDefinition(result, op, merge, reverse_merge);
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override { ProcessLoop(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { ProcessLoop(op); }

 private:
  struct LoopBoundary {
    int start;
    int end;
    size_t reverse_end;
  };
  int current_order_ = 0;
  size_t current_reverse_ = 0;
  std::map<const Stmt*, size_t> reverse_order_;
  std::map<const Stmt*, size_t> reverse_end_order_;
  std::vector<LoopBoundary> loops_;
  std::vector<size_t> active_loops_;
  std::map<VarPtr, std::set<size_t>> recurring_uses_;
  std::vector<VarPtr> ordered_defs_;
  std::map<VarPtr, int> var_def_order_;
  std::map<VarPtr, int> var_raw_last_use_;
  std::map<VarPtr, StmtPtr> var_def_stmt_;
  std::map<VarPtr, VarPtr> return_var_to_init_var_;
  std::map<VarPtr, VarPtr> carrier_cache_;
  std::map<VarPtr, std::pair<int, int>> var_reverse_liveness_;

  void RecordDefinition(const VarPtr& var, const StmtPtr& stmt, int order, size_t reverse_order) {
    ordered_defs_.push_back(var);
    var_def_order_[var] = order;
    if (stmt) var_def_stmt_[var] = stmt;
    const int reverse = static_cast<int>(reverse_order);
    var_reverse_liveness_[var] = {reverse, reverse};
  }

  [[nodiscard]] VarPtr ResolveCarryInitCarrier(const ExprPtr& value) {
    auto var = AsVarLike(value);
    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr carrier;
    while (var) {
      if (auto cached = carrier_cache_.find(var); cached != carrier_cache_.end()) {
        carrier = cached->second;
        break;
      }
      if (!seen.insert(var.get()).second) break;
      path.push_back(var);
      if (auto iter_arg = As<IterArg>(var)) {
        var = AsVarLike(iter_arg->initValue_);
      } else if (auto found = return_var_to_init_var_.find(var); found != return_var_to_init_var_.end()) {
        var = found->second;
      } else {
        carrier = var;
        break;
      }
    }
    for (const auto& member : path) carrier_cache_[member] = carrier;
    return carrier;
  }

  template <typename LoopPtr>
  void ProcessLoop(const LoopPtr& op) {
    for (const auto& argument : op->iter_args_) {
      CollectUsesFromExpr(argument->initValue_, current_order_);
    }
    const size_t index = loops_.size();
    loops_.push_back({current_order_, current_order_, reverse_end_order_.at(op.get()) + 1});
    active_loops_.push_back(index);
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      CollectUsesFromExpr(op->condition_, current_order_);
    }
    VisitStmt(op->body_);
    active_loops_.pop_back();
    loops_[index].end = std::max(loops_[index].start, current_order_ - 1);
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto carrier = ResolveCarryInitCarrier(op->iter_args_[i]->initValue_);
      if (!carrier || !var_def_order_.count(carrier)) continue;
      // The carry's physical slot is required on every iteration, including a
      // zero-trip exit. A fresh body result is not its initializer.
      recurring_uses_[carrier].insert(index);
      if (i < op->return_vars_.size()) return_var_to_init_var_[op->return_vars_[i]] = carrier;
    }
  }

  void CollectUsesFromExpr(const ExprPtr& expr, int order) {
    if (!expr) return;
    VarUseCollector collector;
    collector.VisitExpr(expr);
    for (const auto& var : collector.used_vars) RecordRawUse(var, order);
  }
  void CollectUsesFromValues(const std::vector<ExprPtr>& values) {
    const int order = current_order_++;
    for (const auto& value : values) CollectUsesFromExpr(value, order);
  }
  void RecordRawUse(const VarPtr& var, int order) {
    const auto target = ResolveCarryInitCarrier(var);
    if (!target || !var_def_order_.count(target)) return;
    var_raw_last_use_[target] = std::max(var_raw_last_use_[target], order);
    auto& reverse = var_reverse_liveness_.at(target);
    reverse.second = std::max(reverse.second, static_cast<int>(current_reverse_));
    const int definition = var_def_order_.at(target);
    const auto loop = std::upper_bound(active_loops_.begin(), active_loops_.end(), definition,
                                       [&](int point, size_t index) { return point < loops_[index].start; });
    if (loop != active_loops_.end()) recurring_uses_[target].insert(*loop);
  }

  std::map<VarPtr, int> ComputeEffectiveLastUse() {
    std::map<VarPtr, int> effective;
    for (const auto& var : ordered_defs_) {
      int last = var_def_order_.at(var);
      if (auto use = var_raw_last_use_.find(var); use != var_raw_last_use_.end()) {
        last = use->second;
      }
      if (auto recurring = recurring_uses_.find(var); recurring != recurring_uses_.end()) {
        for (const size_t index : recurring->second) {
          const auto& loop = loops_[index];
          last = std::max(last, loop.end);
          auto& reverse = var_reverse_liveness_.at(var);
          reverse.second = std::max(reverse.second, static_cast<int>(loop.reverse_end));
        }
      }
      effective[var] = last;
    }
    return effective;
  }
};

/**
 * @brief Compute lifetime intervals by walking the full IR tree.
 *
 * Walks ALL statements
 * including those inside nested control flow (IfStmt/ForStmt/WhileStmt bodies).
 */
LifetimeAnalysisResult AnalyzeAllocationLifetimesImpl(const StmtPtr& func_body,
                                                      const std::vector<VarPtr>& entry_vars = {}) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Walk full IR tree to collect variable defs, uses, and ordering
  LifetimeAnalyzer analyzer;
  auto result = analyzer.Analyze(func_body, entry_vars);

  if (result.ordered_defs.empty()) {
    return {lifetimes, {}};
  }

  // Step 2: Build MemRef sharing groups (keyed by base_ Ptr identity)
  std::map<const Var*, std::vector<VarPtr>> memref_groups;
  for (const auto& var : result.ordered_defs) {
    auto tile_type = As<TileType>(var->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      const Var* base_ptr = tile_type->memref_.value()->base_.get();
      memref_groups[base_ptr].push_back(var);
    }
  }

  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  for (const auto& [base_ptr, vars] : memref_groups) {
    if (vars.size() > 1) {
      for (const auto& var : vars) {
        var_sharing_groups[var] = vars;
      }
      LOG_DEBUG << "MemRef sharing group: " << vars.size() << " variables share same MemRef";
    }
  }

  // Step 3: Compute lifetime intervals (with MemRef sharing group merging)
  std::set<VarPtr> processed_vars;

  // Pipeline-stage membership per interval representative (parsed once from the
  // defining Call's pipeline_membership attr, set by LowerPipelineLoops), plus
  // the subset whose defining op is a load (vs compute).
  std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>> pipeline_membership;
  std::set<const Var*> pipeline_load_tiles;
  std::set<const Var*> subrange_unsafe_groups;

  for (const auto& var : result.ordered_defs) {
    if (processed_vars.count(var)) {
      continue;
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;
    }

    std::vector<VarPtr> sharing_group;
    if (var_sharing_groups.count(var)) {
      sharing_group = var_sharing_groups[var];
    } else {
      sharing_group = {var};
    }

    // Compute MERGED lifetime for all variables in the sharing group
    int min_def_point = INT_MAX;
    int max_last_use = INT_MIN;
    int min_reverse_def = INT_MAX;
    int max_reverse_last_use = INT_MIN;

    for (const auto& group_var : sharing_group) {
      int def_point = result.var_def_order.count(group_var) ? result.var_def_order.at(group_var) : 0;
      int last_use = result.var_last_use.count(group_var) ? result.var_last_use.at(group_var) : def_point;

      LOG_DEBUG << "Variable " << group_var->name_hint_ << " def=" << def_point << " last_use=" << last_use;

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
      const auto& reverse = result.var_reverse_liveness.at(group_var);
      min_reverse_def = std::min(min_reverse_def, reverse.first);
      max_reverse_last_use = std::max(max_reverse_last_use, reverse.second);
    }

    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    interval.reverse_def_point = min_reverse_def;
    interval.reverse_last_use_point = max_reverse_last_use;
    auto representative_tile_type = As<TileType>(sharing_group[0]->GetType());
    INTERNAL_CHECK_SPAN(representative_tile_type != nullptr, sharing_group[0]->span_)
        << "Expected TileType for reuse interval";
    auto memory_space = representative_tile_type->GetMemorySpace();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), sharing_group[0]->span_)
        << "TileType with MemRef must have memory_space for reuse analysis";
    interval.memory_space = *memory_space;
    interval.size = GetDefinedMemRef(representative_tile_type)->size_;

    // A direct dynamic tile.slice is lowered through pto.subview, which rebuilds
    // its runtime address from the source tile and slice operands. Every other
    // dynamic-offset member is unsafe for capacity-triggered subdivision:
    // reshape/reinterpret/transpose views have no offset operand, and an unknown
    // producer must fail closed. Mark the complete sharing group so the packer
    // can keep its whole memory space on the legacy layout. Restricting only
    // this interval is insufficient — subdividing other roots can make the
    // space fit and let this pre-existing dynamic address loss reach codegen.
    for (const auto& group_var : sharing_group) {
      auto member_memref = GetTypeMemRef(group_var->GetType());
      if (!member_memref.has_value() || !*member_memref || As<ConstInt>((*member_memref)->byte_offset_)) {
        continue;
      }
      auto dit = result.var_def_stmt.find(group_var);
      auto assign = dit != result.var_def_stmt.end() ? As<AssignStmt>(dit->second) : nullptr;
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call || !IsOp(call, "tile.slice")) {
        subrange_unsafe_groups.insert(interval.variable.get());
        break;
      }
    }

    lifetimes.push_back(interval);

    // Record pipeline-stage membership for this interval's representative so the
    // reuse packer can forbid cross-stage coalescing. Every sharing-group member
    // is a tile of one pipeline clone and carries the same membership; take the
    // first non-empty one (views with no Call def simply contribute nothing).
    for (const auto& group_var : sharing_group) {
      auto dit = result.var_def_stmt.find(group_var);
      if (dit == result.var_def_stmt.end()) continue;
      auto assign = As<AssignStmt>(dit->second);
      if (!assign) continue;
      auto call = As<Call>(assign->value_);
      if (!call) continue;
      auto packed = call->GetAttr<std::string>(kPipelineMembershipAttr, std::string());
      if (!packed.empty()) {
        // Parse the membership string once here; the packer compares pre-parsed
        // vectors so it never re-parses in its O(N²) pairwise loop.
        pipeline_membership[interval.variable.get()] = ParsePipelineMembership(packed);
        if (IsOp(call, "tile.load") || IsOp(call, "tile.read")) {
          pipeline_load_tiles.insert(interval.variable.get());
        }
        break;
      }
    }

    for (const auto& group_var : sharing_group) {
      processed_vars.insert(group_var);
    }

    LOG_DEBUG << "Lifetime for sharing group (representative: " << sharing_group[0]->name_hint_
              << ", size: " << sharing_group.size() << "): [" << min_def_point << ", " << max_last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  // Primary intervals remain available to allocation constraints. The packer
  // also uses the precomputed reverse interval of each allocation group.
  std::map<const Var*, std::pair<int, int>> var_liveness;
  for (const auto& [var, def_point] : result.var_def_order) {
    int last_use = result.var_last_use.count(var) ? result.var_last_use.at(var) : def_point;
    var_liveness[var.get()] = {def_point, last_use};
  }

  return {lifetimes,
          var_sharing_groups,
          std::move(var_liveness),
          std::move(pipeline_membership),
          std::move(pipeline_load_tiles),
          std::move(subrange_unsafe_groups)};
}

// NOTE: The former tile-type reuse-compatibility gate (AreTileTypesCompatible)
// has been removed. PTO codegen binds a per-var alloc_tile to each tile, so two
// tiles that share a physical MemRef can legally carry different shapes, dtypes,
// or TileView attributes. In-place read/write hazards are handled precisely by
// not_inplace_safe() and per-operand forbid_output_alias() markers (see
// ForbidAliasCollector below).

/**
 * @brief Check if two primary lifetime intervals overlap.
 *
 * Uses <= to allow "touching" lifetimes (last_use == def_point) to share
 * buffers: within a single statement, inputs are consumed before outputs
 * are produced.
 */
static bool LifetimesOverlap(const LifetimeInterval& a, const LifetimeInterval& b) {
  return !(a.last_use_point <= b.def_point || b.last_use_point <= a.def_point);
}

// ---------------------------------------------------------------------------
// Ascend910 split-AIV  load + tpop_from_aic  in-place hazard
// ---------------------------------------------------------------------------
//
// On Ascend910 AIV functions with a non-None SplitMode, an op that consumes
// BOTH a tile.load result (or a legal-view descendant of one) AND a
// tile.tpop_from_aic value must not write its output into the same physical
// buffer as the load result it reads.  Letting the writer's output reuse that
// buffer (an in-place touch) yields silently wrong results on this hardware.
//
// MemoryReuse owns every buffer-coalescing decision, so the cleanest fix is to
// never create the hazardous sharing in the first place.  This used to be
// undone after the fact by a dedicated LegalizePTOBufferReuse split pass; the
// guard below folds that responsibility into the reuse decision.
//
// `HazardInputs` is collected by walking the function body (see
// `HazardInputCollector::Run` for why that walk runs twice):
//   - load_derived: vars produced by tile.load or by a legal view op chained
//     from a load (these alias the load's physical buffer).
//   - reads_tpop:   vars whose defining op consumes a tile.tpop_from_aic value.
// Both sets are keyed on Var identity and checked against the reuse-decision
// representatives (a sharing group's earliest-defined member, which for a
// writer is the writer's own output and for a load is the load itself).
//
// Reading an *operand* takes one extra step, because a value can reach the op
// through a loop carry.  An IterArg is never an AssignStmt def, so it is never a
// member of either set; but the carry chain it belongs to (init value, IterArg,
// yield value) has already been fused onto one MemRef base by
// MaterializeSemanticAliases, so an IterArg operand is classified by that base
// instead — see `HazardInputCollector::IsTainted`.  A carry also breaks program
// order, since the producer that taints the base may stand after the use it
// taints; `HazardInputCollector::Run` covers that with a second traversal.

// Resolve a tile Var to its MemRef base pointer (nullptr if it is not a tile or
// has no MemRef yet).  View ops share their source's base, so a view and its
// source resolve to the same pointer here.
static const Var* TileMemRefBase(const VarPtr& v) {
  auto t = As<TileType>(v->GetType());
  if (t && t->memref_.has_value() && t->memref_.value()->base_) return t->memref_.value()->base_.get();
  return nullptr;
}

/// Ops through which a `tile.load` result's taint reaches the writer below.
///
/// This is deliberately NOT one of the registry alias predicates, and the
/// distinction is the whole point.  Those answer a *semantic* question — does
/// this operator declare that its result names an argument's buffer.  The taint
/// closure needs a *physical* one — can this result end up occupying the load's
/// buffer — and after opportunistic reuse the two differ.  `tile.fillpad`
/// declares no aliasing at all, yet it is in-place safe, so `RetargetAssign`
/// may coalesce its output straight onto its input's buffer; that is asserted
/// by `test_fillpad_output_can_reuse_input`.  Deriving this set from the
/// registry drops `tile.fillpad` and reopens
/// `load -> fillpad -> writer(result, tpop_from_aic)`.
///
/// The list is therefore kept as-is, and it is an approximation with two known
/// gaps, both pre-existing:
///   * it omits `tile.assemble` / `tile.set_validshape` / `tile.tget_scale_addr`,
///     which do declare aliasing, so a load reaching the writer through one of
///     them is not tainted; and
///   * it omits every other in-place-safe op (`tile.add`, ...) whose output
///     `RetargetAssign` may likewise coalesce onto a load buffer.
/// Closing either needs the taint to follow post-reuse physical bases rather
/// than op names, which is a change to the reuse decision itself.
static bool IsLegalTileViewOp(const OpPtr& op) {
  return IsOp(op, "tile.reshape") || IsOp(op, "tile.extract") || IsOp(op, "tile.slice") ||
         IsOp(op, "tile.fillpad") || IsOp(op, "tile.fillpad_inplace") || IsOp(op, "tile.transpose_view") ||
         IsOp(op, "tile.reinterpret_view") || IsOp(op, "tensor.slice");
}

using HazardInputs = AllocationHazardInputs;

class HazardInputCollector : public IRVisitor {
 public:
  /// Classify `body` in TWO traversals, then hand the result to `Take()`.
  ///
  /// One forward walk is not enough, because a value can reach a use through a
  /// loop back edge — from a producer that stands *after* that use in program
  /// order:
  ///
  /// ```
  /// for (carry,) in range(..., init=[non_tpop]):
  ///   w = tile.add(l, carry)        // from iteration 1 on, `carry` is a tpop value
  ///   p = tile.tpop_from_aic()      // ... but the producer is only seen here
  ///   yield p                       // p must-aliases carry's base
  /// ```
  ///
  /// A single walk classifies `w` before `p` taints the carry's base, so `w`
  /// escapes `reads_tpop` and its output may be coalesced onto `l`'s buffer.
  ///
  /// Two traversals suffice, and a third could never add anything: the base
  /// sets are complete after the first pass.  `tpop_bases_` only ever takes a
  /// `tile.tpop_from_aic` def, which is order-independent; `load_derived_bases_`
  /// additionally takes view outputs, and a view shares its source's base, so
  /// marking one inserts a base that is already present.  Only the *Var* sets
  /// grow in the second pass, and those propagate in program order within it.
  /// Both traversals are O(N), so the collector stays linear.
  void Run(const StmtPtr& body) {
    VisitStmt(body);
    VisitStmt(body);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (GetTileTypeWithMemRef(op->var_->GetType())) {
      if (auto call = As<Call>(op->value_)) {
        // AsVarLike, not As<Var>: an operand may be an enclosing loop's IterArg
        // — the loop-carry shape of `tile.add(x, c_iter)` / `tile.matmul_acc(
        // c_iter, ...)`.  IterArg has its own ObjectKind, so As<Var> returns
        // null there and would drop the operand, and with it the load / tpop
        // taint the carry brings into the body.
        std::vector<VarPtr> input_vars;
        for (const auto& arg : call->args_) {
          if (auto v = AsVarLike(arg)) input_vars.push_back(v);
        }
        // load_derived closure: within one traversal defs precede uses, so a
        // view's source is already classified by the time we reach the view;
        // a source that only becomes tainted across a loop back edge is picked
        // up by the second traversal (see `Run`).
        if (IsOp(call, "tile.load")) {
          MarkLoadDerived(op->var_);
        } else if (IsLegalTileViewOp(call->op_)) {
          for (const VarPtr& in : input_vars) {
            if (IsLoadDerived(in)) {
              MarkLoadDerived(op->var_);
              break;
            }
          }
        }
        if (IsOp(call, "tile.tpop_from_aic")) MarkTpop(op->var_);
        for (const VarPtr& in : input_vars) {
          if (IsTpop(in)) {
            inputs_.reads_tpop.insert(op->var_.get());
            break;
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  HazardInputs Take() { return std::move(inputs_); }

 private:
  // Taint a var and the physical buffer it occupies.  The buffer mirror exists
  // for loop carries: an IterArg is never itself an AssignStmt def, so it can
  // never be a member of the Var-keyed sets no matter how it is spelled at the
  // use site.  MaterializeSemanticAliases (which runs immediately before this
  // pass) has already fused each carry — init value, IterArg, yield value —
  // onto ONE MemRef base, so a carry's taint is exactly the taint of the base it
  // must-aliases.
  void MarkLoadDerived(const VarPtr& v) {
    inputs_.load_derived.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) load_derived_bases_.insert(base);
  }

  void MarkTpop(const VarPtr& v) {
    tpop_vars_.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) tpop_bases_.insert(base);
  }

  [[nodiscard]] bool IsLoadDerived(const VarPtr& v) const {
    return IsTainted(inputs_.load_derived, load_derived_bases_, v);
  }

  [[nodiscard]] bool IsTpop(const VarPtr& v) const { return IsTainted(tpop_vars_, tpop_bases_, v); }

  // Var identity stays the exact test for an ordinary operand — two tiles that
  // merely share a base (a view and its source) are still classified by the view
  // allowlist above, not by the buffer.  Only a carry, which identity provably
  // cannot answer for, falls back to its must-aliased base.
  [[nodiscard]] static bool IsTainted(const std::unordered_set<const Var*>& vars,
                                      const std::unordered_set<const Var*>& bases, const VarPtr& v) {
    if (vars.count(v.get()) != 0) return true;
    if (!As<IterArg>(v)) return false;
    const Var* base = TileMemRefBase(v);
    return base != nullptr && bases.count(base) != 0;
  }

  HazardInputs inputs_;
  std::unordered_set<const Var*> tpop_vars_;
  std::unordered_set<const Var*> load_derived_bases_;  ///< MemRef bases holding a load_derived value
  std::unordered_set<const Var*> tpop_bases_;          ///< MemRef bases holding a tpop_from_aic value
};

// ---------------------------------------------------------------------------
// "Output must not alias this input" map
// ---------------------------------------------------------------------------
//
// An op may forbid its output from sharing a buffer with one or more of its
// input operands, because the hardware reads those inputs while writing the
// output (aliasing them corrupts results).  Two registry declarations feed the
// same per-output forbidden-input set, unified here:
//
//   * not_inplace_safe()        -> the op cannot run src == dst at all, so the
//                                  output must not alias ANY of its inputs
//                                  (e.g. tile.recip, tile.mrgsort_format1).
//   * forbid_output_alias(i)    -> the op is in-place-safe w.r.t. its value
//                                  operands but reads a *specific* operand
//                                  while writing dst (e.g. tile.sel's predicate
//                                  mask / tmp scratch), so dst must not alias
//                                  that one operand's buffer.
//
// Because every read input of `op->var_`'s defining call appears in `args_`,
// "forbid all inputs" is exactly "forbid every args_ entry" — so this single
// walk records, per output tile var, the input tile vars its op forbids the
// output from aliasing.  MemoryReuse then refuses to place the output on any of
// those inputs' buffers.  Distinct from the load+tpop hazard (backend-gated) —
// this is op-semantic and always collected.
// Maps each output tile Var to the input operand Vars its defining op forbids
// the output from sharing a buffer with.  Enforcement resolves each operand to
// the *physical buffer* it ends up on (following both reuse-map reassignment and
// VIEW inheritance) and blocks the output from landing there — see the use site.
using ForbidAliasMap = AllocationForbidAliasMap;

class ForbidAliasCollector : public IRVisitor {
 public:
  // `sharing_groups` (from ComputeLifetimes) maps every var that shares a MemRef
  // to its group. IdentifyReuseOpportunities keys lifetimes on each group's
  // *representative* (`sharing_group[0]`), so the forbidden set must be keyed on
  // the representative of the defining op's output too — otherwise a forbidden
  // op whose output is a non-representative group member (e.g. its result is
  // also viewed/reshaped elsewhere) would never be looked up. Forbidden input
  // *values* need no such mapping: enforcement compares physical MemRef bases,
  // which every group member already shares.
  ForbidAliasCollector(const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
                       const BufferViewAnalysis* views, const std::set<const Var*>& coalescible_transfers)
      : views_(views), coalescible_transfers_(coalescible_transfers) {
    for (const auto& [var, group] : sharing_groups) {
      if (!group.empty()) member_to_rep_[var.get()] = group[0].get();
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // An exact copy can be eliminated by copy coalescing.
    // Every remaining transfer still has to satisfy tile.move's disjointness.
    if (coalescible_transfers_.count(op->var_.get()) != 0) return;
    // Multi-result calls are assigned to a tuple temporary and unpacked by
    // subsequent TupleGetItem assignments. Carry the call's semantic
    // no-alias inputs onto every physical tile result; the tuple temporary
    // itself has no allocation interval to constrain.
    if (const auto get_item = As<TupleGetItemExpr>(op->value_)) {
      if (const VarPtr tuple = AsVarLike(get_item->tuple_)) {
        const auto pending = tuple_forbidden_.find(tuple.get());
        if (pending != tuple_forbidden_.end()) RecordForOutput(op->var_, pending->second);
      }
    }

    if (auto call = As<Call>(op->value_); call && call->op_) {
      const auto& reg = OpRegistry::GetInstance();
      if (reg.IsRegistered(call->op_->name_)) {
        const auto& entry = reg.GetEntry(call->op_->name_);
        std::vector<VarPtr> forbidden_inputs;
        auto forbid_arg = [&](size_t i) {
          if (i < call->args_.size()) {
            if (auto v = AsVarLike(call->args_[i])) forbidden_inputs.push_back(v);
          }
        };
        if (!entry.IsInplaceSafe()) {
          // Non-in-place ops forbid output aliasing active inputs. A5 TPRELU
          // retains tmp (arg 2) in the ABI but does not read it, so only src
          // and slope remain active.
          const size_t forbidden_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
          for (size_t i = 0; i < forbidden_arg_count; ++i) forbid_arg(i);
        } else {
          for (size_t i : entry.ForbidOutputAliasArgs()) forbid_arg(i);
        }
        // Narrowing is also unsafe on A2/A3: TCVT writes every row's head
        // before reading the tails, clobbering earlier rows' unread source.
        if (op_predicates::CastRequiresDisjointSource(call, !IsA5Target())) {
          forbid_arg(0);
        }
        // In-place safety permits exact aliases, not a packed output placed
        // inside a strided input window. Constrain the input's whole storage
        // family before either planner chooses addresses.
        if (views_ && !op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
          for (size_t i = 0; i < call->args_.size(); ++i) {
            if (views_->IsWindow(call->args_[i])) forbid_arg(i);
          }
        }
        if (views_ && IsOp(call, "tile.assemble")) {
          const auto target = GetTypeMemRef(call->args_[0]->GetType()).value_or(nullptr);
          const auto source = GetTypeMemRef(call->args_[1]->GetType()).value_or(nullptr);
          if (target && source && target->base_ != source->base_) forbid_arg(1);
        }
        RecordForOutput(op->var_, forbidden_inputs);
        // tile.transpose is registered not_inplace_safe(), so its output is
        // already forbidden from aliasing any input above (pto.ttrans writes
        // dst directly from src on the scalar path — dst == src corrupts).
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  ForbidAliasMap Take() { return std::move(forbidden_); }

 private:
  void RecordForOutput(const VarPtr& output, const std::vector<VarPtr>& forbidden_inputs) {
    if (!output || forbidden_inputs.empty()) return;
    if (As<TupleType>(output->GetType())) {
      tuple_forbidden_[output.get()] = forbidden_inputs;
      return;
    }
    if (!As<TileType>(output->GetType())) return;
    const auto rep_it = member_to_rep_.find(output.get());
    const Var* out_key = rep_it != member_to_rep_.end() ? rep_it->second : output.get();
    auto& recorded = forbidden_[out_key];
    recorded.insert(recorded.end(), forbidden_inputs.begin(), forbidden_inputs.end());
  }

  ForbidAliasMap forbidden_;
  const BufferViewAnalysis* views_;
  const std::set<const Var*>& coalescible_transfers_;
  std::map<const Var*, std::vector<VarPtr>> tuple_forbidden_;
  std::map<const Var*, const Var*> member_to_rep_;  ///< sharing-group member -> representative
};

/// Collect every tile operand an operator declares as a WRITTEN WORKSPACE.
///
/// `set_workspace_arg(i)` means "compiler-supplied scratch: the hardware writes
/// it, but it carries no result the caller reads", and `set_arg_effect(i, ...)`
/// says whether that write happens. Such a buffer is written by the intrinsic
/// itself, on whatever pipe its implementation chooses -- A2/A3's TSELS stages
/// its scalar into `tmp` from the SCALAR pipe while every tile operand is read
/// on the vector pipe. Program order does not order a write against a read
/// still in flight on another pipe, and the write never reaches the generated
/// `.pto` as a write (operands are emitted positionally), so no downstream sync
/// insertion can see the WAR. The buffer therefore must not be inherited from a
/// value that dies before the workspace is defined.
///
/// Keyed on the registry rather than on operator names or a backend flag: any
/// operator declaring a written workspace is exposed to the same hazard, and
/// PyPTO has no way to know which pipe a given intrinsic writes from.
class WrittenWorkspaceCollector : public IRVisitor {
 public:
  void VisitExpr_(const CallPtr& op) override {
    Record(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  // The registry keys both call-like kinds the same way; classifying only Call
  // would silently skip an operator launched through pl.submit.
  void VisitExpr_(const SubmitPtr& op) override {
    Record(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  // Only these two repeat a body once MemoryReuse runs: `pl.pipeline` is
  // lowered to a loop by LowerPipelineToSlots / LowerPipelineLoops (passes
  // 30-31), well before this pass.
  void VisitStmt_(const ForStmtPtr& op) override {
    ++loop_depth_;
    IRVisitor::VisitStmt_(op);
    --loop_depth_;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ++loop_depth_;
    IRVisitor::VisitStmt_(op);
    --loop_depth_;
  }

  std::unordered_set<const Var*> TakeVars() { return std::move(workspaces_); }
  std::unordered_set<const Var*> TakeBases() { return std::move(bases_); }
  std::unordered_set<const Var*> TakeLoopingVars() { return std::move(looping_workspaces_); }
  std::unordered_set<const Var*> TakeLoopingBases() { return std::move(looping_bases_); }

 private:
  void Record(const OpPtr& op, const std::vector<ExprPtr>& args) {
    // LookupOpEntry, not a name lookup: a Submit's callee is a GlobalVar naming
    // a *function*, and resolving that against the operator registry would hand
    // a user function whose name collides with an operator that operator's
    // workspace contract.
    const auto* entry = LookupOpEntry(op);
    if (entry == nullptr) return;
    for (const size_t index : entry->GetWorkspaceArgs()) {
      if (index >= args.size() || !entry->MayWriteArg(index)) continue;
      // AsVarLike, not As<Var>: a workspace can reach the op as a loop IterArg
      // when the call sits inside a loop body.
      auto workspace = AsVarLike(args[index]);
      if (!workspace) continue;
      workspaces_.insert(workspace.get());
      // Record the allocation base as well -- see `written_workspace_bases`.
      const Var* base = TileMemRefBase(workspace);
      if (base != nullptr) bases_.insert(base);
      if (loop_depth_ > 0) {
        looping_workspaces_.insert(workspace.get());
        if (base != nullptr) looping_bases_.insert(base);
      }
    }
  }

  int loop_depth_ = 0;
  std::unordered_set<const Var*> workspaces_;
  std::unordered_set<const Var*> bases_;
  std::unordered_set<const Var*> looping_workspaces_;
  std::unordered_set<const Var*> looping_bases_;
};

/// True only for Ascend910 AIV split-mode functions, which need the load +
/// tpop_from_aic in-place hazard guard.  All other backends / function kinds
/// reuse buffers freely.  Defensive against unit-test contexts that run
/// MemoryReuse without a configured backend.
bool NeedsLoadTpopHazardGuard(const FunctionPtr& func) {
  if (func->func_type_ != FunctionType::AIV) return false;
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  if (ctx == nullptr) return false;
  if (!ctx->GetBackendHandler()->RequiresSplitLoadTpopWorkaround()) return false;
  // A split AIV function needs the guard. The split may be signalled either as a
  // function-level mode (single-mode / standalone kernels carry ``split``) or
  // ONLY as the mode-agnostic ``split_aiv`` marker — multi-mode explicit
  // pl.split_aiv regions have their per-region modes lowered + erased by
  // LowerAutoVectorSplit, so no single function-level mode survives. Key on the
  // marker too so the guard still fires for them.
  if (func->HasAttr(kAttrSplitAiv) && func->GetAttr<bool>(kAttrSplitAiv, false)) return true;
  const auto split_mode = func->GetSplitMode();
  return split_mode.has_value() && *split_mode != SplitMode::None;
}

/// Optional early coalescing must leave target hazards and pipeline allocation
/// choices to the planner. Reuse its collectors rather than duplicating their
/// operator/backend rules, and protect both sides of any proposed merge.
BranchStorageProtection CollectBranchStorageProtection(const FunctionPtr& function, const StmtPtr& body) {
  BranchStorageProtection protection;
  auto& hazard_bases = protection.no_mutable_alias_bases;
  if (NeedsLoadTpopHazardGuard(function)) {
    HazardInputCollector collector;
    collector.Run(body);
    const auto hazard = collector.Take();
    for (const auto* value : hazard.load_derived) {
      if (auto tile = GetTileTypeWithMemRef(value->GetType())) {
        hazard_bases.insert(GetDefinedMemRef(tile)->base_.get());
      }
    }
  }

  WrittenWorkspaceCollector workspaces;
  workspaces.VisitStmt(body);
  const auto workspace_bases = workspaces.TakeBases();
  hazard_bases.insert(workspace_bases.begin(), workspace_bases.end());
  protection.immovable_bases = hazard_bases;

  class PipelineAllocationCollector : public IRVisitor {
   public:
    explicit PipelineAllocationCollector(std::set<const Var*>& bases) : bases_(bases) {}

    void VisitExpr(const ExprPtr&) override {}

    void VisitStmt_(const AssignStmtPtr& assign) override {
      const auto call = As<Call>(assign->value_);
      if (call && !call->GetAttr<std::string>(kPipelineMembershipAttr, std::string()).empty()) {
        if (const auto* base = TileMemRefBase(assign->var_)) bases_.insert(base);
      }
    }

   private:
    std::set<const Var*>& bases_;
  } pipelines(protection.immovable_bases);
  pipelines.VisitStmt(body);
  return protection;
}

/**
 * @brief Identify memory reuse opportunities from lifetime intervals.
 *
 * Global first-fit-decreasing packing: within each memory space, intervals are
 * placed **largest-first**, and every later interval joins the first existing
 * buffer all of whose members it can share with (see `can_share` below).  A
 * buffer's allocated size is fixed by its first (largest) member, so admitting a
 * smaller member afterwards is free — which makes first-fit cost-optimal for
 * this objective and, crucially, lets a *later, larger* interval host an
 * *earlier, smaller* one.  The former definition-order greedy had a
 * one-directional size gate (`source.size >= target.size`) so it could only let
 * a smaller interval reuse an earlier, larger buffer; two lifetime-disjoint L0
 * cube-input tiles whose small one was defined first were never coalesced.
 *
 * The representative (each buffer's first/largest member) is the MemRef every
 * other member is rebased onto downstream; its alloc dominates the whole
 * function (InitMemRef hoists every tile.alloc to the function-body head), so a
 * representative defined *after* some of its members is safe.  Because the
 * packer no longer processes intervals in program order, every pairwise gate
 * below (load+tpop hazard, no-alias) is checked in **both directions**.
 *
 * @param hazard  Ascend910 load + tpop_from_aic guard inputs.  Empty when the
 *                guard is inactive (non-910B / non-split-AIV), in which case the
 *                hazard check is a no-op and reuse behaviour is unchanged.
 * @param forbid_alias  Per-output forbidden-input map (ForbidAliasCollector):
 *                an op may forbid its output from sharing a buffer with specific
 *                input operands it reads while writing the output in place.
 *
 * Complexity: the existing first-fit pack compares O(M^2) allocation pairs.
 * Each pair's lifetime check compares precomputed interval bounds in O(1),
 * without scanning aliases or querying individual values. The added reverse
 * numbering and summaries take O(N log N) work and O(N) storage. Capacity
 * retries retain the existing kMaxShedRepacks bound: the packing search remains
 * O(K * M^2) for M allocation groups and K attempts. This preserves its existing
 * super-linear optimization boundary rather than adding a logarithmic factor.

 */
// Cross-group shed objective (see docs/en/dev/passes/37-memory_reuse.md, pipeline-stage guard): when a
// space overflows at every group's max-affordable depth,
// the packer lowers one pipeline group's double-buffering depth by a residue. The MaxRelief heuristic
// selects **which** group loses a level — lower score sheds first; ties always break by lowest group id
// (deterministic). MaxRelief is used unconditionally: the pluggable objective (and its ArrivalOrder
// alternative) was retired because it never diverged from MaxRelief on any real kernel — the multi-group
// forced-shed it needs to differ does not occur in practice.
struct ShedCandidate {
  int32_t group;         ///< pipeline group id (the tie-break key)
  uint64_t bytes_freed;  ///< align(slot_g): bytes reclaimed by dropping one residue buffer of this group
};

/// MaxRelief: shed the group that frees the most bytes per level ⇒ fewest levels lost to fit the space.
inline double ScoreMaxRelief(const ShedCandidate& c) { return -static_cast<double>(c.bytes_freed); }

/// One tile independently bound to a declared allocation, with its lifetime.
struct BoundMember {
  VarPtr var;
  int def_point;
  int last_use_point;
};

/// Group the tiles a user *independently bound* to each pinned buffer.
///
/// A tile whose defining op inherits its source's buffer (a view, an in-place
/// op, or a bare SSA alias) lands on the same base without the author binding it
/// there — it is the same data, so overlapping with its source is expected and
/// legal. Only independently produced tiles compete for the buffer's bytes, and
/// only those are checked for lifetime overlap.
class IndependentlyBoundCollector : public IRVisitor {
 public:
  IndependentlyBoundCollector(const std::set<const Var*>& pinned_bases,
                              const std::map<const Var*, std::pair<int, int>>& var_liveness)
      : pinned_bases_(pinned_bases), var_liveness_(var_liveness) {}

  /// One slot of a declared allocation: its base, and the byte offset that tells
  /// slots of the same allocation apart. `InitMemRef` has already consumed the
  /// slot index into that offset, so a null offset (a dynamic index) collapses to
  /// "some slot" — see the note on `by_slot`.
  struct SlotKey {
    const Var* base;
    int64_t byte_offset;

    bool operator<(const SlotKey& other) const {
      if (base != other.base) return base < other.base;
      return byte_offset < other.byte_offset;
    }
  };

  /// Slot -> the tiles bound to it, with their lifetimes.
  ///
  /// Keyed per *slot*, not per allocation: two tiles on different slots of one
  /// declared allocation are supposed to be live together — that is what a
  /// ping-pong is — and only tiles landing on the *same* slot can corrupt each
  /// other. Tiles whose offset is a runtime expression are skipped entirely,
  /// since which slot they occupy is not a static fact.
  std::map<SlotKey, std::vector<BoundMember>> by_slot;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) Record(op);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void Record(const AssignStmtPtr& op) {
    auto memref = GetTypeMemRef(op->var_->GetType());
    if (!memref.has_value() || !(*memref)->base_) return;
    const Var* base = (*memref)->base_.get();
    if (pinned_bases_.count(base) == 0) return;
    if (AsVarLike(op->value_)) return;  // bare SSA alias — same data
    if (auto call = As<Call>(op->value_);
        call && call->op_ && op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
      return;
    }
    auto liveness = var_liveness_.find(op->var_.get());
    if (liveness == var_liveness_.end()) return;
    // A runtime slot index leaves no static slot to attribute the tile to, so
    // there is nothing to compare it against. The author owns the rotation in
    // that case; isolation from other allocations still holds.
    auto const_offset = As<ConstInt>((*memref)->byte_offset_);
    if (!const_offset) return;
    by_slot[{base, const_offset->value_}].push_back(
        {op->var_, liveness->second.first, liveness->second.second});
  }

  const std::set<const Var*>& pinned_bases_;
  const std::map<const Var*, std::pair<int, int>>& var_liveness_;
};

/// Reject a declared allocation whose independently bound tiles are co-live.
///
/// Binding two co-live tiles to one buffer is not a reuse decision, it is data
/// corruption: the later write lands on bytes the earlier tile still needs. The
/// packer refuses such a pairing among its own candidates; a hand-written
/// binding bypasses that path, so it is re-checked here. Touching is allowed —
/// one tile's last read may be the statement that produces the next.
///
/// Sorted sweep rather than a pairwise scan: a *valid* program is exactly one
/// where every pair is disjoint, so a pairwise check never exits early and would
/// be quadratic in the member count — which grows with the IR once a binding
/// inside a loop body is unrolled.
void ValidateDeclaredAllocs(const StmtPtr& body, const std::set<const Var*>& pinned_bases,
                            const std::map<const Var*, std::pair<int, int>>& var_liveness) {
  if (pinned_bases.empty()) return;
  IndependentlyBoundCollector collector(pinned_bases, var_liveness);
  collector.VisitStmt(body);

  for (auto& [slot, members] : collector.by_slot) {
    const Var* base = slot.base;
    std::sort(members.begin(), members.end(),
              [](const BoundMember& a, const BoundMember& b) { return a.def_point < b.def_point; });
    for (size_t i = 1; i < members.size(); ++i) {
      const BoundMember& prev = members[i - 1];
      const BoundMember& cur = members[i];
      CHECK_SPAN(prev.last_use_point <= cur.def_point, cur.var->span_)
          << "Tiles '" << prev.var->name_hint_ << "' and '" << cur.var->name_hint_
          << "' are both bound to the same slot of the declared allocation '" << base->name_hint_
          << "' but are live at the same time (" << prev.var->name_hint_ << ": [" << prev.def_point << ", "
          << prev.last_use_point << "], " << cur.var->name_hint_ << ": [" << cur.def_point << ", "
          << cur.last_use_point
          << "]). Sharing one slot would overwrite data that is still needed — put them on different "
             "slots, declare separate allocations, or bind only tiles whose lifetimes do not overlap.";
    }
  }
}

/// Collect the full byte extent of allocations declared by the author.
///
/// InitMemRef hoists every alloc to the function body's top level. When an
/// allocation exists, prepending it forms a SeqStmts; a function with no
/// allocation may legitimately retain a singleton body such as ReturnStmt.
/// A multi-slot declaration is larger than any one member MemRef, so DSA-RP
/// must take the extent from the alloc statement rather than reconstruct it
/// from members.
std::map<const Var*, uint64_t> CollectPinnedAllocSizes(const StmtPtr& body, const Span& span,
                                                       const std::string& consumer) {
  INTERNAL_CHECK_SPAN(body, span) << consumer << " expected a non-null function body";

  std::map<const Var*, uint64_t> pinned_allocations;
  auto collect = [&](const StmtPtr& stmt) {
    auto base = GetPinnedAllocBase(stmt);
    if (!base) return;
    auto assign = As<AssignStmt>(stmt);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    auto size = call && call->args_.size() >= 2 ? As<ConstInt>(call->args_[1]) : nullptr;
    INTERNAL_CHECK_SPAN(size && size->value_ > 0, stmt->span_)
        << consumer << " expected declared allocation '" << base->name_hint_
        << "' to have a positive constant byte extent after InitMemRef";
    pinned_allocations.emplace(base.get(), static_cast<uint64_t>(size->value_));
  };

  if (auto top_level = As<SeqStmts>(body)) {
    for (const auto& stmt : top_level->stmts_) collect(stmt);
  } else {
    collect(body);
  }
  return pinned_allocations;
}

struct ReusePlacement {
  VarPtr representative;
  uint64_t byte_offset = 0;
};

using ReuseMap = std::map<VarPtr, ReusePlacement>;

struct PackedMember {
  size_t interval_index = 0;
  uint64_t byte_offset = 0;
};

struct PackedBuffer {
  std::vector<PackedMember> members;
  uint64_t capacity = 0;
  bool pinned = false;
  BufferRootCompatibility::Root buffer_root;
};

ReuseMap IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const ForbidAliasMap& copy_separation,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>>& pipeline_membership,
    const std::set<const Var*>& pipeline_load_tiles, const std::set<const Var*>& subrange_unsafe_groups,
    const std::map<MemorySpace, uint64_t>& reserved_end_by_space, const std::set<const Var*>& pinned_bases,
    const FunctionPtr& func, std::vector<Diagnostic>* out_hints) {
  ReuseMap reuse_map;
  std::vector<BufferRootCompatibility::Root> buffer_roots(lifetimes.size());
  const auto* buffer_context = PassContext::Current();
  if (buffer_context && buffer_context->GetEnableBufferIR()) {
    BufferRootCompatibility compatibility(func->body_);
    for (size_t i = 0; i < lifetimes.size(); ++i) buffer_roots[i] = compatibility.Get(lifetimes[i].variable);
  }

  // A declared allocation (`pl.Tile[..., pl.MemRef("name"), ...]`) has exactly the membership
  // the kernel author wrote. The packer neither adds tiles to it nor moves its
  // tiles into someone else's buffer — that is the whole point of naming it:
  // coalescing tiles whose lifetimes merely happen not to overlap introduces a
  // false dependency that serializes them.
  //
  // Resolved once per interval rather than inside `can_share`: that predicate is
  // the innermost step of the O(M^2) pack, which itself re-runs per shed step, so
  // a type cast + set lookup there would be paid M^2 times — and by every kernel,
  // including the vast majority that declare no allocation at all. Same reason
  // `pipeline_membership` is pre-parsed (see LifetimeAnalysisResult).
  std::vector<bool> is_pinned(lifetimes.size(), false);
  if (!pinned_bases.empty()) {
    for (size_t i = 0; i < lifetimes.size(); ++i) {
      if (!lifetimes[i].variable) continue;
      auto memref = GetTypeMemRef(lifetimes[i].variable->GetType());
      is_pinned[i] = memref.has_value() && (*memref)->base_ && pinned_bases.count((*memref)->base_.get()) > 0;
    }
  }

  // Members of a sharing group (the vars that already physically share one base).
  // Returns a pointer (or nullptr) to avoid copying the vector in the O(M^2) loop.
  auto group_members = [&sharing_groups](const VarPtr& rep) -> const std::vector<VarPtr>* {
    auto it = sharing_groups.find(rep);
    return it != sharing_groups.end() ? &it->second : nullptr;
  };
  // Maps a tile's *original* MemRef base to the base its buffer is coalesced
  // onto by a committed reuse decision.  A reuse retargets the reused var (and
  // every VIEW that inherited its base) from its own base onto the
  // representative's base; resolve_base() chases that chain so the no-alias
  // guard sees a forbidden operand's *physical* buffer even when the operand is
  // a view whose own base is now stale.  Populated as decisions are committed.
  std::map<const Var*, const Var*> base_remap;
  std::function<const Var*(const Var*)> resolve_base = [&](const Var* b) -> const Var* {
    while (b != nullptr) {
      auto it = base_remap.find(b);
      if (it == base_remap.end() || it->second == b) break;
      b = it->second;
    }
    return b;
  };

  // The physical buffer base a tile var currently occupies: follow any committed
  // reuse onto its representative, then chase base coalescing / VIEW inheritance.
  auto physical_base = [&](const VarPtr& v) -> const Var* {
    VarPtr root = v;
    while (reuse_map.count(root)) root = reuse_map.at(root).representative;
    return resolve_base(TileMemRefBase(root));
  };

  // Ascend910 split-AIV hazard (directional): `writer` consumes a
  // tile.tpop_from_aic value and is defined exactly where the load-derived
  // `input` is last read, so placing writer's output on input's buffer forms the
  // in-place load+tpop touch.  `hazard` is empty off-910B, so this is a no-op.
  auto hazard_blocks = [&hazard](const LifetimeInterval& writer, const LifetimeInterval& input) {
    return writer.def_point == input.last_use_point && hazard.reads_tpop.count(writer.variable.get()) != 0 &&
           hazard.load_derived.count(input.variable.get()) != 0;
  };

  // Written-workspace hazard (directional): the intrinsic writes this operand
  // itself, on a pipe PyPTO cannot see, and nothing orders that write against a
  // read of the same buffer still in flight.  Block only the direction that
  // lets the workspace INHERIT storage from a value that dies before it is
  // defined -- a later value joining the workspace's buffer is safe (it has
  // long retired), which keeps the legal `dst` reuse that
  // `test_sels_output_may_reuse_dead_tmp` pins.
  auto in_workspace_set = [](const LifetimeInterval& interval, const std::unordered_set<const Var*>& vars,
                             const std::unordered_set<const Var*>& bases) {
    if (vars.count(interval.variable.get()) != 0) return true;
    const Var* base = TileMemRefBase(interval.variable);
    return base != nullptr && bases.count(base) != 0;
  };
  auto is_written_workspace = [&hazard, &in_workspace_set](const LifetimeInterval& interval) {
    return in_workspace_set(interval, hazard.written_workspaces, hazard.written_workspace_bases);
  };
  auto is_looping_written_workspace = [&hazard, &in_workspace_set](const LifetimeInterval& interval) {
    return in_workspace_set(interval, hazard.looping_written_workspaces,
                            hazard.looping_written_workspace_bases);
  };
  auto workspace_blocks = [&is_written_workspace, &is_looping_written_workspace](
                              const LifetimeInterval& workspace, const LifetimeInterval& other) {
    // Inside a loop the directional argument collapses: the back edge re-writes
    // the workspace from the scalar pipe on iteration i+1 while iteration i's
    // vector reads of anything sharing that buffer -- the TSELS `dst`, say, and
    // whatever consumes it -- can still be in flight. Lifetime order within one
    // iteration proves nothing about the next, so a looping workspace takes an
    // exclusive buffer.
    if (is_looping_written_workspace(workspace)) return true;
    return is_written_workspace(workspace) && other.last_use_point <= workspace.def_point;
  };

  // No-alias guard (directional): `writer`'s defining op forbids its output from
  // sharing a buffer with one or more input operands — either because the op is
  // not_inplace_safe (forbids ALL inputs, e.g. tile.recip) or because it marks a
  // specific operand via forbid_output_alias (e.g. tile.sel's mask/tmp, read by
  // the TSEL intrinsic while writing dst).  Both feed `forbid_alias`.  Block if
  // any forbidden operand resolves to the physical buffer `occupant` sits on.
  // Each operand's buffer is found by following its reuse chain to its owner and
  // taking the MemRef base — catching both an operand reassigned by an earlier
  // reuse decision AND one occupying the buffer via VIEW inheritance (a reshape
  // / slice / extract shares its source's base with no reuse-map entry).
  auto forbid_blocks = [&](const LifetimeInterval& writer, const LifetimeInterval& occupant,
                           const ForbidAliasMap& constraints) {
    auto fa_it = constraints.find(writer.variable.get());
    if (fa_it == constraints.end()) return false;
    const Var* occ_base = physical_base(occupant.variable);
    if (occ_base == nullptr) return false;
    for (const VarPtr& operand : fa_it->second) {
      if (physical_base(operand) == occ_base) return true;
    }
    return false;
  };

  // Pipeline ping-pong guard (symmetric): two tiles that share a pipeline group
  // with a *different* stage index belong to replicated clones the scheduler
  // overlaps, so collapsing them onto one buffer injects a false write-after-read
  // that serializes the stages — even though their program-order lifetimes are
  // disjoint (that disjointness is exactly what pipelining hides). Non-pipelined
  // tiles carry no membership and never trigger this. See ``kPipelineMembershipAttr``.
  //
  // Default (capacity-gated, #1475): protect concurrent clones in EVERY space —
  // including the L0 matmul spaces (Left/Right/Acc/Bias) and regardless of whether
  // a tile is a load or a `tile.move` result — up to the max-affordable
  // double-buffering depth `F_g` (see below). This is what fixes the L0b operand
  // collapse: the binding matmul operands are L0 `tile.move` results, so the legacy
  // predicate below skipped them and merged 8 → 1.
  //
  // Legacy predicate (the `!gated` fallback, used only when a space's capacity is
  // unknown or when `force_legacy` re-packs after an unfittable shed): the L0
  // spaces are exempt entirely and only *load* buffers get per-stage privacy.
  // Forbidding *all* cross-stage reuse without a capacity check is infeasible — F
  // full copies of every intermediate overflow the budget (e.g. stage=4 RMSNorm
  // needs 4x67KB > 188KB UB), and 2x64KB Right > 64KB L0b — so the ungated path
  // blocks iff the two tiles are same-group / different-stage AND at least one is a
  // load. The capacity gate replaces this coarse heuristic with an exact per-space
  // fit; the predicate is retained verbatim only as the never-worse-than-legacy floor.
  auto is_l0_space = [](MemorySpace s) {
    return s == MemorySpace::Left || s == MemorySpace::Right || s == MemorySpace::Acc ||
           s == MemorySpace::Bias || s == MemorySpace::LeftScale || s == MemorySpace::RightScale;
  };
  // Capacity-gated (#1475): keep software-pipelined operands in separate buffers so the pipeline
  // stages double-buffer instead of serializing on a shared buffer. #1900's `pipeline_membership` tags
  // give each operand its (group, stage); `stage` is the clone index 0..F-1 of a `pl.pipeline(stage=F)`
  // group whose clones run concurrently and must occupy distinct buffers. Each group `g` gets an
  // affordable residue count `F_g = min(D_g, ⌊C_s / slot_g⌋)`; clone stage k lands in residue
  // `ordinal(k) mod F_g`, so clones < F_g apart never share (exact double-buffering when F_g == D_g,
  // maximal spread when capacity forces F_g < D_g). This adjacency guarantee is structural and
  // model-free — it is `mod k`, kept. `F_g` is *mutable*: the FFD driver's shed loop lowers one group's
  // depth (re-packing, exact SpaceFootprint fit) when a space overflows, and `F_g == 1` collapses that
  // group to the legacy merge. Unknown capacity (`cap == 0`) ⇒ `F_g = 1` (conservative merge — never a
  // separation we cannot verify fits). The whole-space fit is now the allocator's realized footprint
  // (Primitive A, in the driver below), not a `Σ` estimate — so co-resident tiles and alignment are exact.
  std::map<std::pair<MemorySpace, int32_t>, uint64_t> group_slot;  // (space, group) -> max tile
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_depth;  // (space, group) -> F_g (mutable)
  std::map<std::pair<MemorySpace, int32_t>, std::map<int32_t, int32_t>>
      group_stage_ordinal;                                                   // stage->ordinal
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_requested_depth;  // (space, group) -> D_g (stages)
  std::set<MemorySpace> force_legacy_spaces;  // spaces whose gated packing overflowed → legacy fallback fired
  // Spaces with a *known* (non-zero) capacity — only these are capacity-gated. A space whose bound is
  // unknown (`cap == 0`, incl. no backend configured) falls through to the legacy predicate: gating it to
  // F_g == 1 would merge *everything*, dropping even the legacy non-L0 load-only separation (#1900) and
  // thus separating strictly LESS than legacy. Legacy-for-unknown makes "never worse than legacy" hold
  // for separation too, not just for overflow.
  std::set<MemorySpace> capacity_known_spaces;
  {
    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
    std::map<std::pair<MemorySpace, int32_t>, std::set<int32_t>> group_stages;
    for (const auto& iv : lifetimes) {
      auto it = pipeline_membership.find(iv.variable.get());
      if (it == pipeline_membership.end()) continue;
      for (const auto& [g, st] : it->second) {
        auto key = std::make_pair(iv.memory_space, g);
        group_slot[key] = std::max(group_slot[key], iv.size);
        group_stages[key].insert(st);
      }
    }
    for (const auto& [key, slot] : group_slot) {
      // Optimistic per-slot depth using the *raw* space size (not cap − reserved_start): this is only the
      // initial F_g. The shed loop below re-checks with the exact reserved-aware SpaceFootprint and lowers
      // F_g further if the reserved region or co-resident tiles don't actually leave room.
      const uint64_t cap = be ? be->GetMemSize(key.first) : 0;
      const int32_t depth = static_cast<int32_t>(group_stages[key].size());
      int32_t f = 1;  // cap == 0 (unknown capacity) ⇒ this space is NOT gated (legacy predicate below)
      if (cap != 0 && slot != 0) {
        f = static_cast<int32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(static_cast<uint64_t>(depth), cap / slot)));
        capacity_known_spaces.insert(key.first);
      }
      group_depth[key] = f;
      group_requested_depth[key] = depth;
      int32_t ordinal = 0;  // stages may be sparse; compare dense ordinals mod F, not raw stage mod F
      for (const int32_t st : group_stages[key]) group_stage_ordinal[key][st] = ordinal++;
    }
  }
  // `force_legacy` (set by the driver's fallback below) makes the guard behave exactly like the
  // legacy predicate for a space whose gated packing overflowed — so the fallback is legacy by construction.
  bool force_legacy = false;
  auto pipeline_blocks = [&pipeline_membership, &pipeline_load_tiles, &is_l0_space, &group_depth,
                          &group_stage_ordinal, &force_legacy,
                          &capacity_known_spaces](const LifetimeInterval& a, const LifetimeInterval& b) {
    // The binding matmul operands live in L0 and are `tile.move` results, so the legacy guard's
    // `is_l0_space` exemption AND its load-only restriction both skip them — the per-stage operands
    // merge (8 → 1) and the cube matmuls serialize. The capacity-gated path (always on) protects
    // pipeline operands in every space and regardless of load/move, up to the max-affordable
    // double-buffering depth. A space whose capacity is unknown (`cap == 0`) is NOT gated — it uses the
    // legacy predicate, so gated packing is never worse than legacy there. (a and b share a memory
    // space — reuse only happens within a space.)
    const bool gated = !force_legacy && capacity_known_spaces.count(a.memory_space) != 0;
    if (!gated && is_l0_space(a.memory_space)) return false;  // legacy: L0 exempt
    auto ia = pipeline_membership.find(a.variable.get());
    if (ia == pipeline_membership.end()) return false;
    auto ib = pipeline_membership.find(b.variable.get());
    if (ib == pipeline_membership.end()) return false;
    if (gated) {
      // Keep separate iff a shared group's dense stage ordinals fall in different residues mod F_g (the
      // current, possibly-shed, affordable residue count). F_g == 1 ⇒ same residue ⇒ mergeable (legacy).
      // The whole-space fit is decided by the driver's exact SpaceFootprint, not here.
      for (const auto& [ga, sa] : ia->second) {
        for (const auto& [gb, sb] : ib->second) {
          if (ga != gb) continue;
          const auto key = std::make_pair(a.memory_space, ga);
          auto dit = group_depth.find(key);
          const int32_t k = (dit != group_depth.end() && dit->second > 0) ? dit->second : 1;
          int32_t ra = sa;
          int32_t rb = sb;
          auto omit = group_stage_ordinal.find(key);
          if (omit != group_stage_ordinal.end()) {
            auto oa = omit->second.find(sa);
            if (oa != omit->second.end()) ra = oa->second;
            auto ob = omit->second.find(sb);
            if (ob != omit->second.end()) rb = ob->second;
          }
          if (((ra % k) + k) % k != ((rb % k) + k) % k) return true;  // different buffer → separate
        }
      }
      return false;
    }
    if (!PipelineMembershipsConflict(ia->second, ib->second)) return false;  // not cross-stage
    // Legacy: block only if at least one side is a load buffer.
    return pipeline_load_tiles.count(a.variable.get()) != 0 ||
           pipeline_load_tiles.count(b.variable.get()) != 0;
  };

  // Can `cand` join a single physical buffer that already holds `member`?
  // Lifetimes must not overlap (touching is allowed: a buffer's reader is
  // consumed before the writer at the same statement produces its output), and
  // neither directional gate may block. No tile-type / size check is needed:
  // PTO binds a per-var alloc_tile so differing shapes/dtypes legally alias one
  // base, and largest-first ordering guarantees the buffer is sized to its
  // representative (no member is ever larger than the buffer it joins).
  bool placing_subranges = false;
  auto can_share = [&](const LifetimeInterval& cand, const LifetimeInterval& member) {
    // Disjointness in either legal sibling order proves that the groups cannot
    // be co-live. Overlap in both orders is conservative for nested branches.
    // Both hulls already include every mandatory alias, so the existing
    // quadratic root-pair search needs only constant-time bounds checks.
    const bool reverse_overlap = !(cand.reverse_last_use_point <= member.reverse_def_point ||
                                   member.reverse_last_use_point <= cand.reverse_def_point);
    if (LifetimesOverlap(cand, member) && reverse_overlap) return false;
    if (hazard_blocks(cand, member) || hazard_blocks(member, cand)) return false;
    if (workspace_blocks(cand, member) || workspace_blocks(member, cand)) return false;
    if (forbid_blocks(cand, member, forbid_alias) || forbid_blocks(member, cand, forbid_alias)) return false;
    // A remaining copy permits identical or disjoint windows, never a partial
    // overlap. Only whole-root placement proves identity for these transfers.
    if (placing_subranges &&
        (forbid_blocks(cand, member, copy_separation) || forbid_blocks(member, cand, copy_separation))) {
      return false;
    }
    if (pipeline_blocks(cand, member)) return false;  // symmetric — one call suffices
    return true;
  };

  // Group interval indices by memory space — reuse only happens within a space.
  std::map<MemorySpace, std::vector<size_t>> by_space;
  for (size_t i = 0; i < lifetimes.size(); ++i) {
    by_space[lifetimes[i].memory_space].push_back(i);
  }

  // Allocator policy for the exact per-space footprint (Primitive A) — only when a backend is configured.
  const backend::Backend* pack_be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
  auto alloc_policy = pack_be ? pack_be->CreateMemoryAllocatorPolicy() : nullptr;

  for (auto& [space_binding, indices_binding] : by_space) {
    // Bind the structured bindings to regular local references: the lambdas below capture by `[&]`,
    // and capturing a structured binding name directly is only a C++20 extension (project is C++17).
    auto& space = space_binding;
    auto& indices = indices_binding;
    // Largest-first; ties broken by definition order for determinism and so that
    // equal-size workloads reproduce the prior definition-order grouping.
    std::sort(indices.begin(), indices.end(), [&lifetimes](size_t a, size_t b) {
      if (lifetimes[a].size != lifetimes[b].size) return lifetimes[a].size > lifetimes[b].size;
      if (lifetimes[a].def_point != lifetimes[b].def_point) {
        return lifetimes[a].def_point < lifetimes[b].def_point;
      }
      return a < b;
    });

    // Dynamic non-slice views lose their symbolic address in
    // AllocateMemoryAddr and cannot reconstruct it in PTO codegen. Disable
    // subdivision for the complete space: compacting an unrelated group can be
    // enough to turn a legacy overflow into silently wrong generated code.
    const bool subrange_space_safe = std::none_of(indices.begin(), indices.end(), [&](size_t idx) {
      return subrange_unsafe_groups.count(lifetimes[idx].variable.get()) != 0;
    });

    // Subrange placement moves the complete root allocation, not the live
    // windows of its alias/view members.  The root must therefore have a static
    // allocation offset, while a symbolic member offset is safe: rebasing keeps
    // that expression relative to the root and the packer continues to reserve
    // the root's complete [placement, placement + size) range.  Constant member
    // offsets are still checked here so malformed static ranges fail closed.
    std::vector<bool> whole_root_subrange_eligible(lifetimes.size(), false);
    for (size_t idx : indices) {
      if (subrange_unsafe_groups.count(lifetimes[idx].variable.get()) != 0) continue;
      const auto root_memref = GetTypeMemRef(lifetimes[idx].variable->GetType());
      if (!root_memref.has_value() || !*root_memref) continue;
      const auto root_offset = As<ConstInt>((*root_memref)->byte_offset_);
      if (!root_offset || root_offset->value_ < 0) continue;
      bool eligible = true;
      const auto* members = group_members(lifetimes[idx].variable);
      const std::vector<VarPtr> singleton =
          members ? std::vector<VarPtr>{} : std::vector<VarPtr>{lifetimes[idx].variable};
      for (const VarPtr& member : (members ? *members : singleton)) {
        const auto member_memref = GetTypeMemRef(member->GetType());
        if (!member_memref.has_value() || !*member_memref ||
            (*member_memref)->base_.get() != (*root_memref)->base_.get()) {
          eligible = false;
          break;
        }
        const auto member_offset = As<ConstInt>((*member_memref)->byte_offset_);
        // A remaining dynamic member is a direct tile.slice: it does not expose
        // holes to the packer, and PTO rebuilds its address from the translated
        // source root plus the slice operands.
        if (!member_offset) continue;
        const __int128 relative = static_cast<__int128>(member_offset->value_) - root_offset->value_;
        const __int128 end = relative + (*member_memref)->size_;
        if (relative < 0 || end > lifetimes[idx].size) {
          eligible = false;
          break;
        }
      }
      whole_root_subrange_eligible[idx] = eligible;
    }

    // First-fit-decreasing pack with the current per-group residue counts
    // (pipeline_blocks reads the mutable F_g). The ordinary mode reproduces the
    // legacy whole-buffer FFD exactly. Subrange mode instead treats members that
    // cannot share bytes with the candidate as address blockers and picks the
    // lowest aligned gap. Members stay sorted by offset, so one candidate scans
    // each arena once, preserving the existing quadratic root-pair search.
    auto pack = [&](bool allow_subranges) {
      placing_subranges = allow_subranges;
      std::vector<PackedBuffer> buffers;
      for (size_t idx : indices) {
        if (is_pinned[idx]) {
          buffers.push_back(
              PackedBuffer{{PackedMember{idx, 0}}, lifetimes[idx].size, true, buffer_roots[idx]});
          continue;
        }
        const auto& cand = lifetimes[idx];
        bool placed = false;
        for (PackedBuffer& buffer : buffers) {
          if (buffer.pinned || cand.size > buffer.capacity) continue;
          // A disjoint subrange is still the same symbolic byte root. Dynamic
          // metadata requires a compatible full-capacity typed allocation even
          // when the original lifetimes do not overlap.
          if (!BufferRootCompatibility::Compatible(buffer_roots[idx], buffer.buffer_root)) continue;

          std::optional<uint64_t> chosen_offset;
          if (!allow_subranges || !whole_root_subrange_eligible[idx] ||
              !whole_root_subrange_eligible[buffer.members.front().interval_index]) {
            bool fits_at_zero = true;
            for (const PackedMember& member : buffer.members) {
              if (!can_share(cand, lifetimes[member.interval_index])) {
                fits_at_zero = false;
                break;
              }
            }
            if (fits_at_zero) chosen_offset = 0;
          } else {
            uint64_t cursor = 0;
            for (const PackedMember& member : buffer.members) {
              if (can_share(cand, lifetimes[member.interval_index])) continue;
              const uint64_t start = member.byte_offset;
              INTERNAL_CHECK_SPAN(
                  lifetimes[member.interval_index].size <= std::numeric_limits<uint64_t>::max() - start,
                  lifetimes[member.interval_index].variable->span_)
                  << "Internal error: reuse placement range overflows uint64";
              const uint64_t end = start + lifetimes[member.interval_index].size;
              const uint64_t aligned = alloc_policy->AlignAddress(cursor, space);
              if (aligned <= start && cand.size <= start - aligned) {
                chosen_offset = aligned;
                break;
              }
              cursor = std::max(cursor, end);
            }
            if (!chosen_offset.has_value()) {
              const uint64_t aligned = alloc_policy->AlignAddress(cursor, space);
              if (aligned <= buffer.capacity && cand.size <= buffer.capacity - aligned) {
                chosen_offset = aligned;
              }
            }
          }

          if (!chosen_offset.has_value()) continue;
          const PackedMember placement{idx, *chosen_offset};
          auto pos = std::upper_bound(buffer.members.begin(), buffer.members.end(), placement,
                                      [](const PackedMember& lhs, const PackedMember& rhs) {
                                        return lhs.byte_offset < rhs.byte_offset;
                                      });
          buffer.members.insert(pos, placement);
          buffer.buffer_root = BufferRootCompatibility::Merge(buffer.buffer_root, buffer_roots[idx]);
          placed = true;
          break;
        }
        if (!placed) {
          buffers.push_back(PackedBuffer{{PackedMember{idx, 0}}, cand.size, false, buffer_roots[idx]});
        }
      }
      return buffers;
    };

    std::vector<PackedBuffer> buffers = pack(/*allow_subranges=*/false);

    // Capacity relief uses the exact allocator footprint (SpaceFootprint, Primitive A). First retry with
    // subranges; while that still overflows, lower the largest-slot pipeline group's depth by one residue
    // and re-pack. Re-running the FFD (not merging in place) avoids in-place non-monotonicity, but the FFD
    // is still not monotone
    // under the *relaxed* (F_g==1) predicate, so exhausting the shed does NOT guarantee a fit: the loop
    // ends in a from-scratch legacy re-pack + diagnostic (see the `!shed_group` branch below), which
    // guarantees no fit regression vs legacy. The shed objective is MaxRelief (largest slot first, tie
    // by lowest group id), applied unconditionally.
    if (alloc_policy) {
      const uint64_t cap = pack_be->GetMemSize(space);
      auto rit = reserved_end_by_space.find(space);
      const uint64_t reserved_start = rit != reserved_end_by_space.end() ? rit->second : 0;
      auto footprint = [&](const std::vector<PackedBuffer>& bufs) {
        SpaceFootprint fp(space, *alloc_policy, reserved_start);
        for (const auto& buf : bufs) {
          (void)fp.OpenBuffer(buf.capacity);
        }
        return fp.HighWater();
      };
      auto retry_with_subranges = [&]() {
        if (!subrange_space_safe || cap == 0 || footprint(buffers) <= cap) return;
        auto subdivided = pack(/*allow_subranges=*/true);
        if (footprint(subdivided) < footprint(buffers)) buffers = std::move(subdivided);
      };
      // Preserve the legacy layout whenever it fits. Only an actual overflow
      // enables subdivision, before sacrificing software-pipeline depth.
      retry_with_subranges();
      // Bound the shed re-packs so the loop stays within the pass-complexity budget
      // (.claude/rules/pass-complexity.md). A legitimate shed converges in Σ_g(F_g−1) steps over the few
      // co-live groups, but the pipeline-group count can grow with generated IR, so cap the full re-packs
      // at a constant. The cap only trips in a pathological many-group overflow, where the legacy fallback
      // is the correct safe outcome anyway. A fixed retry budget preserves the complexity
      // of one FFD pack up to a constant factor.
      constexpr int kMaxShedRepacks = 256;
      int shed_repacks = 0;
      while (cap != 0 && footprint(buffers) > cap) {
        // Pick the group to shed by MaxRelief; ties break by lowest group id (deterministic). Once the
        // re-pack budget is spent, stop picking (shed_group stays empty) and take the legacy fallback.
        const bool within_budget = (++shed_repacks <= kMaxShedRepacks);
        std::optional<int32_t> shed_group;
        double best_score = 0.0;
        if (within_budget) {
          for (const auto& [key, f] : group_depth) {
            if (key.first != space || f <= 1) continue;
            const uint64_t slot = group_slot.count(key) ? group_slot.at(key) : 0;
            const double s =
                ScoreMaxRelief(ShedCandidate{key.second, alloc_policy->AlignAddress(slot, space)});
            if (!shed_group || s < best_score || (s == best_score && key.second < *shed_group)) {
              best_score = s;
              shed_group = key.second;
            }
          }
        }
        if (!shed_group) {
          // No group left to shed — either every group is at depth 1 (shed exhausted) or the re-pack
          // budget was spent (pathological many-group overflow). FFD is NOT monotone under the relaxed
          // (F_g==1) predicate, so this packing may be worse than legacy; re-pack in legacy mode
          // (`force_legacy`, byte-identical to the legacy predicate) — no fit regression vs legacy, and if
          // legacy also overflows it is a genuine overflow legacy would hit too (AllocateMemoryAddr surfaces
          // it).
          force_legacy = true;
          buffers = pack(/*allow_subranges=*/false);
          retry_with_subranges();
          force_legacy = false;
          force_legacy_spaces.insert(space);
          // Fold into the diagnostic channel (Warning) so all capacity-degradation signals go through one
          // place (perf hint for a partial reduction below; this Warning for a space that fits at no depth).
          if (out_hints != nullptr) {
            const bool still_overflows = footprint(buffers) > cap;
            // A single physical buffer larger than the entire space is not a
            // reuse or pipeline-depth failure.  Legacy packing cannot repair
            // it, and the final allocator (or an earlier operation-specific
            // check such as AutoTileMatmulL0's explicit-L0 operand diagnostic)
            // reports the impossible allocation.  Do not obscure that error
            // with a fallback warning that suggests reuse or stage count is
            // responsible.  Keep warning for aggregate pressure: every buffer
            // fits individually, but their required co-residency does not.
            const bool has_intrinsically_oversized_buffer = std::any_of(
                buffers.begin(), buffers.end(), [&](const PackedBuffer& buf) { return buf.capacity > cap; });
            const std::string why =
                within_budget ? "at any double-buffering depth" : "within the shed-repack budget";
            if (!has_intrinsically_oversized_buffer) {
              out_hints->emplace_back(
                  DiagnosticSeverity::Warning, "MemoryReuse", 0,
                  "capacity-gated reuse could not fit memory space " + MemorySpaceToString(space) + " " +
                      why + "; fell back to the legacy packing" +
                      (still_overflows ? " (which also overflows — reduce tile size or stage count)" : "") +
                      ".",
                  func ? func->span_ : Span::unknown());
            }
          }
          break;
        }
        group_depth[std::make_pair(space, *shed_group)] -= 1;
        buffers = pack(/*allow_subranges=*/false);
        retry_with_subranges();
      }
    }

    // Commit the final packing. Every member maps directly to the arena owner;
    // byte_offset is zero for legacy whole-buffer reuse and may be non-zero only
    // for the overflow-triggered subrange fallback.
    for (const auto& buf : buffers) {
      const size_t representative_index = buf.members.front().interval_index;
      const VarPtr& representative = lifetimes[representative_index].variable;
      for (const PackedMember& member : buf.members) {
        if (member.interval_index == representative_index) continue;
        const auto& cand = lifetimes[member.interval_index];
        reuse_map[cand.variable] = ReusePlacement{representative, member.byte_offset};
        if (const Var* cb = TileMemRefBase(cand.variable)) {
          if (const Var* rb = physical_base(representative)) {
            if (cb != rb) base_remap[cb] = rb;
          }
        }
      }
    }
  }

  // Loud diagnostic (perf hint): a pipeline group whose achieved depth F_g fell below the requested D_g
  // means the capacity gate could not honor the programmer's `pl.pipeline(stage=D)` — stages k and k+F_g
  // share a buffer and re-serialize (the false WAR the pipeline meant to avoid). Surface it (correct, just
  // slower) with the concrete fix, rather than silently degrading. Spaces that hit the legacy fallback
  // already emitted a Warning above, so skip them here to avoid a double signal.
  if (out_hints != nullptr) {
    for (const auto& [key, achieved] : group_depth) {
      if (capacity_known_spaces.count(key.first) == 0) continue;  // unknown capacity ⇒ not gated
      if (force_legacy_spaces.count(key.first) != 0) continue;    // already warned at the space level
      auto rit = group_requested_depth.find(key);
      const int32_t requested = rit != group_requested_depth.end() ? rit->second : achieved;
      if (achieved >= requested) continue;  // fully double-buffered as requested — nothing to report
      const uint64_t slot = group_slot.count(key) != 0 ? group_slot.at(key) : 0;
      const uint64_t cap = pack_be != nullptr ? pack_be->GetMemSize(key.first) : 0;
      // Effective (free) capacity is the total minus the reserved region — the same reserved_start the exact
      // SpaceFootprint fit begins at. Co-resident non-pipeline tiles and other pipeline groups also consume
      // the space, but they are not a single subtractable constant, so the byte threshold is only exact when
      // this operand's own footprint is the binding constraint. Distinguish the two shed causes so the fix
      // stays honest: (a) slot-bound — `slot*requested` overflows even an otherwise-empty free region, so
      // shrink to `free_cap/requested`; (b) space-pressure — it would fit alone, so the fix is to relieve the
      // co-residents, not shrink this tile (which already satisfies the per-slot bound).
      const uint64_t reserved =
          reserved_end_by_space.count(key.first) != 0 ? reserved_end_by_space.at(key.first) : 0;
      const uint64_t free_cap = cap > reserved ? cap - reserved : 0;
      // Use the *aligned* slot — the same per-buffer increment SpaceFootprint bumps by — so the slot-bound
      // vs space-pressure classification and the byte threshold match the real fit (raw `slot` under-counts
      // when the tile isn't an alignment multiple).
      const uint64_t aligned_slot =
          alloc_policy != nullptr ? alloc_policy->AlignAddress(slot, key.first) : slot;
      const uint64_t need = aligned_slot * static_cast<uint64_t>(requested);
      const bool slot_bound = free_cap == 0 || need > free_cap;
      // Source-agnostic wording: `pipeline_membership` is stamped both by an explicit `pl.pipeline(stage=)`
      // and by compiler-synthesized pipelines (e.g. #1900's cross-core skew clones), so blame "software
      // pipelining", not the user's `stage=`, and offer `pl.pipeline(stage=)` only as an example.
      std::ostringstream msg;
      msg << "software pipelining requested depth " << requested << " for pipeline group " << key.second
          << " in " << MemorySpaceToString(key.first) << ", but only " << achieved << " of " << requested
          << " buffers fit (" << aligned_slot << " B per stage, " << free_cap << " B free";
      if (reserved != 0) msg << " after " << reserved << " B reserved";
      msg << ") — stages " << achieved << " apart share storage and serialize. ";
      if (slot_bound) {
        msg << "This operand alone needs " << need << " B for depth " << requested
            << "; shrink the per-stage tile to <= "
            << (requested > 0 ? free_cap / static_cast<uint64_t>(requested) : free_cap)
            << " B, or reduce the pipeline depth (e.g. `pl.pipeline(stage=)`) to " << achieved << ".";
      } else {
        msg << "The operand would fit depth " << requested
            << " on its own, but co-resident buffers / other pipeline groups over-subscribe the space; "
            << "relieve the co-residents (smaller or fewer co-live tiles) or reduce the pipeline depth to "
            << achieved << ".";
      }
      out_hints->emplace_back(DiagnosticSeverity::PerfHint, "MemoryReuse", 0, "PH-MR-001", msg.str(),
                              func ? func->span_ : Span::unknown());
    }
  }

  return reuse_map;
}

/**
 * @brief Apply MemRef sharing to the statement tree
 */
StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const ReuseMap& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups) {
  // Custom IRMutator for MemRef sharing
  class MemRefSharingMutator : public IRMutator {
   public:
    explicit MemRefSharingMutator(const ReuseMap& reuse_map,
                                  const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups)
        : reuse_map_(reuse_map), sharing_groups_(sharing_groups) {}

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto variable = RewriteDefinition(op->var_);
      auto value = VisitExpr(op->value_);
      if (variable == op->var_ && value == op->value_) return op;
      return std::make_shared<AssignStmt>(variable, value, op->span_);
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override {
      auto result = As<IfStmt>(IRMutator::VisitStmt_(op));
      auto returns = result->return_vars_;
      bool changed = false;
      for (size_t i = 0; i < returns.size(); ++i) {
        auto replacement = RewriteDefinition(op->return_vars_[i]);
        changed |= replacement != returns[i];
        returns[i] = std::move(replacement);
      }
      if (!changed) return result;
      auto updated = MutableCopy(result);
      updated->return_vars_ = std::move(returns);
      return updated;
    }

    VarPtr RewriteDefinition(const VarPtr& var) {
      // Check if this variable should reuse another's MemRef
      if (reuse_map_.count(var)) {
        const ReusePlacement& placement = reuse_map_.at(var);
        INTERNAL_CHECK_SPAN(
            placement.byte_offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), var->span_)
            << "Internal error: reuse placement offset does not fit an INDEX ConstInt";
        VarPtr source_var = placement.representative;

        // Get source's TileType and MemRef
        auto source_tile_type = As<TileType>(source_var->GetType());

        if (!source_tile_type || !source_tile_type->memref_.has_value()) {
          LOG_ERROR << "Source variable " << source_var->name_hint_ << " does not have MemRef";
          return var;
        }

        std::optional<MemRefPtr> source_memref = source_tile_type->memref_;

        // Get current variable's TileType
        auto curr_tile_type = As<TileType>(var->GetType());

        if (!curr_tile_type) {
          LOG_ERROR << "Current variable " << var->name_hint_ << " is not TileType";
          return var;
        }

        // Rebase a sharing-group member onto the reuse target, keeping its
        // offset relative to the representative and its own size — substituting
        // the target MemRef wholesale would collapse every member onto the
        // target's base offset (issue #1723: all per-head row slices read row 0).
        // Members coinciding with the representative reuse the target MemRef
        // object so plain reuse keeps MemRef identity.
        //
        // Symbolic offsets retain the exact relative-address expression:
        // target_offset + (member_offset - representative_offset). Collapsing a
        // dynamic member onto the target MemRef would silently select the
        // representative's window.
        const MemRefPtr curr_memref = curr_tile_type->memref_.value_or(nullptr);
        auto rebase_memref = [&](const TileTypePtr& tile) -> std::optional<MemRefPtr> {
          if (!tile->memref_.has_value() || !curr_memref) return source_memref;
          const auto& old = tile->memref_.value();
          auto old_off = As<ConstInt>(old->byte_offset_);
          auto curr_off = As<ConstInt>(curr_memref->byte_offset_);
          if (!old_off || !curr_off) {
            ExprPtr relative;
            if (structural_equal(old->byte_offset_, curr_memref->byte_offset_)) {
              relative = MakeZeroByteOffset();
            } else if (curr_off && curr_off->value_ == 0) {
              relative = old->byte_offset_;
            } else {
              relative = MakeSub(old->byte_offset_, curr_memref->byte_offset_, old->span_);
            }
            auto placement_expr = std::make_shared<ConstInt>(static_cast<int64_t>(placement.byte_offset),
                                                             DataType::INDEX, Span::unknown());
            auto target_offset = AddByteOffsets((*source_memref)->byte_offset_, placement_expr);
            auto rebased_offset = AddByteOffsets(target_offset, relative);
            return std::make_shared<MemRef>((*source_memref)->base_, rebased_offset, old->size_, old->span_,
                                            (*source_memref)->is_pinned_, (*source_memref)->slot_count_,
                                            (*source_memref)->slot_index_);
          }
          const __int128 wide_rel = static_cast<__int128>(old_off->value_) - curr_off->value_;
          const __int128 wide_placement = placement.byte_offset;
          INTERNAL_CHECK_SPAN(wide_rel >= 0 && wide_rel <= std::numeric_limits<int64_t>::max() &&
                                  wide_placement + wide_rel + old->size_ <= (*source_memref)->size_,
                              old->span_)
              << "Internal error: sharing-group member offset " << old_off->value_
              << " cannot be represented inside reuse target (size " << (*source_memref)->size_ << ")";
          const int64_t placed_rel = static_cast<int64_t>(wide_placement + wide_rel);
          if (placed_rel == 0 && old->size_ == (*source_memref)->size_) return source_memref;
          auto rel_expr = std::make_shared<ConstInt>(placed_rel, DataType::INDEX, Span::unknown());
          return std::make_shared<MemRef>((*source_memref)->base_,
                                          AddByteOffsets((*source_memref)->byte_offset_, rel_expr),
                                          old->size_, old->span_, (*source_memref)->is_pinned_,
                                          (*source_memref)->slot_count_, (*source_memref)->slot_index_);
        };

        // Rebase the representative through the same path as every member of
        // its semantic sharing group.  A smaller tile may reuse a larger
        // allocation; substituting source_memref wholesale here would give the
        // representative the larger window while its aliases retain their
        // original window, splitting one loop-carried accumulator family.
        auto new_tile_type = std::dynamic_pointer_cast<const TileType>(
            CloneTypeWithMemRefAndRemapExprs(curr_tile_type, rebase_memref(curr_tile_type),
                                             [this](const ExprPtr& expr) { return VisitExpr(expr); }));

        // Create new Var
        auto new_var = std::make_shared<const Var>(var->name_hint_, new_tile_type, var->span_);

        // Record the variable substitution mapping (old -> new)
        // This ensures that all subsequent references to the old variable will be replaced with the new one
        var_substitution_map_[var] = new_var;

        // CRITICAL: If this variable shares MemRef with others (view operations),
        // we need to update ALL of them to use the new MemRef
        if (sharing_groups_.count(var)) {
          const auto& sharing_group = sharing_groups_.at(var);
          for (const auto& shared_var : sharing_group) {
            if (shared_var != var) {
              // Create new Var for shared variable with same reused MemRef
              auto shared_tile_type = As<TileType>(shared_var->GetType());
              if (shared_tile_type) {
                auto new_shared_tile_type =
                    std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
                        shared_tile_type, rebase_memref(shared_tile_type),
                        [this](const ExprPtr& expr) { return VisitExpr(expr); }));
                auto new_shared_var = std::make_shared<const Var>(shared_var->name_hint_,
                                                                  new_shared_tile_type, shared_var->span_);
                var_substitution_map_[shared_var] = new_shared_var;

                LOG_DEBUG << "Propagating reuse to sharing group member: " << shared_var->name_hint_;
              }
            }
          }
        }

        return new_var;
      }
      const auto found = var_substitution_map_.find(var);
      return found == var_substitution_map_.end() ? var : found->second;
    }

    // Override VisitExpr_ to replace variable references with their new versions
    ExprPtr VisitExpr_(const VarPtr& op) override {
      // Check if this variable has been replaced (i.e., it's the old version of a reused variable)
      if (var_substitution_map_.count(op)) {
        // Return the new version of the variable
        return var_substitution_map_.at(op);
      }
      // Otherwise, keep the variable as-is
      return op;
    }

   private:
    const ReuseMap& reuse_map_;
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups_;  // var -> sharing group
    // Maps old variable objects to new variable objects (with reused MemRef)
    // This is needed because IR nodes are immutable, so we create new Var objects
    // and need to replace all references to the old ones
    std::map<VarPtr, VarPtr> var_substitution_map_;
  };

  MemRefSharingMutator mutator(reuse_map, var_sharing_groups);
  return mutator.VisitStmt(stmt);
}

// A Vec view retains its parent's physical pitch. TMOV requires equal physical
// shapes after subview resolution, whereas TEXTRACT explicitly repacks the
// requested window into the already planned destination. Index the final SSA
// body so retargeting and carry repair cannot leave stale view provenance.
class StorageTransferLegalizer : public IRMutator {
 public:
  StorageTransferLegalizer(const StmtPtr& body, const std::vector<VarPtr>& parameters)
      : views_(body), geometry_(views_, names_) {
    names_.Collect(body, parameters);
  }

 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override { return argument; }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    const auto call = As<Call>(assign->value_);
    if (call && IsOp(call, "tile.slice")) return geometry_.Index(assign, call);
    if (!call || !IsOp(call, "tile.move") || call->args_.size() != 1) return assign;
    const auto source = GetTileTypeWithMemRef(call->args_[0]->GetType());
    const auto target = GetTileTypeWithMemRef(assign->var_->GetType());
    const auto view = views_.Get(call->args_[0]);
    const auto root = view.root ? GetTileTypeWithMemRef(view.root->GetType()) : nullptr;
    if (!source || !target || source->GetMemorySpace() != MemorySpace::Vec ||
        target->GetMemorySpace() != MemorySpace::Vec || !view.window || !root || source->shape_.size() != 2 ||
        root->shape_.size() != 2 || SamePhysicalWindow(GetDefinedMemRef(source), GetDefinedMemRef(target))) {
      return assign;
    }
    // A post-placement repair cannot introduce a wider read that aliases the
    // destination. Preplanning transfers expose this root to no-alias analysis.
    if (MemRef::MayAlias(GetDefinedMemRef(root), GetDefinedMemRef(target))) return assign;
    const auto offsets = geometry_.RootOffsets(view, call->span_);
    auto& registry = OpRegistry::GetInstance();
    auto extract = MutableCopy(registry.Create(
        "tile.extract",
        {view.root, offsets[0], offsets[1], std::make_shared<MakeTuple>(source->shape_, call->span_)},
        {{"target_memory", MemorySpace::Vec}}, call->span_));
    // Keep scheduling metadata; argument directions depend on the new arity.
    for (const auto& attribute : call->attrs_) {
      if (attribute.first != kAttrArgDirections) extract->attrs_.push_back(attribute);
    }
    const auto expected = CloneTypeWithoutMemRef(target);
    if (structural_equal(CloneTypeWithoutMemRef(extract->GetType()), expected)) {
      return std::make_shared<AssignStmt>(assign->var_, extract, assign->span_);
    }

    // A slice/set_validshape may narrow validity beyond the root's window.
    // Preserve that logical requirement without changing the extraction pitch
    // or inventing a differently laid-out transfer.
    const auto extracted = std::make_shared<Var>(
        names_.Fresh("storage_window_"), CloneTypeWithMemRef(extract->GetType(), GetDefinedMemRef(target)),
        assign->span_);
    const auto valid = tile_view_semantics::GetEffectiveTileView(*target).valid_shape;
    const auto refine =
        registry.Create("tile.set_validshape", {extracted, valid[0], valid[1]}, {}, assign->span_);
    if (!structural_equal(CloneTypeWithoutMemRef(refine->GetType()), expected)) return assign;
    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(extracted, extract, assign->span_),
                             std::make_shared<AssignStmt>(assign->var_, refine, assign->span_)},
        assign->span_);
  }

 private:
  BufferViewAnalysis views_;
  StorageNameSupply names_;
  BufferWindowGeometry geometry_;
};

StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body);

/**
 * @brief Fix yield/return_var MemRef mismatch after MemoryReuse.
 *
 * Handles ForStmt, WhileStmt and IfStmt:
 * - Loops: MemoryReuse may assign different MemRefs to iter_arg and yield value.
 *   Since yield value becomes the next iteration's iter_arg, they must use the
 *   same buffer. When they differ, insert a tile.move before yield, except that
 *   divergent Acc buffers are rejected because Acc->Acc moves are unsupported.
 * - IfStmt: each result already has its canonical destination. Reconcile both
 *   incoming edges with that window; reject an Acc-to-Acc copy only if a
 *   producer could not be retargeted and an actual transfer remains necessary.
 *   Scope::IfResults traverses loops while leaving their carry scheduling to
 *   the later planner-specific reconciliation stage.
 */
class YieldFixupMutator : public StorageBoundaryMutator {
 public:
  enum class Scope { AllControlFlow, IfResults };

  YieldFixupMutator() = default;

  StmtPtr Run(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    StmtPtr aligned = body;
    if (scope_ == Scope::AllControlFlow) {
      aligned = RetypeApplier({}).VisitStmt(aligned);
    }
    // Carry alignment can move a result; propagate its required aliases before
    // reconciling transfers while keeping unresolved if results provisional.
    // An if-only run must leave loop storage to its later legalization stage.
    aligned = NormalizeIdentityCopyBuffersBeforeYieldFixup(aligned);
    names_.Collect(aligned, parameters);
    const auto rewritten = VisitStmt(aligned);
    return StorageTransferLegalizer(rewritten, parameters).VisitStmt(rewritten);
  }

  /// Region targets are already planned. This stage only realizes edge
  /// transfers; legacy PYPTO retains its established loop-copy scheduler.
  explicit YieldFixupMutator(bool canonical_loop_copies, Scope scope = Scope::AllControlFlow)
      : canonical_loop_copies_(canonical_loop_copies), scope_(scope) {}

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(result);
    if (scope_ == Scope::IfResults) return result;
    if (!for_stmt || for_stmt->iter_args_.empty()) return result;
    if (canonical_loop_copies_) return FixupCanonicalLoop(for_stmt);

    auto yield_stmt = FindYieldStmt(for_stmt->body_);
    if (!yield_stmt) return result;

    // Check each (iter_arg, yield_value) pair for MemRef mismatch.
    // Use initValue's MemRef as the target because codegen maps iter_arg to initValue's buffer,
    // and MemoryReuse may have updated initValue's MemRef without updating iter_arg's.
    // One index over the carry buffers serves both the overlap rejection below
    // and the conflict graph in OrderCarryCopies.
    const std::vector<MemRefPtr> carry_ranges = CollectCarryRanges(for_stmt);
    const CarryRangeIndex carry_index(carry_ranges);
    RejectOverlappingCarryBuffers(for_stmt, carry_index);

    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < yield_stmt->value_.size() && i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike, not As<Var>: a nested loop both yields and seeds carries with
      // the enclosing loop's IterArg, whose own ObjectKind As<Var> does not match.
      // Skipping those left the nested carry with no writeback at all.
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;

      // Yielding this position's own iter_arg leaves the carry untouched, so
      // there is nothing to write back. Short-circuit before comparing buffers:
      // it is the one case where the two sides are the same storage by
      // construction, whatever their (possibly still stale) types say.
      if (yield_var.get() == static_cast<const Var*>(for_stmt->iter_args_[i].get())) continue;

      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);

      // Get the initValue's MemRef (the actual buffer used at runtime)
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Compare addresses, not allocations: two slots of one pl.MemRef(slots=N)
      // share a base, so "same allocation" would skip the copy that moves the
      // value from one slot to the other and silently carry the wrong slot.
      const AddressRelation relation = CompareBaseAddress(yield_memref, init_memref);
      if (relation == AddressRelation::kSame) continue;
      // Two symbolic offsets into one allocation may or may not name the same
      // bytes. Copying is wrong if they coincide (a tile.move onto itself, which
      // Acc cannot lower at all) and skipping is wrong if they do not, so say so
      // rather than pick one and hope.
      CHECK_SPAN(relation == AddressRelation::kDifferent, for_stmt->span_)
          << "Loop-carried value '" << for_stmt->iter_args_[i]->name_hint_
          << "' is yielded at a slot of its own allocation that cannot be shown equal to, or "
             "distinct from, the slot it is carried in, so whether the carry needs a copy is "
             "undecidable here. Subscript the declared allocation with the same expression at both "
             "ends, or give the carry its own allocation.";

      // MemRef mismatch — create tile.move to copy yield value into initValue's buffer
      auto target_memory = init_tile->GetMemorySpace();
      auto [moved_var, move_stmt] = CreateTileMove(yield_var, init_memref, target_memory);

      copies.push_back(CarryCopy{i, yield_var, std::move(moved_var), std::move(move_stmt), yield_memref,
                                 init_memref, target_memory});
    }

    if (copies.empty()) {
      // Even without tile.move insertion, iter_arg and return_var may have stale MemRefs
      // (MemoryReuse updates initValue/yield but not iter_arg/return_var types).
      // Ensure all 4 loop-carry variables share the same MemRef (= initValue's).
      return PatchIterArgsAndReturnVars(for_stmt, yield_stmt);
    }

    // `pl.yield_` rebinds every carry simultaneously, so the copies that realize
    // it must not observe each other's writes. Order them first.
    std::vector<StmtPtr> spill_stmts;
    const std::vector<size_t> order = OrderCarryCopies(&copies, carry_index, &spill_stmts);

    // Build new yield values with moved vars substituted
    std::vector<ExprPtr> new_yield_values = yield_stmt->value_;
    for (const auto& copy : copies) {
      new_yield_values[copy.index] = copy.moved_var;
    }
    auto new_yield = MutableCopy(yield_stmt);
    new_yield->value_ = std::move(new_yield_values);

    // A spill reads a carry buffer before any copy writes one, so every spill
    // precedes every copy.
    std::vector<StmtPtr> move_stmts = std::move(spill_stmts);
    move_stmts.reserve(move_stmts.size() + order.size());
    for (size_t idx : order) move_stmts.push_back(copies[idx].move_stmt);

    // Insert tile.move stmts before yield and replace yield in body
    auto new_body = InsertMovesAndReplaceYield(for_stmt->body_, new_yield, move_stmts);

    // Build intermediate ForStmt with new body, then patch iter_args/return_vars
    auto intermediate_for = MutableCopy(for_stmt);
    intermediate_for->body_ = new_body;

    return PatchIterArgsAndReturnVars(intermediate_for, new_yield);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    if (scope_ == Scope::IfResults) return result;
    return FixupCanonicalLoop(As<WhileStmt>(result));
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = As<IfStmt>(IRMutator::VisitStmt_(op));
    return result && !result->return_vars_.empty() ? FixupCanonicalIf(result) : result;
  }

 public:
  /// Allocations this fixup had to create (cycle spills), for the caller to
  /// hoist onto the function-body head. Empty for every loop whose carries do
  /// not form a copy cycle, which is the overwhelmingly common case.
  std::vector<StmtPtr> TakePendingAllocs() { return std::move(pending_allocs_); }

 private:
  bool canonical_loop_copies_ = false;
  Scope scope_ = Scope::AllControlFlow;
  StorageNameSupply names_;

  enum class BoundaryKind { IfResult, LoopCarry };

  // Each phi's target is established before planning. Never
  // choose a branch input as a new target here: it may remain independently live
  // after the IfStmt. Both arms must explicitly write the declared destination.
  StmtPtr FixupCanonicalIf(const IfStmtPtr& branch) {
    auto then_body = FixupCanonicalBoundary(branch->then_body_, branch->return_vars_, branch->span_,
                                            BoundaryKind::IfResult);
    auto else_body = branch->else_body_;
    if (else_body) {
      else_body =
          FixupCanonicalBoundary(*else_body, branch->return_vars_, branch->span_, BoundaryKind::IfResult);
    }
    if (then_body == branch->then_body_ && else_body == branch->else_body_) return branch;
    auto result = MutableCopy(branch);
    result->then_body_ = std::move(then_body);
    result->else_body_ = std::move(else_body);
    return result;
  }

  template <typename LoopPtr>
  StmtPtr FixupCanonicalLoop(const LoopPtr& loop) {
    std::vector<VarPtr> targets(loop->iter_args_.begin(), loop->iter_args_.end());
    auto body = FixupCanonicalBoundary(loop->body_, targets, loop->span_, BoundaryKind::LoopCarry);
    if (body == loop->body_) return loop;
    auto result = MutableCopy(loop);
    result->body_ = std::move(body);
    return result;
  }

  StmtPtr FixupCanonicalBoundary(const StmtPtr& body, const std::vector<VarPtr>& targets, const Span& span,
                                 BoundaryKind kind) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    std::vector<ExprPtr> values = yield ? yield->value_ : std::vector<ExprPtr>{};
    std::vector<MemRefPtr> destinations(targets.size());
    std::vector<VarPtr> sources(targets.size());
    std::vector<size_t> owners(targets.size());
    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (const auto tile = GetTileTypeWithMemRef(targets[i]->GetType())) {
        destinations[i] = GetDefinedMemRef(tile);
      }
    }
    const CarryRangeIndex index(destinations);
    for (size_t i = 0; i < targets.size(); ++i) {
      owners[i] = i;
      auto target_tile = GetTileTypeWithMemRef(targets[i]->GetType());
      if (!target_tile) continue;
      CHECK_SPAN(yield && i < values.size(), span)
          << "Buffer IR storage legalization requires every region to yield every tile result";
      auto source = AsVarLike(values[i]);
      auto source_tile = source ? GetTileTypeWithMemRef(source->GetType()) : nullptr;
      sources[i] = source;
      CHECK_SPAN(source && As<TileType>(source->GetType()), span)
          << "Buffer IR storage legalization requires tile yields to be storage variables";
      if (!source_tile) {
        // Legacy tile handles can be supplied without a physical MemRef (for
        // example cross-core inputs). Their edge still targets the declared r;
        // codegen realizes that transfer once it resolves the opaque handle.
        // Selected Buffer IR requires explicit source storage and cannot defer.
        const auto* context = PassContext::Current();
        CHECK_SPAN(context == nullptr || !context->GetEnableBufferIR(), span)
            << "Buffer IR storage legalization requires known storage for tile yield inputs";
        continue;
      }
      sources[i] = source;
      auto target = GetDefinedMemRef(target_tile);
      destinations[i] = target;
      auto source_memref = GetDefinedMemRef(source_tile);
      if (SamePhysicalWindow(source_memref, target)) continue;
      CHECK_SPAN(CompareBaseAddress(source_memref, target) == AddressRelation::kDifferent, span)
          << "Buffer IR storage legalization cannot reconcile ambiguous views of one destination";
      const auto source_memory = source_tile->GetMemorySpace();
      const auto target_memory = target_tile->GetMemorySpace();
      const bool supported =
          source_memory && target_memory && IsTileMoveEverSupported(*source_memory, *target_memory);
      CHECK_SPAN(kind != BoundaryKind::LoopCarry || supported || !index.AnyOverlap(source_memref), span)
          << "Buffer IR cannot reconcile region storage with an unsupported tile.move memory-space pair";
      // Constructing an actual copy preserves the accumulator invariant's
      // existing InternalError; fresh Acc producers were allowed to retarget
      // earlier without requiring a transfer at all.
      auto [moved, move] = CreateTileMove(source, target, target_memory);
      CHECK_SPAN(supported, span)
          << "Buffer IR cannot reconcile region storage with an unsupported tile.move memory-space pair";
      copies.push_back({i, source, moved, move, source_memref, target, target_tile->GetMemorySpace()});
    }
    if (copies.empty()) return body;
    std::vector<bool> written_slots(targets.size(), false);
    for (const auto& copy : copies) written_slots[copy.index] = true;
    // Multiple SSA results may return the same value through one window. Merge
    // only equivalent transfers, including their descriptors; equal destinations
    // alone would silently overwrite distinct results. Mutable loop carries
    // retain their independent-slot contract.
    const auto overlaps =
        kind == BoundaryKind::IfResult ? index.OverlappingPairs(written_slots) : index.OverlappingPairs();
    for (const auto& [lhs, rhs] : overlaps) {
      auto lhs_source = GetTileTypeWithMemRef(sources[lhs]->GetType());
      auto rhs_source = GetTileTypeWithMemRef(sources[rhs]->GetType());
      CHECK_SPAN(lhs_source && rhs_source && kind == BoundaryKind::IfResult &&
                     SamePhysicalWindow(destinations[lhs], destinations[rhs]) &&
                     SamePhysicalWindow(GetDefinedMemRef(lhs_source), GetDefinedMemRef(rhs_source)) &&
                     SameTransferDescriptor(lhs_source, rhs_source) &&
                     SameTransferDescriptor(GetTileTypeWithMemRef(targets[lhs]->GetType()),
                                            GetTileTypeWithMemRef(targets[rhs]->GetType())),
                 span)
          << "Buffer IR region transfers require independent destination windows";
      UnionSets(owners, lhs, rhs);
    }
    for (size_t i = 0; i < owners.size(); ++i) owners[i] = FindSet(owners, i);
    copies.erase(
        std::remove_if(copies.begin(), copies.end(),
                       [&owners](const CarryCopy& copy) { return owners[copy.index] != copy.index; }),
        copies.end());
    std::vector<MemRefPtr> written_destinations(targets.size());
    for (const auto& copy : copies) written_destinations[copy.index] = copy.dst_memref;
    const CarryRangeIndex writes(written_destinations);
    // Snapshot each source that another transfer may overwrite. This deliberately
    // uses overlap existence queries, not an O(k^2) graph for broad/fanout views.
    // Every snapshot precedes every destination write: swaps, cycles, partial
    // overlap and fanout therefore share one O(k log k) algorithm.
    std::vector<StmtPtr> moves;
    for (auto& copy : copies) {
      if (writes.AnyOverlap(copy.src_memref)) SpillCarrySource(&copy, &moves);
    }
    std::vector<VarPtr> moved_values(targets.size());
    for (const auto& copy : copies) {
      moves.push_back(copy.move_stmt);
      moved_values[copy.index] = copy.moved_var;
    }
    // Spilling may replace the representative's moved var. Publish aliases only
    // after the complete copy plan has settled, so every slot sees that final var.
    for (size_t i = 0; i < owners.size(); ++i) {
      if (const auto& moved = moved_values[owners[i]]) values[i] = moved;
    }
    auto new_yield = MutableCopy(yield);
    new_yield->value_ = std::move(values);
    return InsertMovesAndReplaceYield(body, new_yield, moves);
  }

  /// One reconciling copy for one loop carry: `dst_memref <- source`.
  struct CarryCopy {
    size_t index;                           ///< iter_arg / yield position
    VarPtr source;                          ///< what the copy reads
    VarPtr moved_var;                       ///< the copy's LHS, replacing the yield value
    StmtPtr move_stmt;                      ///< the `tile.move` itself
    MemRefPtr src_memref;                   ///< byte range the copy reads
    MemRefPtr dst_memref;                   ///< the carry buffer the copy writes
    std::optional<MemorySpace> dst_memory;  ///< memory space of `dst_memref`
  };

  std::vector<StmtPtr> pending_allocs_;
  uint64_t spill_counter_ = 0;

  /// Byte ranges grouped by allocation, so overlap questions are index lookups
  /// rather than all-pairs scans.
  ///
  /// Within an allocation the constant-offset ranges are sorted by start.
  /// Prefix maxima provide linear overlap witnesses even when only a subset
  /// of the destinations will be written. Existence queries use binary search.
  /// A range whose offset is not constant
  /// has no position on that axis, so it is kept aside and treated as touching
  /// everything in its allocation — the same conservative answer
  /// `MemRef::MayAlias` gives.
  class CarryRangeIndex {
   public:
    /// `ranges[i]` is the byte range owned by carry `i`; a null entry is skipped.
    explicit CarryRangeIndex(const std::vector<MemRefPtr>& ranges) {
      for (size_t i = 0; i < ranges.size(); ++i) {
        if (!ranges[i]) continue;
        const Var* base = ranges[i]->base_.get();
        auto offset = As<ConstInt>(ranges[i]->byte_offset_);
        if (!offset) {
          unpositioned_[base].push_back(i);
          continue;
        }
        const int64_t begin = offset->value_;
        positioned_[base].push_back(Entry{begin, RangeEnd(begin, ranges[i]->size_, ranges[i]->span_), i});
      }
      for (auto& [base, entries] : positioned_) {
        static_cast<void>(base);
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
          return std::tie(a.begin, a.end, a.index) < std::tie(b.begin, b.end, b.index);
        });
        auto& maxima = prefix_end_[base];
        for (const auto& entry : entries) {
          maxima.push_back(maxima.empty() ? entry.end : std::max(maxima.back(), entry.end));
        }
      }
    }

    /// Bounded existence query for general overlapping source windows.
    [[nodiscard]] bool AnyOverlap(const MemRefPtr& query) const {
      const auto* base = query->base_.get();
      if (unpositioned_.count(base) != 0) return true;
      auto found = positioned_.find(base);
      if (found == positioned_.end() || found->second.empty()) return false;
      auto offset = As<ConstInt>(query->byte_offset_);
      if (!offset) return true;
      const auto& entries = found->second;
      const auto end = RangeEnd(offset->value_, query->size_, query->span_);
      auto limit = std::lower_bound(entries.begin(), entries.end(), end,
                                    [](const Entry& entry, int64_t value) { return entry.begin < value; });
      if (limit == entries.begin()) return false;
      return prefix_end_.at(base)[static_cast<size_t>(limit - entries.begin() - 1)] > offset->value_;
    }

    /// Call `fn(i)` for every indexed range that may overlap `query`.
    template <typename Fn>
    void ForEachOverlapping(const MemRefPtr& query, const Fn& fn) const {
      const Var* base = query->base_.get();
      if (auto loose = unpositioned_.find(base); loose != unpositioned_.end()) {
        for (size_t i : loose->second) fn(i);
      }
      auto bucket = positioned_.find(base);
      if (bucket == positioned_.end()) return;
      const auto& entries = bucket->second;

      auto offset = As<ConstInt>(query->byte_offset_);
      if (!offset) {  // unknown position — conservatively touches the whole allocation
        for (const Entry& entry : entries) fn(entry.index);
        return;
      }
      const int64_t begin = offset->value_;
      const int64_t end = RangeEnd(begin, query->size_, query->span_);
      // Entries are sorted by start, so the first candidate is the first one
      // starting at or after `begin`, less the single earlier one that may still
      // cover `begin`.
      auto it = std::lower_bound(entries.begin(), entries.end(), begin,
                                 [](const Entry& entry, int64_t value) { return entry.begin < value; });
      if (it != entries.begin() && std::prev(it)->end > begin) --it;
      for (; it != entries.end() && it->begin < end; ++it) {
        if (it->end > begin) fn(it->index);
      }
    }

    /// Linear overlap witnesses, as (lower index, higher index). An empty mask
    /// marks every slot active; otherwise each witness touches an active slot.
    /// Adjacent equal windows stay connected, and prefix maxima catch an active
    /// window hidden behind smaller inactive windows inside an enclosing range.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> OverlappingPairs(
        const std::vector<bool>& active = {}) const {
      std::vector<std::pair<size_t, size_t>> pairs;
      const auto is_active = [&active](size_t index) { return active.empty() || active[index]; };
      const auto append = [&pairs, &is_active](size_t lhs, size_t rhs) {
        if (lhs == rhs || (!is_active(lhs) && !is_active(rhs))) return;
        const std::pair<size_t, size_t> pair = std::minmax(lhs, rhs);
        if (pairs.empty() || pairs.back() != pair) pairs.push_back(pair);
      };
      for (const auto& [base, entries] : positioned_) {
        std::optional<size_t> widest;
        std::optional<size_t> widest_active;
        std::optional<size_t> first_active;
        for (size_t n = 0; n < entries.size(); ++n) {
          const auto& entry = entries[n];
          const bool current_active = is_active(entry.index);
          if (n > 0 && entry.begin < entries[n - 1].end) append(entries[n - 1].index, entry.index);
          const auto previous = current_active ? widest : widest_active;
          if (previous && entry.begin < entries[*previous].end) append(entries[*previous].index, entry.index);
          if (!widest || entry.end > entries[*widest].end) widest = n;
          if (current_active) {
            if (!widest_active || entry.end > entries[*widest_active].end) widest_active = n;
            if (!first_active) first_active = entry.index;
          }
        }
        // Symbolic offsets may touch every positioned window. One active
        // cross-group witness suffices to reject their unproven separation.
        const auto loose = unpositioned_.find(base);
        if (loose != unpositioned_.end() && !entries.empty()) {
          append(loose->second.front(), entries.front().index);
          if (first_active) append(loose->second.front(), *first_active);
          const auto active_loose = std::find_if(loose->second.begin(), loose->second.end(), is_active);
          if (active_loose != loose->second.end()) append(*active_loose, entries.front().index);
        }
      }
      for (const auto& [base, loose] : unpositioned_) {
        static_cast<void>(base);
        const auto representative = std::find_if(loose.begin(), loose.end(), is_active);
        for (size_t i = 1; i < loose.size(); ++i) {
          append(loose[i - 1], loose[i]);
        }
        if (representative != loose.end()) {
          for (size_t index : loose) append(*representative, index);
        }
      }
      return pairs;
    }

   private:
    struct Entry {
      int64_t begin;
      int64_t end;
      size_t index;
    };

    /// One past the last byte of `[begin, begin + size)`, as a signed offset.
    ///
    /// `MemRef::size_` is unsigned and the interval arithmetic below is signed,
    /// so a size past `INT64_MAX` or a sum that wraps would silently corrupt both
    /// the sort order and every overlap answer derived from it. No on-chip buffer
    /// comes anywhere near that, so a violation is a corrupt MemRef rather than
    /// something to accommodate.
    static int64_t RangeEnd(int64_t begin, uint64_t size, const Span& span) {
      constexpr auto kMax = std::numeric_limits<int64_t>::max();
      INTERNAL_CHECK_SPAN(size <= static_cast<uint64_t>(kMax), span)
          << "Internal error: MemRef size " << size << " does not fit a signed byte offset";
      const auto signed_size = static_cast<int64_t>(size);
      INTERNAL_CHECK_SPAN(begin <= kMax - signed_size, span)
          << "Internal error: MemRef byte range [" << begin << ", +" << size << ") overflows";
      return begin + signed_size;
    }
    std::map<const Var*, std::vector<Entry>> positioned_;
    std::map<const Var*, std::vector<int64_t>> prefix_end_;
    std::map<const Var*, std::vector<size_t>> unpositioned_;
  };

  /// The byte range each carry owns, indexed by iter_arg position (null when the
  /// carry is not a tile with a MemRef).
  static std::vector<MemRefPtr> CollectCarryRanges(const ForStmtPtr& for_stmt) {
    std::vector<MemRefPtr> ranges(for_stmt->iter_args_.size());
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike: a nested loop seeds its carries from the enclosing loop's
      // IterArg, which As<Var> would skip -- and skipping it is exactly how two
      // nested carries seeded from one outer carry would slip past the check below.
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      ranges[i] = GetDefinedMemRef(init_tile);
    }
    return ranges;
  }

  /// Reject a loop whose carries do not each own their buffer.
  ///
  /// Two carries whose buffers overlap cannot both survive an iteration: whatever
  /// order the writebacks run in, the second one destroys the first. The usual
  /// source is seeding two `pl.range` carries from one tile (`lag1 = one;
  /// lag2 = one`), which makes both carries that tile's buffer. That is a real
  /// limitation of the carry lowering rather than a compiler invariant, so it is
  /// reported to the author -- and reported here, where the carries are still
  /// named, rather than downstream as a degenerate `tile.move` onto itself.
  ///
  /// Byte ranges, not allocations: disjoint slices of one declared multi-slot
  /// buffer are independent carries and stay legal.
  static void RejectOverlappingCarryBuffers(const ForStmtPtr& for_stmt, const CarryRangeIndex& index) {
    for (const auto& [lhs, rhs] : index.OverlappingPairs()) {
      CHECK_SPAN(false, for_stmt->span_)
          << "Loop-carried values '" << for_stmt->iter_args_[lhs]->name_hint_ << "' and '"
          << for_stmt->iter_args_[rhs]->name_hint_
          << "' share the same on-chip buffer, so one iteration cannot preserve both. Give each "
             "loop-carried tile its own initial value -- build them with separate ops instead of "
             "seeding both from the same tile.";
    }
  }

  /// Order the carry copies so none overwrites a buffer another still reads.
  ///
  /// `pl.yield_` rebinds every carry at once, but the copies realizing it run in
  /// sequence, so copy A must precede copy B whenever A reads the buffer B
  /// writes. Emitting them in iter_arg order instead collapses a shift register:
  /// for `lag2 = lag1; lag1 = v`, writing lag1's buffer first makes lag2 read the
  /// new lag1 and the loop silently carries `lag2 == lag1` ([#2481]).
  ///
  /// Returns indices into `*copies` in emission order. A copy cycle — the swap
  /// `a, b = b, a`, where each carry's value lives in the other's buffer — has no
  /// valid order, so its members are spilled into scratch buffers first (appended
  /// to `*spills`, which the caller emits ahead of every copy) and their copies
  /// read the spills instead. A spill destination is a fresh allocation nothing
  /// else writes, so spilling removes every outgoing edge of the spilled copy.
  ///
  /// The graph is built once and consumed incrementally: a spill drops only the
  /// spilled copy's own outgoing edges, so the traversal resumes where it stalled
  /// instead of rebuilding anything. Victims come from the residual graph's
  /// strongly connected components, so a node merely downstream of a cycle is
  /// never spilled, and one victim set is enough to leave the residual acyclic —
  /// the traversal stalls at most once.
  ///
  /// Cost is O(k log k + E) in one loop's carry count k: `carry_index` answers each
  /// source with a binary search plus one step per range it actually overlaps, so
  /// k carries on disjoint slots of one allocation cost k lookups rather than k^2
  /// pairwise tests, and every edge is relaxed at most once.
  std::vector<size_t> OrderCarryCopies(std::vector<CarryCopy>* copies, const CarryRangeIndex& carry_index,
                                       std::vector<StmtPtr>* spills) {
    const size_t count = copies->size();

    // A copy writes its carry's byte range, so `carry_index` — built over exactly
    // those ranges — already indexes every destination. Map the carry positions it
    // reports back onto copy positions; a carry with no copy has nothing to order.
    std::map<size_t, size_t> copy_of_carry;
    for (size_t j = 0; j < count; ++j) copy_of_carry.emplace((*copies)[j].index, j);

    // A copy reads its source's byte range and writes its destination's, so two
    // copies conflict exactly when one's source overlaps the other's destination.
    std::vector<std::vector<size_t>> successors(count);
    std::vector<size_t> in_degree(count, 0);
    for (size_t i = 0; i < count; ++i) {
      carry_index.ForEachOverlapping((*copies)[i].src_memref, [&](size_t carry) {
        auto hit = copy_of_carry.find(carry);
        if (hit == copy_of_carry.end() || hit->second == i) return;
        successors[i].push_back(hit->second);  // i reads what hit->second overwrites
        ++in_degree[hit->second];
      });
    }

    std::vector<size_t> order;
    order.reserve(count);
    std::vector<bool> emitted(count, false);
    std::vector<size_t> ready;
    for (size_t i = 0; i < count; ++i) {
      if (in_degree[i] == 0) ready.push_back(i);
    }

    auto release = [&](size_t node) {
      for (size_t next : successors[node]) {
        if (--in_degree[next] == 0 && !emitted[next]) ready.push_back(next);
      }
      successors[node].clear();
    };

    while (order.size() < count) {
      if (!ready.empty()) {
        const size_t node = ready.back();
        ready.pop_back();
        if (emitted[node]) continue;
        // Kahn: every predecessor is already emitted, and an edge means "must run
        // before", so appending here builds the emission order directly.
        emitted[node] = true;
        order.push_back(node);
        release(node);
        continue;
      }

      // Nothing is runnable, so every remaining copy sits on or behind a cycle.
      // Take victims from the strongly connected components rather than from the
      // first node that still has an outgoing edge: such a node need not be on a
      // cycle at all, and spilling one that merely sits downstream of one costs a
      // scratch buffer without unblocking anything. One victim per component, all
      // of them in this round, so k/2 independent two-cycles cost a single pass.
      const std::vector<size_t> victims = CycleVictims(successors, emitted);
      INTERNAL_CHECK_SPAN(!victims.empty(), (*copies)[0].source->span_)
          << "Internal error: loop-carry copy ordering stalled with no cycle to break";
      // CycleVictims returns a set that leaves the residual acyclic, so Kahn
      // drains the rest without stalling again.
      for (size_t victim : victims) {
        SpillCarrySource(&(*copies)[victim], spills);
        release(victim);
        if (in_degree[victim] == 0 && !emitted[victim]) ready.push_back(victim);
      }
    }
    return order;
  }

  /// The strongly connected components of `nodes`, using only edges between
  /// nodes still marked active.
  ///
  /// Tarjan, iteratively so a deep chain cannot overflow the stack. Restricting
  /// to a node subset is what lets the caller re-decompose one component without
  /// walking the whole graph again.
  static std::vector<std::vector<size_t>> StronglyConnectedComponents(
      const std::vector<std::vector<size_t>>& successors, const std::vector<char>& active,
      const std::vector<size_t>& nodes) {
    constexpr size_t kUnvisited = std::numeric_limits<size_t>::max();
    const size_t count = successors.size();
    std::vector<size_t> index(count, kUnvisited);
    std::vector<size_t> lowlink(count, 0);
    std::vector<char> in_subset(count, 0);
    std::vector<char> on_stack(count, 0);
    for (size_t node : nodes) in_subset[node] = 1;

    std::vector<size_t> component_stack;
    std::vector<std::vector<size_t>> components;
    size_t next_index = 0;

    for (size_t root : nodes) {
      if (!active[root] || index[root] != kUnvisited) continue;
      index[root] = lowlink[root] = next_index++;
      component_stack.push_back(root);
      on_stack[root] = 1;
      std::vector<std::pair<size_t, size_t>> frames{{root, size_t{0}}};

      while (!frames.empty()) {
        // Re-read the frame each step: descending can reallocate the vector.
        const size_t node = frames.back().first;
        if (frames.back().second < successors[node].size()) {
          const size_t next = successors[node][frames.back().second++];
          if (!in_subset[next] || !active[next]) continue;
          if (index[next] == kUnvisited) {
            index[next] = lowlink[next] = next_index++;
            component_stack.push_back(next);
            on_stack[next] = 1;
            frames.emplace_back(next, size_t{0});
          } else if (on_stack[next]) {
            lowlink[node] = std::min(lowlink[node], index[next]);
          }
          continue;
        }

        if (lowlink[node] == index[node]) {
          std::vector<size_t> component;
          while (true) {
            const size_t member = component_stack.back();
            component_stack.pop_back();
            on_stack[member] = 0;
            component.push_back(member);
            if (member == node) break;
          }
          components.push_back(std::move(component));
        }

        frames.pop_back();
        if (!frames.empty()) {
          const size_t parent = frames.back().first;
          lowlink[parent] = std::min(lowlink[parent], lowlink[node]);
        }
      }
    }
    return components;
  }

  /// A set of copies whose spilling leaves the residual graph acyclic.
  ///
  /// Every cycle lies inside one strongly connected component, so it is enough
  /// to make each component acyclic. Spilling a node drops all of its outgoing
  /// edges, which breaks every cycle through it -- but a component holding
  /// several cycles that do not all pass through one node stays cyclic after a
  /// single spill, so what is left of that component is decomposed again. The
  /// re-decomposition is confined to the component's own nodes: the caller gets
  /// one acyclic residual from one call and never re-walks the whole graph.
  ///
  /// Victims are always on a cycle at the moment they are chosen, so no scratch
  /// buffer is spent on a copy that merely sits downstream of one. The set is
  /// not guaranteed minimal -- a minimum feedback vertex set is NP-hard, and the
  /// lowest-numbered member is picked instead.
  ///
  /// Cost is O(k + E) for the first decomposition plus, per component that needs
  /// more than one spill, another pass over that component alone.
  static std::vector<size_t> CycleVictims(const std::vector<std::vector<size_t>>& successors,
                                          const std::vector<bool>& emitted) {
    const size_t count = successors.size();
    std::vector<char> active(count, 0);
    std::vector<size_t> all_nodes;
    for (size_t i = 0; i < count; ++i) {
      if (emitted[i]) continue;
      active[i] = 1;
      all_nodes.push_back(i);
    }

    std::vector<size_t> victims;
    std::vector<std::vector<size_t>> pending{std::move(all_nodes)};
    while (!pending.empty()) {
      const std::vector<size_t> subset = std::move(pending.back());
      pending.pop_back();
      for (auto& component : StronglyConnectedComponents(successors, active, subset)) {
        const bool has_self_edge = component.size() == 1 &&
                                   std::find(successors[component[0]].begin(), successors[component[0]].end(),
                                             component[0]) != successors[component[0]].end();
        if (component.size() < 2 && !has_self_edge) continue;

        const size_t victim = *std::min_element(component.begin(), component.end());
        victims.push_back(victim);
        active[victim] = 0;  // its outgoing edges are gone once it reads a spill

        // The rest of this component may still hold a cycle the victim was not on.
        std::vector<size_t> remainder;
        for (size_t member : component) {
          if (member != victim) remainder.push_back(member);
        }
        if (remainder.size() > 1) pending.push_back(std::move(remainder));
      }
    }
    return victims;
  }

  /// Redirect one carry copy through a fresh scratch buffer, so the copy no
  /// longer reads a buffer its siblings overwrite. Emits the scratch allocation
  /// into `pending_allocs_` for the caller to hoist.
  void SpillCarrySource(CarryCopy* copy, std::vector<StmtPtr>* spills) {
    auto src_tile = GetTileTypeWithMemRef(copy->source->GetType());
    INTERNAL_CHECK_SPAN(src_tile, copy->source->span_)
        << "Internal error: a loop-carry copy source must be a TileType with a MemRef";
    auto src_memref = GetDefinedMemRef(src_tile);
    auto src_memory = src_tile->GetMemorySpace();
    if (canonical_loop_copies_) {
      CHECK_SPAN(src_memory && IsTileMoveEverSupported(*src_memory, *src_memory), copy->source->span_)
          << "Buffer IR cannot snapshot a transfer source without a supported same-space tile.move";
    }
    INTERNAL_CHECK_SPAN(src_memory.has_value(), copy->source->span_)
        << "Internal error: a loop-carry copy source must carry a memory space to spill";

    // `mem_<space>_carry_spill_N` cannot collide with InitMemRef's
    // `mem_<space>_<N>`, which never has a non-numeric segment before its
    // counter, and the counter is per-function so two spills cannot collide
    // either. The name is cosmetic, but it has to stay unique for the
    // print -> parse roundtrip to rebind the same allocation.
    std::string space_str = MemorySpaceToString(*src_memory);
    std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const std::string base_name =
        canonical_loop_copies_ ? names_.Fresh("mem_" + space_str + "_carry_spill_")
                               : "mem_" + space_str + "_carry_spill_" + std::to_string(spill_counter_++);
    auto scratch_base = std::make_shared<Var>(base_name, GetPtrType(), copy->source->span_);
    auto scratch_memref = std::make_shared<MemRef>(scratch_base, static_cast<int64_t>(0), src_memref->size_,
                                                   copy->source->span_);
    pending_allocs_.push_back(CreateAllocStatement(scratch_memref, *src_memory));

    auto [spilled_var, spill_stmt] = CreateTileMove(copy->source, scratch_memref, src_memory);
    spills->push_back(std::move(spill_stmt));

    auto [moved_var, move_stmt] = CreateTileMove(spilled_var, copy->dst_memref, copy->dst_memory);
    copy->source = spilled_var;
    copy->moved_var = std::move(moved_var);
    copy->move_stmt = std::move(move_stmt);
    copy->src_memref = scratch_memref;
  }

  // Create a tile.move operation that copies source into target_memref's buffer.
  // Returns (moved_var, move_assign_stmt).
  std::pair<VarPtr, StmtPtr> CreateTileMove(const VarPtr& source, const MemRefPtr& target_memref,
                                            std::optional<MemorySpace> target_memory) {
    return CreateStorageMove(
        source, target_memref, target_memory, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        canonical_loop_copies_ ? names_.Fresh("storage_transfer_") : source->name_hint_ + "_mv");
  }

  // Patch iter_args and return_vars to share initValue's MemRef.
  // Returns the original ForStmt if no patching is needed.
  StmtPtr PatchIterArgsAndReturnVars(const ForStmtPtr& for_stmt, const YieldStmtPtr& yield_stmt) {
    bool changed = false;
    std::vector<IterArgPtr> new_iter_args = for_stmt->iter_args_;
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Patch iter_arg if its MemRef differs from initValue's. Addresses, not
      // allocations: an iter_arg left on a sibling slot of one multi-slot
      // allocation is a different buffer and still has to be re-pointed. These
      // sites only retype a node, they never move data, so anything not provably
      // init's address is repointed: an unprovable pair costs nothing to realign
      // and would otherwise leave a possibly-wrong slot in the metadata.
      auto ia_tile = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (ia_tile && ia_tile->memref_.has_value()) {
        auto ia_memref = GetDefinedMemRef(ia_tile);
        if (CompareBaseAddress(ia_memref, init_memref) != AddressRelation::kSame) {
          auto new_ia_type = CloneTypeWithMemRefAndRemapExprs(
              ia_tile, init_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
              init_tile->GetMemorySpace());
          new_iter_args[i] =
              std::make_shared<IterArg>(for_stmt->iter_args_[i]->name_hint_, new_ia_type,
                                        for_stmt->iter_args_[i]->initValue_, for_stmt->iter_args_[i]->span_);
          var_remap_[for_stmt->iter_args_[i].get()] = new_iter_args[i];
          changed = true;
        }
      }

      // Patch return_var to share yield value's MemRef (which should == initValue's after fixup)
      if (i >= new_return_vars.size() || !yield_stmt || i >= yield_stmt->value_.size()) continue;
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;
      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!rv_tile || !rv_tile->memref_.has_value()) continue;
      auto rv_memref = GetDefinedMemRef(rv_tile);
      if (CompareBaseAddress(rv_memref, yield_memref) != AddressRelation::kSame) {
        auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
            rv_tile, yield_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
            yield_tile->GetMemorySpace());
        new_return_vars[i] =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
        // Register old→new so downstream references (e.g., ReturnStmt) are updated
        var_remap_[for_stmt->return_vars_[i].get()] = new_return_vars[i];
        changed = true;
      }
    }

    if (!changed) return for_stmt;

    auto patched_body = VisitStmt(for_stmt->body_);

    for (const auto& old_iter_arg : for_stmt->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    auto new_for = MutableCopy(for_stmt);
    new_for->iter_args_ = new_iter_args;
    new_for->body_ = patched_body;
    new_for->return_vars_ = std::move(new_return_vars);
    return new_for;
  }

  // Follow the same trailing path as GetLastYieldStmt. Keep transfers inside
  // transparent scopes and leave yields of nested control-flow regions alone.
  static StmtPtr InsertMovesAndReplaceYield(const StmtPtr& body, const YieldStmtPtr& new_yield,
                                            const std::vector<StmtPtr>& move_stmts) {
    if (As<YieldStmt>(body)) {
      // Body is just a yield — wrap moves + yield in SeqStmts
      std::vector<StmtPtr> stmts;
      stmts.insert(stmts.end(), move_stmts.begin(), move_stmts.end());
      stmts.push_back(new_yield);
      return SeqStmts::Flatten(std::move(stmts), body->span_);
    }
    if (auto seq = As<SeqStmts>(body)) {
      if (seq->stmts_.empty()) return body;
      std::vector<StmtPtr> new_children = seq->stmts_;
      new_children.back() = InsertMovesAndReplaceYield(new_children.back(), new_yield, move_stmts);
      return SeqStmts::Flatten(std::move(new_children), body->span_);
    }
    if (auto scope = As<SplitAivScopeStmt>(body)) {
      auto result = MutableCopy(scope);
      result->body_ = InsertMovesAndReplaceYield(scope->body_, new_yield, move_stmts);
      return result;
    }
    if (auto scope = As<RuntimeScopeStmt>(body)) {
      auto result = MutableCopy(scope);
      result->body_ = InsertMovesAndReplaceYield(scope->body_, new_yield, move_stmts);
      return result;
    }
    return body;
  }
};

// Check if a statement is a tile.alloc AssignStmt for an unused MemRef
bool IsUnusedAllocStmt(const StmtPtr& stmt, const std::set<const Var*>& used_bases) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return false;
  auto call = As<Call>(assign->value_);
  if (!call) return false;
  if (!IsOp(call, "tile.alloc") && !IsOp(call, "tensor.alloc")) return false;
  // Alloc LHS is a Ptr Var — check if any MemRef's base_ still references it
  return used_bases.find(assign->var_.get()) == used_bases.end();
}

// Remove unused alloc statements from a SeqStmts body
StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body, const std::set<const Var*>& used_bases) {
  auto seq = As<SeqStmts>(body);
  if (!seq) return body;

  std::vector<StmtPtr> new_seq_stmts;
  bool changed = false;

  for (const auto& child : seq->stmts_) {
    if (IsUnusedAllocStmt(child, used_bases)) {
      changed = true;
      continue;
    }
    new_seq_stmts.push_back(child);
  }

  if (!changed) return body;
  return SeqStmts::Flatten(std::move(new_seq_stmts), body->span_);
}

/// Strip the transient ``pipeline_membership`` attr from every Call once
/// MemoryReuse has consumed it (in ComputeLifetimes). The attr exists only to
/// carry pipeline-stage identity from LowerPipelineLoops to here; leaving it on
/// the IR would ride downstream into later passes and codegen. Stripping at the
/// end of MemoryReuse keeps the post-reuse IR clean.
class StripPipelineMembershipMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call || !call->HasAttr(kPipelineMembershipAttr)) return visited;
    auto new_attrs = StripAttr(call->attrs_, kPipelineMembershipAttr);
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                  call->GetType(), call->span_);
  }
};

/// Reconcile required storage aliases after producer/control-flow retyping.
///
/// An AssignStmt whose value is a bare Var (`x = y`, not a Call) is a pure SSA
/// rename — `x` must alias `y`'s buffer.  Accumulator target planning retargets an
/// if-phi's producers + return_var onto the accumulator buffer, which can leave a
/// downstream copy of the (now-moved) return_var still typed on the old buffer:
///   c: Tile[..., mem_acc_17] = c_phi_2   // c_phi_2 was retargeted to mem_acc_5
/// Codegen would then store from the stale buffer.  This single forward pass
/// retypes every such copy's LHS to its RHS's MemRef and substitutes the LHS's
/// downstream uses, so the whole rename chain follows.  A no-op when no identity
/// copy has a buffer mismatch (the common case). Registered writeback results
/// and first-input buffer aliases obey the same forward-propagation rule.
class NormalizeIdentityCopyBuffersMutator : public StorageBoundaryMutator {
 public:
  NormalizeIdentityCopyBuffersMutator() = default;

  /// `unsettled` seeds the IfStmt return_vars whose buffer YieldFixup has not
  /// chosen yet; the traversal grows it with every required alias. See
  /// `ReanchorAliasingOutput` for why anchoring onto one is wrong.
  explicit NormalizeIdentityCopyBuffersMutator(std::set<const Var*> unsettled)
      : unsettled_(std::move(unsettled)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = subst_.find(op);
    return it != subst_.end() ? it->second : op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto found = var_remap_.find(op.get());
    if (found != var_remap_.end()) return found->second;
    auto visited = As<IterArg>(StorageBoundaryMutator::VisitExpr_(op));
    INTERNAL_CHECK_SPAN(visited, op->span_) << "Internal error: IterArg visit must yield an IterArg";
    if (!AsTensorTypeLike(visited->GetType())) return visited;
    // GM loop carries are mutable references: codegen binds their handles to
    // the initializer, including on zero-trip loops. A renamed initializer may
    // have moved even when this carry's own type has not changed.
    if (auto type = TensorBoundaryType(visited->GetType(), visited->initValue_)) {
      visited = std::make_shared<IterArg>(visited->name_hint_, type, visited->initValue_, visited->span_);
    }
    if (IsUnsettled(AsVarLike(op->initValue_)) || IsUnsettled(AsVarLike(visited->initValue_))) {
      unsettled_.insert(op.get());
      unsettled_.insert(visited.get());
    }
    var_remap_[op.get()] = visited;
    return visited;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return NormalizeTensorLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return NormalizeTensorLoop(op); }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Only bare-Var identity copies are pure renames; Call producers are not.
    auto src_var = AsVarLike(op->value_);
    if (src_var) {
      auto new_src = AsVarLike(VisitExpr(op->value_));  // follow prior substitutions
      // A rename of an unsettled value denotes that same storage, so it is just as
      // unsettled: `tmp = phi` then `t = <in-place>(tmp, ...)` would otherwise walk
      // straight past the guard in ReanchorAliasingOutput and anchor onto the phi's
      // stale buffer through the alias.
      if (IsUnsettled(src_var) || IsUnsettled(new_src)) {
        unsettled_.insert(op->var_.get());
      }
      auto lhs = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
      auto rhs = new_src ? std::dynamic_pointer_cast<const ShapedType>(new_src->GetType()) : nullptr;
      if (lhs && rhs && lhs->memref_ && rhs->memref_ &&
          (!SamePhysicalWindow(*lhs->memref_, *rhs->memref_) ||
           lhs->GetMemorySpace() != rhs->GetMemorySpace())) {
        auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_src->GetType(), op->var_->span_);
        if (IsUnsettled(op->var_)) unsettled_.insert(new_lhs.get());
        subst_[op->var_] = new_lhs;
        return std::make_shared<AssignStmt>(new_lhs, new_src, op->span_);
      }
    }
    // Required aliases must follow the current input even when another mutator
    // changed it earlier. Normalize views in this same walk so a writeback
    // repair cannot leave its downstream views on the previous buffer.
    if (auto reanchored = ReanchorAliasingOutput(op)) return reanchored;
    return IRMutator::VisitStmt_(op);
  }

 private:
  TypePtr TensorBoundaryType(const TypePtr& type, const ExprPtr& source) {
    auto target = AsTensorTypeLike(type);
    auto input = source ? AsTensorTypeLike(source->GetType()) : nullptr;
    if (!target || !input || !target->memref_ || !input->memref_ ||
        (SamePhysicalWindow(*target->memref_, *input->memref_) &&
         target->GetMemorySpace() == input->GetMemorySpace())) {
      return nullptr;
    }
    return CloneTypeWithMemRef(type, input->memref_, input->GetMemorySpace());
  }

  template <typename LoopPtr>
  StmtPtr NormalizeTensorLoop(const LoopPtr& op) {
    auto visited = std::static_pointer_cast<const typename LoopPtr::element_type>(IRMutator::VisitStmt_(op));
    auto result = MutableCopy(visited);
    bool changed = false;
    for (size_t i = 0; i < visited->iter_args_.size() && i < visited->return_vars_.size(); ++i) {
      const auto& argument = visited->iter_args_[i];
      const auto& return_var = visited->return_vars_[i];
      if (!AsTensorTypeLike(return_var->GetType())) continue;
      auto replacement = return_var;
      if (auto type = TensorBoundaryType(return_var->GetType(), argument)) {
        replacement = std::make_shared<Var>(return_var->name_hint_, type, return_var->span_);
        subst_[op->return_vars_[i]] = replacement;
        result->return_vars_[i] = replacement;
        changed = true;
      }
      if (IsUnsettled(argument)) {
        unsettled_.insert(op->return_vars_[i].get());
        unsettled_.insert(replacement.get());
      }
    }
    // Only the loop boundary aliases its initializer. Independently produced
    // yielded tensors keep their own storage and are never retargeted here.
    return changed ? result : visited;
  }

  /// True when `var` denotes storage whose buffer YieldFixup has not chosen yet —
  /// an IfStmt return_var seeded into `unsettled_`, or any required alias of one.
  [[nodiscard]] bool IsUnsettled(const VarPtr& var) const {
    return var != nullptr && unsettled_.count(var.get()) != 0;
  }

  /// Before branch destinations are established, aliases of a provisional
  /// phi must stay unsettled too. Otherwise a downstream writeback could pin
  /// itself to the temporary buffer inherited from one arm. Once destinations
  /// are established, rebuild required aliases from their current input while
  /// preserving shape/dtype/view metadata. Allocating view operations are excluded.
  StmtPtr ReanchorAliasingOutput(const AssignStmtPtr& op) {
    auto call = As<Call>(op->value_);
    if (!call || !call->op_) return nullptr;
    auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
    const bool is_view = op_predicates::IsBufferAliasingViewOp(call->op_->name_);
    if (!reuse_idx && (!is_view || call->args_.empty())) return nullptr;
    // An explicit writeback contract wins over memory-space inheritance.
    auto in_var = AsVarLike(call->args_[reuse_idx ? *reuse_idx : 0]);
    if (!in_var) return nullptr;
    auto new_in = AsVarLike(VisitExpr(in_var));  // follow prior subst_ renames
    if (!new_in) return nullptr;
    if (IsUnsettled(in_var) || IsUnsettled(new_in)) {
      // A view or another writeback cannot hide an unresolved phi from the
      // next writer in the chain. YieldFixup must choose that storage first.
      unsettled_.insert(op->var_.get());
      return nullptr;
    }
    auto lhs = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
    auto input = std::dynamic_pointer_cast<const ShapedType>(new_in->GetType());
    if (!lhs || !input || !lhs->memref_ || !input->memref_) return nullptr;
    auto target = *input->memref_;
    if (!reuse_idx) {
      // Derive only the view-relative offset. A writeback's transfer offset
      // selects what it writes; its result still denotes the whole destination.
      auto relative = ComputeViewByteOffset(call, new_in->GetType());
      target = MakeAliasingMemRef(target, relative, (*lhs->memref_)->size_, (*lhs->memref_)->span_);
    }
    if (SamePhysicalWindow(*lhs->memref_, target) && lhs->GetMemorySpace() == input->GetMemorySpace()) {
      return nullptr;
    }
    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), target, input->GetMemorySpace());
    auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    subst_[op->var_] = new_lhs;
    auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(recursed, op->span_) << "Internal error: AssignStmt visit must yield an AssignStmt";
    return std::make_shared<AssignStmt>(new_lhs, recursed->value_, recursed->span_);
  }

  std::map<VarPtr, ExprPtr> subst_;
  std::set<const Var*> unsettled_;
};

/// Every IfStmt result reachable in a body. During pre-boundary normalization,
/// these can still carry a provisional input buffer.
class IfPhiReturnVarCollector : public IRVisitor {
 public:
  std::set<const Var*> phis;

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) phis.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
};

/// The pre-YieldFixup half of the identity-copy normalization bracket: identical
/// to the plain run except that an in-place producer is never anchored onto a phi
/// whose buffer YieldFixup has yet to choose (`ReanchorAliasingOutput`).
StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body) {
  IfPhiReturnVarCollector collector;
  collector.VisitStmt(body);
  return NormalizeIdentityCopyBuffersMutator(std::move(collector.phis)).VisitStmt(body);
}

/**
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities by walking the full
 * IR tree to compute variable lifetimes, then applying greedy MemRef sharing.
 * Variables that can share memory will point to the same MemRef object.
 * After sharing, redundant alloc operations are removed.
 */
// Semantic must-alias materialization — the "Step 0" formerly at the head of
// MemoryReuse. Propagates each ForStmt iter_arg/initValue's canonical MemRef
// down the yield/producer chain so accumulator producers (and other loop-carry
// / in-place chains) write directly into the carried buffer. This is a
// *semantics-required* aliasing (the loop accumulator must live in one buffer),
// as opposed to the opportunistic lifetime coalescing in MemoryReuse.
//
// Split into its own pass so it can run without the (skippable) lifetime-reuse
// phase: when ptoas owns lifetime reuse (memory_planner=PTOAS), this still runs
// while MemoryReuse is skipped, so codegen can emit a shared tile_buf handle for
// the must-alias buffers and ptoas PlanMemory does the reuse.
FunctionPtr TransformMaterializeSemanticAliases(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MaterializeSemanticAliases cannot run on null function";
  BufferIRFunctionScope buffer_scope(func);

  // Orchestration bodies submit tasks and never hold TileType variables; a
  // Graph body is orchestration too, so it is skipped for the same reason.
  if (IsOrchestrationLike(func->func_type_)) return func;

  StmtPtr new_body = func->body_;
  const auto* ctx = PassContext::Current();
  if (ctx != nullptr && ctx->GetEnableBufferIR()) {
    auto current_function = MutableCopy(func);
    current_function->body_ = new_body;
    SnapshotAssembleSources snapshots(current_function);
    new_body = snapshots.VisitStmt(new_body);
    new_body = InsertAllocsIntoBody(new_body, snapshots.allocations);
  }
  // Required aliases and entry isolation precede the immutable planning
  // snapshot on every path. A loop may update a private carry while its old
  // initializer remains independently observable after the loop.
  new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);
  new_body = RetypeApplier({}).VisitStmt(new_body);
  LoopInputIsolation isolation;
  isolation.Plan(new_body, func->params_, func->param_directions_);
  if (!isolation.initializers.empty()) {
    RetypeApplier applier(std::move(isolation.rewrites), std::move(isolation.initializers),
                          std::move(isolation.entries));
    new_body = applier.VisitStmt(new_body);
    new_body = InsertAllocsIntoBody(new_body, isolation.allocations);
    new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);
  }
  new_body = RetypeApplier({}).VisitStmt(new_body);
  // Logical result identities and both incoming edges are planned together.
  // Required loop/Acc targets and optional reuse share one SSA snapshot and
  // one rewrite transaction; planner selection happens only after this contract.
  std::vector<StmtPtr> allocations;
  TopDownRetargeter storage;
  auto rewrites = storage.PlanStorage(func, new_body, &allocations);
  if (!rewrites.empty()) new_body = RetypeApplier(std::move(rewrites)).VisitStmt(new_body);
  new_body = InsertAllocsIntoBody(new_body, allocations);
  new_body = RetypeApplier({}).VisitStmt(new_body);

  // Tile PYPTO still reconciles loop writebacks after packing. If edges must
  // already be complete so every planner sees their actual reads and writes.
  const bool all_boundaries =
      ctx != nullptr && (ctx->GetEnableBufferIR() || ctx->GetMemoryPlanner() != MemoryPlanner::PyPTO);
  YieldFixupMutator yield_fixup(
      /*canonical_loop_copies=*/true,
      all_boundaries ? YieldFixupMutator::Scope::AllControlFlow : YieldFixupMutator::Scope::IfResults);
  new_body = yield_fixup.Run(new_body, func->params_);
  new_body = InsertAllocsIntoBody(new_body, yield_fixup.TakePendingAllocs());
  new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);
  new_body = RemoveUnusedAllocStatements(new_body, memref_collectors::CollectUsedBasePtrs(new_body));

  if (new_body == func->body_) return func;

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_,
                                          func->requires_runtime_binding_, func->ir_stage_);
}

struct TransferCoalescing {
  std::set<const Var*> definitions;
  ForbidAliasMap subrange_separation;
};

/// Whole-root, same-descriptor copies can become SSA identities when placement
/// coalesces their storage. Remaining copies must keep disjoint byte windows.
TransferCoalescing CollectCoalescibleTransfers(const StmtPtr& body, const LifetimeAnalysisResult& analysis,
                                               const std::vector<VarPtr>& parameters) {
  class Collector : public IRVisitor {
   public:
    Collector(const LifetimeAnalysisResult& analysis, const std::vector<VarPtr>& parameters) {
      // Carries and region results are not necessarily sharing-group members.
      // Resolve every handle through its allocation identity instead of
      // treating a missing variable entry as proof that it is the whole root.
      for (const auto& interval : analysis.lifetimes) {
        if (const auto memref = GetTypeMemRef(interval.variable->GetType()); memref && *memref) {
          representatives_.emplace((*memref)->base_.get(), interval.variable);
        }
      }
      // This planner tracks body definitions. Preserve the existing no-alias
      // protection for incoming storage, including aliases of Tile parameters.
      for (const auto& parameter : parameters) {
        if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
          parameter_bases_.insert(GetDefinedMemRef(tile)->base_.get());
        }
      }
    }
    TransferCoalescing transfers;

    // Expressions cannot contain assignments; do not expand carry initializers.
    void VisitExpr(const ExprPtr&) override {}

    void VisitStmt_(const AssignStmtPtr& assign) override {
      auto call = As<Call>(assign->value_);
      if (!IsOp(call, "tile.move") || call->args_.size() != 1) return;
      auto input = AsVarLike(call->args_[0]);
      auto source = input ? GetTileTypeWithMemRef(input->GetType()) : nullptr;
      auto output = GetTileTypeWithMemRef(assign->var_->GetType());
      if (!SameTransferDescriptor(source, output)) return;
      auto source_memref = GetDefinedMemRef(source);
      auto output_memref = GetDefinedMemRef(output);
      if (parameter_bases_.count(source_memref->base_.get()) ||
          parameter_bases_.count(output_memref->base_.get())) {
        return;
      }
      const bool same_slot =
          source_memref->slot_count_ == output_memref->slot_count_ &&
          source_memref->slot_index_.has_value() == output_memref->slot_index_.has_value() &&
          (!source_memref->slot_index_ ||
           structural_equal(*source_memref->slot_index_, *output_memref->slot_index_));
      auto source_root = Representative(input);
      auto output_root = Representative(assign->var_);
      if (!source_root || !output_root) return;
      if (same_slot && source_memref->size_ == output_memref->size_ &&
          SamePhysicalWindow(GetTypeMemRef(source_root->GetType()).value_or(nullptr), source_memref) &&
          SamePhysicalWindow(GetTypeMemRef(output_root->GetType()).value_or(nullptr), output_memref)) {
        transfers.definitions.insert(assign->var_.get());
        transfers.subrange_separation[output_root.get()].push_back(input);
      }
    }

   private:
    std::map<const Var*, VarPtr> representatives_;
    std::set<const Var*> parameter_bases_;
    [[nodiscard]] VarPtr Representative(const VarPtr& value) const {
      const auto memref = GetTypeMemRef(value->GetType());
      if (!memref || !*memref) return nullptr;
      const auto found = representatives_.find((*memref)->base_.get());
      return found == representatives_.end() ? nullptr : found->second;
    }
  } collector(analysis, parameters);
  collector.VisitStmt(body);
  return std::move(collector.transfers);
}

/// Copy coalescing removes the operation instead of executing an in-place TMOV.
/// Exact descriptor equality makes source substitution preserve the type/view.
class RemoveIdentityTransfers : public IRMutator {
 public:
  explicit RemoveIdentityTransfers(const std::set<const Var*>& fixed_bases) : fixed_bases_(fixed_bases) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = As<AssignStmt>(IRMutator::VisitStmt_(op));
    auto call = As<Call>(result->value_);
    if (!IsOp(call, "tile.move") || call->args_.size() != 1) return result;
    auto source = GetTileTypeWithMemRef(call->args_[0]->GetType());
    auto target = GetTileTypeWithMemRef(result->var_->GetType());
    if (!SameTransferDescriptor(source, target) ||
        !SamePhysicalWindow(GetDefinedMemRef(source), GetDefinedMemRef(target))) {
      return result;
    }
    // Fixed storage never participates in optional copy coalescing. Preserve
    // its declared operations so their address restrictions remain validated.
    if (fixed_bases_.count(GetDefinedMemRef(source)->base_.get())) return result;
    if (!AsVarLike(call->args_[0])) return result;
    var_remap_[op->var_.get()] = call->args_[0];
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
  }

 private:
  const std::set<const Var*>& fixed_bases_;
};

FunctionPtr TransformMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MemoryReusePass cannot run on null function";
  BufferIRFunctionScope buffer_scope(func);

  // Orchestration bodies submit tasks and never hold TileType variables, so
  // there is nothing for memory reuse to do — skip them silently. A Graph body
  // is orchestration too.
  if (IsOrchestrationLike(func->func_type_)) return func;

  // Step 0 (semantic must-alias retarget) now runs in the preceding
  // MaterializeSemanticAliases pass, so the body here is already retargeted.
  StmtPtr new_body = func->body_;

  // MaterializeSemanticAliases has already settled every if-result target
  // for all planners. Placement changes allocation locations, not result classes.

  // Step 1: Compute lifetimes by walking full IR tree
  auto analysis_result = AnalyzeAllocationLifetimesImpl(new_body);

  if (analysis_result.lifetimes.empty()) {
    LOG_DEBUG << "No TileType variables found in function '" << func->name_ << "', skipping memory reuse";
    return func;
  }

  // Step 2: Identify reuse opportunities.  On Ascend910 split-AIV functions,
  // collect the load + tpop_from_aic hazard inputs so the reuse decision never
  // forms the hazardous in-place sharing (folds in the former
  // LegalizePTOBufferReuse responsibility).  Off-910B the inputs stay empty and
  // reuse behaviour is unchanged.
  const auto transfers = CollectCoalescibleTransfers(new_body, analysis_result, func->params_);
  auto planning_function = MutableCopy(func);
  planning_function->body_ = new_body;
  const AllocationConstraintAnalysis constraint_analysis =
      AnalyzeAllocationConstraints(planning_function, analysis_result, "MemoryReuse", transfers.definitions);
  const HazardInputs& hazard = constraint_analysis.target_hazard_inputs;
  const ForbidAliasMap& forbid_alias = constraint_analysis.forbid_alias;

  // Per-space reserved end (the SpaceFootprint reserved_start for the exact fit check). Only meaningful
  // with a configured backend; empty otherwise ⇒ reserved_start defaults to 0.
  std::map<MemorySpace, uint64_t> reserved_end_by_space;
  if (backend::BackendConfig::IsConfigured()) {
    auto policy = backend::GetBackend()->CreateMemoryAllocatorPolicy();
    if (policy) {
      // Shared with AllocateMemoryAddr — the reserved start is parity-by-construction, not comment-synced.
      const auto resolution = ResolveReserveBufferBases(func, *policy);
      for (const auto& [space, end] : resolution.reserved_end_by_space) reserved_end_by_space[space] = end;
    }
  }
  // Bases the author declared via a one-argument `pl.MemRef(...)`, materialized by
  // InitMemRef as a pinned alloc. Their membership is off-limits to the packer.
  // InitMemRef hoists every alloc to the body's top-level SeqStmts, so this scans
  // that list in place rather than flattening a copy of the whole body. Missing
  // that shape must not fail open: an empty `pinned_bases` silently disables both
  // the packer isolation and the co-liveness check below, handing the author back
  // exactly the coalescing the binding was written to prevent.
  auto pinned_bases = constraint_analysis.declared_allocation_bases;
  // Tile parameters borrow caller-owned storage. Body-local aliases and if
  // results must not let the packer move that storage or add new occupants.
  // Keep declaration validation separate: parameters are fixed by the ABI,
  // rather than by a user-declared allocation in this function.
  for (const auto& parameter : func->params_) {
    if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
      pinned_bases.insert(GetDefinedMemRef(tile)->base_.get());
    }
  }

  std::vector<Diagnostic> hints;
  auto reuse_map = IdentifyReuseOpportunities(
      analysis_result.lifetimes, hazard, forbid_alias, transfers.subrange_separation,
      analysis_result.var_sharing_groups, analysis_result.pipeline_membership,
      analysis_result.pipeline_load_tiles, analysis_result.subrange_unsafe_groups, reserved_end_by_space,
      pinned_bases, planning_function, &hints);
  // Surface capacity-forced pipeline-depth reductions (perf hints) and legacy-fallback overflows
  // (warnings) through the unified diagnostic channel → perf_hints.log / stderr.
  if (!hints.empty()) EmitDiagnostics(hints, "MemoryReuse");

  // Step 3: Apply MemRef sharing (skip if no reuse candidates)
  if (!reuse_map.empty()) {
    new_body = ApplyMemRefSharing(new_body, reuse_map, analysis_result.var_sharing_groups);

    // Step 3.5: Re-align loop-carried iter_arg/return_var MemRefs to their
    // (now-reused) initValue, top-down.  Reuse retypes producer/init AssignStmt
    // vars but leaves loop-carry nodes stale; for nested pipelined accumulators
    // that split chain would otherwise force YieldFixupMutator to emit invalid
    // acc->acc tile.move ops (#1352).
    new_body = RetypeApplier({}).VisitStmt(new_body);
  }

  // Reconcile bare-Var SSA identity copies whose buffers diverged
  // from the value they rename.  No-op when no mismatch exists.
  //
  // This runs on both sides of YieldFixup because both sides create such a
  // divergence, from opposite directions:
  //
  //  - before, because placement or carry alignment leaves a rename
  //    stranded on the pre-coalesce buffer, and because a carry rename such as
  //    `lag2 = lag1` only lands on lag1's buffer here. YieldFixup has to see the
  //    settled buffer to order the carry writebacks against each other: against
  //    the pre-normalization buffers every rename still sits on a buffer of its
  //    own, so no conflict is visible and a shift register silently collapses
  //    ([#2481]).
  //  - after, because boundary reconciliation introduces typed handles for
  //    transfers and loop results. Their downstream renames must follow the
  //    settled windows without choosing a different if-result destination.
  //
  // The mutator is idempotent, so the second run is a no-op whenever the first
  // already settled everything.
  new_body = RemoveIdentityTransfers(pinned_bases).VisitStmt(new_body);
  new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);

  // Step 4: Fix ForStmt/IfStmt yield/return_var MemRef mismatches
  const auto* ctx = PassContext::Current();
  YieldFixupMutator yield_fixup(/*canonical_loop_copies=*/ctx != nullptr && ctx->GetEnableBufferIR());
  new_body = yield_fixup.Run(new_body, func->params_);
  // A carry-copy cycle needs a scratch buffer; its alloc belongs on the body head
  // like every other allocation, and Step 5 below keeps it because it is in use.
  new_body = InsertAllocsIntoBody(new_body, yield_fixup.TakePendingAllocs());

  // Step 4.5: the second half of the bracket described at the pre-boundary normalization.
  new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);

  // Step 5: Remove alloc statements for MemRefs no longer in use
  auto used_bases = memref_collectors::CollectUsedBasePtrs(new_body);
  new_body = RemoveUnusedAllocStatements(new_body, used_bases);

  // Step 6: Strip the now-consumed pipeline_membership attr so it does not ride
  // downstream into later passes / codegen. It was only needed to carry stage
  // identity from LowerPipelineLoops to the reuse decision above.
  new_body = StripPipelineMembershipMutator().VisitStmt(new_body);

  auto result = std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_,
                                                 func->requires_runtime_binding_, func->ir_stage_);
  return result;
}

}  // namespace

AllocationConstraintAnalysis AnalyzeAllocationConstraints(const FunctionPtr& func,
                                                          const LifetimeAnalysisResult& lifetimes,
                                                          const char* consumer,
                                                          const std::set<const Var*>& coalescible_transfers) {
  AllocationConstraintAnalysis result;
  result.declared_allocation_sizes = CollectPinnedAllocSizes(func->body_, func->span_, consumer);
  for (const auto& [base, size] : result.declared_allocation_sizes) {
    static_cast<void>(size);
    result.declared_allocation_bases.insert(base);
  }
  ValidateDeclaredAllocs(func->body_, result.declared_allocation_bases, lifetimes.var_liveness);

  result.needs_load_tpop_hazard_guard = NeedsLoadTpopHazardGuard(func);
  if (result.needs_load_tpop_hazard_guard) {
    HazardInputCollector collector;
    collector.Run(func->body_);
    result.target_hazard_inputs = collector.Take();
  }

  WrittenWorkspaceCollector workspace_collector;
  workspace_collector.VisitStmt(func->body_);
  result.target_hazard_inputs.written_workspaces = workspace_collector.TakeVars();
  result.target_hazard_inputs.written_workspace_bases = workspace_collector.TakeBases();
  result.target_hazard_inputs.looping_written_workspaces = workspace_collector.TakeLoopingVars();
  result.target_hazard_inputs.looping_written_workspace_bases = workspace_collector.TakeLoopingBases();

  const auto* context = PassContext::Current();
  std::optional<BufferViewAnalysis> views;
  if (context && context->GetEnableBufferIR() && IsBufferIRSelected(func)) views.emplace(func->body_);
  ForbidAliasCollector forbid_collector(lifetimes.var_sharing_groups, views ? &*views : nullptr,
                                        coalescible_transfers);
  forbid_collector.VisitStmt(func->body_);
  result.forbid_alias = forbid_collector.Take();
  return result;
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const StmtPtr& func_body) {
  return AnalyzeAllocationLifetimesImpl(func_body);
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const FunctionPtr& func) {
  INTERNAL_CHECK(func != nullptr) << "Cannot analyze allocation lifetimes for a null function";
  return AnalyzeAllocationLifetimesImpl(func->body_, func->params_);
}

namespace pass {
Pass MaterializeSemanticAliases() {
  return CreateFunctionPass(TransformMaterializeSemanticAliases, "MaterializeSemanticAliases",
                            kMaterializeSemanticAliasesProperties);
}
Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, "MemoryReuse", kMemoryReuseProperties); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
