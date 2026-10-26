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
#include <array>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
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
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_ir_selection.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/storage_access_index.h"
#include "pypto/ir/transforms/utils/storage_reference_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

bool LoopMayExecute(const StmtPtr& statement, bool after_first) {
  if (auto loop = As<ForStmt>(statement)) {
    const auto start = As<ConstInt>(loop->start_);
    const auto stop = As<ConstInt>(loop->stop_);
    const auto step = As<ConstInt>(loop->step_);
    if (!start || !stop || !step || step->value_ == 0) return true;
    const __int128 current =
        static_cast<__int128>(start->value_) + (after_first ? static_cast<__int128>(step->value_) : 0);
    return step->value_ > 0 ? current < stop->value_ : current > stop->value_;
  }
  if (auto loop = As<WhileStmt>(statement)) {
    if (auto condition = As<ConstBool>(loop->condition_)) return condition->value_;
  }
  return true;
}

namespace {

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
  std::map<const Stmt*, std::set<VarPtr>> condition_reads;
  std::map<const Stmt*, std::set<VarPtr>> observation_reads;
  using BindingReads = std::map<VarPtr, std::vector<std::pair<const Stmt*, VarPtr>>>;
  BindingReads binding_reads;
  std::vector<StmtPtr> loops;  // For/While in lexical preorder.
  StmtPtr root;

  void Run(const StmtPtr& body);

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
      if (payload_reachable_ && WritesTileResult(As<Call>(op->value_))) {
        data_writes[op.get()].insert(op->var_);
      }
    }
    // Renames and zero-copy views bind storage without reading its payload.
    // Keep shape/offset operands as ordinary reads; defer only the source.
    if (As<TileType>(op->var_->GetType())) {
      if (auto source = AsVarLike(op->value_)) {
        if (payload_reachable_) binding_reads[op->var_].emplace_back(op.get(), source);
        reading_initializer_ = true;
        VisitExpr(op->value_);
        reading_initializer_ = false;
        return;
      }
      if (auto call = As<Call>(op->value_);
          call && call->op_ && op_predicates::IsBufferAliasingViewOp(call->op_->name_) &&
          OpRegistry::GetInstance().GetEntry(call->op_->name_).GetExecutionMemoryAccessEvidence() ==
              ExecutionMemoryAccessEvidence::NoAccess &&
          !call->args_.empty()) {
        if (auto source = AsVarLike(call->args_[0])) {
          if (payload_reachable_) binding_reads[op->var_].emplace_back(op.get(), source);
          reading_initializer_ = true;
          VisitExpr(call->args_[0]);
          reading_initializer_ = false;
          for (size_t i = 1; i < call->args_.size(); ++i) VisitExpr(call->args_[i]);
          // Debug/dependency attributes can independently consume payloads.
          for (const auto& [key, value] : call->attrs_) {
            (void)key;
            ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
          }
          return;
        }
      }
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
    binding_stack_.push_back(op);
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(op->else_body_.value());
    binding_stack_.pop_back();
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
    loops.push_back(loop);
    for (const auto& argument : loop->iter_args_) {
      reading_initializer_ = AsVarLike(argument->initValue_) != nullptr;
      VisitExpr(argument->initValue_);
    }
    reading_initializer_ = false;
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
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      reading_condition_ = true;
      VisitExpr(loop->condition_);
      reading_condition_ = false;
    }
    binding_stack_.push_back(loop);
    const bool saved_reachable = payload_reachable_;
    payload_reachable_ = payload_reachable_ && LoopMayExecute(loop, false);
    VisitStmt(loop->body_);
    payload_reachable_ = saved_reachable;
    binding_stack_.pop_back();
    loop_stack_.pop_back();
  }

  void VisitExpr_(const CallPtr& call) override {
    const auto assignment = As<AssignStmt>(current_stmt_);
    RecordInputWrites(call->op_, call->args_, assignment && assignment->value_ == call);
    auto& registry = OpRegistry::GetInstance();
    const auto* entry =
        call->op_ && registry.IsRegistered(call->op_->name_) ? &registry.GetEntry(call->op_->name_) : nullptr;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const bool saved_binding = reading_initializer_;
      // A write-only destination does not consume its previous payload. Still
      // traverse computed operands: evaluating their subexpressions can read.
      if (entry && entry->GetIRStage() == OpIRStage::Functional &&
          !ArgEffectReads(entry->GetArgEffect(i, call->kwargs_)) && AsVarLike(call->args_[i])) {
        reading_initializer_ = true;
      }
      VisitExpr(call->args_[i]);
      reading_initializer_ = saved_binding;
    }
    for (const auto& [key, value] : call->attrs_) {
      (void)key;
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
    }
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    RecordInputWrites(submit->op_, submit->args_);
    IRVisitor::VisitExpr_(submit);
  }

  void RecordInputWrites(const OpPtr& op, const std::vector<ExprPtr>& arguments,
                         bool direct_assignment = false) {
    if (!payload_reachable_) return;
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

  void VisitStmt_(const YieldStmtPtr& op) override {
    for (size_t i = 0; i < op->value_.size(); ++i) {
      auto source = AsVarLike(op->value_[i]);
      // Forwarding a handle only observes its payload when the bound result
      // or next iteration's carry is consumed. Computed yields still read it.
      reading_initializer_ = source != nullptr && !binding_stack_.empty();
      VisitExpr(op->value_[i]);
      reading_initializer_ = false;
      // Body reachability was established on entry; only a possible backedge
      // needs a separate check when registering the next carry binding.
      if (!payload_reachable_ || !source || binding_stack_.empty()) continue;
      auto bind = [&](const auto& owner) {
        if (i < owner->return_vars_.size()) {
          binding_reads[owner->return_vars_[i]].emplace_back(op.get(), source);
        }
      };
      auto bind_loop = [&](const auto& loop) {
        bind(loop);
        if (LoopMayExecute(loop, true) && i < loop->iter_args_.size()) {
          binding_reads[loop->iter_args_[i]].emplace_back(op.get(), source);
        }
      };
      const auto& owner = binding_stack_.back();
      if (auto branch = As<IfStmt>(owner)) {
        bind(branch);
      } else if (auto loop = As<ForStmt>(owner)) {
        bind_loop(loop);
      } else if (auto loop = As<WhileStmt>(owner)) {
        bind_loop(loop);
      }
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  void VisitVarLike_(const VarPtr& op) override {
    if (current_stmt_) reads[current_stmt_.get()].insert(op);
    if (payload_reachable_ && reading_condition_ && !reading_initializer_) {
      condition_reads[current_stmt_.get()].insert(op);
    }
    if (payload_reachable_ && !reading_initializer_ && current_stmt_) {
      observation_reads[current_stmt_.get()].insert(op);
    }
    IRVisitor::VisitVarLike_(op);
  }

 private:
  bool payload_reachable_ = true;
  bool reading_condition_ = false;
  bool reading_initializer_ = false;
  std::vector<StmtPtr> binding_stack_;
  StmtPtr current_stmt_;
  std::vector<StmtPtr> loop_stack_;
  size_t next_order_ = 0;
};

// Share payload demand with both legacy coalescing and Buffer IR loop-input
// isolation. Forwarding bindings must not become reads in either storage path.
std::map<const Stmt*, std::set<VarPtr>> BuildPayloadObservations(
    const std::map<VarPtr, VarDef>& defs, const std::map<const Stmt*, std::set<VarPtr>>& observation_reads,
    const DefMapVisitor::BindingReads& binding_reads) {
  auto effective_observations = observation_reads;
  std::set<VarPtr> observed_loop_values;
  std::vector<VarPtr> pending_loop_values;
  auto observe_loop_value = [&](const VarPtr& var) {
    const auto def = defs.find(var);
    if (def != defs.end() &&
        (def->second.kind == VarDef::kIterArg || def->second.kind == VarDef::kLoopReturn ||
         def->second.kind == VarDef::kIfReturn || binding_reads.count(var) != 0) &&
        observed_loop_values.insert(var).second) {
      pending_loop_values.push_back(var);
    }
  };
  for (const auto& [stmt, vars] : observation_reads) {
    (void)stmt;
    for (const auto& var : vars) observe_loop_value(var);
  }
  while (!pending_loop_values.empty()) {
    const auto value = pending_loop_values.back();
    pending_loop_values.pop_back();
    // Activate forwarding edges only on demand; an unused yield is not a
    // physical read. Each binding and edge is processed at most once.
    if (auto bindings = binding_reads.find(value); bindings != binding_reads.end()) {
      for (const auto& [stmt, source] : bindings->second) {
        effective_observations[stmt].insert(source);
        observe_loop_value(source);
      }
    }
    const auto& def = defs.at(value);
    IterArgPtr argument = def.iter_arg;
    if (def.kind == VarDef::kLoopReturn) {
      if (auto loop = As<ForStmt>(def.control_stmt)) {
        if (def.return_idx >= loop->iter_args_.size()) continue;
        auto start = As<ConstInt>(loop->start_);
        auto stop = As<ConstInt>(loop->stop_);
        auto step = As<ConstInt>(loop->step_);
        // Widen neither a difference nor trip count: compare bounds directly.
        if (start && stop && step &&
            ((step->value_ > 0 && start->value_ < stop->value_) ||
             (step->value_ < 0 && start->value_ > stop->value_))) {
          continue;
        }
        argument = loop->iter_args_[def.return_idx];
      } else if (auto loop = As<WhileStmt>(def.control_stmt)) {
        if (def.return_idx >= loop->iter_args_.size()) continue;
        argument = loop->iter_args_[def.return_idx];
      }
    }
    if (!argument) continue;
    if (auto initial = AsVarLike(argument->initValue_)) {
      effective_observations[def.definition_stmt.get()].insert(initial);
      observe_loop_value(initial);
    }
  }
  return effective_observations;
}

// Resolve forwarding demand once, then index only payload reads. The same
// facts feed loop isolation, exact window observations and future liveness.
void DefMapVisitor::Run(const StmtPtr& body) {
  root = body;
  VisitStmt(body);
  for (auto& [var, def] : defs) {
    (void)var;
    if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kLoopReturn) {
      def.order = stmt_end_order.at(def.control_stmt.get());
    }
  }
  reads = BuildPayloadObservations(defs, observation_reads, binding_reads);
  std::vector<std::pair<size_t, const std::set<VarPtr>*>> ordered_reads;
  ordered_reads.reserve(reads.size());
  for (const auto& [statement, values] : reads) {
    ordered_reads.emplace_back(stmt_order.at(statement), &values);
  }
  std::sort(ordered_reads.begin(), ordered_reads.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::vector<std::pair<size_t, const Stmt*>> active_loops;
  size_t next_loop = 0;
  for (const auto& [position, values] : ordered_reads) {
    // A loop initializer is outside its own backedge. Conditions are added
    // separately below because While re-evaluates them at its own header.
    while (next_loop < loops.size() && stmt_order.at(loops[next_loop].get()) < position) {
      const auto& loop = loops[next_loop++];
      if (!LoopMayExecute(loop, true)) continue;
      const size_t begin = stmt_order.at(loop.get());
      while (!active_loops.empty() && stmt_end_order.at(active_loops.back().second) < begin) {
        active_loops.pop_back();
      }
      active_loops.emplace_back(begin, loop.get());
    }
    while (!active_loops.empty() && stmt_end_order.at(active_loops.back().second) < position) {
      active_loops.pop_back();
    }
    for (const auto& value : *values) {
      value_last_use[value] = std::max(value_last_use[value], position);
      const auto definition = defs.find(value);
      const auto loop =
          definition == defs.end()
              ? active_loops.begin()
              : std::upper_bound(active_loops.begin(), active_loops.end(), definition->second.order,
                                 [](size_t order, const auto& candidate) { return order < candidate.first; });
      if (loop != active_loops.end()) recurring_reads[value].insert(loop->second);
    }
  }
  for (const auto& [loop, values] : condition_reads) {
    const auto* condition = static_cast<const WhileStmt*>(loop);
    if (const auto constant = As<ConstBool>(condition->condition_); constant && !constant->value_) {
      continue;
    }
    for (const auto& value : values) {
      const auto definition = defs.find(value);
      if (definition == defs.end() || definition->second.order < stmt_order.at(loop)) {
        recurring_reads[value].insert(loop);
      }
    }
  }
  for (const auto& [value, enclosing] : recurring_reads) {
    for (const auto* loop : enclosing) {
      value_last_use[value] = std::max(value_last_use[value], stmt_end_order.at(loop));
    }
  }
}

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
  std::set<VarPtr> reference_carries;

  /// Collect one immutable snapshot of actual writes and possible carry
  /// writebacks. The same proof drives isolation and result mutability; merely
  /// entering a loop does not write its initializer. Use a fresh instance for
  /// each snapshot, since projected windows and value versions belong to it.
  void AnalyzeEffects(const StmtPtr& body) {
    DefMapVisitor demand;
    demand.Run(body);
    payload_observations_ = std::move(demand.reads);
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
      if (!LoopMayExecute(loop, false)) return;
      for (const auto& argument : loop->iter_args_) {
        if (!GetTileTypeWithMemRef(argument->GetType())) continue;
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
    const auto* saved_statement = current_statement_;
    current_statement_ = statement.get();
    auto saved = current_;
    const auto saved_reverse = current_reverse_;
    current_reverse_ = reverse_order_.at(statement.get()) + 1;
    current_ = ++order_;
    positions_[statement.get()] = current_;
    IRVisitor::VisitStmt(statement);
    current_ = saved;
    current_reverse_ = saved_reverse;
    current_statement_ = saved_statement;
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
        // data reads or nested Call/Submit effects and must be visited. Later
        // demanded consumers observe this view's inherited value version and
        // exact projected window, without a whole-parent read at its binding.
        if (call) {
          RecordInputWrites(call->op_, call->args_);
          for (size_t i = 1; i < call->args_.size(); ++i) VisitExpr(call->args_[i]);
          for (const auto& [key, value] : call->attrs_) {
            (void)key;
            ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpr(expression); });
          }
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
      const bool same_value = lhs_tile && rhs_tile && ValueOrigin(lhs) == ValueOrigin(rhs) &&
                              SameKnownTileMemorySpace(tile, lhs_tile) &&
                              SameKnownTileMemorySpace(tile, rhs_tile);
      // This is a value-version projection for effect queries, not a physical
      // result binding. Pure renames/views retain their origin; native writers
      // start fresh origins and cannot make an updated/unchanged join a no-op.
      if (same_value && (!fixed || SamePhysicalWindow(target, StorageWindow(lhs)))) {
        const auto origin = ValueOrigin(lhs);
        value_origins_[result.get()] = origin;
        const auto lhs_definition = definitions_.find(lhs.get());
        const auto rhs_definition = definitions_.find(rhs.get());
        definitions_[result.get()] =
            std::min(lhs_definition == definitions_.end() ? 0 : lhs_definition->second,
                     rhs_definition == definitions_.end() ? 0 : rhs_definition->second);
      }
      const bool common = same_value && SamePhysicalWindow(StorageWindow(lhs), StorageWindow(rhs));
      if (common && (!fixed || SamePhysicalWindow(target, StorageWindow(lhs)))) {
        storage_windows_[result.get()] = StorageWindow(lhs);
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
  const Stmt* current_statement_ = nullptr;
  bool effects_reachable_ = true;
  std::map<const Stmt*, std::set<VarPtr>> payload_observations_;

  [[nodiscard]] bool MayWriteInLoop(const StmtPtr& loop, const MemRefPtr& window) const {
    return writes_index_.MayOverlapAfter(window, positions_.at(loop.get()), reverse_order_.at(loop.get()),
                                         loop_ends_.at(loop.get()));
  }

  void RecordRead(const VarPtr& value, const Var* excluded_slot) {
    const auto observation = payload_observations_.find(current_statement_);
    if (observation == payload_observations_.end() || observation->second.count(value) == 0) return;
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
    if (!effects_reachable_) return;
    ForEachStorageWindow(value, [&](const MemRefPtr& window) {
      write_events_.push_back({window, primary, reverse});
      ++write_generations_[window->base_.get()];
    });
  }
  void RecordWrite(const VarPtr& value) { RecordWrite(value, current_, current_reverse_); }
  void RecordInputWrites(const OpPtr& op, const std::vector<ExprPtr>& arguments) {
    if (!effects_reachable_) return;
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
    const bool repeats = LoopMayExecute(loop, true);
    if (repeats) loop_stack_.emplace_back(positions_.at(loop.get()), loop.get());
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      // Separate condition reads from initializer reads at the loop header.
      current_ = ++order_;
      VisitExpr(loop->condition_);
    }
    const bool saved_reachable = effects_reachable_;
    const bool may_enter = LoopMayExecute(loop, false);
    effects_reachable_ = effects_reachable_ && may_enter;
    VisitStmt(loop->body_);
    effects_reachable_ = saved_reachable;
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    std::vector<bool> unchanged_values;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      const auto source = yield && i < yield->value_.size() ? AsVarLike(yield->value_[i]) : nullptr;
      const bool unchanged = !may_enter || (source && ValueOrigin(source) == ValueOrigin(argument));
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
    if (repeats) loop_stack_.pop_back();
    loop_ends_[loop.get()] = order_;
    for (const auto& result : loop->return_vars_) definitions_[result.get()] = ++order_;
  }

  template <typename LoopPtr>
  void PlanLoop(const LoopPtr& loop) {
    if (!LoopMayExecute(loop, false)) return;
    const auto position = positions_.at(loop.get());
    std::set<size_t> isolate;
    std::vector<StorageAccessIndex::Event> carry_windows;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      if (!GetTileTypeWithMemRef(loop->iter_args_[i]->GetType())) continue;
      ForEachStorageWindow(loop->iter_args_[i]->initValue_,
                           [&](const MemRefPtr& window) { carry_windows.push_back({window, i + 1, 1}); });
    }
    StorageAccessIndex other_carries;
    other_carries.Build(carry_windows);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      // A reference carry changes its handle on the backedge. It has no
      // destination window to isolate or to copy into; actual native writes
      // through that handle are checked by storage-reference validation.
      if (!GetTileTypeWithMemRef(loop->iter_args_[i]->GetType())) continue;
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
      if (memory && !IsTileMoveEverSupported(*memory, *memory)) {
        // Entry isolation need not copy a read-only initializer into a mutable
        // destination. A carry can instead select its initializer at entry and
        // independently produced storage on each backedge. Actual writes
        // through this handle still require the shared old-version proof;
        // turning the binding into a reference cannot authorize clobbering an
        // independently observed initializer or previous carried version.
        reference_carries.insert(argument);
        continue;
      }
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
    // A reference may still read any incoming allocation after this selection.
    // Keep its actual owners stable while planning other branch destinations;
    // this is protection, not mandatory coalescing of the source allocations.
    const auto references = AnalyzeStorageReferences(func_body, function->params_);
    reference_source_bases_ = references.source_bases;
    immovable_branch_bases_.insert(references.source_bases.begin(), references.source_bases.end());
    std::vector<StorageAccessIndex::Event> readonly_caller_windows;
    for (size_t i = 0; i < function->params_.size(); ++i) {
      const auto& parameter = function->params_[i];
      if (const auto memory = GetTypeMemRef(parameter->GetType());
          function->param_directions_[i] == ParamDirection::In && memory && *memory) {
        readonly_caller_windows.push_back({*memory, 1, 1});
      }
      // A Ptr ABI parameter can name a directly supplied result window. Its
      // binding is caller-owned, just like the storage of a Tile parameter.
      if (As<PtrType>(parameter->GetType())) {
        pinned_branch_bases_.insert(parameter.get());
      }
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
  WindowedReachableEventIndex recurring_access_index_;
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
  std::set<const Var*> reference_source_bases_;
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
    if (LoopMayExecute(loop, false)) {
      branch_loop_headers_.push_back(stmt_order_.at(loop.get()));
      VisitBranchStorage(loop->body_, allocations, select_results);
      branch_loop_headers_.pop_back();
    }
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
    std::vector<WindowedReachableEventIndex::Event> recurring_accesses;
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
      for (const auto* loop : loops) {
        record(loop, value, true, true);
        if (const auto tile = GetTileTypeWithMemRef(value->GetType())) {
          recurring_accesses.push_back(
              {GetDefinedMemRef(tile), stmt_order_.at(loop), stmt_end_order_.at(loop) + 1});
        }
      }
    }
    for (auto& [component, events] : physical_events) {
      physical_access_index_[component].Build(events);
    }
    physical_read_index_.Build(physical_reads);
    reverse_physical_read_index_.Build(reverse_physical_reads);
    recurring_access_index_.Build(recurring_accesses);
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
      if (!LoopMayExecute(for_stmt, false)) return;
      PropagateFromLoop(for_stmt);
      VisitLoopConstraints(for_stmt->body_);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      if (!LoopMayExecute(loop, false)) return;
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
      if (!LoopMayExecute(for_stmt, false)) return;
      VisitIfPhisForAccumulator(for_stmt->body_);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      if (!LoopMayExecute(loop, false)) return;
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
      // A selected handle is already a complete representation plan. Native
      // accumulator coalescing must not resurrect its abandoned destination
      // or overwrite one of the independent selected source allocations.
      if (!GetTileTypeWithMemRef(if_stmt->return_vars_[i]->GetType())) continue;
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
    if (As<TileType>(var->GetType()) && !GetTileTypeWithMemRef(var->GetType())) return false;
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
    // Existing selected owners can retain native required aliases on their
    // current windows. A new producer cannot become an additional occupant of
    // such a window: the handle may still select its original value later.
    // Apply this at the actual producer, not at forwarding If/loop wrappers,
    // whose existing common-window and old-observer proofs remain authoritative.
    if (reference_source_bases_.count(target->base_.get()) &&
        !SamePhysicalWindow(CurrentMemRef(var), target)) {
      return false;
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

    if (!LoopMayExecute(for_stmt, false)) {
      if (idx >= for_stmt->iter_args_.size()) return false;
      const auto& argument = for_stmt->iter_args_[idx];
      const auto initial = AsVarLike(argument->initValue_);
      if (!initial || !TryRetargetVar(initial, target, target_memory)) return false;
      PlanRewrite(argument, target, target_memory);
      PlanRewrite(var, target, target_memory);
      return true;
    }

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
    // A later enclosing iteration can observe the old value beyond the
    // nearest inner-loop/branch boundary. Check those demanded recurrence
    // intervals first, retaining exact byte/slot geometry and event pairing.
    if (recurring_access_index_.HasAfter(target, 0, def.order, def.order)) return false;
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

  std::set<const Var*> RemapReferenceCarries(const std::set<VarPtr>& carries) const {
    std::set<const Var*> result;
    for (const auto& carry : carries) {
      const auto found = var_substitution_.find(carry);
      result.insert(found == var_substitution_.end() ? carry.get() : found->second.get());
    }
    return result;
  }

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

}  // namespace

// Retyping also propagates a retargeted input through its aliasing views.
StmtPtr RefreshStorageTypes(const StmtPtr& body) { return RetypeApplier({}).VisitStmt(body); }

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

  // Resolve required aliases before classifying independent incoming storage.
  // InitMemRef may leave a rename/view with a provisional allocation request;
  // that request must not turn a readonly same-window carry into a selector.
  StmtPtr new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(func->body_);
  new_body = RetypeApplier({}).VisitStmt(new_body);
  new_body = NormalizeStorageReferences(new_body, func->params_);
  ValidateStorageReferenceWrites(new_body, func->params_, func->param_directions_);
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
  std::set<const Var*> reference_carries;
  if (!isolation.initializers.empty()) {
    // Consume entries while loop statement identities still match the frozen
    // isolation plan. A selected carry in the same loop must not rebuild that
    // loop before its independent copyable carries receive their entry copies.
    RetypeApplier applier(std::move(isolation.rewrites), std::move(isolation.initializers),
                          std::move(isolation.entries));
    new_body = applier.VisitStmt(new_body);
    reference_carries = applier.RemapReferenceCarries(isolation.reference_carries);
    new_body = InsertAllocsIntoBody(new_body, isolation.allocations);
  } else {
    for (const auto& carry : isolation.reference_carries) reference_carries.insert(carry.get());
  }
  if (!reference_carries.empty()) {
    // Keep concrete bindings until the shared graph has recognized every
    // inherited alias. Clearing a carry first would make its native writeback's
    // inherited window look like an explicit result destination.
    new_body = NormalizeStorageReferences(new_body, func->params_, reference_carries);
    ValidateStorageReferenceWrites(new_body, func->params_, func->param_directions_);
  }
  new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);
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
  // Native Acc updates still have value-version obligations when their joins
  // remain concrete. Check them after destination planning, before an illegal
  // Acc-to-Acc edge repair can obscure the independently observed old value.
  ValidateStorageReferenceWrites(new_body, func->params_, func->param_directions_,
                                 /*include_concrete_acc_regions=*/true);

  // Tile PYPTO still reconciles loop writebacks after packing. If edges must
  // already be complete so every planner sees their actual reads and writes.
  const bool all_boundaries =
      ctx != nullptr && (ctx->GetEnableBufferIR() || ctx->GetMemoryPlanner() != MemoryPlanner::PyPTO);
  new_body = FixupYields(new_body, func->params_, /*canonical_loop_copies=*/true,
                         /*if_results_only=*/!all_boundaries);
  new_body = NormalizeIdentityCopyBuffers(new_body);
  // Fixed destinations can introduce transfers during boundary repair. Check
  // their writes too; the pre-planning proof alone cannot cover new effects.
  ValidateStorageReferenceWrites(new_body, func->params_, func->param_directions_);
  new_body = RemoveUnusedAllocStatements(new_body, memref_collectors::CollectUsedBasePtrs(new_body));

  if (new_body == func->body_) return func;

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_,
                                          func->requires_runtime_binding_, func->ir_stage_);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto
