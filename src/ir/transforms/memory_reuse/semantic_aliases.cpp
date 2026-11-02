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
// are later in exactly one and are therefore excluded. Read/write events are
// indexed by overlapping physical-window component, so each query is O(log N)
// and the complete retargeting analysis remains O(N log N).
// ============================================================================

// Preserve an assemble source before its destination can overwrite the same
// storage family. The copy is explicit before planning, so every planner sees
// its lifetime, capacity and no-alias constraints. Exact self-writes stay views.
class SnapshotAssembleSources : public IRMutator {
 public:
  explicit SnapshotAssembleSources(const FunctionPtr& function) : views_(function->body_) {
    names_.Collect(function->body_, function->params_);
  }

  std::vector<StmtPtr> allocations;

 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override { return argument; }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    const auto call = As<Call>(assign->value_);
    if (call && IsOp(call, "tile.slice")) return IndexWindowOffsets(assign, call);
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
      const auto offsets = RootOffsets(view, call->span_);
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
  std::vector<ExprPtr> RootOffsets(const BufferViewAnalysis::View& view, const Span& span) const {
    if (view.offsets) {
      return {std::make_shared<ConstInt>((*view.offsets)[0], DataType::INDEX, span),
              std::make_shared<ConstInt>((*view.offsets)[1], DataType::INDEX, span)};
    }
    const auto found = root_offsets_.find(view.slice.get());
    INTERNAL_CHECK_SPAN(found != root_offsets_.end(), span)
        << "Internal error: runtime window offsets must dominate their snapshot";
    return found->second;
  }

  StmtPtr IndexWindowOffsets(const AssignStmtPtr& assign, const CallPtr& call) {
    const auto view = views_.Get(assign->var_);
    if (view.offsets || !view.root) return assign;
    const auto parent = RootOffsets(views_.Get(call->args_[0]), call->span_);
    const auto offsets = As<MakeTuple>(call->args_[2]);
    INTERNAL_CHECK_SPAN(offsets && offsets->elements_.size() == 2, call->span_)
        << "Internal error: rank-2 slice offsets required";
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
  std::unordered_map<const Call*, std::vector<ExprPtr>> root_offsets_;
};

/// Describes where a TileType Var is defined.
struct VarDef {
  enum Kind { kAssign, kIfReturn, kForReturn, kWhileReturn, kIterArg, kUnknown };
  Kind kind = kUnknown;
  StmtPtr assign_stmt;   // AssignStmt (for kAssign)
  StmtPtr control_stmt;  // IfStmt/ForStmt/WhileStmt defining a control-flow result
  StmtPtr definition_stmt;
  ForStmtPtr nearest_for;
  size_t return_idx = 0;  // index into control-flow return_vars_ or iter_args_
  IterArgPtr iter_arg;    // for kIterArg
  size_t order = 0;       // lexical statement order of the definition
};

// Known bounds exclude dead bodies and loops without a backedge. Unknown
// conditions remain conservative; widen arithmetic before advancing a For.

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
  std::map<const Stmt*, std::set<VarPtr>> condition_reads;
  std::map<const Stmt*, std::set<VarPtr>> observation_reads;
  using BindingReads = std::map<VarPtr, std::vector<std::pair<const Stmt*, VarPtr>>>;
  BindingReads binding_reads;
  std::vector<StmtPtr> loops;  // ForStmt/WhileStmt in lexical preorder
  StmtPtr root;

  void Run(const StmtPtr& body) {
    root = body;
    VisitStmt(body);
    for (auto& [var, def] : defs) {
      (void)var;
      if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kForReturn ||
          def.kind == VarDef::kWhileReturn) {
        def.order = stmt_end_order.at(def.control_stmt.get());
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
      d.nearest_for = for_stack_.empty() ? nullptr : for_stack_.back();
      d.order = stmt_order.at(op.get());
      defs[op->var_] = d;
      writes[op.get()].insert(op->var_);
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
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIfReturn;
      d.control_stmt = op;
      d.definition_stmt = op;
      d.nearest_for = for_stack_.empty() ? nullptr : for_stack_.back();
      d.return_idx = i;
      d.order = stmt_order.at(op.get());
      defs[rv] = d;
    }
    binding_stack_.push_back(op);
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(op->else_body_.value());
    binding_stack_.pop_back();
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    loops.push_back(op);
    VisitExpr(op->loop_var_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    VisitLoopInitializers(op);

    RegisterLoopBindings(op, VarDef::kForReturn, op);
    for_stack_.push_back(op);
    binding_stack_.push_back(op);
    const bool saved_reachable = payload_reachable_;
    payload_reachable_ = payload_reachable_ && LoopMayExecute(op, false);
    VisitStmt(op->body_);
    payload_reachable_ = saved_reachable;
    binding_stack_.pop_back();
    for_stack_.pop_back();
  }

  void VisitExpr_(const CallPtr& call) override {
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

  void VisitExpr_(const IterArgPtr& argument) override {
    // The initializer is read at loop entry. A bound carry reads the current
    // iteration's value, not its initializer's old logical version again.
    VisitVarLike_(argument);
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    loops.push_back(loop);
    VisitLoopInitializers(loop);
    RegisterLoopBindings(loop, VarDef::kWhileReturn, for_stack_.empty() ? nullptr : for_stack_.back());
    reading_condition_ = true;
    VisitExpr(loop->condition_);
    reading_condition_ = false;
    binding_stack_.push_back(loop);
    const bool saved_reachable = payload_reachable_;
    payload_reachable_ = payload_reachable_ && LoopMayExecute(loop, false);
    VisitStmt(loop->body_);
    payload_reachable_ = saved_reachable;
    binding_stack_.pop_back();
  }

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
  template <typename LoopPtr>
  void VisitLoopInitializers(const LoopPtr& loop) {
    for (const auto& argument : loop->iter_args_) {
      reading_initializer_ = AsVarLike(argument->initValue_) != nullptr;
      VisitExpr(argument->initValue_);
    }
    reading_initializer_ = false;
  }

  template <typename LoopPtr>
  void RegisterLoopBindings(const LoopPtr& loop, VarDef::Kind return_kind, const ForStmtPtr& nearest_for) {
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (!As<TileType>(argument->GetType())) continue;
      VarDef def;
      def.kind = VarDef::kIterArg;
      def.control_stmt = loop;
      def.definition_stmt = loop;
      def.nearest_for = nearest_for;
      def.return_idx = i;
      def.iter_arg = argument;
      def.order = stmt_order.at(loop.get());
      defs[argument] = def;
    }
    for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
      const auto& result = loop->return_vars_[i];
      if (!As<TileType>(result->GetType())) continue;
      VarDef def;
      def.kind = return_kind;
      def.control_stmt = loop;
      def.definition_stmt = loop;
      def.nearest_for = nearest_for;
      def.return_idx = i;
      def.order = stmt_order.at(loop.get());
      defs[result] = def;
    }
  }

  StmtPtr current_stmt_;
  bool payload_reachable_ = true;
  bool reading_condition_ = false;
  bool reading_initializer_ = false;
  std::vector<ForStmtPtr> for_stack_;
  std::vector<StmtPtr> binding_stack_;
  size_t next_order_ = 0;
};

// Plan uniform compiler-owned storage selections before YieldFixup. The plan
// separates slot ownership/lifetime decisions from the structured-SSA rewrite.
// Definitions and binding edges are indexed once; union/find and cached slot
// resolution keep the analysis O(N log N). Loop storage retains its existing
// path; this repair only selects between two existing producer windows.
StorageSelectionPlan BuildStorageSelectionPlan(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
  DefMapVisitor index;
  index.Run(body);
  StorageNameSupply names;
  names.Collect(body, parameters);
  struct StorageContracts : IRVisitor {
    std::set<const Var*> bases;
    std::set<const Var*> written_values;
    void VisitExpr_(const IterArgPtr&) override {}
    // Protect every window touched by a loop. Do not recurse again here:
    // collecting each outermost loop subtree once keeps the scan linear.
    void ProtectLoop(const StmtPtr& loop) {
      struct LoopWindows : memref_collectors::MemRefWithSpaceCollector {
        std::set<const IterArg*> seen;
        void VisitExpr_(const IterArgPtr& argument) override {
          if (seen.insert(argument.get()).second) IRVisitor::VisitExpr_(argument);
        }
      } windows;
      windows.VisitStmt(loop);
      for (const auto& entry : windows.memrefs) bases.insert(entry.first->base_.get());
    }
    void VisitStmt_(const ForStmtPtr& loop) override { ProtectLoop(loop); }
    void VisitStmt_(const WhileStmtPtr& loop) override { ProtectLoop(loop); }
    void VisitExpr_(const CallPtr& call) override {
      if (const auto target = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) {
        if (const auto value = AsVarLike(call->args_[*target])) written_values.insert(value.get());
      }
      IRVisitor::VisitExpr_(call);
    }
    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (const auto call = As<Call>(assign->value_); call && IsOp(call, "tile.alloc")) {
        for (const auto& [name, value] : call->kwargs_) {
          if (name == "pinned" && AnyCast<bool>(value, "pinned")) bases.insert(assign->var_.get());
        }
      }
      IRVisitor::VisitStmt_(assign);
    }
  } contracts;
  for (const auto& parameter : parameters) {
    if (const auto memory = GetTypeMemRef(parameter->GetType())) {
      contracts.bases.insert((*memory)->base_.get());
    }
  }
  contracts.VisitStmt(body);

  std::map<const Var*, const Var*> parents;
  std::map<const Var*, size_t> ranks;
  std::function<const Var*(const Var*)> find = [&](const Var* value) -> const Var* {
    auto [position, fresh] = parents.emplace(value, value);
    if (!fresh && position->second != value) position->second = find(position->second);
    return position->second;
  };
  auto join = [&](const Var* a, const Var* b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (ranks[a] < ranks[b]) std::swap(a, b);
    parents[b] = a;
    if (ranks[a] == ranks[b]) ++ranks[a];
  };
  for (const auto& [value, bindings] : index.binding_reads) {
    for (const auto& [statement, source] : bindings) {
      (void)statement;
      join(value.get(), source.get());
    }
  }
  std::map<const Var*, std::vector<VarPtr>> components;
  std::map<const Var*, const Var*> storage_owners;
  for (const auto& [value, definition] : index.defs) {
    (void)definition;
    const auto owner = find(value.get());
    components[owner].push_back(value);
    if (const auto memory = GetTypeMemRef(value->GetType())) {
      const auto* base = (*memory)->base_.get();
      const auto [position, fresh] = storage_owners.emplace(base, owner);
      // Resolved aliases can share a window without an SSA binding edge. Do
      // not migrate one family while leaving such another owner behind.
      if (!fresh && position->second != owner) contracts.bases.insert(base);
    }
  }

  StorageSelectionPlan plan;
  const auto integer = [](uint64_t value, const Span& span) -> ExprPtr {
    return std::make_shared<ConstInt>(static_cast<int64_t>(value), DataType::INDEX, span);
  };
  // A branch-local SSA rename can still select an incoming value, as in
  // `else: selected = b`. Cache the origin order across rename chains.
  std::map<const Var*, size_t> origin_orders;
  std::function<size_t(const VarPtr&)> origin_order = [&](const VarPtr& value) -> size_t {
    const auto [position, fresh] = origin_orders.emplace(value.get(), std::numeric_limits<size_t>::max());
    if (!fresh) return position->second;
    const auto found = index.defs.find(value);
    if (found == index.defs.end()) return position->second;
    const auto& definition = found->second;
    position->second = definition.order;
    if (definition.kind == VarDef::kAssign) {
      if (const auto source = AsVarLike(As<AssignStmt>(definition.assign_stmt)->value_)) {
        position->second = origin_order(source);
      }
    }
    return position->second;
  };
  for (const auto& [root, members] : components) {
    (void)root;
    bool valid = true;
    bool selection = false;
    bool divergent = false;
    MemRefPtr first;
    TypePtr signature;
    std::vector<VarPtr> producers;
    for (const auto& value : members) {
      const auto& definition = index.defs.at(value);
      if (definition.kind != VarDef::kAssign && definition.kind != VarDef::kIfReturn) {
        valid = false;
        break;
      }
      const auto tile = GetTileTypeWithMemRef(value->GetType());
      if (!tile || !tile->GetMemorySpace() || tile->shape_.empty()) {
        valid = false;
        break;
      }
      const auto static_extent = [](const ExprPtr& extent) {
        const auto constant = As<ConstInt>(extent);
        return constant && constant->value_ > 0;
      };
      const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
      // Only full implicit descriptors can be reconstructed by tile.create.
      // Keep other descriptor adaptations outside this control-flow repair.
      if (view != tile_view_semantics::GetImplicitTileView(tile->shape_, tile->GetMemorySpace()) ||
          (*tile->GetMemorySpace() != MemorySpace::Vec && *tile->GetMemorySpace() != MemorySpace::Mat &&
           *tile->GetMemorySpace() != MemorySpace::Acc)) {
        valid = false;
        break;
      }
      // This lowering selects storage, not geometry. Dynamic validity and
      // strided views need an explicit descriptor contract beyond slot choice.
      const auto memory = GetDefinedMemRef(tile);
      if (!std::all_of(tile->shape_.begin(), tile->shape_.end(), static_extent) || memory->slot_count_ != 1 ||
          contracts.bases.count(memory->base_.get()) || contracts.written_values.count(value.get()) ||
          !As<ConstInt>(memory->byte_offset_) || As<ConstInt>(memory->byte_offset_)->value_ != 0) {
        valid = false;
        break;
      }
      const auto stripped = CloneTypeWithoutMemRef(tile);
      if (signature && (!structural_equal(signature, stripped) || first->size_ != memory->size_)) {
        valid = false;
        break;
      }
      signature = stripped;
      if (!first) first = memory;
      divergent |= !SamePhysicalWindow(first, memory);
      if (definition.kind == VarDef::kAssign) {
        const auto assign = As<AssignStmt>(definition.assign_stmt);
        if (const auto source = AsVarLike(assign->value_)) {
          valid &= index.defs.count(source) != 0;
        } else {
          const auto call = As<Call>(assign->value_);
          auto& registry = OpRegistry::GetInstance();
          // Only a registered fresh output may choose another allocation.
          // In-place/writeback results and NoAccess views retain their source
          // storage; an ordinary SSA rename was handled above.
          if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_) ||
              op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
            valid = false;
            break;
          }
          producers.push_back(value);
        }
      } else {
        if (definition.kind == VarDef::kIfReturn) {
          // Fresh branch-local results retain the existing packing/retargeting
          // path. A pool is needed when an arm selects an already existing value.
          if (const auto found = index.binding_reads.find(value); found != index.binding_reads.end()) {
            for (const auto& [statement, source] : found->second) {
              (void)statement;
              selection |= origin_order(source) < index.stmt_order.at(definition.control_stmt.get());
            }
          }
        }
      }
    }
    for (const auto& value : members) {
      if (const auto found = index.binding_reads.find(value); found != index.binding_reads.end()) {
        for (const auto& [statement, source] : found->second) {
          (void)statement;
          if (!index.defs.count(source)) valid = false;  // Borrowed source.
        }
      }
    }
    if (!valid || !selection || !divergent || producers.size() != 2) continue;
    std::sort(producers.begin(), producers.end(), [&](const VarPtr& a, const VarPtr& b) {
      return index.defs.at(a).order < index.defs.at(b).order;
    });
    constexpr uint64_t count = 2;
    if (first->size_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / count) continue;
    std::map<const Var*, uint64_t> producer_slots;
    for (size_t i = 0; i < producers.size(); ++i) producer_slots[producers[i].get()] = i;
    const auto bytes = first->size_;
    const auto base =
        std::make_shared<Var>(names.Fresh("mem_storage_selection_"), GetPtrType(), first->span_);
    const auto allocation = std::make_shared<MemRef>(base, int64_t{0}, bytes * count, first->span_);
    plan.allocations.push_back(
        CreateAllocStatement(allocation, *As<TileType>(members.front()->GetType())->GetMemorySpace()));
    std::function<ExprPtr(const VarPtr&)> slot = [&](const VarPtr& value) -> ExprPtr {
      if (const auto found = plan.indices.find(value.get()); found != plan.indices.end()) {
        return found->second;
      }
      const auto& definition = index.defs.at(value);
      ExprPtr result;
      if (definition.kind == VarDef::kAssign) {
        if (const auto source = AsVarLike(As<AssignStmt>(definition.assign_stmt)->value_)) {
          result = slot(source);
        } else {
          result = integer(producer_slots.at(value.get()), value->span_);
        }
      } else {
        result = std::make_shared<Var>(names.Fresh(value->name_hint_ + "_slot_"),
                                       std::make_shared<ScalarType>(DataType::INDEX), value->span_);
      }
      plan.indices[value.get()] = result;
      return result;
    };
    for (const auto& value : members) {
      const auto selected = slot(value);
      auto memory = std::make_shared<MemRef>(base, MulByteOffsets(selected, integer(bytes, value->span_)),
                                             bytes, value->span_, false, count, selected);
      auto tile = std::make_shared<Var>(value->name_hint_, CloneTypeWithMemRef(value->GetType(), memory),
                                        value->span_);
      plan.indices[tile.get()] = selected;
      plan.tiles[value.get()] = std::move(tile);
    }
  }
  return plan;
}

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
        (def->second.kind == VarDef::kIterArg || def->second.kind == VarDef::kForReturn ||
         def->second.kind == VarDef::kWhileReturn || def->second.kind == VarDef::kIfReturn ||
         binding_reads.count(var) != 0) &&
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
    if (def.kind == VarDef::kForReturn) {
      auto loop = As<ForStmt>(def.control_stmt);
      if (!loop || def.return_idx >= loop->iter_args_.size()) continue;
      auto start = As<ConstInt>(loop->start_);
      auto stop = As<ConstInt>(loop->stop_);
      auto step = As<ConstInt>(loop->step_);
      // Compare bounds directly: computing stop-start or a trip count could
      // overflow. Unknown bounds conservatively retain the zero-trip path.
      if (start && stop && step &&
          ((step->value_ > 0 && start->value_ < stop->value_) ||
           (step->value_ < 0 && start->value_ > stop->value_))) {
        continue;
      }
      argument = loop->iter_args_[def.return_idx];
    } else if (def.kind == VarDef::kWhileReturn) {
      auto loop = As<WhileStmt>(def.control_stmt);
      if (!loop || def.return_idx >= loop->iter_args_.size()) continue;
      argument = loop->iter_args_[def.return_idx];
    }
    if (!argument) continue;
    if (auto initial = AsVarLike(argument->initValue_)) {
      effective_observations[def.definition_stmt.get()].insert(initial);
      observe_loop_value(initial);
    }
  }
  return effective_observations;
}

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

/// Visits a stmt subtree and collects the MemRef base Ptrs of every *read*
/// of a TileType Var.  Writes (the LHS of an AssignStmt) are intentionally
/// excluded; every other stmt/expression kind dispatches through the default
/// IRVisitor traversal, so new stmt types are covered automatically.
class ExprReadBaseCollector : public IRVisitor {
 public:
  std::set<const Var*> bases;

  void VisitVarLike_(const VarPtr& var) override {
    if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
      bases.insert(GetDefinedMemRef(tile)->base_.get());
    }
    IRVisitor::VisitVarLike_(var);
  }

  void VisitExpr_(const IterArgPtr& argument) override {
    const auto* context = PassContext::Current();
    if (context && context->GetEnableBufferIR()) {
      VisitVarLike_(argument);
    } else {
      IRVisitor::VisitExpr_(argument);
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Skip op->var_ — the LHS is a definition, not a read.
    if (op->value_) VisitExpr(op->value_);
  }
};

/// Range-maximum index over events (primary lexical order, reverse-branch
/// lexical order). A query succeeds when an event lies in the primary interval
/// and is also after the lower bound in the reverse ordering.
class ReachableEventIndex {
 public:
  void Build(std::vector<std::pair<size_t, size_t>> events, bool active = true) {
    if (events.empty()) return;
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    size_ = 1;
    while (size_ < events.size()) size_ *= 2;
    primary_.reserve(events.size());
    max_reverse_.assign(size_ * 2, 0);
    for (size_t i = 0; i < events.size(); ++i) {
      primary_.push_back(events[i].first);
      max_reverse_[size_ + i] = active ? events[i].second + 1 : 0;
    }
    for (size_t i = size_; i-- > 1;) {
      max_reverse_[i] = std::max(max_reverse_[i * 2], max_reverse_[i * 2 + 1]);
    }
  }

  // Activate observations in logical-definition order for loop-entry queries.
  void Activate(size_t primary, size_t reverse) {
    size_t node = size_ + static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary) -
                                              primary_.begin());
    max_reverse_[node] = std::max(max_reverse_[node], reverse + 1);
    while (node > 1) {
      node /= 2;
      max_reverse_[node] = std::max(max_reverse_[node * 2], max_reverse_[node * 2 + 1]);
    }
  }

  [[nodiscard]] bool HasAfter(size_t primary_begin, size_t primary_end, size_t reverse_lower) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    size_t left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                      primary_.begin());
    size_t right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                       primary_.begin());
    left += size_;
    right += size_;
    size_t maximum = 0;
    while (left < right) {
      if (left & 1U) maximum = std::max(maximum, max_reverse_[left++]);
      if (right & 1U) maximum = std::max(maximum, max_reverse_[--right]);
      left /= 2;
      right /= 2;
    }
    return maximum > reverse_lower + 1;
  }

 private:
  std::vector<size_t> primary_;
  std::vector<size_t> max_reverse_;
  size_t size_ = 0;
};

/// Range-minimum index over read events and their logical definition versions.
/// Aliases can share a physical allocation while naming different values over
/// time (notably an IterArg's initializer and the corresponding ForStmt
/// return). A post-If read is an independent observation of the old value only
/// when the handle being read was defined before that IfStmt.
///
/// These queries begin after the complete IfStmt subtree, where primary lexical
/// order alone proves reachability (both arms precede the suffix in either
/// branch ordering). The index therefore stores the minimum handle-definition
/// order per primary interval. Build is O(N log N), each query is O(log N), and
/// the complete pass remains O(N log N).
class VersionedReachableEventIndex {
 public:
  struct Event {
    size_t primary;
    size_t definition;
  };

  void Build(std::vector<Event> events) {
    if (events.empty()) return;
    std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
      return std::tie(lhs.primary, lhs.definition) < std::tie(rhs.primary, rhs.definition);
    });

    primary_.reserve(events.size());
    std::vector<size_t> leaf_minimum;
    leaf_minimum.reserve(events.size());
    for (size_t i = 0; i < events.size();) {
      primary_.push_back(events[i].primary);
      size_t minimum = events[i].definition;
      size_t j = i + 1;
      while (j < events.size() && events[j].primary == events[i].primary) {
        minimum = std::min(minimum, events[j].definition);
        ++j;
      }
      leaf_minimum.push_back(minimum);
      i = j;
    }

    size_ = 1;
    while (size_ < primary_.size()) size_ *= 2;
    minimum_definition_.assign(size_ * 2, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < leaf_minimum.size(); ++i) {
      minimum_definition_[size_ + i] = leaf_minimum[i];
    }
    for (size_t i = size_; i-- > 1;) {
      minimum_definition_[i] = std::min(minimum_definition_[i * 2], minimum_definition_[i * 2 + 1]);
    }
  }

  [[nodiscard]] bool HasAfterFromHandleDefinedBefore(size_t primary_begin, size_t primary_end,
                                                     size_t definition_upper) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    size_t left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                      primary_.begin());
    size_t right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                       primary_.begin());
    left += size_;
    right += size_;
    size_t minimum = std::numeric_limits<size_t>::max();
    while (left < right) {
      if (left & 1U) minimum = std::min(minimum, minimum_definition_[left++]);
      if (right & 1U) minimum = std::min(minimum, minimum_definition_[--right]);
      left /= 2;
      right /= 2;
    }
    return minimum < definition_upper;
  }

 private:
  std::vector<size_t> primary_;
  std::vector<size_t> minimum_definition_;
  size_t size_ = 0;
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

  void Plan(const StmtPtr& body, const std::vector<VarPtr>& parameters, bool recurrence_only = false) {
    recurrence_only_ = recurrence_only;
    names_.Collect(body, parameters);
    DefMapVisitor demand;
    demand.Run(body);
    payload_observations_ =
        BuildPayloadObservations(demand.defs, demand.observation_reads, demand.binding_reads);
    ReverseBranchOrderVisitor reverse;
    reverse.Run(body);
    reverse_order_ = std::move(reverse.stmt_order);
    for (const auto& parameter : parameters) {
      if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
        incoming_bases_.insert(GetDefinedMemRef(tile)->base_.get());
      }
    }
    VisitStmt(body);
    struct Observation {
      const Var* base;
      size_t position;
      size_t reverse;
      size_t definition;
    };
    std::vector<Observation> observations;
    std::map<const Var*, std::vector<std::pair<size_t, size_t>>> reachable_events;
    std::map<const Var*, std::vector<std::pair<size_t, size_t>>> recurring_events;
    for (const auto& [var, loop] : recurring_reads_) {
      const auto* base = GetDefinedMemRef(GetTileTypeWithMemRef(var->GetType()))->base_.get();
      recurring_events[base].emplace_back(positions_.at(loop), loop_ends_.at(loop));
    }
    for (const auto& [position, var] : reads_) {
      const auto tile = GetTileTypeWithMemRef(var->GetType());
      auto definition = definitions_.find(var.get());
      const auto* base = GetDefinedMemRef(tile)->base_.get();
      const size_t version = definition == definitions_.end() ? 0 : definition->second;
      const size_t reverse = read_reverse_.at(position);
      reachable_events[base].emplace_back(position, reverse);
      observations.push_back({base, position, reverse, version});
    }
    for (auto& [base, values] : recurring_events) recurring_indexes_[base].Build(std::move(values));
    for (auto& [base, values] : reachable_events) reachable_indexes_[base].Build(std::move(values), false);
    std::sort(observations.begin(), observations.end(),
              [](const auto& a, const auto& b) { return a.definition < b.definition; });
    size_t next = 0;
    for (const auto& loop : loops_) {
      // The preorder loop sweep activates only handles naming older values.
      // A range maximum then tests both branch orderings in O(log N).
      while (next < observations.size() && observations[next].definition < positions_.at(loop.get())) {
        const auto& event = observations[next++];
        reachable_indexes_[event.base].Activate(event.position, event.reverse);
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
    bool identity = source != nullptr;
    if (auto call = As<Call>(statement->value_); call && call->op_ && !call->args_.empty() &&
                                                 op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
      source = AsVarLike(call->args_[0]);
      const auto input = source ? GetTileTypeWithMemRef(source->GetType()) : nullptr;
      const auto output = GetTileTypeWithMemRef(statement->var_->GetType());
      identity = input && output && SameKnownTileMemorySpace(input, output) &&
                 SamePhysicalWindow(GetDefinedMemRef(input), GetDefinedMemRef(output)) &&
                 OpRegistry::GetInstance().GetEntry(call->op_->name_).GetExecutionMemoryAccessEvidence() ==
                     ExecutionMemoryAccessEvidence::NoAccess;
    }
    if (source && GetTileTypeWithMemRef(source->GetType())) {
      // Metadata views do not read data. Their later readers observe the same
      // logical version, including a view created inside a loop from old input.
      auto origin = definitions_.find(source.get());
      definitions_[statement->var_.get()] = origin == definitions_.end() ? 0 : origin->second;
      // Flatten immutable renames/full-window views once in definition order.
      // Mutating aliases and partial windows do not establish value identity.
      if (identity) identity_roots_[statement->var_.get()] = IdentityRoot(source);
      // Visit the expression through the demand filter: non-source operands
      // and reference attributes can be real observations even for a view.
      VisitExpr(statement->value_);
      return;
    }
    definitions_[statement->var_.get()] = current_;
    VisitExpr(statement->value_);
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    for (const auto& result : branch->return_vars_) definitions_[result.get()] = ++order_;
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    VisitLoop(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override { VisitLoop(loop); }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitVarLike_(const VarPtr& value) override {
    const auto observation = payload_observations_.find(current_statement_);
    if (observation == payload_observations_.end() || observation->second.count(value) == 0) return;
    if (GetTileTypeWithMemRef(value->GetType())) {
      reads_.emplace_back(current_, value);
      read_reverse_[current_] = current_reverse_;
      const auto definition = definitions_.find(value.get());
      const size_t version = definition == definitions_.end() ? 0 : definition->second;
      // The stack is ordered by header position. The first header after the
      // definition is the outermost loop that can repeat this same value's read.
      const auto loop =
          std::upper_bound(loop_stack_.begin(), loop_stack_.end(), version,
                           [](size_t position, const auto& entry) { return position < entry.first; });
      if (loop != loop_stack_.end()) recurring_reads_.emplace_back(value, loop->second);
    }
  }

 private:
  bool recurrence_only_ = false;
  const Stmt* current_statement_ = nullptr;
  std::map<const Stmt*, std::set<VarPtr>> payload_observations_;
  std::map<const Var*, VarPtr> identity_roots_;

  [[nodiscard]] VarPtr IdentityRoot(const VarPtr& value) const {
    auto root = identity_roots_.find(value.get());
    return root == identity_roots_.end() ? value : root->second;
  }

  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& loop) {
    loops_.push_back(loop);
    for (const auto& argument : loop->iter_args_) {
      VisitExpr(argument->initValue_);
      definitions_[argument.get()] = current_;
    }
    // Initializers execute once at entry, outside this loop's recurrence.
    const bool repeats = LoopMayExecute(loop, true);
    if (repeats) loop_stack_.emplace_back(positions_.at(loop.get()), loop.get());
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      // Separate condition reads from initializer reads at the loop header.
      current_ = ++order_;
      VisitExpr(loop->condition_);
    }
    VisitStmt(loop->body_);
    if (repeats) loop_stack_.pop_back();
    loop_ends_[loop.get()] = order_;
    for (const auto& result : loop->return_vars_) definitions_[result.get()] = ++order_;
  }

  template <typename LoopPtr>
  void PlanLoop(const LoopPtr& loop) {
    // A loop that cannot enter returns its initializer without modifying it.
    // Even independently observed inputs need no private copy on that path.
    if (!LoopMayExecute(loop, false)) return;
    auto body = loop->body_;
    if (auto sequence = As<SeqStmts>(body); sequence && sequence->stmts_.size() == 1) {
      body = sequence->stmts_[0];
    }
    if (auto yield = As<YieldStmt>(body); yield && yield->value_.size() == loop->iter_args_.size()) {
      bool unchanged = true;
      for (size_t i = 0; i < yield->value_.size(); ++i) {
        const auto source = AsVarLike(yield->value_[i]);
        const auto& argument = loop->iter_args_[i];
        const auto initial = AsVarLike(argument->initValue_);
        unchanged = unchanged && source &&
                    (source == argument || (initial && IdentityRoot(source) == IdentityRoot(initial)));
      }
      // Forwarding a carry, its initializer, or an immutable alias of that
      // initializer cannot overwrite input payloads,
      // even when another handle observes that input after this loop. Such a
      // binding needs no private buffer (and no unsupported Acc-to-Acc copy).
      if (unchanged) return;
    }
    const auto position = positions_.at(loop.get());
    std::set<size_t> isolate;
    std::map<const Var*, std::vector<std::pair<int64_t, size_t>>> windows;
    std::set<const Var*> unknown_bases;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& initial = loop->iter_args_[i]->initValue_;
      auto tile = GetTileTypeWithMemRef(initial->GetType());
      if (!tile) continue;
      const auto memref = GetDefinedMemRef(tile);
      const auto* base = memref->base_.get();
      const auto reachable = reachable_indexes_.find(base);
      const auto recurring = recurring_indexes_.find(base);
      // Recurrence intervals use (loop begin, loop end) in the range-maximum
      // index: some begin <= position must have end >= position. A read outside
      // all such loops cannot observe this carry on a later iteration.
      const bool nested_observation =
          recurring != recurring_indexes_.end() && recurring->second.HasAfter(0, position, position - 1);
      if (recurrence_only_) {
        // Legacy fixup copies yields back into the carry. Preserve independently
        // read seed payloads with a private carry when a legal entry copy exists;
        // unsupported spaces retain the retarget guard and explicit copy error.
        const auto memory = tile->GetMemorySpace();
        if (nested_observation && memory && IsTileMoveEverSupported(*memory, *memory)) isolate.insert(i);
        continue;
      }
      if (incoming_bases_.count(base) != 0 || nested_observation ||
          (reachable != reachable_indexes_.end() &&
           reachable->second.HasAfter(position + 1, order_, reverse_order_.at(loop.get())))) {
        isolate.insert(i);
      }
      auto offset = As<ConstInt>(memref->byte_offset_);
      windows[base].emplace_back(offset ? offset->value_ : 0, i);
      if (!offset) unknown_bases.insert(base);
    }
    for (auto& [base, ranges] : windows) {
      if (ranges.size() < 2) continue;
      if (unknown_bases.count(base) != 0) {
        for (const auto& [offset, index] : ranges) {
          (void)offset;
          isolate.insert(index);
        }
        continue;
      }
      std::sort(ranges.begin(), ranges.end());
      __int128 end = 0;
      size_t owner = ranges.front().second;
      bool first = true;
      for (const auto& [offset, index] : ranges) {
        auto tile = GetTileTypeWithMemRef(loop->iter_args_[index]->initValue_->GetType());
        if (!first && offset < end) {
          isolate.insert(owner);
          isolate.insert(index);
        }
        const auto next_end = static_cast<__int128>(offset) + GetDefinedMemRef(tile)->size_;
        if (first || next_end > end) {
          end = next_end;
          owner = index;
        }
        first = false;
      }
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
  std::map<const Var*, ReachableEventIndex> reachable_indexes_;
  std::vector<std::pair<size_t, const Stmt*>> loop_stack_;
  std::map<const Stmt*, size_t> loop_ends_;
  std::vector<std::pair<VarPtr, const Stmt*>> recurring_reads_;
  std::map<const Var*, ReachableEventIndex> recurring_indexes_;
  StorageNameSupply names_;
  std::map<const Stmt*, size_t> positions_;
  std::map<const Var*, size_t> definitions_;
  std::vector<std::pair<size_t, VarPtr>> reads_;
  std::vector<StmtPtr> loops_;
  std::set<const Var*> incoming_bases_;
};

/// Plans top-down retypes. Produces (old Var -> new Type) map.
class TopDownRetargeter {
 public:
  /// Runs the analysis. Returns map: old VarPtr -> new Type (with target MemRef).
  std::map<VarPtr, TypePtr> Compute(const StmtPtr& func_body) {
    BuildAnalysis(func_body);
    VisitForStmts(func_body);
    return std::move(rewrites_);
  }

  /// Coalesce peeled loop-carried accumulator if-phis.
  ///
  /// DIRECTION OF TRAVEL (long horizon — do not act on this without a migration).
  /// The shape this repairs is the *hand-peeled* split-K idiom:
  ///
  ///     if k == 0: acc = tile.matmul(a, b)          // fresh Acc buffer
  ///     else:      acc = tile.matmul_acc(acc, a, b) // in-place, another buffer
  ///
  /// It gives one logical value two producers on two L0C buffers, which the
  /// machine cannot represent — there is no Acc->Acc copy, so this repair exists
  /// only to rewrite buffer identity after the fact.  The predicated spelling
  /// says the same thing on one buffer and needs no repair:
  ///
  ///     acc = tile.matmul_acc(acc, a, b, init_cond=(k == 0))
  ///
  /// `AutoTileMatmulL0` already *generates* the predicated form (and head-peels
  /// the bias K-loop, which has no `init_cond` operand), so no compiler-generated
  /// IR reaches this function any more — every remaining caller is a peel in the
  /// author's own source.  We keep repairing those: existing kernels written with
  /// the peel must keep compiling, and pypto-lib alone has 95 such sites.
  ///
  /// Retiring this repair is therefore a *long-horizon* intent, not a near-term
  /// plan. It would need the source-level migration to land first. Until then,
  /// prefer `init_cond` in new code and leave existing peels alone.
  ///
  /// LowerPipelineLoops peels a stage=2 K-loop into an epilogue IfStmt whose live
  /// branch is an in-place accumulator (matmul_acc, output aliasing input on the
  /// accumulator buffer) and whose dead `if k==0` branch is a fresh matmul seed on a
  /// *different* buffer.  Left alone, YieldFixupMutator reconciles the two by copying
  /// the accumulator onto the seed's buffer via an Acc->Acc tile.move — both a second
  /// co-live L0C buffer (overflow) and an op TMovOp::verify rejects on every target
  /// (there is no legal Acc->Acc tmov pair).  We instead retarget the seed producer
  /// onto the accumulator buffer so both branches share it and no move is emitted,
  /// matching mad_acc's shared-%dst in-place semantics.
  ///
  /// The seed retype bypasses the *global* dead-at-assign check (the accumulator
  /// buffer is legitimately live at the post-if phi consumer, which the global
  /// check would treat as a conflict), but only after `TryCoalesceAccIfPhi`
  /// verifies the two preconditions branch exclusivity actually needs: (a) the
  /// seed producer is lexically inside the branch, and (b) a branch-scoped
  /// liveness scan (`IsTargetDeadAtAssign(..., stop_at=if)`) finds no same-branch
  /// tail read of the accumulator buffer. For both a direct `matmul_acc` and a
  /// branch-local loop seeded outside the IfStmt, an alias-aware post-if scan
  /// proves that no external handle to the reused input remains observable
  /// independently of the phi. When any condition fails, that phi is left to
  /// YieldFixup instead of being coalesced.
  /// Returns the rewrite map (apply via RetypeApplier).  A needed-but-declined
  /// retarget (after the preconditions hold) is a hard error — no legal Acc->Acc
  /// move exists to fall back to.
  std::map<VarPtr, TypePtr> CoalesceAccumulatorIfPhis(const StmtPtr& func_body) {
    BuildAnalysis(func_body);
    VisitIfPhisForAccumulator(func_body);
    return std::move(rewrites_);
  }

  /// Give divergent branch results independent destinations before planning.
  /// Direct, unaliased branch-local producers may write these fresh destinations;
  /// the shared YieldFixup materializes the remaining arm transfers afterwards.
  std::map<VarPtr, TypePtr> CanonicalizeBranchStorage(const StmtPtr& func_body,
                                                      std::vector<StmtPtr>* allocations) {
    BuildAnalysis(func_body, /*index_branch_storage=*/true);
    // InitMemRef clears MemRef::is_pinned_; the hoisted alloc then carries the
    // authoritative declaration contract. Preserve those output bindings.
    if (auto body = As<SeqStmts>(func_body)) {
      for (const auto& stmt : body->stmts_) {
        if (auto base = GetPinnedAllocBase(stmt)) pinned_branch_bases_.insert(base.get());
      }
    }
    VisitBranchStorage(func_body, allocations);
    return std::move(rewrites_);
  }

 private:
  struct RewriteJournalEntry {
    VarPtr var;
    std::optional<TypePtr> previous;
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

  std::map<VarPtr, VarDef> defs_;
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
  std::map<const Var*, VersionedReachableEventIndex> alias_read_index_;
  std::map<const Var*, ReachableEventIndex> recurring_alias_read_index_;
  std::map<size_t, ReachableEventIndex> recurring_physical_read_index_;
  std::map<VarPtr, TypePtr> rewrites_;
  std::vector<RewriteJournalEntry> rewrite_journal_;
  std::set<VarPtr> visiting_;  // cycle guard
  std::map<const Var*, size_t> storage_handle_counts_;
  std::map<const Var*, const Var*> branch_storage_handle_bases_;
  std::set<const Var*> pinned_branch_bases_;
  uint64_t branch_storage_counter_ = 0;

  void BuildAnalysis(const StmtPtr& func_body, bool index_branch_storage = false) {
    storage_handle_counts_.clear();
    branch_storage_handle_bases_.clear();
    DefMapVisitor def_v;
    def_v.Run(func_body);
    defs_ = std::move(def_v.defs);
    stmt_order_ = std::move(def_v.stmt_order);
    stmt_end_order_ = std::move(def_v.stmt_end_order);
    analysis_root_ = def_v.root;

    ReverseBranchOrderVisitor reverse_v;
    reverse_v.Run(func_body);
    reverse_stmt_order_ = std::move(reverse_v.stmt_order);
    reverse_stmt_end_order_ = std::move(reverse_v.stmt_end_order);
    BuildAliasFamilies();
    BuildAccessIndexes(def_v.reads, def_v.writes);
    BuildRecurringReadIndexes(def_v.condition_reads, def_v.observation_reads, def_v.binding_reads,
                              def_v.loops);
    if (!index_branch_storage) return;

    // Count distinct storage handles once, including incoming handles with no
    // local definition. If-phi handles are consumers of branch storage rather
    // than independently observable aliases. Other aliases conservatively block
    // producer retargeting. This fixed walk avoids per-phi subtree/use scans.
    std::set<VarPtr> counted;
    auto count_storage = [&](const VarPtr& var) {
      if (!counted.insert(var).second) return;
      auto def = defs_.find(var);
      if (def != defs_.end() && def->second.kind == VarDef::kIfReturn) return;
      auto tile = GetTileTypeWithMemRef(var->GetType());
      if (tile) {
        const auto* base = GetDefinedMemRef(tile)->base_.get();
        ++storage_handle_counts_[base];
        branch_storage_handle_bases_[var.get()] = base;
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

  void RetargetBranchProducer(const VarPtr& var, const StmtPtr& branch, const MemRefPtr& target,
                              std::optional<MemorySpace> memory) {
    auto def = defs_.find(var);
    if (def == defs_.end() || def->second.kind != VarDef::kAssign ||
        !IsInside(def->second.definition_stmt, branch) || rewrites_.count(var) != 0) {
      return;
    }
    auto tile = GetTileTypeWithMemRef(var->GetType());
    if (!tile || pinned_branch_bases_.count(GetDefinedMemRef(tile)->base_.get()) != 0 ||
        storage_handle_counts_[GetDefinedMemRef(tile)->base_.get()] != 1) {
      return;
    }
    auto assign = As<AssignStmt>(def->second.assign_stmt);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    const auto& registry = OpRegistry::GetInstance();
    if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_)) return;
    const auto& entry = registry.GetEntry(call->op_->name_);
    // Reusing an input, including a metadata view, is an existing storage
    // contract. Do not walk that producer chain onto a fresh phi allocation.
    if (entry.OutputMemoryInheritsInput() ||
        op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size()).has_value()) {
      return;
    }
    // The target is freshly allocated and the producer is branch-local, so it
    // has no pre-existing live contents. Retain every operation-legality check.
    RetargetAssign(var, def->second, target, memory, /*check_liveness=*/false);
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
  void VisitBranchStorageLoop(const LoopPtr& loop, std::vector<StmtPtr>* allocations) {
    for (const auto& argument : loop->iter_args_) {
      FollowBranchStorageAlias(argument, AsVarLike(argument->initValue_));
    }
    VisitBranchStorage(loop->body_, allocations);
    for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
      // A zero-trip loop returns its initializer, not its body's yield storage.
      FollowBranchStorageAlias(loop->return_vars_[i], loop->iter_args_[i]);
    }
  }

  void VisitBranchStorage(const StmtPtr& stmt, std::vector<StmtPtr>* allocations) {
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) VisitBranchStorage(child, allocations);
    } else if (auto assign = As<AssignStmt>(stmt)) {
      auto source = alias_source_.find(assign->var_);
      if (source != alias_source_.end()) {
        FollowBranchStorageAlias(assign->var_, source->second, As<Call>(assign->value_));
      }
    } else if (auto loop = As<ForStmt>(stmt)) {
      VisitBranchStorageLoop(loop, allocations);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      VisitBranchStorageLoop(loop, allocations);
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitBranchStorage(scope->body_, allocations);
    } else if (auto branch = As<IfStmt>(stmt)) {
      VisitBranchStorage(branch->then_body_, allocations);
      const auto else_body = branch->else_body_;
      if (else_body) VisitBranchStorage(*else_body, allocations);
      auto then_yield = transform_utils::GetLastYieldStmt(branch->then_body_);
      auto else_yield = else_body ? transform_utils::GetLastYieldStmt(*else_body) : nullptr;
      for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
        auto result = branch->return_vars_[i];
        auto result_tile = CurrentTileType(result);
        if (!result_tile) continue;
        CHECK_SPAN(else_body.has_value() && then_yield && else_yield && i < then_yield->value_.size() &&
                       i < else_yield->value_.size(),
                   branch->span_)
            << "Buffer IR storage legalization requires both arms to yield every tile result";
        auto then_var = AsVarLike(then_yield->value_[i]);
        auto else_var = AsVarLike(else_yield->value_[i]);
        auto then_tile = then_var ? CurrentTileType(then_var) : nullptr;
        auto else_tile = else_var ? CurrentTileType(else_var) : nullptr;
        CHECK_SPAN(then_tile && else_tile, branch->span_)
            << "Buffer IR storage legalization requires tile branch yields to be storage variables";
        auto then_memref = GetDefinedMemRef(then_tile);
        auto else_memref = GetDefinedMemRef(else_tile);
        if (SamePhysicalWindow(then_memref, else_memref)) {
          PlanRewrite(result, then_memref, then_tile->GetMemorySpace());
          continue;
        }
        auto memory = result_tile->GetMemorySpace();
        CHECK_SPAN(memory.has_value() && *memory != MemorySpace::Acc, branch->span_)
            << "Buffer IR storage legalization requires divergent Acc branch results to be "
               "coalesced; Acc-to-Acc transfers are unsupported";
        auto base = std::make_shared<Var>("mem_branch_result_" + std::to_string(branch_storage_counter_++),
                                          GetPtrType(), result->span_);
        const auto size =
            std::max({GetDefinedMemRef(result_tile)->size_, then_memref->size_, else_memref->size_});
        auto target = std::make_shared<MemRef>(base, static_cast<int64_t>(0), size, result->span_);
        allocations->push_back(CreateAllocStatement(target, *memory));
        PlanRewrite(result, target, memory);
        RetargetBranchProducer(then_var, branch->then_body_, target, memory);
        RetargetBranchProducer(else_var, *else_body, target, memory);
      }
    }
  }

  static size_t FindSet(std::vector<size_t>& parent, size_t value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  }

  static void UnionSets(std::vector<size_t>& parent, size_t lhs, size_t rhs) {
    lhs = FindSet(parent, lhs);
    rhs = FindSet(parent, rhs);
    if (lhs != rhs) parent[rhs] = lhs;
  }

  void BuildAccessIndexes(const std::map<const Stmt*, std::set<VarPtr>>& reads,
                          const std::map<const Stmt*, std::set<VarPtr>>& writes) {
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
    std::map<const Var*, std::vector<VersionedReachableEventIndex::Event>> alias_events;
    auto record = [&](const Stmt* stmt, const VarPtr& var, bool is_read) {
      auto primary = stmt_order_.find(stmt);
      auto reverse = reverse_stmt_order_.find(stmt);
      if (primary == stmt_order_.end() || reverse == reverse_stmt_order_.end()) return;
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
        alias_events[ResolveAliasRoot(var).get()].push_back(
            VersionedReachableEventIndex::Event{primary->second, definition});
      }
    };
    for (const auto& [stmt, vars] : reads) {
      for (const auto& var : vars) record(stmt, var, true);
    }
    for (const auto& [stmt, vars] : writes) {
      for (const auto& var : vars) record(stmt, var, false);
    }
    for (auto& [component, events] : physical_events) {
      physical_access_index_[component].Build(std::move(events));
    }
    for (auto& [root, events] : alias_events) alias_read_index_[root].Build(std::move(events));
  }

  /// A read of an older handle can repeat after an IfStmt through an enclosing
  /// loop's backedge, even when that read lexically precedes the IfStmt. Sweep
  /// the read sites and preorder loop intervals once, then binary-search the
  /// active loop stack for the outermost loop entered after each handle's
  /// definition. Each loop is pushed/popped once; event indexing and queries
  /// remain O(N log N), with no per-phi subtree or ancestor scans.
  void BuildRecurringReadIndexes(const std::map<const Stmt*, std::set<VarPtr>>& condition_reads,
                                 const std::map<const Stmt*, std::set<VarPtr>>& observation_reads,
                                 const DefMapVisitor::BindingReads& binding_reads,
                                 const std::vector<StmtPtr>& loops) {
    recurring_alias_read_index_.clear();
    recurring_physical_read_index_.clear();
    // A carry's value advances on its own backedge, but re-entering its
    // binding from an outer loop consumes the initializer again. Attribute an
    // observed carry's entry read to that binding header, not to every use in
    // its body. Propagate observed bindings once along initializer chains so
    // intermediate carry versions survive intervening loops as well.
    // A consumed loop result also observes the initializer on a zero-trip
    // path. Definitely nonempty For results need no such entry observation.
    // Unused bindings remain non-observing; this worklist stays O(N log N).
    auto effective_observations = BuildPayloadObservations(defs_, observation_reads, binding_reads);
    std::vector<std::pair<size_t, const std::set<VarPtr>*>> ordered_reads;
    ordered_reads.reserve(effective_observations.size());
    for (const auto& [stmt, vars] : effective_observations) {
      ordered_reads.emplace_back(stmt_order_.at(stmt), &vars);
    }
    std::sort(ordered_reads.begin(), ordered_reads.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    // Half-open lexical intervals [begin, end) of the currently enclosing loops.
    std::vector<std::pair<size_t, size_t>> loop_stack;
    std::map<const Var*, std::vector<std::pair<size_t, size_t>>> recurring_events;
    std::map<size_t, std::vector<std::pair<size_t, size_t>>> physical_events;
    auto record_physical = [&](const VarPtr& var, const std::pair<size_t, size_t>& interval) {
      // Semantic aliases may still carry stale physical metadata before the
      // first normalization. Protect their canonical allocation as well.
      for (const auto& handle : {var, ResolveAliasRoot(var)}) {
        if (auto tile = GetTileTypeWithMemRef(handle->GetType())) {
          auto component = memref_component_.find(GetDefinedMemRef(tile).get());
          if (component != memref_component_.end()) physical_events[component->second].push_back(interval);
        }
      }
    };
    size_t next_loop = 0;
    for (const auto& [position, vars] : ordered_reads) {
      // A loop's own initializer is evaluated outside its recurrence. Its
      // header reads therefore see only loops with strictly earlier headers.
      while (next_loop < loops.size() && stmt_order_.at(loops[next_loop].get()) < position) {
        const auto& current_loop = loops[next_loop++];
        if (!LoopMayExecute(current_loop, true)) continue;
        const auto* loop = current_loop.get();
        const size_t begin = stmt_order_.at(loop);
        while (!loop_stack.empty() && loop_stack.back().second <= begin) loop_stack.pop_back();
        loop_stack.emplace_back(begin, stmt_end_order_.at(loop) + 1);
      }
      while (!loop_stack.empty() && loop_stack.back().second <= position) loop_stack.pop_back();
      for (const auto& var : *vars) {
        const auto def = defs_.find(var);
        const auto recurring =
            def == defs_.end()
                ? loop_stack.begin()  // Incoming handles predate even a root loop at order zero.
                : std::upper_bound(
                      loop_stack.begin(), loop_stack.end(), def->second.order,
                      [](size_t definition, const auto& interval) { return definition < interval.first; });
        if (recurring != loop_stack.end()) {
          recurring_events[ResolveAliasRoot(var).get()].push_back(*recurring);
          // Alias and physical recurrence must share payload demand. Unused
          // forwarding bindings belong only to the ordinary access indexes.
          record_physical(var, *recurring);
        }
      }
    }
    // While conditions, unlike initializers, are evaluated again at the same
    // header. Preserve observations of older handles on that loop's backedge.
    for (const auto& [loop, vars] : condition_reads) {
      const size_t begin = stmt_order_.at(loop);
      for (const auto& var : vars) {
        const auto def = defs_.find(var);
        if (def == defs_.end() || def->second.order < begin) {
          const auto interval = std::make_pair(begin, stmt_end_order_.at(loop) + 1);
          recurring_events[ResolveAliasRoot(var).get()].push_back(interval);
          record_physical(var, interval);
        }
      }
    }
    for (auto& [root, events] : recurring_events) {
      recurring_alias_read_index_[root].Build(std::move(events));
    }
    for (auto& [component, events] : physical_events) {
      recurring_physical_read_index_[component].Build(std::move(events));
    }
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
      if (def.kind == VarDef::kForReturn) {
        auto loop = As<ForStmt>(def.control_stmt);
        if (loop && def.return_idx < loop->iter_args_.size()) {
          alias_source_[var] = std::static_pointer_cast<const Var>(loop->iter_args_[def.return_idx]);
        }
        continue;
      }
      if (def.kind == VarDef::kWhileReturn) {
        auto loop = As<WhileStmt>(def.control_stmt);
        if (loop && def.return_idx < loop->iter_args_.size()) {
          alias_source_[var] = std::static_pointer_cast<const Var>(loop->iter_args_[def.return_idx]);
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
  void VisitForStmts(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) VisitForStmts(s);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      PropagateFromForStmt(for_stmt);
      VisitForStmts(for_stmt->body_);
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      VisitForStmts(while_stmt->body_);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      VisitForStmts(if_stmt->then_body_);
      if (if_stmt->else_body_.has_value()) VisitForStmts(if_stmt->else_body_.value());
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitForStmts(scope->body_);
    }
  }

  void PropagateFromForStmt(const ForStmtPtr& for_stmt) {
    if (for_stmt->iter_args_.empty()) return;
    auto yield = FindYieldStmt(for_stmt->body_);
    if (!yield) return;

    for (size_t i = 0; i < for_stmt->iter_args_.size() && i < yield->value_.size(); ++i) {
      auto ia = for_stmt->iter_args_[i];
      // If a prior PropagateFromForStmt (an enclosing ForStmt) already retyped
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
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      VisitIfPhisForAccumulator(while_stmt->body_);
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
    if (it == defs_.end() || it->second.kind != VarDef::kForReturn) return std::nullopt;
    auto loop = As<ForStmt>(it->second.control_stmt);
    if (!loop || it->second.return_idx >= loop->iter_args_.size()) return std::nullopt;

    if (!IsInside(loop, branch_scope)) return std::nullopt;

    auto init = AsVarLike(loop->iter_args_[it->second.return_idx]->initValue_);
    auto init_tile = init ? CurrentTileType(init) : nullptr;
    if (!init || !CurrentTileType(var) || !init_tile) return std::nullopt;

    auto init_def = defs_.find(init);
    if (init_def != defs_.end() && IsInside(init_def->second.definition_stmt, branch_scope)) {
      return std::nullopt;
    }

    auto body_yield = FindYieldStmt(loop->body_);
    if (!body_yield || it->second.return_idx >= body_yield->value_.size()) return std::nullopt;
    auto yielded = AsVarLike(body_yield->value_[it->second.return_idx]);
    auto iter_arg = std::static_pointer_cast<const Var>(loop->iter_args_[it->second.return_idx]);
    if (!yielded || !IsInplaceChainRootedAt(yielded, iter_arg)) return std::nullopt;
    return AccumulatorTarget{GetDefinedMemRef(init_tile), init_tile->GetMemorySpace(), init};
  }

  // For an IfStmt whose branches yield an in-place accumulator on one side and a
  // fresh seed on the other (a different L0C buffer), retarget the seed onto the
  // accumulator buffer.  Scoped to Acc — the ISA case with no legal reconciling
  // move.  A declined retarget is a hard error (see CoalesceAccumulatorIfPhis).
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
        } else if (seed_def->second.kind == VarDef::kForReturn) {
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

  TileTypePtr CurrentTileType(const VarPtr& var) {
    auto it = rewrites_.find(var);
    auto type = (it != rewrites_.end()) ? it->second : var->GetType();
    return GetTileTypeWithMemRef(type);
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
    if (SamePhysicalWindow(CurrentMemRef(var), target)) return true;  // already aligned
    if (!visiting_.insert(var).second) return false;                  // cycle
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
    } else if (def.kind == VarDef::kForReturn) {
      retargeted = RetargetForReturn(var, def, target, target_memory);
    } else if (def.kind == VarDef::kIterArg) {
      retargeted = RetargetIterArg(var, def, target, target_memory);
    }
    if (retargeted) transaction.Commit();
    return retargeted;
  }

  /// Retype a Var defined by an AssignStmt.
  ///
  /// `check_liveness` gates the general dead-at-assign check (IsTargetDeadAtAssign).
  /// It is true for the normal loop-carry retarget path.  It is set false only by
  /// CoalesceAccumulatorIfPhis: coalescing an IfStmt phi's two branch yields onto one
  /// buffer is always safe (the phi is redefined by exactly one branch at runtime, so
  /// the branches are mutually exclusive and the target's downstream liveness cannot be
  /// violated by a branch-local producer).  The op-legality checks below still apply.
  bool RetargetAssign(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                      std::optional<MemorySpace> target_memory, bool check_liveness = true) {
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
    //      with a real move (see AlignLoopCarriesToInitMutator's contract).
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
    if (!entry.IsInplaceSafe()) {
      const size_t read_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
      if (CallReadsBase(*call, target->base_.get(), read_arg_count)) return false;
    }
    if (op_predicates::CastRequiresDisjointSource(call, !IsA5Target()) &&
        CallReadsBase(*call, target->base_.get(), 1)) {
      return false;
    }

    // Unconstrained: check liveness, then plan retype.  (Skipped for if-phi
    // branch coalescing, where branch exclusivity is a stronger guarantee.)
    if (check_liveness && !IsTargetDeadAtAssign(def, target)) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// True if any argument of the call is a TileType Var whose MemRef base
  /// is `target_base`.  Used to detect would-be in-place execution before
  /// we retype the output onto the same buffer.
  static bool CallReadsBase(const Call& call, const Var* target_base, size_t arg_count) {
    ExprReadBaseCollector c;
    for (size_t i = 0; i < std::min(arg_count, call.args_.size()); ++i) {
      c.VisitExpr(call.args_[i]);
    }
    return c.bases.count(target_base) > 0;
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

  /// Retype an IfStmt return_var: recurse into both branches' yield values.
  bool RetargetIfReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                        std::optional<MemorySpace> target_memory) {
    auto if_stmt = As<IfStmt>(def.control_stmt);
    if (!if_stmt) return false;
    size_t idx = def.return_idx;

    auto visit_branch = [&](const StmtPtr& body) -> bool {
      auto y = FindYieldStmt(body);
      if (!y || idx >= y->value_.size()) return false;
      auto yv = AsVarLike(y->value_[idx]);
      if (!yv) return false;
      return TryRetargetVar(yv, target, target_memory);
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
  bool RetargetForReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                         std::optional<MemorySpace> target_memory) {
    auto for_stmt = As<ForStmt>(def.control_stmt);
    if (!for_stmt) return false;
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
    auto tracked = branch_storage_handle_bases_.find(var.get());
    if (tracked == branch_storage_handle_bases_.end()) return;
    auto tile = GetTileTypeWithMemRef(type);
    if (!tile) return;
    const auto* base = GetDefinedMemRef(tile)->base_.get();
    if (base == tracked->second) return;
    auto& previous_count = storage_handle_counts_[tracked->second];
    INTERNAL_CHECK_SPAN(previous_count > 0, var->span_)
        << "Internal error: branch storage handle count must remain positive";
    --previous_count;
    ++storage_handle_counts_[base];
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
    auto new_type = CloneTypeWithMemRef(var->GetType(), target, target_memory);
    UpdateBranchStorageHandle(var, new_type);
    auto previous = rewrites_.find(var);
    rewrite_journal_.push_back(RewriteJournalEntry{
        var, previous == rewrites_.end() ? std::nullopt : std::optional<TypePtr>(previous->second)});
    rewrites_[var] = new_type;
  }

  void RollbackRewrites(size_t checkpoint) {
    INTERNAL_CHECK(checkpoint <= rewrite_journal_.size()) << "Invalid rewrite transaction checkpoint";
    while (rewrite_journal_.size() > checkpoint) {
      auto entry = std::move(rewrite_journal_.back());
      rewrite_journal_.pop_back();
      UpdateBranchStorageHandle(entry.var, entry.previous.value_or(entry.var->GetType()));
      if (entry.previous.has_value()) {
        rewrites_[entry.var] = *entry.previous;
      } else {
        rewrites_.erase(entry.var);
      }
    }
  }

  /// True when the reused input is independently observed after the IfStmt,
  /// either in its lexical suffix or on a later enclosing-loop iteration.
  /// Retargeting the fresh branch must not add a write visible to that observer.
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
    const auto* alias_root = ResolveAliasRoot(target.reused_input).get();
    auto if_begin = stmt_order_.find(if_stmt.get());
    if (if_begin == stmt_order_.end()) return true;
    auto recurring = recurring_alias_read_index_.find(alias_root);
    if (recurring != recurring_alias_read_index_.end() &&
        recurring->second.HasAfter(0, if_begin->second, if_begin->second)) {
      return true;
    }
    auto reads = alias_read_index_.find(alias_root);
    return reads != alias_read_index_.end() &&
           reads->second.HasAfterFromHandleDefinedBefore(primary_end->second + 1, root_end->second,
                                                         if_begin->second);
  }

  /// Is target's physical window untouched after the definition, including
  /// observations reached through an enclosing-loop backedge? Check recurrence
  /// before the nearest-loop/branch suffix so For-carry retargeting cannot
  /// overwrite old values before the accumulator-phi coalescer runs. The dual
  /// lexical ordering excludes mutually exclusive sibling arms.
  bool IsTargetDeadAfterDefinition(const VarDef& def, const MemRefPtr& target,
                                   const Stmt* stop_at = nullptr) {
    if (!target || !def.definition_stmt) return false;
    auto component = memref_component_.find(target.get());
    if (component == memref_component_.end()) return false;
    // Include every recurrence interval spanning this producer, including its
    // own For: a handle defined before that For can observe the old seed on
    // the next iteration. Reads of the advancing carry itself already exclude
    // its own loop when the recurrence index compares logical definitions.
    auto recurring = recurring_physical_read_index_.find(component->second);
    if (recurring != recurring_physical_read_index_.end() &&
        recurring->second.HasAfter(0, def.order, def.order)) {
      return false;
    }
    auto accesses = physical_access_index_.find(component->second);
    if (accesses == physical_access_index_.end()) return true;

    const Stmt* boundary = stop_at;
    if (!boundary && def.nearest_for) boundary = def.nearest_for.get();
    if (!boundary && analysis_root_) boundary = analysis_root_.get();
    auto boundary_end = stmt_end_order_.find(boundary);
    auto reverse_definition = reverse_stmt_order_.find(def.definition_stmt.get());
    if (boundary_end == stmt_end_order_.end() || reverse_definition == reverse_stmt_order_.end()) {
      return false;
    }
    if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kForReturn || def.kind == VarDef::kWhileReturn) {
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
    const auto* context = PassContext::Current();
    if ((context == nullptr || !context->GetEnableBufferIR()) && entries_.empty() && initializers_.empty()) {
      // An isolation application normalizes every dependent loop binding,
      // including nested loops without an entry copy of their own. Ordinary
      // legacy retypes without isolation retain their old behavior.
      // Pre-register the IterArg's TileType retype so body references to the
      // IterArg are substituted via VisitExpr_(IterArgPtr). The init value
      // recursion is handled inside VisitExpr_(IterArgPtr).
      for (const auto& ia : op->iter_args_) {
        auto var_key = std::static_pointer_cast<const Var>(ia);
        auto rit = rewrites_.find(var_key);
        if (rit != rewrites_.end()) {
          auto new_iter_arg =
              std::make_shared<IterArg>(ia->name_hint_, rit->second, ia->initValue_, ia->span_);
          var_substitution_[var_key] = new_iter_arg;
        }
      }
      for (const auto& rv : op->return_vars_) {
        auto rit = rewrites_.find(rv);
        if (rit != rewrites_.end()) {
          var_substitution_[rv] = std::make_shared<Var>(rv->name_hint_, rit->second, rv->span_);
        }
      }
      return IRMutator::VisitStmt_(op);
    }
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

StmtPtr CoalesceAccumulatorIfPhis(const StmtPtr& body) {
  TopDownRetargeter coalescer;
  auto rewrites = coalescer.CoalesceAccumulatorIfPhis(body);
  if (rewrites.empty()) return body;
  return RetypeApplier(std::move(rewrites)).VisitStmt(body);
}

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
    SnapshotAssembleSources snapshots(func);
    new_body = snapshots.VisitStmt(new_body);
    new_body = InsertAllocsIntoBody(new_body, snapshots.allocations);
    // Resolve identity/view lineage before indexing old-value observations.
    // Branch storage is still provisional here: every dependent alias must
    // remain unsettled until CanonicalizeBranchStorage/YieldFixup selects it.
    new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);
    new_body = RetypeApplier({}).VisitStmt(new_body);
  }
  {
    const bool buffer_ir = ctx != nullptr && ctx->GetEnableBufferIR();
    LoopInputIsolation isolation;
    isolation.Plan(new_body, func->params_, /*recurrence_only=*/!buffer_ir);
    if (!isolation.initializers.empty()) {
      RetypeApplier applier(std::move(isolation.rewrites), std::move(isolation.initializers),
                            std::move(isolation.entries));
      new_body = applier.VisitStmt(new_body);
      new_body = InsertAllocsIntoBody(new_body, isolation.allocations);
      new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);
    }
    if (buffer_ir) new_body = RetypeApplier({}).VisitStmt(new_body);
  }
  TopDownRetargeter retargeter;
  auto rewrites = retargeter.Compute(new_body);
  if (!rewrites.empty()) {
    RetypeApplier applier(std::move(rewrites));
    new_body = applier.VisitStmt(new_body);
  }

  // Peeled accumulator phis are semantic must-alias boundaries. Coalesce them
  // before lifetime analysis so nested K-loop branches enter every memory
  // planner as one allocation family. The legacy planner repeats this repair
  // after packing because packing may introduce new carry mismatches.
  new_body = CoalesceAccumulatorIfPhis(new_body);
  // Accumulator-phi retargeting can expose bare SSA aliases or in-place outputs
  // whose types still name the pre-retarget allocation. Those are semantic
  // aliases too, so normalize them before any planner observes lifetimes or
  // emits handles. Planner-specific repetitions remain below because later
  // YieldFixup/reuse steps can create fresh mismatches.
  //
  // This is the earliest normalization on every path, so no IfStmt YieldFixup has
  // run yet under any planner: use the pre-fixup form, or an in-place producer
  // re-anchored onto a phi's not-yet-chosen buffer here survives into the planner
  // (the guarded runs below only decline to *add* such an anchor, they do not undo
  // one) and is read as the canonical branch target when the fixup finally runs.
  new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);

  // Under memory_planner=PtoAS or DsaRP the whole MemoryReuse pass is skipped.
  // DsaRP must therefore run the remaining correctness normalizations from
  // MemoryReuse Steps 4 through 4.5 in the same order. They are not optimizations: when a
  // peeled accumulator if-phi or
  // loop yields a value living in a different buffer than its iter_arg/return_var,
  // it inserts the `tile.move` that writes the result back into the carry. Without
  // it the carry is never updated and the loop silently becomes a no-op — the
  // `[N, 1]` col-vector carry of an online softmax is the shape that hits this,
  // because its branch producer runs on a `[1, N]` view in its own buffer.
  //
  // Run it here so both alternative planners reconcile carries. Under the
  // legacy PyPTO planner it stays where it is: Step 4 must run *after* reuse
  // decisions, which can themselves create fresh mismatches.
  //
  // PTOAS reconciles ForStmt carries and preserves existing same-window Mat
  // if aliases. Addr-less codegen still handles distinct branch buffers by
  // re-pointing a branch-local producer at the if-phi handle. DSA-RP instead
  // materializes both IfStmt and ForStmt fixups and repairs bare-Var identity
  // copies before lifetime analysis and placement.
  // Resolve storage choices before fixed-buffer yield repair. Functions already
  // selected for Buffer IR retain their existing storage legalization path.
  if (ctx == nullptr || !ctx->GetEnableBufferIR()) {
    auto plan = BuildStorageSelectionPlan(new_body, func->params_);
    new_body = MaterializeStorageSelections(new_body, func->params_, std::move(plan));
  }
  if (ctx != nullptr && ctx->GetEnableBufferIR()) {
    // The Buffer path cannot rely on codegen to repair branch producers or
    // invent copies. Establish independent phi destinations while storage can
    // still be allocated, before any of the three planners sees the program.
    std::vector<StmtPtr> allocations;
    TopDownRetargeter branch_storage;
    auto branch_rewrites = branch_storage.CanonicalizeBranchStorage(new_body, &allocations);
    if (!branch_rewrites.empty()) {
      RetypeApplier applier(std::move(branch_rewrites));
      new_body = applier.VisitStmt(new_body);
    }
    new_body = InsertAllocsIntoBody(new_body, allocations);
    new_body = RetypeApplier({}).VisitStmt(new_body);
    new_body =
        FixupYields(new_body, func->params_, /*fixup_if_stmts=*/true, /*canonical_branch_storage=*/true);
    new_body = NormalizeIdentityCopyBuffers(new_body);
    // Retargeted producers no longer use their original allocations. Removing
    // those now is also necessary for PTOAS, which skips MemoryReuse entirely.
    new_body = RemoveUnusedAllocStatements(new_body, memref_collectors::CollectUsedBasePtrs(new_body));
  } else if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::DsaRP) {
    // Identity-copy normalization brackets YieldFixup on both sides; see the
    // matching step in TransformMemoryReuse for why each side is needed.
    new_body = NormalizeIdentityCopyBuffersBeforeYieldFixup(new_body);

    new_body = FixupYields(new_body, func->params_, /*fixup_if_stmts=*/true);
    new_body = NormalizeIdentityCopyBuffers(new_body);
  } else if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS) {
    new_body = FixupYields(new_body, func->params_, /*fixup_if_stmts=*/false);
  }

  if (new_body == func->body_) return func;

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_,
                                          func->requires_runtime_binding_, func->ir_stage_);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto
