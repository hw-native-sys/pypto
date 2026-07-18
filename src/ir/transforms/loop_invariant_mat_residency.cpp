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

#include "src/ir/transforms/loop_invariant_mat_residency.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace loop_invariant_mat_residency {
namespace {

// ConvertTensorToTileOps deliberately inserts a tensor operand's GM->Mat load
// at its use site. For a tensor.matmul inside a user-written loop that means a
// stationary operand is reloaded on every iteration. Once memory-space
// inference has made every space explicit, this module can recognize the
// complete
//
//   tile.load(GM -> Mat) -> transpose_view* -> tile.move/extract(Mat -> L0)
//
// chain and move the invariant prefix to the loop preheader. The rewrite is
// intentionally conservative: only read-only function parameters, direct
// top-level statements in a statically non-empty sequential loop, and static
// capacity-safe Mat/Left/Right footprints are accepted. Each loop is analyzed
// exactly once against its original direct body. Nested loops are rewritten
// independently, so a chain moves across at most one lexical loop per pass
// invocation instead of being repeatedly rescanned and bubbled through every
// enclosing loop. Together with the one-time inventory and complete-use map,
// the rewrite is O(N).

class VarUseCollector : public IRVisitor {
 public:
  [[nodiscard]] const std::unordered_set<const Expr*>& GetUses() const { return uses_; }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) uses_.insert(op.get());
  }

 private:
  std::unordered_set<const Expr*> uses_;
};

/// Count every syntactic use while excluding SSA definition sites.  The
/// residency recognizer intentionally accepts only an exact single-consumer
/// chain, so an under-count is unsound: plain aliases, Submit arguments,
/// nested expressions, loop initializers, yields, and returns must all count.
class CompleteVarUseCollector : public IRVisitor {
 public:
  void Analyze(const StmtPtr& body) { VisitStmt(body); }

  [[nodiscard]] size_t GetUseCount(const Expr* var) const {
    auto it = use_counts_.find(var);
    return it == use_counts_.end() ? 0 : it->second;
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) ++use_counts_[op.get()];
    IRVisitor::VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    if (op->then_body_) VisitStmt(op->then_body_);
    if (op->else_body_.has_value() && *op->else_body_) VisitStmt(*op->else_body_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }

 private:
  std::unordered_map<const Expr*, size_t> use_counts_;
};

bool IsLoopOrderingBoundaryCall(const CallPtr& call) {
  if (!call || !call->op_) return false;
  const auto& op_name = call->op_->name_;
  // Interprocedural effect summaries are not available here. A function call
  // may hide a fence, so moving a load across iterations of that call would be
  // speculative even when its declared writable roots are disjoint.
  if (!op_predicates::IsBuiltinOp(op_name)) return true;
  const auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return true;
  const auto& category = registry.GetEntry(op_name).GetOpCategory();
  // Cross-core communication has iteration-ordering semantics just like an
  // explicit fence.  A trailing tpush/tpop/tfree would otherwise let a load
  // move across the previous iteration's FIFO handshake.
  return category == "SyncOp" || category == "CrossCoreOp";
}

struct LoopResidencyInfo {
  uint64_t preorder{0};
  uint64_t postorder{0};
  std::unordered_set<const Expr*> yielded_values;
  bool has_ordering_boundary{false};
};

std::optional<uint64_t> StaticTileBytes(const TileTypePtr& tile) {
  if (!tile) return std::nullopt;
  uint64_t bytes = tile->dtype_.GetByte();
  if (bytes == 0) return std::nullopt;
  for (const auto& dim : tile->shape_) {
    auto extent = As<ConstInt>(dim);
    if (!extent || extent->value_ <= 0) return std::nullopt;
    const auto value = static_cast<uint64_t>(extent->value_);
    if (value != 0 && bytes > std::numeric_limits<uint64_t>::max() / value) return std::nullopt;
    bytes *= value;
  }
  return bytes;
}

class LoopResidencyInventory : public IRVisitor {
 public:
  void Analyze(const StmtPtr& body) {
    complete_uses_.Analyze(body);
    VisitStmt(body);
    BuildResidencyChains();
  }

  [[nodiscard]] const LoopResidencyInfo* GetLoopInfo(const ForStmtPtr& loop) const {
    auto it = loop_info_.find(loop.get());
    return it == loop_info_.end() ? nullptr : &it->second;
  }

  [[nodiscard]] const ForStmt* GetDefiningLoop(const Expr* var) const {
    auto it = defining_loop_.find(var);
    return it == defining_loop_.end() ? nullptr : it->second;
  }

  [[nodiscard]] bool IsDefinedInLoopSubtree(const ForStmtPtr& loop, const Expr* var) const {
    const auto* defining_loop = GetDefiningLoop(var);
    if (!defining_loop) return false;
    auto outer_it = loop_info_.find(loop.get());
    auto inner_it = loop_info_.find(defining_loop);
    if (outer_it == loop_info_.end() || inner_it == loop_info_.end()) return true;
    return outer_it->second.preorder <= inner_it->second.preorder &&
           inner_it->second.postorder <= outer_it->second.postorder;
  }

  [[nodiscard]] bool IsYieldedFromLoopSubtree(const ForStmtPtr& loop, const Expr* var) const {
    auto it = loop_info_.find(loop.get());
    return it != loop_info_.end() && it->second.yielded_values.count(var) != 0;
  }

  [[nodiscard]] const std::map<MemorySpace, std::vector<uint64_t>>& GetOwnedBufferSizes() const {
    return owned_buffer_sizes_;
  }

  [[nodiscard]] bool HasUnknownSize(MemorySpace space) const {
    return unknown_static_size_.count(space) != 0;
  }

  [[nodiscard]] bool HasExplicitReservation() const { return has_explicit_reservation_; }

  [[nodiscard]] bool IsResidencyChainVar(const Expr* var) const {
    return residency_chain_vars_.count(var) != 0;
  }

  [[nodiscard]] bool HasUnresolvedPipelineExpansion(MemorySpace space) const {
    return pipeline_expansion_spaces_.count(space) != 0;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (current_loop_ && op->var_) {
      defining_loop_[op->var_.get()] = current_loop_;
    }
    RecordOwnedTileFootprint(op->var_, op->value_);
    RecordDefinition(op);
    if (auto call = As<Call>(op->value_); call && IsOp(call, "system.reserve_buffer")) {
      has_explicit_reservation_ = true;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (current_loop_ && IsLoopOrderingBoundaryCall(op)) {
      loop_info_[current_loop_].has_ordering_boundary = true;
    }
    RecordDirectCallUses(op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    // Submit is asynchronous and may carry explicit dependency edges.  It is
    // a sibling IR kind of Call, so the Call visitor above never sees it.
    // Conservatively keep resident loads inside any loop subtree that
    // launches a task: otherwise a preheader load could move across a Submit
    // from the preceding iteration.
    if (current_loop_) {
      loop_info_[current_loop_].has_ordering_boundary = true;
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_); call && IsOp(call, "system.reserve_buffer")) {
      has_explicit_reservation_ = true;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (current_loop_) {
      VarUseCollector collector;
      for (const auto& value : op->value_) collector.VisitExpr(value);
      auto& yielded = loop_info_[current_loop_].yielded_values;
      yielded.insert(collector.GetUses().begin(), collector.GetUses().end());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    BindToCurrentLoop(op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    BindToCurrentLoop(op->return_vars_);
    BindToCurrentLoop(op->iter_args_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const ForStmt* parent = current_loop_;
    BindToCurrentLoop(op->return_vars_);
    auto& info = loop_info_[op.get()];
    info.preorder = next_order_++;

    current_loop_ = op.get();
    // This visitor replaces the base ForStmt traversal, so visit the header
    // expressions explicitly while the child loop is current.  Otherwise an
    // effectful iter-arg initializer (or a nested loop bound) is invisible to
    // the ordering analysis and a body load could move ahead of it.
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    const bool is_pipeline = op->kind_ == ForKind::Pipeline;
    if (is_pipeline) ++pipeline_depth_;
    if (op->loop_var_) defining_loop_[op->loop_var_.get()] = op.get();
    for (const auto& iter_arg : op->iter_args_) {
      if (!iter_arg) continue;
      defining_loop_[iter_arg.get()] = op.get();
      if (iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    VisitStmt(op->body_);
    if (is_pipeline) --pipeline_depth_;
    current_loop_ = parent;

    info.postorder = next_order_++;
    if (parent) MergeIntoParent(info, loop_info_[parent]);
  }

 private:
  std::map<const ForStmt*, LoopResidencyInfo> loop_info_;
  std::unordered_map<const Expr*, const ForStmt*> defining_loop_;
  std::map<MemorySpace, std::vector<uint64_t>> owned_buffer_sizes_;
  std::set<MemorySpace> unknown_static_size_;
  std::set<MemorySpace> pipeline_expansion_spaces_;
  struct DirectCallUse {
    CallPtr call;
    size_t arg_index;
  };
  struct CallDefinition {
    VarPtr var;
    CallPtr call;
  };
  std::unordered_map<const Expr*, CallDefinition> definitions_;
  std::unordered_map<const Expr*, std::vector<DirectCallUse>> direct_call_uses_;
  std::unordered_set<const Expr*> residency_chain_vars_;
  CompleteVarUseCollector complete_uses_;
  bool has_explicit_reservation_{false};
  const ForStmt* current_loop_{nullptr};
  size_t pipeline_depth_{0};
  uint64_t next_order_{0};

  template <typename VarContainer>
  void BindToCurrentLoop(const VarContainer& vars) {
    if (!current_loop_) return;
    for (const auto& var : vars) {
      if (var) defining_loop_[var.get()] = current_loop_;
    }
  }

  static bool OwnsAllocation(const ExprPtr& value) {
    if (AsVarLike(value)) return false;
    auto call = As<Call>(value);
    if (!call || !call->op_) return true;
    if (op_predicates::IsBufferAliasingViewOp(call->op_->name_)) return false;
    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(call->op_->name_)) return true;
    return !registry.GetEntry(call->op_->name_).GetOutputReusesInputArg().has_value();
  }

  void RecordOwnedTileFootprint(const VarPtr& var, const ExprPtr& value) {
    if (!var || !OwnsAllocation(value)) return;
    auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memory_space_.has_value()) return;
    const auto space = *tile->memory_space_;
    if (pipeline_depth_ != 0) pipeline_expansion_spaces_.insert(space);
    auto bytes = StaticTileBytes(tile);
    if (!bytes.has_value()) {
      unknown_static_size_.insert(space);
      return;
    }
    owned_buffer_sizes_[space].push_back(*bytes);
  }

  void RecordDefinition(const AssignStmtPtr& op) {
    if (!op || !op->var_) return;
    auto call = As<Call>(op->value_);
    if (!call) return;
    definitions_[op->var_.get()] = {op->var_, call};
  }

  void RecordDirectCallUses(const CallPtr& call) {
    if (!call) return;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      if (auto arg = AsVarLike(call->args_[i])) direct_call_uses_[arg.get()].push_back({call, i});
    }
  }

  static std::optional<std::pair<size_t, size_t>> MatmulOperandIndices(const CallPtr& call) {
    if (!call || !call->op_) return std::nullopt;
    if (IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_bias")) {
      return call->args_.size() >= 2 ? std::optional<std::pair<size_t, size_t>>({0, 1}) : std::nullopt;
    }
    if (IsOp(call, "tile.matmul_acc")) {
      return call->args_.size() >= 3 ? std::optional<std::pair<size_t, size_t>>({1, 2}) : std::nullopt;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool IsMatchingMatmulUse(const VarPtr& var) const {
    auto tile = var ? As<TileType>(var->GetType()) : nullptr;
    if (!tile || !tile->memory_space_.has_value()) return false;
    auto use_it = direct_call_uses_.find(var.get());
    if (complete_uses_.GetUseCount(var.get()) != 1 || use_it == direct_call_uses_.end() ||
        use_it->second.size() != 1) {
      return false;
    }
    const auto& use = use_it->second.front();
    const auto operand_indices = MatmulOperandIndices(use.call);
    if (!operand_indices.has_value()) return false;
    if (*tile->memory_space_ == MemorySpace::Left) return use.arg_index == operand_indices->first;
    if (*tile->memory_space_ == MemorySpace::Right) return use.arg_index == operand_indices->second;
    return false;
  }

  [[nodiscard]] bool HasOnlyUseBy(const VarPtr& var, const CallPtr& consumer) const {
    auto it = direct_call_uses_.find(var.get());
    return complete_uses_.GetUseCount(var.get()) == 1 && it != direct_call_uses_.end() &&
           it->second.size() == 1 && it->second.front().call.get() == consumer.get();
  }

  void BuildResidencyChains() {
    for (const auto& [_, terminal_def] : definitions_) {
      const auto& terminal_call = terminal_def.call;
      if ((!IsOp(terminal_call, "tile.move") && !IsOp(terminal_call, "tile.extract")) ||
          terminal_call->args_.empty() || !IsMatchingMatmulUse(terminal_def.var)) {
        continue;
      }
      auto terminal_tile = As<TileType>(terminal_def.var->GetType());
      auto source = AsVarLike(terminal_call->args_[0]);
      auto source_tile = source ? As<TileType>(source->GetType()) : nullptr;
      if (!terminal_tile || !source_tile || source_tile->GetMemorySpace() != MemorySpace::Mat) continue;

      std::vector<const Expr*> chain = {terminal_def.var.get()};
      CallPtr consumer = terminal_call;
      bool found_load = false;
      while (source && HasOnlyUseBy(source, consumer)) {
        auto def_it = definitions_.find(source.get());
        if (def_it == definitions_.end()) break;
        const auto& definition = def_it->second;
        auto result_tile = As<TileType>(definition.var->GetType());
        if (!result_tile || result_tile->GetMemorySpace() != MemorySpace::Mat) break;
        chain.push_back(definition.var.get());
        if (IsOp(definition.call, "tile.load")) {
          found_load = true;
          break;
        }
        if (!IsOp(definition.call, "tile.transpose_view") || definition.call->args_.empty()) break;
        consumer = definition.call;
        source = AsVarLike(definition.call->args_[0]);
      }
      if (found_load) residency_chain_vars_.insert(chain.begin(), chain.end());
    }
  }

  static void MergeIntoParent(const LoopResidencyInfo& child, LoopResidencyInfo& parent) {
    parent.yielded_values.insert(child.yielded_values.begin(), child.yielded_values.end());
    parent.has_ordering_boundary = parent.has_ordering_boundary || child.has_ordering_boundary;
  }
};

std::vector<StmtPtr> DirectStatements(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  return {body};
}

const Var* ResolveCanonicalRoot(const std::unordered_map<const Var*, const Var*>& roots, const Var* var) {
  const Var* current = var;
  std::unordered_set<const Var*> seen;
  while (current && seen.insert(current).second) {
    auto it = roots.find(current);
    if (it == roots.end() || !it->second) return nullptr;
    if (it->second == current) return current;
    current = it->second;
  }
  return nullptr;
}

/// Refine BufferRootCollector's dependency-region roots into storage roots for
/// the small set needed by residency.  In particular, tensor.slice and
/// tensor.view have their own dependency/metadata values but still alias their
/// source allocation, while tensor.assemble returns its target allocation.
/// Only tensor.create is treated as a trusted fresh external-disjoint
/// allocation; all other unknown producers remain untrusted.
class CallerStorageProvenanceCollector : public IRVisitor {
 public:
  explicit CallerStorageProvenanceCollector(std::unordered_map<const Var*, const Var*> roots)
      : storage_roots_(std::move(roots)) {}

  [[nodiscard]] const std::unordered_map<const Var*, const Var*>& GetStorageRoots() const {
    return storage_roots_;
  }

  [[nodiscard]] const std::unordered_set<const Var*>& GetFreshRoots() const { return fresh_roots_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      if (auto call = As<Call>(op->value_)) {
        if (IsOp(call, "tensor.create")) {
          storage_roots_[op->var_.get()] = op->var_.get();
          fresh_roots_.insert(op->var_.get());
        } else if ((IsOp(call, "tensor.slice") || IsOp(call, "tensor.assemble") ||
                    IsOp(call, "tensor.view")) &&
                   !call->args_.empty()) {
          SetAliasedRoot(op->var_.get(), call->args_[0]);
        }
      } else if (auto source = AsVarLike(op->value_)) {
        SetAliasedRoot(op->var_.get(), source);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::unordered_map<const Var*, const Var*> storage_roots_;
  std::unordered_set<const Var*> fresh_roots_;

  void SetAliasedRoot(const Var* result, const ExprPtr& source_expr) {
    auto source = AsVarLike(source_expr);
    const Var* root = source ? ResolveCanonicalRoot(storage_roots_, source.get()) : nullptr;
    if (root) {
      storage_roots_[result] = root;
    } else {
      storage_roots_.erase(result);
    }
  }
};

class FunctionWriteRootCollector : public IRVisitor {
 public:
  FunctionWriteRootCollector(ProgramPtr program,
                             const std::unordered_map<const Var*, const Var*>& buffer_roots)
      : program_(std::move(program)), buffer_roots_(buffer_roots) {}

  [[nodiscard]] const std::unordered_set<const Var*>& GetWriteRoots() const { return write_roots_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    RecordWrites(op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordWrites(transform_utils::AsCallOrSubmitView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  ProgramPtr program_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  std::unordered_set<const Var*> write_roots_;

  void RecordArgRoot(const ExprPtr& arg) {
    auto var = AsVarLike(arg);
    if (!var) return;
    if (const Var* root = ResolveCanonicalRoot(buffer_roots_, var.get())) write_roots_.insert(root);
  }

  void RecordWrites(const CallPtr& call) {
    if (!call || !call->op_) return;
    if ((IsOp(call, "tile.store") || IsOp(call, "tile.mscatter")) && call->args_.size() >= 3) {
      RecordArgRoot(call->args_[2]);
      return;
    }
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return;
    const size_t count = std::min(call->args_.size(), callee->param_directions_.size());
    for (size_t i = 0; i < count; ++i) {
      if (callee->param_directions_[i] == ParamDirection::Out ||
          callee->param_directions_[i] == ParamDirection::InOut) {
        RecordArgRoot(call->args_[i]);
      }
    }
  }
};

struct ReadParamEvidence {
  size_t call_sites{0};
  bool all_sites_safe{true};
};

class InProgramCallTargetCollector : public IRVisitor {
 public:
  explicit InProgramCallTargetCollector(ProgramPtr program) : program_(std::move(program)) {}

  [[nodiscard]] const std::unordered_set<const Function*>& GetCalledFunctions() const {
    return called_functions_;
  }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    RecordCallTarget(op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordCallTarget(transform_utils::AsCallOrSubmitView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  ProgramPtr program_;
  std::unordered_set<const Function*> called_functions_;

  void RecordCallTarget(const CallPtr& call) {
    if (!call || !call->op_ || !program_) return;
    auto callee = program_->GetFunction(call->op_->name_);
    if (callee) called_functions_.insert(callee.get());
  }
};

class CallSiteReadSafetyCollector : public IRVisitor {
 public:
  CallSiteReadSafetyCollector(ProgramPtr program, bool caller_is_root_orchestration,
                              const std::unordered_map<const Var*, const Var*>& buffer_roots,
                              const std::unordered_set<const Var*>& fresh_roots,
                              std::unordered_map<const Var*, ReadParamEvidence>* evidence)
      : program_(std::move(program)),
        caller_is_root_orchestration_(caller_is_root_orchestration),
        buffer_roots_(buffer_roots),
        fresh_roots_(fresh_roots),
        evidence_(evidence) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    RecordEvidence(op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    PoisonEvidence(transform_utils::AsCallOrSubmitView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  ProgramPtr program_;
  bool caller_is_root_orchestration_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_set<const Var*>& fresh_roots_;
  std::unordered_map<const Var*, ReadParamEvidence>* evidence_;

  [[nodiscard]] const Var* ResolveRoot(const ExprPtr& arg) const {
    auto var = AsVarLike(arg);
    return var ? ResolveCanonicalRoot(buffer_roots_, var.get()) : nullptr;
  }

  void PoisonEvidence(const CallPtr& call) {
    if (!call || !call->op_ || !program_) return;
    auto callee = program_->GetFunction(call->op_->name_);
    if (!callee || callee->func_type_ != FunctionType::InCore) return;
    for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] != ParamDirection::In ||
          !As<TensorType>(callee->params_[i]->GetType())) {
        continue;
      }
      auto& evidence = (*evidence_)[callee->params_[i].get()];
      ++evidence.call_sites;
      evidence.all_sites_safe = false;
    }
  }

  void RecordEvidence(const CallPtr& call) {
    if (!call || !call->op_ || !program_) return;
    auto callee = program_->GetFunction(call->op_->name_);
    if (!callee || callee->func_type_ != FunctionType::InCore) return;

    // Only a direct root-orchestration call may contribute evidence, and even
    // there an external parameter is not a no-alias fact.  Non-root calls and
    // Submit sites poison every Tensor In parameter they reach.
    if (!caller_is_root_orchestration_) {
      PoisonEvidence(call);
      return;
    }

    std::unordered_set<const Var*> write_roots;
    bool all_write_roots_known = true;
    for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
      const auto direction = callee->param_directions_[i];
      if (direction != ParamDirection::Out && direction != ParamDirection::InOut) continue;
      if (!As<TensorType>(callee->params_[i]->GetType())) continue;
      const Var* root = i < call->args_.size() ? ResolveRoot(call->args_[i]) : nullptr;
      if (root) {
        write_roots.insert(root);
      } else {
        all_write_roots_known = false;
      }
    }

    for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] != ParamDirection::In ||
          !As<TensorType>(callee->params_[i]->GetType())) {
        continue;
      }
      auto& evidence = (*evidence_)[callee->params_[i].get()];
      ++evidence.call_sites;
      const Var* read_root = i < call->args_.size() ? ResolveRoot(call->args_[i]) : nullptr;
      if (!read_root || fresh_roots_.count(read_root) == 0 || !all_write_roots_known ||
          write_roots.count(read_root) != 0) {
        evidence.all_sites_safe = false;
      }
    }
  }
};

std::unordered_set<const Var*> CollectProvenSafeReadParams(const ProgramPtr& program) {
  std::unordered_map<const Var*, ReadParamEvidence> evidence;
  for (const auto& [_, func] : program->functions_) {
    if (!func || !func->body_) continue;
    if (func->func_type_ != FunctionType::InCore) continue;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (func->param_directions_[i] == ParamDirection::In && As<TensorType>(func->params_[i]->GetType())) {
        evidence.emplace(func->params_[i].get(), ReadParamEvidence{});
      }
    }
  }

  InProgramCallTargetCollector call_targets(program);
  for (const auto& [_, func] : program->functions_) {
    if (!func || !func->body_) continue;
    call_targets.VisitStmt(func->body_);
  }
  const auto& called_functions = call_targets.GetCalledFunctions();

  for (const auto& [_, func] : program->functions_) {
    if (!func || !func->body_) continue;
    buffer_root::BufferRootCollector roots(program);
    roots.Initialize(func->params_);
    roots.VisitStmt(func->body_);
    CallerStorageProvenanceCollector storage(roots.buffer_roots);
    storage.VisitStmt(func->body_);
    const bool is_root_orchestration =
        func->func_type_ == FunctionType::Orchestration && called_functions.count(func.get()) == 0;
    CallSiteReadSafetyCollector calls(program, is_root_orchestration, storage.GetStorageRoots(),
                                      storage.GetFreshRoots(), &evidence);
    calls.VisitStmt(func->body_);
  }
  std::unordered_set<const Var*> safe;
  for (const auto& [_, func] : program->functions_) {
    if (!func || !func->body_) continue;
    if (func->func_type_ != FunctionType::InCore) continue;
    for (const auto& func_param : func->params_) {
      const auto* param = func_param.get();
      auto it = evidence.find(param);
      if (it != evidence.end() && it->second.call_sites != 0 && it->second.all_sites_safe) {
        safe.insert(param);
      }
    }
  }
  return safe;
}

class LoopInvariantTileLoadHoister : public IRMutator {
 public:
  LoopInvariantTileLoadHoister(const FunctionPtr& func, const ProgramPtr& program,
                               const LoopResidencyInventory& inventory,
                               const std::unordered_set<const Var*>& proven_safe_read_params)
      : inventory_(inventory), proven_safe_read_params_(proven_safe_read_params) {
    INTERNAL_CHECK_SPAN(func->params_.size() == func->param_directions_.size(), func->span_)
        << "Internal error: function parameter and direction counts differ";
    for (size_t i = 0; i < func->params_.size(); ++i) {
      param_directions_[func->params_[i].get()] = func->param_directions_[i];
    }
    buffer_root::BufferRootCollector roots(program);
    roots.Initialize(func->params_);
    roots.VisitStmt(func->body_);
    CallerStorageProvenanceCollector storage(std::move(roots.buffer_roots));
    storage.VisitStmt(func->body_);
    buffer_roots_ = storage.GetStorageRoots();
    FunctionWriteRootCollector writes(program, buffer_roots_);
    writes.VisitStmt(func->body_);
    write_roots_ = writes.GetWriteRoots();
    // Memory-space inference historically works without a configured backend.
    // Residency needs concrete capacity limits, so leave it disabled in that
    // backend-neutral mode instead of making the whole pass require a target.
    if (backend::BackendConfig::IsConfigured()) {
      const auto* ctx = PassContext::Current();
      handler_ = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
      allocation_policy_ = pypto::backend::GetBackend()->CreateMemoryAllocatorPolicy();
      BuildFunctionFootprints();
    }
  }

 protected:
  static bool IsHoistPrefixEffectBoundary(const StmtPtr& stmt, const CallPtr& call) {
    if (dce::IsSideEffectOp(stmt)) return true;
    if (!call || !call->op_) return false;

    return IsLoopOrderingBoundaryCall(call);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Decide from the original direct body before recursively rewriting child
    // loops. This prevents a preheader produced for an inner loop from being
    // rescanned and bubbled again at every enclosing level (O(N * depth)).
    // Nested loops are still visited by the base mutator below and may each
    // hoist their own original direct chain by one lexical level.
    if (!IsStaticNonEmptySequential(op)) return IRMutator::VisitStmt_(op);
    const auto* loop_info = inventory_.GetLoopInfo(op);
    // A load moved to the preheader would cross an ordering operation from
    // every preceding iteration even when that operation is textually after
    // the load. Reject any known or potentially hidden boundary in the loop
    // subtree.
    if (!loop_info || !handler_ || loop_info->has_ordering_boundary) return IRMutator::VisitStmt_(op);

    auto stmts = DirectStatements(op->body_);
    std::vector<bool> hoist(stmts.size(), false);
    std::unordered_set<const Expr*> invariant_chain;
    bool changed = false;

    for (size_t i = 0; i < stmts.size(); ++i) {
      auto assign = As<AssignStmt>(stmts[i]);
      // A direct control-flow/effect statement before a candidate can bypass
      // the remainder of the iteration (for example, `continue`). Stop at
      // that boundary rather than speculating a later load into the preheader.
      if (!assign) break;
      if (!assign->var_ || !inventory_.IsResidencyChainVar(assign->var_.get()) ||
          inventory_.IsYieldedFromLoopSubtree(op, assign->var_.get())) {
        // An assigned store, cross-core synchronization op, Submit, or call to
        // another function is still an effect boundary. Do not move a later
        // load ahead of it merely because its result is assigned to a Var.
        auto preceding_call = assign->value_ ? As<Call>(assign->value_) : nullptr;
        if (IsHoistPrefixEffectBoundary(stmts[i], preceding_call)) {
          break;
        }
        continue;
      }
      auto call = As<Call>(assign->value_);
      if (!call) continue;  // Submit and non-call expressions are never moved.

      bool eligible = false;
      if (IsOp(call, "tile.load")) {
        eligible = IsReadOnlyMatLoad(call, assign->var_) && UsesAreLoopInvariant(call, op, invariant_chain);
      } else if (IsResidencyChainOp(call)) {
        auto source = call->args_.empty() ? nullptr : AsVarLike(call->args_[0]);
        eligible = source && invariant_chain.count(source.get()) != 0 && IsResidencySpace(assign->var_) &&
                   UsesAreLoopInvariant(call, op, invariant_chain);
      }

      if (!eligible || !CapacitySafe(assign->var_)) continue;
      hoist[i] = true;
      invariant_chain.insert(assign->var_.get());
      changed = true;
    }

    if (!changed) return IRMutator::VisitStmt_(op);

    std::vector<StmtPtr> preheader;
    std::vector<StmtPtr> body;
    preheader.reserve(stmts.size());
    body.reserve(stmts.size());
    for (size_t i = 0; i < stmts.size(); ++i) {
      (hoist[i] ? preheader : body).push_back(stmts[i]);
    }

    auto new_loop = MutableCopy(op);
    new_loop->body_ = SeqStmts::Flatten(std::move(body), op->body_->span_);
    // Invoke the base implementation directly for this loop so only its
    // remaining body is recursively rewritten; do not re-enter this override
    // and reconsider the same direct statements.
    ForStmtPtr rewritten_loop = new_loop;
    preheader.push_back(IRMutator::VisitStmt_(rewritten_loop));
    return SeqStmts::Flatten(std::move(preheader), op->span_);
  }

 private:
  const LoopResidencyInventory& inventory_;
  const std::unordered_set<const Var*>& proven_safe_read_params_;
  std::unordered_map<const Expr*, ParamDirection> param_directions_;
  std::unordered_map<const Var*, const Var*> buffer_roots_;
  std::unordered_set<const Var*> write_roots_;
  const backend::BackendHandler* handler_{nullptr};
  MemoryAllocatorPolicyPtr allocation_policy_;
  std::map<MemorySpace, uint64_t> function_footprints_;
  std::set<MemorySpace> invalid_footprints_;

  static bool IsStaticNonEmptySequential(const ForStmtPtr& loop) {
    if (loop->kind_ != ForKind::Sequential) return false;
    auto start = As<ConstInt>(loop->start_);
    auto stop = As<ConstInt>(loop->stop_);
    auto step = As<ConstInt>(loop->step_);
    return start && stop && step && step->value_ > 0 && start->value_ < stop->value_;
  }

  [[nodiscard]] bool IsReadOnlyMatLoad(const CallPtr& call, const VarPtr& result) const {
    auto result_tile = As<TileType>(result->GetType());
    if (!result_tile || result_tile->GetMemorySpace() != MemorySpace::Mat || call->args_.empty() ||
        !call->GetAttr<bool>(kCompilerTensorToTileMatBridgeAttr, false)) {
      return false;
    }
    auto source = AsVarLike(call->args_[0]);
    if (!source || !As<TensorType>(source->GetType())) return false;
    auto it = param_directions_.find(source.get());
    if (it == param_directions_.end() || it->second != ParamDirection::In ||
        proven_safe_read_params_.count(source.get()) == 0) {
      return false;
    }
    auto root_it = buffer_roots_.find(source.get());
    return root_it != buffer_roots_.end() && root_it->second && write_roots_.count(root_it->second) == 0;
  }

  static bool IsResidencyChainOp(const CallPtr& call) {
    return IsOp(call, "tile.transpose_view") || IsOp(call, "tile.move") || IsOp(call, "tile.extract");
  }

  static bool IsResidencySpace(const VarPtr& result) {
    auto tile = As<TileType>(result->GetType());
    if (!tile || !tile->memory_space_.has_value()) return false;
    const auto space = *tile->memory_space_;
    return space == MemorySpace::Mat || space == MemorySpace::Left || space == MemorySpace::Right;
  }

  [[nodiscard]] bool UsesAreLoopInvariant(const ExprPtr& expr, const ForStmtPtr& loop,
                                          const std::unordered_set<const Expr*>& already_hoisted) const {
    VarUseCollector collector;
    collector.VisitExpr(expr);
    for (const auto* use : collector.GetUses()) {
      if (already_hoisted.count(use) != 0) continue;
      if (inventory_.IsDefinedInLoopSubtree(loop, use)) return false;
    }
    return true;
  }

  [[nodiscard]] std::optional<uint64_t> Capacity(MemorySpace space) const {
    switch (space) {
      case MemorySpace::Mat:
        return handler_->GetMatCapacityBytes();
      case MemorySpace::Left:
        return handler_->GetL0aCapacityBytes();
      case MemorySpace::Right:
        return handler_->GetL0bCapacityBytes();
      default:
        return std::nullopt;
    }
  }

  void BuildFunctionFootprints() {
    if (!allocation_policy_) return;
    for (const auto& [space, sizes] : inventory_.GetOwnedBufferSizes()) {
      SpaceFootprint footprint(space, *allocation_policy_);
      uint64_t high_water = 0;
      for (uint64_t bytes : sizes) {
        if (high_water > std::numeric_limits<uint64_t>::max() - bytes) {
          invalid_footprints_.insert(space);
          break;
        }
        (void)footprint.OpenBuffer(bytes);
        high_water = footprint.HighWater();
      }
      if (invalid_footprints_.count(space) == 0) {
        function_footprints_[space] = footprint.HighWater();
      }
    }
  }

  [[nodiscard]] bool CapacitySafe(const VarPtr& result) const {
    auto tile = As<TileType>(result->GetType());
    if (!tile || !tile->memory_space_.has_value()) return false;
    const auto space = *tile->memory_space_;
    auto capacity = Capacity(space);
    if (!capacity.has_value() || *capacity == 0 || !allocation_policy_ || inventory_.HasUnknownSize(space) ||
        inventory_.HasUnresolvedPipelineExpansion(space) || inventory_.HasExplicitReservation() ||
        invalid_footprints_.count(space) != 0) {
      return false;
    }

    // Both memory planners run after this transform and see the extended
    // lifetime. Requiring every allocation-owning tile in the whole function
    // to fit simultaneously is stronger than either planner's liveness
    // packing, but it guarantees residency cannot turn a compilable function
    // into a later capacity failure. Align every owner exactly as
    // AllocateMemoryAddr does.
    auto it = function_footprints_.find(space);
    return it == function_footprints_.end() || it->second <= *capacity;
  }
};

/// Strip the private bridge marker after residency has consumed it. Keeping
/// the provenance transient avoids leaking an implementation detail into
/// downstream allocation and codegen IR.
class StripCompilerMatBridgeAttrMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call || !call->HasAttr(kCompilerTensorToTileMatBridgeAttr)) return visited;
    auto attrs = StripAttr(call->attrs_, kCompilerTensorToTileMatBridgeAttr);
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                  call->span_);
  }
};

FunctionPtr StripCompilerMatBridgeAttrs(const FunctionPtr& func) {
  if (!func || !func->body_) return func;
  StripCompilerMatBridgeAttrMutator stripper;
  auto body = stripper.VisitStmt(func->body_);
  if (body.get() == func->body_.get()) return func;
  auto clean = MutableCopy(func);
  clean->body_ = std::move(body);
  return clean;
}

FunctionPtr RewriteFunction(const FunctionPtr& func, const ProgramPtr& program,
                            const std::unordered_set<const Var*>& proven_safe_read_params) {
  if (!func || !func->body_ || func->func_type_ != FunctionType::InCore) {
    return StripCompilerMatBridgeAttrs(func);
  }
  LoopResidencyInventory inventory;
  inventory.Analyze(func->body_);
  LoopInvariantTileLoadHoister hoister(func, program, inventory, proven_safe_read_params);
  auto rewritten = MutableCopy(func);
  rewritten->body_ = hoister.VisitStmt(func->body_);
  return NormalizeStmtStructure(StripCompilerMatBridgeAttrs(rewritten));
}

}  // namespace

ProgramPtr Apply(const ProgramPtr& program) {
  if (!program) return program;
  const auto proven_safe_read_params = CollectProvenSafeReadParams(program);
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  for (const auto& [gvar, func] : program->functions_) {
    new_functions[gvar] = RewriteFunction(func, program, proven_safe_read_params);
  }
  return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace loop_invariant_mat_residency
}  // namespace ir
}  // namespace pypto
