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
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::GetLastYieldStmt;

namespace {

// Unregistered cube ops (not yet registered via REGISTER_OP but still need Acc output)
const std::unordered_set<std::string> kUnregisteredCubeOps = {"tile.matmul_mx", "tile.matmul_mx_acc",
                                                              "tile.matmul_mx_bias"};

// Look up input constraints for an op. Returns nullptr if none.
const std::vector<std::vector<MemorySpace>>* GetInputConstraints(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return nullptr;
  const auto& spec_opt = registry.GetEntry(op_name).GetMemorySpec();
  if (!spec_opt.has_value()) return nullptr;
  return &spec_opt->input_constraints;
}

// Prefer the non-Vec space when two demands collide on the same var. Vec acts as
// the permissive default, so a specialized demand (Mat, Left, Right, Acc) wins.
bool ShouldOverrideDemand(MemorySpace existing, MemorySpace incoming) {
  return existing == MemorySpace::Vec && incoming != MemorySpace::Vec;
}

// ============================================================================
// Phase 0: Backward demand collection
//
// For each op with `input_constraints`, record "this input var is demanded to
// live in this space". Then propagate demands backward through ops registered
// with `set_output_memory_inherit_input()` to a fixed point so that chains like
//   slice(tensor) -> fillpad -> matmul
// push the matmul's Mat demand back through fillpad onto the slice's output,
// enabling the downstream Phase 1 analyzer to resolve the slice-produced tile
// directly to Mat instead of routing through Vec.
// ============================================================================

class DemandCollector : public IRVisitor {
 public:
  [[nodiscard]] const std::map<VarPtr, MemorySpace>& GetDemands() const { return demands_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      RecordDirectDemands(call);
      RecordInheritInputEdge(op->var_, call);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_)) RecordDirectDemands(call);
    IRVisitor::VisitStmt_(op);
  }

  /// Propagate demand backward through OutputMemoryInheritsInput() ops.
  /// Edges `dst -> src` are captured in program order during the forward visit;
  /// since the inherit-input relation flows strictly backward (dst defined
  /// after src), a single reverse-order sweep reaches the fixed point in O(N).
  void PropagateThroughInheritInputOps() {
    for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
      const auto& [dst, src] = *it;
      auto out_it = demands_.find(dst);
      if (out_it == demands_.end()) continue;
      auto [ins_it, inserted] = demands_.try_emplace(src, out_it->second);
      if (!inserted && ShouldOverrideDemand(ins_it->second, out_it->second)) {
        ins_it->second = out_it->second;
      }
    }
  }

 private:
  std::map<VarPtr, MemorySpace> demands_;
  // `dst -> src` edges for ops with OutputMemoryInheritsInput(), captured in
  // program order. Walked in reverse in PropagateThroughInheritInputOps.
  std::vector<std::pair<VarPtr, VarPtr>> edges_;

  void RecordDirectDemands(const CallPtr& call) {
    auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(call->op_->name_)) return;
    const auto& spec = reg.GetEntry(call->op_->name_).GetMemorySpec();
    if (!spec.has_value()) return;
    for (size_t i = 0; i < spec->input_constraints.size() && i < call->args_.size(); ++i) {
      const auto& allowed = spec->input_constraints[i];
      if (allowed.empty()) continue;
      auto var = As<Var>(call->args_[i]);
      if (!var) continue;
      // Preferred space: the first allowed entry. Backends are expected to list
      // the canonical choice first (e.g. tile.store uses {Vec, Acc} — a Vec
      // producer needs no move, and Acc-origin tiles keep their space).
      MemorySpace demand = allowed[0];
      auto [it, inserted] = demands_.try_emplace(var, demand);
      if (!inserted && ShouldOverrideDemand(it->second, demand)) {
        it->second = demand;
      }
    }
  }

  void RecordInheritInputEdge(const VarPtr& dst, const CallPtr& call) {
    if (!dst) return;
    auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(call->op_->name_)) return;
    if (!reg.GetEntry(call->op_->name_).OutputMemoryInheritsInput()) return;
    for (const auto& arg : call->args_) {
      auto var = As<Var>(arg);
      if (!var) continue;
      if (!As<TileType>(var->GetType()) && !As<TensorType>(var->GetType())) continue;
      edges_.emplace_back(dst, var);
      break;  // first tile-typed input only (matches inherit-input semantics)
    }
  }
};

// ============================================================================
// Phase 1: Analyze - infer memory_space for each tile variable
// ============================================================================

class TileMemorySpaceAnalyzer : public IRVisitor {
 public:
  TileMemorySpaceAnalyzer(const std::vector<VarPtr>& params, const std::map<VarPtr, MemorySpace>& demands)
      : demands_(demands) {
    for (const auto& var : params) {
      CHECK(!As<TileType>(var->GetType())) << "InCore function parameter '" << var->name_hint_
                                           << "' has TileType, but InCore parameters must be TensorType";
    }
  }

  [[nodiscard]] const std::map<VarPtr, MemorySpace>& GetVarMemory() const { return var_memory_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op->var_ || !As<TileType>(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    if (auto call = As<Call>(op->value_)) {
      const std::string& op_name = call->op_->name_;
      if (op_name.rfind("tile.", 0) == 0) {
        var_memory_[op->var_] = InferFromOp(op_name, call, op->var_);
      } else {
        // Non-tile ops producing TileType: default to Vec
        var_memory_[op->var_] = MemorySpace::Vec;
      }
    } else if (auto src_var = As<Var>(op->value_)) {
      // Plain SSA alias `y = x`. Inherit x's memory space onto y so later
      // phases (MoveCollector, Phase 3) see a consistent memory_space on the
      // alias. The Python frontend emits these when eliding no-op
      // tensor.fillpad(pad=zero) calls whose input already has a matching
      // valid_shape — the alias is value-identical to its source.
      auto src_it = var_memory_.find(src_var);
      if (src_it != var_memory_.end()) {
        var_memory_[op->var_] = src_it->second;
      }
    }

    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);

    if (op->return_vars_.empty()) return;

    auto yield_stmt = GetLastYieldStmt(op->body_);
    if (!yield_stmt) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!As<TileType>(op->return_vars_[i]->GetType())) continue;
      if (i >= yield_stmt->value_.size()) continue;
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      if (!yield_var) continue;

      // Fallback to the TileType annotation handles IfStmt return_vars — they
      // carry a memory_space_ set by earlier passes but never get re-tracked
      // in var_memory_ since this analyzer only visits AssignStmts.
      std::optional<MemorySpace> yield_memory;
      if (auto it = var_memory_.find(yield_var); it != var_memory_.end()) {
        yield_memory = it->second;
      } else if (auto yt = As<TileType>(yield_var->GetType()); yt) {
        yield_memory = yt->memory_space_;
      }
      if (!yield_memory.has_value()) continue;

      var_memory_[op->return_vars_[i]] = *yield_memory;

      // Back-propagation handles the accumulator pattern: a tile.create
      // conservatively defaults to Mem.Vec but the loop body writes a
      // different space (e.g. Acc from matmul_acc). Without this override the
      // final tile.store reads a Vec tile and ExpandMixedKernel misclassifies
      // the kernel as mixed, producing broken AIC/AIV IR.
      if (i < op->iter_args_.size()) {
        var_memory_[op->iter_args_[i]] = *yield_memory;
        // Any TileType init carrier needs to agree with the promoted iter_arg,
        // whether or not the analyzer has already recorded it — e.g. an IfStmt
        // return_var used as the loop init is never visited by the AssignStmt
        // path, so it would otherwise keep its old memory space.
        if (auto init_var = As<Var>(op->iter_args_[i]->initValue_);
            init_var && As<TileType>(init_var->GetType())) {
          var_memory_[init_var] = *yield_memory;
        }
      }
    }
  }

 private:
  const std::map<VarPtr, MemorySpace>& demands_;
  std::map<VarPtr, MemorySpace> var_memory_;

  MemorySpace InferFromOp(const std::string& op_name, const CallPtr& call, const VarPtr& out_var) {
    auto& registry = OpRegistry::GetInstance();

    // Handle unregistered ops (backward compat)
    if (!registry.IsRegistered(op_name)) {
      if (kUnregisteredCubeOps.count(op_name) > 0) return MemorySpace::Acc;
      return MemorySpace::Vec;
    }

    const auto& entry = registry.GetEntry(op_name);
    const auto& spec_opt = entry.GetMemorySpec();
    if (!spec_opt.has_value() || !spec_opt->deduce_output_memory) {
      // no_memory_spec ops (e.g. tile.tpop_*): read memory_space from Call return type
      if (auto tile_type = As<TileType>(call->GetType())) {
        if (tile_type->memory_space_.has_value() && *tile_type->memory_space_ != MemorySpace::DDR) {
          return *tile_type->memory_space_;
        }
      }
      return MemorySpace::Vec;
    }

    auto result = spec_opt->deduce_output_memory(call->kwargs_);
    if (result.has_value()) {
      return *result;
    }

    // Resolver returned nullopt — kwarg absent. Two cases:
    // (1) Inherit-input op (fillpad/slice/...): output = first tile input's
    //     space. Demand back-prop ensures input is or will be resolved to
    //     match consumer demand.
    // (2) Retargetable producer whose kwarg is absent (e.g. a converter chose
    //     to let the pass decide): consult backward demand, then fall back.
    // We never override a present kwarg — a Left/Right/Acc demand from a
    // compute op (matmul) cannot be satisfied by a DDR load directly and must
    // still route through Mat with a subsequent tile.move.
    if (spec_opt->output_inherits_input) {
      return InheritFromInput(call).value_or(MemorySpace::Vec);
    }
    if (entry.HasRetargetableMemoryKwarg()) {
      auto demand_it = demands_.find(out_var);
      if (demand_it != demands_.end()) {
        MemorySpace demand = demand_it->second;
        // Retargetable DDR-facing producers (tile.load) can only directly
        // produce {Vec, Mat}; specialized demands (Left/Right/Acc/Bias) from
        // downstream compute ops (matmul etc.) must be reached via a
        // tile.move inserted by Phase 2 MoveCollector. Clamping here keeps
        // the producer's output hardware-valid and preserves the move chain.
        if (demand == MemorySpace::Vec || demand == MemorySpace::Mat) return demand;
      }
    }
    return InheritFromInput(call).value_or(MemorySpace::Vec);
  }

  std::optional<MemorySpace> InheritFromInput(const CallPtr& call) {
    for (const auto& arg : call->args_) {
      if (auto var = As<Var>(arg)) {
        auto it = var_memory_.find(var);
        if (it != var_memory_.end()) {
          return it->second;
        }
      }
    }
    return std::nullopt;
  }
};

// ============================================================================
// Phase 2: Collect needed tile.move insertions for input constraint mismatches
// ============================================================================

// Key: (producer variable, target memory space)
using MoveKey = std::pair<VarPtr, MemorySpace>;
struct MoveKeyLess {
  bool operator()(const MoveKey& a, const MoveKey& b) const {
    if (a.first != b.first) return a.first < b.first;
    return static_cast<int>(a.second) < static_cast<int>(b.second);
  }
};

class MoveCollector : public IRVisitor {
 public:
  explicit MoveCollector(const std::map<VarPtr, MemorySpace>& var_memory) : var_memory_(var_memory) {}

  [[nodiscard]] const std::set<MoveKey, MoveKeyLess>& GetNeededMoves() const { return needed_moves_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      CheckInputConstraints(call);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_)) {
      CheckInputConstraints(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_;
  std::set<MoveKey, MoveKeyLess> needed_moves_;

  void CheckInputConstraints(const CallPtr& call) {
    const auto* constraints = GetInputConstraints(call->op_->name_);
    if (!constraints) return;

    for (size_t i = 0; i < constraints->size() && i < call->args_.size(); ++i) {
      const auto& allowed_spaces = (*constraints)[i];
      if (allowed_spaces.empty()) continue;

      auto var = As<Var>(call->args_[i]);
      if (!var) continue;
      auto it = var_memory_.find(var);
      if (it == var_memory_.end()) continue;

      bool allowed =
          std::find(allowed_spaces.begin(), allowed_spaces.end(), it->second) != allowed_spaces.end();
      if (!allowed) {
        needed_moves_.insert({var, allowed_spaces[0]});
      }
    }
  }
};

// ============================================================================
// Phase 3: Mutate - set memory_space_, insert tile.move, substitute args
// ============================================================================

class TileMemorySpaceMutator : public IRMutator {
 public:
  TileMemorySpaceMutator(const std::map<VarPtr, MemorySpace>& var_memory,
                         const std::set<MoveKey, MoveKeyLess>& needed_moves)
      : var_memory_(var_memory), needed_moves_(needed_moves) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }

    auto tile_type = As<TileType>(op->GetType());
    auto mem_it = var_memory_.find(op);

    if (tile_type && mem_it != var_memory_.end()) {
      // When promoting to a new memory space, refresh the TileView to the
      // implicit view for the target — the old view (e.g. Vec-style defaults
      // from a tile.create) becomes a layout mismatch once the memory space
      // changes (e.g. Acc expects col_major/row_major/fractal=1024 rather
      // than the Vec-style row_major/none_box/fractal=512).
      std::optional<TileView> new_view = tile_type->tile_view_;
      if (tile_type->memory_space_ != mem_it->second) {
        new_view = tile_view_semantics::GetImplicitTileView(tile_type->shape_, mem_it->second);
      }
      auto new_type = std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                                 std::move(new_view), mem_it->second);
      auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
      var_cache_[op] = new_var;
      return new_var;
    }

    var_cache_[op] = op;
    return op;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    const auto* constraints = GetInputConstraints(op->op_->name_);

    std::vector<ExprPtr> new_args;
    bool changed = false;
    new_args.reserve(op->args_.size());

    for (size_t i = 0; i < op->args_.size(); ++i) {
      bool substituted = false;
      if (constraints && i < constraints->size() && !(*constraints)[i].empty()) {
        if (auto var = As<Var>(op->args_[i])) {
          MoveKey key = {var, (*constraints)[i][0]};
          auto move_it = created_moves_.find(key);
          if (move_it != created_moves_.end()) {
            new_args.push_back(move_it->second);
            changed = true;
            substituted = true;
          }
        }
      }
      if (!substituted) {
        auto new_arg = IRMutator::VisitExpr(op->args_[i]);
        new_args.push_back(new_arg);
        if (new_arg.get() != op->args_[i].get()) changed = true;
      }
    }

    if (!changed) return op;
    // GlobalVar calls and unregistered ops bypass OpRegistry — reconstruct directly.
    auto& registry = OpRegistry::GetInstance();
    if (As<GlobalVar>(op->op_) || !registry.IsRegistered(op->op_->name_)) {
      return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, op->GetType(), op->span_);
    }
    return registry.Create(op->op_->name_, new_args, op->kwargs_, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    auto new_var = As<Var>(new_var_expr);
    if (!new_var) {
      if (new_var_expr.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
      return std::make_shared<AssignStmt>(As<Var>(new_var_expr), new_value, op->span_);
    }

    // Rewrite retargetable producers' target_memory kwarg so it matches the
    // resolved memory space. Covers tile.create / tile.load / any op registered
    // with HasRetargetableMemoryKwarg(): if Phase 1 resolved the output to a
    // different space than the kwarg says (or the kwarg is absent because the
    // converter let the pass decide), we rewrite the call so codegen reads a
    // consistent value and the result type gets a fresh implicit TileView.
    if (auto call = As<Call>(new_value); call) {
      auto& registry = OpRegistry::GetInstance();
      const std::string& call_op_name = call->op_->name_;
      if (registry.IsRegistered(call_op_name) &&
          registry.GetEntry(call_op_name).HasRetargetableMemoryKwarg()) {
        auto mem_it = var_memory_.find(op->var_);
        auto old_call_type = As<TileType>(call->GetType());
        if (mem_it != var_memory_.end() && old_call_type) {
          MemorySpace promoted = mem_it->second;
          std::optional<MemorySpace> kwarg_target;
          for (const auto& [key, value] : call->kwargs_) {
            if (key == "target_memory") {
              kwarg_target = AnyCast<MemorySpace>(value, "target_memory");
              break;
            }
          }
          if (!kwarg_target.has_value() || *kwarg_target != promoted) {
            std::vector<std::pair<std::string, std::any>> new_kwargs;
            new_kwargs.reserve(call->kwargs_.size() + 1);
            bool saw_target_memory = false;
            for (const auto& [key, value] : call->kwargs_) {
              if (key == "target_memory") {
                saw_target_memory = true;
                new_kwargs.emplace_back(key, std::any(promoted));
              } else {
                new_kwargs.emplace_back(key, value);
              }
            }
            if (!saw_target_memory) {
              new_kwargs.emplace_back("target_memory", std::any(promoted));
            }
            auto promoted_view = tile_view_semantics::GetImplicitTileView(old_call_type->shape_, promoted);
            auto promoted_type = std::make_shared<TileType>(old_call_type->shape_, old_call_type->dtype_,
                                                            old_call_type->memref_, promoted_view, promoted);
            new_value = std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs),
                                               std::move(promoted_type), call->span_);
          }
        }
      }
    }

    // Sync LHS Var type with the rebuilt Call's result type.  When VisitExpr_(CallPtr)
    // rebuilds the Call via OpRegistry after substituting moved arguments, the deduced
    // result type may differ from the LHS Var's original type (e.g. tile_view changes
    // because the inputs now have different layouts).  Without this sync, the Var
    // annotation and the Call result type disagree, which breaks roundtrip equality.
    auto new_call = As<Call>(new_value);
    auto old_tile_type = As<TileType>(new_var->GetType());
    if (new_call && old_tile_type) {
      auto new_tile_type = As<TileType>(new_call->GetType());
      if (new_tile_type && new_tile_type.get() != old_tile_type.get()) {
        // Preserve the Var's memory_space (set by VisitExpr_(VarPtr) based on var_memory_).
        auto synced_type =
            std::make_shared<TileType>(new_tile_type->shape_, new_tile_type->dtype_, new_tile_type->memref_,
                                       new_tile_type->tile_view_, old_tile_type->memory_space_);
        auto synced_var = std::make_shared<Var>(new_var->name_hint_, std::move(synced_type), new_var->span_);
        var_cache_[op->var_] = synced_var;
        new_var = synced_var;
      }
    }

    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    auto new_stmts = VisitAndInsertMoves(op->stmts_, changed);
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_;
  const std::set<MoveKey, MoveKeyLess>& needed_moves_;
  std::map<VarPtr, ExprPtr> var_cache_;
  std::map<MoveKey, ExprPtr, MoveKeyLess> created_moves_;
  // One entry per active SeqStmts scope holding the keys inserted into
  // created_moves_ within that scope. Popping a scope erases only those keys,
  // avoiding a full-map copy on every SeqStmts visit (O(N^2) on nested IR).
  std::vector<std::vector<MoveKey>> scope_inserted_stack_;

  std::vector<StmtPtr> VisitAndInsertMoves(const std::vector<StmtPtr>& stmts, bool& changed) {
    // Scope created_moves_ to this SeqStmts so moves emitted in one branch
    // of an IfStmt (or other sibling scope) are not treated as available in
    // later sibling blocks. Otherwise the cache would skip re-emitting a
    // required tile.move in the else branch while the target var is defined
    // only in the then branch, leaving a dangling SSA reference.
    scope_inserted_stack_.emplace_back();
    std::vector<StmtPtr> new_stmts;
    for (const auto& stmt : stmts) {
      InsertMovesForConsumer(new_stmts, stmt, changed);
      auto new_stmt = IRMutator::VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) changed = true;
      new_stmts.push_back(new_stmt);
    }
    for (const auto& key : scope_inserted_stack_.back()) {
      created_moves_.erase(key);
    }
    scope_inserted_stack_.pop_back();
    return new_stmts;
  }

  void InsertMovesForConsumer(std::vector<StmtPtr>& stmts, const StmtPtr& stmt, bool& changed) {
    CallPtr call;
    Span span = stmt ? stmt->span_ : Span::unknown();
    if (auto assign = As<AssignStmt>(stmt)) {
      call = As<Call>(assign->value_);
    } else if (auto eval = As<EvalStmt>(stmt)) {
      call = As<Call>(eval->expr_);
    }
    if (!call) return;

    const auto* constraints = GetInputConstraints(call->op_->name_);
    if (!constraints) return;

    // Look up backend layout spec so tile.move carries the correct layout for the consumer.
    // This avoids a later ResolveBackendOpLayouts repair pass needing to insert tile.reshape.
    const backend::BackendTileLayoutSpec* layout_spec = nullptr;
    if (backend::BackendConfig::IsConfigured()) {
      layout_spec = backend::GetBackend()->GetTileLayoutSpec(call->op_->name_);
    }

    for (size_t i = 0; i < constraints->size() && i < call->args_.size(); ++i) {
      if ((*constraints)[i].empty()) continue;
      auto var = As<Var>(call->args_[i]);
      if (!var) continue;

      MoveKey key = {var, (*constraints)[i][0]};
      if (needed_moves_.count(key) == 0 || created_moves_.count(key) > 0) {
        continue;
      }

      // Get required layout for this input from backend spec.
      // blayout comes from the spec; slayout is set to none_box only for Vec targets
      // because Vec/scalar-processing spaces use ND format (no scatter layout).
      // For other memory spaces (Mat, Left, Right), the scatter layout is preserved.
      std::optional<TileLayout> required_blayout;
      std::optional<TileLayout> required_slayout;
      if (layout_spec && i < layout_spec->input_layouts.size() && layout_spec->input_layouts[i].has_value()) {
        required_blayout = layout_spec->input_layouts[i];
        if (key.second == MemorySpace::Vec) {
          required_slayout = TileLayout::none_box;
        }
      }

      InsertMoveStmt(stmts, var, key.second, span, required_blayout, required_slayout);
      changed = true;
    }
  }

  void InsertMoveStmt(std::vector<StmtPtr>& stmts, const VarPtr& original_var, MemorySpace target,
                      const Span& span, std::optional<TileLayout> required_blayout = std::nullopt,
                      std::optional<TileLayout> required_slayout = std::nullopt) {
    auto mutated_producer = IRMutator::VisitExpr(original_var);
    auto mutated_producer_var = As<Var>(mutated_producer);
    INTERNAL_CHECK_SPAN(mutated_producer_var, span)
        << "Internal error: inferred tile-memory producer is not a Var expression";

    // Create tile.move call via OpRegistry
    auto& op_reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", std::any(target)}};
    if (required_blayout.has_value()) {
      kwargs.emplace_back("blayout", std::any(*required_blayout));
    }
    if (required_slayout.has_value()) {
      kwargs.emplace_back("slayout", std::any(*required_slayout));
    }
    auto move_call = op_reg.Create("tile.move", {mutated_producer}, kwargs, span);

    // Create moved var with memory_space_ set
    auto move_type = As<TileType>(move_call->GetType());
    INTERNAL_CHECK_SPAN(move_type, span) << "Internal error: tile.move return type is not TileType";
    auto moved_type = std::make_shared<TileType>(move_type->shape_, move_type->dtype_, move_type->memref_,
                                                 move_type->tile_view_, target);
    auto moved_var = std::make_shared<Var>(
        mutated_producer_var->name_hint_ + "_" + MemorySpaceToString(target), std::move(moved_type), span);

    // Register for substitution and in var_cache_ so VisitExpr_(VarPtr) returns it as-is.
    // Record the key in the current scope so it is erased when the SeqStmts exits.
    MoveKey key = {original_var, target};
    created_moves_[key] = moved_var;
    if (!scope_inserted_stack_.empty()) {
      scope_inserted_stack_.back().push_back(key);
    }
    var_cache_[moved_var] = moved_var;

    stmts.push_back(std::make_shared<AssignStmt>(moved_var, move_call, span));
  }
};

// ============================================================================
// Transform: combine analysis, move collection, and mutation
// ============================================================================

FunctionPtr TransformInferTileMemorySpace(const FunctionPtr& func) {
  // Phase 0: Collect backward demand from op input_constraints; propagate
  // through OutputMemoryInheritsInput() ops so demand reaches retargetable
  // producers (tile.load/tile.create) even through view chains (slice/fillpad).
  DemandCollector demand_collector;
  demand_collector.VisitStmt(func->body_);
  demand_collector.PropagateThroughInheritInputOps();

  // Phase 1: Analyze — infer memory space for each tile variable, using Phase-0
  // demand as fallback for retargetable producers whose target_memory is absent.
  TileMemorySpaceAnalyzer analyzer(func->params_, demand_collector.GetDemands());
  analyzer.VisitStmt(func->body_);

  const auto& var_memory = analyzer.GetVarMemory();
  if (var_memory.empty()) {
    return func;
  }

  // Phase 2: Collect needed tile.move insertions for residual input-constraint
  // mismatches (producer and demand both resolved to different fixed spaces).
  MoveCollector collector(var_memory);
  collector.VisitStmt(func->body_);

  // Phase 3: Mutate — set memory_space_ on types, insert moves, substitute args,
  // rewrite target_memory kwargs on retargetable producers to stay consistent.
  TileMemorySpaceMutator mutator(var_memory, collector.GetNeededMoves());
  auto new_body = mutator.VisitStmt(func->body_);

  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

// ============================================================================
// Pass factory function
// ============================================================================

namespace pass {

Pass InferTileMemorySpace() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        new_functions[gvar] = TransformInferTileMemorySpace(func);
      } else {
        new_functions[gvar] = func;
      }
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "InferTileMemorySpace", kInferTileMemorySpaceProperties);
}

}  // namespace pass

// ============================================================================
// TileMemoryInferred property verifier
// ============================================================================

namespace {

class TileMemoryInferredVerifier : public IRVisitor {
 public:
  explicit TileMemoryInferredVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      auto tile_type = As<TileType>(op->var_->GetType());
      if (tile_type && !tile_type->memory_space_.has_value()) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileMemoryInferred", 0,
                                  "InCore function '" + func_name_ + "': TileType variable '" +
                                      op->var_->name_hint_ + "' has no memory_space set",
                                  op->var_->span_);
      }
    }

    // Verify input memory space constraints
    if (auto call = As<Call>(op->value_)) {
      VerifyInputConstraints(call);
    }

    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_)) {
      VerifyInputConstraints(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;

  void VerifyInputConstraints(const CallPtr& call) {
    const auto* constraints = GetInputConstraints(call->op_->name_);
    if (!constraints) return;

    for (size_t i = 0; i < constraints->size() && i < call->args_.size(); ++i) {
      const auto& allowed_spaces = (*constraints)[i];
      if (allowed_spaces.empty()) continue;

      auto var = As<Var>(call->args_[i]);
      if (!var) continue;
      auto tile_type = As<TileType>(var->GetType());
      if (!tile_type || !tile_type->memory_space_.has_value()) continue;

      MemorySpace actual = *tile_type->memory_space_;
      bool allowed = std::find(allowed_spaces.begin(), allowed_spaces.end(), actual) != allowed_spaces.end();
      if (!allowed) {
        std::string allowed_str;
        for (size_t j = 0; j < allowed_spaces.size(); ++j) {
          if (j > 0) allowed_str += "/";
          allowed_str += MemorySpaceToString(allowed_spaces[j]);
        }
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileMemoryInferred", 0,
                                  "InCore function '" + func_name_ + "': Op '" + call->op_->name_ +
                                      "' input " + std::to_string(i) + " ('" + var->name_hint_ +
                                      "') requires " + allowed_str + " but is in " +
                                      MemorySpaceToString(actual),
                                  var->span_);
      }
    }
  }
};

}  // namespace

class TileMemoryInferredPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileMemoryInferred"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      TileMemoryInferredVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateTileMemoryInferredPropertyVerifier() {
  return std::make_shared<TileMemoryInferredPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
