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
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
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
      INTERNAL_CHECK(!As<TileType>(var->GetType()))
          << "InCore function parameter '" << var->name_hint_
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
    // Seed each TileType iter-arg's memory space from its init value before
    // analysing the body, so an inherit-input op in the body inherits the
    // carried-in space instead of InheritFromInput falling through to a
    // co-argument. Notably tile.assemble(target, source, offset) is
    // output_inherits_input on its *target* (arg0); for a full-K Mat-scratch the
    // target is the Mat scratch iter-arg, which is still unresolved when the body
    // is analysed — without this seed InheritFromInput skips it and returns the
    // Acc *source* (arg1), forcing the whole [M, N] scratch chain into Acc and
    // overflowing L0c. The post-body override below still promotes a
    // conservatively-Vec init that the body writes as Acc (matmul_acc accumulator).
    // AsVarLike (not As<Var>) so an inner loop whose init is the outer iter-arg is
    // also seeded. When the init carrier was never visited by the AssignStmt path
    // (e.g. an IfStmt return var), it is absent from var_memory_ but still carries a
    // memory_space_ in its TileType — fall back to that so the seed resolves
    // regardless of the init's statement shape (mirrors the yield_memory lookup).
    for (const auto& iter_arg : op->iter_args_) {
      if (!As<TileType>(iter_arg->GetType())) continue;
      if (auto init_var = AsVarLike(iter_arg->initValue_)) {
        if (auto it = var_memory_.find(init_var); it != var_memory_.end()) {
          var_memory_[iter_arg] = it->second;
        } else if (auto init_tile_type = As<TileType>(init_var->GetType());
                   init_tile_type && init_tile_type->memory_space_.has_value()) {
          var_memory_[iter_arg] = *init_tile_type->memory_space_;
        }
      }
    }

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
    // AsVarLike (not As<Var>) so an IterArg argument is matched — e.g.
    // tile.assemble's Mat scratch target (arg0) inside a full-K pipeline loop.
    // With As<Var> the IterArg is skipped and the inherit falls through to a
    // co-argument (the Acc source, arg1), forcing the scratch into Acc.
    for (const auto& arg : call->args_) {
      if (auto var = AsVarLike(arg)) {
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
  // When promoting to a new memory_space, refresh the layout pieces (blayout/
  // slayout/fractal) to the target's implicit view — the source's layout
  // (e.g. Vec defaults from tile.create) becomes a mismatch once the space
  // changes (Acc expects col_major/row_major). Other metadata (valid_shape,
  // stride, start_offset, pad) reflects the actual data and is preserved.
  std::optional<TypePtr> ComputeRewrittenType(const VarPtr& op) const {
    auto tile_type = As<TileType>(op->GetType());
    auto mem_it = var_memory_.find(op);
    if (!tile_type || mem_it == var_memory_.end()) return std::nullopt;

    std::optional<TileView> new_view = tile_type->tile_view_;
    if (tile_type->memory_space_ != mem_it->second) {
      TileView source = tile_view_semantics::GetEffectiveTileView(*tile_type);
      TileView target_layout = tile_view_semantics::GetImplicitTileView(tile_type->shape_, mem_it->second);
      source.blayout = target_layout.blayout;
      source.slayout = target_layout.slayout;
      source.fractal = target_layout.fractal;
      new_view = std::move(source);
    }
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                      std::move(new_view), mem_it->second);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }

    if (auto new_type = ComputeRewrittenType(op)) {
      auto new_var = std::make_shared<Var>(op->name_hint_, *new_type, op->span_);
      var_cache_[op] = new_var;
      return new_var;
    }

    var_cache_[op] = op;
    return op;
  }

  // IterArg dispatches through its own visitor (per kind_traits — As<Var> does
  // not match IterArg). Without this override the base IRMutator preserves the
  // IterArg's old type, leaving iter_arg.type.memory_space stale while
  // init_value and yield are promoted — breaking AssignStmt symmetry and
  // print/parse round-trip.
  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }

    auto new_type_opt = ComputeRewrittenType(op);
    auto new_init_value = VisitExpr(op->initValue_);

    bool type_changed = new_type_opt.has_value();
    bool init_changed = new_init_value.get() != op->initValue_.get();
    if (!type_changed && !init_changed) {
      var_cache_[op] = op;
      return op;
    }

    auto new_iter_arg = std::make_shared<const IterArg>(
        op->name_hint_, type_changed ? *new_type_opt : op->GetType(), std::move(new_init_value), op->span_);
    var_cache_[op] = new_iter_arg;
    return new_iter_arg;
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
      return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, op->attrs_, op->GetType(),
                                    op->span_);
    }
    auto deduced = registry.Create(op->op_->name_, new_args, op->kwargs_, op->span_);
    return std::make_shared<Call>(deduced->op_, deduced->args_, deduced->kwargs_, op->attrs_,
                                  deduced->GetType(), deduced->span_);
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
            new_value = std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->attrs_,
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
        // When the producing Call's result type still lacks the resolved memory
        // space, rebuild it so the RHS Call and the LHS Var agree. Retargetable
        // producers (tile.load / tile.create) are already promoted above via
        // their target_memory kwarg; this covers tile producers with no such
        // kwarg (e.g. pld.tile.remote_load), whose deduced TileType keeps
        // memory_space unset. Without it the Var carries the inferred space but
        // the Call does not, so a print->parse roundtrip — which re-derives the
        // Call type from the LHS annotation — sees a memory_space presence
        // mismatch on body[*].value.type.
        if (new_tile_type->memory_space_ != old_tile_type->memory_space_) {
          new_value = std::make_shared<Call>(new_call->op_, new_call->args_, new_call->kwargs_,
                                             new_call->attrs_, synced_type, new_call->span_);
        }
        auto synced_var = std::make_shared<Var>(new_var->name_hint_, synced_type, new_var->span_);
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

      // ISA constraint on the Acc→Vec data path: the destination tile is ND
      // (row_major, none_box). The hardware cube→vec pipe (tpush_to_aiv /
      // tpop_from_aic) un-fractalizes the data during transfer, so the tile
      // arriving in Vec is physically ND regardless of the source's NZ form
      // in Acc. Label the move's dst accordingly so downstream consumers see
      // the correct layout without a redundant repair tmov.
      auto producer_mem_it = var_memory_.find(var);
      if (producer_mem_it != var_memory_.end() && producer_mem_it->second == MemorySpace::Acc &&
          key.second == MemorySpace::Vec) {
        required_blayout = TileLayout::row_major;
        required_slayout = TileLayout::none_box;
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
// Phase 4: conservative loop-invariant residency
//
// ConvertTensorToTileOps deliberately inserts a tensor operand's GM->Mat load
// at its use site.  For a tensor.matmul inside a user-written loop that means a
// stationary operand is reloaded on every iteration.  Once Phase 3 has made
// every memory space explicit, this phase can recognize the complete
//
//   tile.load(GM -> Mat) -> transpose_view* -> tile.move/extract(Mat -> L0)
//
// chain and move the invariant prefix to the loop preheader.  The rewrite is
// intentionally conservative: only read-only function parameters, direct
// top-level statements in a statically non-empty sequential loop, and static
// capacity-safe Mat/Left/Right footprints are accepted.  A nested invariant
// chain bubbles outward one loop at a time because the mutator visits the inner
// loop first.  The inventory is computed once, bottom-up, so the pass stays
// O(N log N) rather than rescanning every nested loop body.
// ============================================================================

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

struct LoopResidencyInfo {
  uint64_t preorder{0};
  uint64_t postorder{0};
  std::unordered_set<const Expr*> yielded_values;
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
    RecordDirectCallUses(op);
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
    const bool is_pipeline = op->kind_ == ForKind::Pipeline;
    if (is_pipeline) ++pipeline_depth_;
    if (op->loop_var_) defining_loop_[op->loop_var_.get()] = op.get();
    for (const auto& iter_arg : op->iter_args_) {
      if (!iter_arg) continue;
      defining_loop_[iter_arg.get()] = op.get();
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
    if (use_it == direct_call_uses_.end() || use_it->second.size() != 1) return false;
    const auto& use = use_it->second.front();
    const auto operand_indices = MatmulOperandIndices(use.call);
    if (!operand_indices.has_value()) return false;
    if (*tile->memory_space_ == MemorySpace::Left) return use.arg_index == operand_indices->first;
    if (*tile->memory_space_ == MemorySpace::Right) return use.arg_index == operand_indices->second;
    return false;
  }

  [[nodiscard]] bool HasOnlyUseBy(const VarPtr& var, const CallPtr& consumer) const {
    auto it = direct_call_uses_.find(var.get());
    return it != direct_call_uses_.end() && it->second.size() == 1 &&
           it->second.front().call.get() == consumer.get();
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
  }
};

std::vector<StmtPtr> DirectStatements(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  return {body};
}

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
    auto it = buffer_roots_.find(var.get());
    if (it != buffer_roots_.end() && it->second) write_roots_.insert(it->second);
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
                              std::unordered_map<const Var*, ReadParamEvidence>* evidence)
      : program_(std::move(program)),
        caller_is_root_orchestration_(caller_is_root_orchestration),
        buffer_roots_(buffer_roots),
        evidence_(evidence) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    RecordEvidence(op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordEvidence(transform_utils::AsCallOrSubmitView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  ProgramPtr program_;
  bool caller_is_root_orchestration_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  std::unordered_map<const Var*, ReadParamEvidence>* evidence_;

  [[nodiscard]] const Var* ResolveRoot(const ExprPtr& arg) const {
    auto var = AsVarLike(arg);
    if (!var) return nullptr;
    auto it = buffer_roots_.find(var.get());
    return it == buffer_roots_.end() ? nullptr : it->second;
  }

  void RecordEvidence(const CallPtr& call) {
    if (!call || !call->op_ || !program_) return;
    auto callee = program_->GetFunction(call->op_->name_);
    if (!callee || callee->func_type_ != FunctionType::InCore) return;

    // Only a root orchestration call site is a trusted alias-proof boundary.
    // Distinct helper/wrapper parameters may still alias at their unproven
    // caller, so a non-root call must poison (not merely be ignored by) every
    // Tensor In parameter it reaches.  This deliberately declines transitive
    // helper evidence until the pass has an explicit provenance analysis.
    if (!caller_is_root_orchestration_) {
      for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
        if (callee->param_directions_[i] != ParamDirection::In ||
            !As<TensorType>(callee->params_[i]->GetType())) {
          continue;
        }
        auto& evidence = (*evidence_)[callee->params_[i].get()];
        ++evidence.call_sites;
        evidence.all_sites_safe = false;
      }
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
      if (!read_root || !all_write_roots_known || write_roots.count(read_root) != 0) {
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
    const bool is_root_orchestration =
        func->func_type_ == FunctionType::Orchestration && called_functions.count(func.get()) == 0;
    CallSiteReadSafetyCollector calls(program, is_root_orchestration, roots.buffer_roots, &evidence);
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
    buffer_roots_ = std::move(roots.buffer_roots);
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
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Visit first so an invariant chain in a nested loop can bubble into this
    // loop's direct body and, when legal here too, into this preheader.
    auto visited = IRMutator::VisitStmt_(op);
    auto loop = As<ForStmt>(visited);
    INTERNAL_CHECK_SPAN(loop, op->span_) << "Internal error: ForStmt mutation returned a non-ForStmt";

    if (!IsStaticNonEmptySequential(loop)) return visited;
    const auto* loop_info = inventory_.GetLoopInfo(op);
    if (!loop_info || !handler_) return visited;

    auto stmts = DirectStatements(loop->body_);
    std::vector<bool> hoist(stmts.size(), false);
    std::unordered_set<const Expr*> invariant_chain;
    bool changed = false;

    for (size_t i = 0; i < stmts.size(); ++i) {
      auto assign = As<AssignStmt>(stmts[i]);
      // A direct control-flow/effect statement before a candidate can bypass
      // the remainder of the iteration (for example, `continue`).  Stop at
      // that boundary rather than speculating a later load into the preheader.
      if (!assign) break;
      if (!assign->var_ || !inventory_.IsResidencyChainVar(assign->var_.get()) ||
          inventory_.IsYieldedFromLoopSubtree(op, assign->var_.get())) {
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

    if (!changed) return visited;

    std::vector<StmtPtr> preheader;
    std::vector<StmtPtr> body;
    preheader.reserve(stmts.size());
    body.reserve(stmts.size());
    for (size_t i = 0; i < stmts.size(); ++i) {
      (hoist[i] ? preheader : body).push_back(stmts[i]);
    }

    auto new_loop = MutableCopy(loop);
    new_loop->body_ = SeqStmts::Flatten(std::move(body), loop->body_->span_);
    preheader.push_back(new_loop);
    return SeqStmts::Flatten(std::move(preheader), loop->span_);
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
      if (invalid_footprints_.count(space) == 0) function_footprints_[space] = footprint.HighWater();
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

    // Both memory planners run after this pass and see the extended lifetime.
    // Requiring every allocation-owning tile in the whole function to fit
    // simultaneously is stronger than either planner's liveness packing, but
    // it guarantees residency cannot turn a compilable function into a later
    // capacity failure.  Align every owner exactly as AllocateMemoryAddr does.
    auto it = function_footprints_.find(space);
    return it == function_footprints_.end() || it->second <= *capacity;
  }
};

/// Strip the private bridge marker after residency has consumed it.  Keeping
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
  StripCompilerMatBridgeAttrMutator stripper;
  auto body = stripper.VisitStmt(func->body_);
  if (body.get() == func->body_.get()) return func;
  auto clean = MutableCopy(func);
  clean->body_ = std::move(body);
  return clean;
}

// ============================================================================
// Transform: combine analysis, move collection, and mutation
// ============================================================================

FunctionPtr TransformInferTileMemorySpace(const FunctionPtr& func, const ProgramPtr& program,
                                          const std::unordered_set<const Var*>& proven_safe_read_params) {
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
    return StripCompilerMatBridgeAttrs(func);
  }

  // Phase 2: Collect needed tile.move insertions for residual input-constraint
  // mismatches (producer and demand both resolved to different fixed spaces).
  MoveCollector collector(var_memory);
  collector.VisitStmt(func->body_);

  // Phase 3: Mutate — set memory_space_ on types, insert moves, substitute args,
  // rewrite target_memory kwargs on retargetable producers to stay consistent.
  TileMemorySpaceMutator mutator(var_memory, collector.GetNeededMoves());
  auto new_body = mutator.VisitStmt(func->body_);

  auto inferred_func = MutableCopy(func);
  inferred_func->body_ = new_body;

  // Phase 4: hoist loop-invariant GM->Mat loads and any independently
  // invariant Mat->L0 prefix.  Build the inventory once on the fully inferred
  // IR; the mutator consumes it without nested whole-body rescans.
  LoopResidencyInventory inventory;
  inventory.Analyze(inferred_func->body_);
  LoopInvariantTileLoadHoister hoister(inferred_func, program, inventory, proven_safe_read_params);
  inferred_func->body_ = hoister.VisitStmt(inferred_func->body_);
  return NormalizeStmtStructure(StripCompilerMatBridgeAttrs(inferred_func));
}

}  // namespace

// ============================================================================
// Pass factory function
// ============================================================================

namespace pass {

Pass InferTileMemorySpace() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    const auto proven_safe_read_params = CollectProvenSafeReadParams(program);
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        new_functions[gvar] = TransformInferTileMemorySpace(func, program, proven_safe_read_params);
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
