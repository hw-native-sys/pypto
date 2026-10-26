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

#include "pypto/ir/transforms/utils/lifetime_analysis.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/storage_reference_analysis.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace {

using memory_reuse::ReverseBranchOrderVisitor;

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitVarLike_(const VarPtr& var) override { used_vars.insert(var); }
  void VisitExpr_(const IterArgPtr& var) override { VisitVarLike_(var); }
};

/**
 * @brief Full IR tree walker for lifetime analysis.
 *
 * Walks the entire IR tree (including nested control flow bodies) to
 * assign sequential order to all leaf statements and collect variable
 * definitions and uses.
 *
 * Two-phase design:
 *   Phase 1 (Analyze): Walk tree, assign orders, collect raw var uses, record loop boundaries.
 *   Phase 2 (ComputeEffectiveLastUse): For each var, extend lifetime to the end of any
 *           enclosing loop where the var is defined before the loop.
 */
class LifetimeAnalyzer : public IRVisitor {
 public:
  struct Result {
    std::vector<VarPtr> ordered_defs;
    std::map<VarPtr, int> var_def_order;
    std::map<VarPtr, int> var_last_use;
    std::map<VarPtr, StmtPtr> var_def_stmt;
    std::map<VarPtr, std::pair<int, int>> var_reverse_liveness;
    int function_end;
    int reverse_function_end;
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
    const int reverse_end = func_body ? static_cast<int>(reverse_end_order_.at(func_body.get()) + 1) : 0;
    return {std::move(ordered_defs_),
            std::move(var_def_order_),
            std::move(last_use),
            std::move(var_def_stmt_),
            std::move(var_reverse_liveness_),
            current_order_,
            reverse_end};
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
  const auto references = AnalyzeStorageReferences(func_body, entry_vars);

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
    // A selector owns no allocation. Every concrete origin reachable through
    // its reference graph remains exclusive for the complete function in both
    // branch orders. This first implementation intentionally spends memory
    // instead of expanding an origin set at each selector. Extend allocation
    // occupancy only: per-value liveness below still describes the original
    // SSA versions, so sequential author bindings and old-value observations
    // do not acquire artificial conflicts merely by supplying a reference.
    if (references.source_bases.count(GetDefinedMemRef(representative_tile_type)->base_.get())) {
      interval.def_point = 0;
      interval.last_use_point = result.function_end;
      interval.reverse_def_point = 0;
      interval.reverse_last_use_point = result.reverse_function_end;
    }

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

}  // namespace

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const StmtPtr& func_body) {
  return AnalyzeAllocationLifetimesImpl(func_body);
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const FunctionPtr& func) {
  INTERNAL_CHECK(func != nullptr) << "Cannot analyze allocation lifetimes for a null function";
  return AnalyzeAllocationLifetimesImpl(func->body_, func->params_);
}

}  // namespace ir
}  // namespace pypto
