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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
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

  void VisitVarLike_(const VarPtr& var) override {
    used_vars.insert(var);
    IRVisitor::VisitVarLike_(var);
  }
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
    std::vector<VarPtr> ordered_defs;        // TileType vars in definition order
    std::map<VarPtr, int> var_def_order;     // var -> def order
    std::map<VarPtr, int> var_last_use;      // var -> effective last use (with loop extension)
    std::map<VarPtr, StmtPtr> var_def_stmt;  // var -> defining AssignStmt (for op name extraction)
    // Phi families, keyed per (scf.if, return-slot): each family id groups one
    // phi return_var with its branch-local yield-sources (the same logical value
    // across mutually-exclusive paths), so they may always share a buffer.  A
    // var may belong to several families (e.g. a source yielded at >1 slot)
    // without those families merging, so we store a *set* of ids per var.
    std::map<const Var*, std::set<int>> phi_family_ids;
  };

  Result Analyze(const StmtPtr& func_body, const std::vector<VarPtr>& entry_vars = {}) {
    // Function parameters have no defining AssignStmt. Seed Tile parameters at
    // the entry point so function-wide allocation planning includes their
    // MemRefs and body uses extend their lifetimes normally.
    for (const VarPtr& var : entry_vars) {
      const auto tile_type = As<TileType>(var->GetType());
      if (!tile_type || !tile_type->memref_.has_value()) continue;
      ordered_defs_.push_back(var);
      var_def_order_[var] = current_order_;
    }

    // Phase 1: Walk IR tree
    if (func_body) {
      VisitStmt(func_body);
    }

    // Phase 2: Apply loop-aware lifetime extension
    auto effective_last_use = ComputeEffectiveLastUse();

    return {std::move(ordered_defs_), std::move(var_def_order_), std::move(effective_last_use),
            std::move(var_def_stmt_), std::move(var_family_ids_)};
  }

 protected:
  // Container statements: recurse into children
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  // Leaf statements: assign order + collect defs/uses
  void VisitStmt_(const AssignStmtPtr& op) override {
    int order = current_order_++;

    // Record TileType variable definitions
    auto tile_type = As<TileType>(op->var_->GetType());
    if (tile_type) {
      ordered_defs_.push_back(op->var_);
      var_def_order_[op->var_] = order;
      var_def_stmt_[op->var_] = op;
    }

    // Collect variable uses from value expression (for ALL AssignStmt)
    CollectUsesFromExpr(op->value_, order);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    int order = current_order_++;
    CollectUsesFromExpr(op->expr_, order);
  }

  void VisitStmt_(const YieldStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  void VisitStmt_(const ReturnStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  // Control flow: recurse into bodies with scope tracking
  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit condition uses at the current order point (before branches)
    CollectUsesFromExpr(op->condition_, current_order_);

    // Visit then body first, else body second (sequential ordering), recording
    // each branch's order range so we can tell branch-local yield-sources apart.
    int then_start = current_order_;
    VisitStmt(op->then_body_);
    int then_end = current_order_ - 1;
    int else_start = -1;
    int else_end = -1;
    if (op->else_body_.has_value()) {
      else_start = current_order_;
      VisitStmt(*op->else_body_);
      else_end = current_order_ - 1;
    }

    // An scf.if return_var (phi) is NOT itself a tracked def -- it is materialized
    // by YieldFixup into a branch yield-source's buffer downstream, and tracking
    // it here would perturb that lowering.  But reads of the phi *after* the if
    // keep that buffer live; if we drop them (RecordRawUse early-returns on
    // untracked vars), the buffer's live range collapses at the yield and a later
    // temporary is packed onto the still-live phi value (merge_norm mi/li/oi,
    // #1821).  So we redirect the phi's reads onto its *branch-local* yield-
    // sources (the tiles actually holding the value), recorded in
    // `phi_local_sources_`.  A source defined OUTSIDE the branch (loaded before
    // the if and live elsewhere) is left alone -- it genuinely conflicts and
    // YieldFixup moves it into the phi buffer.
    //
    // Extending both branch sources to the phi's last read makes the two
    // mutually-exclusive sources overlap, so we also tag them with a fresh
    // per-slot family id; the reuse packer exempts intra-family pairs from the
    // interference check.  Ids are *per return slot* (not a transitive union): a
    // source shared across slots joins several families without merging them, so
    // two distinct phi results are never exempted.
    auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
    auto else_yield =
        op->else_body_.has_value() ? transform_utils::GetLastYieldStmt(*op->else_body_) : nullptr;
    auto branch_local = [this](const VarPtr& v, int start, int end) {
      if (start < 0) return false;
      auto it = var_def_order_.find(v);
      return it != var_def_order_.end() && it->second >= start && it->second <= end;
    };
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      auto tile_type = As<TileType>(rv->GetType());
      if (!tile_type || !tile_type->memref_.has_value()) continue;
      int family_id = next_family_id_++;
      std::vector<VarPtr> local_sources;
      std::set<const Var*> seen_phi;
      // Flatten a nested-if phi source down to its terminal tracked tiles, and
      // tag THOSE with this (outer) slot's family id -- otherwise the terminals
      // only carry the inner slot's id and `overlap_blocks_sharing` (which sees
      // terminal vars, not the untracked phi wrapper) would reject safe nested
      // mutually-exclusive sharing.
      std::function<void(const VarPtr&, int, int)> add_source_var = [&](const VarPtr& src, int start,
                                                                        int end) {
        auto pit = phi_local_sources_.find(src.get());
        if (pit != phi_local_sources_.end()) {
          if (!seen_phi.insert(src.get()).second) return;
          for (const auto& nested : pit->second) add_source_var(nested, start, end);
          return;
        }
        if (!branch_local(src, start, end)) return;
        local_sources.push_back(src);
        var_family_ids_[src.get()].insert(family_id);
      };
      auto add_source = [&](const ExprPtr& val, int start, int end) {
        if (auto src = AsVarLike(val)) add_source_var(src, start, end);
      };
      if (then_yield && i < then_yield->value_.size()) {
        add_source(then_yield->value_[i], then_start, then_end);
      }
      if (else_yield && i < else_yield->value_.size()) {
        add_source(else_yield->value_[i], else_start, else_end);
      }
      if (!local_sources.empty()) {
        var_family_ids_[rv.get()].insert(family_id);
        phi_local_sources_[rv.get()] = std::move(local_sources);
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->body_);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->body_);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
  }

 private:
  int current_order_ = 0;

  // Loop boundaries recorded during Phase 1
  struct LoopBoundary {
    int start_order;
    int end_order;
  };
  std::vector<LoopBoundary> loop_scopes_;

  // Variable tracking (Phase 1)
  std::vector<VarPtr> ordered_defs_;
  std::map<VarPtr, int> var_def_order_;
  std::map<VarPtr, int> var_raw_last_use_;  // Raw last use (without loop extension)
  std::map<VarPtr, StmtPtr> var_def_stmt_;

  // Maps ForStmt/WhileStmt return_vars to their corresponding initValue vars.
  // YieldFixup will place return_vars into initValue's MemRef buffer,
  // so any post-loop use of a return_var must extend the initValue's lifetime.
  std::map<VarPtr, VarPtr> return_var_to_init_var_;

  // Per-slot phi families: var -> set of family ids (see VisitStmt_(IfStmtPtr)).
  std::map<const Var*, std::set<int>> var_family_ids_;
  int next_family_id_ = 0;

  // scf.if phi return_var -> its branch-local yield-sources, so RecordRawUse can
  // redirect phi reads onto the tiles that actually hold the value.
  std::map<const Var*, std::vector<VarPtr>> phi_local_sources_;

  void CollectUsesFromExpr(const ExprPtr& expr, int use_order) {
    if (!expr) return;
    VarUseCollector collector;
    collector.VisitExpr(expr);
    for (const auto& var : collector.used_vars) {
      RecordRawUse(var, use_order);
    }
  }

  void CollectUsesFromValues(const std::vector<ExprPtr>& values) {
    int order = current_order_++;
    for (const auto& val : values) {
      CollectUsesFromExpr(val, order);
    }
  }

  void ProcessLoopBody(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body) {
    for (const auto& iter_arg : iter_args) {
      if (iter_arg->initValue_) {
        CollectUsesFromExpr(iter_arg->initValue_, current_order_);
      }
    }
    int loop_start = current_order_;
    VisitStmt(body);
    int loop_end = current_order_ - 1;
    loop_scopes_.push_back({loop_start, loop_end});
  }

  /// Resolve a loop carry's init to the variable that actually owns the buffer.
  ///
  /// A nested loop seeds its carry from the *enclosing* loop's IterArg, and an
  /// IterArg is an alias of its own init rather than a definition in its own
  /// right -- like return_vars, iter-args are deliberately absent from
  /// `ordered_defs_` / `var_def_order_`, so they carry no lifetime of their own.
  /// Walking the carry chain to the first non-IterArg therefore lands on the
  /// AssignStmt-defined tile whose buffer the whole loop nest shares, which is
  /// the variable a use of any return_var in that nest has to keep alive.
  ///
  /// Returns nullptr when the chain does not end at a tracked carrier; the
  /// caller then records no mapping, exactly as before.
  [[nodiscard]] VarPtr ResolveCarryInitCarrier(const ExprPtr& init_value) const {
    auto var = AsVarLike(init_value);
    // An init is always defined before its own loop, so the chain is acyclic in
    // well-formed IR; the visited set bounds it anyway rather than hanging.
    std::set<const Var*> seen;
    while (var) {
      auto iter_arg = As<IterArg>(var);
      if (!iter_arg) return var;
      if (!seen.insert(var.get()).second) return nullptr;
      var = AsVarLike(iter_arg->initValue_);
    }
    return nullptr;
  }

  void RegisterReturnVars(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    size_t count = std::min(return_vars.size(), iter_args.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& rv = return_vars[i];
      if (!As<TileType>(rv->GetType())) continue;

      // Map return_var -> the init's buffer carrier so that post-loop uses of the
      // return_var extend that carrier's lifetime (not the return_var's).
      // We do NOT register return_vars in ordered_defs_ -- they must not
      // participate in sharing group computation, which would inflate
      // group lifetimes and block unrelated reuse opportunities.
      auto init_var = ResolveCarryInitCarrier(iter_args[i]->initValue_);
      if (init_var && var_def_order_.count(init_var)) {
        return_var_to_init_var_[rv] = init_var;
      }
    }
  }

  void RecordRawUse(const VarPtr& var, int use_order) {
    // If var is an scf.if phi return_var, redirect the use to its branch-local
    // yield-sources (the tiles holding the value), keeping their buffers live to
    // the phi's last read.  Sources may themselves be phis (chained scf.if), so
    // recurse.
    auto pit = phi_local_sources_.find(var.get());
    if (pit != phi_local_sources_.end()) {
      for (const auto& src : pit->second) RecordRawUse(src, use_order);
      return;
    }

    // If var is a loop return_var, redirect the use to its initValue var.
    // YieldFixup will alias the return_var to the initValue's MemRef,
    // so keeping the initValue live prevents premature buffer reuse.
    auto it = return_var_to_init_var_.find(var);
    const VarPtr& target = (it != return_var_to_init_var_.end()) ? it->second : var;

    if (!var_def_order_.count(target)) {
      return;
    }
    // operator[] default-inserts 0 for missing keys; use_order is always >= 0.
    var_raw_last_use_[target] = std::max(var_raw_last_use_[target], use_order);
  }

  /**
   * @brief Phase 2: Compute effective last use with loop-aware extension.
   *
   * For each variable, if it's used inside a loop body and defined before
   * that loop, extend its lifetime to the end of the loop. This handles
   * the case where the loop re-executes and the variable is needed again.
   */
  [[nodiscard]] std::map<VarPtr, int> ComputeEffectiveLastUse() const {
    std::map<VarPtr, int> effective;

    for (const auto& var : ordered_defs_) {
      int def_order = var_def_order_.at(var);
      int raw_last = var_raw_last_use_.count(var) ? var_raw_last_use_.at(var) : def_order;
      int extended = raw_last;

      // Check all loop scopes: if var is defined before the loop and
      // has any use inside the loop [start, end], extend to loop end.
      for (const auto& loop : loop_scopes_) {
        if (def_order < loop.start_order && raw_last >= loop.start_order) {
          // Variable is defined before loop and used inside it
          extended = std::max(extended, loop.end_order);
        }
      }

      effective[var] = extended;
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
  const auto allocation_sizes = memref_collectors::CollectAllocSizes(func_body);

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
    bool has_slots = false;

    for (const auto& group_var : sharing_group) {
      int def_point = result.var_def_order.count(group_var) ? result.var_def_order.at(group_var) : 0;
      int last_use = result.var_last_use.count(group_var) ? result.var_last_use.at(group_var) : def_point;

      LOG_DEBUG << "Variable " << group_var->name_hint_ << " def=" << def_point << " last_use=" << last_use;

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
      const auto tile = GetTileTypeWithMemRef(group_var->GetType());
      has_slots |= tile && GetDefinedMemRef(tile)->slot_count_ > 1;
    }

    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    auto representative_tile_type = As<TileType>(sharing_group[0]->GetType());
    INTERNAL_CHECK_SPAN(representative_tile_type != nullptr, sharing_group[0]->span_)
        << "Expected TileType for reuse interval";
    auto memory_space = representative_tile_type->GetMemorySpace();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), sharing_group[0]->span_)
        << "TileType with MemRef must have memory_space for reuse analysis";
    interval.memory_space = *memory_space;
    interval.size = GetDefinedMemRef(representative_tile_type)->size_;
    // Reuse moves the complete allocation. Keep its capacity separate from
    // each slot's window size and from the author's pinned ownership contract.
    if (has_slots) {
      const auto capacity = allocation_sizes.find(GetDefinedMemRef(representative_tile_type)->base_.get());
      if (capacity != allocation_sizes.end()) interval.size = std::max(interval.size, capacity->second);
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

  // Per-var individual liveness (def + effective last use, including phi/loop
  // extension) for the packer's precise pairwise interference check.
  std::map<const Var*, std::pair<int, int>> var_liveness;
  for (const auto& [var, def_point] : result.var_def_order) {
    int last_use = result.var_last_use.count(var) ? result.var_last_use.at(var) : def_point;
    var_liveness[var.get()] = {def_point, last_use};
  }

  return {lifetimes,
          var_sharing_groups,
          std::move(result.phi_family_ids),
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
