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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Lifetime interval for a TileType variable (based on topological order)
 */
struct LifetimeInterval {
  VarPtr variable;           ///< The variable
  int def_point;             ///< Definition point (topological order)
  int last_use_point;        ///< Last use point (topological order)
  MemorySpace memory_space;  ///< Memory space
  uint64_t size;             ///< Size in bytes
  std::string def_op_name;   ///< Op name that defines this variable (empty if unknown)
};

namespace {

/**
 * @brief Result of lifetime computation
 */
struct LifetimeAnalysisResult {
  std::vector<LifetimeInterval> lifetimes;
  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
};

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitExpr_(const VarPtr& var) override {
    used_vars.insert(var);
    IRVisitor::VisitExpr_(var);
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
  };

  Result Analyze(const StmtPtr& func_body) {
    // Phase 1: Walk IR tree
    if (func_body) {
      VisitStmt(func_body);
    }

    // Phase 2: Apply loop-aware lifetime extension
    auto effective_last_use = ComputeEffectiveLastUse();

    return {std::move(ordered_defs_), std::move(var_def_order_), std::move(effective_last_use),
            std::move(var_def_stmt_)};
  }

 protected:
  // Container statements: recurse into children
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const ScopeStmtPtr& op) override { VisitStmt(op->body_); }

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

    // Visit then body first, else body second (sequential ordering)
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) {
      VisitStmt(*op->else_body_);
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

  void RegisterReturnVars(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    size_t count = std::min(return_vars.size(), iter_args.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& rv = return_vars[i];
      if (!As<TileType>(rv->GetType())) continue;

      // Map return_var -> initValue var so that post-loop uses of the
      // return_var extend the initValue's lifetime (not the return_var's).
      // We do NOT register return_vars in ordered_defs_ -- they must not
      // participate in sharing group computation, which would inflate
      // group lifetimes and block unrelated reuse opportunities.
      auto init_var = As<Var>(iter_args[i]->initValue_);
      if (init_var && var_def_order_.count(init_var)) {
        return_var_to_init_var_[rv] = init_var;
      }
    }
  }

  void RecordRawUse(const VarPtr& var, int use_order) {
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
LifetimeAnalysisResult ComputeLifetimes(const StmtPtr& func_body) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Walk full IR tree to collect variable defs, uses, and ordering
  LifetimeAnalyzer analyzer;
  auto result = analyzer.Analyze(func_body);

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

  for (const auto& var : result.ordered_defs) {
    if (processed_vars.count(var)) {
      continue;
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;
    }

    const auto& memref = tile_type->memref_.value();

    std::vector<VarPtr> sharing_group;
    if (var_sharing_groups.count(var)) {
      sharing_group = var_sharing_groups[var];
    } else {
      sharing_group = {var};
    }

    // Compute MERGED lifetime for all variables in the sharing group
    int min_def_point = INT_MAX;
    int max_last_use = INT_MIN;

    for (const auto& group_var : sharing_group) {
      int def_point = result.var_def_order.count(group_var) ? result.var_def_order.at(group_var) : 0;
      int last_use = result.var_last_use.count(group_var) ? result.var_last_use.at(group_var) : def_point;

      LOG_DEBUG << "Variable " << group_var->name_hint_ << " def=" << def_point << " last_use=" << last_use;

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
    }

    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    auto representative_tile_type = As<TileType>(sharing_group[0]->GetType());
    CHECK(representative_tile_type != nullptr) << "Expected TileType for reuse interval";
    auto memory_space = representative_tile_type->GetMemorySpace();
    CHECK(memory_space.has_value()) << "TileType with MemRef must have memory_space for reuse analysis";
    interval.memory_space = *memory_space;
    interval.size = memref->size_;

    // Extract the defining op name for use in the in-place safety check
    if (result.var_def_stmt.count(sharing_group[0])) {
      if (auto assign = As<AssignStmt>(result.var_def_stmt.at(sharing_group[0]))) {
        if (auto call = As<Call>(assign->value_)) {
          interval.def_op_name = call->op_->name_;
        }
      }
    }

    lifetimes.push_back(interval);

    for (const auto& group_var : sharing_group) {
      processed_vars.insert(group_var);
    }

    LOG_DEBUG << "Lifetime for sharing group (representative: " << sharing_group[0]->name_hint_
              << ", size: " << sharing_group.size() << "): [" << min_def_point << ", " << max_last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  return {lifetimes, var_sharing_groups};
}

/**
 * @brief Check if two TileType variables have fully compatible tile attributes
 *
 * PTO codegen binds a single alloc_tile declaration (shape, dtype, blayout, pad, etc.)
 * to each buffer.  All operations referencing that buffer share the same declaration.
 * Reuse between tiles with different attributes would cause attribute mismatches in
 * the generated PTO IR, leading to incorrect codegen or hardware behaviour.
 *
 * Checked attributes: shape, dtype, and TileView (all fields via TileView::operator==).
 */
bool AreTileTypesCompatible(const VarPtr& var1, const VarPtr& var2) {
  auto t1 = As<TileType>(var1->GetType());
  auto t2 = As<TileType>(var2->GetType());
  if (!t1 || !t2) return true;

  if (t1->dtype_ != t2->dtype_) return false;
  if (!AreExprVectorsEqual(t1->shape_, t2->shape_)) return false;

  bool has_view1 = t1->tile_view_.has_value();
  bool has_view2 = t2->tile_view_.has_value();
  if (has_view1 != has_view2) return false;
  if (has_view1 && t1->tile_view_.value() != t2->tile_view_.value()) return false;
  return true;
}

/**
 * @brief Check if two lifetimes overlap.
 *
 * Uses <= to allow "touching" lifetimes (last_use == def_point) to share
 * buffers: within a single statement, inputs are consumed before outputs
 * are produced.
 */
static bool LifetimesOverlap(const LifetimeInterval& a, const LifetimeInterval& b) {
  return !(a.last_use_point <= b.def_point || b.last_use_point <= a.def_point);
}

/**
 * @brief Identify memory reuse opportunities from lifetime intervals
 */
std::map<VarPtr, VarPtr> IdentifyReuseOpportunities(const std::vector<LifetimeInterval>& lifetimes) {
  std::map<VarPtr, VarPtr> reuse_map;

  // Build a fast lookup map: VarPtr -> LifetimeInterval for O(1) access
  // This avoids repeated std::find_if calls which were O(n)
  std::map<VarPtr, const LifetimeInterval*> var_to_lifetime;
  for (const auto& interval : lifetimes) {
    var_to_lifetime[interval.variable] = &interval;
  }

  // Track which variables are reusing each source variable's MemRef
  // This is critical to avoid multiple variables with overlapping lifetimes
  // sharing the same MemRef, which would cause memory corruption
  std::map<VarPtr, std::vector<VarPtr>> memref_users;  // source_var -> list of vars reusing it

  // Group variables by memory_space (preserve order within each group)
  std::map<MemorySpace, std::vector<size_t>> groups;  // memory_space -> indices in lifetimes
  for (size_t i = 0; i < lifetimes.size(); i++) {
    groups[lifetimes[i].memory_space].push_back(i);
  }

  // For each memory space, find reuse opportunities
  for (auto& [space, indices] : groups) {
    // Greedy matching: for each variable, try to reuse from previous variables
    for (size_t i = 1; i < indices.size(); i++) {
      size_t curr_idx = indices[i];
      const auto& curr_lifetime = lifetimes[curr_idx];
      VarPtr curr_var = curr_lifetime.variable;

      // Find best candidate to reuse from (earliest with sufficient size)
      for (size_t j = 0; j < i; j++) {
        size_t prev_idx = indices[j];
        const auto& prev_lifetime = lifetimes[prev_idx];
        VarPtr prev_var = prev_lifetime.variable;

        // Check if lifetimes overlap with source variable.
        // Use <= to allow "touching" lifetimes (last_use == def_point) to be
        // merged: within a single statement, inputs are consumed before outputs
        // are produced, so a variable whose last use is in the same statement as
        // another variable's definition can safely share the same buffer.
        bool overlaps_with_source = LifetimesOverlap(prev_lifetime, curr_lifetime);

        // Check if size is sufficient
        bool size_ok = prev_lifetime.size >= curr_lifetime.size;

        if (overlaps_with_source || !size_ok) {
          continue;  // Cannot reuse due to overlap with source or insufficient size
        }

        // Check full TileType compatibility (shape, dtype, TileView attributes)
        if (!AreTileTypesCompatible(curr_var, prev_var)) {
          continue;
        }

        // CRITICAL: Check if current variable's lifetime overlaps with ANY variable
        // that is already reusing the same MemRef (transitive reuse check).
        // Follow the reuse chain to the root, since all variables in the chain
        // share the same physical MemRef.
        VarPtr root = prev_var;
        while (reuse_map.count(root)) {
          root = reuse_map.at(root);
        }
        bool overlaps_with_users = false;
        // When prev_var itself reuses another variable, we must also check
        // against root (the ultimate MemRef owner) since it's not tracked
        // in memref_users.
        if (root != prev_var) {
          const LifetimeInterval* root_lifetime = var_to_lifetime[root];
          if (root_lifetime) {
            bool overlaps = LifetimesOverlap(*root_lifetime, curr_lifetime);
            if (overlaps) {
              overlaps_with_users = true;
              LOG_DEBUG << "Variable " << curr_var->name_hint_ << " cannot reuse " << prev_var->name_hint_
                        << " due to overlap with root MemRef owner " << root->name_hint_;
            }
          }
        }
        if (!overlaps_with_users && memref_users.count(root)) {
          for (const auto& user_var : memref_users[root]) {
            if (user_var == prev_var) continue;  // Already checked in source overlap
            const LifetimeInterval* user_lifetime = var_to_lifetime[user_var];
            if (user_lifetime) {
              bool overlaps = LifetimesOverlap(*user_lifetime, curr_lifetime);

              if (overlaps) {
                overlaps_with_users = true;
                LOG_DEBUG << "Variable " << curr_var->name_hint_ << " cannot reuse " << prev_var->name_hint_
                          << " due to overlap with existing user " << user_var->name_hint_ << " (lifetime ["
                          << curr_lifetime.def_point << ", " << curr_lifetime.last_use_point << "] vs ["
                          << user_lifetime->def_point << ", " << user_lifetime->last_use_point << "])";
                break;
              }
            }
          }
        }

        if (!overlaps_with_users) {
          // For inplace-unsafe ops (src buffer == dst buffer not supported), block reuse
          // whenever the buffer is still occupied at curr_var's definition statement.
          // A conflict exists if root or any variable sharing root's buffer has
          // last_use == curr_var.def — meaning it is still being read as an input
          // to the inplace-unsafe op that defines curr_var (src == dst).
          if (!curr_lifetime.def_op_name.empty()) {
            auto& registry = OpRegistry::GetInstance();
            if (registry.IsRegistered(curr_lifetime.def_op_name) &&
                !registry.GetEntry(curr_lifetime.def_op_name).IsInplaceSafe()) {
              // Check root (the ultimate buffer owner)
              const LifetimeInterval* root_lifetime =
                  var_to_lifetime.count(root) ? var_to_lifetime.at(root) : nullptr;
              bool inplace_conflict =
                  root_lifetime && root_lifetime->last_use_point == curr_lifetime.def_point;

              // Check all variables sharing root's buffer
              if (!inplace_conflict && memref_users.count(root)) {
                for (const auto& user_var : memref_users.at(root)) {
                  const LifetimeInterval* user_lifetime =
                      var_to_lifetime.count(user_var) ? var_to_lifetime.at(user_var) : nullptr;
                  if (user_lifetime && user_lifetime->last_use_point == curr_lifetime.def_point) {
                    inplace_conflict = true;
                    break;
                  }
                }
              }

              if (inplace_conflict) {
                LOG_DEBUG << "Variable " << curr_var->name_hint_ << " cannot reuse " << prev_var->name_hint_
                          << " (op=" << curr_lifetime.def_op_name
                          << " does not support in-place execution, buffer still occupied at def)";
                continue;
              }
            }
          }

          // Can safely reuse!
          reuse_map[curr_var] = prev_var;
          memref_users[root].push_back(curr_var);  // Track under root MemRef owner
          LOG_DEBUG << "Variable " << curr_var->name_hint_ << " can reuse " << prev_var->name_hint_
                    << " (lifetime [" << curr_lifetime.def_point << ", " << curr_lifetime.last_use_point
                    << "]"
                    << " vs [" << prev_lifetime.def_point << ", " << prev_lifetime.last_use_point << "])";
          break;  // Found a reuse target, stop searching
        }
      }
    }
  }

  return reuse_map;
}

/**
 * @brief Apply MemRef sharing to the statement tree
 */
StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const std::map<VarPtr, VarPtr>& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups) {
  // Custom IRMutator for MemRef sharing
  class MemRefSharingMutator : public IRMutator {
   public:
    explicit MemRefSharingMutator(const std::map<VarPtr, VarPtr>& reuse_map,
                                  const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups)
        : reuse_map_(reuse_map), sharing_groups_(sharing_groups) {}

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      // Check if this variable should reuse another's MemRef
      if (reuse_map_.count(op->var_)) {
        VarPtr source_var = reuse_map_.at(op->var_);

        // Get source's TileType and MemRef
        auto source_tile_type = As<TileType>(source_var->GetType());

        if (!source_tile_type || !source_tile_type->memref_.has_value()) {
          LOG_ERROR << "Source variable " << source_var->name_hint_ << " does not have MemRef";
          return IRMutator::VisitStmt_(op);
        }

        std::optional<MemRefPtr> source_memref = source_tile_type->memref_;

        // Get current variable's TileType
        auto curr_tile_type = As<TileType>(op->var_->GetType());

        if (!curr_tile_type) {
          LOG_ERROR << "Current variable " << op->var_->name_hint_ << " is not TileType";
          return IRMutator::VisitStmt_(op);
        }

        // Create new TileType with shared MemRef
        auto new_tile_type = std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
            curr_tile_type, source_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); }));

        // Create new Var
        auto new_var = std::make_shared<const Var>(op->var_->name_hint_, new_tile_type, op->var_->span_);

        // Record the variable substitution mapping (old -> new)
        // This ensures that all subsequent references to the old variable will be replaced with the new one
        var_substitution_map_[op->var_] = new_var;

        // CRITICAL: If this variable shares MemRef with others (view operations),
        // we need to update ALL of them to use the new MemRef
        if (sharing_groups_.count(op->var_)) {
          const auto& sharing_group = sharing_groups_.at(op->var_);
          for (const auto& shared_var : sharing_group) {
            if (shared_var != op->var_) {
              // Create new Var for shared variable with same reused MemRef
              auto shared_tile_type = As<TileType>(shared_var->GetType());
              if (shared_tile_type) {
                auto new_shared_tile_type =
                    std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
                        shared_tile_type, source_memref,
                        [this](const ExprPtr& expr) { return VisitExpr(expr); }));
                auto new_shared_var = std::make_shared<const Var>(shared_var->name_hint_,
                                                                  new_shared_tile_type, shared_var->span_);
                var_substitution_map_[shared_var] = new_shared_var;

                LOG_DEBUG << "Propagating reuse to sharing group member: " << shared_var->name_hint_;
              }
            }
          }
        }

        // Visit value expression (this will recursively apply substitutions)
        ExprPtr new_value = VisitExpr(op->value_);

        return std::make_shared<const AssignStmt>(new_var, new_value, op->span_);
      }

      return IRMutator::VisitStmt_(op);
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
    const std::map<VarPtr, VarPtr>& reuse_map_;
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups_;  // var -> sharing group
    // Maps old variable objects to new variable objects (with reused MemRef)
    // This is needed because IR nodes are immutable, so we create new Var objects
    // and need to replace all references to the old ones
    std::map<VarPtr, VarPtr> var_substitution_map_;
  };

  MemRefSharingMutator mutator(reuse_map, var_sharing_groups);
  return mutator.VisitStmt(stmt);
}

/**
 * @brief Fix yield/return_var MemRef mismatch after MemoryReuse.
 *
 * Handles both ForStmt and IfStmt:
 * - ForStmt: MemoryReuse may assign different MemRefs to iter_arg and yield value.
 *   Since yield value becomes the next iteration's iter_arg, they must use the
 *   same buffer. When they differ, insert a tile.move before yield.
 * - IfStmt: MemoryReuse may change MemRefs of variables inside branches.
 *   Patch return_vars to match the yield value's MemRef.
 */
class YieldFixupMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(result);
    if (!for_stmt || for_stmt->iter_args_.empty()) return result;

    auto yield_stmt = FindYieldStmt(for_stmt->body_);
    if (!yield_stmt) return result;

    // Check each (iter_arg, yield_value) pair for MemRef mismatch.
    // Use initValue's MemRef as the target because codegen maps iter_arg to initValue's buffer,
    // and MemoryReuse may have updated initValue's MemRef without updating iter_arg's.
    std::vector<std::pair<size_t, VarPtr>> moves_to_insert;  // (index, new_moved_var)
    std::vector<StmtPtr> move_stmts;
    for (size_t i = 0; i < yield_stmt->value_.size() && i < for_stmt->iter_args_.size(); ++i) {
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      if (!yield_var) continue;

      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);

      // Get the initValue's MemRef (the actual buffer used at runtime)
      auto init_var = As<Var>(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      if (MemRef::SameAllocation(yield_memref, init_memref)) continue;

      // MemRef mismatch — create tile.move to copy yield value into initValue's buffer
      auto target_memory = init_tile->GetMemorySpace();
      auto [moved_var, move_stmt] = CreateTileMove(yield_var, init_memref, target_memory);

      move_stmts.emplace_back(std::move(move_stmt));
      moves_to_insert.emplace_back(i, moved_var);
    }

    if (moves_to_insert.empty()) {
      // Even without tile.move insertion, iter_arg and return_var may have stale MemRefs
      // (MemoryReuse updates initValue/yield but not iter_arg/return_var types).
      // Ensure all 4 loop-carry variables share the same MemRef (= initValue's).
      return PatchIterArgsAndReturnVars(for_stmt, yield_stmt);
    }

    // Build new yield values with moved vars substituted
    std::vector<ExprPtr> new_yield_values = yield_stmt->value_;
    for (const auto& [idx, moved_var] : moves_to_insert) {
      new_yield_values[idx] = moved_var;
    }
    auto new_yield = std::make_shared<YieldStmt>(new_yield_values, yield_stmt->span_);

    // Insert tile.move stmts before yield and replace yield in body
    auto new_body = InsertMovesAndReplaceYield(for_stmt->body_, new_yield, move_stmts);

    // Build intermediate ForStmt with new body, then patch iter_args/return_vars
    auto intermediate_for =
        std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                  for_stmt->iter_args_, new_body, for_stmt->return_vars_, for_stmt->span_,
                                  for_stmt->kind_, for_stmt->chunk_config_, for_stmt->attrs_);

    return PatchIterArgsAndReturnVars(intermediate_for, new_yield);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto if_stmt = As<IfStmt>(result);
    if (!if_stmt || if_stmt->return_vars_.empty()) return result;

    // Find yield statements in each branch
    auto then_yield = FindYieldStmt(if_stmt->then_body_);
    auto else_yield = if_stmt->else_body_.has_value() ? FindYieldStmt(if_stmt->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return result;

    // For each return_var position, check if the two branches yield tiles
    // with different MemRefs.  When they differ, insert tile.move in the
    // branch whose MemRef ≠ the canonical target (then-branch is canonical).
    bool body_changed = false;
    std::vector<std::pair<size_t, VarPtr>> else_moves;
    std::vector<StmtPtr> else_move_stmts;
    std::vector<VarPtr> new_return_vars = if_stmt->return_vars_;

    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      VarPtr then_var =
          (then_yield && i < then_yield->value_.size()) ? As<Var>(then_yield->value_[i]) : nullptr;
      VarPtr else_var =
          (else_yield && i < else_yield->value_.size()) ? As<Var>(else_yield->value_[i]) : nullptr;

      auto then_tile = then_var ? GetTileTypeWithMemRef(then_var->GetType()) : nullptr;
      auto else_tile = else_var ? GetTileTypeWithMemRef(else_var->GetType()) : nullptr;
      if (!then_tile && !else_tile) continue;

      // Use then-branch MemRef as canonical target (fall back to else if then absent)
      auto target_tile = then_tile ? then_tile : else_tile;
      auto target_memref = GetDefinedMemRef(target_tile);
      auto target_memory = target_tile->GetMemorySpace();

      // Check else branch: if MemRef differs from target, insert tile.move
      if (then_tile && else_tile) {
        auto else_memref = GetDefinedMemRef(else_tile);
        if (!MemRef::SameAllocation(else_memref, target_memref)) {
          auto [moved_var, move_stmt] = CreateTileMove(else_var, target_memref, target_memory);
          else_move_stmts.emplace_back(std::move(move_stmt));
          else_moves.emplace_back(i, moved_var);
          body_changed = true;
        }
      }

      // Patch return_var to share target MemRef
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (rv_tile && rv_tile->memref_.has_value()) {
        auto rv_memref = GetDefinedMemRef(rv_tile);
        if (!MemRef::SameAllocation(rv_memref, target_memref)) {
          auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
              rv_tile, target_memref, [this](const ExprPtr& e) { return VisitExpr(e); }, target_memory);
          new_return_vars[i] =
              std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
          var_remap_[if_stmt->return_vars_[i].get()] = new_return_vars[i];
          body_changed = true;
        }
      }
    }

    if (!body_changed) return result;

    // Rebuild else body with tile.move stmts inserted before yield
    auto new_else_body = if_stmt->else_body_;
    if (!else_moves.empty() && else_yield && if_stmt->else_body_.has_value()) {
      std::vector<ExprPtr> new_else_yield_values = else_yield->value_;
      for (const auto& [idx, moved_var] : else_moves) {
        new_else_yield_values[idx] = moved_var;
      }
      auto new_yield = std::make_shared<YieldStmt>(new_else_yield_values, else_yield->span_);
      new_else_body = InsertMovesAndReplaceYield(if_stmt->else_body_.value(), new_yield, else_move_stmts);
    }

    return std::make_shared<IfStmt>(if_stmt->condition_, if_stmt->then_body_, new_else_body,
                                    std::move(new_return_vars), if_stmt->span_);
  }

 private:
  // Create a tile.move operation that copies source into target_memref's buffer.
  // Returns (moved_var, move_assign_stmt).
  std::pair<VarPtr, StmtPtr> CreateTileMove(const VarPtr& source, const MemRefPtr& target_memref,
                                            std::optional<MemorySpace> target_memory) {
    INTERNAL_CHECK(target_memory.has_value())
        << "Internal error: target TileType must have memory_space for tile.move";
    auto& op_reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_memory", std::any(target_memory.value())}};
    auto move_call = op_reg.Create("tile.move", {source}, kwargs, source->span_);
    auto moved_type = CloneTypeWithMemRefAndRemapExprs(
        source->GetType(), target_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        target_memory);
    auto moved_var = std::make_shared<Var>(source->name_hint_ + "_mv", moved_type, source->span_);
    auto move_stmt = std::make_shared<AssignStmt>(moved_var, move_call, source->span_);
    return {moved_var, move_stmt};
  }

  // Patch iter_args and return_vars to share initValue's MemRef.
  // Returns the original ForStmt if no patching is needed.
  StmtPtr PatchIterArgsAndReturnVars(const ForStmtPtr& for_stmt, const YieldStmtPtr& yield_stmt) {
    bool changed = false;
    std::vector<IterArgPtr> new_iter_args = for_stmt->iter_args_;
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto init_var = As<Var>(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Patch iter_arg if its MemRef differs from initValue's
      auto ia_tile = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (ia_tile && ia_tile->memref_.has_value()) {
        auto ia_memref = GetDefinedMemRef(ia_tile);
        if (!MemRef::SameAllocation(ia_memref, init_memref)) {
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
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      if (!yield_var) continue;
      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!rv_tile || !rv_tile->memref_.has_value()) continue;
      auto rv_memref = GetDefinedMemRef(rv_tile);
      if (!MemRef::SameAllocation(rv_memref, yield_memref)) {
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

    return std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                     new_iter_args, patched_body, std::move(new_return_vars), for_stmt->span_,
                                     for_stmt->kind_, for_stmt->chunk_config_, for_stmt->attrs_);
  }

  // Replace YieldStmt in body and insert move AssignStmts before it.
  // Body structure is typically SeqStmts([...assigns..., YieldStmt]).
  // Move stmts go directly into the SeqStmts before the yield.
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
      std::vector<StmtPtr> new_children;
      for (const auto& child : seq->stmts_) {
        if (As<YieldStmt>(child)) {
          // Insert move stmts directly before the new yield
          new_children.insert(new_children.end(), move_stmts.begin(), move_stmts.end());
          new_children.push_back(new_yield);
        } else {
          new_children.push_back(child);
        }
      }
      return SeqStmts::Flatten(std::move(new_children), body->span_);
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
  if (call->op_->name_ != "tile.alloc" && call->op_->name_ != "tensor.alloc") return false;
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

/**
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities by walking the full
 * IR tree to compute variable lifetimes, then applying greedy MemRef sharing.
 * Variables that can share memory will point to the same MemRef object.
 * After sharing, redundant alloc operations are removed.
 */
FunctionPtr TransformMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MemoryReusePass cannot run on null function";

  // Step 1: Compute lifetimes by walking full IR tree
  auto analysis_result = ComputeLifetimes(func->body_);

  if (analysis_result.lifetimes.empty()) {
    LOG_WARN << "No TileType variables found, skipping memory reuse";
    return func;
  }

  // Step 2: Identify reuse opportunities
  auto reuse_map = IdentifyReuseOpportunities(analysis_result.lifetimes);

  // Step 3: Apply MemRef sharing (skip if no reuse candidates)
  StmtPtr new_body = func->body_;
  if (!reuse_map.empty()) {
    new_body = ApplyMemRefSharing(func->body_, reuse_map, analysis_result.var_sharing_groups);
  }

  // Step 4: Fix ForStmt/IfStmt yield/return_var MemRef mismatches
  YieldFixupMutator yield_fixup;
  new_body = yield_fixup.VisitStmt(new_body);

  // Step 5: Remove alloc statements for MemRefs no longer in use
  auto used_bases = memref_collectors::CollectUsedBasePtrs(new_body);
  new_body = RemoveUnusedAllocStatements(new_body, used_bases);

  auto result = std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_);
  return result;
}

}  // namespace

namespace pass {
Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, "MemoryReuse", kMemoryReuseProperties); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
