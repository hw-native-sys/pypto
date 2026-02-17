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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dependency_analyzer.h"
#include "pypto/ir/transforms/dependency_graph.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
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
};

namespace {

/**
 * @brief Assign topological order to all statements in basic blocks
 */
std::map<StmtPtr, int> AssignDeclarationOrder(const std::vector<BasicBlock>& blocks) {
  std::map<StmtPtr, int> order;
  int current_order = 0;

  // Traverse blocks in order and assign order to each statement
  for (const auto& block : blocks) {
    for (const auto& stmt : block.statements) {
      order[stmt] = current_order++;
    }
  }

  LOG_DEBUG << "Assigned declaration order to " << order.size() << " statements";

  return order;
}

/**
 * @brief Result of lifetime computation
 */
struct LifetimeAnalysisResult {
  std::vector<LifetimeInterval> lifetimes;
  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
};

/**
 * @brief Compute lifetime intervals from dependencies
 *
 * This function identifies memory reuse opportunities using ONLY dependency
 * relationships (topological ordering), NOT execution timing simulation.
 */
LifetimeAnalysisResult ComputeLifetimesFromDependencies(const std::vector<BasicBlock>& blocks,
                                                        const std::vector<DependencyEdge>& dependencies) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Assign topological order to all statements
  auto stmt_order = AssignDeclarationOrder(blocks);

  if (stmt_order.empty()) {
    LOG_WARN << "Failed to compute topological order";
    return {lifetimes, {}};
  }

  // Step 2: Build maps: var -> defining stmt, var -> list of using stmts
  // Use vectors to preserve original statement order
  std::vector<VarPtr> ordered_vars;  // Variables in definition order
  std::map<VarPtr, StmtPtr> var_def_stmt;
  std::map<VarPtr, std::vector<StmtPtr>> var_use_stmts;

  // Helper to collect variable uses from an expression
  class VarUseCollector : public IRVisitor {
   public:
    std::set<VarPtr> used_vars;

    void VisitExpr_(const VarPtr& var) override {
      used_vars.insert(var);
      IRVisitor::VisitExpr_(var);
    }
  };

  for (const auto& block : blocks) {
    for (const auto& stmt : block.statements) {
      // Check if this is an AssignStmt defining a TileType variable
      if (auto assign = As<AssignStmt>(stmt)) {
        auto tile_type = As<TileType>(assign->var_->GetType());
        if (tile_type) {
          ordered_vars.push_back(assign->var_);  // Preserve definition order
          var_def_stmt[assign->var_] = stmt;
        }

        // Collect variables used in the value expression (for ALL AssignStmt, not just TileType)
        // This ensures we capture uses in statements like: result = store(tile_e, ...)
        VarUseCollector collector;
        collector.VisitExpr(assign->value_);

        for (const auto& used_var : collector.used_vars) {
          var_use_stmts[used_var].push_back(stmt);
        }
      } else if (auto eval_stmt = As<EvalStmt>(stmt)) {
        // Collect variable uses
        VarUseCollector collector;
        collector.VisitExpr(eval_stmt->expr_);

        for (const auto& used_var : collector.used_vars) {
          var_use_stmts[used_var].push_back(stmt);
        }
      }
    }
  }

  // Step 2.5: Build MemRef sharing groups
  // Variables that share the same MemRef object (via shared_ptr) should be treated
  // as a single logical buffer with merged lifetime
  std::map<const MemRef*, std::vector<VarPtr>> memref_groups;
  for (const auto& var : ordered_vars) {
    auto tile_type = As<TileType>(var->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      const MemRef* memref_ptr = tile_type->memref_.value().get();
      memref_groups[memref_ptr].push_back(var);
    }
  }

  // Build reverse map: var -> all vars sharing same MemRef
  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  for (const auto& [memref_ptr, vars] : memref_groups) {
    if (vars.size() > 1) {
      // Multiple variables share this MemRef
      for (const auto& var : vars) {
        var_sharing_groups[var] = vars;
      }
      LOG_DEBUG << "MemRef sharing group: " << vars.size() << " variables share same MemRef";
    }
  }

  // Step 3: For each TileType variable with MemRef, compute lifetime (in definition order)
  // For variables sharing MemRef, use MERGED lifetime
  std::set<VarPtr> processed_vars;  // Track which vars we've already processed

  for (const auto& var : ordered_vars) {
    if (processed_vars.count(var)) {
      continue;  // Already processed as part of a sharing group
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;  // Skip variables without MemRef
    }

    const auto& memref = tile_type->memref_.value();

    // Check if this variable shares MemRef with others
    std::vector<VarPtr> sharing_group;
    if (var_sharing_groups.count(var)) {
      sharing_group = var_sharing_groups[var];
    } else {
      sharing_group = {var};  // Single variable
    }

    // Compute MERGED lifetime for all variables in the sharing group
    int min_def_point = INT_MAX;
    int max_last_use = INT_MIN;

    for (const auto& group_var : sharing_group) {
      const auto& def_stmt = var_def_stmt[group_var];
      int def_point = stmt_order[def_stmt];
      int last_use = def_point;

      if (var_use_stmts.count(group_var)) {
        LOG_DEBUG << "Variable " << group_var->name_ << " has " << var_use_stmts[group_var].size()
                  << " use statements";
        for (const auto& use_stmt : var_use_stmts[group_var]) {
          if (stmt_order.count(use_stmt)) {
            int use_order = stmt_order[use_stmt];
            LOG_DEBUG << "  Use at order " << use_order;
            last_use = std::max(last_use, use_order);
          }
        }
      } else {
        LOG_DEBUG << "Variable " << group_var->name_ << " has no recorded uses";
      }

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
    }

    // Create ONE lifetime interval for the entire sharing group
    // Use the first variable as the representative
    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    interval.memory_space = memref->memory_space_;
    interval.size = memref->size_;

    lifetimes.push_back(interval);

    // Mark all variables in the group as processed
    for (const auto& group_var : sharing_group) {
      processed_vars.insert(group_var);
    }

    LOG_DEBUG << "Lifetime for sharing group (representative: " << sharing_group[0]->name_
              << ", size: " << sharing_group.size() << "): [" << min_def_point << ", " << max_last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  return {lifetimes, var_sharing_groups};
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

        // Check if lifetimes overlap with source variable
        bool overlaps_with_source = !(prev_lifetime.last_use_point < curr_lifetime.def_point ||
                                      curr_lifetime.last_use_point < prev_lifetime.def_point);

        // Check if size is sufficient
        bool size_ok = prev_lifetime.size >= curr_lifetime.size;

        if (overlaps_with_source || !size_ok) {
          continue;  // Cannot reuse due to overlap with source or insufficient size
        }

        // CRITICAL: Check if current variable's lifetime overlaps with ANY variable
        // that is already reusing the same MemRef (transitive reuse check)
        bool overlaps_with_users = false;
        if (memref_users.count(prev_var)) {
          for (const auto& user_var : memref_users[prev_var]) {
            // Use the fast lookup map instead of std::find_if
            const LifetimeInterval* user_lifetime = var_to_lifetime[user_var];
            if (user_lifetime) {
              bool overlaps = !(user_lifetime->last_use_point < curr_lifetime.def_point ||
                                curr_lifetime.last_use_point < user_lifetime->def_point);

              if (overlaps) {
                overlaps_with_users = true;
                LOG_DEBUG << "Variable " << curr_var->name_ << " cannot reuse " << prev_var->name_
                          << " due to overlap with existing user " << user_var->name_ << " (lifetime ["
                          << curr_lifetime.def_point << ", " << curr_lifetime.last_use_point << "] vs ["
                          << user_lifetime->def_point << ", " << user_lifetime->last_use_point << "])";
                break;
              }
            }
          }
        }

        if (!overlaps_with_users) {
          // Can safely reuse!
          reuse_map[curr_var] = prev_var;
          memref_users[prev_var].push_back(curr_var);  // Track this reuse relationship
          LOG_DEBUG << "Variable " << curr_var->name_ << " can reuse " << prev_var->name_ << " (lifetime ["
                    << curr_lifetime.def_point << ", " << curr_lifetime.last_use_point << "]"
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
          LOG_ERROR << "Source variable " << source_var->name_ << " does not have MemRef";
          return IRMutator::VisitStmt_(op);
        }

        std::optional<MemRefPtr> source_memref = source_tile_type->memref_;

        // Get current variable's TileType
        auto curr_tile_type = As<TileType>(op->var_->GetType());

        if (!curr_tile_type) {
          LOG_ERROR << "Current variable " << op->var_->name_ << " is not TileType";
          return IRMutator::VisitStmt_(op);
        }

        // Create new TileType with shared MemRef
        auto new_tile_type = std::make_shared<const TileType>(curr_tile_type->shape_, curr_tile_type->dtype_,
                                                              source_memref,  // Share MemRef!
                                                              curr_tile_type->tile_view_);

        // Create new Var
        auto new_var = std::make_shared<const Var>(op->var_->name_, new_tile_type, op->var_->span_);

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
                    std::make_shared<const TileType>(shared_tile_type->shape_, shared_tile_type->dtype_,
                                                     source_memref,  // Same reused MemRef!
                                                     shared_tile_type->tile_view_);
                auto new_shared_var =
                    std::make_shared<const Var>(shared_var->name_, new_shared_tile_type, shared_var->span_);
                var_substitution_map_[shared_var] = new_shared_var;

                LOG_DEBUG << "Propagating reuse to sharing group member: " << shared_var->name_;
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
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities using ONLY dependency
 * relationships (topological ordering), NOT execution timing simulation.
 * Variables that can share memory will point to the same MemRef object.
 */
FunctionPtr TransformBasicMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "BasicMemoryReusePass cannot run on null function";

  // Step 1: Use DependencyAnalyzer to get dependency graph
  DependencyAnalyzer analyzer;
  DependencyGraph graph = analyzer.Analyze(func);

  if (graph.blocks.empty()) {
    LOG_WARN << "No basic blocks found, skipping memory reuse";
    return func;
  }

  // Step 2: Compute lifetimes based on dependency graph
  auto analysis_result = ComputeLifetimesFromDependencies(graph.blocks, graph.dependencies);

  if (analysis_result.lifetimes.empty()) {
    LOG_WARN << "No TileType variables found, skipping memory reuse";
    return func;
  }

  // Step 3: Identify reuse opportunities
  auto reuse_map = IdentifyReuseOpportunities(analysis_result.lifetimes);

  if (reuse_map.empty()) {
    return func;
  }

  // Step 4: Apply MemRef sharing
  StmtPtr new_body = ApplyMemRefSharing(func->body_, reuse_map, analysis_result.var_sharing_groups);

  return std::make_shared<const Function>(func->name_, func->params_, func->return_types_, new_body,
                                          func->span_, func->func_type_);
}

}  // namespace

namespace pass {
Pass BasicMemoryReuse() {
  return CreateFunctionPass(TransformBasicMemoryReuse, "BasicMemoryReuse", kBasicMemoryReuseProperties);
}
}  // namespace pass
}  // namespace ir
}  // namespace pypto
