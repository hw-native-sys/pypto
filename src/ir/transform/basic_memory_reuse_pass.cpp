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

#include "pypto/ir/transform/basic_memory_reuse_pass.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/mutator.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/ir/transform/dependency_analyzer.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

FunctionPtr BasicMemoryReusePass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "BasicMemoryReusePass cannot run on null function";

  // Step 1: Use DependencyAnalyzer to get dependency graph
  DependencyAnalyzer analyzer;
  DependencyGraph graph = analyzer.Analyze(func);

  LOG_INFO << "Analyzed " << graph.blocks.size() << " blocks, " << graph.dependencies.size() << " edges";

  if (graph.blocks.empty()) {
    LOG_WARN << "No basic blocks found, skipping memory reuse";
    return func;
  }

  // Step 2: Compute lifetimes based on dependency graph
  auto lifetimes = ComputeLifetimesFromDependencies(graph.blocks, graph.dependencies);
  LOG_INFO << "Computed lifetimes for " << lifetimes.size() << " variables";

  if (lifetimes.empty()) {
    LOG_INFO << "No TileType variables found, skipping memory reuse";
    return func;
  }

  // Step 3: Identify reuse opportunities
  auto reuse_map = IdentifyReuseOpportunities(lifetimes);
  LOG_INFO << "Found " << reuse_map.size() << " memory reuse opportunities";

  if (reuse_map.empty()) {
    LOG_INFO << "No memory reuse opportunities, returning original function";
    return func;
  }

  // Step 4: Apply MemRef sharing
  StmtPtr new_body = ApplyMemRefSharing(func->body_, reuse_map);

  return std::make_shared<const Function>(func->name_, func->params_, func->return_types_, new_body,
                                          func->span_);
}

std::vector<LifetimeInterval> BasicMemoryReusePass::ComputeLifetimesFromDependencies(
    const std::vector<BasicBlock>& blocks, const std::vector<DependencyEdge>& dependencies) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Assign declaration order to all statements
  // ASSUMPTION: Input Function IR already satisfies topological ordering
  auto stmt_order = AssignDeclarationOrder(blocks);

  if (stmt_order.empty()) {
    LOG_WARN << "No statements found in blocks";
    return lifetimes;
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

          // Collect variables used in the value expression
          VarUseCollector collector;
          collector.VisitExpr(assign->value_);

          for (const auto& used_var : collector.used_vars) {
            var_use_stmts[used_var].push_back(stmt);
          }
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

  // Step 3: For each TileType variable with MemRef, compute lifetime (in definition order)
  for (const auto& var : ordered_vars) {
    const auto& def_stmt = var_def_stmt[var];
    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;  // Skip variables without MemRef
    }

    const auto& memref = tile_type->memref_.value();

    // Def point
    int def_point = stmt_order[def_stmt];

    // Last use point (find maximum order among all use statements)
    int last_use = def_point;  // At least def point
    if (var_use_stmts.count(var)) {
      for (const auto& use_stmt : var_use_stmts[var]) {
        if (stmt_order.count(use_stmt)) {
          last_use = std::max(last_use, stmt_order[use_stmt]);
        }
      }
    }

    // Create lifetime interval
    LifetimeInterval interval;
    interval.variable = var;
    interval.def_point = def_point;
    interval.last_use_point = last_use;
    interval.memory_space = memref->memory_space_;
    interval.size = memref->size_;

    lifetimes.push_back(interval);

    LOG_DEBUG << "Lifetime for " << var->name_ << ": [" << def_point << ", " << last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  return lifetimes;
}

std::map<VarPtr, VarPtr> BasicMemoryReusePass::IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes) {
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
          LOG_INFO << "Variable " << curr_var->name_ << " can reuse " << prev_var->name_ << " (lifetime ["
                   << curr_lifetime.def_point << ", " << curr_lifetime.last_use_point << "]"
                   << " vs [" << prev_lifetime.def_point << ", " << prev_lifetime.last_use_point << "])";
          break;  // Found a reuse target, stop searching
        }
      }
    }
  }

  return reuse_map;
}

StmtPtr BasicMemoryReusePass::ApplyMemRefSharing(const StmtPtr& stmt,
                                                 const std::map<VarPtr, VarPtr>& reuse_map) {
  // Custom IRMutator for MemRef sharing
  class MemRefSharingMutator : public IRMutator {
   public:
    explicit MemRefSharingMutator(const std::map<VarPtr, VarPtr>& reuse_map) : reuse_map_(reuse_map) {}

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
    // Maps old variable objects to new variable objects (with reused MemRef)
    // This is needed because IR nodes are immutable, so we create new Var objects
    // and need to replace all references to the old ones
    std::map<VarPtr, VarPtr> var_substitution_map_;
  };

  MemRefSharingMutator mutator(reuse_map);
  return mutator.VisitStmt(stmt);
}

std::map<StmtPtr, int> BasicMemoryReusePass::AssignDeclarationOrder(const std::vector<BasicBlock>& blocks) {
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

}  // namespace ir
}  // namespace pypto
