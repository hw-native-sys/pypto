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
#include <queue>
#include <set>
#include <vector>

#include "pypto/core/logging.h"
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

  // Step 1: Assign topological order to all statements
  auto stmt_order = AssignTopologicalOrder(blocks, dependencies);

  if (stmt_order.empty()) {
    LOG_WARN << "Failed to compute topological order";
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
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        auto tile_type = std::dynamic_pointer_cast<const TileType>(assign->var_->GetType());
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
      } else if (auto eval_stmt = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
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
    auto tile_type = std::dynamic_pointer_cast<const TileType>(var->GetType());
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

        // Check if lifetimes overlap
        bool overlaps = !(prev_lifetime.last_use_point < curr_lifetime.def_point ||
                          curr_lifetime.last_use_point < prev_lifetime.def_point);

        // Check if size is sufficient
        bool size_ok = prev_lifetime.size >= curr_lifetime.size;

        if (!overlaps && size_ok) {
          // Can reuse!
          reuse_map[curr_var] = prev_var;
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
        auto source_tile_type = std::dynamic_pointer_cast<const TileType>(source_var->GetType());

        if (!source_tile_type || !source_tile_type->memref_.has_value()) {
          LOG_ERROR << "Source variable " << source_var->name_ << " does not have MemRef";
          return IRMutator::VisitStmt_(op);
        }

        std::optional<MemRefPtr> source_memref = source_tile_type->memref_;

        // Get current variable's TileType
        auto curr_tile_type = std::dynamic_pointer_cast<const TileType>(op->var_->GetType());

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

        // Visit value expression
        ExprPtr new_value = VisitExpr(op->value_);

        return std::make_shared<const AssignStmt>(new_var, new_value, op->span_);
      }

      return IRMutator::VisitStmt_(op);
    }

   private:
    const std::map<VarPtr, VarPtr>& reuse_map_;
  };

  MemRefSharingMutator mutator(reuse_map);
  return mutator.VisitStmt(stmt);
}

std::map<StmtPtr, int> BasicMemoryReusePass::AssignTopologicalOrder(
    const std::vector<BasicBlock>& blocks, const std::vector<DependencyEdge>& dependencies) {
  std::map<StmtPtr, int> order;

  // Build adjacency list for dependency graph
  std::map<StmtPtr, std::vector<StmtPtr>> successors;  // stmt -> stmts that depend on it
  std::map<StmtPtr, int> in_degree;

  // Collect all statements in order (for deterministic ordering)
  std::vector<StmtPtr> all_stmts;
  for (const auto& block : blocks) {
    for (const auto& stmt : block.statements) {
      all_stmts.push_back(stmt);
      in_degree[stmt] = 0;
    }
  }

  // Build graph from dependencies
  for (const auto& edge : dependencies) {
    successors[edge.producer].push_back(edge.consumer);
    in_degree[edge.consumer]++;
  }

  // Topological sort using Kahn's algorithm
  std::queue<StmtPtr> queue;

  // Find all nodes with in-degree 0 (preserve original order for determinism)
  for (const auto& stmt : all_stmts) {
    if (in_degree[stmt] == 0) {
      queue.push(stmt);
    }
  }

  int current_order = 0;
  while (!queue.empty()) {
    StmtPtr stmt = queue.front();
    queue.pop();

    order[stmt] = current_order++;

    // Decrease in-degree for successors
    if (successors.count(stmt)) {
      for (const auto& succ : successors[stmt]) {
        in_degree[succ]--;
        if (in_degree[succ] == 0) {
          queue.push(succ);
        }
      }
    }
  }

  // Check if all statements were processed (no cycles)
  if (order.size() != all_stmts.size()) {
    LOG_WARN << "Dependency graph has cycles, topological order is incomplete";
  }

  LOG_DEBUG << "Assigned topological order to " << order.size() << " statements";

  return order;
}

}  // namespace ir
}  // namespace pypto
