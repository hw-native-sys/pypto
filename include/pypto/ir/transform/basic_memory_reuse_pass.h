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

#ifndef PYPTO_IR_TRANSFORM_BASIC_MEMORY_REUSE_PASS_H_
#define PYPTO_IR_TRANSFORM_BASIC_MEMORY_REUSE_PASS_H_

#include <map>
#include <memory>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/transform/base/pass.h"
#include "pypto/ir/transform/dependency_graph.h"

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

/**
 * @brief Pass for basic memory reuse based on dependency graph
 *
 * This pass identifies memory reuse opportunities using ONLY dependency
 * relationships (topological ordering), NOT execution timing simulation.
 * Variables that can share memory will point to the same MemRef object.
 *
 * Strategy:
 * - Uses DependencyAnalyzer to compute dependency graph
 * - Computes lifetimes based on topological order in dependency graph
 * - Two variables can reuse memory if:
 *   1. Same memory_space
 *   2. Non-overlapping lifetimes (based on def/use points)
 * - Reuse is implemented by MemRef pointer sharing (not address copying)
 */
class BasicMemoryReusePass : public Pass {
 public:
  BasicMemoryReusePass() = default;
  ~BasicMemoryReusePass() override = default;

  /**
   * @brief Execute the basic memory reuse pass
   *
   * @param func Input function
   * @return Function with MemRef sharing applied
   */
  FunctionPtr Run(const FunctionPtr& func) override;

 private:
  /**
   * @brief Compute lifetime intervals based on dependency graph
   *
   * Uses topological ordering of statements, not execution timing.
   *
   * @param blocks Basic blocks
   * @param dependencies Dependency edges
   * @return Vector of LifetimeIntervals (ordered by def_point)
   */
  std::vector<LifetimeInterval> ComputeLifetimesFromDependencies(
      const std::vector<BasicBlock>& blocks, const std::vector<DependencyEdge>& dependencies);

  /**
   * @brief Identify variables that can share memory
   *
   * Greedy algorithm: for each variable, find the earliest previous variable
   * in the same memory space with non-overlapping lifetime.
   *
   * @param lifetimes Lifetime intervals (must be ordered by def_point)
   * @return Reuse map: var_new -> var_old (var_new reuses var_old's MemRef)
   */
  std::map<VarPtr, VarPtr> IdentifyReuseOpportunities(const std::vector<LifetimeInterval>& lifetimes);

  /**
   * @brief Apply MemRef sharing to function body
   *
   * Updates TileType variables to share MemRef objects.
   *
   * @param stmt Original statement
   * @param reuse_map Reuse relationships
   * @return Transformed statement with MemRef sharing
   */
  StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const std::map<VarPtr, VarPtr>& reuse_map);

  /**
   * @brief Assign topological order to statements
   *
   * @param blocks Basic blocks
   * @param dependencies Dependency edges
   * @return Map of Stmt -> topological order
   */
  std::map<StmtPtr, int> AssignTopologicalOrder(const std::vector<BasicBlock>& blocks,
                                                const std::vector<DependencyEdge>& dependencies);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_BASIC_MEMORY_REUSE_PASS_H_
