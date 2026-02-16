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

#ifndef PYPTO_IR_TRANSFORMS_DEPENDENCY_ANALYZER_H_
#define PYPTO_IR_TRANSFORMS_DEPENDENCY_ANALYZER_H_

#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/dependency_graph.h"

namespace pypto {
namespace ir {

/**
 * @brief Dependency analyzer for IR functions
 *
 * This class analyzes data dependencies and control flow in IR functions.
 * It is NOT a pass - it's a pure analysis tool that can be called by
 * any pass that needs dependency information.
 *
 * Inherits from IRMutator to leverage function information retrieval capabilities.
 *
 * Usage:
 *   DependencyAnalyzer analyzer;
 *   DependencyGraph graph = analyzer.Analyze(func);
 *   // Use graph.blocks and graph.dependencies
 */
class DependencyAnalyzer : public IRMutator {
 public:
  DependencyAnalyzer() = default;
  ~DependencyAnalyzer() override = default;

  /**
   * @brief Analyze a function and return its dependency graph
   *
   * This is the main entry point. Call this to get complete dependency
   * analysis results for a function.
   *
   * @param func Function to analyze
   * @return Complete dependency graph (blocks + edges)
   */
  DependencyGraph Analyze(const FunctionPtr& func);

  /**
   * @brief Analyze only basic blocks (without dependency edges)
   *
   * Useful when you only need CFG structure.
   *
   * @param func Function to analyze
   * @return Vector of basic blocks
   */
  std::vector<BasicBlock> AnalyzeBasicBlocks(const FunctionPtr& func);

  /**
   * @brief Analyze only dependencies (without basic blocks)
   *
   * Analyzes dependencies assuming a single basic block.
   *
   * @param func Function to analyze
   * @return Vector of dependency edges
   */
  std::vector<DependencyEdge> AnalyzeDependencies(const FunctionPtr& func);

 private:
  /**
   * @brief Identify basic blocks from control flow
   *
   * @param stmt Function body
   * @return Vector of basic blocks
   */
  std::vector<BasicBlock> IdentifyBasicBlocks(const StmtPtr& stmt);

  /**
   * @brief Build dependency graph for a basic block
   *
   * @param block Basic block to analyze
   * @return Vector of dependency edges
   */
  std::vector<DependencyEdge> AnalyzeBlockDependencies(const BasicBlock& block);

  /**
   * @brief Extract pipe type from a statement
   *
   * @param stmt Statement to analyze
   * @return Pipe type string (CUBE, VECTOR, MTE1, etc.)
   */
  std::string GetPipeTypeFromStmt(const StmtPtr& stmt);

  /**
   * @brief Extract pipe type from a Call expression
   *
   * @param call_expr Call expression
   * @return Pipe type string
   */
  std::string GetPipeType(const CallPtr& call_expr);

  /**
   * @brief Merge dependencies from multiple control flow paths
   *
   * @param path_dependencies Dependencies from each path
   * @return Merged dependencies (union)
   */
  std::vector<DependencyEdge> MergeDependencies(
      const std::vector<std::vector<DependencyEdge>>& path_dependencies);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DEPENDENCY_ANALYZER_H_
