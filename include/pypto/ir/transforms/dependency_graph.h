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

#ifndef PYPTO_IR_TRANSFORMS_DEPENDENCY_GRAPH_H_
#define PYPTO_IR_TRANSFORMS_DEPENDENCY_GRAPH_H_

#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Represents a basic block in the control flow graph
 */
struct BasicBlock {
  int id;                           // Unique identifier
  std::vector<StmtPtr> statements;  // Statements in this block
  std::vector<int> predecessors;    // Predecessor block IDs
  std::vector<int> successors;      // Successor block IDs
  bool is_loop_body = false;        // True if this block is a loop body
};

/**
 * @brief Represents a data dependency edge in the dependency graph
 */
struct DependencyEdge {
  enum Type { RAW, WAR, WAW };  // Read-After-Write, Write-After-Read, Write-After-Write

  StmtPtr producer;           // Producer statement
  StmtPtr consumer;           // Consumer statement
  VarPtr variable;            // Variable creating the dependency
  Type type;                  // Dependency type
  std::string producer_pipe;  // Pipe type of producer (e.g., "VECTOR")
  std::string consumer_pipe;  // Pipe type of consumer
};

/**
 * @brief Complete dependency graph analysis result
 */
struct DependencyGraph {
  std::vector<BasicBlock> blocks;            // All basic blocks
  std::vector<DependencyEdge> dependencies;  // All dependency edges

  DependencyGraph() = default;
  DependencyGraph(std::vector<BasicBlock> blks, std::vector<DependencyEdge> deps)
      : blocks(std::move(blks)), dependencies(std::move(deps)) {}
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DEPENDENCY_GRAPH_H_
