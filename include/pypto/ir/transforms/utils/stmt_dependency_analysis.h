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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_STMT_DEPENDENCY_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_STMT_DEPENDENCY_ANALYSIS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace stmt_dep {

/**
 * @brief Dataflow dependency graph over the top-level statements of a region.
 *
 * Nodes are the region's top-level statements:
 *   - If the region is a SeqStmts, each child of `stmts_` becomes a node.
 *   - Otherwise the region itself is the single node.
 *
 * Edges are predecessor relationships at block granularity: a compound child
 * (IfStmt/ForStmt/...) aggregates all the variable uses and defs from its
 * subtree. `predecessors[X]` is the set of nodes whose defs X directly reads.
 *
 * The graph is sound under the InOut-use discipline (RFC #1026): physical
 * memory mutation is mirrored by SSA version changes, so SSA def-use captures
 * all real dependencies. Callers needing soundness must first run
 * CheckInOutUseDiscipline and refuse to proceed on any violation.
 */
struct StmtDependencyGraph {
  /// Top-level stmts in region order.
  std::vector<StmtPtr> stmts;

  /// predecessors[X] = set of stmts in `stmts` that X directly depends on.
  /// Keyed by raw pointer; the StmtPtr ownership is held in `stmts`.
  std::unordered_map<const Stmt*, std::unordered_set<const Stmt*>> predecessors;
};

/**
 * @brief Build the statement dependency graph for a region.
 *
 * Pure dataflow analysis over SSA def-use. Does not check the InOut-use
 * discipline; callers that need soundness should call CheckInOutUseDiscipline
 * first and refuse to proceed on any violation.
 *
 * Complexity: O(N * avg_fanout) where N is the number of statements in the
 * region — a single pass with per-stmt use/def collection.
 *
 * @param region The region (typically a SeqStmts) to analyze. If `region` is
 *               not a SeqStmts, the graph has a single node and no edges.
 * @return Dependency graph with nodes and predecessor edges.
 */
StmtDependencyGraph BuildStmtDependencyGraph(const StmtPtr& region);

/**
 * @brief Check that the InOut-use discipline (RFC #1026) holds over a region.
 *
 * The discipline: for any user-function call that passes variable `v` as an
 * InOut or Out parameter, `v` must not be read by any statement reachable
 * from the call in CFG order. Post-mutation values flow exclusively through
 * the call's return slots.
 *
 * InOut and Out are treated identically — both cause `v` to be "dead for read"
 * after the call.
 *
 * Built-in ops (tile.*, tensor.*, system.*) do not contribute to the dead set
 * here; their memory mutations are handled separately (Mode B in RFC #1026)
 * and are out of scope for this dataflow analysis.
 *
 * @param region The region to validate (typically a SeqStmts).
 * @param program The program — used to resolve Call::op_ to a Function and
 *                look up param_directions_.
 * @return A list of Diagnostics; empty iff the discipline holds.
 */
std::vector<Diagnostic> CheckInOutUseDiscipline(const StmtPtr& region, const ProgramPtr& program);

}  // namespace stmt_dep
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_STMT_DEPENDENCY_ANALYSIS_H_
