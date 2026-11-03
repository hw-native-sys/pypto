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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Conservative lifetime of one physical allocation identity.
 *
 * Views and mandatory aliases that share one base MemRef are represented by a
 * single interval. Opportunistic reuse between different intervals remains a
 * placement decision.
 */
struct LifetimeInterval {
  VarPtr variable;
  int def_point;
  int last_use_point;
  MemorySpace memory_space;
  uint64_t size;
};

// Direct descriptor bindings, rather than a must-alias equivalence relation.
// A native If result can name either yielded allocation; its representative
// MemRef is only a geometry hint. Renames, views and registered writebacks
// preserve this provenance. Loop bindings retain their fixed initializer edge.
using TileStorageSources = std::map<const Var*, std::vector<VarPtr>>;

/// Native choices and the loop boundaries connected to them through descriptor
/// aliases. A carry's initializer alone does not describe all runtime storage.
struct TileSelectionInfo {
  std::set<const Var*> values;
  std::set<const IterArg*> carries;
};

/// Exclude loop backedges when identifying independently retained descriptors:
/// an advancing carry is not an old selection merely because its next value is.
[[nodiscard]] TileSelectionInfo AnalyzeTileSelections(const StmtPtr& body, bool include_backedges = true);

/**
 * @brief Shared result of allocation-lifetime analysis.
 */
struct LifetimeAnalysisResult {
  std::vector<LifetimeInterval> lifetimes;
  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  std::map<const Var*, std::set<int>> phi_family_ids;
  std::map<const Var*, std::pair<int, int>> var_liveness;
  std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>> pipeline_membership;
  std::set<const Var*> pipeline_load_tiles;
  // Sharing-group representatives with a dynamic-offset member that PTO cannot
  // reconstruct from the view op's operands. Any memory space containing one
  // must keep the legacy whole-buffer layout: subdividing other groups could
  // otherwise make an unsafe program reach codegen instead of overflowing.
  std::set<const Var*> subrange_unsafe_groups;
  // Present on the legacy Tile path. Keep the graph compact instead of eagerly
  // materializing every transitive root set (quadratic for growing nested phis).
  TileStorageSources storage_sources;
};

/**
 * @brief Analyze conservative allocation lifetimes and alias families.
 */
[[nodiscard]] LifetimeAnalysisResult AnalyzeAllocationLifetimes(const StmtPtr& func_body);

/**
 * @brief Analyze allocation lifetimes, including on-chip Tile parameters.
 *
 * Function parameters are live on entry and have no defining AssignStmt in the
 * body. Function-wide allocation planners must use this overload so those
 * allocation identities participate in placement and writeback.
 */
[[nodiscard]] LifetimeAnalysisResult AnalyzeAllocationLifetimes(const FunctionPtr& func);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
