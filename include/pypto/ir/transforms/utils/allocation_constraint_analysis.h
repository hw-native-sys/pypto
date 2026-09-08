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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ALLOCATION_CONSTRAINT_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ALLOCATION_CONSTRAINT_ANALYSIS_H_

#include <cstdint>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {

struct AllocationHazardInputs {
  std::unordered_set<const Var*> load_derived;
  std::unordered_set<const Var*> reads_tpop;
  /// Vars passed to an operand the operator declares as a written workspace
  /// (`set_workspace_arg` + a writing `set_arg_effect`). The intrinsic writes
  /// that buffer itself, on a pipe PyPTO cannot see, so it must not inherit a
  /// buffer an earlier value occupies.
  std::unordered_set<const Var*> written_workspaces;
  /// MemRef allocation bases of those same operands. A lifetime interval is
  /// keyed on ONE representative of its sharing group, and an `IterArg` never
  /// gets an interval of its own at all (its carry chain -- init value,
  /// IterArg, yield value -- was fused onto a single MemRef base by
  /// MaterializeSemanticAliases). Matching on Var identity alone therefore
  /// misses a loop-carried workspace, and any workspace that is not its
  /// group's representative. The base is the identity that survives both.
  std::unordered_set<const Var*> written_workspace_bases;
  /// The subset of those enclosed in a loop, with their bases. Across a back
  /// edge "earlier" and "later" stop meaning anything: iteration i+1 re-writes
  /// the workspace from the scalar pipe while iteration i's vector reads of a
  /// value sharing that buffer can still be in flight. A looping workspace
  /// therefore needs an exclusive buffer, not just protection from inheriting
  /// one. Only `ForStmt` and `WhileStmt` repeat a body by the time MemoryReuse
  /// runs -- `pl.pipeline` is lowered to a loop by pass 30/31.
  std::unordered_set<const Var*> looping_written_workspaces;
  std::unordered_set<const Var*> looping_written_workspace_bases;
};

using AllocationForbidAliasMap = std::map<const Var*, std::vector<VarPtr>>;
using AllocationExactOrDisjointMap = std::map<const Var*, std::vector<VarPtr>>;

/**
 * @brief Correctness facts shared by legacy reuse and DSA allocation planning.
 */
struct AllocationConstraintAnalysis {
  std::map<const Var*, uint64_t> declared_allocation_sizes;
  std::set<const Var*> declared_allocation_bases;
  AllocationHazardInputs target_hazard_inputs;
  AllocationForbidAliasMap forbid_alias;
  AllocationExactOrDisjointMap exact_or_disjoint_alias;
  bool needs_load_tpop_hazard_guard = false;
};

/**
 * @brief Collect allocation correctness facts without making a placement decision.
 */
[[nodiscard]] AllocationConstraintAnalysis AnalyzeAllocationConstraints(
    const FunctionPtr& func, const LifetimeAnalysisResult& lifetimes, const char* consumer);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ALLOCATION_CONSTRAINT_ANALYSIS_H_
