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
