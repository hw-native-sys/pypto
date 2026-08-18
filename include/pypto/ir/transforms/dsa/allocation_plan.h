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

#ifndef PYPTO_IR_TRANSFORMS_DSA_ALLOCATION_PLAN_H_
#define PYPTO_IR_TRANSFORMS_DSA_ALLOCATION_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {

enum class AllocationSeparationReason : uint8_t {
  PipelineStage,
  TargetHazard,
  SemanticNoAlias,
  DeclaredAllocation,
};

struct AllocationSeparation {
  size_t first;
  size_t second;
  std::vector<AllocationSeparationReason> reasons;
};

/**
 * @brief Compiler-derived inputs to DSA placement and reuse-hazard recognition.
 */
struct AllocationPlan {
  std::vector<LifetimeInterval> intervals;
  std::vector<AllocationSeparation> separations;
  /// Full byte extent of each author-declared allocation. This can exceed any
  /// member MemRef when the declaration contains multiple runtime-selected
  /// slots.
  std::map<const Var*, uint64_t> declared_allocation_sizes;
};

/**
 * @brief Build conservative DSA lifetimes and mandatory separations.
 */
[[nodiscard]] AllocationPlan BuildDsaAllocationPlan(const FunctionPtr& func);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_ALLOCATION_PLAN_H_
