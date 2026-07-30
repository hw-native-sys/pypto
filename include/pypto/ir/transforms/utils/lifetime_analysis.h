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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"

namespace pypto {
namespace ir {

/**
 * @brief Lifetime interval for one physical allocation identity.
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

enum class AllocationSeparationReason : uint8_t {
  PipelineStage,
  TargetHazard,
  SemanticNoAlias,
  StorageLayout,
  DeclaredAllocation,
};

struct AllocationSeparation {
  size_t first;
  size_t second;
  std::vector<AllocationSeparationReason> reasons;
};

/**
 * @brief Compiler-derived inputs shared by DSA placement and hazard recognition.
 */
struct AllocationPlan {
  std::vector<LifetimeInterval> intervals;
  std::vector<AllocationSeparation> separations;
};

/**
 * @brief Compute conservative allocation lifetimes and mandatory separations.
 */
[[nodiscard]] AllocationPlan ComputeAllocationPlan(const FunctionPtr& func);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
