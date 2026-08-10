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

#ifndef PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_
#define PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"

namespace pypto {
namespace backend {
class Backend;
}
namespace ir {
namespace dsa_adapter {

/**
 * @brief A legal but potentially costly physical-overlap relation.
 *
 * The interval indices refer to AllocationPlan::intervals. Cost is an abstract,
 * non-negative optimization priority; it is not a cycle estimate.
 */
struct RecognizedReusePenalty {
  size_t first_interval;
  size_t second_interval;
  uint64_t cost;
};

/**
 * @brief Recognize the compiler's built-in DSA reuse-penalty policy.
 *
 * The recognizer emits one unit-weight relation per buffer pair for which
 * physical reuse can introduce a cross-pipe WAR or WAW handoff. It requires
 * disjoint shared execution lifetimes, a complete access set, full-allocation
 * handoff endpoints, and a verified initial write. Same-pipe, partial-view,
 * structurally ambiguous, and uncertain handoffs remain unpenalized.
 *
 * The active backend supplies execution-pipe classification for supported
 * operation and direct-memory-route combinations. The recognizer does not
 * invoke or simulate ptoas, and skips calls whose backend pipe or physical
 * access contract is unknown.
 */
[[nodiscard]] std::vector<RecognizedReusePenalty> RecognizeReusePenalties(
    const FunctionPtr& func, const AllocationPlan& allocation_plan, const backend::Backend& backend);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_
