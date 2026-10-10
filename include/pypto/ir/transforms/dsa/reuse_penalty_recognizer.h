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
 * @brief How the recognizer enumerates candidate allocation pairs.
 *
 * Both strategies return the same relations. `ReferenceAllPairs` is the
 * readable specification — every allocation pair, filtered by the promotion
 * policy — and exists so tests can pin the indexed sweep the compiler runs.
 */
enum class ReuseEnumeration : uint8_t {
  IndexedSweep,
  ReferenceAllPairs,
};

/**
 * @brief Recognize the compiler's built-in DSA reuse-penalty policy.
 *
 * The recognizer emits one unit-weight relation per buffer pair for which
 * physical reuse can introduce a cross-resource WAR or WAW handoff. It keeps
 * the maximal accesses of the earlier allocation and the complete minimal
 * initial-write frontier of the later one, then promotes a pair when some
 * maximal access and some first write use two different abstract resources.
 * A pair is eligible only when both allocations have a complete, classified,
 * full-allocation access set and a verified initial write. Same-resource,
 * partial-view, structurally ambiguous, and uncertain handoffs remain
 * unpenalized, as do correctness and pipeline-intent separations.
 *
 * Ordering comes from a chain-cover reachability index over the statement
 * dependency graph: each abstract resource is one completion-ordered issue
 * chain, so one query costs O(1) and the whole index costs O(V + E) for a
 * fixed number of resources. The recognizer does not invoke or simulate ptoas.
 *
 * The route taxonomy is target independent. The active backend decides only
 * whether the selected SoC can perform an operation's transfer at all, and an
 * operation it cannot classify leaves its allocations unpenalized.
 */
[[nodiscard]] std::vector<RecognizedReusePenalty> RecognizeReusePenalties(
    const FunctionPtr& func, const AllocationPlan& allocation_plan, const backend::Backend& backend,
    ReuseEnumeration enumeration = ReuseEnumeration::IndexedSweep);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_
