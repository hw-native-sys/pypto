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

#ifndef PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_
#define PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_

#include <cstdint>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/dsa_reuse_penalty_solver.h"

namespace pypto {
namespace backend {
class Backend;
}
namespace ir {

class MemoryAllocatorPolicy;

namespace dsa_adapter {

using MemRefWithSpace = std::pair<MemRefPtr, MemorySpace>;

/**
 * @brief In-memory DSA-RP problem and transient writeback information.
 *
 * No schema or filesystem representation is involved. ``pipeline_pairs`` are
 * exactly the hard separations that become legal if pipeline intent must be
 * relaxed; target and semantic separations never appear in this list.
 */
struct PreparedProblem {
  dsa::DsaProblem strict_problem;
  std::unordered_map<const Var*, dsa::BufferId> buffer_id_by_base;
  std::set<const Var*> declared_allocation_bases;
  std::vector<dsa::Separation> pipeline_pairs;
};

/**
 * @brief Translate PyPTO allocation facts into the narrow in-tree DSA-RP model.
 *
 * Buffer lifetimes use ``ConvertToDsaExecutionLifetime``, the same conservative
 * read/write event convention as the standalone research adapter.
 */
[[nodiscard]] PreparedProblem BuildProblem(
    const FunctionPtr& func, const AllocationPlan& allocation_plan, const MemoryAllocatorPolicy& policy,
    const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
    const std::unordered_map<MemorySpace, uint64_t>& pool_caps, const backend::Backend* backend);

/**
 * @brief Remove only pipeline-only hard relations and price the newly legal reuse.
 */
[[nodiscard]] dsa::DsaProblem RelaxPipelineIntent(const PreparedProblem& prepared);

/**
 * @brief Convert validated offsets to fresh MemRefs, preserving view offsets.
 */
[[nodiscard]] std::vector<std::pair<const MemRef*, MemRefPtr>> BuildMemRefReplacements(
    const PreparedProblem& prepared, const dsa::DsaSolution& solution,
    const std::vector<MemRefWithSpace>& memrefs, const MemoryAllocatorPolicy& policy);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_
