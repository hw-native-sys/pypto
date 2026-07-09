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
#include <unordered_map>
#include <vector>

#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/dsa/dsa_solver.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {

class MemoryAllocatorPolicy;

namespace dsa {

/**
 * @brief Translate per-allocation lifetimes into a DsaProblem (core mapping).
 *
 * One DSA Buffer per allocation: pool = memory space, interval = [def, last_use],
 * size rounded up to the space's alignment (matching the bump's per-buffer
 * footprint), align = the space's alignment granule.  Off-chip (DDR) and
 * non-allocated spaces are skipped.  reserved_base / pool_caps come from the
 * reserve-buffer resolution and the backend's per-space capacities.
 *
 * This is the CORE mapping only.  Must-aliases and views are already folded into
 * the intervals (via base_ identity); opportunistic reuse is the solver's job.
 * Separations for pipeline double-buffers / cross-pipe hazards are NOT yet
 * emitted here — so the resulting packing is a lower bound (consulting mode),
 * not yet safe to make authoritative.
 */
[[nodiscard]] DsaProblem BuildDsaProblem(const std::vector<LifetimeInterval>& lifetimes,
                                         const MemoryAllocatorPolicy& policy,
                                         const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
                                         const std::unordered_map<MemorySpace, uint64_t>& pool_caps);

}  // namespace dsa
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_
