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

#include "pypto/ir/transforms/dsa/memref_dsa_adapter.h"

#include <algorithm>
#include <cstdint>
#include <set>

#include "pypto/ir/memory_allocator_policy.h"

namespace pypto {
namespace ir {
namespace dsa {

DsaProblem BuildDsaProblem(const std::vector<LifetimeInterval>& lifetimes, const MemoryAllocatorPolicy& policy,
                           const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
                           const std::unordered_map<MemorySpace, uint64_t>& pool_caps) {
  DsaProblem problem;
  std::set<MemorySpace> spaces_seen;

  for (size_t i = 0; i < lifetimes.size(); ++i) {
    const LifetimeInterval& li = lifetimes[i];
    if (li.memory_space == MemorySpace::DDR) continue;      // off-chip — not planned here
    if (!policy.ShouldAllocate(li.memory_space)) continue;  // backend excludes this space

    // Probe the space's alignment granule (smallest aligned address >= 1) and
    // round the buffer's size up to it, so the packed peak is comparable to the
    // bump allocator (which aligns each buffer's end via AlignAddress).
    const uint64_t granule = std::max<uint64_t>(1, policy.AlignAddress(1, li.memory_space));

    Buffer b;
    b.id = static_cast<BufferId>(i);
    b.size = policy.AlignAddress(li.size, li.memory_space);
    b.align = granule;
    b.pool = static_cast<PoolId>(li.memory_space);
    b.interval.start = li.def_point;
    b.interval.end = li.last_use_point;
    problem.buffers.push_back(b);
    spaces_seen.insert(li.memory_space);
  }

  for (MemorySpace space : spaces_seen) {
    const PoolId pool = static_cast<PoolId>(space);
    auto rb = reserved_end_by_space.find(space);
    if (rb != reserved_end_by_space.end()) problem.reserved_base[pool] = rb->second;
    auto cap = pool_caps.find(space);
    if (cap != pool_caps.end()) problem.pool_caps[pool] = cap->second;
  }

  return problem;
}

}  // namespace dsa
}  // namespace ir
}  // namespace pypto
