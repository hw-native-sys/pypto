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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_MEMORY_FOOTPRINT_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MEMORY_FOOTPRINT_H_

#include <cstdint>

#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"

namespace pypto {
namespace ir {

/// The ordering + alignment bump walk shared by AllocateMemoryAddr (to emit each physical buffer's
/// base address) and MemoryReuse (to track the per-space high-water mark). Driving both from one
/// walk guarantees the packer's fit check equals the allocator's realized footprint — the
/// "footprint = allocator" invariant that keeps a capacity-gated reuse decision from separating
/// operands the allocator can't actually place (#1475).
///
/// A space's physical buffers are placed sequentially starting at `reserved_start` (the end of any
/// reserved region). Each buffer of `slot_size` bytes — a base-group sized to its largest member —
/// lands at the current bump position, then the cursor advances to
/// `policy.AlignAddress(base + slot_size, space)`. The high-water mark is the final cursor.
class SpaceFootprint {
 public:
  SpaceFootprint(MemorySpace space, const MemoryAllocatorPolicy& policy, uint64_t reserved_start = 0)
      : space_(space), policy_(&policy), current_addr_(reserved_start) {}

  /// Reserve one physical buffer of `slot_size` bytes. Returns its base address and advances the
  /// high-water mark past it. `slot_size` is the buffer's largest member (a base-group's slot size).
  [[nodiscard]] uint64_t OpenBuffer(uint64_t slot_size) {
    const uint64_t base = current_addr_;
    current_addr_ = policy_->AlignAddress(current_addr_ + slot_size, space_);
    return base;
  }

  /// The realized footprint so far — identical to the address AllocateMemoryAddr bumps to after the
  /// same sequence of buffers. Compare against the space capacity for the reuse fit check.
  [[nodiscard]] uint64_t HighWater() const { return current_addr_; }

 private:
  MemorySpace space_;
  const MemoryAllocatorPolicy* policy_;
  uint64_t current_addr_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MEMORY_FOOTPRINT_H_
