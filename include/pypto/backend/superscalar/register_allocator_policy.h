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

#ifndef PYPTO_BACKEND_SUPERSCALAR_REGISTER_ALLOCATOR_POLICY_H_
#define PYPTO_BACKEND_SUPERSCALAR_REGISTER_ALLOCATOR_POLICY_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"

namespace pypto {
namespace backend {

/**
 * @brief Register-renaming allocation policy for the SuperscalarNPU TREG file.
 *
 * TREG is a register file of 256 fixed 4KB blocks (1MB total), addressed by
 * block index. Liveness-based coalescing (the MemoryReuse pass) has already
 * merged non-overlapping tiles onto shared MemRefs, so each surviving MemRef is
 * a distinct "live value" that needs its own block(s). This policy then assigns
 * block indices to those values — the address-allocation half of linear-scan
 * register allocation.
 *
 * Key behaviours (consumed by AllocateMemoryAddr):
 * - ShouldAllocate: allocate every on-chip space, skip DDR (managed externally).
 * - AddressUnitBytes(TREG) = 4096: the stored MemRef address is a *block index*
 *   rather than a byte address.
 * - AlignAddress(TREG): round up to a 4KB block boundary, so a tile larger than
 *   one block occupies consecutive blocks and the next tile starts on a fresh
 *   block.
 * - MaxAddressUnits(TREG) = 256: allocating more than 256 live blocks raises a
 *   user-facing register-pressure error.
 */
class SuperscalarNPURegisterAllocatorPolicy : public ir::MemoryAllocatorPolicy {
 public:
  /// Size of one TREG block in bytes.
  static constexpr uint64_t kBlockSizeBytes = 4096;
  /// Number of TREG blocks in the register file.
  static constexpr uint64_t kNumBlocks = 256;

  [[nodiscard]] bool ShouldAllocate(ir::MemorySpace space) const override {
    // DDR addresses are managed externally; every on-chip space is allocated.
    return space != ir::MemorySpace::DDR;
  }

  [[nodiscard]] uint64_t AlignAddress(uint64_t addr, ir::MemorySpace space) const override {
    if (space == ir::MemorySpace::TREG) {
      // Round up to a whole 4KB block so each tile occupies full blocks.
      return (addr + kBlockSizeBytes - 1) & ~(kBlockSizeBytes - 1);
    }
    // Any other on-chip space falls back to 32-byte byte addressing.
    return (addr + 31) & ~static_cast<uint64_t>(31);
  }

  void OrderMemRefs(std::vector<ir::MemRefPtr>& refs) const override {
    std::sort(refs.begin(), refs.end(),
              [](const ir::MemRefPtr& a, const ir::MemRefPtr& b) { return a->name_hint_ < b->name_hint_; });
  }

  [[nodiscard]] uint64_t AddressUnitBytes(ir::MemorySpace space) const override {
    return space == ir::MemorySpace::TREG ? kBlockSizeBytes : 1;
  }

  [[nodiscard]] uint64_t MaxAddressUnits(ir::MemorySpace space) const override {
    return space == ir::MemorySpace::TREG ? kNumBlocks : UINT64_MAX;
  }
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_SUPERSCALAR_REGISTER_ALLOCATOR_POLICY_H_
