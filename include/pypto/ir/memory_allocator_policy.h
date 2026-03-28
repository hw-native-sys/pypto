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

#ifndef PYPTO_IR_MEMORY_ALLOCATOR_POLICY_H_
#define PYPTO_IR_MEMORY_ALLOCATOR_POLICY_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"

namespace pypto {
namespace ir {

/**
 * @brief Abstract interface for memory address allocation strategy
 *
 * Encapsulates the placement policy used by AllocateMemoryAddr to assign
 * addresses to MemRefs. Backends can provide custom implementations to
 * control alignment requirements, space filtering, and allocation ordering.
 */
class MemoryAllocatorPolicy {
 public:
  virtual ~MemoryAllocatorPolicy() = default;

  /**
   * @brief Whether the given memory space should be allocated by this policy
   *
   * Spaces that return false are skipped entirely (their MemRefs keep
   * the original address). For example, DDR addresses are typically
   * managed externally and should not be allocated by this pass.
   *
   * @param space Memory space to check
   * @return true if addresses should be allocated for this space
   */
  [[nodiscard]] virtual bool ShouldAllocate(MemorySpace space) const = 0;

  /**
   * @brief Align an address for the given memory space
   *
   * Returns the smallest address >= addr that satisfies the alignment
   * requirement of the given memory space.
   *
   * @param addr Raw address to align
   * @param space Memory space whose alignment rule applies
   * @return Aligned address
   */
  [[nodiscard]] virtual uint64_t AlignAddress(uint64_t addr, MemorySpace space) const = 0;

  /**
   * @brief Order MemRefs within a single memory space before allocation
   *
   * The allocation pass calls this to sort the MemRef vector for a space
   * before assigning sequential addresses.  The default strategy sorts
   * by MemRef id for deterministic output.
   *
   * @param refs Mutable vector of MemRefs to sort in-place
   */
  virtual void OrderMemRefs(std::vector<MemRefPtr>& refs) const = 0;
};

using MemoryAllocatorPolicyPtr = std::unique_ptr<MemoryAllocatorPolicy>;

/**
 * @brief Default allocation policy matching the original hard-coded behavior
 *
 * - Skips DDR (addresses managed externally)
 * - Uses 32-byte alignment for all on-chip memory spaces
 * - Sorts MemRefs by id_ for deterministic allocation order
 */
class DefaultMemoryAllocatorPolicy : public MemoryAllocatorPolicy {
 public:
  [[nodiscard]] bool ShouldAllocate(MemorySpace space) const override { return space != MemorySpace::DDR; }

  [[nodiscard]] uint64_t AlignAddress(uint64_t addr, [[maybe_unused]] MemorySpace space) const override {
    return (addr + 31) & ~static_cast<uint64_t>(31);
  }

  void OrderMemRefs(std::vector<MemRefPtr>& refs) const override {
    std::sort(refs.begin(), refs.end(),
              [](const MemRefPtr& a, const MemRefPtr& b) { return a->id_ < b->id_; });
  }
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMORY_ALLOCATOR_POLICY_H_
