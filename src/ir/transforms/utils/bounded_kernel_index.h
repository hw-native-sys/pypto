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

#ifndef SRC_IR_TRANSFORMS_UTILS_BOUNDED_KERNEL_INDEX_H_
#define SRC_IR_TRANSFORMS_UTILS_BOUNDED_KERNEL_INDEX_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace pypto::ir::deduplication_detail {
// A candidate pays for both projections before comparing. This bounds collision
// work even when many tiny candidates collide with a very large representative.
// Capacity/budget exhaustion conservatively retains the original definition.
template <typename Unit>
class BoundedIndex {
 public:
  template <typename Equal>
  const Unit* FindOrInsert(uint64_t hash, const Unit& unit, Equal equal) {
    auto& bucket = buckets_[hash];
    size_t budget = unit.weight > std::numeric_limits<size_t>::max() / kCapacity
                        ? std::numeric_limits<size_t>::max()
                        : kCapacity * unit.weight;
    for (const auto* representative : bucket) {
      if (unit.weight > budget || representative->weight > budget - unit.weight) continue;
      budget -= unit.weight + representative->weight;
      if (equal(unit, *representative)) return representative;
    }
    if (bucket.size() < kCapacity) bucket.push_back(&unit);
    return &unit;
  }

 private:
  static constexpr size_t kCapacity = 8;
  std::map<uint64_t, std::vector<const Unit*>> buckets_;
};
}  // namespace pypto::ir::deduplication_detail
#endif  // SRC_IR_TRANSFORMS_UTILS_BOUNDED_KERNEL_INDEX_H_
