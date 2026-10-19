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

#include "ir/transforms/utils/bounded_kernel_index.h"

#include <cassert>
#include <vector>

struct Unit {
  size_t weight;
  int value;
};
int main() {
  pypto::ir::deduplication_detail::BoundedIndex<Unit> index;
  Unit huge{1000000, 0};
  auto equal = [](const Unit& a, const Unit& b) { return a.value == b.value; };
  assert(index.FindOrInsert(0, huge, equal) == &huge);
  std::vector<Unit> small(1000, Unit{1, 1});
  size_t work = 0;
  for (const auto& candidate : small) {
    size_t before = work;
    auto match = index.FindOrInsert(0, candidate, [&](const Unit& a, const Unit& b) {
      work += a.weight + b.weight;
      return equal(a, b);
    });
    assert(work - before <= 8 * candidate.weight);
    assert(match->value == 1);
    assert(match == &small.front());
  }
  Unit unequal{1, 2};
  assert(index.FindOrInsert(0, unequal, equal) == &unequal);
  // Sharing differences affect work only, not structural eligibility.
  Unit expanded{2, 1};
  assert(index.FindOrInsert(0, expanded, equal) == &small.front());
}
