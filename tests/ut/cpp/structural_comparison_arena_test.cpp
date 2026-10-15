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

#include "ir/transforms/utils/structural_comparison_arena.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace {
namespace detail = pypto::ir::structural_comparison_detail;

struct CollidingHash {
  uint64_t operator()(const std::string&) const { return 0; }
};

struct CountingCompare {
  inline static size_t comparisons = 0;
  bool operator()(const detail::Key& left, const detail::Key& right) const {
    ++comparisons;
    return left < right;
  }
};

void TestCollisionsDoNotImplyEquality() {
  detail::ExactArena<CollidingHash> arena;
  auto left = arena.Intern("float", "0x0");
  auto right = arena.Intern("float", "0x8000000000000000");
  assert(arena.Hash(left) == arena.Hash(right));
  assert(left != right);
  for (size_t i = 0; i < 100; ++i) {
    left = arena.Intern("add", "", {{"left", left}, {"right", left}});
    right = arena.Intern("add", "", {{"left", right}, {"right", right}});
    assert(arena.Hash(left) == arena.Hash(right));
    assert(left != right);
  }
}

void TestSharedDagSizeAndLookupBound() {
  detail::ExactArena<CollidingHash, CountingCompare> arena;
  auto root = arena.Intern("leaf", "0");
  constexpr size_t kNodes = 4096;
  for (size_t i = 0; i < kNodes; ++i) {
    auto left = arena.Intern("pair", "", {{"left", root}, {"right", root}});
    auto right = arena.Intern("pair", "", {{"left", root}, {"right", root}});
    assert(left == right);
    root = left;
  }
  assert(arena.Size() == kNodes + 2);
  // Ordered lookup is logarithmic even when every digest collides. A tree
  // expansion would require 2^4096 leaves; quadratic lookup exceeds this bound.
  assert(CountingCompare::comparisons < 100 * kNodes);
}

void TestDigestDoesNotDependOnArenaIds() {
  detail::ExactArena<> left;
  detail::ExactArena<> right;
  right.Intern("unrelated", "record");
  auto a = left.Intern("leaf", "1");
  auto b = right.Intern("leaf", "1");
  assert(a != b);
  a = left.Intern("pair", "", {{"left", a}, {"right", a}});
  b = right.Intern("pair", "", {{"left", b}, {"right", b}});
  assert(left.Hash(a) == right.Hash(b));
}

}  // namespace

int main() {
  TestCollisionsDoNotImplyEquality();
  TestSharedDagSizeAndLookupBound();
  TestDigestDoesNotDependOnArenaIds();
}
