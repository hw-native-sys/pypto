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

#ifndef SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_ARENA_H_
#define SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_ARENA_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pypto::ir::structural_comparison_detail {

// Exact interning, not hash-based equality. Each record contains only scalar
// payloads and child IDs; shared DAGs are never expanded into trees. Ordered
// indexing costs O(N log N) including all fields/string bytes, with O(N) space.
// Digests use child digests, NOT arena IDs, so independent invocations agree.
using Id = size_t;
using Fields = std::vector<std::pair<std::string, Id>>;
using Key = std::tuple<std::string, std::string, Fields>;

inline uint64_t Combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct Record {
  Key key;
  uint64_t hash;
};

template <typename AtomHash = std::hash<std::string>, typename Compare = std::less<Key>>
class ExactArena {
 public:
  ExactArena() { Intern("null", "", {}); }

  Id Intern(std::string tag, std::string atom, Fields fields = {}) {
    Key key{std::move(tag), std::move(atom), std::move(fields)};
    auto found = index_.find(key);
    if (found != index_.end()) return found->second;
    uint64_t hash = Combine(AtomHash{}(std::get<0>(key)), AtomHash{}(std::get<1>(key)));
    for (const auto& [name, child] : std::get<2>(key)) {
      hash = Combine(hash, AtomHash{}(name));
      hash = Combine(hash, records_[child].hash);
    }
    Id id = records_.size();
    index_.emplace(key, id);
    records_.push_back({std::move(key), hash});
    return id;
  }

  [[nodiscard]] uint64_t Hash(Id id) const { return records_[id].hash; }

  [[nodiscard]] const Key& GetKey(Id id) const { return records_[id].key; }
  [[nodiscard]] size_t Size() const { return records_.size(); }

 private:
  std::map<Key, Id, Compare> index_;
  std::vector<Record> records_;
};

}  // namespace pypto::ir::structural_comparison_detail

#endif  // SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_ARENA_H_
