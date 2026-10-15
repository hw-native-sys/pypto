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

#ifndef SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_CACHE_H_
#define SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_CACHE_H_

#include <map>
#include <optional>
#include <set>
#include <vector>

namespace pypto::ir::structural_comparison_detail {

// A variable acquires its mapped identity once. Keep reverse edges rather than
// copying transitive dependency sets into each DAG node. Binding an unrelated
// variable costs one lookup; invalidation visits only cached ancestors of uses
// whose identity actually changed. Edges and entries have stable addresses.
template <typename Key, typename Variable, typename Value>
class BindingAwareCache {
 public:
  struct Entry {
    std::optional<Value> value;
    std::set<Entry*> parents;
  };

  Entry& Get(const Key& key, Entry* parent = nullptr) {
    auto& entry = entries_[key];
    if (parent) entry.parents.insert(parent);
    return entry;
  }

  void DependOn(Entry* entry, const Variable& variable) {
    if (entry) dependents_[variable].insert(entry);
  }

  void Bind(const Variable& variable) {
    auto found = dependents_.find(variable);
    if (found == dependents_.end()) return;
    std::vector<Entry*> pending(found->second.begin(), found->second.end());
    dependents_.erase(found);
    while (!pending.empty()) {
      auto* entry = pending.back();
      pending.pop_back();
      // An invalid ancestor already propagated invalidation. In-progress
      // entries have no cached value and are populated after their traversal.
      if (!entry->value) continue;
      entry->value.reset();
      pending.insert(pending.end(), entry->parents.begin(), entry->parents.end());
    }
  }

 private:
  std::map<Key, Entry> entries_;
  std::map<Variable, std::set<Entry*>> dependents_;
};

}  // namespace pypto::ir::structural_comparison_detail

#endif  // SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_COMPARISON_CACHE_H_
