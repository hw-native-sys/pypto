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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_REUSE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_REUSE_H_

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"

namespace pypto::ir {

// Physical region reuse shared by Tile codegen and Tile-to-Buffer lowering.
// Keys describe exact compatible slot geometry/layout/valid extents/count.
// Returns (logical base, physical owner) in definition order. Reuse preserves
// all previous occupants' no-alias edges and target-specific hazards.
// O((R + E) log R) for R regions and E explicit allocation constraints.
template <typename Key>
std::vector<std::pair<const Var*, const Var*>> PlanMultiBufferReuse(
    const FunctionPtr& func, const std::map<const Var*, Key>& compatibilities) {
  if (compatibilities.empty()) return {};
  const auto lifetime_analysis = ir::AnalyzeAllocationLifetimes(func);
  std::map<const ir::Var*, std::pair<int, int>> base_lifetimes;
  for (const auto& interval : lifetime_analysis.lifetimes) {
    const auto tile_type = As<TileType>(interval.variable->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) continue;
    const auto memref = ir::GetDefinedMemRef(tile_type);
    // AnalyzeAllocationLifetimes already merges every variable sharing this
    // MemRef base into one [min(def), max(last use)] interval.
    base_lifetimes[memref->base_.get()] = {interval.def_point, interval.last_use_point};
  }

  const auto constraints = ir::AnalyzeAllocationConstraints(func, lifetime_analysis, "MultiBufferReuse");
  auto base_of = [](const ir::Var* var) -> const ir::Var* {
    const auto tile_type = var ? As<TileType>(var->GetType()) : nullptr;
    return tile_type && tile_type->memref_.has_value() ? tile_type->memref_.value()->base_.get() : nullptr;
  };
  // The allocator's no-alias facts refer to logical tile Vars. Resolve their
  // MemRef bases once, then check every occupant of a reused physical region.
  // A region's first owner alone is insufficient after a second allocation has
  // reused it. Hard-edge checks total O(E log R), for R bases and E edges.
  std::map<const ir::Var*, std::set<const ir::Var*>> forbidden_bases;
  for (const auto& [writer, inputs] : constraints.forbid_alias) {
    const ir::Var* writer_base = base_of(writer);
    if (!writer_base) continue;
    for (const auto& input : inputs) {
      const ir::Var* input_base = base_of(input.get());
      if (!input_base || input_base == writer_base) continue;
      forbidden_bases[writer_base].insert(input_base);
      forbidden_bases[input_base].insert(writer_base);
    }
  }
  std::set<const ir::Var*> load_derived_bases;
  std::set<const ir::Var*> reads_tpop_bases;
  for (const ir::Var* var : constraints.target_hazard_inputs.load_derived) {
    if (const ir::Var* base = base_of(var)) load_derived_bases.insert(base);
  }
  for (const ir::Var* var : constraints.target_hazard_inputs.reads_tpop) {
    if (const ir::Var* base = base_of(var)) reads_tpop_bases.insert(base);
  }

  std::vector<std::pair<int, const ir::Var*>> allocation_order;
  allocation_order.reserve(compatibilities.size());
  for (const auto& [base, key] : compatibilities) {
    const auto lifetime_it = base_lifetimes.find(base);
    const int def =
        lifetime_it == base_lifetimes.end() ? std::numeric_limits<int>::max() : lifetime_it->second.first;
    allocation_order.emplace_back(def, base);
  }
  std::stable_sort(allocation_order.begin(), allocation_order.end(),
                   [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  struct AvailableRegion {
    int last_use = 0;
    const ir::Var* owner = nullptr;
  };
  struct EarliestAvailable {
    bool operator()(const AvailableRegion& lhs, const AvailableRegion& rhs) const {
      return lhs.last_use > rhs.last_use;
    }
  };
  using RegionHeap = std::priority_queue<AvailableRegion, std::vector<AvailableRegion>, EarliestAvailable>;
  using RegionPools = std::array<RegionHeap, 2>;
  std::map<Key, RegionPools> available_regions;
  std::map<const ir::Var*, const ir::Var*> region_owners;

  std::vector<std::pair<const Var*, const Var*>> result;
  for (const auto& [def, base] : allocation_order) {
    const auto& compatibility = compatibilities.at(base);
    auto lifetime_it = base_lifetimes.find(base);
    bool reused = false;
    const auto& hazard = constraints.target_hazard_inputs;
    const bool looping_workspace = hazard.looping_written_workspace_bases.count(base) != 0;
    // Written workspaces cannot inherit earlier storage; looping workspaces
    // also cannot lend it, so they never enter the reusable pools below.
    if (lifetime_it != base_lifetimes.end() && !looping_workspace &&
        hazard.written_workspace_bases.count(base) == 0) {
      std::set<const ir::Var*> forbidden_owners;
      if (const auto forbidden = forbidden_bases.find(base); forbidden != forbidden_bases.end()) {
        for (const ir::Var* peer : forbidden->second) {
          if (const auto owner = region_owners.find(peer); owner != region_owners.end()) {
            forbidden_owners.insert(owner->second);
          }
        }
      }
      auto& pools = available_regions[compatibility];
      const size_t pool_count = reads_tpop_bases.count(base) != 0 ? 1 : pools.size();
      std::array<std::vector<AvailableRegion>, 2> blocked;
      std::optional<size_t> selected;
      for (size_t pool = 0; pool < pool_count; ++pool) {
        auto& heap = pools[pool];
        while (!heap.empty() && heap.top().last_use <= lifetime_it->second.first &&
               forbidden_owners.count(heap.top().owner) != 0) {
          blocked[pool].push_back(heap.top());
          heap.pop();
        }
        if (!heap.empty() && heap.top().last_use <= lifetime_it->second.first &&
            (!selected || heap.top().last_use < pools[*selected].top().last_use)) {
          selected = pool;
        }
      }
      if (selected) {
        const auto available = pools[*selected].top();
        pools[*selected].pop();
        region_owners.emplace(base, available.owner);
        const size_t next_pool = *selected != 0 || load_derived_bases.count(base) != 0 ? 1 : 0;
        pools[next_pool].push({lifetime_it->second.second, available.owner});
        reused = true;
      }
      for (size_t pool = 0; pool < pool_count; ++pool) {
        for (const auto& entry : blocked[pool]) pools[pool].push(entry);
      }
    }
    if (!reused) {
      region_owners.emplace(base, base);
      if (lifetime_it != base_lifetimes.end() && !looping_workspace) {
        const size_t pool = load_derived_bases.count(base) != 0 ? 1 : 0;
        available_regions[compatibility][pool].push({lifetime_it->second.second, base});
      }
    }

    result.emplace_back(base, region_owners.at(base));
  }
  return result;
}

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_REUSE_H_
