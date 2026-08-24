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

#include "pypto/ir/transforms/dsa/allocation_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {

AllocationPlan BuildDsaAllocationPlan(const FunctionPtr& func) {
  LifetimeAnalysisResult analysis = AnalyzeAllocationLifetimes(func);
  const AllocationConstraintAnalysis constraints = AnalyzeAllocationConstraints(func, analysis, "DSA-RP");

  AllocationPlan plan;
  plan.intervals = std::move(analysis.lifetimes);
  plan.declared_allocation_sizes = constraints.declared_allocation_sizes;

  // DSA places a physical allocation identity. Its extent must cover every
  // mandatory alias/view and the full author-declared allocation, not merely
  // the representative tile.
  for (LifetimeInterval& interval : plan.intervals) {
    uint64_t allocation_size = interval.size;
    const auto sharing = analysis.var_sharing_groups.find(interval.variable);
    if (sharing != analysis.var_sharing_groups.end()) {
      for (const VarPtr& member : sharing->second) {
        const auto tile_type = As<TileType>(member->GetType());
        INTERNAL_CHECK_SPAN(tile_type != nullptr && tile_type->memref_.has_value(), member->span_)
            << "Expected every allocation sharing-group member to carry a MemRef";
        allocation_size = std::max(allocation_size, GetDefinedMemRef(tile_type)->size_);
      }
    }
    const auto representative_memref = GetTypeMemRef(interval.variable->GetType());
    if (representative_memref.has_value() && representative_memref.value()) {
      const auto declared =
          constraints.declared_allocation_sizes.find(representative_memref.value()->base_.get());
      if (declared != constraints.declared_allocation_sizes.end()) {
        allocation_size = std::max(allocation_size, declared->second);
      }
    }
    interval.size = allocation_size;
  }
  const auto& intervals = plan.intervals;

  // Pair-producing analyses are indexed by pipeline group, statement point,
  // or MemRef base. Their cost is O(N log N + E log E), where E is the number
  // of hard relations materialized in the DSA problem.
  std::map<std::pair<size_t, size_t>, std::set<AllocationSeparationReason>> separation_reasons;
  auto add_separation = [&separation_reasons](size_t first, size_t second,
                                              AllocationSeparationReason reason) {
    if (first == second) return;
    if (second < first) std::swap(first, second);
    separation_reasons[{first, second}].insert(reason);
  };

  std::unordered_map<const Var*, size_t> base_to_index;
  std::vector<size_t> pinned_intervals;
  for (size_t index = 0; index < intervals.size(); ++index) {
    const auto memref = GetTypeMemRef(intervals[index].variable->GetType());
    if (!memref.has_value() || !memref.value()) continue;
    base_to_index[memref.value()->base_.get()] = index;
    if (constraints.declared_allocation_bases.count(memref.value()->base_.get()) != 0) {
      pinned_intervals.push_back(index);
    }
  }

  // An author-declared allocation is closed: only values explicitly bound to
  // its base may occupy it. This explicit-pair model is output-sensitive.
  for (size_t pinned : pinned_intervals) {
    for (size_t other = 0; other < intervals.size(); ++other) {
      if (other != pinned && intervals[other].memory_space == intervals[pinned].memory_space) {
        add_separation(pinned, other, AllocationSeparationReason::DeclaredAllocation);
      }
    }
  }

  // A5 Vec storage is representation-sensitive: lifetime-disjoint ND and NZ
  // tiles still need different physical addresses.  Materialize only the
  // cross-class pairs, preserving reuse within each representation family.
  std::vector<size_t> vec_nd_intervals;
  std::vector<size_t> vec_nz_intervals;
  for (size_t index = 0; index < intervals.size(); ++index) {
    const auto layout = constraints.vec_nz_layout_class.find(intervals[index].variable.get());
    if (layout == constraints.vec_nz_layout_class.end()) continue;
    (layout->second ? vec_nz_intervals : vec_nd_intervals).push_back(index);
  }
  auto add_disjoint_layout_pairs = [&intervals, &add_separation](std::vector<size_t> earlier,
                                                                 std::vector<size_t> later) {
    std::sort(earlier.begin(), earlier.end(), [&intervals](size_t lhs, size_t rhs) {
      return std::pair{intervals[lhs].last_use_point, lhs} < std::pair{intervals[rhs].last_use_point, rhs};
    });
    std::sort(later.begin(), later.end(), [&intervals](size_t lhs, size_t rhs) {
      return std::pair{intervals[lhs].def_point, lhs} < std::pair{intervals[rhs].def_point, rhs};
    });

    size_t eligible = 0;
    for (size_t later_index : later) {
      while (eligible < earlier.size() &&
             intervals[earlier[eligible]].last_use_point <= intervals[later_index].def_point) {
        ++eligible;
      }
      for (size_t prior = 0; prior < eligible; ++prior) {
        add_separation(earlier[prior], later_index, AllocationSeparationReason::StorageLayout);
      }
    }
  };
  add_disjoint_layout_pairs(vec_nd_intervals, vec_nz_intervals);
  add_disjoint_layout_pairs(vec_nz_intervals, vec_nd_intervals);

  // Preserve requested software-pipeline depth as hard separations first. The
  // DSA-RP driver alone may later relax this typed policy under capacity pressure.
  using GroupKey = std::pair<MemorySpace, int32_t>;
  std::map<GroupKey, std::map<int32_t, std::vector<size_t>>> group_members;
  for (size_t index = 0; index < intervals.size(); ++index) {
    const LifetimeInterval& interval = intervals[index];
    const auto membership = analysis.pipeline_membership.find(interval.variable.get());
    if (membership == analysis.pipeline_membership.end()) continue;
    for (const auto& [group, stage] : membership->second) {
      group_members[{interval.memory_space, group}][stage].push_back(index);
    }
  }
  for (const auto& [group, members_by_stage] : group_members) {
    static_cast<void>(group);
    for (auto first_bucket = members_by_stage.begin(); first_bucket != members_by_stage.end();
         ++first_bucket) {
      for (auto second_bucket = std::next(first_bucket); second_bucket != members_by_stage.end();
           ++second_bucket) {
        for (size_t first : first_bucket->second) {
          for (size_t second : second_bucket->second) {
            add_separation(first, second, AllocationSeparationReason::PipelineStage);
          }
        }
      }
    }
  }

  // Backend-specific split-AIV load+tpop correctness hazard, bucketed by the
  // one statement boundary at which illegal in-place reuse can occur.
  if (constraints.needs_load_tpop_hazard_guard) {
    std::map<int, std::vector<size_t>> writers_by_def;
    std::map<int, std::vector<size_t>> inputs_by_last_use;
    for (size_t index = 0; index < intervals.size(); ++index) {
      const LifetimeInterval& interval = intervals[index];
      if (constraints.target_hazard_inputs.reads_tpop.count(interval.variable.get()) != 0) {
        writers_by_def[interval.def_point].push_back(index);
      }
      if (constraints.target_hazard_inputs.load_derived.count(interval.variable.get()) != 0) {
        inputs_by_last_use[interval.last_use_point].push_back(index);
      }
    }
    for (const auto& [point, writers] : writers_by_def) {
      const auto inputs = inputs_by_last_use.find(point);
      if (inputs == inputs_by_last_use.end()) continue;
      for (size_t writer : writers) {
        for (size_t input : inputs->second) {
          add_separation(writer, input, AllocationSeparationReason::TargetHazard);
        }
      }
    }
  }

  // Op-semantic no-alias constraints resolve operands through allocation-base identity.
  for (size_t index = 0; index < intervals.size(); ++index) {
    const auto forbidden = constraints.forbid_alias.find(intervals[index].variable.get());
    if (forbidden == constraints.forbid_alias.end()) continue;
    for (const VarPtr& operand : forbidden->second) {
      const auto memref = GetTypeMemRef(operand->GetType());
      if (!memref.has_value() || !memref.value()) continue;
      const auto found = base_to_index.find(memref.value()->base_.get());
      if (found != base_to_index.end()) {
        add_separation(index, found->second, AllocationSeparationReason::SemanticNoAlias);
      }
    }
  }

  plan.separations.reserve(separation_reasons.size());
  for (const auto& [indices, reasons] : separation_reasons) {
    plan.separations.push_back({indices.first, indices.second,
                                std::vector<AllocationSeparationReason>(reasons.begin(), reasons.end())});
  }
  return plan;
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
