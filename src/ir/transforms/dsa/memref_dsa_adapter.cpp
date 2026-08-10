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

#include "pypto/ir/transforms/dsa/memref_dsa_adapter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/dsa_reuse_penalty_solver.h"
#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {
namespace {

dsa::PoolId ToPoolId(MemorySpace space) { return static_cast<dsa::PoolId>(space); }

std::pair<dsa::BufferId, dsa::BufferId> CanonicalPair(dsa::BufferId first, dsa::BufferId second) {
  return first < second ? std::make_pair(first, second) : std::make_pair(second, first);
}

uint64_t SaturatingAdd(uint64_t first, uint64_t second) {
  return first > std::numeric_limits<uint64_t>::max() - second ? std::numeric_limits<uint64_t>::max()
                                                               : first + second;
}

}  // namespace

PreparedProblem BuildProblem(const FunctionPtr& func, const AllocationPlan& allocation_plan,
                             const MemoryAllocatorPolicy& policy,
                             const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
                             const std::unordered_map<MemorySpace, uint64_t>& pool_caps,
                             const backend::Backend* backend) {
  INTERNAL_CHECK(func != nullptr) << "DSA-RP cannot analyze a null function";

  PreparedProblem prepared;
  for (const auto& [base, size] : allocation_plan.declared_allocation_sizes) {
    static_cast<void>(size);
    prepared.declared_allocation_bases.insert(base);
  }
  std::vector<std::optional<dsa::BufferId>> buffer_by_interval(allocation_plan.intervals.size());
  std::map<MemorySpace, dsa::Pool> pools;
  std::map<MemorySpace, uint64_t> fallback_capacity;

  for (size_t index = 0; index < allocation_plan.intervals.size(); ++index) {
    const LifetimeInterval& lifetime = allocation_plan.intervals[index];
    if (lifetime.memory_space == MemorySpace::DDR || !policy.ShouldAllocate(lifetime.memory_space)) {
      continue;
    }

    INTERNAL_CHECK(prepared.strict_problem.buffers.size() <=
                   static_cast<size_t>(std::numeric_limits<dsa::BufferId>::max()))
        << "Too many allocation identities for DSA-RP";
    const auto id = static_cast<dsa::BufferId>(prepared.strict_problem.buffers.size());
    const auto tile_type = As<TileType>(lifetime.variable->GetType());
    INTERNAL_CHECK_SPAN(tile_type != nullptr && tile_type->memref_.has_value(), lifetime.variable->span_)
        << "DSA-RP expected representative '" << lifetime.variable->name_hint_ << "' to carry a MemRef";
    const MemRefPtr memref = GetDefinedMemRef(tile_type);

    const uint64_t alignment = std::max<uint64_t>(1, policy.AlignAddress(1, lifetime.memory_space));
    const DsaExecutionLifetime execution_lifetime = ConvertToDsaExecutionLifetime(lifetime);
    prepared.strict_problem.buffers.push_back({id,
                                               lifetime.size,
                                               alignment,
                                               ToPoolId(lifetime.memory_space),
                                               {execution_lifetime.begin, execution_lifetime.end}});
    buffer_by_interval[index] = id;

    const auto inserted = prepared.buffer_id_by_base.emplace(memref->base_.get(), id);
    INTERNAL_CHECK_SPAN(inserted.second, lifetime.variable->span_)
        << "DSA-RP produced duplicate allocation identity for base '" << memref->base_->name_hint_ << "'";

    dsa::Pool& pool = pools[lifetime.memory_space];
    pool.id = ToPoolId(lifetime.memory_space);
    const auto reserved = reserved_end_by_space.find(lifetime.memory_space);
    if (reserved != reserved_end_by_space.end() && reserved->second > 0 && pool.reserved_ranges.empty()) {
      pool.reserved_ranges.push_back({0, reserved->second});
      fallback_capacity[lifetime.memory_space] = reserved->second;
    }
    const uint64_t aligned_size = policy.AlignAddress(lifetime.size, lifetime.memory_space);
    fallback_capacity[lifetime.memory_space] =
        SaturatingAdd(fallback_capacity[lifetime.memory_space], aligned_size);
  }

  for (auto& [space, pool] : pools) {
    const auto configured = pool_caps.find(space);
    pool.capacity = configured != pool_caps.end() && configured->second > 0 ? configured->second
                                                                            : fallback_capacity[space];
    prepared.strict_problem.pools.push_back(std::move(pool));
  }

  using BufferPair = std::pair<dsa::BufferId, dsa::BufferId>;
  std::map<BufferPair, std::set<AllocationSeparationReason>> hard_reasons;
  for (const AllocationSeparation& separation : allocation_plan.separations) {
    INTERNAL_CHECK(separation.first < buffer_by_interval.size() &&
                   separation.second < buffer_by_interval.size())
        << "DSA-RP separation references an out-of-range interval";
    if (!buffer_by_interval[separation.first] || !buffer_by_interval[separation.second]) {
      continue;
    }
    const BufferPair pair =
        CanonicalPair(*buffer_by_interval[separation.first], *buffer_by_interval[separation.second]);
    // Physical memory spaces are independent DSA problems. A relation between
    // two spaces cannot constrain reuse because those addresses never alias.
    if (prepared.strict_problem.buffers[pair.first].pool !=
        prepared.strict_problem.buffers[pair.second].pool) {
      continue;
    }
    auto& reasons = hard_reasons[pair];
    INTERNAL_CHECK(!separation.reasons.empty()) << "DSA-RP separation must carry at least one typed reason";
    reasons.insert(separation.reasons.begin(), separation.reasons.end());
  }
  for (const auto& [pair, reasons] : hard_reasons) {
    prepared.strict_problem.separations.push_back({pair.first, pair.second});
    const dsa::Buffer& first = prepared.strict_problem.buffers[pair.first];
    const dsa::Buffer& second = prepared.strict_problem.buffers[pair.second];
    // A pipeline relation between co-live buffers is redundant with ordinary
    // DSA interference. It can never become legal physical reuse, so do not
    // turn it into a soft edge during the bounded fallback.
    if (reasons.size() == 1 && reasons.count(AllocationSeparationReason::PipelineStage) != 0 &&
        !first.lifetime.Overlaps(second.lifetime)) {
      prepared.pipeline_pairs.push_back({pair.first, pair.second});
    }
  }

  std::map<BufferPair, uint64_t> penalty_weights;
  const std::vector<RecognizedReusePenalty> recognized =
      backend != nullptr ? RecognizeReusePenalties(func, allocation_plan, *backend)
                         : std::vector<RecognizedReusePenalty>{};
  for (const RecognizedReusePenalty& penalty : recognized) {
    INTERNAL_CHECK(penalty.first_interval < buffer_by_interval.size() &&
                   penalty.second_interval < buffer_by_interval.size())
        << "DSA-RP recognizer returned an out-of-range interval";
    if (!buffer_by_interval[penalty.first_interval] || !buffer_by_interval[penalty.second_interval]) {
      continue;
    }
    const BufferPair pair = CanonicalPair(*buffer_by_interval[penalty.first_interval],
                                          *buffer_by_interval[penalty.second_interval]);
    if (prepared.strict_problem.buffers[pair.first].pool !=
        prepared.strict_problem.buffers[pair.second].pool) {
      continue;
    }
    penalty_weights[pair] = SaturatingAdd(penalty_weights[pair], penalty.cost);
  }
  for (const auto& [pair, weight] : penalty_weights) {
    if (weight != 0) {
      prepared.strict_problem.reuse_penalties.push_back({pair.first, pair.second, weight});
    }
  }

  return prepared;
}

dsa::DsaProblem RelaxPipelineIntent(const PreparedProblem& prepared) {
  return dsa::RelaxSeparationsToPenalties(prepared.strict_problem, prepared.pipeline_pairs, 1);
}

std::vector<std::pair<const MemRef*, MemRefPtr>> BuildMemRefReplacements(
    const PreparedProblem& prepared, const dsa::DsaSolution& solution,
    const std::vector<MemRefWithSpace>& memrefs, const MemoryAllocatorPolicy& policy) {
  std::vector<std::pair<const MemRef*, MemRefPtr>> replacements;
  replacements.reserve(memrefs.size());
  for (const auto& [old_memref, memory_space] : memrefs) {
    if (memory_space == MemorySpace::DDR || !policy.ShouldAllocate(memory_space)) {
      continue;
    }
    const auto buffer = prepared.buffer_id_by_base.find(old_memref->base_.get());
    INTERNAL_CHECK_SPAN(buffer != prepared.buffer_id_by_base.end(), old_memref->span_)
        << "DSA-RP writeback could not find allocation base '" << old_memref->base_->name_hint_ << "'";
    const uint64_t* offset = solution.Find(buffer->second);
    INTERNAL_CHECK_SPAN(offset != nullptr, old_memref->span_)
        << "DSA-RP writeback has no placement for buffer " << buffer->second;

    ExprPtr address;
    if (const auto relative = As<ConstInt>(old_memref->byte_offset_)) {
      INTERNAL_CHECK_SPAN(relative->value_ >= 0, old_memref->span_)
          << "DSA-RP writeback encountered a negative relative MemRef offset";
      const uint64_t relative_value = static_cast<uint64_t>(relative->value_);
      INTERNAL_CHECK_SPAN(
          *offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - relative_value,
          old_memref->span_)
          << "DSA-RP address exceeds PyPTO's signed INT64 representation";
      address = std::make_shared<ConstInt>(static_cast<int64_t>(*offset + relative_value), DataType::INT64,
                                           Span::unknown());
    } else if (prepared.declared_allocation_bases.count(old_memref->base_.get()) != 0) {
      INTERNAL_CHECK_SPAN(*offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                          old_memref->span_)
          << "DSA-RP address exceeds PyPTO's signed INT64 representation";
      auto base = std::make_shared<ConstInt>(static_cast<int64_t>(*offset), DataType::INDEX, Span::unknown());
      address = std::make_shared<Add>(base, old_memref->byte_offset_, DataType::INDEX, Span::unknown());
    } else {
      // Ordinary dynamic view offsets are re-derived by their PTO subview op;
      // only declared runtime slots carry their expression into alloc_tile.
      address = std::make_shared<ConstInt>(static_cast<int64_t>(*offset), DataType::INT64, Span::unknown());
    }

    auto new_memref = std::make_shared<MemRef>(old_memref->name_hint_, old_memref->base_, std::move(address),
                                               old_memref->size_, old_memref->span_, old_memref->is_pinned_,
                                               old_memref->slot_count_, old_memref->slot_index_);
    replacements.emplace_back(old_memref.get(), std::move(new_memref));
  }

  return replacements;
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
