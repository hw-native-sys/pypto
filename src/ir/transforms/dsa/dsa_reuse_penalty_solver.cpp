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

#include "pypto/ir/transforms/dsa/dsa_reuse_penalty_solver.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {
namespace dsa {
namespace {

// The explicit DSA-RP input can contain Theta(B^2) lifetime conflicts and
// penalty relations for B buffers. BuildSearchSpace materializes that graph in
// O(B^2 + E) time; this is an output-sensitive exception used only by the
// opt-in DSA-RP planner, not an implicit nested scan over arbitrary IR nodes.

using NodePair = std::pair<size_t, size_t>;

struct SearchNode {
  BufferId id = 0;
  uint64_t size = 0;
  uint64_t alignment = 1;
  PoolId pool = 0;
  Interval lifetime;
};

struct SearchSpace {
  std::vector<SearchNode> nodes;
  std::unordered_map<BufferId, size_t> node_by_buffer;
  std::vector<std::set<size_t>> hard_neighbors;
  std::map<NodePair, uint64_t> soft_weights;
  std::vector<std::vector<std::pair<size_t, uint64_t>>> soft_neighbors;
};

struct WeightedBoundary {
  uint64_t position = 0;
  uint64_t weight = 0;
};

[[nodiscard]] bool AddOverflows(uint64_t first, uint64_t second) noexcept {
  return first > std::numeric_limits<uint64_t>::max() - second;
}

[[nodiscard]] uint64_t SaturatingAdd(uint64_t first, uint64_t second) noexcept {
  return AddOverflows(first, second) ? std::numeric_limits<uint64_t>::max() : first + second;
}

[[nodiscard]] std::optional<uint64_t> AlignUp(uint64_t value, uint64_t alignment) noexcept {
  if (alignment <= 1) return value;
  const uint64_t remainder = value % alignment;
  if (remainder == 0) return value;
  const uint64_t delta = alignment - remainder;
  if (AddOverflows(value, delta)) return std::nullopt;
  return value + delta;
}

[[nodiscard]] bool RangesOverlap(uint64_t first_offset, uint64_t first_size, uint64_t second_offset,
                                 uint64_t second_size) noexcept {
  if (first_size == 0 || second_size == 0 || AddOverflows(first_offset, first_size) ||
      AddOverflows(second_offset, second_size)) {
    return false;
  }
  return first_offset < second_offset + second_size && second_offset < first_offset + first_size;
}

[[nodiscard]] NodePair CanonicalPair(size_t first, size_t second) noexcept {
  return first < second ? NodePair{first, second} : NodePair{second, first};
}

[[nodiscard]] std::pair<BufferId, BufferId> CanonicalBufferPair(BufferId first, BufferId second) noexcept {
  return first < second ? std::pair{first, second} : std::pair{second, first};
}

[[nodiscard]] std::unordered_map<PoolId, const Pool*> PoolsById(const DsaProblem& problem) {
  std::unordered_map<PoolId, const Pool*> result;
  result.reserve(problem.pools.size());
  for (const Pool& pool : problem.pools) result.emplace(pool.id, &pool);
  return result;
}

[[nodiscard]] std::unordered_map<BufferId, const Buffer*> BuffersById(const DsaProblem& problem) {
  std::unordered_map<BufferId, const Buffer*> result;
  result.reserve(problem.buffers.size());
  for (const Buffer& buffer : problem.buffers) result.emplace(buffer.id, &buffer);
  return result;
}

[[nodiscard]] SearchSpace BuildSearchSpace(const DsaProblem& problem) {
  SearchSpace search;
  search.nodes.reserve(problem.buffers.size());
  search.node_by_buffer.reserve(problem.buffers.size());

  for (const Buffer& buffer : problem.buffers) {
    const size_t index = search.nodes.size();
    search.nodes.push_back({buffer.id, buffer.size, buffer.alignment, buffer.pool, buffer.lifetime});
    search.node_by_buffer.emplace(buffer.id, index);
  }

  const size_t count = search.nodes.size();
  search.hard_neighbors.resize(count);
  search.soft_neighbors.resize(count);

  for (size_t first = 0; first < count; ++first) {
    for (size_t second = first + 1; second < count; ++second) {
      if (search.nodes[first].pool == search.nodes[second].pool &&
          search.nodes[first].lifetime.Overlaps(search.nodes[second].lifetime)) {
        search.hard_neighbors[first].insert(second);
        search.hard_neighbors[second].insert(first);
      }
    }
  }

  for (const Separation& separation : problem.separations) {
    const size_t first = search.node_by_buffer.at(separation.first);
    const size_t second = search.node_by_buffer.at(separation.second);
    search.hard_neighbors[first].insert(second);
    search.hard_neighbors[second].insert(first);
  }

  for (const ReusePenalty& penalty : problem.reuse_penalties) {
    const size_t first = search.node_by_buffer.at(penalty.first);
    const size_t second = search.node_by_buffer.at(penalty.second);
    if (penalty.weight == 0 || search.hard_neighbors[first].count(second) != 0) continue;
    const NodePair pair = CanonicalPair(first, second);
    search.soft_weights[pair] = SaturatingAdd(search.soft_weights[pair], penalty.weight);
  }
  for (const auto& [pair, weight] : search.soft_weights) {
    search.soft_neighbors[pair.first].emplace_back(pair.second, weight);
    search.soft_neighbors[pair.second].emplace_back(pair.first, weight);
  }
  return search;
}

[[nodiscard]] std::vector<size_t> SizeOrder(const SearchSpace& search) {
  std::vector<size_t> order(search.nodes.size());
  for (size_t index = 0; index < order.size(); ++index) order[index] = index;
  std::sort(order.begin(), order.end(), [&](size_t first, size_t second) {
    const SearchNode& first_node = search.nodes[first];
    const SearchNode& second_node = search.nodes[second];
    return std::make_tuple(std::numeric_limits<uint64_t>::max() - first_node.size, first_node.lifetime.begin,
                           first_node.id) <
           std::make_tuple(std::numeric_limits<uint64_t>::max() - second_node.size,
                           second_node.lifetime.begin, second_node.id);
  });
  return order;
}

[[nodiscard]] std::vector<std::vector<size_t>> CanonicalOrders(const SearchSpace& search,
                                                               const CanonicalGreedyOptions& options) {
  std::vector<std::vector<size_t>> orders;
  std::vector<size_t> nodes = SizeOrder(search);
  orders.push_back(nodes);

  std::sort(nodes.begin(), nodes.end(), [&](size_t first, size_t second) {
    return std::tie(search.nodes[first].lifetime.begin, search.nodes[first].id) <
           std::tie(search.nodes[second].lifetime.begin, search.nodes[second].id);
  });
  orders.push_back(nodes);

  std::vector<uint64_t> incident_weight(nodes.size(), 0);
  for (const auto& [pair, weight] : search.soft_weights) {
    incident_weight[pair.first] = SaturatingAdd(incident_weight[pair.first], weight);
    incident_weight[pair.second] = SaturatingAdd(incident_weight[pair.second], weight);
  }
  std::sort(nodes.begin(), nodes.end(), [&](size_t first, size_t second) {
    return std::make_tuple(std::numeric_limits<uint64_t>::max() - incident_weight[first],
                           std::numeric_limits<uint64_t>::max() - search.nodes[first].size,
                           search.nodes[first].id) <
           std::make_tuple(std::numeric_limits<uint64_t>::max() - incident_weight[second],
                           std::numeric_limits<uint64_t>::max() - search.nodes[second].size,
                           search.nodes[second].id);
  });
  orders.push_back(nodes);

  std::mt19937_64 random(options.seed);
  for (size_t restart = 0; restart < options.random_restarts; ++restart) {
    std::shuffle(nodes.begin(), nodes.end(), random);
    orders.push_back(nodes);
  }

  std::vector<std::vector<size_t>> unique_orders;
  std::set<std::vector<size_t>> seen;
  for (std::vector<size_t>& order : orders) {
    if (seen.insert(order).second) {
      unique_orders.push_back(std::move(order));
    }
  }
  return unique_orders;
}

[[nodiscard]] std::vector<AddressRange> BlockingRanges(const DsaProblem& problem, const SearchSpace& search,
                                                       const std::vector<bool>& placed,
                                                       const std::vector<uint64_t>& offsets, size_t current) {
  std::vector<AddressRange> blocked;
  const SearchNode& node = search.nodes[current];
  for (const Pool& pool : problem.pools) {
    if (pool.id == node.pool) {
      blocked = pool.reserved_ranges;
      break;
    }
  }
  for (size_t other : search.hard_neighbors[current]) {
    if (!placed[other] || search.nodes[other].pool != node.pool) continue;
    if (AddOverflows(offsets[other], search.nodes[other].size)) continue;
    blocked.push_back({offsets[other], offsets[other] + search.nodes[other].size});
  }
  return blocked;
}

[[nodiscard]] std::optional<uint64_t> LowestFit(uint64_t size, uint64_t alignment,
                                                std::vector<AddressRange> blocked_ranges) {
  std::sort(blocked_ranges.begin(), blocked_ranges.end(),
            [](const AddressRange& first, const AddressRange& second) {
              return std::tie(first.begin, first.end) < std::tie(second.begin, second.end);
            });

  std::optional<uint64_t> candidate = AlignUp(0, alignment);
  if (!candidate) return std::nullopt;
  for (const AddressRange& blocked : blocked_ranges) {
    if (blocked.end <= *candidate) continue;
    if (AddOverflows(*candidate, size)) return std::nullopt;
    if (blocked.begin >= *candidate + size) break;
    candidate = AlignUp(blocked.end, alignment);
    if (!candidate) return std::nullopt;
  }
  if (AddOverflows(*candidate, size)) return std::nullopt;
  return candidate;
}

[[nodiscard]] std::vector<AddressRange> MergeBlockingRanges(std::vector<AddressRange> ranges) {
  std::sort(ranges.begin(), ranges.end(), [](const AddressRange& first, const AddressRange& second) {
    return std::tie(first.begin, first.end) < std::tie(second.begin, second.end);
  });
  std::vector<AddressRange> merged;
  merged.reserve(ranges.size());
  for (const AddressRange& range : ranges) {
    if (merged.empty() || merged.back().end < range.begin) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

[[nodiscard]] std::pair<std::vector<WeightedBoundary>, std::vector<WeightedBoundary>> BuildSoftBoundaries(
    const SearchSpace& search, const std::vector<bool>& placed, const std::vector<uint64_t>& offsets,
    size_t current) {
  std::vector<WeightedBoundary> starts;
  std::vector<WeightedBoundary> ends;
  starts.reserve(search.soft_neighbors[current].size());
  ends.reserve(search.soft_neighbors[current].size());
  for (const auto& [other, weight] : search.soft_neighbors[current]) {
    if (!placed[other]) continue;
    if (AddOverflows(offsets[other], search.nodes[other].size)) continue;
    starts.push_back({offsets[other], weight});
    ends.push_back({offsets[other] + search.nodes[other].size, weight});
  }
  const auto by_position = [](const WeightedBoundary& first, const WeightedBoundary& second) {
    return std::tie(first.position, first.weight) < std::tie(second.position, second.weight);
  };
  std::sort(starts.begin(), starts.end(), by_position);
  std::sort(ends.begin(), ends.end(), by_position);
  return {std::move(starts), std::move(ends)};
}

[[nodiscard]] std::set<uint64_t> CandidateOffsets(const DsaProblem& problem, const SearchSpace& search,
                                                  const std::vector<bool>& placed,
                                                  const std::vector<uint64_t>& offsets, size_t current) {
  const SearchNode& node = search.nodes[current];
  std::set<uint64_t> candidates{0};
  for (const Pool& pool : problem.pools) {
    if (pool.id != node.pool) continue;
    for (const AddressRange& reserved : pool.reserved_ranges) {
      candidates.insert(reserved.end);
    }
    break;
  }
  for (size_t other : search.hard_neighbors[current]) {
    if (placed[other] && search.nodes[other].pool == node.pool &&
        !AddOverflows(offsets[other], search.nodes[other].size)) {
      candidates.insert(offsets[other] + search.nodes[other].size);
    }
  }
  for (const auto& [other, weight] : search.soft_neighbors[current]) {
    static_cast<void>(weight);
    if (placed[other] && search.nodes[other].pool == node.pool &&
        !AddOverflows(offsets[other], search.nodes[other].size)) {
      candidates.insert(offsets[other] + search.nodes[other].size);
    }
  }

  std::set<uint64_t> aligned;
  for (uint64_t candidate : candidates) {
    const std::optional<uint64_t> offset = AlignUp(candidate, node.alignment);
    if (offset) aligned.insert(*offset);
  }
  return aligned;
}

[[nodiscard]] DsaSolution BuildSolution(const SearchSpace& search, const std::vector<uint64_t>& offsets) {
  DsaSolution solution;
  for (size_t index = 0; index < search.nodes.size(); ++index) {
    solution.offsets.emplace(search.nodes[index].id, offsets[index]);
  }
  return solution;
}

[[nodiscard]] bool IsBetterObjective(const ObjectiveValue& candidate,
                                     const ObjectiveValue& incumbent) noexcept {
  return std::tie(candidate.reuse_cost, candidate.total_peak, candidate.max_peak) <
         std::tie(incumbent.reuse_cost, incumbent.total_peak, incumbent.max_peak);
}

[[nodiscard]] std::optional<DsaSolution> FirstFitIncumbent(const DsaProblem& problem,
                                                           const SearchSpace& search) {
  const std::vector<size_t> order = SizeOrder(search);
  std::vector<bool> placed(search.nodes.size(), false);
  std::vector<uint64_t> offsets(search.nodes.size(), 0);
  const auto pools = PoolsById(problem);

  for (size_t current : order) {
    const SearchNode& node = search.nodes[current];
    const std::optional<uint64_t> offset =
        LowestFit(node.size, node.alignment, BlockingRanges(problem, search, placed, offsets, current));
    const Pool* pool = pools.at(node.pool);
    if (!offset || AddOverflows(*offset, node.size) || *offset + node.size > pool->capacity) {
      return std::nullopt;
    }
    offsets[current] = *offset;
    placed[current] = true;
  }
  return BuildSolution(search, offsets);
}

[[nodiscard]] std::optional<DsaSolution> PlaceCanonicalOrder(const DsaProblem& problem,
                                                             const SearchSpace& search,
                                                             const std::vector<size_t>& order,
                                                             uint64_t* evaluated) {
  std::vector<bool> placed(search.nodes.size(), false);
  std::vector<uint64_t> offsets(search.nodes.size(), 0);
  const auto pools = PoolsById(problem);

  for (size_t current : order) {
    const std::set<uint64_t> candidates = CandidateOffsets(problem, search, placed, offsets, current);
    const std::vector<AddressRange> blocked =
        MergeBlockingRanges(BlockingRanges(problem, search, placed, offsets, current));
    const auto [soft_starts, soft_ends] = BuildSoftBoundaries(search, placed, offsets, current);
    const SearchNode& node = search.nodes[current];
    const Pool* pool = pools.at(node.pool);

    size_t blocked_index = 0;
    size_t start_index = 0;
    size_t end_index = 0;
    uint64_t started_weight = 0;
    uint64_t ended_weight = 0;
    std::optional<std::tuple<uint64_t, uint64_t>> best;
    for (uint64_t offset : candidates) {
      ++(*evaluated);
      if (AddOverflows(offset, node.size) || offset + node.size > pool->capacity) continue;
      const uint64_t candidate_end = offset + node.size;

      while (blocked_index < blocked.size() && blocked[blocked_index].end <= offset) {
        ++blocked_index;
      }
      if (blocked_index < blocked.size() && blocked[blocked_index].begin < candidate_end) continue;

      while (start_index < soft_starts.size() && soft_starts[start_index].position < candidate_end) {
        started_weight += soft_starts[start_index].weight;
        ++start_index;
      }
      while (end_index < soft_ends.size() && soft_ends[end_index].position <= offset) {
        ended_weight += soft_ends[end_index].weight;
        ++end_index;
      }
      if (started_weight < ended_weight) return std::nullopt;
      const auto score = std::make_tuple(started_weight - ended_weight, offset);
      if (!best || score < *best) best = score;
    }
    if (!best) return std::nullopt;
    offsets[current] = std::get<1>(*best);
    placed[current] = true;
  }
  return BuildSolution(search, offsets);
}

}  // namespace

std::vector<std::string> ValidateProblem(const DsaProblem& problem) {
  std::vector<std::string> errors;
  std::unordered_map<PoolId, const Pool*> pools;
  pools.reserve(problem.pools.size());
  for (const Pool& pool : problem.pools) {
    if (!pools.emplace(pool.id, &pool).second) {
      errors.push_back("duplicate pool id " + std::to_string(pool.id));
    }
    if (pool.capacity == 0) {
      errors.push_back("pool " + std::to_string(pool.id) + " must have a positive fixed capacity");
    }
    for (const AddressRange& reserved : pool.reserved_ranges) {
      if (reserved.begin >= reserved.end) {
        errors.push_back("pool " + std::to_string(pool.id) + " has an empty or reversed reserved range");
      } else if (reserved.end > pool.capacity) {
        errors.push_back("pool " + std::to_string(pool.id) + " has a reserved range beyond capacity");
      }
    }
  }

  std::unordered_map<BufferId, const Buffer*> buffers;
  buffers.reserve(problem.buffers.size());
  for (const Buffer& buffer : problem.buffers) {
    if (!buffers.emplace(buffer.id, &buffer).second) {
      errors.push_back("duplicate buffer id " + std::to_string(buffer.id));
    }
    if (buffer.size == 0) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " has zero size");
    }
    if (buffer.alignment == 0) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " has zero alignment");
    }
    if (buffer.lifetime.begin < 0 || buffer.lifetime.begin >= buffer.lifetime.end) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " has an invalid half-open lifetime");
    }
    if (pools.count(buffer.pool) == 0) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " references unknown pool " +
                       std::to_string(buffer.pool));
    }
  }

  const auto validate_pair = [&](BufferId first, BufferId second, const std::string& kind) {
    const auto first_buffer = buffers.find(first);
    const auto second_buffer = buffers.find(second);
    if (first == second) {
      errors.push_back(kind + " has identical endpoints " + std::to_string(first));
    }
    if (first_buffer == buffers.end() || second_buffer == buffers.end()) {
      errors.push_back(kind + " references an unknown buffer");
    } else if (first_buffer->second->pool != second_buffer->second->pool) {
      errors.push_back(kind + " spans different fixed pools");
    }
  };
  for (const Separation& separation : problem.separations) {
    validate_pair(separation.first, separation.second, "separation");
  }
  uint64_t total_penalty_weight = 0;
  for (const ReusePenalty& penalty : problem.reuse_penalties) {
    validate_pair(penalty.first, penalty.second, "reuse penalty");
    const auto first = buffers.find(penalty.first);
    const auto second = buffers.find(penalty.second);
    if (first != buffers.end() && second != buffers.end() && first->second->pool == second->second->pool &&
        first->second->lifetime.Overlaps(second->second->lifetime)) {
      errors.emplace_back("reuse penalty endpoints must have lifetime-compatible buffers");
    }
    if (AddOverflows(total_penalty_weight, penalty.weight)) {
      errors.emplace_back("total reuse-penalty weight exceeds uint64");
    } else {
      total_penalty_weight += penalty.weight;
    }
  }
  return errors;
}

std::vector<std::string> ValidateSolution(const DsaProblem& problem, const DsaSolution& solution) {
  std::vector<std::string> errors = ValidateProblem(problem);
  if (!errors.empty()) return errors;

  const auto pools = PoolsById(problem);
  const auto buffers = BuffersById(problem);
  for (const auto& [id, offset] : solution.offsets) {
    static_cast<void>(offset);
    if (buffers.count(id) == 0) {
      errors.push_back("solution contains unknown buffer " + std::to_string(id));
    }
  }

  for (const Buffer& buffer : problem.buffers) {
    const uint64_t* offset = solution.Find(buffer.id);
    if (offset == nullptr) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " has no offset");
      continue;
    }
    if (*offset % buffer.alignment != 0) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " violates alignment");
    }
    if (AddOverflows(*offset, buffer.size)) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " address range overflows uint64");
      continue;
    }
    const Pool* pool = pools.at(buffer.pool);
    if (*offset + buffer.size > pool->capacity) {
      errors.push_back("buffer " + std::to_string(buffer.id) + " exceeds pool capacity");
    }
    for (const AddressRange& reserved : pool->reserved_ranges) {
      if (RangesOverlap(*offset, buffer.size, reserved.begin, reserved.end - reserved.begin)) {
        errors.push_back("buffer " + std::to_string(buffer.id) + " overlaps a reserved range");
      }
    }
  }

  std::set<std::pair<BufferId, BufferId>> separations;
  for (const Separation& separation : problem.separations) {
    separations.insert(CanonicalBufferPair(separation.first, separation.second));
  }
  for (size_t first = 0; first < problem.buffers.size(); ++first) {
    for (size_t second = first + 1; second < problem.buffers.size(); ++second) {
      const Buffer& first_buffer = problem.buffers[first];
      const Buffer& second_buffer = problem.buffers[second];
      if (first_buffer.pool != second_buffer.pool) continue;
      const bool conflict = first_buffer.lifetime.Overlaps(second_buffer.lifetime) ||
                            separations.count(CanonicalBufferPair(first_buffer.id, second_buffer.id)) != 0;
      if (!conflict) continue;
      const uint64_t* first_offset = solution.Find(first_buffer.id);
      const uint64_t* second_offset = solution.Find(second_buffer.id);
      if (first_offset != nullptr && second_offset != nullptr &&
          RangesOverlap(*first_offset, first_buffer.size, *second_offset, second_buffer.size)) {
        errors.push_back("hard-conflicting buffers " + std::to_string(first_buffer.id) + "," +
                         std::to_string(second_buffer.id) + " overlap in address");
      }
    }
  }
  return errors;
}

ObjectiveValue EvaluateObjective(const DsaProblem& problem, const DsaSolution& solution) {
  ObjectiveValue objective;
  for (const Pool& pool : problem.pools) {
    uint64_t& peak = objective.peak_by_pool[pool.id];
    for (const AddressRange& reserved : pool.reserved_ranges) {
      peak = std::max(peak, reserved.end);
    }
  }
  for (const Buffer& buffer : problem.buffers) {
    const uint64_t* offset = solution.Find(buffer.id);
    if (offset == nullptr || AddOverflows(*offset, buffer.size)) continue;
    uint64_t& peak = objective.peak_by_pool[buffer.pool];
    peak = std::max(peak, *offset + buffer.size);
  }
  for (const auto& [pool, peak] : objective.peak_by_pool) {
    static_cast<void>(pool);
    objective.total_peak = SaturatingAdd(objective.total_peak, peak);
    objective.max_peak = std::max(objective.max_peak, peak);
  }

  const auto buffers = BuffersById(problem);
  for (const ReusePenalty& penalty : problem.reuse_penalties) {
    const Buffer* first = buffers.count(penalty.first) ? buffers.at(penalty.first) : nullptr;
    const Buffer* second = buffers.count(penalty.second) ? buffers.at(penalty.second) : nullptr;
    const uint64_t* first_offset = solution.Find(penalty.first);
    const uint64_t* second_offset = solution.Find(penalty.second);
    if (first == nullptr || second == nullptr || first_offset == nullptr || second_offset == nullptr ||
        first->pool != second->pool || first->lifetime.Overlaps(second->lifetime)) {
      continue;
    }
    if (RangesOverlap(*first_offset, first->size, *second_offset, second->size)) {
      objective.reuse_cost = SaturatingAdd(objective.reuse_cost, penalty.weight);
    }
  }
  return objective;
}

DsaProblem RelaxSeparationsToPenalties(const DsaProblem& problem, const std::vector<Separation>& relaxable,
                                       uint64_t weight) {
  DsaProblem relaxed = problem;
  const auto buffers = BuffersById(problem);
  std::set<std::pair<BufferId, BufferId>> selected;
  for (const Separation& separation : relaxable) {
    selected.insert(CanonicalBufferPair(separation.first, separation.second));
  }
  relaxed.separations.erase(
      std::remove_if(relaxed.separations.begin(), relaxed.separations.end(),
                     [&selected](const Separation& separation) {
                       return selected.count(CanonicalBufferPair(separation.first, separation.second)) != 0;
                     }),
      relaxed.separations.end());

  std::map<std::pair<BufferId, BufferId>, uint64_t> weights;
  for (const ReusePenalty& penalty : relaxed.reuse_penalties) {
    const auto pair = CanonicalBufferPair(penalty.first, penalty.second);
    weights[pair] = SaturatingAdd(weights[pair], penalty.weight);
  }
  for (const auto& pair : selected) {
    const auto first = buffers.find(pair.first);
    const auto second = buffers.find(pair.second);
    // Lifetime-conflicting buffers remain hard through ordinary DSA
    // interference even after a redundant typed separation is removed. They
    // cannot activate a legal-reuse penalty.
    if (first == buffers.end() || second == buffers.end() ||
        first->second->lifetime.Overlaps(second->second->lifetime)) {
      continue;
    }
    weights[pair] = SaturatingAdd(weights[pair], weight);
  }
  relaxed.reuse_penalties.clear();
  for (const auto& [pair, combined_weight] : weights) {
    if (combined_weight != 0) {
      relaxed.reuse_penalties.push_back({pair.first, pair.second, combined_weight});
    }
  }
  return relaxed;
}

size_t CountOverlappingPairs(const DsaProblem& problem, const DsaSolution& solution,
                             const std::vector<Separation>& pairs) {
  const auto buffers = BuffersById(problem);
  size_t count = 0;
  for (const Separation& pair : pairs) {
    const auto first = buffers.find(pair.first);
    const auto second = buffers.find(pair.second);
    const uint64_t* first_offset = solution.Find(pair.first);
    const uint64_t* second_offset = solution.Find(pair.second);
    if (first == buffers.end() || second == buffers.end() || first_offset == nullptr ||
        second_offset == nullptr || first->second->pool != second->second->pool) {
      continue;
    }
    if (RangesOverlap(*first_offset, first->second->size, *second_offset, second->second->size)) {
      ++count;
    }
  }
  return count;
}

CanonicalGreedySolver::CanonicalGreedySolver(CanonicalGreedyOptions options) : options_(options) {}

DsaResult CanonicalGreedySolver::Solve(const DsaProblem& problem) const {
  DsaResult result;
  result.diagnostics = ValidateProblem(problem);
  if (!result.diagnostics.empty()) {
    result.status = SolveStatus::kInvalidProblem;
    return result;
  }

  const SearchSpace search = BuildSearchSpace(problem);
  const std::optional<DsaSolution> first_fit = FirstFitIncumbent(problem, search);
  if (first_fit) {
    const std::vector<std::string> errors = ValidateSolution(problem, *first_fit);
    if (!errors.empty()) {
      result.status = SolveStatus::kInvalidProblem;
      result.diagnostics.emplace_back("internal first-fit incumbent failed independent validation");
      result.diagnostics.insert(result.diagnostics.end(), errors.begin(), errors.end());
      return result;
    }
    result.status = SolveStatus::kFeasible;
    result.solution = *first_fit;
    result.objective = EvaluateObjective(problem, *first_fit);
    result.statistics.first_fit_seed_feasible = true;
    result.statistics.selected_first_fit_seed = true;
  }

  const std::vector<std::vector<size_t>> orders = CanonicalOrders(search, options_);
  for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
    ++result.statistics.orders_evaluated;
    const std::optional<DsaSolution> placement = PlaceCanonicalOrder(
        problem, search, orders[order_index], &result.statistics.candidate_offsets_evaluated);
    if (!placement) continue;

    const std::vector<std::string> errors = ValidateSolution(problem, *placement);
    if (!errors.empty()) {
      result.status = SolveStatus::kInvalidProblem;
      result.solution.reset();
      result.diagnostics.emplace_back(
          "canonical greedy produced a placement that failed independent validation");
      result.diagnostics.insert(result.diagnostics.end(), errors.begin(), errors.end());
      return result;
    }
    const ObjectiveValue objective = EvaluateObjective(problem, *placement);
    const bool replace = !result.solution || IsBetterObjective(objective, result.objective);
    if (replace) {
      result.status = SolveStatus::kFeasible;
      result.solution = *placement;
      result.objective = objective;
      result.statistics.selected_order = order_index;
      result.statistics.selected_first_fit_seed = false;
    }
  }

  if (!result.solution) {
    result.status = SolveStatus::kNoFit;
    result.diagnostics.emplace_back("canonical greedy found no capacity-fitting placement");
  }
  return result;
}

}  // namespace dsa
}  // namespace ir
}  // namespace pypto
