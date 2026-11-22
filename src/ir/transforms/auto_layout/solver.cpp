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

#include "pypto/ir/transforms/auto_layout/solver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/auto_layout/model.h"

namespace pypto {
namespace ir {
namespace auto_layout {

namespace {

double RequireFiniteNonNegative(double value, const std::string& context) {
  if (!std::isfinite(value) || value < 0.0) {
    throw pypto::ValueError(context + " must be finite and nonnegative");
  }
  return value;
}

bool StartsWithTile(const std::string& op_name) { return op_name.rfind("tile.", 0) == 0; }

bool SupportsVecConversionPlacement(const std::string& name) {
  return StartsWithTile(name) || name == "control.if_join" || name == kVecLoopInitOp ||
         name == kVecLoopYieldOp;
}

bool HasIncompatibleJoinLayouts(const OperatorConfig& config) {
  return config.op_name == "control.if_join" &&
         std::any_of(config.input_layouts.begin(), config.input_layouts.end(),
                     [&](const LayoutSpec& input) { return input != config.output_layout; });
}

constexpr LayoutSearchScore kWorstCost{std::numeric_limits<int64_t>::max(),
                                       std::numeric_limits<double>::infinity()};
constexpr LayoutSearchScore kWorstGain{std::numeric_limits<int64_t>::min(),
                                       -std::numeric_limits<double>::infinity()};

bool BetterCost(LayoutSearchScore lhs, LayoutSearchScore rhs, double tolerance) {
  if (lhs.violations != rhs.violations) return lhs.violations < rhs.violations;
  return lhs.cycles < rhs.cycles - tolerance;
}

bool BetterGain(LayoutSearchScore lhs, LayoutSearchScore rhs, double tolerance = 0.0) {
  if (lhs.violations != rhs.violations) return lhs.violations > rhs.violations;
  return lhs.cycles > rhs.cycles + tolerance;
}

bool HasMultipleRepresentations(uint64_t mask) { return (mask & (mask - 1)) != 0; }

int64_t FactorViolations(const IndexedProblem::ValueInfo& value, uint16_t source, uint64_t mask) {
  const int64_t multiple = value.forbid_multi_rep && HasMultipleRepresentations(mask);
  if (value.unsupported_target_masks.empty()) return multiple;
  return multiple + __builtin_popcountll(mask & value.unsupported_target_masks[source]);
}

/// Compact bitset with an inline 64-bit fast path for graphs with <= 64 vertices.
struct SmallBitset {
  uint64_t w0 = 0;
  std::vector<uint64_t> extra;
  size_t num_words = 1;

  explicit SmallBitset(size_t num_bits = 64) {
    num_words = std::max<size_t>(1, (num_bits + 63) / 64);
    if (num_words > 1) {
      extra.assign(num_words - 1, 0);
    }
  }

  void Set(size_t bit) {
    const size_t word = bit >> 6;
    const uint64_t mask = uint64_t{1} << (bit & 63);
    if (word == 0) {
      w0 |= mask;
    } else {
      extra[word - 1] |= mask;
    }
  }

  void Clear(size_t bit) {
    const size_t word = bit >> 6;
    const uint64_t mask = ~(uint64_t{1} << (bit & 63));
    if (word == 0) {
      w0 &= mask;
    } else {
      extra[word - 1] &= mask;
    }
  }

  [[nodiscard]] bool Test(size_t bit) const {
    const size_t word = bit >> 6;
    const uint64_t mask = uint64_t{1} << (bit & 63);
    return word == 0 ? ((w0 & mask) != 0) : ((extra[word - 1] & mask) != 0);
  }

  [[nodiscard]] int Popcount() const {
    int cnt = __builtin_popcountll(w0);
    for (uint64_t w : extra) {
      cnt += __builtin_popcountll(w);
    }
    return cnt;
  }

  void OrWith(const SmallBitset& other) {
    w0 |= other.w0;
    for (size_t i = 0; i < extra.size(); ++i) {
      extra[i] |= other.extra[i];
    }
  }

  [[nodiscard]] int PopcountAndNot(const SmallBitset& other, size_t skip_bit) const {
    const size_t skip_word = skip_bit >> 6;
    const uint64_t skip_mask = ~(uint64_t{1} << (skip_bit & 63));
    uint64_t w = w0 & ~other.w0;
    if (skip_word == 0) {
      w &= skip_mask;
    }
    int cnt = __builtin_popcountll(w);
    for (size_t i = 0; i < extra.size(); ++i) {
      uint64_t wi = extra[i] & ~other.extra[i];
      if (skip_word == i + 1) {
        wi &= skip_mask;
      }
      cnt += __builtin_popcountll(wi);
    }
    return cnt;
  }

  template <typename Fn>
  void ForEachBit(Fn&& fn) const {
    uint64_t cur = w0;
    while (cur != 0) {
      const int bit = __builtin_ctzll(cur);
      fn(static_cast<size_t>(bit));
      cur &= cur - 1;
    }
    for (size_t i = 0; i < extra.size(); ++i) {
      uint64_t wi = extra[i];
      const size_t base = (i + 1) << 6;
      while (wi != 0) {
        const int bit = __builtin_ctzll(wi);
        fn(base + static_cast<size_t>(bit));
        wi &= wi - 1;
      }
    }
  }
};

struct MinFillResult {
  std::vector<OpId> elim_order;
  std::vector<std::vector<OpId>> bags;
  std::vector<std::optional<size_t>> parents;
  int64_t treewidth = 0;
  int64_t max_bag_states = 0;
  int64_t total_states = 0;
};

struct MinFillKey {
  int fill2 = 0;
  int degree = 0;
  OpId original_id = 0;

  bool operator<(const MinFillKey& other) const {
    return std::tie(fill2, degree, original_id) < std::tie(other.fill2, other.degree, other.original_id);
  }
};

MinFillResult MinFillDecomposition(const std::vector<OpId>& vertices,
                                   const std::vector<std::pair<OpId, OpId>>& edges) {
  MinFillResult result;
  if (vertices.empty()) {
    return result;
  }

  std::vector<OpId> sorted_verts = vertices;
  std::sort(sorted_verts.begin(), sorted_verts.end());
  sorted_verts.erase(std::unique(sorted_verts.begin(), sorted_verts.end()), sorted_verts.end());

  const size_t n = sorted_verts.size();
  std::unordered_map<OpId, size_t> local_of;
  local_of.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    local_of.emplace(sorted_verts[i], i);
  }

  std::vector<SmallBitset> adj(n, SmallBitset(n));
  for (const auto& [u_id, v_id] : edges) {
    if (u_id == v_id) {
      continue;
    }
    auto it_u = local_of.find(u_id);
    auto it_v = local_of.find(v_id);
    if (it_u != local_of.end() && it_v != local_of.end()) {
      adj[it_u->second].Set(it_v->second);
      adj[it_v->second].Set(it_u->second);
    }
  }

  auto compute_key = [&](size_t v) -> MinFillKey {
    const SmallBitset& nb = adj[v];
    const int deg = nb.Popcount();
    int fill2 = 0;
    nb.ForEachBit([&](size_t u1) { fill2 += nb.PopcountAndNot(adj[u1], u1); });
    return MinFillKey{fill2 >> 1, deg, sorted_verts[v]};
  };

  std::vector<bool> alive(n, true);
  std::vector<MinFillKey> keys(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = compute_key(i);
  }

  result.elim_order.reserve(n);
  result.bags.reserve(n);
  std::vector<size_t> local_elim_order;
  local_elim_order.reserve(n);
  std::vector<size_t> step_of(n, 0);

  for (size_t step = 0; step < n; ++step) {
    size_t best_local = n;
    for (size_t i = 0; i < n; ++i) {
      if (!alive[i]) {
        continue;
      }
      if (best_local == n || keys[i] < keys[best_local]) {
        best_local = i;
      }
    }

    const SmallBitset nb = adj[best_local];
    std::vector<OpId> bag;
    bag.reserve(static_cast<size_t>(nb.Popcount()) + 1);
    bag.push_back(sorted_verts[best_local]);

    SmallBitset dirty(n);
    dirty.OrWith(nb);
    nb.ForEachBit([&](size_t u) {
      bag.push_back(sorted_verts[u]);
      adj[u].OrWith(nb);
      adj[u].Clear(u);
      adj[u].Clear(best_local);
      dirty.OrWith(adj[u]);
    });

    alive[best_local] = false;
    dirty.Clear(best_local);
    dirty.ForEachBit([&](size_t w) {
      if (alive[w]) {
        keys[w] = compute_key(w);
      }
    });

    step_of[best_local] = step;
    local_elim_order.push_back(best_local);
    result.elim_order.push_back(sorted_verts[best_local]);
    result.bags.push_back(std::move(bag));
  }

  result.parents.reserve(n);
  int64_t max_bag_len = 0;
  for (const auto& bag : result.bags) {
    max_bag_len = std::max(max_bag_len, static_cast<int64_t>(bag.size()));
    if (bag.size() <= 1) {
      result.parents.emplace_back(std::nullopt);
    } else {
      size_t min_step = n;
      for (size_t i = 1; i < bag.size(); ++i) {
        const size_t local_u = local_of[bag[i]];
        min_step = std::min(min_step, step_of[local_u]);
      }
      result.parents.emplace_back(min_step);
    }
  }
  result.treewidth = std::max<int64_t>(0, max_bag_len - 1);
  return result;
}

std::vector<OpId> TopologicalSeed(const IndexedProblem& problem, const LayoutSolverConfig& config,
                                  int64_t& evaluated_moves, std::string& termination_reason) {
  const size_t num_ops = problem.NumOps();
  const size_t num_vals = problem.NumValues();
  std::vector<uint16_t> choices(num_ops, 0);
  std::vector<uint16_t> produced(num_vals);
  std::vector<uint64_t> requested(num_vals, 0);
  for (ValueId v = 0; v < num_vals; ++v) {
    produced[v] = problem.Values()[v].default_source_idx;
    requested[v] = problem.Values()[v].boundary_demand_mask;
  }

  // Pinned readers constrain a GM input even when they occur later in
  // topological order. Seed those demands before scoring movable readers.
  if (problem.Instance().forbid_unmaterializable_gm_conversions) {
    for (const auto& op : problem.Ops()) {
      if (op.costs.size() != 1) continue;
      for (const auto& use : op.config_uses.front()) {
        if (problem.Values()[use.value_id].forbid_multi_rep) requested[use.value_id] |= use.demand_mask;
      }
    }
  }

  for (OpId op_idx : problem.TopoOrder()) {
    const auto& op = problem.Ops()[op_idx];
    uint16_t best_choice = 0;
    LayoutSearchScore best_cost = kWorstCost;
    const auto& out_info = problem.Values()[op.output_value];
    const size_t out_num_layouts = out_info.layouts.size();
    for (uint16_t q = 0; q < static_cast<uint16_t>(op.costs.size()); ++q) {
      if (evaluated_moves + 1 > config.max_evaluations) {
        termination_reason = "evaluation_budget";
        choices[op_idx] = best_choice;
        return std::vector<OpId>(choices.begin(), choices.end());
      }
      ++evaluated_moves;
      LayoutSearchScore cost{op.incompatible_configs[q], op.costs[q]};
      const uint16_t out_lay = op.output_layout_indices[q];
      uint64_t out_mask = requested[op.output_value];
      cost.violations += FactorViolations(out_info, out_lay, out_mask);
      while (out_mask != 0) {
        const int dst_idx = __builtin_ctzll(out_mask);
        if (static_cast<uint16_t>(dst_idx) != out_lay) {
          cost.cycles +=
              out_info
                  .conv_costs[static_cast<size_t>(out_lay) * out_num_layouts + static_cast<size_t>(dst_idx)];
        }
        out_mask &= out_mask - 1;
      }
      for (const auto& use : op.config_uses[q]) {
        const ValueId v = use.value_id;
        const auto& v_info = problem.Values()[v];
        const size_t num_layouts = v_info.layouts.size();
        const uint16_t src_idx = (v == op.output_value) ? out_lay : produced[v];
        const auto old_mask = requested[v];
        const auto new_mask = old_mask | use.demand_mask;
        cost.violations +=
            FactorViolations(v_info, src_idx, new_mask) - FactorViolations(v_info, src_idx, old_mask);
        if (v_info.forbid_multi_rep && !HasMultipleRepresentations(old_mask) &&
            HasMultipleRepresentations(new_mask)) {
          cost.cycles += problem.Instance().cost_model.vec_unsupported_conversion_cycles;
        }
        uint64_t new_bits = use.demand_mask & ~requested[v];
        while (new_bits != 0) {
          const int dst_idx = __builtin_ctzll(new_bits);
          cost.cycles +=
              v_info.conv_costs[static_cast<size_t>(src_idx) * num_layouts + static_cast<size_t>(dst_idx)];
          new_bits &= new_bits - 1;
        }
      }
      if (BetterCost(cost, best_cost, config.tolerance)) {
        best_cost = cost;
        best_choice = q;
      }
    }
    choices[op_idx] = best_choice;
    produced[op.output_value] = op.output_layout_indices[best_choice];
    for (const auto& use : op.config_uses[best_choice]) {
      requested[use.value_id] |= use.demand_mask;
    }
  }
  return std::vector<OpId>(choices.begin(), choices.end());
}

std::pair<std::vector<LayoutConversion>, std::vector<std::string>> BuildConversionReport(
    const IncrementalState& state) {
  std::vector<LayoutConversion> conversions;
  std::vector<std::string> issues;
  const auto& problem = state.Problem();
  const auto& instance = problem.Instance();
  const size_t num_vals = problem.NumValues();

  for (ValueId v = 0; v < num_vals; ++v) {
    const auto& v_info = problem.Values()[v];
    const auto* node = instance.FindValue(v_info.name);
    if (node == nullptr) {
      continue;
    }
    const uint16_t src_idx = state.Sources()[v];
    const LayoutSpec& src_spec = v_info.layouts[src_idx];
    const uint64_t mask = state.DemandMasks()[v];
    if (v_info.producer_op.has_value()) {
      const auto producer = *v_info.producer_op;
      const auto& op = instance.operations[producer];
      if (HasIncompatibleJoinLayouts(op.configs[state.Choices()[producer]])) {
        issues.push_back("Conditional join '" + op.op_id + "' has incompatible branch layouts");
      }
    }
    const bool external_input = !node->producer_op_id.has_value() && node->memory_space == MemorySpace::DDR;
    if (external_input && __builtin_popcountll(mask) > 1) {
      issues.push_back("GM input '" + v_info.name + "' needs multiple representations");
    }
    const size_t num_layouts = v_info.layouts.size();
    uint64_t cur_mask = mask;
    while (cur_mask != 0) {
      const uint16_t dst_idx = static_cast<uint16_t>(__builtin_ctzll(cur_mask));
      cur_mask &= cur_mask - 1;
      if (dst_idx == src_idx) {
        continue;
      }
      const LayoutSpec& dst_spec = v_info.layouts[dst_idx];
      std::vector<OpId> consumer_indices;
      std::vector<std::string> consumer_op_ids;
      for (OpId p : v_info.participants) {
        const uint16_t q = state.Choices()[p];
        for (const auto& use : problem.Ops()[p].config_uses[q]) {
          if (use.value_id == v && (use.demand_mask & (uint64_t{1} << dst_idx)) != 0) {
            consumer_indices.push_back(p);
            consumer_op_ids.push_back(problem.Ops()[p].op_id);
            break;
          }
        }
      }
      const bool boundary_output =
          consumer_indices.empty() && node->is_external_output && dst_spec == node->current_layout;
      const bool boundary = external_input || boundary_output;
      const double conv_cost = v_info.conv_costs[static_cast<size_t>(src_idx) * num_layouts + dst_idx];
      conversions.push_back(
          LayoutConversion{v_info.name, src_spec, dst_spec, conv_cost, consumer_op_ids, boundary});
      if (!boundary) {
        const std::string& prod_op_name =
            v_info.producer_op.has_value() ? problem.Ops()[*v_info.producer_op].op_name : std::string();
        const bool producer_ok = !v_info.producer_op || SupportsVecConversionPlacement(prod_op_name);
        const bool consumers_ok =
            std::all_of(consumer_indices.begin(), consumer_indices.end(),
                        [&](OpId idx) { return SupportsVecConversionPlacement(problem.Ops()[idx].op_name); });
        const bool compat_invs = (v_info.incompatible_target_mask & (uint64_t{1} << dst_idx)) == 0;
        const bool realizable_vec =
            node->memory_space == MemorySpace::Vec &&
            instance.cost_model.SupportsVecTileConversion(*node, src_spec, dst_spec) && producer_ok &&
            consumers_ok && compat_invs;
        if (!realizable_vec) {
          issues.push_back("Value '" + v_info.name + "' needs an internal conversion " +
                           src_spec.ShortName() + " -> " + dst_spec.ShortName());
        }
      }
    }
  }
  return {std::move(conversions), std::move(issues)};
}

LayoutSolverResult BuildResult(const IncrementalState& state, int64_t evaluated_moves, int64_t moves_applied,
                               int64_t greedy_sweeps, double initial_cost_cycles,
                               const std::string& termination_reason,
                               const std::optional<TreewidthStats>& tw_stats = std::nullopt,
                               int64_t unreported_width = 0) {
  const auto& problem = state.Problem();
  CostBreakdown breakdown = EvaluateAssignment(problem.Instance(), state.ToAssignment());
  const double diff = std::abs(state.Cost() - breakdown.total_cost_cycles);
  if (diff > 1e-7 && diff > 1e-10 * std::max(std::abs(state.Cost()), std::abs(breakdown.total_cost_cycles))) {
    throw pypto::InternalError("Layout gain cache disagrees with evaluator");
  }
  auto [conversions, issues] = BuildConversionReport(state);
  LayoutSolverStats stats;
  stats.evaluated_moves = evaluated_moves;
  stats.moves_applied = moves_applied;
  stats.greedy_sweeps = greedy_sweeps;
  stats.initial_cost_cycles = initial_cost_cycles;
  stats.termination_reason = termination_reason;
  stats.treewidth = tw_stats.has_value() ? tw_stats->primal_treewidth : unreported_width;
  stats.active_treewidth = tw_stats.has_value() ? tw_stats->active_primal_treewidth : unreported_width;
  return LayoutSolverResult{std::move(breakdown), std::move(stats), std::move(conversions), std::move(issues),
                            tw_stats};
}

// Range maximum and range addition keep each selected copy's capacity check
// logarithmic in the number of events, even for long-lived shared copies.
class VecPressure {
 public:
  explicit VecPressure(const std::vector<int64_t>& original)
      : size_(original.size()), maximum_(4 * size_), added_(4 * size_) {
    Build(original, 1, 0, size_);
  }

  [[nodiscard]] int64_t Peak(size_t first, size_t last) const { return Peak(1, 0, size_, first, last + 1); }

  void Add(size_t first, size_t last, int64_t bytes) { Add(1, 0, size_, first, last + 1, bytes); }

 private:
  void Build(const std::vector<int64_t>& original, size_t node, size_t begin, size_t end) {
    if (end - begin == 1) {
      maximum_[node] = original[begin];
      return;
    }
    const size_t mid = begin + (end - begin) / 2;
    Build(original, 2 * node, begin, mid);
    Build(original, 2 * node + 1, mid, end);
    maximum_[node] = std::max(maximum_[2 * node], maximum_[2 * node + 1]);
  }

  [[nodiscard]] int64_t Peak(size_t node, size_t begin, size_t end, size_t first, size_t last) const {
    if (first <= begin && end <= last) return maximum_[node];
    const size_t mid = begin + (end - begin) / 2;
    int64_t peak = 0;
    if (first < mid) peak = Peak(2 * node, begin, mid, first, last);
    if (mid < last) peak = std::max(peak, Peak(2 * node + 1, mid, end, first, last));
    return peak + added_[node];
  }

  void Add(size_t node, size_t begin, size_t end, size_t first, size_t last, int64_t bytes) {
    if (first <= begin && end <= last) {
      maximum_[node] += bytes;
      added_[node] += bytes;
      return;
    }
    const size_t mid = begin + (end - begin) / 2;
    if (first < mid) Add(2 * node, begin, mid, first, last, bytes);
    if (mid < last) Add(2 * node + 1, mid, end, first, last, bytes);
    maximum_[node] = std::max(maximum_[2 * node], maximum_[2 * node + 1]) + added_[node];
  }

  size_t size_;
  std::vector<int64_t> maximum_;
  std::vector<int64_t> added_;
};

struct VecCopy {
  const ValueNode* value;
  LayoutSpec source;
  LayoutSpec target;
  VecConversionStorage storage;
  size_t first;
  size_t last;
};

struct VecCapacityConflicts {
  std::vector<std::string> issues;
  std::vector<VecCopy> copies;
  std::vector<std::pair<size_t, OperatorConfig>> operation_configs;
};

bool FitsVecCapacity(int64_t occupied, uint64_t capacity, uint64_t extra) {
  const auto used = static_cast<uint64_t>(occupied);
  return used <= capacity && extra <= capacity - used;
}

VecCapacityConflicts FindVecCapacityConflicts(const LayoutProblemInstance& instance,
                                              const LayoutAssignment& assignment) {
  VecCapacityConflicts conflicts;
  if (instance.vec_memory_regions.empty()) return conflicts;
  std::unordered_map<std::string, LayoutSpec> produced;
  for (const auto& op : instance.operations) {
    const auto config = assignment.op_configs.find(op.op_id);
    if (config != assignment.op_configs.end()) produced.emplace(op.output, config->second.output_layout);
  }
  for (const auto& region : instance.vec_memory_regions) {
    if (region.operation_positions.empty()) continue;
    if (region.first_operation > instance.operations.size() ||
        region.operation_positions.size() > instance.operations.size() - region.first_operation) {
      throw pypto::ValueError("Vec pressure region has operations outside the layout instance");
    }
    std::map<std::pair<std::string, LayoutSpec>, size_t> copy_by_demand;
    std::vector<VecCopy> copies;
    auto original = region.original_bytes;
    for (size_t i = 0; i < region.operation_positions.size(); ++i) {
      const auto& op = instance.operations[region.first_operation + i];
      auto selected = assignment.op_configs.find(op.op_id);
      if (selected == assignment.op_configs.end()) continue;
      const auto& cfg = selected->second;
      const size_t position = region.operation_positions[i];
      if (position >= original.size()) {
        throw pypto::ValueError("Vec pressure operation is outside its timeline");
      }
      // Private copies live through this read only. Their destinations can
      // overlap, and each conversion may release its scratch before the next.
      uint64_t local_destinations = 0;
      uint64_t local_scratch = 0;
      if (cfg.local_vec_load_layout) {
        const auto& value = instance.values.at(op.output);
        const auto storage =
            instance.cost_model.GetVecConversionStorage(value, *cfg.local_vec_load_layout, cfg.output_layout);
        if (!storage) throw pypto::InternalError("Mutable Vec load has no modeled conversion storage");
        // The region already accounts for the backing defined by the load.
        // The native load's temporary has the same size as that backing.
        local_destinations += storage->destination_bytes;
        local_scratch = storage->scratch_bytes;
      }
      for (const auto& [operand, target] : cfg.local_vec_conversions) {
        const auto& value = instance.values.at(op.inputs.at(operand));
        const auto storage =
            instance.cost_model.GetVecConversionStorage(value, cfg.input_layouts.at(operand), target);
        if (!storage) throw pypto::InternalError("Mutable Vec read has no modeled conversion storage");
        local_destinations += storage->destination_bytes;
        local_scratch = std::max(local_scratch, storage->scratch_bytes);
      }
      for (size_t j = 0; j < op.inputs.size() && j < cfg.input_layouts.size(); ++j) {
        const auto& name = op.inputs[j];
        const auto* value = instance.FindValue(name);
        if (!value || value->memory_space != MemorySpace::Vec) continue;
        const LayoutSpec target = cfg.input_layouts[j];
        LayoutSpec src = value->current_layout;
        if (const auto it = produced.find(name); it != produced.end()) {
          src = it->second;
        } else if (const auto assigned = assignment.value_layouts.find(name);
                   assigned != assignment.value_layouts.end()) {
          src = assigned->second;
        }
        const auto storage = instance.cost_model.GetVecConversionStorage(*value, src, target);
        if (storage) {
          const auto backing = region.backings.find(op.output);
          const bool aliases_copy =
              backing != region.backings.end() &&
              std::find(backing->second.begin(), backing->second.end(), name) != backing->second.end();
          const size_t first = region.copy_positions.at(name);
          const size_t last =
              region.EffectiveLastUse(first, aliases_copy ? region.last_uses.at(op.output) : position);
          const auto [it, inserted] = copy_by_demand.try_emplace(std::make_pair(name, target), copies.size());
          if (inserted) {
            copies.push_back(VecCopy{value, src, target, *storage, first, last});
          } else {
            copies[it->second].last = std::max(copies[it->second].last, last);
          }
        }
        // An explicit Vec move can also lower to a physical transpose. Its
        // result is already in the baseline, so only its scratch is additional.
        if (j == 0 && op.op_name == "tile.move") {
          const LayoutSpec execution_layout = cfg.ExecutionInputLayout(0);
          const auto* output = instance.FindValue(op.output);
          const auto workspace =
              instance.cost_model.GetVecConversionStorage(*value, execution_layout, cfg.output_layout);
          if (output && output->memory_space == MemorySpace::Vec && workspace &&
              IsFlatVecTransposePair(execution_layout, cfg.output_layout)) {
            local_scratch = std::max(local_scratch, workspace->scratch_bytes);
          }
        }
      }
      const uint64_t operation_bytes = local_destinations + local_scratch;
      if (operation_bytes != 0) {
        if (!FitsVecCapacity(original[position], region.capacity, operation_bytes)) {
          conflicts.issues.push_back("Vec capacity exceeded by local layout conversion at '" + op.op_id +
                                     "'");
          conflicts.operation_configs.emplace_back(region.first_operation + i, cfg);
        }
        original[position] += static_cast<int64_t>(operation_bytes);
      }
    }
    if (copies.empty()) continue;
    std::stable_sort(copies.begin(), copies.end(),
                     [](const VecCopy& a, const VecCopy& b) { return a.first < b.first; });
    VecPressure pressure(original);
    size_t insertion = copies.front().first;
    uint64_t insertion_scratch = 0;
    for (const auto& copy : copies) {
      if (copy.first > copy.last || copy.last >= original.size()) {
        throw pypto::ValueError("Vec conversion of '" + copy.value->name + "' has an invalid lifetime");
      }
      const auto [destination, scratch] = copy.storage;
      if (copy.first != insertion) {
        insertion = copy.first;
        insertion_scratch = 0;
      }
      const uint64_t scratch_peak = std::max(insertion_scratch, scratch);
      const bool fits_lifetime =
          FitsVecCapacity(pressure.Peak(copy.first, copy.last), region.capacity, destination);
      const bool fits_scratch =
          destination <= region.capacity &&
          FitsVecCapacity(pressure.Peak(copy.first, copy.first), region.capacity - destination, scratch_peak);
      if (fits_lifetime && fits_scratch) {
        pressure.Add(copy.first, copy.last, static_cast<int64_t>(destination));
        // Several prologue conversions can share an insertion point. Reserve
        // space for the largest scratch allocation regardless of their order.
        insertion_scratch = scratch_peak;
      } else {
        conflicts.issues.push_back("Vec capacity exceeded by conversion of '" + copy.value->name + "' " +
                                   copy.source.ShortName() + " -> " + copy.target.ShortName());
        conflicts.copies.push_back(copy);
      }
    }
  }
  return conflicts;
}

struct ActiveDecomposition {
  std::vector<MinFillResult> component_decomps;
  std::vector<std::vector<OpId>> components;
  std::vector<std::vector<ValueId>> component_factors;
};

int64_t SaturatingAdd(int64_t a, int64_t b) {
  const auto limit = std::numeric_limits<int64_t>::max();
  return a > limit - b ? limit : a + b;
}

int64_t BagStates(const IndexedProblem& problem, const std::vector<OpId>& bag) {
  int64_t states = 1;
  for (OpId op : bag) {
    const auto domain = static_cast<int64_t>(problem.Ops()[op].costs.size());
    if (states > std::numeric_limits<int64_t>::max() / domain) return std::numeric_limits<int64_t>::max();
    states *= domain;
  }
  return states;
}

TreewidthStats ComputeProblemTreewidthInternal(const IndexedProblem& problem,
                                               const ActiveDecomposition& decomp) {
  const size_t num_ops = problem.NumOps();
  std::vector<OpId> all_ops(num_ops);
  std::iota(all_ops.begin(), all_ops.end(), 0);
  std::vector<std::pair<OpId, OpId>> dataflow_edges;
  std::vector<std::pair<OpId, OpId>> primal_edges;

  for (const auto& v_info : problem.Values()) {
    const auto& pts = v_info.participants;
    if (v_info.producer_op.has_value()) {
      const OpId prod = *v_info.producer_op;
      for (OpId c : pts) {
        if (c != prod) {
          dataflow_edges.emplace_back(prod, c);
        }
      }
    }
    for (size_t i = 0; i < pts.size(); ++i) {
      for (size_t j = i + 1; j < pts.size(); ++j) {
        const OpId u = pts[i];
        const OpId w = pts[j];
        primal_edges.emplace_back(u, w);
      }
    }
  }

  auto df_res = MinFillDecomposition(all_ops, dataflow_edges);
  auto primal_res = MinFillDecomposition(all_ops, primal_edges);

  int64_t max_bag_size = 0;
  for (const auto& bag : primal_res.bags) {
    max_bag_size = std::max(max_bag_size, static_cast<int64_t>(bag.size()));
  }
  TreewidthStats stats;
  stats.num_values = static_cast<int64_t>(problem.NumValues());
  stats.num_operations = static_cast<int64_t>(num_ops);
  stats.dataflow_treewidth = df_res.treewidth;
  stats.primal_treewidth = primal_res.treewidth;
  stats.max_bag_size = max_bag_size;
  for (const auto& dec : decomp.component_decomps) {
    stats.num_movable_operations += static_cast<int64_t>(dec.bags.size());
    stats.active_primal_treewidth = std::max(stats.active_primal_treewidth, dec.treewidth);
    stats.max_active_bag_size = std::max(stats.max_active_bag_size, dec.treewidth + 1);
    stats.max_bag_domain_product = std::max(stats.max_bag_domain_product, dec.max_bag_states);
    stats.total_dp_states = SaturatingAdd(stats.total_dp_states, dec.total_states);
  }
  if (num_ops > 0 && decomp.components.empty()) {
    stats.max_bag_domain_product = 1;
    stats.total_dp_states = 1;
  }
  stats.elimination_order.reserve(primal_res.elim_order.size());
  for (OpId u : primal_res.elim_order) {
    stats.elimination_order.push_back(problem.Ops()[u].op_id);
  }
  stats.bags.reserve(primal_res.bags.size());
  for (const auto& bag : primal_res.bags) {
    std::vector<std::string> names;
    names.reserve(bag.size());
    for (OpId u : bag) {
      names.push_back(problem.Ops()[u].op_id);
    }
    stats.bags.push_back(std::move(names));
  }
  return stats;
}

// Find components on the op/value incidence graph. A shared value connects its
// participants via one factor; expanding to pairwise edges here would be quadratic.
// nullopt requests just the components for FM, with no elimination ordering.
ActiveDecomposition ComputeActiveDecomposition(const IndexedProblem& problem,
                                               std::optional<int64_t> width_limit) {
  ActiveDecomposition out;
  std::vector<std::vector<ValueId>> factors_by_op(problem.NumOps());
  std::vector<std::vector<OpId>> scopes(problem.NumValues());
  for (ValueId v = 0; v < problem.NumValues(); ++v) {
    for (OpId u : problem.Values()[v].participants) {
      if (problem.Ops()[u].costs.size() > 1) {
        scopes[v].push_back(u);
        factors_by_op[u].push_back(v);
      }
    }
  }

  std::vector<bool> seen_op(problem.NumOps(), false);
  std::vector<bool> seen_value(problem.NumValues(), false);
  for (OpId start = 0; start < problem.NumOps(); ++start) {
    if (seen_op[start] || problem.Ops()[start].costs.size() <= 1) continue;
    auto& ops = out.components.emplace_back();
    ops.push_back(start);
    seen_op[start] = true;
    auto& factors = out.component_factors.emplace_back();
    int64_t lower_bound = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
      for (ValueId v : factors_by_op[ops[i]]) {
        if (seen_value[v]) continue;
        seen_value[v] = true;
        factors.push_back(v);
        lower_bound = std::max(lower_bound, static_cast<int64_t>(scopes[v].size()) - 1);
        for (OpId u : scopes[v]) {
          if (!seen_op[u]) {
            seen_op[u] = true;
            ops.push_back(u);
          }
        }
      }
    }
    std::sort(ops.begin(), ops.end());
    auto& dec = out.component_decomps.emplace_back();
    // Each factor induces a clique. This bound suffices to reject DP without
    // building the clique or running min-fill on this component.
    dec.treewidth = lower_bound;
    if (!width_limit || lower_bound > *width_limit) continue;
    std::vector<std::pair<OpId, OpId>> edges;
    for (ValueId v : factors) {
      const auto& scope = scopes[v];
      for (size_t i = 0; i < scope.size(); ++i) {
        for (size_t j = i + 1; j < scope.size(); ++j) {
          edges.emplace_back(scope[i], scope[j]);
        }
      }
    }
    dec = MinFillDecomposition(ops, edges);
    for (const auto& bag : dec.bags) {
      const auto states = BagStates(problem, bag);
      dec.max_bag_states = std::max(dec.max_bag_states, states);
      dec.total_states = SaturatingAdd(dec.total_states, states);
    }
  }
  return out;
}

/// Precomputed descriptor for one tensor factor assigned to a DP bag.
struct BagFactorDesc {
  ValueId value_id = 0;
  int prod_bag_pos = -1;  ///< -1 when producer is fixed/external; else index in bag.
  uint16_t fixed_source_idx = 0;
  uint64_t base_demand_mask = 0;
  struct ConsumerEntry {
    size_t bag_pos = 0;
    OpId op_id = 0;
    std::vector<uint64_t> demand_by_choice;
  };
  std::vector<ConsumerEntry> consumers;
};

/// Precomputed descriptor for one child bag message lookup.
struct ChildMessageDesc {
  size_t child_step = 0;
  size_t elim_stride = 0;
  std::vector<size_t> sep_strides;  ///< Aligned with parent bag's separator positions.
};

struct DPContext {
  std::vector<bool> is_movable;
  std::vector<uint64_t> base_demand;
  std::vector<uint16_t> base_source;
  std::vector<int> comp_of_op;
  std::vector<size_t> step_of_op;
  std::vector<std::vector<std::pair<ValueId, size_t>>> comp_factors;
};

DPContext BuildDPContext(const IndexedProblem& problem, const ActiveDecomposition& decomp) {
  const size_t num_ops = problem.NumOps();
  const size_t num_vals = problem.NumValues();
  const size_t num_comps = decomp.component_decomps.size();
  DPContext ctx;
  ctx.is_movable.assign(num_ops, false);
  for (OpId u = 0; u < num_ops; ++u) {
    if (problem.Ops()[u].costs.size() > 1) {
      ctx.is_movable[u] = true;
    }
  }

  ctx.base_demand.assign(num_vals, 0);
  ctx.base_source.assign(num_vals, 0);
  for (ValueId v = 0; v < num_vals; ++v) {
    ctx.base_demand[v] = problem.Values()[v].boundary_demand_mask;
    ctx.base_source[v] = problem.Values()[v].default_source_idx;
  }
  for (OpId u = 0; u < num_ops; ++u) {
    if (ctx.is_movable[u]) {
      continue;
    }
    const auto& op = problem.Ops()[u];
    ctx.base_source[op.output_value] = op.output_layout_indices[0];
    for (const auto& use : op.config_uses[0]) {
      ctx.base_demand[use.value_id] |= use.demand_mask;
    }
  }

  ctx.comp_of_op.assign(num_ops, -1);
  ctx.step_of_op.assign(num_ops, 0);
  for (size_t c = 0; c < num_comps; ++c) {
    const auto& dec = decomp.component_decomps[c];
    for (size_t s = 0; s < dec.elim_order.size(); ++s) {
      ctx.comp_of_op[dec.elim_order[s]] = static_cast<int>(c);
      ctx.step_of_op[dec.elim_order[s]] = s;
    }
  }

  // O(V + E) linear-time factor placement across all connected components.
  ctx.comp_factors.resize(num_comps);
  std::vector<size_t> comp_min_step(num_comps, std::numeric_limits<size_t>::max());
  std::vector<size_t> touched_comps;
  for (ValueId v = 0; v < num_vals; ++v) {
    touched_comps.clear();
    for (OpId p : problem.Values()[v].participants) {
      const int c = ctx.comp_of_op[p];
      if (c >= 0) {
        const size_t cu = static_cast<size_t>(c);
        if (comp_min_step[cu] == std::numeric_limits<size_t>::max()) {
          touched_comps.push_back(cu);
        }
        comp_min_step[cu] = std::min(comp_min_step[cu], ctx.step_of_op[p]);
      }
    }
    for (size_t cu : touched_comps) {
      ctx.comp_factors[cu].emplace_back(v, comp_min_step[cu]);
      comp_min_step[cu] = std::numeric_limits<size_t>::max();
    }
  }
  return ctx;
}

bool RunTreewidthDPOnSingleComponent(const IndexedProblem& problem, const ActiveDecomposition& decomp,
                                     const DPContext& ctx, size_t comp_idx, const LayoutSolverConfig& config,
                                     int64_t& evaluated_moves, std::vector<uint16_t>& out_choices) {
  const auto& dec = decomp.component_decomps[comp_idx];
  const size_t num_bags = dec.bags.size();
  if (num_bags == 0) {
    return true;
  }

  std::vector<std::vector<ValueId>> factors_at_bag(num_bags);
  for (const auto& [v, first_step] : ctx.comp_factors[comp_idx]) {
    factors_at_bag[first_step].push_back(v);
  }

  std::vector<std::vector<size_t>> children_of(num_bags);
  for (size_t s = 0; s < num_bags; ++s) {
    if (dec.parents[s].has_value()) {
      children_of[*dec.parents[s]].push_back(s);
    }
  }

  // Precompute mixed-radix separator sizes and strides for each bag.
  std::vector<size_t> sep_sizes(num_bags, 1);
  std::vector<std::vector<size_t>> sep_strides(num_bags);
  for (size_t s = 0; s < num_bags; ++s) {
    const size_t sep_len = dec.bags[s].size() - 1;
    sep_strides[s].assign(sep_len, 1);
    size_t stride = 1;
    for (int j = static_cast<int>(sep_len) - 1; j >= 0; --j) {
      sep_strides[s][static_cast<size_t>(j)] = stride;
      const OpId u = dec.bags[s][static_cast<size_t>(j) + 1];
      stride *= problem.Ops()[u].costs.size();
    }
    sep_sizes[s] = stride;
  }

  std::vector<std::vector<LayoutSearchScore>> messages(num_bags);
  std::vector<std::vector<uint16_t>> backpointers(num_bags);

  std::unordered_map<OpId, size_t> bag_pos;
  std::vector<uint16_t> sep_q;
  std::vector<uint16_t> bag_q;

  for (size_t s = 0; s < num_bags; ++s) {
    const auto& bag = dec.bags[s];
    const OpId elim_u = bag[0];
    const size_t sep_len = bag.size() - 1;
    const size_t sep_size = sep_sizes[s];
    const size_t elim_dom = problem.Ops()[elim_u].costs.size();

    bag_pos.clear();
    for (size_t idx = 0; idx < bag.size(); ++idx) {
      bag_pos[bag[idx]] = idx;
    }

    std::vector<ChildMessageDesc> child_descs;
    child_descs.reserve(children_of[s].size());
    for (size_t c : children_of[s]) {
      ChildMessageDesc cd;
      cd.child_step = c;
      cd.sep_strides.assign(sep_len, 0);
      const auto& c_bag = dec.bags[c];
      for (size_t cj = 1; cj < c_bag.size(); ++cj) {
        const OpId cw = c_bag[cj];
        const size_t p_pos = bag_pos[cw];
        const size_t c_stride = sep_strides[c][cj - 1];
        if (p_pos == 0) {
          cd.elim_stride = c_stride;
        } else {
          cd.sep_strides[p_pos - 1] = c_stride;
        }
      }
      child_descs.push_back(std::move(cd));
    }

    std::vector<BagFactorDesc> factor_descs;
    factor_descs.reserve(factors_at_bag[s].size());
    for (ValueId v : factors_at_bag[s]) {
      const auto& v_info = problem.Values()[v];
      BagFactorDesc fd;
      fd.value_id = v;
      fd.base_demand_mask = ctx.base_demand[v];
      fd.fixed_source_idx = ctx.base_source[v];
      if (v_info.producer_op.has_value() && ctx.is_movable[*v_info.producer_op]) {
        fd.prod_bag_pos = static_cast<int>(bag_pos[*v_info.producer_op]);
      }
      for (OpId p : v_info.participants) {
        if (!ctx.is_movable[p]) {
          continue;
        }
        BagFactorDesc::ConsumerEntry ce;
        ce.bag_pos = bag_pos[p];
        ce.op_id = p;
        const size_t p_dom = problem.Ops()[p].costs.size();
        ce.demand_by_choice.assign(p_dom, 0);
        bool has_any_demand = false;
        for (uint16_t q = 0; q < static_cast<uint16_t>(p_dom); ++q) {
          for (const auto& use : problem.Ops()[p].config_uses[q]) {
            if (use.value_id == v) {
              ce.demand_by_choice[q] = use.demand_mask;
              if (use.demand_mask != 0) {
                has_any_demand = true;
              }
              break;
            }
          }
        }
        if (has_any_demand) {
          fd.consumers.push_back(std::move(ce));
        }
      }
      factor_descs.push_back(std::move(fd));
    }

    messages[s].resize(sep_size);
    backpointers[s].assign(sep_size, 0);
    sep_q.assign(sep_len, 0);
    bag_q.assign(bag.size(), 0);

    const auto& elim_costs = problem.Ops()[elim_u].costs;
    for (size_t sep_idx = 0; sep_idx < sep_size; ++sep_idx) {
      for (size_t j = 0; j < sep_len; ++j) {
        bag_q[j + 1] = sep_q[j];
      }
      LayoutSearchScore best_val = kWorstCost;
      uint16_t best_q = 0;
      for (uint16_t q_elim = 0; q_elim < static_cast<uint16_t>(elim_dom); ++q_elim) {
        if (evaluated_moves + 1 > config.max_evaluations) {
          return false;
        }
        ++evaluated_moves;
        bag_q[0] = q_elim;
        LayoutSearchScore val{problem.Ops()[elim_u].incompatible_configs[q_elim], elim_costs[q_elim]};

        for (const auto& fd : factor_descs) {
          const auto& v_info = problem.Values()[fd.value_id];
          const size_t num_layouts = v_info.layouts.size();
          uint16_t src_idx = fd.fixed_source_idx;
          if (fd.prod_bag_pos >= 0) {
            const OpId prod_op = bag[static_cast<size_t>(fd.prod_bag_pos)];
            const uint16_t prod_q = bag_q[static_cast<size_t>(fd.prod_bag_pos)];
            src_idx = problem.Ops()[prod_op].output_layout_indices[prod_q];
          }
          uint64_t mask = fd.base_demand_mask;
          for (const auto& ce : fd.consumers) {
            mask |= ce.demand_by_choice[bag_q[ce.bag_pos]];
          }
          val.violations += FactorViolations(v_info, src_idx, mask);
          if (v_info.forbid_multi_rep && HasMultipleRepresentations(mask)) {
            val.cycles += problem.Instance().cost_model.vec_unsupported_conversion_cycles;
          }
          const double* row = v_info.conv_costs.data() + static_cast<size_t>(src_idx) * num_layouts;
          while (mask != 0) {
            const int dst = __builtin_ctzll(mask);
            val.cycles += row[dst];
            mask &= mask - 1;
          }
        }

        for (const auto& cd : child_descs) {
          size_t c_idx = static_cast<size_t>(q_elim) * cd.elim_stride;
          for (size_t j = 0; j < sep_len; ++j) {
            c_idx += static_cast<size_t>(sep_q[j]) * cd.sep_strides[j];
          }
          const auto& child_score = messages[cd.child_step][c_idx];
          val.violations += child_score.violations;
          val.cycles += child_score.cycles;
        }

        if (BetterCost(val, best_val, config.tolerance)) {
          best_val = val;
          best_q = q_elim;
        }
      }
      messages[s][sep_idx] = best_val;
      backpointers[s][sep_idx] = best_q;

      // Advance mixed-radix odometer for sep_q.
      for (int j = static_cast<int>(sep_len) - 1; j >= 0; --j) {
        const size_t jj = static_cast<size_t>(j);
        const OpId sep_op = bag[jj + 1];
        const uint16_t dom = static_cast<uint16_t>(problem.Ops()[sep_op].costs.size());
        if (++sep_q[jj] < dom) {
          break;
        }
        sep_q[jj] = 0;
      }
    }
  }

  // Backtrack optimal choices in reverse elimination order.
  for (int s = static_cast<int>(num_bags) - 1; s >= 0; --s) {
    const size_t ss = static_cast<size_t>(s);
    const auto& bag = dec.bags[ss];
    const OpId elim_u = bag[0];
    size_t sep_idx = 0;
    for (size_t j = 1; j < bag.size(); ++j) {
      sep_idx += static_cast<size_t>(out_choices[bag[j]]) * sep_strides[ss][j - 1];
    }
    out_choices[elim_u] = backpointers[ss][sep_idx];
  }

  return true;
}

bool RunTreewidthDPOnComponents(const IndexedProblem& problem, const ActiveDecomposition& decomp,
                                const LayoutSolverConfig& config, int64_t& evaluated_moves,
                                std::vector<uint16_t>& out_choices) {
  out_choices.assign(problem.NumOps(), 0);
  const DPContext ctx = BuildDPContext(problem, decomp);
  for (size_t comp_idx = 0; comp_idx < decomp.component_decomps.size(); ++comp_idx) {
    if (!RunTreewidthDPOnSingleComponent(problem, decomp, ctx, comp_idx, config, evaluated_moves,
                                         out_choices)) {
      return false;
    }
  }
  return true;
}

void GreedyRefineOps(IncrementalState& state, const std::vector<OpId>& ops, const LayoutSolverConfig& config,
                     int64_t& evaluated_moves, int64_t& moves_applied, int64_t& greedy_sweeps,
                     std::string& termination_reason) {
  const auto& problem = state.Problem();
  for (int64_t sweep = 0; sweep < config.max_greedy_sweeps; ++sweep) {
    ++greedy_sweeps;
    bool improved = false;
    for (OpId op_idx : ops) {
      const size_t num_cfg = problem.Ops()[op_idx].costs.size();
      LayoutSearchScore best_gain;
      uint16_t best_choice = 0;
      bool found = false;
      for (uint16_t q = 0; q < static_cast<uint16_t>(num_cfg); ++q) {
        if (q == state.Choices()[op_idx]) {
          continue;
        }
        if (evaluated_moves + 1 > config.max_evaluations) {
          termination_reason = "evaluation_budget";
          return;
        }
        ++evaluated_moves;
        const auto gain = state.GainScore(op_idx, q);
        if (!found || BetterGain(gain, best_gain, config.tolerance)) {
          best_gain = gain;
          best_choice = q;
          found = true;
        }
      }
      if (found && BetterGain(best_gain, {}, config.tolerance)) {
        state.Apply(op_idx, best_choice);
        ++moves_applied;
        improved = true;
      }
    }
    if (!improved) {
      termination_reason = "local_optimum";
      return;
    }
  }
  termination_reason = "greedy_sweep_budget";
}

void GreedyRefine(IncrementalState& state, const LayoutSolverConfig& config, int64_t& evaluated_moves,
                  int64_t& moves_applied, int64_t& greedy_sweeps, std::string& termination_reason) {
  GreedyRefineOps(state, state.Problem().TopoOrder(), config, evaluated_moves, moves_applied, greedy_sweeps,
                  termination_reason);
}

// Report only net changes so the frontier invalidates scores only for affected factors.
template <typename Visit>
void ForEachDemandDelta(const IndexedProblem::OpConfigUse& old_use,
                        const IndexedProblem::OpConfigUse& new_use, Visit visit) {
  const auto find_count = [](const auto& use, uint16_t layout) {
    for (const auto& [index, count] : use.layout_counts) {
      if (index == layout) return count;
    }
    return 0;
  };
  for (const auto& [layout, old_count] : old_use.layout_counts) {
    const int delta = find_count(new_use, layout) - old_count;
    if (delta != 0) visit(layout, delta);
  }
  for (const auto& [layout, new_count] : new_use.layout_counts) {
    if ((old_use.demand_mask & (uint64_t{1} << layout)) == 0) visit(layout, new_count);
  }
}

// The tournament stores one exact maximum per unlocked operation. Leaf order is the
// original FM scan order, and equal gains pick the earlier operation/configuration.
// When tolerance could prevent that maximum from winning, replay the ordered scan
// from cached gains, pruning any subtree whose maximum cannot improve the incumbent.
class FMMoveFrontier {
 public:
  static constexpr uint16_t kNoChoice = std::numeric_limits<uint16_t>::max();

  struct Move {
    size_t local = 0;
    uint16_t choice = kNoChoice;
    LayoutSearchScore gain = kWorstGain;
  };

  FMMoveFrontier(const IncrementalState& state, const std::vector<OpId>& ops,
                 const LayoutSolverConfig& config, int64_t& evaluated_moves)
      : state_(state),
        ops_(ops),
        config_(config),
        evaluated_moves_(evaluated_moves),
        before_best_(ops.size()) {
    while (leaf_base_ < ops.size()) leaf_base_ *= 2;
    tree_.resize(2 * leaf_base_);
    offsets_.reserve(ops.size() + 1);
    offsets_.push_back(0);
    for (OpId op : ops) offsets_.push_back(offsets_.back() + state.Problem().Ops()[op].costs.size());
    gains_.resize(offsets_.back());
  }

  void Reset() { std::fill(tree_.begin(), tree_.end(), Move{}); }

  [[nodiscard]] bool Active(size_t local) const { return tree_[leaf_base_ + local].choice != kNoChoice; }
  [[nodiscard]] bool Empty() const { return tree_[1].choice == kNoChoice; }

  bool Refresh(size_t local) {
    if (!Evaluate(local)) return false;
    UpdatePath(local);
    return true;
  }

  void Lock(size_t local) {
    if (!Active(local)) return;
    tree_[leaf_base_ + local] = {};
    UpdatePath(local);
  }

  [[nodiscard]] Move Best() const {
    const Move winner = tree_[1];
    size_t first_node = 1;
    while (first_node < leaf_base_) {
      first_node *= 2;
      if (tree_[first_node].choice == kNoChoice) ++first_node;
    }
    const size_t first = first_node - leaf_base_;
    const uint16_t first_choice = state_.Choices()[ops_[first]] == 0 ? 1 : 0;
    Move best{first, first_choice, gains_[offsets_[first] + first_choice]};
    // An unordered cycle comparison can only affect moves with equal feasibility.
    if (std::isnan(best.gain.cycles) && best.gain.violations == winner.gain.violations) return best;
    if (config_.tolerance == 0.0 || (best.local == winner.local && best.choice == winner.choice)) {
      return winner;
    }

    // If the maximum beats every preceding alternative by more than tolerance,
    // the scan must accept it and cannot replace it. Query only the left siblings
    // on its path and the cached prefix within its operation.
    auto before = before_best_[winner.local];
    for (size_t node = leaf_base_ + winner.local; node > 1; node /= 2) {
      if ((node & 1) != 0 && BetterGain(tree_[node - 1].gain, before)) before = tree_[node - 1].gain;
    }
    if (BetterGain(winner.gain, before, config_.tolerance)) return winner;
    Scan(1, best);
    return best;
  }

 private:
  bool Evaluate(size_t local) {
    const OpId op = ops_[local];
    const uint16_t current = state_.Choices()[op];
    Move winner{local, kNoChoice, kWorstGain};
    const auto num_choices = static_cast<uint16_t>(offsets_[local + 1] - offsets_[local]);
    for (uint16_t q = 0; q < num_choices; ++q) {
      if (q == current) continue;
      if (evaluated_moves_ >= config_.max_evaluations) return false;
      ++evaluated_moves_;
      const auto gain = state_.GainScore(op, q);
      gains_[offsets_[local] + q] = gain;
      // Ignore NaNs for maxima. Best() separately handles a NaN in the first
      // unlocked candidate, which is the only way the ordered scan can pick one.
      if (winner.choice == kNoChoice || BetterGain(gain, winner.gain)) {
        before_best_[local] = winner.gain;
        winner = {local, q,
                  std::isnan(gain.cycles) ? LayoutSearchScore{gain.violations, kWorstGain.cycles} : gain};
      }
    }
    tree_[leaf_base_ + local] = winner;
    return true;
  }

  void UpdateNode(size_t node) {
    const Move left = tree_[2 * node];
    const Move right = tree_[2 * node + 1];
    tree_[node] = left.choice == kNoChoice || BetterGain(right.gain, left.gain) ? right : left;
  }

  void UpdatePath(size_t local) {
    for (size_t node = (leaf_base_ + local) / 2; node > 0; node /= 2) UpdateNode(node);
  }

  void Scan(size_t node, Move& best) const {
    if (!BetterGain(tree_[node].gain, best.gain, config_.tolerance)) return;
    if (node < leaf_base_) {
      Scan(2 * node, best);
      Scan(2 * node + 1, best);
      return;
    }
    const size_t local = node - leaf_base_;
    const uint16_t current = state_.Choices()[ops_[local]];
    const auto num_choices = static_cast<uint16_t>(offsets_[local + 1] - offsets_[local]);
    for (uint16_t q = 0; q < num_choices; ++q) {
      const auto gain = gains_[offsets_[local] + q];
      if (q != current && BetterGain(gain, best.gain, config_.tolerance)) best = {local, q, gain};
    }
  }

  const IncrementalState& state_;
  const std::vector<OpId>& ops_;
  const LayoutSolverConfig& config_;
  int64_t& evaluated_moves_;
  size_t leaf_base_ = 1;
  std::vector<Move> tree_;
  std::vector<size_t> offsets_;
  std::vector<LayoutSearchScore> gains_;
  std::vector<LayoutSearchScore> before_best_;
};

// The group neighborhood proposes one configuration per changing consumer for
// each common target layout. Rank matching configurations outside the shared
// factor, then score the assembled proposal on all factors jointly.
class FMFanoutFrontier {
 public:
  struct Proposal {
    ValueId value = 0;
    std::vector<IndexedProblem::ValueInfo::Consumer> changing_consumers;
    std::vector<IncrementalState::ChoiceChange> choices;
    IncrementalState::BlockEvaluation evaluation;
    long double rebased_gain = 0.0L;
    int64_t rebased_violation_reduction = 0;
    long double rounding_margin = 0.0L;
    long double rounding_margin_per_lock = 0.0L;
    LayoutSearchScore priority;
    bool unique_choices = false;
    bool needs_restaging = false;
  };

  FMFanoutFrontier(const IncrementalState& state, const std::vector<ValueId>& comp_factors,
                   const LayoutSolverConfig& config, int64_t& evaluated_moves)
      : state_(state), config_(config), evaluated_moves_(evaluated_moves) {
    const auto& problem = state.Problem();
    auto fanouts = comp_factors;
    std::sort(fanouts.begin(), fanouts.end());
    for (ValueId value : fanouts) {
      const auto& info = problem.Values()[value];
      if (info.consumers.size() < 2 || info.common_consumer_layouts == 0) continue;
      const auto movable_consumers =
          std::count_if(info.consumers.begin(), info.consumers.end(),
                        [&](const auto& consumer) { return problem.Ops()[consumer.op_id].costs.size() > 1; });
      if (movable_consumers < 2) continue;

      const size_t index = groups_.size();
      groups_.emplace_back(value, info.layouts.size());
      group_by_value_.emplace(value, index);
      const auto add_dependency = [&](ValueId factor) {
        auto& dependents = dependents_by_factor_[factor];
        if (dependents.empty() || dependents.back() != index) dependents.push_back(index);
      };
      for (const auto& consumer : info.consumers) {
        const auto& op = problem.Ops()[consumer.op_id];
        if (op.costs.size() == 1) continue;
        // A factor may affect either choice ranking or the exact joint gain.
        // All configurations of an operation have the same factor scope.
        add_dependency(op.output_value);
        for (const auto& use : op.config_uses.front()) add_dependency(use.value_id);
      }
    }
  }

  void Reset() {
    scores_.clear();
    dirty_.clear();
    for (auto& group : groups_) {
      const auto& info = state_.Problem().Values()[group.value];
      group.allowed_targets = info.common_consumer_layouts;
      group.scored_targets = 0;
      group.rebased_target = 0;
      group.admitted = false;
      for (auto& proposal : group.proposals) proposal = {};
    }
  }

  void Admit(ValueId value) {
    const auto it = group_by_value_.find(value);
    if (it == group_by_value_.end()) return;
    auto& group = groups_[it->second];
    if (group.admitted || group.allowed_targets == 0) return;
    group.admitted = true;
    std::fill(group.consumers_at_target.begin(), group.consumers_at_target.end(), 0);
    for (const auto& consumer : state_.Problem().Values()[value].consumers) {
      const auto& op = state_.Problem().Ops()[consumer.op_id];
      const uint64_t mask = op.config_uses[state_.Choices()[consumer.op_id]][consumer.input_slot].demand_mask;
      if (IsSingleLayout(mask)) ++group.consumers_at_target[__builtin_ctzll(mask)];
    }
    dirty_.insert(it->second);
  }

  void Lock(OpId op_id, uint16_t new_choice, std::optional<LayoutSearchScore> singleton_gain) {
    const auto& op = state_.Problem().Ops()[op_id];
    const auto& old_uses = op.config_uses[state_.Choices()[op_id]];
    const auto& new_uses = op.config_uses[new_choice];
    for (size_t slot = 0; slot < old_uses.size(); ++slot) {
      const auto it = group_by_value_.find(old_uses[slot].value_id);
      if (it == group_by_value_.end()) continue;
      auto& group = groups_[it->second];
      const uint64_t old_mask = old_uses[slot].demand_mask;
      const uint64_t new_mask = new_uses[slot].demand_mask;
      // Locked consumers may stay in a proposal only if they already request
      // precisely its target. Track that constraint without scanning the group.
      const uint64_t allowed = group.allowed_targets & (IsSingleLayout(new_mask) ? new_mask : 0);
      if (group.admitted && (allowed != group.allowed_targets || old_mask != new_mask)) {
        dirty_.insert(it->second);
      }
      if (singleton_gain && old_mask != new_mask && (allowed & group.scored_targets) != 0) {
        auto& proposal = group.proposals[__builtin_ctzll(allowed)];
        if (proposal.unique_choices && std::isfinite(proposal.priority.cycles) &&
            std::isfinite(proposal.rounding_margin_per_lock)) {
          // The indexed consumer order and every subsequent filter preserve OpId order.
          const IncrementalState::ChoiceChange move{op_id, new_choice};
          const auto member = std::lower_bound(proposal.choices.begin(), proposal.choices.end(), move);
          if (member != proposal.choices.end() && *member == move) {
            // This singleton reached the same choice as the cached proposal.
            // Each remaining member has a unique choice, so the proposal's
            // endpoint survives unchanged even when side factors are shared.
            group.rebased_target = allowed;
            proposal.rebased_gain -= singleton_gain->cycles;
            proposal.rebased_violation_reduction -= singleton_gain->violations;
            proposal.rounding_margin += proposal.rounding_margin_per_lock;
          }
        }
      }
      group.allowed_targets = allowed;
      if (group.admitted && old_mask != new_mask) {
        if (IsSingleLayout(old_mask)) --group.consumers_at_target[__builtin_ctzll(old_mask)];
        if (IsSingleLayout(new_mask)) ++group.consumers_at_target[__builtin_ctzll(new_mask)];
      }
    }
  }

  void Invalidate(ValueId factor) {
    const auto it = dependents_by_factor_.find(factor);
    if (it == dependents_by_factor_.end()) return;
    for (size_t index : it->second) {
      if (groups_[index].admitted && groups_[index].allowed_targets != 0) dirty_.insert(index);
    }
  }

  bool Refresh() {
    for (size_t index : dirty_) {
      auto& group = groups_[index];
      const uint64_t rebased_target = std::exchange(group.rebased_target, 0);
      while (group.scored_targets != 0) {
        const auto target = static_cast<uint16_t>(__builtin_ctzll(group.scored_targets));
        auto& proposal = group.proposals[target];
        scores_.erase({proposal.priority, index, target});
        if ((rebased_target & (uint64_t{1} << target)) == 0) {
          proposal.choices.clear();
          proposal.evaluation = {};
        }
        group.scored_targets &= group.scored_targets - 1;
      }

      const auto& info = state_.Problem().Values()[group.value];
      uint64_t targets = group.allowed_targets;
      while (targets != 0) {
        const auto target = static_cast<uint16_t>(__builtin_ctzll(targets));
        const uint64_t bit = uint64_t{1} << target;
        targets &= targets - 1;
        const size_t block_size = info.consumers.size() - group.consumers_at_target[target];
        if (block_size < 2) continue;
        auto& proposal = group.proposals[target];
        if (rebased_target == bit) {
          // Treat a rebased score as an upper bound. An ambiguous comparison
          // or potential winner forces a fresh exact score before selection.
          const auto upper = static_cast<double>(proposal.rebased_gain + proposal.rounding_margin);
          proposal.priority = {proposal.rebased_violation_reduction,
                               std::nextafter(upper, std::numeric_limits<double>::infinity())};
          proposal.needs_restaging = true;
        } else if (!Evaluate(group.value, bit, block_size, proposal)) {
          return false;
        }
        if (std::isnan(proposal.priority.cycles)) continue;
        group.scored_targets |= bit;
        scores_.insert({proposal.priority, index, target});
      }
    }
    dirty_.clear();
    return true;
  }

  bool Best(LayoutSearchScore singleton_gain, const Proposal*& winner) {
    winner = nullptr;
    while (!scores_.empty() && BetterGain(scores_.begin()->gain, singleton_gain, config_.tolerance)) {
      const auto score = *scores_.begin();
      auto& group = groups_[score.group];
      auto& proposal = group.proposals[score.target];
      if (!proposal.needs_restaging) {
        winner = &proposal;
        return true;
      }
      const auto& info = state_.Problem().Values()[group.value];
      const auto charge =
          static_cast<int64_t>(info.consumers.size() - group.consumers_at_target[score.target]);
      if (charge > config_.max_evaluations - evaluated_moves_) return false;
      evaluated_moves_ += charge;
      auto& choices = proposal.choices;
      const auto already_applied = [&](const auto& choice) {
        return state_.Choices()[choice.first] == choice.second;
      };
      choices.erase(std::remove_if(choices.begin(), choices.end(), already_applied), choices.end());
      scores_.erase(scores_.begin());
      SetExactScore(proposal);
      if (!std::isnan(proposal.priority.cycles)) {
        scores_.insert({proposal.priority, score.group, score.target});
      } else {
        group.scored_targets &= ~(uint64_t{1} << score.target);
      }
    }
    return true;
  }

 private:
  struct Group {
    Group(ValueId value, size_t layout_count)
        : value(value), consumers_at_target(layout_count), proposals(layout_count) {}

    ValueId value;
    uint64_t allowed_targets = 0;
    uint64_t scored_targets = 0;
    uint64_t rebased_target = 0;
    bool admitted = false;
    std::vector<size_t> consumers_at_target;
    std::vector<Proposal> proposals;
  };

  struct Score {
    LayoutSearchScore gain;
    size_t group;
    uint16_t target;

    bool operator<(const Score& other) const {
      if (gain.violations != other.gain.violations || gain.cycles != other.gain.cycles) {
        return BetterGain(gain, other.gain);
      }
      return std::tie(group, target) < std::tie(other.group, other.target);
    }
  };

  static bool IsSingleLayout(uint64_t mask) { return mask != 0 && (mask & (mask - 1)) == 0; }

  void SetExactScore(Proposal& proposal) {
    proposal.evaluation = state_.EvaluateBlock(proposal.choices);
    proposal.priority = {proposal.evaluation.violation_reduction, proposal.evaluation.gain};
    proposal.rebased_gain = proposal.priority.cycles;
    proposal.rebased_violation_reduction = proposal.priority.violations;
    proposal.needs_restaging = false;
    if (!proposal.unique_choices) return;

    // Factor evaluation can add and remove conversions in either order.
    // Bound intermediate magnitudes by the cost of every layout in a row,
    // plus any multi-representation penalty, including currently absent costs.
    const auto& problem = state_.Problem();
    long double scale = 1.0L;
    for (const auto& [op, choice] : proposal.choices) {
      const auto& costs = problem.Ops()[op].costs;
      scale += std::max(costs[choice], costs[state_.Choices()[op]]);
    }
    for (const auto& factor : proposal.evaluation.factors) {
      const auto& info = problem.Values()[factor.value];
      scale += static_cast<long double>(info.max_conversion_cost) * info.layouts.size();
      if (info.forbid_multi_rep) {
        scale += std::abs(problem.Instance().cost_model.vec_unsupported_conversion_cycles);
      }
    }
    // Allow for rounding in the original and re-evaluated sums and in the
    // singleton gain. Each factor contributes at most 64 demand terms.
    const long double unit = std::numeric_limits<double>::epsilon() * scale;
    const size_t terms = proposal.choices.size() + proposal.evaluation.factors.size();
    proposal.rounding_margin_per_lock = unit * (256.0L + 8.0L * terms);
    proposal.rounding_margin = proposal.rounding_margin_per_lock;
  }

  bool Evaluate(ValueId value, uint64_t target, size_t block_size, Proposal& proposal) {
    const auto charge = static_cast<int64_t>(block_size);
    if (charge > config_.max_evaluations - evaluated_moves_) return false;

    proposal.value = value;
    const auto& problem = state_.Problem();
    const auto needs_change = [&](const IndexedProblem::ValueInfo::Consumer& consumer) {
      const auto& uses = problem.Ops()[consumer.op_id].config_uses;
      return uses[state_.Choices()[consumer.op_id]][consumer.input_slot].demand_mask != target;
    };
    auto& changing = proposal.changing_consumers;
    if (changing.empty()) {
      const auto& consumers = problem.Values()[value].consumers;
      changing.reserve(block_size);
      std::copy_if(consumers.begin(), consumers.end(), std::back_inserter(changing), needs_change);
    } else {
      // While the target is allowed, any newly locked consumer must already
      // request it. It can only leave this list. Re-score just the remaining
      // members when a side factor or an eligible lock changes the proposal.
      changing.erase(std::remove_if(changing.begin(), changing.end(),
                                    [&](const auto& consumer) { return !needs_change(consumer); }),
                     changing.end());
    }
    std::vector<uint16_t> matching_choices;
    proposal.choices.clear();
    proposal.choices.reserve(block_size);
    proposal.unique_choices = true;
    for (const auto& consumer : changing) {
      const OpId op = consumer.op_id;
      const auto& uses = problem.Ops()[op].config_uses;
      matching_choices.clear();
      const auto num_choices = static_cast<uint16_t>(uses.size());
      for (uint16_t q = 0; q < num_choices; ++q) {
        if (uses[q][consumer.input_slot].demand_mask == target) matching_choices.push_back(q);
      }
      // Common targets come from intersecting the consumer catalogs.
      uint16_t choice = matching_choices.front();
      if (matching_choices.size() > 1) {
        proposal.unique_choices = false;
        auto best_gain = kWorstGain;
        for (uint16_t q : matching_choices) {
          if (evaluated_moves_ == config_.max_evaluations) return false;
          ++evaluated_moves_;
          const auto gain = state_.GainScore(op, q, value);
          if (BetterGain(gain, best_gain, config_.tolerance)) {
            best_gain = gain;
            choice = q;
          }
        }
      }
      proposal.choices.emplace_back(op, choice);
    }
    // Keep the existing charge of one joint evaluation per changed operation
    // in addition to the evaluations used to rank matching configurations.
    if (charge > config_.max_evaluations - evaluated_moves_) return false;
    evaluated_moves_ += charge;
    SetExactScore(proposal);
    return true;
  }

  const IncrementalState& state_;
  const LayoutSolverConfig& config_;
  int64_t& evaluated_moves_;
  std::vector<Group> groups_;
  std::unordered_map<ValueId, size_t> group_by_value_;
  std::unordered_map<ValueId, std::vector<size_t>> dependents_by_factor_;
  std::set<size_t> dirty_;
  std::set<Score> scores_;
};

// Exact block scores depend on all count changes. Single-op gains only change
// when the source, support, or the consumer's ability to remove a layout changes.
struct FMFactorChange {
  bool source_changed = false;
  bool support_changed = false;
  bool consumer_can_remove_changed = false;
  bool counts_changed = false;

  void NoteCountChange(int before, int after, bool repeated_operand) {
    if (before == after) return;
    counts_changed = true;
    support_changed |= (before == 0) != (after == 0);
    consumer_can_remove_changed |= repeated_operand || (before == 1) != (after == 1);
  }
};

template <typename Visit>
void VisitSingleMoveFactors(const IncrementalState& state, OpId op_idx, uint16_t new_choice, Visit visit) {
  const auto& problem = state.Problem();
  const auto& op = problem.Ops()[op_idx];
  const uint16_t old_choice = state.Choices()[op_idx];
  const auto& old_uses = op.config_uses[old_choice];
  const auto& new_uses = op.config_uses[new_choice];
  if (op.output_layout_indices[old_choice] != op.output_layout_indices[new_choice]) {
    FMFactorChange change;
    change.source_changed = true;
    visit(op.output_value, change);
  }
  for (size_t slot = 0; slot < old_uses.size(); ++slot) {
    const ValueId value = old_uses[slot].value_id;
    const auto& counts = state.Counts()[value];
    const bool repeated = problem.Values()[value].has_repeated_operand;
    FMFactorChange change;
    ForEachDemandDelta(old_uses[slot], new_uses[slot], [&](uint16_t layout, int delta) {
      change.NoteCountChange(counts[layout], counts[layout] + delta, repeated);
    });
    visit(value, change);
  }
}

template <typename Visit>
void VisitBlockMoveFactors(const IncrementalState& state, const IncrementalState::BlockEvaluation& evaluation,
                           Visit visit) {
  for (const auto& factor : evaluation.factors) {
    FMFactorChange change;
    change.source_changed = state.Sources()[factor.value] != factor.source;
    const auto& counts = state.Counts()[factor.value];
    const bool repeated = state.Problem().Values()[factor.value].has_repeated_operand;
    for (size_t layout = 0; layout < counts.size(); ++layout) {
      change.NoteCountChange(counts[layout], factor.counts[layout], repeated);
    }
    visit(factor.value, change);
  }
}

void FMRefineComponent(IncrementalState& state, const std::vector<OpId>& comp_ops,
                       const std::vector<ValueId>& comp_factors, const std::vector<OpId>& topo_ops,
                       const LayoutSolverConfig& config, bool locally_optimal, int64_t& evaluated_moves,
                       int64_t& moves_applied, int64_t& greedy_sweeps, std::string& termination_reason) {
  const size_t n = comp_ops.size();
  FMMoveFrontier singles(state, comp_ops, config, evaluated_moves);
  FMFanoutFrontier groups(state, comp_factors, config, evaluated_moves);
  std::vector<bool> marked(n, false);
  std::vector<bool> seen(n, false);
  size_t seen_count = 0;
  std::unordered_set<ValueId> expanded_values;
  std::vector<size_t> affected;
  std::vector<IncrementalState::ChoiceChange> history;
  std::vector<size_t> move_starts;
  std::vector<IncrementalState::ChoiceChange> undo;
  history.reserve(n);
  move_starts.reserve(n);

  const auto& problem = state.Problem();
  const auto local_index = [&](OpId op) {
    // Every movable participant of a factor belongs to this component.
    return static_cast<size_t>(std::lower_bound(comp_ops.begin(), comp_ops.end(), op) - comp_ops.begin());
  };
  const auto activate = [&](size_t local) {
    if (seen[local]) return true;
    seen[local] = true;
    ++seen_count;
    if (!singles.Refresh(local)) return false;
    const auto& op = problem.Ops()[comp_ops[local]];
    groups.Admit(op.output_value);
    for (const auto& use : op.config_uses.front()) groups.Admit(use.value_id);
    return true;
  };
  const auto expand_value = [&](ValueId value) {
    if (!expanded_values.insert(value).second) return true;
    for (OpId op : problem.Values()[value].participants) {
      if (problem.Ops()[op].costs.size() > 1 && !activate(local_index(op))) return false;
    }
    return true;
  };
  const auto expand_move = [&]() {
    for (size_t i = move_starts.back(); i < history.size(); ++i) {
      const auto& op = problem.Ops()[history[i].first];
      if (!expand_value(op.output_value)) return false;
      for (const auto& use : op.config_uses.front()) {
        if (!expand_value(use.value_id)) return false;
      }
    }
    return true;
  };
  const auto record_and_lock = [&](size_t local, uint16_t choice,
                                   std::optional<LayoutSearchScore> singleton_gain) {
    const OpId op = comp_ops[local];
    history.emplace_back(op, state.Choices()[op]);
    seen_count += !seen[local];
    seen[local] = true;
    singles.Lock(local);
    groups.Lock(op, choice, singleton_gain);
  };
  const auto mark_factor = [&](ValueId value, const FMFactorChange& change) {
    if (change.source_changed || change.counts_changed) groups.Invalidate(value);
    if (singles.Empty() ||
        (!change.source_changed && !change.support_changed && !change.consumer_can_remove_changed)) {
      return;
    }
    const auto& info = problem.Values()[value];
    for (OpId op : info.participants) {
      if (problem.Ops()[op].costs.size() <= 1 || (!change.support_changed && op == info.producer_op)) {
        continue;
      }
      const size_t local = local_index(op);
      if (singles.Active(local) && !marked[local]) {
        marked[local] = true;
        affected.push_back(local);
      }
    }
  };

  // Sample regularly spaced positions in extraction order. A stalled pass
  // broadens the sample and a rotation also changes it when its size stays
  // fixed. Cap the initial work independently of component size.
  constexpr size_t kInitialSeeds = 4;
  constexpr size_t kMaxSeeds = 16;
  const size_t initial_seeds = std::min(n, kInitialSeeds);
  const size_t max_seeds = std::min(n, kMaxSeeds);
  size_t seed_count = initial_seeds;
  size_t rotation = 0;
  std::vector<size_t> seeds;
  seeds.reserve(max_seeds);
  std::optional<size_t> preferred_seed;
  std::optional<FMMoveFrontier::Move> failed_full_frontier_start;
  bool tried_fanout_restart = false;
  for (int64_t pass = 0; pass < config.max_greedy_sweeps; ++pass) {
    ++greedy_sweeps;
    history.clear();
    move_starts.clear();
    std::fill(seen.begin(), seen.end(), false);
    seen_count = 0;
    expanded_values.clear();
    auto best_prefix_cost = state.SearchCost();
    size_t best_prefix_len = 0;
    std::optional<size_t> fanout_seed;
    std::optional<FMMoveFrontier::Move> full_frontier_start;
    LayoutSearchScore fanout_seed_gain;
    seeds.clear();
    const auto add_seed = [&](size_t local) {
      if (seeds.size() < seed_count && std::find(seeds.begin(), seeds.end(), local) == seeds.end()) {
        seeds.push_back(local);
      }
    };
    if (preferred_seed) add_seed(*preferred_seed);
    if (failed_full_frontier_start) add_seed(failed_full_frontier_start->local);
    for (size_t i = 0; seeds.size() < seed_count; ++i) {
      add_seed((i * n / seed_count + rotation) % n);
    }
    rotation = (rotation + 1) % n;
    const auto initially_incident = [&](ValueId value) {
      return std::any_of(seeds.begin(), seeds.end(), [&](size_t local) {
        const auto& op = problem.Ops()[comp_ops[local]];
        return op.output_value == value ||
               std::any_of(op.config_uses.front().begin(), op.config_uses.front().end(),
                           [&](const auto& use) { return use.value_id == value; });
      });
    };

    // Every seed and its incident fan-outs contend before the first move.
    // Other operators enter only through the usual move-driven expansion.
    singles.Reset();
    groups.Reset();
    bool budget_hit = !std::all_of(seeds.begin(), seeds.end(), activate) || !groups.Refresh();
    while (!budget_hit && !singles.Empty()) {
      const auto single = singles.Best();
      const FMFanoutFrontier::Proposal* group = nullptr;
      if (!groups.Best(single.gain, group)) {
        budget_hit = true;
        break;
      }
      // Wider samples can offer a different first move. If the same singleton
      // wins and previously opened the whole component, the exact frontier
      // and its tie-breaking after that move are unchanged.
      if (history.empty() && group == nullptr && failed_full_frontier_start &&
          single.local == failed_full_frontier_start->local &&
          single.choice == failed_full_frontier_start->choice) {
        break;
      }
      move_starts.push_back(history.size());
      // A compatible block changes at least two unlocked operations. Exact
      // block gains compete with singleton gains even when both increase cost.
      if (group != nullptr) {
        // A profitable block discovered after moving away from the seed may
        // have arrived too late to recover earlier losses. Remember one of its
        // consumers so the block can compete before those operations lock.
        const LayoutSearchScore gain{group->evaluation.violation_reduction, group->evaluation.gain};
        if (!tried_fanout_restart && BetterGain(gain, {}, config.tolerance) &&
            BetterGain(gain, fanout_seed_gain) && !initially_incident(group->value)) {
          fanout_seed = local_index(group->choices.front().first);
          fanout_seed_gain = gain;
        }
        for (const auto& [op, choice] : group->choices) {
          record_and_lock(local_index(op), choice, std::nullopt);
        }
        if (history.size() < n) VisitBlockMoveFactors(state, group->evaluation, mark_factor);
        state.ApplyBlock(group->choices, group->evaluation);
      } else {
        record_and_lock(single.local, single.choice, single.gain);
        const OpId op = comp_ops[single.local];
        if (history.size() < n) VisitSingleMoveFactors(state, op, single.choice, mark_factor);
        state.Apply(op, single.choice);
      }
      if (BetterCost(state.SearchCost(), best_prefix_cost, config.tolerance)) {
        best_prefix_cost = state.SearchCost();
        best_prefix_len = history.size();
      }
      if (history.size() == n) break;
      for (size_t local : affected) {
        marked[local] = false;
        if (!singles.Refresh(local)) {
          budget_hit = true;
          break;
        }
      }
      affected.clear();
      // Explore every incident value once, even when the move changes only
      // intrinsic cost. A compatible block can lock undiscovered consumers.
      if (!budget_hit && (!expand_move() || !groups.Refresh())) budget_hit = true;
      if (!budget_hit && move_starts.size() == 1 && group == nullptr && seen_count == n) {
        // The first move must reach the whole component on its own. The only
        // operators it may not have brought into the frontier are the seeds.
        const auto reached_by_first_move = [&](size_t local) {
          const auto& op = problem.Ops()[comp_ops[local]];
          return expanded_values.count(op.output_value) != 0 ||
                 std::any_of(op.config_uses.front().begin(), op.config_uses.front().end(),
                             [&](const auto& use) { return expanded_values.count(use.value_id) != 0; });
        };
        if (std::all_of(seeds.begin(), seeds.end(), reached_by_first_move)) full_frontier_start = single;
      }
    }

    // A whole move is the smallest unit of either the retained prefix or the
    // rolled-back suffix. Undo demand changes jointly for each group.
    while (history.size() > best_prefix_len) {
      const size_t begin = move_starts.back();
      move_starts.pop_back();
      if (begin + 1 == history.size()) {
        const auto [op, choice] = history[begin];
        state.Apply(op, choice);
      } else {
        undo.assign(history.begin() + static_cast<decltype(history)::difference_type>(begin), history.end());
        state.ApplyBlock(undo, state.EvaluateBlock(undo));
      }
      history.resize(begin);
    }
    const int64_t previous_moves = moves_applied;
    moves_applied += static_cast<int64_t>(best_prefix_len);
    // FM locks can hide a profitable second change to an operator. Polish the
    // retained assignment before choosing the next seeds, even after complete
    // rollback unless a prior sweep already verified the same assignment.
    locally_optimal &= best_prefix_len == 0;
    if (!locally_optimal && evaluated_moves < config.max_evaluations) {
      GreedyRefineOps(state, topo_ops, config, evaluated_moves, moves_applied, greedy_sweeps,
                      termination_reason);
      if (termination_reason == "evaluation_budget") return;
      locally_optimal = termination_reason == "local_optimum";
    }
    if (budget_hit || (!locally_optimal && evaluated_moves == config.max_evaluations)) {
      termination_reason = "evaluation_budget";
      return;
    }

    preferred_seed = best_prefix_len == 0 ? fanout_seed : std::nullopt;
    if (moves_applied != previous_moves) {
      seed_count = initial_seeds;
      tried_fanout_restart = false;
      failed_full_frontier_start.reset();
    } else {
      if (full_frontier_start) failed_full_frontier_start = full_frontier_start;
      if (seed_count < max_seeds) {
        seed_count = std::min(2 * seed_count, max_seeds);
      } else if (!preferred_seed) {
        termination_reason = "fm_no_improving_prefix";
        return;
      }
    }
    tried_fanout_restart |= preferred_seed.has_value();
  }
  termination_reason = "greedy_sweep_budget";
}

void GreedyIncumbentAndFM(IncrementalState& state, const ActiveDecomposition& graph,
                          const std::vector<size_t>& components, const LayoutSolverConfig& config,
                          int64_t& evaluated_moves, int64_t& moves_applied, int64_t& greedy_sweeps,
                          std::string& termination_reason) {
  const auto seed_choices = state.Choices();
  std::vector<size_t> component_by_op(state.Problem().NumOps(), graph.components.size());
  for (size_t component : components) {
    for (OpId op : graph.components[component]) component_by_op[op] = component;
  }
  std::vector<std::vector<OpId>> topo_components(graph.components.size());
  std::vector<OpId> greedy_ops;
  for (OpId op : state.Problem().TopoOrder()) {
    const size_t component = component_by_op[op];
    if (component == graph.components.size()) continue;
    greedy_ops.push_back(op);
    topo_components[component].push_back(op);
  }
  GreedyRefineOps(state, greedy_ops, config, evaluated_moves, moves_applied, greedy_sweeps,
                  termination_reason);
  if (termination_reason == "evaluation_budget") return;
  const bool greedy_converged = termination_reason == "local_optimum";

  // Start FM from the original seed to retain paths through choices that
  // coordinate descent has already changed. Retain the greedy assignment
  // whenever the excursion fails to find a better prefix within the budget.
  const auto greedy_choices = state.Choices();
  std::vector<IncrementalState::ChoiceChange> restore;
  bool sweep_budget_hit = false;
  for (size_t component : components) {
    const auto& ops = graph.components[component];
    const auto incumbent_cost = state.SearchCost();
    restore.clear();
    for (OpId op : ops) {
      if (seed_choices[op] != greedy_choices[op]) restore.emplace_back(op, seed_choices[op]);
    }
    if (!restore.empty()) state.ApplyBlock(restore, state.EvaluateBlock(restore));

    int64_t retained = 0;
    FMRefineComponent(state, ops, graph.component_factors[component], topo_components[component], config,
                      greedy_converged && restore.empty(), evaluated_moves, retained, greedy_sweeps,
                      termination_reason);
    if (BetterCost(state.SearchCost(), incumbent_cost, config.tolerance)) {
      moves_applied += retained;
    } else {
      restore.clear();
      for (OpId op : ops) {
        if (state.Choices()[op] != greedy_choices[op]) restore.emplace_back(op, greedy_choices[op]);
      }
      if (!restore.empty()) state.ApplyBlock(restore, state.EvaluateBlock(restore));
    }
    if (termination_reason == "evaluation_budget") return;
    sweep_budget_hit |= termination_reason == "greedy_sweep_budget";
  }
  if (!components.empty()) {
    termination_reason = sweep_budget_hit ? "greedy_sweep_budget" : "fm_no_improving_prefix";
  }
}

}  // namespace

void LayoutSolverConfig::Validate() const {
  if (max_evaluations < 1) {
    throw pypto::ValueError("max_evaluations must be at least 1");
  }
  if (max_greedy_sweeps < 1) {
    throw pypto::ValueError("max_greedy_sweeps must be at least 1");
  }
  if (max_treewidth < 0) {
    throw pypto::ValueError("max_treewidth must be at least 0");
  }
  if (max_bag_states < 1) {
    throw pypto::ValueError("max_bag_states must be at least 1");
  }
  if (!std::isfinite(tolerance) || tolerance < 0.0) {
    throw pypto::ValueError("tolerance must be finite and nonnegative");
  }
}

const LayoutAssignment& LayoutSolverResult::WritebackAssignment() const {
  if (!writeback_issues.empty()) {
    std::string joined;
    for (size_t i = 0; i < writeback_issues.size(); ++i) {
      if (i > 0) joined += "; ";
      joined += writeback_issues[i];
    }
    throw pypto::ValueError("Layout solution requires conversion materialization: " + joined);
  }
  return breakdown.assignment;
}

int IndexedProblem::ValueInfo::FindLayoutIndex(const LayoutSpec& spec) const {
  for (size_t i = 0; i < layouts.size(); ++i) {
    if (layouts[i] == spec) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

IndexedProblem::IndexedProblem(const LayoutProblemInstance& instance) : instance_(instance) {
  instance_.cost_model.ValidateUseHorizon();

  // Deterministic value indexing following value_order first, then sorted remaining keys.
  std::vector<std::string> ordered_names;
  ordered_names.reserve(instance_.values.size());
  std::unordered_set<std::string> seen_names;
  for (const auto& name : instance_.value_order) {
    if (instance_.values.count(name) != 0 && seen_names.insert(name).second) {
      ordered_names.push_back(name);
    }
  }
  std::vector<std::string> rest;
  for (const auto& [name, _] : instance_.values) {
    if (seen_names.insert(name).second) {
      rest.push_back(name);
    }
  }
  std::sort(rest.begin(), rest.end());
  ordered_names.insert(ordered_names.end(), rest.begin(), rest.end());

  values_.resize(ordered_names.size());
  value_by_name_.reserve(ordered_names.size() * 2);
  for (ValueId v = 0; v < static_cast<ValueId>(ordered_names.size()); ++v) {
    const std::string& name = ordered_names[v];
    value_by_name_[name] = v;
    const ValueNode& node = instance_.values.at(name);
    ValueInfo& info = values_[v];
    info.name = name;
    auto append_layout = [&](const LayoutSpec& lay) {
      if (info.FindLayoutIndex(lay) < 0) {
        // Each consumer set is a uint64_t mask. Check before building a mask
        // so a layout index of 64 can never reach a left shift.
        if (info.layouts.size() == std::numeric_limits<uint64_t>::digits) {
          throw pypto::ValueError("Value '" + name + "' exceeds 64 candidate layouts");
        }
        info.layouts.push_back(lay);
      }
    };
    append_layout(node.current_layout);
    for (const auto& lay : node.admissible_layouts) {
      append_layout(lay);
    }
  }

  const size_t num_ops = instance_.operations.size();
  ops_.resize(num_ops);
  op_by_id_.reserve(num_ops * 2);
  std::unordered_map<std::string, const OperationNode*> op_ptr_by_id;
  op_ptr_by_id.reserve(num_ops * 2);

  for (OpId u = 0; u < static_cast<OpId>(num_ops); ++u) {
    const auto& op = instance_.operations[u];
    if (op_by_id_.count(op.op_id) != 0) {
      throw pypto::ValueError("Duplicate operation ID '" + op.op_id + "'");
    }
    op_by_id_[op.op_id] = u;
    op_ptr_by_id[op.op_id] = &op;
    if (op.configs.empty()) {
      throw pypto::ValueError("Operation '" + op.op_id + "' has no legal configurations");
    }
    if (op.configs.size() > std::numeric_limits<uint16_t>::max()) {
      throw pypto::ValueError("Operation '" + op.op_id + "' exceeds 65535 configurations");
    }
    if (op.invocations < 0) {
      throw pypto::ValueError("Invocations of '" + op.op_id + "' must be at least 0");
    }

    auto resolve_val = [&](const std::string& val_name) -> ValueId {
      auto it = value_by_name_.find(val_name);
      if (it == value_by_name_.end()) {
        throw pypto::ValueError("Operation '" + op.op_id + "' references unknown value '" + val_name + "'");
      }
      return it->second;
    };

    std::vector<ValueId> input_ids;
    input_ids.reserve(op.inputs.size());
    for (const auto& inp : op.inputs) {
      const ValueId vid = resolve_val(inp);
      input_ids.push_back(vid);
      values_[vid].participants.push_back(u);
    }
    const ValueId out_vid = resolve_val(op.output);
    values_[out_vid].participants.push_back(u);
    if (values_[out_vid].producer_op.has_value()) {
      throw pypto::ValueError("Value '" + op.output +
                              "' has multiple producers; distinct SSA values are required");
    }
    values_[out_vid].producer_op = u;

    OpInfo& op_info = ops_[u];
    op_info.op_id = op.op_id;
    op_info.op_name = op.op_name;
    op_info.invocations = op.invocations;
    op_info.output_value = out_vid;
    op_info.costs.reserve(op.configs.size());
    op_info.incompatible_configs.reserve(op.configs.size());
    op_info.output_layout_indices.reserve(op.configs.size());
    op_info.config_uses.resize(op.configs.size());
    std::vector<uint64_t> single_layouts_by_input;

    const double cost_mult = RequireFiniteNonNegative(OperatorCostMultiplier(instance_, op), op.op_id);
    for (size_t q = 0; q < op.configs.size(); ++q) {
      const auto& cfg = op.configs[q];
      if (cfg.op_id != op.op_id || cfg.op_name != op.op_name) {
        throw pypto::ValueError("Configuration for '" + cfg.op_id + "'/'" + cfg.op_name +
                                "' does not belong to '" + op.op_id + "'");
      }
      if (cfg.input_layouts.size() != op.inputs.size()) {
        throw pypto::ValueError("Configuration of '" + op.op_id + "' has wrong number of inputs");
      }
      RequireFiniteNonNegative(cfg.op_cost_cycles, "Configuration cost of '" + op.op_id + "'");

      // Validate admissible layouts.
      auto check_admissible = [&](const std::string& val_name, const LayoutSpec& lay) {
        const auto& adm = instance_.values.at(val_name).admissible_layouts;
        if (std::find(adm.begin(), adm.end(), lay) == adm.end()) {
          throw pypto::ValueError("Configuration of '" + op.op_id + "' requests inadmissible layout " +
                                  lay.ShortName() + " for '" + val_name + "'");
        }
      };
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        check_admissible(op.inputs[i], cfg.input_layouts[i]);
      }
      check_admissible(op.output, cfg.output_layout);

      op_info.costs.push_back(
          RequireFiniteNonNegative(cost_mult * cfg.op_cost_cycles, "Cost of '" + op.op_id + "'"));
      op_info.incompatible_configs.push_back(instance_.forbid_unmaterializable_gm_conversions &&
                                             HasIncompatibleJoinLayouts(cfg));
      const int out_lay_idx = values_[out_vid].FindLayoutIndex(cfg.output_layout);
      if (out_lay_idx < 0) {
        throw pypto::ValueError("Internal layout indexing failure for '" + op.output + "'");
      }
      op_info.output_layout_indices.push_back(static_cast<uint16_t>(out_lay_idx));

      // Aggregate per-value layout counts for this config.
      std::vector<OpConfigUse> uses;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        const ValueId vid = input_ids[i];
        const int lay_idx = values_[vid].FindLayoutIndex(cfg.input_layouts[i]);
        if (lay_idx < 0) {
          throw pypto::ValueError("Internal layout indexing failure for '" + op.inputs[i] + "'");
        }
        auto it = std::find_if(uses.begin(), uses.end(),
                               [&](const OpConfigUse& u_entry) { return u_entry.value_id == vid; });
        if (it == uses.end()) {
          OpConfigUse u_entry;
          u_entry.value_id = vid;
          u_entry.demand_mask = uint64_t{1} << lay_idx;
          u_entry.layout_counts.emplace_back(static_cast<uint16_t>(lay_idx), 1);
          uses.push_back(std::move(u_entry));
        } else {
          values_[vid].has_repeated_operand = true;
          it->demand_mask |= (uint64_t{1} << lay_idx);
          auto c_it = std::find_if(it->layout_counts.begin(), it->layout_counts.end(), [&](const auto& pair) {
            return pair.first == static_cast<uint16_t>(lay_idx);
          });
          if (c_it == it->layout_counts.end()) {
            it->layout_counts.emplace_back(static_cast<uint16_t>(lay_idx), 1);
          } else {
            ++c_it->second;
          }
        }
      }
      if (q == 0) single_layouts_by_input.resize(uses.size());
      for (size_t slot = 0; slot < uses.size(); ++slot) {
        if (uses[slot].layout_counts.size() == 1) {
          single_layouts_by_input[slot] |= uses[slot].demand_mask;
        }
      }
      op_info.config_uses[q] = std::move(uses);
    }
    for (size_t slot = 0; slot < single_layouts_by_input.size(); ++slot) {
      auto& value = values_[op_info.config_uses[0][slot].value_id];
      value.consumers.push_back({u, slot});
      value.common_consumer_layouts &= single_layouts_by_input[slot];
    }
  }

  for (ValueInfo& v_info : values_) {
    std::sort(v_info.participants.begin(), v_info.participants.end());
    v_info.participants.erase(std::unique(v_info.participants.begin(), v_info.participants.end()),
                              v_info.participants.end());
    const ValueNode& node = instance_.values.at(v_info.name);
    const std::optional<std::string> actual_prod =
        v_info.producer_op.has_value() ? std::optional<std::string>(ops_[*v_info.producer_op].op_id)
                                       : std::nullopt;
    if (node.producer_op_id != actual_prod) {
      throw pypto::ValueError("Value '" + v_info.name + "' producer mismatch");
    }
  }

  // Kahn min-heap topological sort.
  std::vector<std::vector<OpId>> successors(num_ops);
  std::vector<int> indegree(num_ops, 0);
  for (OpId u = 0; u < static_cast<OpId>(num_ops); ++u) {
    const auto& op = instance_.operations[u];
    std::vector<OpId> preds;
    for (const auto& inp : op.inputs) {
      const ValueId vid = value_by_name_.at(inp);
      if (values_[vid].producer_op.has_value()) {
        preds.push_back(*values_[vid].producer_op);
      }
    }
    std::sort(preds.begin(), preds.end());
    preds.erase(std::unique(preds.begin(), preds.end()), preds.end());
    indegree[u] = static_cast<int>(preds.size());
    for (OpId p : preds) {
      successors[p].push_back(u);
    }
  }
  for (auto& succs : successors) {
    std::sort(succs.begin(), succs.end());
  }

  std::priority_queue<OpId, std::vector<OpId>, std::greater<>> ready;
  for (OpId u = 0; u < static_cast<OpId>(num_ops); ++u) {
    if (indegree[u] == 0) {
      ready.push(u);
    }
  }
  topo_order_.reserve(num_ops);
  while (!ready.empty()) {
    const OpId u = ready.top();
    ready.pop();
    topo_order_.push_back(u);
    for (OpId succ : successors[u]) {
      if (--indegree[succ] == 0) {
        ready.push(succ);
      }
    }
  }
  if (topo_order_.size() != num_ops) {
    throw pypto::ValueError(
        "Layout solver requires an acyclic single-producer graph; found a dependency cycle");
  }

  // Index conversion factors.
  for (ValueId v = 0; v < static_cast<ValueId>(values_.size()); ++v) {
    ValueInfo& v_info = values_[v];
    const ValueNode& node = instance_.values.at(v_info.name);
    const int default_idx = v_info.FindLayoutIndex(node.current_layout);
    v_info.default_source_idx = default_idx >= 0 ? static_cast<uint16_t>(default_idx) : 0;

    std::vector<uint16_t> source_indices;
    if (v_info.producer_op.has_value()) {
      const OpInfo& prod = ops_[*v_info.producer_op];
      for (uint16_t idx : prod.output_layout_indices) {
        if (std::find(source_indices.begin(), source_indices.end(), idx) == source_indices.end()) {
          source_indices.push_back(idx);
        }
      }
    } else {
      source_indices.push_back(v_info.default_source_idx);
    }

    std::set<int64_t> conv_invs_set;
    std::set<int64_t> compat_conv_invs_set;
    std::vector<uint16_t> target_indices;
    uint64_t incompatible_target_mask = 0;
    auto add_target = [&](uint16_t idx) {
      if (std::find(target_indices.begin(), target_indices.end(), idx) == target_indices.end()) {
        target_indices.push_back(idx);
      }
    };

    const bool has_onchip_placement =
        node.memory_space != MemorySpace::DDR && (node.conversion_invocations || v_info.producer_op);
    const int64_t prod_inv =
        node.conversion_invocations.value_or(v_info.producer_op ? ops_[*v_info.producer_op].invocations : 0);

    for (OpId p : v_info.participants) {
      const OpInfo& op_info = ops_[p];
      const bool inv_compat =
          !has_onchip_placement || CanShareOnChipConversion(prod_inv, op_info.invocations);
      for (const auto& uses_vec : op_info.config_uses) {
        for (const auto& use : uses_vec) {
          if (use.value_id != v) {
            continue;
          }
          bool any_diff = false;
          for (const auto& [lay_idx, _] : use.layout_counts) {
            add_target(lay_idx);
            bool diff_for_lay = false;
            for (uint16_t src_idx : source_indices) {
              if (src_idx != lay_idx) {
                diff_for_lay = true;
                any_diff = true;
                break;
              }
            }
            if (diff_for_lay && !inv_compat) {
              incompatible_target_mask |= (uint64_t{1} << lay_idx);
            }
          }
          if (any_diff) {
            conv_invs_set.insert(op_info.invocations);
            if (inv_compat) {
              compat_conv_invs_set.insert(op_info.invocations);
            }
          }
        }
      }
    }

    const size_t num_layouts = v_info.layouts.size();
    v_info.boundary_counts.assign(num_layouts, 0);
    if (node.is_external_output && !instance_.cost_model.allow_external_output_relayout) {
      const uint16_t b_idx = v_info.default_source_idx;
      add_target(b_idx);
      v_info.boundary_demand_mask |= (uint64_t{1} << b_idx);
      v_info.boundary_counts[b_idx] = 1;
      for (uint16_t src_idx : source_indices) {
        if (src_idx != b_idx) {
          conv_invs_set.insert(instance_.cost_model.expected_uses);
          const bool b_compat =
              !has_onchip_placement || CanShareOnChipConversion(prod_inv, instance_.cost_model.expected_uses);
          if (b_compat) {
            compat_conv_invs_set.insert(instance_.cost_model.expected_uses);
          } else {
            incompatible_target_mask |= (uint64_t{1} << b_idx);
          }
          break;
        }
      }
    }

    v_info.forbid_multi_rep = instance_.forbid_unmaterializable_gm_conversions &&
                              !node.producer_op_id.has_value() && node.memory_space == MemorySpace::DDR;

    bool has_conversion_free_layout = false;
    if (incompatible_target_mask != 0) {
      for (uint16_t s_idx : source_indices) {
        if (node.is_external_output && !instance_.cost_model.allow_external_output_relayout &&
            s_idx != v_info.default_source_idx) {
          continue;
        }
        bool ok_all_consumers = true;
        for (OpId p : v_info.participants) {
          const OpInfo& op_info = ops_[p];
          bool op_consumes_v = false;
          bool op_has_compat_cfg = false;
          for (const auto& uses_vec : op_info.config_uses) {
            for (const auto& use : uses_vec) {
              if (use.value_id == v) {
                op_consumes_v = true;
                if (use.demand_mask == (uint64_t{1} << s_idx)) {
                  op_has_compat_cfg = true;
                }
              }
            }
          }
          if (op_consumes_v && !op_has_compat_cfg) {
            ok_all_consumers = false;
            break;
          }
        }
        if (ok_all_consumers) {
          has_conversion_free_layout = true;
          break;
        }
      }
    }

    v_info.incompatible_target_mask = incompatible_target_mask;

    const std::vector<int64_t> conv_invs =
        (incompatible_target_mask != 0 && has_conversion_free_layout)
            ? std::vector<int64_t>(compat_conv_invs_set.begin(), compat_conv_invs_set.end())
            : std::vector<int64_t>(conv_invs_set.begin(), conv_invs_set.end());
    const double mult = ConversionMultiplier(instance_, node, op_ptr_by_id, conv_invs);
    v_info.conv_costs.assign(num_layouts * num_layouts, 0.0);
    const bool check_writeback = instance_.forbid_unmaterializable_gm_conversions;
    const bool check_conversion = check_writeback || !node.blocked_vec_conversions.empty();
    if (check_conversion) v_info.unsupported_target_masks.resize(num_layouts);
    const bool external_input = !node.producer_op_id && node.memory_space == MemorySpace::DDR;
    const bool producer_ok =
        !v_info.producer_op || SupportsVecConversionPlacement(ops_[*v_info.producer_op].op_name);
    for (uint16_t src_idx : source_indices) {
      for (uint16_t dst_idx : target_indices) {
        if (src_idx == dst_idx) {
          continue;
        }
        const uint64_t dst_bit = uint64_t{1} << dst_idx;
        if (check_conversion) {
          const auto& src = v_info.layouts[src_idx];
          const auto& dst = v_info.layouts[dst_idx];
          // An external output's declared layout may also be requested by a
          // consumer. Leave that ambiguity to the selected-consumer check.
          const bool boundary_layout = node.is_external_output && dst == node.current_layout;
          const bool supports_conversion = node.memory_space == MemorySpace::Vec && producer_ok &&
                                           (incompatible_target_mask & dst_bit) == 0 &&
                                           instance_.cost_model.SupportsVecTileConversion(node, src, dst);
          const bool blocked =
              std::find(node.blocked_vec_conversions.begin(), node.blocked_vec_conversions.end(),
                        std::make_pair(src, dst)) != node.blocked_vec_conversions.end();
          if (blocked || (check_writeback && !external_input && !boundary_layout && !supports_conversion)) {
            v_info.unsupported_target_masks[src_idx] |= dst_bit;
          }
        }
        if ((incompatible_target_mask & dst_bit) != 0 ||
            (mult == 0 && node.memory_space == MemorySpace::Vec &&
             !instance_.cost_model.SupportsVecTileConversion(node, v_info.layouts[src_idx],
                                                             v_info.layouts[dst_idx]))) {
          v_info.conv_costs[static_cast<size_t>(src_idx) * num_layouts + dst_idx] =
              instance_.cost_model.vec_unsupported_conversion_cycles;
          continue;
        }
        const double c = RequireFiniteNonNegative(
            mult * instance_.EstimateConversionCost(node, v_info.layouts[src_idx], v_info.layouts[dst_idx]),
            "Conversion cost for '" + v_info.name + "'");
        v_info.conv_costs[static_cast<size_t>(src_idx) * num_layouts + dst_idx] = c;
      }
    }
    v_info.max_conversion_cost = *std::max_element(v_info.conv_costs.begin(), v_info.conv_costs.end());
  }
}

std::optional<OpId> IndexedProblem::FindOpById(const std::string& op_id) const {
  auto it = op_by_id_.find(op_id);
  if (it == op_by_id_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<ValueId> IndexedProblem::FindValueByName(const std::string& name) const {
  auto it = value_by_name_.find(name);
  if (it == value_by_name_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<uint16_t> IndexedProblem::InitialChoices(const LayoutAssignment& initial) const {
  for (const auto& [op_id, _] : initial.op_configs) {
    if (op_by_id_.count(op_id) == 0) {
      throw pypto::ValueError("Initial assignment references unknown operation '" + op_id + "'");
    }
  }
  if (initial.op_configs.size() != ops_.size()) {
    throw pypto::ValueError("Initial assignment needs exactly one configuration per operation");
  }
  for (const auto& [val_name, _] : initial.value_layouts) {
    if (value_by_name_.count(val_name) == 0) {
      throw pypto::ValueError("Initial assignment references unknown value '" + val_name + "'");
    }
  }

  std::vector<uint16_t> choices;
  choices.reserve(ops_.size());
  for (size_t i = 0; i < ops_.size(); ++i) {
    const auto& op = instance_.operations[i];
    const OperatorConfig& cfg = initial.op_configs.at(op.op_id);
    auto match_it = std::find(op.configs.begin(), op.configs.end(), cfg);
    if (match_it == op.configs.end()) {
      throw pypto::ValueError("Initial configuration of '" + op.op_id + "' is not in its legal catalog");
    }
    auto val_it = initial.value_layouts.find(op.output);
    if (val_it != initial.value_layouts.end() && val_it->second != cfg.output_layout) {
      throw pypto::ValueError("Initial value layout for '" + op.output +
                              "' contradicts its producer configuration");
    }
    choices.push_back(static_cast<uint16_t>(std::distance(op.configs.begin(), match_it)));
  }
  return choices;
}

IncrementalState::IncrementalState(const IndexedProblem& problem, std::vector<uint16_t> choices)
    : problem_(problem), choices_(std::move(choices)) {
  const size_t num_vals = problem_.NumValues();
  const size_t num_ops = problem_.NumOps();
  sources_.resize(num_vals);
  demand_masks_.resize(num_vals);
  counts_.resize(num_vals);
  source_costs_.resize(num_vals);
  source_cost_masks_.resize(num_vals);

  for (ValueId v = 0; v < num_vals; ++v) {
    const auto& v_info = problem_.Values()[v];
    sources_[v] = v_info.default_source_idx;
    demand_masks_[v] = v_info.boundary_demand_mask;
    counts_[v] = v_info.boundary_counts;
  }

  for (OpId u = 0; u < num_ops; ++u) {
    const auto& op = problem_.Ops()[u];
    const uint16_t q = choices_[u];
    sources_[op.output_value] = op.output_layout_indices[q];
    for (const auto& use : op.config_uses[q]) {
      const ValueId v = use.value_id;
      demand_masks_[v] |= use.demand_mask;
      for (const auto& [lay_idx, cnt] : use.layout_counts) {
        counts_[v][lay_idx] += cnt;
      }
    }
  }

  for (OpId u = 0; u < num_ops; ++u) {
    const auto& op = problem_.Ops()[u];
    cost_.Add(op.costs[choices_[u]]);
    cost_.violations += op.incompatible_configs[choices_[u]];
  }
  for (ValueId v = 0; v < num_vals; ++v) {
    source_costs_[v].resize(problem_.Values()[v].layouts.size());
    AddFactorCostDelta(cost_, v, sources_[v], 0, sources_[v], demand_masks_[v]);
  }
}

void IncrementalState::CostSum::Add(double value) {
  const double next = sum + value;
  // Neumaier summation retains small terms while an unsupported layout penalty is active.
  correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
  sum = next;
}

void IncrementalState::CostSum::Replace(double before, double after) {
  if (before != after) {
    Add(-before);
    Add(after);
  }
}

void IncrementalState::AddFactorCostDelta(CostSum& cost, ValueId value, uint16_t old_source,
                                          uint64_t old_mask, uint16_t new_source, uint64_t new_mask) const {
  uint64_t changed = old_source == new_source ? old_mask ^ new_mask : old_mask | new_mask;
  if (changed == 0) return;
  const auto& info = problem_.Values()[value];
  cost.violations +=
      FactorViolations(info, new_source, new_mask) - FactorViolations(info, old_source, old_mask);
  const size_t layout_count = info.layouts.size();
  const double* old_row = info.conv_costs.data() + old_source * layout_count;
  const double* new_row = info.conv_costs.data() + new_source * layout_count;
  while (changed != 0) {
    const auto target = __builtin_ctzll(changed);
    const uint64_t bit = uint64_t{1} << target;
    cost.Replace((old_mask & bit) ? old_row[target] : 0.0, (new_mask & bit) ? new_row[target] : 0.0);
    changed &= changed - 1;
  }
  if (info.forbid_multi_rep) {
    const double penalty = problem_.Instance().cost_model.vec_unsupported_conversion_cycles;
    cost.Replace(HasMultipleRepresentations(old_mask) ? penalty : 0.0,
                 HasMultipleRepresentations(new_mask) ? penalty : 0.0);
  }
}

double IncrementalState::Gain(OpId op_idx, uint16_t new_choice, std::optional<ValueId> omitted_factor) const {
  return GainScore(op_idx, new_choice, omitted_factor).cycles;
}

LayoutSearchScore IncrementalState::GainScore(OpId op_idx, uint16_t new_choice,
                                              std::optional<ValueId> omitted_factor) const {
  const auto& op = problem_.Ops()[op_idx];
  const uint16_t old_choice = choices_[op_idx];
  if (old_choice == new_choice) {
    return {};
  }
  CostSum delta;
  delta.Replace(op.costs[old_choice], op.costs[new_choice]);
  delta.violations += op.incompatible_configs[new_choice] - op.incompatible_configs[old_choice];

  const auto& old_uses = op.config_uses[old_choice];
  const auto& new_uses = op.config_uses[new_choice];

  auto add_value_delta = [&](ValueId v, const IndexedProblem::OpConfigUse* old_u,
                             const IndexedProblem::OpConfigUse* new_u) {
    if (v == omitted_factor) return;
    const uint16_t new_src = (v == op.output_value) ? op.output_layout_indices[new_choice] : sources_[v];

    uint64_t new_mask = demand_masks_[v];
    if (old_u != nullptr) {
      for (const auto& [lay_idx, cnt] : old_u->layout_counts) {
        int net = counts_[v][lay_idx] - cnt;
        if (new_u != nullptr) {
          for (const auto& [nl, nc] : new_u->layout_counts) {
            if (nl == lay_idx) {
              net += nc;
              break;
            }
          }
        }
        if (net > 0) {
          new_mask |= (uint64_t{1} << lay_idx);
        } else {
          new_mask &= ~(uint64_t{1} << lay_idx);
        }
      }
    }
    if (new_u != nullptr) {
      for (const auto& [lay_idx, _] : new_u->layout_counts) {
        // Any layout present in new_u has net > 0 unless subtracted above (which already handled it).
        bool in_old = false;
        if (old_u != nullptr) {
          for (const auto& [ol, _oc] : old_u->layout_counts) {
            if (ol == lay_idx) {
              in_old = true;
              break;
            }
          }
        }
        if (!in_old) {
          new_mask |= (uint64_t{1} << lay_idx);
        }
      }
    }

    AddFactorCostDelta(delta, v, sources_[v], demand_masks_[v], new_src, new_mask);
  };

  bool out_seen = false;
  for (const auto& old_u : old_uses) {
    if (old_u.value_id == op.output_value) {
      out_seen = true;
    }
    const IndexedProblem::OpConfigUse* match_new = nullptr;
    for (const auto& new_u : new_uses) {
      if (new_u.value_id == old_u.value_id) {
        match_new = &new_u;
        break;
      }
    }
    add_value_delta(old_u.value_id, &old_u, match_new);
  }
  for (const auto& new_u : new_uses) {
    bool in_old = false;
    for (const auto& old_u : old_uses) {
      if (old_u.value_id == new_u.value_id) {
        in_old = true;
        break;
      }
    }
    if (!in_old) {
      if (new_u.value_id == op.output_value) {
        out_seen = true;
      }
      add_value_delta(new_u.value_id, nullptr, &new_u);
    }
  }
  if (!out_seen) {
    add_value_delta(op.output_value, nullptr, nullptr);
  }

  return {-delta.violations, -delta.Value()};
}

void IncrementalState::Apply(OpId op_idx, uint16_t new_choice) {
  const uint16_t old_choice = choices_[op_idx];
  if (old_choice == new_choice) {
    return;
  }
  const auto& op = problem_.Ops()[op_idx];

  const auto& old_uses = op.config_uses[old_choice];
  const auto& new_uses = op.config_uses[new_choice];
  for (size_t i = 0; i < old_uses.size(); ++i) {
    const auto value = old_uses[i].value_id;
    const auto old_mask = demand_masks_[value];
    ForEachDemandDelta(old_uses[i], new_uses[i],
                       [&](uint16_t layout, int delta) { UpdateDemandCount(value, layout, delta); });
    CommitFactor(value, old_mask, sources_[value]);
  }

  CommitFactor(op.output_value, demand_masks_[op.output_value], op.output_layout_indices[new_choice]);
  cost_.Replace(op.costs[old_choice], op.costs[new_choice]);
  cost_.violations += op.incompatible_configs[new_choice] - op.incompatible_configs[old_choice];
  choices_[op_idx] = new_choice;
}

IncrementalState::BlockEvaluation IncrementalState::EvaluateBlock(
    const std::vector<ChoiceChange>& choices) const {
  // Keep factors in first-touch order so floating-point summation follows a
  // deterministic order. Each factor sees every consumer and source change.
  BlockEvaluation evaluation;
  auto& factors = evaluation.factors;
  std::unordered_map<ValueId, size_t> factor_positions;
  const auto get_factor = [&](ValueId value) -> BlockEvaluation::Factor& {
    auto [it, inserted] = factor_positions.try_emplace(value, factors.size());
    if (inserted) factors.push_back({value, sources_[value], counts_[value]});
    return factors[it->second];
  };

  CostSum delta;
  for (const auto& [op_idx, new_choice] : choices) {
    const auto& op = problem_.Ops()[op_idx];
    const auto old_choice = choices_[op_idx];
    delta.Replace(op.costs[old_choice], op.costs[new_choice]);
    delta.violations += op.incompatible_configs[new_choice] - op.incompatible_configs[old_choice];
    const uint16_t new_source = op.output_layout_indices[new_choice];
    if (new_source != sources_[op.output_value]) get_factor(op.output_value).source = new_source;

    const auto& old_uses = op.config_uses[old_choice];
    const auto& new_uses = op.config_uses[new_choice];
    for (size_t i = 0; i < old_uses.size(); ++i) {
      ForEachDemandDelta(old_uses[i], new_uses[i], [&](uint16_t layout, int count_delta) {
        get_factor(old_uses[i].value_id).counts[layout] += count_delta;
      });
    }
  }
  for (auto& factor : factors) {
    for (size_t layout = 0; layout < factor.counts.size(); ++layout) {
      if (factor.counts[layout] > 0) factor.demand_mask |= uint64_t{1} << layout;
    }
    AddFactorCostDelta(delta, factor.value, sources_[factor.value], demand_masks_[factor.value],
                       factor.source, factor.demand_mask);
  }
  evaluation.gain = -delta.Value();
  evaluation.violation_reduction = -delta.violations;
  return evaluation;
}

void IncrementalState::ApplyBlock(const std::vector<ChoiceChange>& choices,
                                  const BlockEvaluation& evaluation) {
  for (const auto& factor : evaluation.factors) {
    const auto old_mask = demand_masks_[factor.value];
    counts_[factor.value] = factor.counts;
    demand_masks_[factor.value] = factor.demand_mask;
    CommitFactor(factor.value, old_mask, factor.source);
  }
  for (const auto& [op_idx, new_choice] : choices) {
    const auto& op = problem_.Ops()[op_idx];
    cost_.Replace(op.costs[choices_[op_idx]], op.costs[new_choice]);
    cost_.violations += op.incompatible_configs[new_choice] - op.incompatible_configs[choices_[op_idx]];
    choices_[op_idx] = new_choice;
  }
}

bool IncrementalState::ApplyImprovingBlock(const std::vector<ChoiceChange>& choices, double tolerance) {
  const auto evaluation = EvaluateBlock(choices);
  if (!BetterGain({evaluation.violation_reduction, evaluation.gain}, {}, tolerance)) return false;
  ApplyBlock(choices, evaluation);
  return true;
}

void IncrementalState::UpdateDemandCount(ValueId value, uint16_t layout, int delta) {
  const int previous = counts_[value][layout];
  const int next = previous + delta;
  counts_[value][layout] = next;
  const int support_delta = static_cast<int>(next > 0) - static_cast<int>(previous > 0);
  if (support_delta == 0) return;
  if (next > 0) {
    demand_masks_[value] |= uint64_t{1} << layout;
  } else {
    demand_masks_[value] &= ~(uint64_t{1} << layout);
  }
}

const std::vector<std::vector<double>>& IncrementalState::SourceCosts() const {
  for (ValueId value = 0; value < problem_.NumValues(); ++value) {
    if (source_cost_masks_[value] != demand_masks_[value]) RebuildSourceCosts(value);
  }
  return source_costs_;
}

void IncrementalState::RebuildSourceCosts(ValueId value) const {
  const auto& info = problem_.Values()[value];
  const size_t layout_count = info.layouts.size();
  for (size_t source = 0; source < layout_count; ++source) {
    const double* row = info.conv_costs.data() + source * layout_count;
    uint64_t targets = demand_masks_[value];
    CostSum cost;
    while (targets != 0) {
      cost.Add(row[__builtin_ctzll(targets)]);
      targets &= targets - 1;
    }
    source_costs_[value][source] = cost.Value();
  }
  source_cost_masks_[value] = demand_masks_[value];
}

void IncrementalState::CommitFactor(ValueId value, uint64_t old_mask, uint16_t new_source) {
  const auto new_mask = demand_masks_[value];
  AddFactorCostDelta(cost_, value, sources_[value], old_mask, new_source, new_mask);
  sources_[value] = new_source;
}

LayoutAssignment IncrementalState::ToAssignment() const {
  LayoutAssignment assignment;
  const auto& instance = problem_.Instance();
  for (ValueId v = 0; v < problem_.NumValues(); ++v) {
    const auto& v_info = problem_.Values()[v];
    const ValueNode* node = instance.FindValue(v_info.name);
    LayoutSpec chosen = v_info.layouts[sources_[v]];
    if (node != nullptr && !node->producer_op_id.has_value() && node->memory_space == MemorySpace::DDR) {
      if (__builtin_popcountll(demand_masks_[v]) == 1) {
        const int dst_idx = __builtin_ctzll(demand_masks_[v]);
        chosen = v_info.layouts[static_cast<size_t>(dst_idx)];
      }
    }
    assignment.value_layouts.emplace(v_info.name, std::move(chosen));
  }
  for (OpId u = 0; u < problem_.NumOps(); ++u) {
    const auto& op = instance.operations[u];
    assignment.op_configs.emplace(op.op_id, op.configs[choices_[u]]);
  }
  return assignment;
}

TreewidthStats ComputeTreewidth(const LayoutProblemInstance& instance) {
  IndexedProblem problem(instance);
  return ComputeProblemTreewidthInternal(
      problem, ComputeActiveDecomposition(problem, std::numeric_limits<int64_t>::max()));
}

std::vector<std::string> CheckVecConversionCapacity(const LayoutProblemInstance& instance,
                                                    const LayoutAssignment& assignment) {
  return FindVecCapacityConflicts(instance, assignment).issues;
}

static LayoutSolverResult SolveLocalSearch(const LayoutProblemInstance& instance,
                                           const std::optional<LayoutAssignment>& initial,
                                           const LayoutSolverConfig& config, bool refine_with_fm) {
  config.Validate();
  IndexedProblem problem(instance);
  int64_t evaluated_moves = 0;
  int64_t moves_applied = 0;
  int64_t greedy_sweeps = 0;
  std::string termination_reason = "local_optimum";

  std::vector<uint16_t> choices;
  if (initial.has_value()) {
    choices = problem.InitialChoices(*initial);
  } else {
    auto seed = TopologicalSeed(problem, config, evaluated_moves, termination_reason);
    choices.assign(seed.begin(), seed.end());
  }

  IncrementalState state(problem, std::move(choices));
  const double initial_cost = state.Cost();
  if (termination_reason != "evaluation_budget") {
    if (!refine_with_fm) {
      GreedyRefine(state, config, evaluated_moves, moves_applied, greedy_sweeps, termination_reason);
    } else {
      const auto graph = ComputeActiveDecomposition(problem, std::nullopt);
      std::vector<size_t> components(graph.components.size());
      std::iota(components.begin(), components.end(), size_t{0});
      GreedyIncumbentAndFM(state, graph, components, config, evaluated_moves, moves_applied, greedy_sweeps,
                           termination_reason);
    }
  }
  return BuildResult(state, evaluated_moves, moves_applied, greedy_sweeps, initial_cost, termination_reason);
}

static LayoutSolverResult SolveTreewidthDPCore(const LayoutProblemInstance& instance,
                                               const LayoutSolverConfig& config) {
  config.Validate();
  IndexedProblem problem(instance);
  const auto decomp = ComputeActiveDecomposition(
      problem, config.collect_treewidth_stats ? std::numeric_limits<int64_t>::max() : config.max_treewidth);
  const auto stats = config.collect_treewidth_stats
                         ? std::optional{ComputeProblemTreewidthInternal(problem, decomp)}
                         : std::nullopt;

  if (problem.NumOps() == 0) {
    IncrementalState empty_state(problem, {});
    return BuildResult(empty_state, 0, 0, 0, 0.0, "optimal", stats, -1);
  }
  for (const auto& dec : decomp.component_decomps) {
    if (dec.treewidth > config.max_treewidth) {
      throw pypto::ValueError("Treewidth DP component exceeds max_treewidth=" +
                              std::to_string(config.max_treewidth));
    }
    if (dec.max_bag_states > config.max_bag_states) {
      throw pypto::ValueError("Treewidth DP exceeds max_bag_states=" + std::to_string(config.max_bag_states));
    }
  }

  int64_t evaluated_moves = 0;
  std::vector<uint16_t> choices;
  if (!RunTreewidthDPOnComponents(problem, decomp, config, evaluated_moves, choices)) {
    throw pypto::ValueError("Treewidth DP exceeded max_evaluations=" +
                            std::to_string(config.max_evaluations));
  }
  IncrementalState state(problem, std::move(choices));
  return BuildResult(state, evaluated_moves, 0, 0, state.Cost(), "optimal", stats, -1);
}

static LayoutSolverResult SolveAutoLayoutCore(const LayoutProblemInstance& instance,
                                              const std::optional<LayoutAssignment>& initial,
                                              const LayoutSolverConfig& config) {
  config.Validate();
  IndexedProblem problem(instance);
  const auto decomp = ComputeActiveDecomposition(
      problem, config.collect_treewidth_stats ? std::numeric_limits<int64_t>::max() : config.max_treewidth);
  const auto stats = config.collect_treewidth_stats
                         ? std::optional{ComputeProblemTreewidthInternal(problem, decomp)}
                         : std::nullopt;
  int64_t evaluated_moves = 0;

  if (problem.NumOps() == 0) {
    IncrementalState empty_state(problem, {});
    return BuildResult(empty_state, 0, 0, 0, 0.0, "optimal", stats, -1);
  }

  const size_t num_comps = decomp.component_decomps.size();
  std::vector<size_t> dp_comps;
  std::vector<size_t> fallback_comps;
  for (size_t c = 0; c < num_comps; ++c) {
    const auto& dec = decomp.component_decomps[c];
    if (dec.treewidth <= config.max_treewidth && dec.max_bag_states <= config.max_bag_states &&
        dec.total_states <= config.max_evaluations) {
      dp_comps.push_back(c);
    } else {
      fallback_comps.push_back(c);
    }
  }

  std::vector<uint16_t> dp_choices(problem.NumOps(), 0);
  int64_t dp_evals = 0;
  std::vector<size_t> solved_dp_comps;
  if (!dp_comps.empty()) {
    const DPContext dp_ctx = BuildDPContext(problem, decomp);
    for (size_t c : dp_comps) {
      if (RunTreewidthDPOnSingleComponent(problem, decomp, dp_ctx, c, config, dp_evals, dp_choices)) {
        solved_dp_comps.push_back(c);
      } else {
        fallback_comps.push_back(c);
      }
    }
  }

  if (fallback_comps.empty()) {
    IncrementalState state(problem, std::move(dp_choices));
    return BuildResult(state, dp_evals, 0, 0, state.Cost(), "optimal", stats, -1);
  }

  evaluated_moves = dp_evals;
  int64_t moves_applied = 0;
  int64_t greedy_sweeps = 0;
  std::string termination_reason = "local_optimum";
  std::vector<uint16_t> choices;
  if (initial.has_value()) {
    choices = problem.InitialChoices(*initial);
  } else {
    auto seed = TopologicalSeed(problem, config, evaluated_moves, termination_reason);
    choices.assign(seed.begin(), seed.end());
  }
  // Preserve exact DP solutions on every component that fit the DP budget.
  for (size_t c : solved_dp_comps) {
    for (OpId u : decomp.components[c]) {
      choices[u] = dp_choices[u];
    }
  }

  IncrementalState state(problem, std::move(choices));
  const double initial_cost = state.Cost();
  if (termination_reason != "evaluation_budget") {
    if (initial.has_value()) {
      std::vector<bool> in_fallback(problem.NumOps(), false);
      for (size_t c : fallback_comps) {
        for (OpId u : decomp.components[c]) in_fallback[u] = true;
      }
      std::vector<OpId> fallback_ops;
      for (OpId u : problem.TopoOrder()) {
        if (in_fallback[u]) fallback_ops.push_back(u);
      }
      GreedyRefineOps(state, fallback_ops, config, evaluated_moves, moves_applied, greedy_sweeps,
                      termination_reason);
    } else {
      GreedyIncumbentAndFM(state, decomp, fallback_comps, config, evaluated_moves, moves_applied,
                           greedy_sweeps, termination_reason);
    }
  }
  return BuildResult(state, evaluated_moves, moves_applied, greedy_sweeps, initial_cost, termination_reason,
                     stats, -1);
}

namespace {

std::optional<LayoutAssignment> DeclaredLayoutAssignment(const LayoutProblemInstance& instance) {
  LayoutAssignment assignment;
  for (const auto& op : instance.operations) {
    const auto config = std::find_if(op.configs.begin(), op.configs.end(), [&](const OperatorConfig& cfg) {
      if (cfg.output_layout != instance.values.at(op.output).current_layout) return false;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        if (cfg.input_layouts[i] != instance.values.at(op.inputs[i]).current_layout) return false;
      }
      return true;
    });
    if (config == op.configs.end()) return std::nullopt;
    assignment.op_configs.emplace(op.op_id, *config);
  }
  return assignment;
}

template <typename Solve>
LayoutSolverResult SolveWithVecCapacity(const LayoutProblemInstance& instance,
                                        const LayoutSolverConfig& config,
                                        const std::optional<LayoutAssignment>& initial, Solve solve) {
  auto result = solve(instance, config, false);
  const auto original_conflicts = FindVecCapacityConflicts(instance, result.Assignment());
  const bool capacity_conflict = !original_conflicts.issues.empty();
  if (!capacity_conflict &&
      (!instance.forbid_unmaterializable_gm_conversions || result.writeback_issues.empty())) {
    return result;
  }

  auto totals = result.stats;
  auto restore_original_score = [&](const LayoutAssignment& assignment, const std::string& reason) {
    IndexedProblem problem(instance);
    IncrementalState state(problem, problem.InitialChoices(assignment));
    return BuildResult(state, totals.evaluated_moves, totals.moves_applied, totals.greedy_sweeps,
                       totals.initial_cost_cycles, reason, result.treewidth_stats, result.stats.treewidth);
  };
  auto feasible_incumbent = [&]() -> std::optional<LayoutSolverResult> {
    std::optional<LayoutSolverResult> best;
    const auto consider = [&](const LayoutAssignment& assignment, const std::string& reason) {
      auto candidate = restore_original_score(assignment, reason);
      if (candidate.writeback_issues.empty() &&
          (!best || candidate.TotalCostCycles() < best->TotalCostCycles() - config.tolerance) &&
          CheckVecConversionCapacity(instance, candidate.Assignment()).empty()) {
        best = std::move(candidate);
      }
    };
    if (initial) consider(*initial, "feasible_initial_layouts");
    if (const auto declared = DeclaredLayoutAssignment(instance)) {
      consider(*declared, capacity_conflict ? "vec_capacity_declared_layouts" : "feasible_declared_layouts");
    }
    return best;
  };
  if (!capacity_conflict) {
    if (auto incumbent = feasible_incumbent()) return std::move(*incumbent);
    return result;
  }

  auto constrained = instance;
  auto retry_config = config;
  auto conflicts = original_conflicts;
  constexpr int kMaxCapacityRetries = 3;
  for (int attempt = 0; attempt < kMaxCapacityRetries; ++attempt) {
    retry_config.max_evaluations = config.max_evaluations - totals.evaluated_moves;
    if (retry_config.max_evaluations <= 0) break;
    bool changed = false;
    for (const auto& copy : conflicts.copies) {
      auto& blocked = constrained.values.at(copy.value->name).blocked_vec_conversions;
      const auto pair = std::make_pair(copy.source, copy.target);
      if (std::find(blocked.begin(), blocked.end(), pair) == blocked.end()) {
        blocked.push_back(pair);
        changed = true;
      }
    }
    for (const auto& [index, selected] : conflicts.operation_configs) {
      auto& configs = constrained.operations[index].configs;
      if (configs.size() <= 1) continue;
      if (auto it = std::find(configs.begin(), configs.end(), selected); it != configs.end()) {
        configs.erase(it);
        changed = true;
      }
    }
    if (!changed) break;
    auto candidate = solve(constrained, retry_config, true);
    totals.evaluated_moves += candidate.stats.evaluated_moves;
    totals.moves_applied += candidate.stats.moves_applied;
    totals.greedy_sweeps += candidate.stats.greedy_sweeps;
    conflicts = FindVecCapacityConflicts(instance, candidate.Assignment());
    if (conflicts.issues.empty()) {
      auto repaired = restore_original_score(candidate.Assignment(), "vec_capacity_repaired");
      if (repaired.writeback_issues.empty()) {
        if (initial) {
          auto incumbent = restore_original_score(*initial, "feasible_initial_layouts");
          if (incumbent.writeback_issues.empty() &&
              incumbent.TotalCostCycles() < repaired.TotalCostCycles() - config.tolerance &&
              CheckVecConversionCapacity(instance, incumbent.Assignment()).empty()) {
            return incumbent;
          }
        }
        return repaired;
      }
      break;
    }
  }

  if (auto incumbent = feasible_incumbent()) return std::move(*incumbent);
  totals.termination_reason = "vec_capacity_conflict";
  result.stats = std::move(totals);
  result.writeback_issues.insert(result.writeback_issues.end(), original_conflicts.issues.begin(),
                                 original_conflicts.issues.end());
  return result;
}

}  // namespace

LayoutSolverResult SolveGreedy(const LayoutProblemInstance& instance,
                               const std::optional<LayoutAssignment>& initial,
                               const LayoutSolverConfig& config) {
  return SolveWithVecCapacity(
      instance, config, initial, [&](const auto& constrained, const auto& limits, bool retry) {
        return SolveLocalSearch(constrained, retry ? std::nullopt : initial, limits, false);
      });
}

LayoutSolverResult SolveGreedyFM(const LayoutProblemInstance& instance, const LayoutSolverConfig& config) {
  return SolveWithVecCapacity(instance, config, std::nullopt,
                              [&](const auto& constrained, const auto& limits, bool) {
                                return SolveLocalSearch(constrained, std::nullopt, limits, true);
                              });
}

LayoutSolverResult SolveTreewidthDP(const LayoutProblemInstance& instance, const LayoutSolverConfig& config) {
  return SolveWithVecCapacity(instance, config, std::nullopt,
                              [&](const auto& constrained, const auto& limits, bool retry) {
                                if (retry) return SolveAutoLayoutCore(constrained, std::nullopt, limits);
                                return SolveTreewidthDPCore(constrained, limits);
                              });
}

LayoutSolverResult SolveAutoLayout(const LayoutProblemInstance& instance,
                                   const std::optional<LayoutAssignment>& initial,
                                   const LayoutSolverConfig& config) {
  return SolveWithVecCapacity(
      instance, config, initial, [&](const auto& constrained, const auto& limits, bool retry) {
        return SolveAutoLayoutCore(constrained, retry ? std::nullopt : initial, limits);
      });
}

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto
