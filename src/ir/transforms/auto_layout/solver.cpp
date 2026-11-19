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

  for (OpId op_idx : problem.TopoOrder()) {
    const auto& op = problem.Ops()[op_idx];
    uint16_t best_choice = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    const auto& out_info = problem.Values()[op.output_value];
    const size_t out_num_layouts = out_info.layouts.size();
    for (uint16_t q = 0; q < static_cast<uint16_t>(op.costs.size()); ++q) {
      if (evaluated_moves + 1 > config.max_evaluations) {
        termination_reason = "evaluation_budget";
        choices[op_idx] = best_choice;
        return std::vector<OpId>(choices.begin(), choices.end());
      }
      ++evaluated_moves;
      double cost = op.costs[q];
      const uint16_t out_lay = op.output_layout_indices[q];
      uint64_t out_mask = requested[op.output_value];
      while (out_mask != 0) {
        const int dst_idx = __builtin_ctzll(out_mask);
        if (static_cast<uint16_t>(dst_idx) != out_lay) {
          cost +=
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
        if (v_info.forbid_multi_rep && __builtin_popcountll(requested[v]) <= 1 &&
            __builtin_popcountll(requested[v] | use.demand_mask) > 1) {
          cost += problem.Instance().cost_model.vec_unsupported_conversion_cycles;
        }
        uint64_t new_bits = use.demand_mask & ~requested[v];
        while (new_bits != 0) {
          const int dst_idx = __builtin_ctzll(new_bits);
          cost +=
              v_info.conv_costs[static_cast<size_t>(src_idx) * num_layouts + static_cast<size_t>(dst_idx)];
          new_bits &= new_bits - 1;
        }
      }
      if (cost < best_cost - config.tolerance) {
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
      if (op.op_name == "control.if_join") {
        const auto& selected = op.configs[state.Choices()[producer]];
        // A join has no instruction that converts one branch to the result layout.
        if (std::any_of(selected.input_layouts.begin(), selected.input_layouts.end(),
                        [&](const LayoutSpec& input) { return input != selected.output_layout; })) {
          issues.push_back("Conditional join '" + op.op_id + "' has incompatible branch layouts");
        }
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
        const bool producer_ok = !v_info.producer_op.has_value() || StartsWithTile(prod_op_name) ||
                                 prod_op_name == "control.if_join";
        const bool consumers_ok =
            std::all_of(consumer_indices.begin(), consumer_indices.end(), [&](OpId idx) {
              const std::string& cname = problem.Ops()[idx].op_name;
              return StartsWithTile(cname) || cname == "control.if_join";
            });
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
                               int64_t active_tw = 0) {
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
  stats.treewidth = tw_stats.has_value() ? tw_stats->primal_treewidth : active_tw;
  stats.active_treewidth = tw_stats.has_value() ? tw_stats->active_primal_treewidth : active_tw;
  return LayoutSolverResult{std::move(breakdown), std::move(stats), std::move(conversions), std::move(issues),
                            tw_stats};
}

struct ActiveDecomposition {
  TreewidthStats tw_stats;
  std::vector<MinFillResult> component_decomps;
  std::vector<std::vector<OpId>> components;
};

TreewidthStats ComputeProblemTreewidthInternal(const IndexedProblem& problem) {
  const size_t num_ops = problem.NumOps();
  std::vector<OpId> all_ops(num_ops);
  std::iota(all_ops.begin(), all_ops.end(), 0);

  std::vector<OpId> movable;
  std::vector<bool> is_movable(num_ops, false);
  for (OpId u = 0; u < num_ops; ++u) {
    if (problem.Ops()[u].costs.size() > 1) {
      movable.push_back(u);
      is_movable[u] = true;
    }
  }

  std::vector<std::pair<OpId, OpId>> dataflow_edges;
  std::vector<std::pair<OpId, OpId>> primal_edges;
  std::vector<std::pair<OpId, OpId>> active_edges;

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
        if (is_movable[u] && is_movable[w]) {
          active_edges.emplace_back(u, w);
        }
      }
    }
  }

  auto df_res = MinFillDecomposition(all_ops, dataflow_edges);
  auto primal_res = MinFillDecomposition(all_ops, primal_edges);
  auto active_res = MinFillDecomposition(movable, active_edges);

  int64_t max_bag_size = 0;
  for (const auto& bag : primal_res.bags) {
    max_bag_size = std::max(max_bag_size, static_cast<int64_t>(bag.size()));
  }
  int64_t max_active_bag_size = 0;
  int64_t max_prod = num_ops > 0 ? 1 : 0;
  int64_t total_states = 0;
  if (!active_res.bags.empty()) {
    max_prod = 0;
    for (const auto& bag : active_res.bags) {
      max_active_bag_size = std::max(max_active_bag_size, static_cast<int64_t>(bag.size()));
      int64_t prod = 1;
      for (OpId u : bag) {
        const int64_t dom = static_cast<int64_t>(problem.Ops()[u].costs.size());
        if (prod > std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, dom)) {
          prod = std::numeric_limits<int64_t>::max();
          break;
        }
        prod *= dom;
      }
      max_prod = std::max(max_prod, prod);
      if (total_states > std::numeric_limits<int64_t>::max() - prod) {
        total_states = std::numeric_limits<int64_t>::max();
      } else {
        total_states += prod;
      }
    }
  } else {
    total_states = num_ops > 0 ? 1 : 0;
  }

  TreewidthStats stats;
  stats.num_values = static_cast<int64_t>(problem.NumValues());
  stats.num_operations = static_cast<int64_t>(num_ops);
  stats.num_movable_operations = static_cast<int64_t>(movable.size());
  stats.dataflow_treewidth = df_res.treewidth;
  stats.primal_treewidth = primal_res.treewidth;
  stats.active_primal_treewidth = active_res.treewidth;
  stats.max_bag_size = max_bag_size;
  stats.max_active_bag_size = max_active_bag_size;
  stats.max_bag_domain_product = max_prod;
  stats.total_dp_states = total_states;
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

ActiveDecomposition ComputeActiveDecomposition(const IndexedProblem& problem) {
  ActiveDecomposition out;
  const size_t num_ops = problem.NumOps();
  std::vector<bool> is_movable(num_ops, false);
  std::vector<OpId> movable;
  for (OpId u = 0; u < num_ops; ++u) {
    if (problem.Ops()[u].costs.size() > 1) {
      is_movable[u] = true;
      movable.push_back(u);
    }
  }

  // Build active primal adjacency for connected-component splitting.
  std::vector<std::vector<OpId>> adj(num_ops);
  for (const auto& v_info : problem.Values()) {
    std::vector<OpId> act;
    for (OpId u : v_info.participants) {
      if (is_movable[u]) {
        act.push_back(u);
      }
    }
    for (size_t i = 0; i < act.size(); ++i) {
      for (size_t j = i + 1; j < act.size(); ++j) {
        adj[act[i]].push_back(act[j]);
        adj[act[j]].push_back(act[i]);
      }
    }
  }

  std::vector<bool> visited(num_ops, false);
  for (OpId start : movable) {
    if (visited[start]) {
      continue;
    }
    std::vector<OpId> comp;
    std::queue<OpId> q;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
      const OpId u = q.front();
      q.pop();
      comp.push_back(u);
      for (OpId w : adj[u]) {
        if (!visited[w]) {
          visited[w] = true;
          q.push(w);
        }
      }
    }
    std::sort(comp.begin(), comp.end());
    std::vector<std::pair<OpId, OpId>> comp_edges;
    for (OpId u : comp) {
      for (OpId w : adj[u]) {
        if (u < w) {
          comp_edges.emplace_back(u, w);
        }
      }
    }
    out.component_decomps.push_back(MinFillDecomposition(comp, comp_edges));
    out.components.push_back(std::move(comp));
  }

  out.tw_stats = ComputeProblemTreewidthInternal(problem);
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

  std::vector<std::vector<double>> messages(num_bags);
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

    messages[s].assign(sep_size, std::numeric_limits<double>::infinity());
    backpointers[s].assign(sep_size, 0);
    sep_q.assign(sep_len, 0);
    bag_q.assign(bag.size(), 0);

    const auto& elim_costs = problem.Ops()[elim_u].costs;
    for (size_t sep_idx = 0; sep_idx < sep_size; ++sep_idx) {
      for (size_t j = 0; j < sep_len; ++j) {
        bag_q[j + 1] = sep_q[j];
      }
      double best_val = std::numeric_limits<double>::infinity();
      uint16_t best_q = 0;
      for (uint16_t q_elim = 0; q_elim < static_cast<uint16_t>(elim_dom); ++q_elim) {
        if (evaluated_moves + 1 > config.max_evaluations) {
          return false;
        }
        ++evaluated_moves;
        bag_q[0] = q_elim;
        double val = elim_costs[q_elim];

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
          if (v_info.forbid_multi_rep && __builtin_popcountll(mask) > 1) {
            val += problem.Instance().cost_model.vec_unsupported_conversion_cycles;
          }
          const double* row = v_info.conv_costs.data() + static_cast<size_t>(src_idx) * num_layouts;
          while (mask != 0) {
            const int dst = __builtin_ctzll(mask);
            val += row[dst];
            mask &= mask - 1;
          }
        }

        for (const auto& cd : child_descs) {
          size_t c_idx = static_cast<size_t>(q_elim) * cd.elim_stride;
          for (size_t j = 0; j < sep_len; ++j) {
            c_idx += static_cast<size_t>(sep_q[j]) * cd.sep_strides[j];
          }
          val += messages[cd.child_step][c_idx];
        }

        if (val < best_val - config.tolerance) {
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
      double best_gain = 0.0;
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
        const double gain = state.Gain(op_idx, q);
        if (!found || gain > best_gain + config.tolerance) {
          best_gain = gain;
          best_choice = q;
          found = true;
        }
      }
      if (found && best_gain > config.tolerance) {
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

void FMRefineComponent(IncrementalState& state, const std::vector<OpId>& comp_ops,
                       const LayoutSolverConfig& config, int64_t& evaluated_moves, int64_t& moves_applied,
                       int64_t& greedy_sweeps, std::string& termination_reason) {
  if (comp_ops.size() <= 1) {
    return;
  }
  const auto& problem = state.Problem();
  const size_t n = comp_ops.size();
  std::vector<bool> locked(n, false);
  struct AppliedMove {
    OpId op_idx;
    uint16_t old_choice;
  };
  std::vector<AppliedMove> history;
  history.reserve(n);

  for (int64_t pass = 0; pass < config.max_greedy_sweeps; ++pass) {
    ++greedy_sweeps;
    std::fill(locked.begin(), locked.end(), false);
    history.clear();
    double best_prefix_cost = state.Cost();
    size_t best_prefix_len = 0;
    bool budget_hit = false;

    for (size_t step = 0; step < n; ++step) {
      size_t best_local = n;
      uint16_t best_q = 0;
      double best_gain = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < n; ++i) {
        if (locked[i]) {
          continue;
        }
        const OpId op_idx = comp_ops[i];
        const size_t num_cfg = problem.Ops()[op_idx].costs.size();
        const uint16_t cur_q = state.Choices()[op_idx];
        for (uint16_t q = 0; q < static_cast<uint16_t>(num_cfg); ++q) {
          if (q == cur_q) {
            continue;
          }
          if (evaluated_moves + 1 > config.max_evaluations) {
            budget_hit = true;
            break;
          }
          ++evaluated_moves;
          const double g = state.Gain(op_idx, q);
          if (best_local == n || g > best_gain + config.tolerance) {
            best_gain = g;
            best_local = i;
            best_q = q;
          }
        }
        if (budget_hit) {
          break;
        }
      }
      if (budget_hit || best_local == n) {
        break;
      }
      const OpId chosen_op = comp_ops[best_local];
      const uint16_t old_q = state.Choices()[chosen_op];
      state.Apply(chosen_op, best_q);
      locked[best_local] = true;
      history.push_back(AppliedMove{chosen_op, old_q});
      const double cur_cost = state.Cost();
      if (cur_cost < best_prefix_cost - config.tolerance) {
        best_prefix_cost = cur_cost;
        best_prefix_len = history.size();
      }
    }

    for (size_t idx = history.size(); idx > best_prefix_len; --idx) {
      const auto& mv = history[idx - 1];
      state.Apply(mv.op_idx, mv.old_choice);
    }
    if (budget_hit) {
      termination_reason = "evaluation_budget";
      return;
    }
    if (best_prefix_len == 0) {
      termination_reason = "local_optimum";
      return;
    }
    moves_applied += static_cast<int64_t>(best_prefix_len);
  }
  termination_reason = "greedy_sweep_budget";
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
  if (instance_.cost_model.expected_uses < 1) {
    throw pypto::ValueError("expected_uses must be at least 1");
  }

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
    op_info.output_layout_indices.reserve(op.configs.size());
    op_info.config_uses.resize(op.configs.size());

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
      op_info.config_uses[q] = std::move(uses);
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

    const bool is_onchip_produced = node.memory_space != MemorySpace::DDR && v_info.producer_op.has_value();
    const int64_t prod_inv = is_onchip_produced ? ops_[*v_info.producer_op].invocations : 0;

    for (OpId p : v_info.participants) {
      const OpInfo& op_info = ops_[p];
      const bool inv_compat = !is_onchip_produced || (op_info.invocations >= prod_inv && prod_inv > 0 &&
                                                      (op_info.invocations % prod_inv) == 0);
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
              !is_onchip_produced || (instance_.cost_model.expected_uses >= prod_inv && prod_inv > 0 &&
                                      (instance_.cost_model.expected_uses % prod_inv) == 0);
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
    for (uint16_t src_idx : source_indices) {
      for (uint16_t dst_idx : target_indices) {
        if (src_idx == dst_idx) {
          continue;
        }
        const uint64_t dst_bit = uint64_t{1} << dst_idx;
        if ((incompatible_target_mask & dst_bit) != 0) {
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

  cost_ = 0.0;
  for (OpId u = 0; u < num_ops; ++u) {
    cost_ += problem_.Ops()[u].costs[choices_[u]];
  }
  for (ValueId v = 0; v < num_vals; ++v) {
    const auto& v_info = problem_.Values()[v];
    const size_t num_layouts = v_info.layouts.size();
    source_costs_[v].assign(num_layouts, 0.0);
    for (size_t s = 0; s < num_layouts; ++s) {
      double c = 0.0;
      uint64_t mask = demand_masks_[v];
      const double* row = v_info.conv_costs.data() + s * num_layouts;
      while (mask != 0) {
        const int dst = __builtin_ctzll(mask);
        c += row[dst];
        mask &= mask - 1;
      }
      source_costs_[v][s] = c;
    }
    cost_ += source_costs_[v][sources_[v]];
    if (v_info.forbid_multi_rep && __builtin_popcountll(demand_masks_[v]) > 1) {
      cost_ += problem_.Instance().cost_model.vec_unsupported_conversion_cycles;
    }
  }
}

double IncrementalState::Gain(OpId op_idx, uint16_t new_choice) const {
  const auto& op = problem_.Ops()[op_idx];
  const uint16_t old_choice = choices_[op_idx];
  if (old_choice == new_choice) {
    return 0.0;
  }
  double delta = op.costs[new_choice] - op.costs[old_choice];

  const auto& old_uses = op.config_uses[old_choice];
  const auto& new_uses = op.config_uses[new_choice];

  auto compute_val_delta = [&](ValueId v, const IndexedProblem::OpConfigUse* old_u,
                               const IndexedProblem::OpConfigUse* new_u) -> double {
    const auto& v_info = problem_.Values()[v];
    const size_t num_layouts = v_info.layouts.size();
    const uint16_t old_src = sources_[v];
    const uint16_t new_src = (v == op.output_value) ? op.output_layout_indices[new_choice] : old_src;

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

    const double multi_rep_penalty = problem_.Instance().cost_model.vec_unsupported_conversion_cycles;
    const double mr_delta = v_info.forbid_multi_rep
                                ? (static_cast<double>(__builtin_popcountll(new_mask) > 1) -
                                   static_cast<double>(__builtin_popcountll(demand_masks_[v]) > 1)) *
                                      multi_rep_penalty
                                : 0.0;

    if (new_mask == demand_masks_[v]) {
      return source_costs_[v][new_src] - source_costs_[v][old_src] + mr_delta;
    }

    double factor = source_costs_[v][new_src] + mr_delta;
    uint64_t diff_on = new_mask & ~demand_masks_[v];
    uint64_t diff_off = demand_masks_[v] & ~new_mask;
    const double* row = v_info.conv_costs.data() + static_cast<size_t>(new_src) * num_layouts;
    while (diff_on != 0) {
      const int dst = __builtin_ctzll(diff_on);
      factor += row[dst];
      diff_on &= diff_on - 1;
    }
    while (diff_off != 0) {
      const int dst = __builtin_ctzll(diff_off);
      factor -= row[dst];
      diff_off &= diff_off - 1;
    }
    return factor - source_costs_[v][old_src];
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
    delta += compute_val_delta(old_u.value_id, &old_u, match_new);
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
      delta += compute_val_delta(new_u.value_id, nullptr, &new_u);
    }
  }
  if (!out_seen) {
    delta += compute_val_delta(op.output_value, nullptr, nullptr);
  }

  return -delta;
}

void IncrementalState::Apply(OpId op_idx, uint16_t new_choice) {
  const uint16_t old_choice = choices_[op_idx];
  if (old_choice == new_choice) {
    return;
  }
  const double g = Gain(op_idx, new_choice);
  const auto& op = problem_.Ops()[op_idx];

  auto update_count = [&](ValueId v, uint16_t lay_idx, int delta) {
    const int old_cnt = counts_[v][lay_idx];
    const int new_cnt = old_cnt + delta;
    counts_[v][lay_idx] = new_cnt;
    const int support_delta = static_cast<int>(new_cnt > 0) - static_cast<int>(old_cnt > 0);
    if (support_delta != 0) {
      if (new_cnt > 0) {
        demand_masks_[v] |= (uint64_t{1} << lay_idx);
      } else {
        demand_masks_[v] &= ~(uint64_t{1} << lay_idx);
      }
      const auto& v_info = problem_.Values()[v];
      const size_t num_layouts = v_info.layouts.size();
      for (size_t s = 0; s < num_layouts; ++s) {
        source_costs_[v][s] +=
            static_cast<double>(support_delta) * v_info.conv_costs[s * num_layouts + lay_idx];
      }
    }
  };

  for (const auto& old_u : op.config_uses[old_choice]) {
    for (const auto& [lay_idx, cnt] : old_u.layout_counts) {
      update_count(old_u.value_id, lay_idx, -cnt);
    }
  }
  for (const auto& new_u : op.config_uses[new_choice]) {
    for (const auto& [lay_idx, cnt] : new_u.layout_counts) {
      update_count(new_u.value_id, lay_idx, cnt);
    }
  }

  sources_[op.output_value] = op.output_layout_indices[new_choice];
  choices_[op_idx] = new_choice;
  cost_ -= g;
  if (std::abs(g) > 1e9) {
    // Recompute exact total cost after entering/leaving a large penalty state so IEEE-754
    // cancellation cannot leave residual drift in `cost_`.
    double exact = 0.0;
    const size_t num_ops = problem_.NumOps();
    for (OpId u = 0; u < num_ops; ++u) {
      exact += problem_.Ops()[u].costs[choices_[u]];
    }
    const size_t num_vals = problem_.NumValues();
    const double multi_rep_penalty = problem_.Instance().cost_model.vec_unsupported_conversion_cycles;
    for (ValueId v = 0; v < num_vals; ++v) {
      exact += source_costs_[v][sources_[v]];
      if (problem_.Values()[v].forbid_multi_rep && __builtin_popcountll(demand_masks_[v]) > 1) {
        exact += multi_rep_penalty;
      }
    }
    cost_ = exact;
  }
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
  return ComputeProblemTreewidthInternal(problem);
}

LayoutSolverResult SolveGreedy(const LayoutProblemInstance& instance,
                               const std::optional<LayoutAssignment>& initial,
                               const LayoutSolverConfig& config) {
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
    GreedyRefine(state, config, evaluated_moves, moves_applied, greedy_sweeps, termination_reason);
  }
  return BuildResult(state, evaluated_moves, moves_applied, greedy_sweeps, initial_cost, termination_reason);
}

LayoutSolverResult SolveTreewidthDP(const LayoutProblemInstance& instance, const LayoutSolverConfig& config) {
  config.Validate();
  IndexedProblem problem(instance);
  const auto decomp = ComputeActiveDecomposition(problem);

  if (problem.NumOps() == 0) {
    IncrementalState empty_state(problem, {});
    return BuildResult(empty_state, 0, 0, 0, 0.0, "optimal", decomp.tw_stats);
  }
  if (decomp.tw_stats.active_primal_treewidth > config.max_treewidth) {
    throw pypto::ValueError("Treewidth DP active primal treewidth " +
                            std::to_string(decomp.tw_stats.active_primal_treewidth) +
                            " exceeds max_treewidth=" + std::to_string(config.max_treewidth));
  }
  if (decomp.tw_stats.max_bag_domain_product > config.max_bag_states) {
    throw pypto::ValueError("Treewidth DP max bag states " +
                            std::to_string(decomp.tw_stats.max_bag_domain_product) +
                            " exceeds max_bag_states=" + std::to_string(config.max_bag_states));
  }

  int64_t evaluated_moves = 0;
  std::vector<uint16_t> choices;
  if (!RunTreewidthDPOnComponents(problem, decomp, config, evaluated_moves, choices)) {
    throw pypto::ValueError("Treewidth DP exceeded max_evaluations=" +
                            std::to_string(config.max_evaluations));
  }
  IncrementalState state(problem, std::move(choices));
  return BuildResult(state, evaluated_moves, 0, 0, state.Cost(), "optimal", decomp.tw_stats);
}

LayoutSolverResult SolveAutoLayout(const LayoutProblemInstance& instance,
                                   const std::optional<LayoutAssignment>& initial,
                                   const LayoutSolverConfig& config) {
  config.Validate();
  IndexedProblem problem(instance);
  const auto decomp = ComputeActiveDecomposition(problem);
  int64_t evaluated_moves = 0;

  if (problem.NumOps() == 0) {
    IncrementalState empty_state(problem, {});
    return BuildResult(empty_state, 0, 0, 0, 0.0, "optimal", decomp.tw_stats);
  }

  const size_t num_comps = decomp.component_decomps.size();
  std::vector<size_t> dp_comps;
  std::vector<size_t> fallback_comps;
  for (size_t c = 0; c < num_comps; ++c) {
    const auto& dec = decomp.component_decomps[c];
    int64_t comp_max_prod = 1;
    int64_t comp_total_states = 0;
    for (const auto& bag : dec.bags) {
      int64_t prod = 1;
      for (OpId u : bag) {
        const int64_t dom = static_cast<int64_t>(problem.Ops()[u].costs.size());
        if (prod > std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, dom)) {
          prod = std::numeric_limits<int64_t>::max();
          break;
        }
        prod *= dom;
      }
      comp_max_prod = std::max(comp_max_prod, prod);
      if (comp_total_states > std::numeric_limits<int64_t>::max() - prod) {
        comp_total_states = std::numeric_limits<int64_t>::max();
      } else {
        comp_total_states += prod;
      }
    }
    if (dec.treewidth <= config.max_treewidth && comp_max_prod <= config.max_bag_states &&
        comp_total_states <= config.max_evaluations) {
      dp_comps.push_back(c);
    } else {
      fallback_comps.push_back(c);
    }
  }

  const DPContext dp_ctx = BuildDPContext(problem, decomp);
  std::vector<uint16_t> dp_choices(problem.NumOps(), 0);
  int64_t dp_evals = 0;
  std::vector<size_t> solved_dp_comps;
  for (size_t c : dp_comps) {
    if (RunTreewidthDPOnSingleComponent(problem, decomp, dp_ctx, c, config, dp_evals, dp_choices)) {
      solved_dp_comps.push_back(c);
    } else {
      fallback_comps.push_back(c);
    }
  }

  if (fallback_comps.empty()) {
    IncrementalState state(problem, std::move(dp_choices));
    return BuildResult(state, dp_evals, 0, 0, state.Cost(), "optimal", decomp.tw_stats);
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

  std::vector<bool> in_fallback(problem.NumOps(), false);
  for (size_t c : fallback_comps) {
    for (OpId u : decomp.components[c]) {
      in_fallback[u] = true;
    }
  }
  std::vector<OpId> fallback_ops;
  for (OpId u : problem.TopoOrder()) {
    if (in_fallback[u]) {
      fallback_ops.push_back(u);
    }
  }

  IncrementalState state(problem, std::move(choices));
  const double initial_cost = state.Cost();
  if (termination_reason != "evaluation_budget" && !fallback_ops.empty()) {
    GreedyRefineOps(state, fallback_ops, config, evaluated_moves, moves_applied, greedy_sweeps,
                    termination_reason);
    if (!initial.has_value() && termination_reason != "evaluation_budget") {
      for (size_t c : fallback_comps) {
        if (termination_reason == "evaluation_budget") {
          break;
        }
        FMRefineComponent(state, decomp.components[c], config, evaluated_moves, moves_applied, greedy_sweeps,
                          termination_reason);
      }
    }
  }
  return BuildResult(state, evaluated_moves, moves_applied, greedy_sweeps, initial_cost, termination_reason,
                     decomp.tw_stats);
}

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto
