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

#ifndef PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_SOLVER_H_
#define PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_SOLVER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/auto_layout/model.h"

namespace pypto {
namespace ir {
namespace auto_layout {

using ValueId = uint32_t;
using OpId = uint32_t;

/// Deterministic search limits, with costs expressed in modeled cycles.
struct LayoutSolverConfig {
  int64_t max_evaluations = 100000;
  int64_t max_greedy_sweeps = 100;
  int64_t max_treewidth = 8;
  int64_t max_bag_states = 100000;
  double tolerance = 1e-9;

  void Validate() const;
};

/// One shared conversion required by the scored assignment.
struct LayoutConversion {
  std::string value_name;
  LayoutSpec source_layout;
  LayoutSpec target_layout;
  double cost_cycles = 0.0;
  std::vector<std::string> consumer_op_ids;
  bool is_boundary = false;
};

/// Treewidth and DP state-space diagnostics for a `LayoutProblemInstance`.
struct TreewidthStats {
  int64_t num_values = 0;
  int64_t num_operations = 0;
  int64_t num_movable_operations = 0;
  int64_t dataflow_treewidth = 0;
  int64_t primal_treewidth = 0;
  int64_t active_primal_treewidth = 0;
  int64_t max_bag_size = 0;
  int64_t max_active_bag_size = 0;
  int64_t max_bag_domain_product = 0;
  int64_t total_dp_states = 0;
  std::vector<std::string> elimination_order;
  std::vector<std::vector<std::string>> bags;
};

/// Search accounting and structural treewidth metadata.
struct LayoutSolverStats {
  int64_t evaluated_moves = 0;
  int64_t moves_applied = 0;
  int64_t greedy_sweeps = 0;
  double initial_cost_cycles = 0.0;
  std::string termination_reason = "local_optimum";
  int64_t treewidth = 0;
  int64_t active_treewidth = 0;
};

/// Best modeled solution, search diagnostics, and its writeback requirements.
struct LayoutSolverResult {
  CostBreakdown breakdown;
  LayoutSolverStats stats;
  std::vector<LayoutConversion> conversions;
  std::vector<std::string> writeback_issues;
  std::optional<TreewidthStats> treewidth_stats;

  [[nodiscard]] const LayoutAssignment& Assignment() const { return breakdown.assignment; }
  [[nodiscard]] double TotalCostCycles() const { return breakdown.total_cost_cycles; }
  [[nodiscard]] const LayoutAssignment& WritebackAssignment() const;
};

/// Dense integer-indexed problem representation for fast DP and incremental search.
class IndexedProblem {
 public:
  struct ValueInfo {
    std::string name;
    std::optional<OpId> producer_op;
    std::vector<OpId> participants;
    std::vector<LayoutSpec> layouts;
    uint16_t default_source_idx = 0;
    uint64_t boundary_demand_mask = 0;
    std::vector<int> boundary_counts;
    /// Flat `[src_idx * num_layouts + dst_idx]` precomputed conversion costs.
    std::vector<double> conv_costs;
    uint64_t incompatible_target_mask = 0;
    bool forbid_multi_rep = false;

    [[nodiscard]] int FindLayoutIndex(const LayoutSpec& spec) const;
  };

  struct OpConfigUse {
    ValueId value_id = 0;
    uint64_t demand_mask = 0;
    /// Parallel `(local_layout_idx, count)` pairs for incremental reference counting.
    std::vector<std::pair<uint16_t, int>> layout_counts;
  };

  struct OpInfo {
    std::string op_id;
    std::string op_name;
    int64_t invocations = 1;
    ValueId output_value = 0;
    std::vector<double> costs;
    std::vector<uint16_t> output_layout_indices;
    std::vector<std::vector<OpConfigUse>> config_uses;
  };

  explicit IndexedProblem(const LayoutProblemInstance& instance);

  [[nodiscard]] const LayoutProblemInstance& Instance() const { return instance_; }
  [[nodiscard]] size_t NumValues() const { return values_.size(); }
  [[nodiscard]] size_t NumOps() const { return ops_.size(); }
  [[nodiscard]] const std::vector<ValueInfo>& Values() const { return values_; }
  [[nodiscard]] const std::vector<OpInfo>& Ops() const { return ops_; }
  [[nodiscard]] const std::vector<OpId>& TopoOrder() const { return topo_order_; }
  [[nodiscard]] std::optional<OpId> FindOpById(const std::string& op_id) const;
  [[nodiscard]] std::optional<ValueId> FindValueByName(const std::string& name) const;
  [[nodiscard]] std::vector<uint16_t> InitialChoices(const LayoutAssignment& initial) const;

 private:
  const LayoutProblemInstance& instance_;
  std::vector<ValueInfo> values_;
  std::vector<OpInfo> ops_;
  std::vector<OpId> topo_order_;
  std::unordered_map<std::string, ValueId> value_by_name_;
  std::unordered_map<std::string, OpId> op_by_id_;
};

/// Demand-count and source-cost caches for O(degree) incremental move evaluation.
class IncrementalState {
 public:
  IncrementalState(const IndexedProblem& problem, std::vector<uint16_t> choices);

  [[nodiscard]] double Gain(OpId op_idx, uint16_t new_choice) const;
  void Apply(OpId op_idx, uint16_t new_choice);
  [[nodiscard]] LayoutAssignment ToAssignment() const;

  [[nodiscard]] double Cost() const { return cost_; }
  [[nodiscard]] const std::vector<uint16_t>& Choices() const { return choices_; }
  [[nodiscard]] const std::vector<uint16_t>& Sources() const { return sources_; }
  [[nodiscard]] const std::vector<uint64_t>& DemandMasks() const { return demand_masks_; }
  [[nodiscard]] const std::vector<std::vector<int>>& Counts() const { return counts_; }
  [[nodiscard]] const std::vector<std::vector<double>>& SourceCosts() const { return source_costs_; }
  [[nodiscard]] const IndexedProblem& Problem() const { return problem_; }

 private:
  const IndexedProblem& problem_;
  std::vector<uint16_t> choices_;
  std::vector<uint16_t> sources_;
  std::vector<uint64_t> demand_masks_;
  std::vector<std::vector<int>> counts_;
  std::vector<std::vector<double>> source_costs_;
  double cost_ = 0.0;
};

/// Compute exact min-fill treewidth and DP state-space statistics for `instance`.
TreewidthStats ComputeTreewidth(const LayoutProblemInstance& instance);

/// Solve via topological initialization followed by coordinate-descent refinement.
LayoutSolverResult SolveGreedy(const LayoutProblemInstance& instance,
                               const std::optional<LayoutAssignment>& initial = std::nullopt,
                               const LayoutSolverConfig& config = {});

/// Solve to global optimality via connected-component min-fill treewidth DP.
LayoutSolverResult SolveTreewidthDP(const LayoutProblemInstance& instance,
                                    const LayoutSolverConfig& config = {});

/// Solve via exact treewidth DP when `active_treewidth <= max_treewidth` and within budgets, else greedy.
LayoutSolverResult SolveAutoLayout(const LayoutProblemInstance& instance,
                                   const std::optional<LayoutAssignment>& initial = std::nullopt,
                                   const LayoutSolverConfig& config = {});

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_SOLVER_H_
