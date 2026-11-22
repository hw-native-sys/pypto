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
  /// Auto/DP report full min-fill diagnostics for graphs of up to 256 operations.
  /// When the report is disabled or exceeds this limit, both reported widths are -1.
  bool collect_treewidth_stats = true;

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
    struct Consumer {
      OpId op_id;
      size_t input_slot;
    };

    std::string name;
    std::optional<OpId> producer_op;
    std::vector<OpId> participants;
    /// One entry per consuming operation, even when it uses the value multiple times.
    std::vector<Consumer> consumers;
    /// Layouts that every consumer can request for all its occurrences of this value.
    uint64_t common_consumer_layouts = UINT64_MAX;
    std::vector<LayoutSpec> layouts;
    uint16_t default_source_idx = 0;
    uint64_t boundary_demand_mask = 0;
    std::vector<int> boundary_counts;
    /// Flat `[src_idx * num_layouts + dst_idx]` precomputed conversion costs.
    std::vector<double> conv_costs;
    double max_conversion_cost = 0.0;
    uint64_t incompatible_target_mask = 0;
    bool forbid_multi_rep = false;
    /// Per-source masks of layout pairs that writeback cannot materialize.
    std::vector<uint64_t> unsupported_target_masks;
    bool has_repeated_operand = false;

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
    std::vector<bool> incompatible_configs;
    std::vector<uint16_t> output_layout_indices;
    /// One entry per distinct input value, in the same order for every configuration.
    std::vector<std::vector<OpConfigUse>> config_uses;
  };

  explicit IndexedProblem(const LayoutProblemInstance& instance);

  [[nodiscard]] const LayoutProblemInstance& Instance() const { return instance_; }
  [[nodiscard]] size_t NumValues() const { return values_.size(); }
  [[nodiscard]] size_t NumOps() const { return ops_.size(); }
  [[nodiscard]] const std::vector<ValueInfo>& Values() const { return values_; }
  [[nodiscard]] const std::vector<OpInfo>& Ops() const { return ops_; }
  [[nodiscard]] const std::vector<OpId>& TopoOrder() const { return topo_order_; }
  [[nodiscard]] std::vector<uint16_t> InitialChoices(const LayoutAssignment& initial) const;

 private:
  const LayoutProblemInstance& instance_;
  std::vector<ValueInfo> values_;
  std::vector<OpInfo> ops_;
  std::vector<OpId> topo_order_;
  std::unordered_map<std::string, ValueId> value_by_name_;
  std::unordered_map<std::string, OpId> op_by_id_;
};

/// Search orders writeback violations before the modeled cycle cost. A gain
/// uses the same fields for the reductions in each part of that cost.
struct LayoutSearchScore {
  int64_t violations = 0;
  double cycles = 0.0;
};

/// Demand counts and conversion terms for incremental move evaluation.
class IncrementalState {
 public:
  using ChoiceChange = std::pair<OpId, uint16_t>;

  struct BlockEvaluation {
    struct Factor {
      ValueId value;
      uint16_t source;
      std::vector<int> counts;
      uint64_t demand_mask = 0;
    };

    double gain = 0.0;
    int64_t violation_reduction = 0;
    std::vector<Factor> factors;
  };

  IncrementalState(const IndexedProblem& problem, std::vector<uint16_t> choices);

  /// Omit a factor when ranking choices within a fan-out move with a prescribed input layout.
  [[nodiscard]] LayoutSearchScore GainScore(OpId op_idx, uint16_t new_choice,
                                            std::optional<ValueId> omitted_factor = std::nullopt) const;
  void Apply(OpId op_idx, uint16_t new_choice);
  /// Stage the net changes from distinct operations without changing the current assignment.
  [[nodiscard]] BlockEvaluation EvaluateBlock(const std::vector<ChoiceChange>& choices) const;
  /// Apply a staged evaluation while its operation choices and factors are still current.
  void ApplyBlock(const std::vector<ChoiceChange>& choices, const BlockEvaluation& evaluation);
  [[nodiscard]] LayoutAssignment ToAssignment() const;

  [[nodiscard]] double Cost() const { return cost_.Value(); }
  [[nodiscard]] LayoutSearchScore SearchCost() const { return {cost_.violations, Cost()}; }
  [[nodiscard]] const std::vector<uint16_t>& Choices() const { return choices_; }
  [[nodiscard]] const std::vector<uint16_t>& Sources() const { return sources_; }
  [[nodiscard]] const std::vector<uint64_t>& DemandMasks() const { return demand_masks_; }
  [[nodiscard]] const std::vector<std::vector<int>>& Counts() const { return counts_; }
  [[nodiscard]] const IndexedProblem& Problem() const { return problem_; }

 private:
  struct CostSum {
    void Add(double value);
    void Replace(double before, double after);
    [[nodiscard]] double Value() const { return sum + correction; }

    double sum = 0.0;
    double correction = 0.0;
    int64_t violations = 0;
  };

  void AddFactorCostDelta(CostSum& cost, ValueId value, uint16_t old_source, uint64_t old_mask,
                          uint16_t new_source, uint64_t new_mask) const;
  void CommitFactor(ValueId value, uint64_t old_mask, uint16_t new_source);
  void UpdateDemandCount(ValueId value, uint16_t layout, int delta);

  const IndexedProblem& problem_;
  std::vector<uint16_t> choices_;
  std::vector<uint16_t> sources_;
  std::vector<uint64_t> demand_masks_;
  std::vector<std::vector<int>> counts_;
  CostSum cost_;
};

/// Check Vec capacity for the conversion demands in the assignment's selected operation configurations.
std::vector<std::string> CheckVecConversionCapacity(const LayoutProblemInstance& instance,
                                                    const LayoutAssignment& assignment);

/// Solve via topological initialization followed by coordinate-descent refinement.
LayoutSolverResult SolveGreedy(const LayoutProblemInstance& instance,
                               const std::optional<LayoutAssignment>& initial = std::nullopt,
                               const LayoutSolverConfig& config = {});

/// Retain a greedy incumbent and try best-prefix FM from the topological seed, within one budget.
/// Single-op and fan-out moves compete at each frontier step.
LayoutSolverResult SolveGreedyFM(const LayoutProblemInstance& instance,
                                 const LayoutSolverConfig& config = {});

/// Solve the layout cost factors exactly via connected-component min-fill treewidth DP.
/// Each movable component can contain at most 256 operations.
/// When the result exceeds Vec capacity, use bounded capacity repair within the evaluation budget.
LayoutSolverResult SolveTreewidthDP(const LayoutProblemInstance& instance,
                                    const LayoutSolverConfig& config = {});

/// Solve via exact treewidth DP for each component within the decomposition,
/// width and evaluation limits. Use local search for other components.
LayoutSolverResult SolveAutoLayout(const LayoutProblemInstance& instance,
                                   const std::optional<LayoutAssignment>& initial = std::nullopt,
                                   const LayoutSolverConfig& config = {});

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_SOLVER_H_
