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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/auto_layout/solver.h"

namespace {

namespace al = pypto::ir::auto_layout;
using pypto::DataType;
using pypto::backend::BackendType;
using pypto::ir::MemorySpace;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RequireNear(double actual, double expected, double tol, const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tol) {
    throw std::runtime_error(message + " (got " + std::to_string(actual) + ", expected " +
                             std::to_string(expected) + ")");
  }
}

al::ValueNode MakeValue(const std::string& name, const std::optional<std::string>& producer = std::nullopt,
                        MemorySpace space = MemorySpace::DDR, bool reusable = false,
                        bool external_output = false, std::vector<al::LayoutSpec> admissible = {}) {
  if (admissible.empty()) {
    if (space == MemorySpace::DDR) {
      admissible = {al::LayoutSpec::ND(), al::LayoutSpec::NZ()};
    } else {
      admissible = {al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::ColMajorFlat()};
    }
  }
  al::ValueNode node;
  node.name = name;
  node.shape = {16, 16};
  node.dtype = DataType::FP16;
  node.memory_space = space;
  node.current_layout = admissible.front();
  node.admissible_layouts = std::move(admissible);
  node.producer_op_id = producer;
  node.is_reusable_weight = reusable;
  node.is_external_output = external_output;
  return node;
}

al::OperationNode MakeOp(
    const std::string& op_id, std::vector<std::string> inputs, std::string output,
    const std::vector<std::tuple<std::vector<al::LayoutSpec>, al::LayoutSpec, double>>& specs,
    int64_t invocations = 1, const std::string& op_name = "synthetic") {
  al::OperationNode op;
  op.op_id = op_id;
  op.op_name = op_name;
  op.inputs = std::move(inputs);
  op.output = std::move(output);
  op.invocations = invocations;
  for (const auto& [in_lays, out_lay, cost] : specs) {
    op.configs.push_back(al::OperatorConfig{op_id, op_name, in_lays, out_lay, cost, ""});
  }
  return op;
}

double BruteForceExactBest(const al::LayoutProblemInstance& instance) {
  const size_t n = instance.operations.size();
  std::vector<size_t> idx(n, 0);
  double best = std::numeric_limits<double>::infinity();
  while (true) {
    al::LayoutAssignment assign;
    for (size_t i = 0; i < n; ++i) {
      assign.op_configs[instance.operations[i].op_id] = instance.operations[i].configs[idx[i]];
    }
    const auto bd = al::EvaluateAssignment(instance, assign);
    if (bd.total_cost_cycles < best) {
      best = bd.total_cost_cycles;
    }
    size_t pos = n;
    while (pos > 0) {
      --pos;
      if (++idx[pos] < instance.operations[pos].configs.size()) {
        break;
      }
      idx[pos] = 0;
      if (pos == 0) {
        return best;
      }
    }
    if (n == 0) {
      return best;
    }
  }
}

void TestCostModelHardwareEquations() {
  al::LayoutCostModel a2a3;
  a2a3.backend_type = BackendType::Ascend910;
  al::LayoutCostModel a5;
  a5.backend_type = BackendType::Ascend950;

  Require(a2a3.TargetArch() == "a2a3", "Ascend910 should resolve to a2a3");
  Require(a5.TargetArch() == "a5", "Ascend950 should resolve to a5");
  Require(a2a3.GmAccessGranularityBytes() == 512, "a2a3 granularity should be 512");
  Require(a5.GmAccessGranularityBytes() == 128, "a5 granularity should be 128");
  RequireNear(a2a3.EstimateVbrcbCost(32), 4.0 + 32 * 0.5, 1e-9, "a2a3 vbrcb cost mismatch");
  RequireNear(a5.EstimateVbrcbCost(32), 0.0, 1e-9, "a5 vbrcb cost should be 0");

  auto gm_nd = MakeValue("x");
  gm_nd.shape = {64, 128};
  auto mat_tile = MakeValue("xt", "op0", MemorySpace::Mat, false, false, {al::LayoutSpec::FractalNZ()});
  mat_tile.shape = {64, 128};
  Require(a2a3.IsValidLoadConfig(gm_nd, mat_tile, al::LayoutSpec::NZ(), al::LayoutSpec::FractalNZ()),
          "Mat load NZ2NZ must be legal");
  const double nd_cost =
      a2a3.EstimateLoadCost(gm_nd, mat_tile, al::LayoutSpec::ND(), al::LayoutSpec::FractalNZ());
  const double nz_cost =
      a2a3.EstimateLoadCost(gm_nd, mat_tile, al::LayoutSpec::NZ(), al::LayoutSpec::FractalNZ());
  Require(nz_cost < nd_cost, "Contiguous NZ2NZ load should be cheaper than ND2NZ");

  // Acc -> NZ store is supported on a2a3 FP32 16-aligned, rejected on a5.
  auto acc_tile = MakeValue("acc", "mm", MemorySpace::Acc, false, false, {al::LayoutSpec::FractalAccNZ()});
  acc_tile.shape = {64, 64};
  acc_tile.dtype = DataType::FP32;
  auto gm_out = MakeValue("out", "st", MemorySpace::DDR, false, true);
  gm_out.shape = {64, 64};
  gm_out.dtype = DataType::FP32;
  Require(a2a3.SupportsAccNzStore(acc_tile, gm_out), "a2a3 must support FP32 64x64 Acc->NZ store");
  Require(!a5.SupportsAccNzStore(acc_tile, gm_out), "a5 must reject Acc->NZ store");

  // MTE1 L0A vs L0B and Cube FP32 passes (a2a3 vs a5).
  auto l0a = MakeValue("l0a", "mv0", MemorySpace::Left, false, false, {al::LayoutSpec::FractalNZ()});
  l0a.shape = {64, 128};
  auto l0b = MakeValue("l0b", "mv1", MemorySpace::Right, false, false, {al::LayoutSpec::FractalZN()});
  l0b.shape = {128, 64};
  Require(a2a3.EstimateMoveCost(mat_tile, l0a, al::LayoutSpec::FractalNZ(), al::LayoutSpec::FractalNZ()) <
              a2a3.EstimateMoveCost(mat_tile, l0b, al::LayoutSpec::FractalZN(), al::LayoutSpec::FractalZN()),
          "L0A MTE1 bandwidth should exceed L0B");

  auto lhs_fp32 = l0a;
  lhs_fp32.dtype = DataType::FP32;
  auto rhs_fp32 = l0b;
  rhs_fp32.dtype = DataType::FP32;
  auto acc_fp32 = acc_tile;
  Require(a5.EstimateMatmulCost(lhs_fp32, rhs_fp32, acc_fp32) >
              a2a3.EstimateMatmulCost(lhs_fp32, rhs_fp32, acc_fp32),
          "a5 FP32 matmul uses 8 passes vs 2 on a2a3");

  // MTE3 Vec store & ROW_BCAST_FLAT conversion.
  auto vec_col = MakeValue(
      "rsum", "rs", MemorySpace::Vec, false, false,
      {al::LayoutSpec::ColMajorFlat(), al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::RowBcastFlat()});
  vec_col.shape = {32, 1};
  vec_col.dtype = DataType::FP32;
  Require(
      a2a3.SupportsVecTileConversion(vec_col, al::LayoutSpec::ColMajorFlat(), al::LayoutSpec::RowBcastFlat()),
      "Vec [32,1] must support conversion to ROW_BCAST_FLAT");
  RequireNear(
      a2a3.EstimateConversionCost(vec_col, al::LayoutSpec::ColMajorFlat(), al::LayoutSpec::RowMajorFlat()),
      a2a3.vec_reshape_metadata_cycles, 1e-9, "1D [32,1]<->[1,32] conversion must use reshape cost");

  auto vec_bool = MakeValue("mask", "cmp", MemorySpace::Vec, false, false, {al::LayoutSpec::RowMajorFlat()});
  vec_bool.shape = {16, 32};
  vec_bool.dtype = DataType::BOOL;
  Require(vec_bool.ByteSize() == 16LL * 32LL, "BOOL Vec tile ByteSize must use 1 byte/element storage");
  Require(!al::IsNzCompatibleShape(vec_bool.shape, DataType::BOOL), "BOOL must not be NZ-compatible");
}

void TestTreewidthStatsChainDiamondAndMultiComponent() {
  // Chain: P -> Q -> R (tw=1)
  al::LayoutProblemInstance chain;
  chain.AddValue(MakeValue("x"));
  chain.AddValue(MakeValue("t1", "P"));
  chain.AddValue(MakeValue("t2", "Q"));
  chain.AddValue(MakeValue("y", "R"));
  chain.operations = {
      MakeOp("P", {"x"}, "t1",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
      MakeOp("Q", {"t1"}, "t2",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
      MakeOp("R", {"t2"}, "y",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
  };
  const auto tw_chain = al::ComputeTreewidth(chain);
  Require(tw_chain.dataflow_treewidth == 1 && tw_chain.primal_treewidth == 1 &&
              tw_chain.active_primal_treewidth == 1,
          "Chain treewidth must be 1");
  Require(tw_chain.max_bag_size == 2 && tw_chain.max_bag_domain_product == 4, "Chain bag stats mismatch");

  // Diamond: P -> (Q1, Q2) -> R (tw=2)
  al::LayoutProblemInstance diamond;
  diamond.AddValue(MakeValue("x"));
  diamond.AddValue(MakeValue("t1", "P"));
  diamond.AddValue(MakeValue("u1", "Q1"));
  diamond.AddValue(MakeValue("u2", "Q2"));
  diamond.AddValue(MakeValue("y", "R"));
  diamond.operations = {
      MakeOp("P", {"x"}, "t1",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
      MakeOp("Q1", {"t1"}, "u1",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
      MakeOp("Q2", {"t1"}, "u2",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
      MakeOp("R", {"u1", "u2"}, "y",
             {{{al::LayoutSpec::ND(), al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ(), al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 1.0}}),
  };
  const auto tw_diamond = al::ComputeTreewidth(diamond);
  Require(tw_diamond.dataflow_treewidth == 2 && tw_diamond.primal_treewidth == 2,
          "Diamond treewidth must be 2");
  Require(tw_diamond.max_bag_size == 3 && tw_diamond.max_bag_domain_product == 8,
          "Diamond bag stats mismatch");

  // Multi-component disconnected problem + fixed op.
  al::LayoutProblemInstance multi;
  multi.AddValue(MakeValue("a0"));
  multi.AddValue(MakeValue("a1", "A"));
  multi.AddValue(MakeValue("b0"));
  multi.AddValue(MakeValue("b1", "B"));
  multi.AddValue(MakeValue("c0"));
  multi.AddValue(MakeValue("c1", "C"));
  multi.cost_model.gm_pack_fixed_cycles = 5.0;
  multi.cost_model.gm_pack_bytes_per_cycle = 1e18;
  multi.operations = {
      MakeOp("A", {"a0"}, "a1",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 4.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::ND(), 1.0}}),
      MakeOp("B", {"b0"}, "b1",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 2.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::ND(), 10.0}}),
      MakeOp("C", {"c0"}, "c1", {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 3.0}}),
  };
  const auto dp_multi = al::SolveTreewidthDP(multi);
  RequireNear(dp_multi.TotalCostCycles(), BruteForceExactBest(multi), 1e-9,
              "Multi-component DP must match exact oracle");
  Require(dp_multi.stats.termination_reason == "optimal", "Multi-component DP must be optimal");
}

void TestSharedConversionBarrierAndOracle() {
  for (bool repeated_operand : {false, true}) {
    al::LayoutProblemInstance inst;
    inst.AddValue(MakeValue("x"));
    inst.AddValue(MakeValue("y", "Q1"));
    inst.AddValue(MakeValue("z", "Q2"));
    inst.cost_model.gm_pack_fixed_cycles = 6.0;
    inst.cost_model.gm_pack_bytes_per_cycle = 1e18;

    std::vector<std::string> q1_in =
        repeated_operand ? std::vector<std::string>{"x", "x"} : std::vector<std::string>{"x"};
    std::vector<al::LayoutSpec> q1_nz(q1_in.size(), al::LayoutSpec::NZ());
    std::vector<al::LayoutSpec> q1_nd(q1_in.size(), al::LayoutSpec::ND());

    inst.operations = {
        MakeOp("Q1", q1_in, "y", {{q1_nz, al::LayoutSpec::ND(), 0.0}, {q1_nd, al::LayoutSpec::ND(), 2.0}}),
        MakeOp("Q2", {"x"}, "z",
               {{{al::LayoutSpec::NZ()}, al::LayoutSpec::ND(), 0.0},
                {{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 2.0}}),
    };

    al::LayoutAssignment bad_init;
    bad_init.op_configs["Q1"] = inst.operations[0].configs[0];
    bad_init.op_configs["Q2"] = inst.operations[1].configs[0];
    const auto greedy = al::SolveGreedy(inst, bad_init);
    const auto dp = al::SolveTreewidthDP(inst);
    RequireNear(greedy.TotalCostCycles(), 6.0, 1e-9, "Greedy should get trapped in shared barrier");
    RequireNear(dp.TotalCostCycles(), 4.0, 1e-9, "DP must escape shared conversion barrier");
    RequireNear(dp.TotalCostCycles(), BruteForceExactBest(inst), 1e-9, "DP must match exact oracle");
    Require(dp.conversions.empty(), "Optimal barrier solution has zero conversions");
  }
}

void TestIncrementalStateAndAsymmetricCosts() {
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  const auto DN = al::LayoutSpec::DN();

  al::LayoutProblemInstance inst;
  inst.cost_model.expected_uses = 3;
  inst.AddValue(MakeValue("x", std::nullopt, MemorySpace::DDR, true, false, {ND, NZ, DN}));
  inst.AddValue(MakeValue("t", "P", MemorySpace::DDR, false, false, {ND, NZ, DN}));
  inst.AddValue(MakeValue("u", "Q", MemorySpace::DDR, false, false, {ND, NZ, DN}));
  inst.AddValue(MakeValue("y", "R", MemorySpace::DDR, false, true, {ND, NZ, DN}));
  inst.operations = {
      MakeOp("P", {"x"}, "t", {{{ND}, ND, 3.0}, {{NZ}, NZ, 1.0}, {{DN}, DN, 2.0}}, 3),
      MakeOp("Q", {"t", "t"}, "u", {{{ND, ND}, NZ, 5.0}, {{NZ, DN}, ND, 2.0}, {{DN, NZ}, DN, 4.0}}, 6),
      MakeOp("R", {"t", "u", "t"}, "y",
             {{{NZ, ND, NZ}, NZ, 0.0}, {{DN, DN, DN}, DN, 1.0}, {{ND, NZ, DN}, ND, 3.0}}, 3),
  };

  auto rank_of = [&](const al::LayoutSpec& l) -> int {
    if (l == ND) return 0;
    if (l == NZ) return 1;
    return 2;
  };
  const std::vector<al::LayoutSpec> all_l = {ND, NZ, DN};
  for (const auto& [name, node] : inst.values) {
    for (const auto& s : all_l) {
      for (const auto& d : all_l) {
        if (s != d) {
          inst.custom_conversion_costs[name][{s, d}] =
              static_cast<double>(7 * (rank_of(s) + 1) + 3 * (rank_of(d) + 1) + name.size());
        }
      }
    }
  }

  const auto dp = al::SolveTreewidthDP(inst);
  RequireNear(dp.TotalCostCycles(), BruteForceExactBest(inst), 1e-9,
              "Asymmetric conversion DP must match brute-force oracle");

  al::IndexedProblem prob(inst);
  for (uint16_t c0 = 0; c0 < 3; ++c0) {
    for (uint16_t c1 = 0; c1 < 3; ++c1) {
      for (uint16_t c2 = 0; c2 < 3; ++c2) {
        std::vector<uint16_t> choices = {c0, c1, c2};
        al::IncrementalState state(prob, choices);
        const double base_cost = al::EvaluateAssignment(inst, state.ToAssignment()).total_cost_cycles;
        RequireNear(state.Cost(), base_cost, 1e-9, "IncrementalState init cost mismatch");
        const auto counts_before = state.Counts();
        const auto sources_before = state.Sources();
        const auto source_costs_before = state.SourceCosts();

        for (al::OpId op = 0; op < 3; ++op) {
          const uint16_t old_c = choices[op];
          for (uint16_t cand = 0; cand < 3; ++cand) {
            if (cand == old_c) continue;
            auto next = choices;
            next[op] = cand;
            al::LayoutAssignment next_assign;
            for (size_t i = 0; i < 3; ++i) {
              next_assign.op_configs[inst.operations[i].op_id] = inst.operations[i].configs[next[i]];
            }
            const double expected = al::EvaluateAssignment(inst, next_assign).total_cost_cycles;
            RequireNear(state.Gain(op, cand), base_cost - expected, 1e-9, "IncrementalState::Gain mismatch");
            state.Apply(op, cand);
            RequireNear(state.Cost(), expected, 1e-9, "IncrementalState::Apply cost mismatch");
            state.Apply(op, old_c);
            RequireNear(state.Cost(), base_cost, 1e-9, "IncrementalState undo cost mismatch");
            Require(state.Counts() == counts_before, "Undo counts mismatch");
            Require(state.Sources() == sources_before, "Undo sources mismatch");
            Require(state.SourceCosts() == source_costs_before, "Undo source_costs mismatch");
          }
        }
      }
    }
  }
}

void TestBudgetAndTreewidthLimits() {
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("x"));
  inst.AddValue(MakeValue("t", "op0", MemorySpace::Mat, false, false, {al::LayoutSpec::FractalNZ()}));
  inst.operations = {
      MakeOp("op0", {"x"}, "t",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::FractalNZ(), 10.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::FractalNZ(), 5.0}}),
  };

  al::LayoutSolverConfig budget_cfg;
  budget_cfg.max_evaluations = 1;
  const auto auto_res = al::SolveAutoLayout(inst, std::nullopt, budget_cfg);
  Require(auto_res.stats.termination_reason == "evaluation_budget",
          "SolveAutoLayout must report evaluation_budget when max_evaluations=1");

  bool threw = false;
  try {
    (void)al::SolveTreewidthDP(inst, budget_cfg);
  } catch (const pypto::ValueError&) {
    threw = true;
  }
  Require(threw, "SolveTreewidthDP must throw ValueError on max_evaluations=1");

  // Treewidth exceeds limit check.
  al::LayoutProblemInstance k4;
  k4.AddValue(MakeValue("s"));
  k4.AddValue(MakeValue("a", "A"));
  k4.AddValue(MakeValue("b", "B"));
  k4.AddValue(MakeValue("c", "C"));
  k4.AddValue(MakeValue("d", "D"));
  k4.operations = {
      MakeOp("A", {"s"}, "a",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 2.0}}),
      MakeOp("B", {"s"}, "b",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 2.0}}),
      MakeOp("C", {"s"}, "c",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 2.0}}),
      MakeOp("D", {"s"}, "d",
             {{{al::LayoutSpec::ND()}, al::LayoutSpec::ND(), 1.0},
              {{al::LayoutSpec::NZ()}, al::LayoutSpec::NZ(), 2.0}}),
  };
  al::LayoutSolverConfig tw0;
  tw0.max_treewidth = 0;
  bool tw_threw = false;
  try {
    (void)al::SolveTreewidthDP(k4, tw0);
  } catch (const pypto::ValueError&) {
    tw_threw = true;
  }
  Require(tw_threw, "SolveTreewidthDP must throw ValueError when active_treewidth > max_treewidth");
}

void TestIndexCapacityLimits() {
  const auto flat = al::LayoutSpec::RowMajorFlat();
  auto value = MakeValue("tile", std::nullopt, MemorySpace::Vec, false, false, {flat});
  for (int i = 1; i <= 64; ++i) {
    auto layout = flat;
    layout.tag = std::to_string(i);
    value.admissible_layouts.push_back(layout);
  }
  al::LayoutProblemInstance many_layouts;
  many_layouts.AddValue(value);
  auto rejected = [](const al::LayoutProblemInstance& instance, const std::string& expected) {
    try {
      al::IndexedProblem indexed(instance);
    } catch (const pypto::ValueError& e) {
      return std::string(e.what()).find(expected) != std::string::npos;
    }
    return false;
  };
  Require(rejected(many_layouts, "exceeds 64 candidate layouts"), "Reject a layout beyond the mask width");

  al::LayoutProblemInstance many_configs;
  many_configs.AddValue(MakeValue("out", "producer"));
  auto op = MakeOp("producer", {}, "out", {{{}, al::LayoutSpec::ND(), 0.0}});
  op.configs.resize(65536, op.configs.front());
  many_configs.operations.push_back(std::move(op));
  Require(rejected(many_configs, "exceeds 65535 configurations"), "Reject a truncated configuration count");
}

void TestTreewidthAbove64AndPerComponentDPFallback() {
  // 1) Above 64 operations: ComputeTreewidth and SolveAutoLayout(instance).treewidth_stats must agree.
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  al::LayoutProblemInstance big;
  big.AddValue(MakeValue("v0"));
  for (int i = 0; i < 70; ++i) {
    const std::string out_name = "v" + std::to_string(i + 1);
    const std::string op_name = "Op" + std::to_string(i);
    big.AddValue(MakeValue(out_name, op_name));
    big.operations.push_back(MakeOp(op_name, {"v" + std::to_string(i)}, out_name,
                                    {{{ND}, ND, (i % 2 == 0) ? 1.0 : 2.0}, {{NZ}, NZ, 1.5}}));
  }
  const auto expected_tw = al::ComputeTreewidth(big);
  const auto auto_big = al::SolveAutoLayout(big);
  Require(auto_big.treewidth_stats.has_value(), "treewidth_stats must be populated");
  Require(auto_big.treewidth_stats->primal_treewidth == expected_tw.primal_treewidth,
          "SolveAutoLayout primal_treewidth must match ComputeTreewidth above 64 ops");
  Require(auto_big.treewidth_stats->active_primal_treewidth == expected_tw.active_primal_treewidth,
          "SolveAutoLayout active_primal_treewidth must match ComputeTreewidth above 64 ops");

  // Both components start at all-ND and need coordinated moves. The two-node
  // chain fits width one. The four consumers form a clique of width three.
  al::LayoutProblemInstance mixed;
  mixed.cost_model.gm_pack_fixed_cycles = 10.0;
  mixed.cost_model.gm_pack_bytes_per_cycle = 1e18;
  mixed.AddValue(MakeValue("x0"));
  mixed.AddValue(MakeValue("t0", "A0"));
  mixed.AddValue(MakeValue("y0", "A1"));
  mixed.operations.push_back(MakeOp("A0", {"x0"}, "t0", {{{ND}, ND, 0.0}, {{ND}, NZ, 1.0}}));
  mixed.operations.push_back(MakeOp("A1", {"t0"}, "y0", {{{ND}, ND, 6.0}, {{NZ}, ND, 0.0}}));
  mixed.AddValue(MakeValue("x1"));
  for (int i = 0; i < 4; ++i) {
    const std::string id = "B" + std::to_string(i);
    const std::string out = "b" + std::to_string(i);
    mixed.AddValue(MakeValue(out, id));
    mixed.operations.push_back(MakeOp(id, {"x1"}, out, {{{ND}, ND, 3.0}, {{NZ}, ND, 0.0}}));
  }
  al::LayoutSolverConfig cfg_tw1;
  cfg_tw1.max_treewidth = 1;
  const auto greedy = al::SolveGreedy(mixed, std::nullopt, cfg_tw1);
  RequireNear(greedy.TotalCostCycles(), 18.0, 1e-9, "Each individual flip is unprofitable");
  const auto result = al::SolveAutoLayout(mixed, std::nullopt, cfg_tw1);
  const double optimum = BruteForceExactBest(mixed);
  RequireNear(optimum, 11.0, 1e-9, "Chain costs one and all four consumers share one ten-cycle pack");
  RequireNear(result.TotalCostCycles(), optimum, 1e-9, "Hybrid solve must recover both coordinated moves");
  Require(result.stats.termination_reason != "optimal", "A heuristic component has no exact certificate");
}

void TestIfJoinAndClassFCrossLoopForbidden() {
  const auto RM = al::LayoutSpec::RowMajorFlat();
  const auto CM = al::LayoutSpec::ColMajorFlat();

  // 1) control.if_join forces both branches to match the join layout, inserting a
  // realizable Vec branch-yield conversion only when direct matching costs more.
  al::LayoutProblemInstance join_inst;
  join_inst.AddValue(MakeValue("t_then", "P_then", MemorySpace::Vec, false, false, {RM, CM}));
  join_inst.AddValue(MakeValue("t_else", "P_else", MemorySpace::Vec, false, false, {RM, CM}));
  join_inst.AddValue(MakeValue("t_join", "J_if", MemorySpace::Vec, false, false, {RM, CM}));
  join_inst.AddValue(MakeValue("y_out", "Cons", MemorySpace::Vec, false, false, {RM}));

  al::OperationNode p_then = MakeOp("P_then", {}, "t_then", {{{}, RM, 100.0}, {{}, CM, 1.0}}, 1, "tile.load");
  al::OperationNode p_else = MakeOp("P_else", {}, "t_else", {{{}, RM, 1.0}, {{}, CM, 100.0}}, 1, "tile.load");
  al::OperationNode j_if = MakeOp("J_if", {"t_then", "t_else"}, "t_join",
                                  {{{RM, RM}, RM, 0.0}, {{CM, CM}, CM, 0.0}}, 1, "control.if_join");
  al::OperationNode cons = MakeOp("Cons", {"t_join"}, "y_out", {{{RM}, RM, 0.0}}, 1, "tile.store");
  join_inst.operations = {p_then, p_else, j_if, cons};

  const auto join_dp = al::SolveTreewidthDP(join_inst);
  Require(join_dp.writeback_issues.empty(), "Valid if_join must have no conversion issues");
  Require(join_dp.breakdown.assignment.op_configs.at("P_then").output_layout == CM,
          "then branch must keep CM and insert a branch-yield Vec conversion to RM");
  Require(join_dp.breakdown.assignment.op_configs.at("P_else").output_layout == RM,
          "else branch must match join RM");
  Require(join_dp.breakdown.assignment.op_configs.at("J_if").output_layout == RM,
          "control.if_join output must match consumer RM");
  Require(join_dp.conversions.size() == 1, "Expected exactly one branch-yield conversion for t_then");

  // When a branch conversion into control.if_join is not cheaper than matching directly,
  // the branch switches directly to the join layout with zero conversions.
  al::LayoutProblemInstance direct_join_inst = join_inst;
  direct_join_inst.operations[0] =
      MakeOp("P_then", {}, "t_then", {{{}, RM, 1.5}, {{}, CM, 1.0}}, 1, "tile.load");
  const auto direct_dp = al::SolveTreewidthDP(direct_join_inst);
  Require(direct_dp.writeback_issues.empty(), "Direct-match if_join must have no conversion issues");
  Require(direct_dp.breakdown.assignment.op_configs.at("P_then").output_layout == RM,
          "Cheaper direct-match branch must switch to RM without conversion");
  Require(direct_dp.conversions.empty(), "Direct-match if_join must emit zero conversions");

  // With disjoint branch domains the extractor records the original join as a
  // pinned baseline. It cannot be written as a zero-cost layout conversion.
  p_then.configs = {p_then.configs.front()};
  p_else.configs = {p_else.configs.back()};
  j_if.configs.resize(1);
  j_if.configs.front().input_layouts = {RM, CM};
  join_inst.operations = {p_then, p_else, j_if, cons};
  const auto pinned = al::SolveAutoLayout(join_inst);
  Require(pinned.writeback_issues ==
              std::vector<std::string>{"Conditional join 'J_if' has incompatible branch layouts"},
          "An incompatible pinned join must prevent writeback");

  // 2) Class F: on-chip value produced with invocations=1 and consumed with invocations=4
  // prefers conversion-free match without throwing.
  al::LayoutProblemInstance loop_mismatch;
  loop_mismatch.AddValue(MakeValue("v", "Prod", MemorySpace::Vec, false, false, {RM, CM}));
  loop_mismatch.AddValue(MakeValue("w", "Cons", MemorySpace::Vec, false, false, {RM}));
  loop_mismatch.operations = {
      MakeOp("Prod", {}, "v", {{{}, RM, 2.0}, {{}, CM, 1.0}}, 4),
      MakeOp("Cons", {"v"}, "w", {{{RM}, RM, 0.0}}, 1),
  };
  const auto lm_res = al::SolveAutoLayout(loop_mismatch);
  Require(lm_res.writeback_issues.empty(), "Class F conversion-free config must succeed without issues");
  Require(lm_res.breakdown.assignment.op_configs.at("Prod").output_layout == RM,
          "Class F producer must match consumer layout");

  // 3) Class F gain-cache/evaluator agreement under budget=1 when one consumer has incompatible
  // invocations=1 and another has compatible invocations=2.
  al::LayoutProblemInstance budget1_inst;
  budget1_inst.AddValue(MakeValue("v", "prod", MemorySpace::Vec, false, false, {RM, CM}));
  budget1_inst.AddValue(MakeValue("w1", "c1", MemorySpace::Vec, false, false, {RM, CM}));
  budget1_inst.AddValue(MakeValue("w2", "c2", MemorySpace::Vec, false, false, {RM, CM}));
  budget1_inst.operations = {
      MakeOp("prod", {}, "v", {{{}, RM, 1.0}, {{}, CM, 1000.0}}, 2),
      MakeOp("c1", {"v"}, "w1", {{{CM}, CM, 1.0}, {{RM}, RM, 500.0}}, 2),
      MakeOp("c2", {"v"}, "w2", {{{RM}, RM, 1.0}, {{CM}, CM, 1.0}}, 1),
  };
  al::LayoutSolverConfig cfg_b1;
  cfg_b1.max_treewidth = 0;
  cfg_b1.max_evaluations = 1;
  const auto b1_res = al::SolveAutoLayout(budget1_inst, std::nullopt, cfg_b1);
  Require(std::isfinite(b1_res.TotalCostCycles()),
          "Class F evaluator and gain cache must agree even at max_evaluations=1");
}

}  // namespace

int main() {
  try {
    TestCostModelHardwareEquations();
    TestTreewidthStatsChainDiamondAndMultiComponent();
    TestSharedConversionBarrierAndOracle();
    TestIncrementalStateAndAsymmetricCosts();
    TestBudgetAndTreewidthLimits();
    TestIndexCapacityLimits();
    TestTreewidthAbove64AndPerComponentDPFallback();
    TestIfJoinAndClassFCrossLoopForbidden();
    std::cout << "All pypto_auto_layout_solver_test checks passed!\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed: " << e.what() << '\n';
    return 1;
  }
}
