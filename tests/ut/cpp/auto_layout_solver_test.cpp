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
  // Transposing [32, 1] to [1, 32] changes the reduction axis. Keeping
  // [32, 1] in row-major is also unaligned and cannot use a compact TMOV.
  Require(!a2a3.SupportsVecTileConversion(vec_col, al::LayoutSpec::ColMajorFlat(),
                                          al::LayoutSpec::RowMajorFlat()),
          "A compact column cannot be relaid out without changing its shape");
  vec_col.vec_conversion_headroom = 0;
  Require(!a2a3.SupportsVecTileConversion(vec_col, al::LayoutSpec::ColMajorFlat(),
                                          al::LayoutSpec::RowBcastFlat()),
          "Broadcast requires separate storage even for a compact column");
  vec_col.vec_conversion_headroom = 1023;
  Require(!a2a3.SupportsVecTileConversion(vec_col, al::LayoutSpec::ColMajorFlat(),
                                          al::LayoutSpec::RowBcastFlat()),
          "Row broadcast must budget the expanded physical tile");
  vec_col.vec_conversion_headroom = 1024;
  Require(
      a2a3.SupportsVecTileConversion(vec_col, al::LayoutSpec::ColMajorFlat(), al::LayoutSpec::RowBcastFlat()),
      "Row broadcast fits at the exact storage boundary");

  al::LayoutProblemInstance pressure;
  pressure.cost_model = a2a3;
  pressure.custom_conversion_costs[vec_col.name]
                                  [{al::LayoutSpec::ColMajorFlat(), al::LayoutSpec::RowBcastFlat()}] = 0;
  RequireNear(pressure.EstimateConversionCost(vec_col, al::LayoutSpec::ColMajorFlat(),
                                              al::LayoutSpec::RowBcastFlat()),
              0, 1e-9, "Feasible conversion may use a calibrated cost");
  vec_col.vec_conversion_headroom = 1023;
  RequireNear(pressure.EstimateConversionCost(vec_col, al::LayoutSpec::ColMajorFlat(),
                                              al::LayoutSpec::RowBcastFlat()),
              a2a3.vec_unsupported_conversion_cycles, 1e-9,
              "A price override cannot provide missing Vec storage");

  auto matrix = MakeValue("matrix", "load", MemorySpace::Vec, false, false,
                          {al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::ColMajorFlat()});
  matrix.shape = {16, 32};
  matrix.dtype = DataType::FP32;
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  matrix.vec_conversion_headroom = 2 * matrix.ByteSize() - 1;
  Require(!a2a3.SupportsVecTileConversion(matrix, row, col),
          "TTRANS requires enough space for both temporary and destination tiles");
  matrix.vec_conversion_headroom = 2 * matrix.ByteSize();
  Require(
      a2a3.SupportsVecTileConversion(matrix, row, col) && a2a3.SupportsVecTileConversion(matrix, col, row),
      "A rectangular dense tile can use either transpose/view ordering");
  RequireNear(
      a2a3.EstimateConversionCost(matrix, row, col),
      a2a3.vec_transpose_fixed_cycles + 2.0 * matrix.ByteSize() / a2a3.onchip_relayout_bytes_per_cycle, 1e-9,
      "The conversion charges transpose traffic, with a zero-copy dual view");
  matrix.is_vec_slice = true;
  Require(
      a2a3.SupportsVecTileConversion(matrix, row, col) && !a2a3.SupportsVecTileConversion(matrix, col, row),
      "A compact subview can feed TTRANS but cannot feed its transposed dual");

  auto packed = MakeValue("packed", "pack", MemorySpace::Vec, false, false,
                          {al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::ColMajorFlat()});
  packed.shape = {32, 64};
  packed.dtype = DataType::FP4E2M1X2;
  packed.current_layout = al::LayoutSpec::RowMajorFlat();
  auto boxed = packed;
  boxed.shape = {16, 128};
  boxed.current_layout = al::LayoutSpec::FractalNZ();
  Require(a5.IsValidMoveConfig(packed, boxed, packed.current_layout, boxed.current_layout),
          "The explicit packed ND-to-NZ move can retarget its byte-carrier shape");
  Require(!a5.IsValidMoveConfig(packed, boxed, al::LayoutSpec::ColMajorFlat(), boxed.current_layout),
          "An alternate packed source layout cannot reuse the validated descriptor pair");
  pressure.cost_model = a5;
  pressure.custom_conversion_costs[packed.name]
                                  [{al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::ColMajorFlat()}] = 0;
  RequireNear(
      pressure.EstimateConversionCost(packed, al::LayoutSpec::RowMajorFlat(), al::LayoutSpec::ColMajorFlat()),
      a5.vec_unsupported_conversion_cycles, 1e-9,
      "A price override cannot introduce an unsupported packed Vec conversion");

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

void TestConversionPenaltyRollbackPreservesSmallCosts() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  const auto dn = al::LayoutSpec::DN();
  al::LayoutProblemInstance instance;
  instance.forbid_unmaterializable_gm_conversions = true;
  instance.AddValue(MakeValue("input", "P", MemorySpace::DDR, false, false, {nd, nz, dn}));
  instance.AddValue(MakeValue("a", "A"));
  instance.AddValue(MakeValue("b", "B"));
  instance.custom_conversion_costs["input"][{nd, nz}] = 0.1;
  instance.operations = {
      MakeOp("P", {}, "input", {{{}, nd, 400003.9}}),
      MakeOp("A", {"input"}, "a", {{{nz}, nd, 0.0}, {{nd}, nd, 0.0}}),
      MakeOp("B", {"input"}, "b", {{{nd}, nd, 0.0}, {{dn}, nd, 0.0}}),
  };
  const al::IndexedProblem problem(instance);
  al::IncrementalState state(problem, {0, 0, 0});
  const auto original_rows = state.SourceCosts();
  const auto penalty = instance.cost_model.vec_unsupported_conversion_cycles;
  const auto verify_rollback = [&]() {
    Require(state.SourceCosts() == original_rows, "A penalty must not damage the other conversion terms");
    const double independent = al::EvaluateAssignment(instance, state.ToAssignment()).total_cost_cycles;
    RequireNear(state.Cost(), independent, 1e-9, "Rollback must agree with the independent evaluator");
    RequireNear(state.Cost(), 400004.0, 1e-9, "Keep the full normal cost after removing the penalty");
  };
  for (int i = 0; i < 2; ++i) {
    RequireNear(state.Gain(2, 1), -penalty, 1e-9, "A singleton adds only the large conversion");
    state.Apply(2, 1);
    RequireNear(state.Gain(1, 1), 0.1, 1e-12, "An active penalty cannot hide a small singleton gain");
    (void)state.SourceCosts();
    state.Apply(2, 0);
    verify_rollback();

    const std::vector<al::IncrementalState::ChoiceChange> block{{1, 1}, {2, 1}};
    const auto evaluation = state.EvaluateBlock(block);
    RequireNear(evaluation.gain, 0.1 - penalty, 1e-9, "A block replaces the small cost with the penalty");
    state.ApplyBlock(block, evaluation);
    RequireNear(state.Gain(1, 0), -0.1, 1e-12, "A small gain stays precise after a joint move");
    (void)state.SourceCosts();
    const std::vector<al::IncrementalState::ChoiceChange> undo{{1, 0}, {2, 0}};
    state.ApplyBlock(undo, state.EvaluateBlock(undo));
    verify_rollback();
  }
}

void TestFMRetainsGreedyIncumbent() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  al::LayoutProblemInstance instance;
  instance.AddValue(MakeValue("a", "A"));
  instance.AddValue(MakeValue("p", "P"));
  instance.AddValue(MakeValue("q", "Q"));
  for (const std::string name : {"a", "p"}) {
    instance.custom_conversion_costs[name] = {{{nd, nz}, 6.0}, {{nz, nd}, 6.0}};
  }
  instance.operations = {
      MakeOp("A", {}, "a", {{{}, nd, 0.0}, {{}, nz, 100.0}}),
      MakeOp("P", {"a"}, "p", {{{nd}, nd, 0.0}, {{nd}, nz, 3.0}}),
      MakeOp("Q", {"p"}, "q", {{{nd}, nd, 10.0}, {{nz}, nd, 0.0}}),
  };
  al::LayoutSolverConfig config;
  config.max_treewidth = 0;
  for (int64_t evaluations : {8, 12, 20, 100}) {
    config.max_evaluations = evaluations;
    const auto greedy = al::SolveGreedy(instance, std::nullopt, config);
    const auto fm = al::SolveGreedyFM(instance, config);
    const auto automatic = al::SolveAutoLayout(instance, std::nullopt, config);
    const auto repeated = al::SolveGreedyFM(instance, config);
    RequireNear(fm.TotalCostCycles(), greedy.TotalCostCycles(), 1e-9,
                "A losing FM excursion must retain its greedy incumbent");
    RequireNear(automatic.TotalCostCycles(), fm.TotalCostCycles(), 1e-9,
                "Auto fallback must retain the same incumbent");
    Require(fm.stats.evaluated_moves <= evaluations && automatic.stats.evaluated_moves <= evaluations &&
                fm.stats.evaluated_moves == repeated.stats.evaluated_moves &&
                fm.Assignment().op_configs == repeated.Assignment().op_configs,
            "Both searches must observe the common budget and FM must be deterministic");
  }
  RequireNear(al::SolveAutoLayout(instance, std::nullopt, config).TotalCostCycles(), 3.0, 1e-9,
              "The fallback must preserve the three-cycle layout");
}

// Test oracle for a single movable component. Keep the original ordered candidate
// scan so an FM cache change cannot silently change the selected best prefix.
std::pair<al::LayoutAssignment, int64_t> ScanFM(const al::LayoutProblemInstance& instance,
                                                const al::LayoutSolverResult& greedy,
                                                const al::LayoutSolverConfig& config) {
  const al::IndexedProblem problem(instance);
  al::IncrementalState state(problem, problem.InitialChoices(greedy.Assignment()));
  int64_t moves = greedy.stats.moves_applied;
  for (int64_t pass = 0; pass < config.max_greedy_sweeps; ++pass) {
    double best_cost = state.Cost();
    std::vector<bool> locked(problem.NumOps(), false);
    std::vector<std::pair<al::OpId, uint16_t>> history;
    size_t best_prefix = 0;
    for (size_t step = 0; step < problem.NumOps(); ++step) {
      std::optional<std::pair<al::OpId, uint16_t>> winner;
      double best_gain = 0.0;
      for (al::OpId op = 0; op < problem.NumOps(); ++op) {
        if (locked[op]) continue;
        for (uint16_t q = 0; q < problem.Ops()[op].costs.size(); ++q) {
          if (q == state.Choices()[op]) continue;
          const double gain = state.Gain(op, q);
          if (!winner || gain > best_gain + config.tolerance) {
            winner = {op, q};
            best_gain = gain;
          }
        }
      }
      if (!winner) break;
      const auto [op, choice] = *winner;
      history.emplace_back(op, state.Choices()[op]);
      state.Apply(op, choice);
      locked[op] = true;
      if (state.Cost() < best_cost - config.tolerance) {
        best_cost = state.Cost();
        best_prefix = history.size();
      }
    }
    for (size_t i = history.size(); i > best_prefix; --i) {
      const auto [op, old_choice] = history[i - 1];
      state.Apply(op, old_choice);
    }
    if (best_prefix == 0) break;
    moves += static_cast<int64_t>(best_prefix);
  }
  return {state.ToAssignment(), moves};
}

// A fixed third demand keeps the single-move cache tests independent of fan-out moves.
// Its conversion costs are zero from either movable source choice.
void PinThirdLayout(al::LayoutProblemInstance& inst) {
  const auto dn = al::LayoutSpec::DN();
  inst.values.at("shared").admissible_layouts.push_back(dn);
  inst.custom_conversion_costs["shared"][{al::LayoutSpec::ND(), dn}] = 0.0;
  inst.custom_conversion_costs["shared"][{al::LayoutSpec::NZ(), dn}] = 0.0;
  inst.AddValue(MakeValue("fixed", "F"));
  inst.operations.push_back(MakeOp("F", {"shared"}, "fixed", {{{dn}, al::LayoutSpec::ND(), 0.0}}));
}

void TestCachedFMDemandMultiplicityAndBudget() {
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  for (size_t uses : {1, 2}) {
    al::LayoutProblemInstance inst;
    inst.AddValue(MakeValue("shared", "P"));
    inst.AddValue(MakeValue("a", "A"));
    inst.AddValue(MakeValue("b", "B"));
    inst.custom_conversion_costs["shared"] = {{{ND, NZ}, 10.0}, {{NZ, ND}, 2.0}};
    const std::vector<std::string> inputs(uses, "shared");
    const std::vector<al::LayoutSpec> all_nd(uses, ND);
    const std::vector<al::LayoutSpec> all_nz(uses, NZ);
    inst.operations = {
        MakeOp("P", {}, "shared", {{{}, ND, 0.0}, {{}, NZ, 1.0}}),
        MakeOp("A", inputs, "a", {{all_nd, ND, 4.0}, {all_nz, ND, 0.0}}),
        MakeOp("B", inputs, "b", {{all_nd, ND, 4.0}, {all_nz, ND, 0.0}}),
    };
    PinThirdLayout(inst);
    al::LayoutSolverConfig config;
    config.max_greedy_sweeps = 1;
    RequireNear(al::SolveGreedy(inst, std::nullopt, config).TotalCostCycles(), 8.0, 1e-9,
                "Greedy must stop at the conversion barrier");

    // Seven seed scores, three greedy scores and one FM gain for P.
    // Moving P activates both consumers.
    // A then changes the ND demand from 2 to 1
    // (from 4 to 2 for repeats), leaving the same support but making B the last user.
    config.max_evaluations = 10;
    const auto incomplete_initial = al::SolveGreedyFM(inst, config);
    RequireNear(incomplete_initial.TotalCostCycles(), 8.0, 1e-9, "FM requires a complete initial frontier");
    Require(incomplete_initial.stats.termination_reason == "evaluation_budget", "Initial frontier budget");

    config.max_evaluations = 13;
    const auto incomplete_update = al::SolveGreedyFM(inst, config);
    RequireNear(incomplete_update.TotalCostCycles(), 7.0, 1e-9,
                "FM retains the best prefix on update budget");
    Require(incomplete_update.stats.termination_reason == "evaluation_budget",
            "Stale consumer update budget");
    Require(incomplete_update.stats.moves_applied == 2 && incomplete_update.stats.evaluated_moves == 13,
            "FM counts retained moves when the evaluation budget is exhausted");
    Require(incomplete_update.Assignment().op_configs.at("B") == inst.operations[2].configs[0],
            "FM must refresh B before choosing its move");

    config.max_evaluations = 14;
    const auto complete = al::SolveGreedyFM(inst, config);
    RequireNear(complete.TotalCostCycles(), 1.0, 1e-9, "FM must take all three coordinated moves");
    Require(complete.stats.moves_applied == 3 && complete.stats.evaluated_moves == 14,
            "FM charges the initial frontier and affected inputs only");

    // The fourth operation is connected through B's output and never profitable.
    // Its suffix move must be undone, and the next pass must start from that rollback.
    inst.AddValue(MakeValue("tail", "T"));
    inst.operations.push_back(MakeOp("T", {"b"}, "tail", {{{ND}, ND, 0.0}, {{ND}, ND, 100.0}}));
    config.max_evaluations = 1000;
    config.max_greedy_sweeps = 3;
    const auto rollback = al::SolveGreedyFM(inst, config);
    const auto [expected, moves] = ScanFM(inst, al::SolveGreedy(inst, std::nullopt, config), config);
    RequireNear(rollback.TotalCostCycles(), 1.0, 1e-9, "FM must undo the unprofitable tail move");
    Require(rollback.Assignment().op_configs == expected.op_configs && rollback.stats.moves_applied == moves,
            "Cached FM must agree with the scan after suffix rollback and restart");
    Require(rollback.stats.termination_reason == "fm_no_improving_prefix",
            "An empty prefix from the attempted seeds is not a local optimum certificate");
  }
}

void TestCachedFMToleranceScanOrder() {
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("shared", "P"));
  inst.AddValue(MakeValue("a", "A"));
  inst.AddValue(MakeValue("b", "B"));
  inst.AddValue(MakeValue("tail", "T"));
  inst.custom_conversion_costs["shared"] = {{{ND, NZ}, 10.0}, {{NZ, ND}, 3.0}};
  inst.operations = {
      MakeOp("A", {"shared"}, "a", {{{ND}, ND, 5.0}, {{NZ}, ND, 0.0}}),
      MakeOp("B", {"shared"}, "b", {{{ND}, ND, 6.25}, {{NZ}, ND, 0.5}, {{NZ}, ND, 0.0}}),
      MakeOp("P", {}, "shared", {{{}, ND, 0.0}, {{}, NZ, 2.0}}),
      MakeOp("T", {"a"}, "tail", {{{ND}, ND, 0.0}, {{ND}, ND, 3.0}}),
  };
  // Moving the zero-cost seed opens the same ordered A, B, P, T frontier as the oracle.
  inst.AddValue(MakeValue("seed", "S"));
  for (auto& op : inst.operations) {
    op.inputs.push_back("seed");
    for (auto& cfg : op.configs) cfg.input_layouts.push_back(ND);
  }
  inst.operations.insert(inst.operations.begin(), MakeOp("S", {}, "seed", {{{}, ND, 0.0}, {{}, ND, 0.0}}));
  inst.operations.front().configs[1].notes = "alternate seed";
  PinThirdLayout(inst);
  al::LayoutSolverConfig config;
  config.max_greedy_sweeps = 1;
  config.tolerance = 1.0;
  const auto greedy = al::SolveGreedy(inst, std::nullopt, config);
  RequireNear(greedy.TotalCostCycles(), 11.25, 1e-9, "Greedy must stop before the tolerance test");
  const auto [expected, moves] = ScanFM(inst, greedy, config);
  const auto result = al::SolveGreedyFM(inst, config);
  // A first sets the incumbent to -5. B's -4.25 is skipped and its -3.75
  // becomes the incumbent. T's -3 cannot then replace B. Selecting T first
  // would keep its extra cost in the best prefix as A, B and P recover the loss.
  RequireNear(result.TotalCostCycles(), 2.0, 1e-9, "FM must keep the ordered tolerance comparison");
  Require(result.Assignment().op_configs == expected.op_configs && result.stats.moves_applied == moves,
          "Cached FM must match the scan across tolerance ties within and across operations");
}

void TestFMExpandsOneHopPerMove() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("a", "A"));
  inst.AddValue(MakeValue("b", "B"));
  inst.AddValue(MakeValue("c", "C"));
  inst.custom_conversion_costs["a"] = {{{nd, nz}, 10.0}, {{nz, nd}, 0.0}};
  inst.custom_conversion_costs["b"] = {{{nd, nz}, 10.0}, {{nz, nd}, 0.0}};
  inst.operations = {
      MakeOp("A", {}, "a", {{{}, nd, 0.0}, {{}, nz, 1.0}}),
      MakeOp("B", {"a"}, "b", {{{nd}, nd, 5.0}, {{nz}, nz, 0.0}}),
      MakeOp("C", {"b"}, "c", {{{nd}, nd, 8.0}, {{nz}, nz, 0.0}}),
  };

  al::LayoutSolverConfig cfg;
  cfg.max_greedy_sweeps = 1;
  // Six seed scores and three greedy scores leave all values in ND. The
  // remaining two scores permit the uphill A move and then profitable B.
  // C only becomes profitable and is activated after B moves.
  cfg.max_evaluations = 11;
  const auto limited = al::SolveGreedyFM(inst, cfg);
  const auto& selected = limited.Assignment().op_configs;
  Require(selected.at("A").output_layout == nz && selected.at("B").output_layout == nz &&
              selected.at("C").output_layout == nd && limited.stats.moves_applied == 2,
          "A move must open only its immediate neighbors, without eagerly scoring remote C");
  ++cfg.max_evaluations;
  const auto complete = al::SolveGreedyFM(inst, cfg);
  Require(complete.Assignment().op_configs.at("C").output_layout == nz && complete.stats.moves_applied == 3,
          "B must bring C into the active set");
}

void TestFanoutBlockEscapesFMLocks() {
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("shared"));
  inst.custom_conversion_costs["shared"] = {{{ND, NZ}, 10.0}};
  inst.forbid_unmaterializable_gm_conversions = true;
  inst.AddValue(MakeValue("a", "A"));
  inst.operations.push_back(MakeOp("A", {"shared", "shared"}, "a",
                                   {{{ND, ND}, ND, 3.0},
                                    {{ND, ND}, ND, 3.25},
                                    {{NZ, ND}, ND, 0.0},
                                    {{NZ, NZ}, ND, 2.0},
                                    {{NZ, NZ}, ND, 0.0}}));
  for (int i = 0; i < 3; ++i) {
    const std::string id = "B" + std::to_string(i);
    const std::string output = "b" + std::to_string(i);
    inst.AddValue(MakeValue(output, id));
    inst.operations.push_back(
        MakeOp(id, {"shared"}, output, {{{ND}, ND, 3.0}, {{ND}, ND, 3.25}, {{NZ}, ND, 0.0}}));
  }

  al::LayoutSolverConfig config;
  config.max_greedy_sweeps = 1;
  config.max_treewidth = 0;
  config.collect_treewidth_stats = false;
  const auto greedy = al::SolveGreedy(inst, std::nullopt, config);
  RequireNear(greedy.TotalCostCycles(), 12.0, 1e-9, "Greedy cannot afford the shared pack alone");
  // FM locks each consumer into its less costly ND alternative and rolls back
  // the whole pass. Switching every consumer to NZ avoids the GM multi-rep
  // penalty and replaces four three-cycle operator costs with one shared pack.
  const auto [scan_assignment, scan_moves] = ScanFM(inst, greedy, config);
  Require(scan_moves == 0 && scan_assignment.op_configs == greedy.Assignment().op_configs,
          "Single-op FM must roll back every move");
  const auto fm = al::SolveGreedyFM(inst, config);
  RequireNear(fm.TotalCostCycles(), BruteForceExactBest(inst), 1e-9,
              "Fan-out block must reach the exact optimum");
  RequireNear(fm.TotalCostCycles(), 10.0, 1e-9, "Charge the NZ conversion once for the whole block");
  Require(fm.Assignment().op_configs.at("A") == inst.operations[0].configs[4],
          "Every occurrence must request NZ and the cheaper matching configuration must win");
  Require(fm.stats.moves_applied == 4 && fm.conversions.size() == 1 &&
              fm.conversions[0].consumer_op_ids.size() == 4,
          "The block must change four distinct consumers and list each once");
  const auto automatic = al::SolveAutoLayout(inst, std::nullopt, config);
  Require(automatic.Assignment().op_configs == fm.Assignment().op_configs,
          "Auto fallback must consider the fan-out block");

  // Fourteen seed scores, ten greedy scores, four single moves at seed A,
  // two NZ choices to rank for A and four members to score.
  config.max_evaluations = 14 + 10 + 4 + 2 + 4;
  const auto first = al::SolveGreedyFM(inst, config);
  Require(first.Assignment().op_configs == fm.Assignment().op_configs && first.stats.moves_applied == 4,
          "Fan-out must be able to win the very first decision with only A active");
  --config.max_evaluations;
  const auto limited = al::SolveGreedyFM(inst, config);
  Require(limited.stats.termination_reason == "evaluation_budget" &&
              limited.stats.evaluated_moves <= config.max_evaluations && limited.stats.moves_applied == 0 &&
              limited.Assignment().op_configs == greedy.Assignment().op_configs,
          "A partly scored group must not change any consumer");

  inst.AddValue(MakeValue("fixed", "F"));
  inst.operations.push_back(MakeOp("F", {"shared"}, "fixed", {{{ND}, ND, 0.0}}));
  config.max_evaluations = 100000;
  const auto pinned = al::SolveGreedyFM(inst, config);
  RequireNear(pinned.TotalCostCycles(), 12.0, 1e-9, "A pinned ND consumer rules out a common NZ layout");
  Require(pinned.stats.moves_applied == 0 && pinned.conversions.empty(),
          "The move must include demands from pinned consumers");
}

void TestSeedComparesPositiveSingleAndFanoutGains() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("shared"));
  inst.custom_conversion_costs["shared"][{nd, nz}] = 10.0;
  inst.AddValue(MakeValue("a", "A"));
  inst.operations.push_back(
      MakeOp("A", {"shared"}, "a", {{{nd}, nd, 3.0}, {{nd}, nz, 3.5}, {{nz}, nd, 0.0}}));
  for (const std::string id : {"B", "C", "D"}) {
    inst.AddValue(MakeValue(id, id));
    inst.operations.push_back(MakeOp(id, {"shared"}, id, {{{nd}, nd, 3.0}, {{nz}, nd, 0.0}}));
  }
  inst.AddValue(MakeValue("out", "read"));
  inst.operations.push_back(MakeOp("read", {"a"}, "out", {{{nz}, nd, 0.0}}));

  al::LayoutSolverConfig config;
  config.max_greedy_sweeps = 1;
  // Ten topological and five greedy scores, then two single changes at A
  // and four grouped consumers. No neighbor is evaluated as a singleton by FM.
  config.max_evaluations = 10 + 5 + 2 + 4;
  for (const double cost : {2.0, 2.5, 3.0}) {
    inst.custom_conversion_costs["a"][{nd, nz}] = cost;
    const auto result = al::SolveGreedyFM(inst, config);
    // Unifying the shared input saves 12 - 10 = 2. Changing A's output saves cost - 0.5.
    const bool group_wins = cost < 2.5;
    Require(result.stats.moves_applied == (group_wins ? 5 : 1),
            "Keep the greedy move and compare its cost with the first FM prefix");
    Require(result.Assignment().op_configs.at("A") == inst.operations.front().configs[group_wins ? 2 : 1],
            "The highest-gain candidate must win the first decision");
  }
}

void TestCapacityRepairOfExplicitTranspose() {
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  al::LayoutProblemInstance instance;
  instance.AddValue(MakeValue("a", "load", MemorySpace::Vec));
  instance.AddValue(MakeValue("b", "move", MemorySpace::Vec));
  instance.operations = {
      MakeOp("load", {}, "a", {{{}, col, 0}}, 1, "tile.load"),
      MakeOp("move", {"a"}, "b", {{{col}, row, 0}, {{col}, col, 10}}, 1, "tile.move"),
  };
  al::VecMemoryRegion region{};
  region.capacity = 1024;
  region.operation_positions = {1, 3};
  region.original_bytes = {0, 512, 512, 1024};
  instance.vec_memory_regions.push_back(region);
  al::LayoutAssignment initial;
  for (const auto& op : instance.operations) initial.op_configs.emplace(op.op_id, op.configs.front());
  Require(al::CheckVecConversionCapacity(instance, initial).size() == 1,
          "Two 512-byte tiles leave no space for transpose scratch");
  for (const auto& result : {al::SolveAutoLayout(instance, initial), al::SolveTreewidthDP(instance),
                             al::SolveGreedy(instance, initial), al::SolveGreedyFM(instance)}) {
    Require(result.writeback_issues.empty() && result.Assignment().op_configs.at("move").output_layout == col,
            "Repair the explicit move even though the load has no declared-layout configuration");
  }
}

void TestFMRestartsAtLateFanout() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.cost_model.expected_uses = 10;
  inst.forbid_unmaterializable_gm_conversions = true;
  auto weight = MakeValue("weight", std::nullopt, MemorySpace::DDR, true);
  weight.shape = {128, 128};
  inst.AddValue(weight);
  for (const std::string id : {"A", "B", "C", "D", "E"}) inst.AddValue(MakeValue(id, id));
  inst.operations = {
      MakeOp("A", {}, "A", {{{}, nd, 10.0}, {{}, nd, 80.0}}, 10),
      MakeOp("B", {"A"}, "B", {{{nd}, nd, 10.0}, {{nd}, nd, 80.0}}, 10),
      MakeOp("C", {"B"}, "C", {{{nd}, nd, 10.0}, {{nd}, nd, 80.0}}, 10),
      MakeOp("D", {"C", "weight"}, "D", {{{nd, nd}, nd, 70.0}, {{nd, nz}, nd, 10.0}}, 10),
      MakeOp("E", {"C", "weight"}, "E", {{{nd, nd}, nd, 70.0}, {{nd, nz}, nd, 10.0}}, 10),
  };

  al::LayoutSolverConfig config;
  config.max_treewidth = 0;
  config.max_greedy_sweeps = 1;
  const auto greedy = al::SolveGreedy(inst, std::nullopt, config);
  const auto first = al::SolveGreedyFM(inst, config);
  // The group saves 568 cycles but reaching it from A or the midpoint C
  // first locks in a 700-cycle increase. One FM pass must keep the incumbent.
  RequireNear(first.TotalCostCycles(), 1700.0, 1e-9, "An upstream seed misses the profitable prefix");
  Require(first.Assignment().op_configs == greedy.Assignment().op_configs &&
              first.stats.termination_reason == "greedy_sweep_budget",
          "A skipped fan-out restart must count as sweep budget exhaustion");

  config.max_greedy_sweeps = 2;
  const auto fm = al::SolveGreedyFM(inst, config);
  const auto exact = al::SolveTreewidthDP(inst);
  RequireNear(fm.TotalCostCycles(), 1132.0, 1e-9, "Restart at the shared weight before locking the chain");
  Require(fm.Assignment().op_configs == exact.Assignment().op_configs && fm.stats.moves_applied == 2,
          "Only the two weight consumers belong to the best prefix");
  Require(al::SolveGreedyFM(inst).Assignment().op_configs == fm.Assignment().op_configs,
          "The default solver must find the fan-out improvement");
  Require(
      al::SolveAutoLayout(inst, std::nullopt, config).Assignment().op_configs == fm.Assignment().op_configs,
      "Auto fallback must use the fan-out restart");

  // Restarting needs a singleton score and two block scores. Leave the
  // restart one score short of considering the whole block.
  config.max_evaluations = first.stats.evaluated_moves + 2;
  const auto limited = al::SolveGreedyFM(inst, config);
  Require(limited.stats.termination_reason == "evaluation_budget" &&
              limited.stats.evaluated_moves <= config.max_evaluations && limited.stats.moves_applied == 0 &&
              limited.Assignment().op_configs == greedy.Assignment().op_configs,
          "Exhaustion before scoring the whole restart block must keep the greedy incumbent");
  ++config.max_evaluations;
  const auto retained = al::SolveGreedyFM(inst, config);
  Require(retained.stats.termination_reason == "evaluation_budget" &&
              retained.stats.evaluated_moves == config.max_evaluations && retained.stats.moves_applied == 2 &&
              retained.Assignment().op_configs == fm.Assignment().op_configs,
          "Exhaustion after scoring the restart block must keep the new best prefix");
}

void TestUphillFanoutMovesAndRollsBackAtomically() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("value", "P"));
  inst.custom_conversion_costs["value"] = {{{nd, nz}, 10.0}, {{nz, nd}, 2.0}};
  inst.operations.push_back(MakeOp("P", {}, "value", {{{}, nd, 0.0}, {{}, nz, 1.0}}));
  for (const std::string id : {"A", "B"}) {
    inst.AddValue(MakeValue(id, id));
    inst.operations.push_back(MakeOp(id, {"value"}, id, {{{nd}, nd, 4.0}, {{nz}, nd, 0.0}}));
  }
  // P is the seed. Moving P alone costs 3; switching both its consumers costs 2.
  // The winning group must survive as a single speculative move until P is rescored.
  al::LayoutSolverConfig config;
  config.max_greedy_sweeps = 1;
  config.max_evaluations = 6 + 3 + 1 + 2;
  const auto interrupted = al::SolveGreedyFM(inst, config);
  RequireNear(interrupted.TotalCostCycles(), 8.0, 1e-9, "Roll back the entire uphill group on exhaustion");
  Require(interrupted.stats.moves_applied == 0, "Neither group member belongs to the best prefix yet");
  ++config.max_evaluations;
  const auto completed = al::SolveGreedyFM(inst, config);
  RequireNear(completed.TotalCostCycles(), 1.0, 1e-9, "Rescoring P makes the joint prefix profitable");
  Require(completed.stats.moves_applied == 3, "Both consumers and P belong to the retained prefix");
}

void TestFMReusesShrinkingFanoutGain() {
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  constexpr int consumers = 128;
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("shared"));
  inst.custom_conversion_costs["shared"][{nd, nz}] = 10.0;
  for (int i = 0; i < consumers; ++i) {
    const auto name = "consumer" + std::to_string(i);
    inst.AddValue(MakeValue(name, name));
    inst.operations.push_back(MakeOp(name, {"shared"}, name, {{{nd}, nd, 1.0}, {{nz}, nz, 0.5}}));
  }
  al::LayoutSolverConfig config;
  config.max_evaluations = 12 * consumers;
  const auto result = al::SolveGreedyFM(inst, config);
  RequireNear(result.TotalCostCycles(), 10.0 + 0.5 * consumers, 1e-9, "Unify the wide fan-out");
  // On the second pass each single move toward ND leaves the all-ND group's
  // final assignment unchanged. Re-evaluating its remaining members costs quadratic time.
  Require(result.stats.termination_reason == "fm_no_improving_prefix",
          "The shrinking fan-out must fit a linear budget");
}

void TestFanoutBlockChangesSourceAndDemandTogether() {
  const auto ND = al::LayoutSpec::ND();
  const auto NZ = al::LayoutSpec::NZ();
  al::LayoutProblemInstance inst;
  inst.AddValue(MakeValue("shared"));
  inst.AddValue(MakeValue("between", "A"));
  inst.AddValue(MakeValue("out", "B"));
  inst.custom_conversion_costs["shared"] = {{{ND, NZ}, 10.0}};
  inst.custom_conversion_costs["between"] = {{{ND, NZ}, 30.0}, {{NZ, ND}, 40.0}};
  inst.operations = {
      MakeOp("A", {"shared"}, "between", {{{ND}, ND, 12.0}, {{NZ}, NZ, 0.0}}),
      MakeOp("B", {"shared", "between"}, "out", {{{ND, ND}, ND, 12.0}, {{NZ, NZ}, ND, 0.0}}),
  };
  const al::IndexedProblem problem(inst);
  al::IncrementalState state(problem, {0, 0});
  Require(state.Gain(0, 1) < 0.0 && state.Gain(1, 1) < 0.0,
          "Either move alone introduces a conversion of the intermediate value");
  // A changes the source of 'between' and B changes its demand. The final
  // state needs neither the ND-to-NZ nor the NZ-to-ND intermediate conversion.
  Require(state.ApplyImprovingBlock({{0, 1}, {1, 1}}, 1e-9),
          "The block must score the new source against the new demand");
  const al::IncrementalState rebuilt(problem, {1, 1});
  RequireNear(state.Cost(), 10.0, 1e-9, "Only the shared input conversion remains");
  Require(!state.ApplyImprovingBlock({{0, 0}, {1, 0}}, 1e-9), "The reverse block increases the total cost");
  RequireNear(state.Cost(), rebuilt.Cost(), 1e-9, "Incremental block cost must match the rebuilt state");
  Require(state.Choices() == rebuilt.Choices() && state.Sources() == rebuilt.Sources() &&
              state.Counts() == rebuilt.Counts() && state.DemandMasks() == rebuilt.DemandMasks() &&
              state.SourceCosts() == rebuilt.SourceCosts(),
          "Only the improving block may change choices or factor caches");
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
  cfg_tw1.collect_treewidth_stats = false;
  const auto fast = al::SolveAutoLayout(mixed, std::nullopt, cfg_tw1);
  Require(!fast.treewidth_stats && fast.stats.active_treewidth == -1, "Omitted width must be explicit");
  Require(fast.Assignment().op_configs == result.Assignment().op_configs &&
              fast.stats.evaluated_moves == result.stats.evaluated_moves,
          "Skipping a proven-wide clique must preserve both the fallback and the separate exact component");
  RequireNear(al::SolveGreedyFM(mixed, cfg_tw1).TotalCostCycles(), optimum, 1e-9,
              "FM must cross the shared conversion barrier with no DP");
  bool rejected_clique = false;
  try {
    (void)al::SolveTreewidthDP(mixed, cfg_tw1);
  } catch (const pypto::ValueError&) {
    rejected_clique = true;
  }
  Require(rejected_clique, "Strict DP must still reject a component skipped by the clique bound");
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
    TestConversionPenaltyRollbackPreservesSmallCosts();
    TestFMRetainsGreedyIncumbent();
    TestCachedFMDemandMultiplicityAndBudget();
    TestCachedFMToleranceScanOrder();
    TestFMExpandsOneHopPerMove();
    TestFanoutBlockEscapesFMLocks();
    TestSeedComparesPositiveSingleAndFanoutGains();
    TestFMRestartsAtLateFanout();
    TestCapacityRepairOfExplicitTranspose();
    TestUphillFanoutMovesAndRollsBackAtomically();
    TestFMReusesShrinkingFanoutGain();
    TestFanoutBlockChangesSourceAndDemandTogether();
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
