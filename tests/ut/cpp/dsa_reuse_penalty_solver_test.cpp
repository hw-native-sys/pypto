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

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace dsa = pypto::ir::dsa;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const dsa::DsaSolution& RequireSolution(const dsa::DsaResult& result, const std::string& message) {
  if (!result.solution.has_value()) throw std::runtime_error(message);
  return result.solution.value();
}

uint64_t OffsetOf(const dsa::DsaSolution& solution, dsa::BufferId id) {
  const uint64_t* offset = solution.Find(id);
  Require(offset != nullptr, "expected buffer " + std::to_string(id) + " to be placed");
  return *offset;
}

void TestFeasiblePackingAndTemporalConflict() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
      {2, 64, 32, 0, {6, 8}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected a feasible temporal packing");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "lifetime-overlapping buffers must be disjoint");
  Require(result.objective.max_peak <= 128, "feasible placement must respect capacity");
  Require(dsa::ValidateSolution(problem, RequireSolution(result, "expected solution")).empty(),
          "solver result must pass independent validation");
}

void TestNoFitDoesNotProveInfeasibility() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 16, {}}};
  problem.buffers = {
      {0, 6, 2, 0, {3, 5}}, {1, 6, 2, 0, {2, 7}}, {2, 4, 4, 0, {4, 5}},
      {3, 8, 4, 0, {0, 3}}, {4, 2, 2, 0, {5, 7}}, {5, 1, 1, 0, {5, 10}},
  };

  // Independently verified strict witness. Canonical greedy's bounded order
  // set misses it; kNoFit therefore means search failure, not infeasibility.
  dsa::DsaSolution witness;
  witness.offsets = {{0, 4}, {1, 10}, {2, 0}, {3, 0}, {4, 0}, {5, 2}};
  Require(dsa::ValidateSolution(problem, witness).empty(),
          "strict witness must prove that the characterization instance fits");

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kNoFit && !result.solution.has_value(),
          "bounded canonical greedy is expected to miss the strict witness");
}

void TestExplicitHardSeparation() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
  };
  problem.separations = {{0, 1}};

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected separated lifetime-disjoint buffers to fit");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "explicit separation must prevent address reuse");
}

void TestWeightedPenaltyAvoidance() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
      {2, 64, 32, 0, {4, 6}},
  };
  problem.reuse_penalties = {
      {0, 1, 10},
      {0, 2, 1},
      {1, 2, 1},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected weighted instance to fit");
  Require(result.objective.reuse_cost == 1, "canonical greedy must avoid the weight-10 overlap");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "the highest-weight pair should be separated");
  Require(dsa::EvaluateObjective(problem, RequireSolution(result, "expected solution")).reuse_cost == 1,
          "independent objective evaluation must match the solver result");
}

void TestCapacityNoFit() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {0, 4}},
      {2, 64, 32, 0, {0, 4}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kNoFit && !result.solution.has_value(),
          "three co-live 64-byte buffers cannot fit in 128 bytes");
}

void TestAlignmentAndReservedRanges() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 256, {{0, 32}, {96, 128}}}};
  problem.buffers = {
      {0, 48, 32, 0, {0, 2}},
      {1, 64, 64, 0, {0, 2}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected aligned placement around reserved ranges");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) % 32 == 0,
          "buffer 0 must satisfy 32-byte alignment");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 1) % 64 == 0,
          "buffer 1 must satisfy 64-byte alignment");
  Require(dsa::ValidateSolution(problem, RequireSolution(result, "expected solution")).empty(),
          "aligned placement must avoid both reserved ranges");
}

void TestDeterministicResult() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 192, {}}};
  problem.buffers = {
      {7, 64, 32, 0, {0, 2}},
      {3, 64, 32, 0, {2, 4}},
      {9, 64, 32, 0, {4, 6}},
      {1, 64, 32, 0, {6, 8}},
  };
  problem.reuse_penalties = {
      {7, 3, 5},
      {7, 9, 2},
      {3, 1, 4},
      {9, 1, 1},
  };
  const dsa::CanonicalGreedyOptions options{17, 5};

  const dsa::DsaResult first = dsa::CanonicalGreedySolver(options).Solve(problem);
  const dsa::DsaResult second = dsa::CanonicalGreedySolver(options).Solve(problem);
  Require(first.status == dsa::SolveStatus::kFeasible && first.solution.has_value(),
          "first deterministic solve must be feasible");
  Require(second.status == dsa::SolveStatus::kFeasible && second.solution.has_value(),
          "second deterministic solve must be feasible");
  Require(RequireSolution(first, "expected first solution").offsets ==
              RequireSolution(second, "expected second solution").offsets,
          "same seed and problem must produce identical offsets");
  Require(first.objective.reuse_cost == second.objective.reuse_cost,
          "same seed and problem must produce identical objective");
  Require(first.statistics.candidate_offsets_evaluated == second.statistics.candidate_offsets_evaluated,
          "deterministic solve must evaluate the same number of candidates");
}

void TestSecondaryPeakObjective() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 13, {}}};
  problem.buffers = {
      {0, 5, 1, 0, {4, 8}},
      {1, 5, 1, 0, {5, 8}},
      {2, 2, 1, 0, {5, 8}},
      {3, 6, 1, 0, {3, 5}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "secondary-objective instance must fit");
  Require(result.objective.reuse_cost == 0, "instance has no reuse penalties");
  Require(result.objective.total_peak == 12 && result.objective.max_peak == 12,
          "canonical greedy must retain the lowest peak found on a reuse-cost tie");
}

void TestRejectsPenaltyOnLifetimeConflict() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
  };
  problem.reuse_penalties = {{0, 1, 1}};

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kInvalidProblem && !result.solution.has_value(),
          "soft edges on lifetime-conflicting buffers must be rejected");
}

void TestPipelineOnlyRelaxation() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 128, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
      {2, 64, 32, 0, {3, 5}},
  };
  strict.separations = {{0, 1}, {0, 2}};

  const dsa::DsaResult strict_result = dsa::CanonicalGreedySolver().Solve(strict);
  Require(strict_result.status == dsa::SolveStatus::kNoFit,
          "three pairwise-separated buffers cannot fit in two slots");

  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}}, 1);
  Require(relaxed.separations.size() == 1 && relaxed.separations[0].first == 0 &&
              relaxed.separations[0].second == 2,
          "only the selected pipeline relation may be relaxed");
  Require(relaxed.reuse_penalties.size() == 1 && relaxed.reuse_penalties[0].first == 0 &&
              relaxed.reuse_penalties[0].second == 1 && relaxed.reuse_penalties[0].weight == 1,
          "the relaxed pipeline relation must become a unit penalty");

  const dsa::DsaResult relaxed_result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(relaxed_result.status == dsa::SolveStatus::kFeasible && relaxed_result.solution.has_value(),
          "pipeline-only relaxation must recover a capacity-fitting placement");
  Require(
      dsa::CountOverlappingPairs(strict, RequireSolution(relaxed_result, "expected solution"), {{0, 1}}) == 1,
      "the selected fallback activates exactly the relaxed pair");
  Require(dsa::ValidateSolution(relaxed, RequireSolution(relaxed_result, "expected solution")).empty(),
          "the relaxed placement must retain every non-pipeline hard constraint");
}

void TestRelaxedPairCanRemainInactive() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 128, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
  };
  strict.separations = {{0, 1}};
  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}}, 1);
  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "relaxed two-buffer problem must fit");
  Require(result.objective.reuse_cost == 0 &&
              dsa::CountOverlappingPairs(strict, RequireSolution(result, "expected solution"), {{0, 1}}) == 0,
          "a relaxed relation emits no warning when the selected placement still separates it");
}

void TestPipelineRelaxationIgnoresLifetimeConflict() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 192, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
      {2, 64, 32, 0, {6, 8}},
  };
  strict.separations = {{0, 1}, {1, 2}};

  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}, {1, 2}}, 1);
  Require(relaxed.separations.empty(), "selected typed separations must be removed");
  Require(relaxed.reuse_penalties.size() == 1 && relaxed.reuse_penalties[0].first == 1 &&
              relaxed.reuse_penalties[0].second == 2,
          "only the lifetime-compatible pipeline relation may become a soft edge");
  Require(dsa::ValidateProblem(relaxed).empty(),
          "mixed co-live/disjoint pipeline fallback must remain a valid DSA-RP problem");

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "mixed pipeline fallback must remain solvable");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "ordinary lifetime interference must remain hard after typed relaxation");
}

void TestIndependentValidationRejectsCorruption() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {0, 4}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected a valid baseline placement");

  dsa::DsaSolution corrupted = RequireSolution(result, "expected solution");
  corrupted.offsets[1] = corrupted.offsets[0];
  Require(!dsa::ValidateSolution(problem, corrupted).empty(),
          "independent validation must reject an overlapping hard conflict");

  corrupted = RequireSolution(result, "expected solution");
  corrupted.offsets[0] = 1;
  Require(!dsa::ValidateSolution(problem, corrupted).empty(),
          "independent validation must reject misalignment");
}

}  // namespace

int main() {
  try {
    TestFeasiblePackingAndTemporalConflict();
    TestNoFitDoesNotProveInfeasibility();
    TestExplicitHardSeparation();
    TestWeightedPenaltyAvoidance();
    TestCapacityNoFit();
    TestAlignmentAndReservedRanges();
    TestDeterministicResult();
    TestSecondaryPeakObjective();
    TestRejectsPenaltyOnLifetimeConflict();
    TestPipelineOnlyRelaxation();
    TestRelaxedPairCanRemainInactive();
    TestPipelineRelaxationIgnoresLifetimeConflict();
    TestIndependentValidationRejectsCorruption();
  } catch (const std::exception& error) {
    std::cerr << "DSA-RP solver test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
