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

#ifndef PYPTO_IR_TRANSFORMS_DSA_DSA_REUSE_PENALTY_SOLVER_H_
#define PYPTO_IR_TRANSFORMS_DSA_DSA_REUSE_PENALTY_SOLVER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {
namespace dsa {

/**
 * @file
 * @brief IR-independent capacity-constrained DSA with weighted reuse penalties.
 *
 * The model is deliberately limited to the structure produced by PyPTO after
 * semantic aliases have been materialized: each buffer belongs to one fixed
 * memory pool and has one conservative half-open lifetime. Correctness
 * constraints are hard. A reuse penalty is activated when two lifetime-
 * compatible buffers occupy overlapping byte ranges.
 *
 * For R configured orders, the constructive search runs in
 * O(R * n^2 log n): each buffer considers O(n) canonical boundary offsets,
 * while hard-range feasibility and soft-overlap scoring are indexed. The
 * input hard/soft relation graph can itself contain O(n^2) edges. This is a
 * storage-allocation algorithm over tens to low hundreds of buffers, not a
 * nested traversal over PyPTO IR.
 */

using BufferId = uint32_t;
using PoolId = int32_t;

struct Interval {
  int64_t begin = 0;
  int64_t end = 0;

  [[nodiscard]] bool Overlaps(const Interval& other) const noexcept {
    return begin < other.end && other.begin < end;
  }
};

struct AddressRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct Pool {
  PoolId id = 0;
  uint64_t capacity = 0;
  std::vector<AddressRange> reserved_ranges;
};

struct Buffer {
  BufferId id = 0;
  uint64_t size = 0;
  uint64_t alignment = 1;
  PoolId pool = 0;
  Interval lifetime;
};

struct Separation {
  BufferId first = 0;
  BufferId second = 0;
};

struct ReusePenalty {
  BufferId first = 0;
  BufferId second = 0;
  uint64_t weight = 0;
};

struct DsaProblem {
  std::vector<Pool> pools;
  std::vector<Buffer> buffers;
  std::vector<Separation> separations;
  std::vector<ReusePenalty> reuse_penalties;
};

struct DsaSolution {
  std::map<BufferId, uint64_t> offsets;

  [[nodiscard]] const uint64_t* Find(BufferId id) const noexcept {
    const auto found = offsets.find(id);
    return found == offsets.end() ? nullptr : &found->second;
  }
};

struct ObjectiveValue {
  uint64_t reuse_cost = 0;
  uint64_t total_peak = 0;
  uint64_t max_peak = 0;
  std::map<PoolId, uint64_t> peak_by_pool;
};

enum class SolveStatus {
  kFeasible,
  // The bounded constructive search found no capacity-fitting placement. This
  // is not a mathematical infeasibility proof.
  kNoFit,
  kInvalidProblem,
};

struct SolverStatistics {
  uint64_t candidate_offsets_evaluated = 0;
  size_t orders_evaluated = 0;
  size_t selected_order = 0;
  bool first_fit_seed_feasible = false;
  bool selected_first_fit_seed = false;
};

struct DsaResult {
  SolveStatus status = SolveStatus::kInvalidProblem;
  std::optional<DsaSolution> solution;
  ObjectiveValue objective;
  SolverStatistics statistics;
  std::vector<std::string> diagnostics;
};

struct CanonicalGreedyOptions {
  uint64_t seed = 0;
  size_t random_restarts = 4;
};

/**
 * @brief Deterministic canonical greedy solver for the PyPTO DSA-RP model.
 *
 * For each buffer, the solver tests offset zero, every reserved-range end, and
 * the aligned tops of placed hard or soft neighbors. It rejects candidates
 * that violate hard constraints or capacity, then chooses the minimum
 * incremental reuse cost and lowest offset. It evaluates size-, birth-, and
 * incident-weight orders plus deterministic seeded restarts.
 *
 * A capacity-fitting penalty-blind first-fit result is retained as an
 * incumbent. Consequently, canonical greedy never discards a placement known
 * to fit merely because one of its locally greedy orders gets stuck.
 */
class CanonicalGreedySolver {
 public:
  explicit CanonicalGreedySolver(CanonicalGreedyOptions options = {});

  [[nodiscard]] DsaResult Solve(const DsaProblem& problem) const;

 private:
  CanonicalGreedyOptions options_;
};

/**
 * @brief Validate the model independently of the placement algorithm.
 *
 * An empty result means the problem is well formed.
 */
[[nodiscard]] std::vector<std::string> ValidateProblem(const DsaProblem& problem);

/**
 * @brief Recompute every structural and capacity invariant for a solution.
 *
 * An empty result means the solution is valid.
 */
[[nodiscard]] std::vector<std::string> ValidateSolution(const DsaProblem& problem,
                                                        const DsaSolution& solution);

/**
 * @brief Recompute activated reuse cost and per-pool peaks from geometry.
 */
[[nodiscard]] ObjectiveValue EvaluateObjective(const DsaProblem& problem, const DsaSolution& solution);

/**
 * @brief Replace selected hard separations with weighted soft penalties.
 *
 * The caller is responsible for selecting only policy-relaxable relations.
 * All other hard separations remain unchanged.
 */
[[nodiscard]] DsaProblem RelaxSeparationsToPenalties(const DsaProblem& problem,
                                                     const std::vector<Separation>& relaxable,
                                                     uint64_t weight);

/**
 * @brief Count selected pairs whose placed byte ranges overlap.
 */
[[nodiscard]] size_t CountOverlappingPairs(const DsaProblem& problem, const DsaSolution& solution,
                                           const std::vector<Separation>& pairs);

}  // namespace dsa
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_DSA_REUSE_PENALTY_SOLVER_H_
