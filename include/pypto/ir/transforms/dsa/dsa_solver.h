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

#ifndef PYPTO_IR_TRANSFORMS_DSA_DSA_SOLVER_H_
#define PYPTO_IR_TRANSFORMS_DSA_DSA_SOLVER_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {
namespace dsa {

/// \file
/// IR-free Dynamic Storage Allocation (DSA) solver interface + a first-fit
/// heuristic.  This library is deliberately independent of PyPTO IR: it consumes
/// buffers with lifetimes + sizes and produces offsets, so it can be unit-tested
/// on synthetic instances (and benchmarked against MiniMalloc CSV instances)
/// without building any IR.  The PyPTO adapter (a separate layer) translates IR
/// memory-planning state into a DsaProblem and writes offsets back to MemRefs.
///
/// This is the *core* formulation only (min peak, no concurrent overlap for
/// lifetime-overlapping buffers).  The RFC's optional sync/bank cost overlay is
/// NOT modelled here (see SolverCapabilities::cost_model == false); it is a
/// later refinement that re-ranks the feasible set.

using BufferId = uint32_t;
/// Pool identity — one independent allocation arena.  The PyPTO adapter maps a
/// MemorySpace (L0A/L0B/L0C/L1/UB/...) to a PoolId; pools are solved independently.
using PoolId = int32_t;

/// A half-open-agnostic *inclusive* lifetime interval over topological order
/// points (matching MemoryReuse's `var_liveness` [def, last_use]).  Touching
/// intervals ([.,p] and [p,.]) are treated as overlapping — the conservative
/// choice, since a shared boundary point may be a read-then-write of the same slot.
struct Interval {
  int32_t start = 0;  ///< Definition point (topological order).
  int32_t end = 0;    ///< Last-use point (topological order), inclusive.

  [[nodiscard]] bool Overlaps(const Interval& o) const { return start <= o.end && o.start <= end; }
};

/// One buffer to place.  In the PyPTO mapping, one Buffer == one allocation
/// (base-group); view sub-regions are re-derived by the adapter on write-back.
struct Buffer {
  BufferId id = 0;
  uint64_t size = 0;        ///< Bytes.
  uint64_t align = 1;       ///< Power-of-two alignment; 0 or 1 == no constraint.
  PoolId pool = 0;
  Interval interval;        ///< Single hull for v1 (multi-interval is a later refinement).
};

/// A DSA instance.  colocations/separations are hard constraints; pool caps make
/// over-capacity a reported status rather than a silent overflow.
struct DsaProblem {
  std::vector<Buffer> buffers;
  /// Hard SAME-offset pairs (must-alias: loop-carry / in-place).
  std::vector<std::pair<BufferId, BufferId>> colocations;
  /// Hard KEEP-APART pairs even when lifetime-disjoint (hazard guard / pipeline clones).
  std::vector<std::pair<BufferId, BufferId>> separations;
  /// Per-pool low watermark: placement starts at reserved_base (end of the reserved region).
  std::map<PoolId, uint64_t> reserved_base;
  /// Per-pool capacity in bytes; 0 (or absent) == unbounded.
  std::map<PoolId, uint64_t> pool_caps;
};

enum class SolveStatus {
  kFeasible,          ///< A valid placement within all caps was found.
  kInfeasibleProven,  ///< Proven no placement fits (a complete solver only).
  kBestEffortNoFit,   ///< A placement was produced but exceeds a pool cap.
  kTimeout,           ///< Time-boxed solver gave up; solution may be a fallback.
  kUnsupported,       ///< The problem uses a feature this solver cannot honor.
};

/// The placement: every buffer id maps to its byte offset within its pool.
struct DsaSolution {
  std::map<BufferId, uint64_t> offsets;

  [[nodiscard]] uint64_t OffsetOf(BufferId id) const {
    auto it = offsets.find(id);
    return it == offsets.end() ? 0 : it->second;
  }
};

/// Objective readout.  peak == max over pools of peak_by_pool.
struct ObjectiveValue {
  uint64_t peak = 0;
  std::map<PoolId, uint64_t> peak_by_pool;
};

struct DsaResult {
  SolveStatus status = SolveStatus::kUnsupported;
  std::optional<DsaSolution> solution;
  ObjectiveValue objective;
  std::vector<std::string> diagnostics;
};

/// What a concrete solver honors.  A core-only solver drops the overlay and
/// multi-interval; the adapter reads this to decide preprocessing.
struct SolverCapabilities {
  bool multi_interval = false;
  bool cost_model = false;
  bool colocations = true;
  bool separations = true;
  bool pinned = false;
  bool multi_pool = true;
};

/// Abstract solver.  Solve() never "just returns a solution" — it always returns
/// a DsaResult with a status, so the caller can distinguish a real OOM
/// (kInfeasibleProven) from a best-effort over-cap placement (kBestEffortNoFit).
class DsaSolver {
 public:
  virtual ~DsaSolver() = default;
  [[nodiscard]] virtual SolverCapabilities Capabilities() const = 0;
  [[nodiscard]] virtual DsaResult Solve(const DsaProblem& problem) = 0;
};

/// First-fit-by-lifetime, buffers placed in decreasing-size order — the same
/// heuristic class as ptoas's modern memplan and XLA's GlobalDecreasingSizeBestFit.
///
/// Complexity: O(P * n^2 log n) where n is the number of buffers in the largest
/// pool and P the pool count (a fixed small constant).  This exceeds the project
/// O(N log N) pass bound; it is the accepted bound for the interval-packing
/// problem class (see XLA heap_simulator, ptoas first-fit) and n is bounded by
/// the buffers *per pool per function*, which is small (tens to low hundreds).
/// The dominant O(n^2) comes from each placement scanning already-placed
/// conflicting buffers; buffer-granularity packing with hole reclaim is what
/// dissolves the group-bump fragmentation (#1908).
class FirstFitByLifetimeSolver : public DsaSolver {
 public:
  [[nodiscard]] SolverCapabilities Capabilities() const override;
  [[nodiscard]] DsaResult Solve(const DsaProblem& problem) override;
};

/// Independent checker: recompute every invariant against a candidate solution.
/// Returns an empty vector iff the solution is valid.  Used in tests and as a
/// runtime safety net — never trusts the solver's own bookkeeping.
[[nodiscard]] std::vector<std::string> Validate(const DsaProblem& problem, const DsaSolution& solution);

}  // namespace dsa
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_DSA_SOLVER_H_
