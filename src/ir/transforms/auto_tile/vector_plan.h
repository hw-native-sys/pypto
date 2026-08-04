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

#ifndef PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_PLAN_H_
#define PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_PLAN_H_

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "src/ir/transforms/auto_tile/vector_graph.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

enum class VectorScheduleKind : uint8_t {
  Materialized,
  PointwiseStream,
  ReductionFolded,
  ReductionSpanning,
  Softmax,
};

enum class VectorPhase : uint8_t {
  Body = 0,
  Stats = 1,
  Apply = 2,
  Finalize = 3,
};

inline constexpr size_t PhaseIndex(VectorPhase phase) { return static_cast<size_t>(phase); }

struct AxisPartition {
  int64_t parts = 1;
  int64_t small = 0;
  int64_t big = 0;
  int64_t num_big = 0;
};

struct VectorInputUse {
  size_t op = 0;
  size_t arg = 0;

  friend bool operator==(const VectorInputUse& lhs, const VectorInputUse& rhs) {
    return lhs.op == rhs.op && lhs.arg == rhs.arg;
  }
};

struct VectorInputLifetime {
  size_t tensor = 0;
  size_t first_use = 0;
  size_t last_use = 0;
  std::vector<VectorInputUse> uses;
};

struct VectorPhasePlan {
  std::vector<size_t> ops;
  std::vector<VectorInputLifetime> inputs;
  int64_t first_chunk = 0;
  int64_t trip_count = 0;
  int pipeline_stages = 1;
};

struct VectorSchedulePlan {
  bool feasible = false;
  VectorScheduleKind kind = VectorScheduleKind::Materialized;
  AxisPartition m_partition;
  AxisPartition n_partition;
  int64_t work_units = 0;
  int64_t tile_h = 0;
  int64_t tile_w = 0;
  int64_t strip_h = 0;
  int64_t strip_w = 0;
  int64_t row_strips = 1;
  int64_t width_strips = 1;
  int64_t full_peak_ub_bytes = 0;
  int64_t chunk_peak_ub_bytes = 0;
  int64_t dma_alignment_bytes = 0;
  int reduced_axis = 0;
  int64_t free_tile = 0;
  int64_t free_tile_alloc = 0;
  int64_t reduced_extent = 0;
  int64_t chunk = 0;
  int64_t largest_feasible_chunk = 0;
  int64_t feasible_chunk_candidates = 0;
  int64_t full_chunks = 0;
  int64_t tail = 0;
  std::array<VectorPhasePlan, 4> phases;
  std::array<double, 4> modeled_phase_compute_cycles{};
  std::array<double, 4> modeled_phase_transfer_cycles{};
  std::array<double, 4> modeled_phase_input_bytes{};
  std::array<double, 4> modeled_phase_output_bytes{};
  double modeled_cycles = std::numeric_limits<double>::infinity();
  double modeled_compute_cycles = 0.0;
  double modeled_transfer_cycles = 0.0;
  bool used_reduction_fallback = false;
};

struct VectorHardware {
  int vector_cores = 0;
  int64_t ub_bytes = 0;
  int64_t dma_alignment_bytes = 0;
  int64_t vector_register_bytes = 0;
};

class VectorPlanner910B {
 public:
  explicit VectorPlanner910B(VectorHardware hardware) : hardware_(hardware) {}

  [[nodiscard]] VectorSchedulePlan Plan(const VectorGraph& graph) const;

 private:
  VectorHardware hardware_;
};

[[nodiscard]] const char* ScheduleKindName(VectorScheduleKind kind);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_PLAN_H_
