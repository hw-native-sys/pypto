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

#include "src/ir/transforms/auto_tile/vector_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile/vector_cost_910b.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

// Ascend910B vector-model constants are ported unchanged from the
// silicon-grounded scheduler.  This pass changes ownership (PyPTO now plans
// and emits the whole marked graph); it does not refit the performance model.
// Bandwidths are GiB/s and all latency constants are cycles.
constexpr double kCoreFrequencyHz = 1.85e9;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kGmToUbGiBps = 100.9;
constexpr double kUbToGmGiBps = 188.46;
constexpr double kHbmAggregateGiBps = 900.0;
constexpr double kKernelFillCycles = 10000.0;
constexpr double kPerTaskCycles = 64.0;

bool IsReduction(VectorOpKind kind) {
  return kind == VectorOpKind::RowSum || kind == VectorOpKind::RowMax || kind == VectorOpKind::ColSum ||
         kind == VectorOpKind::ColMax;
}

AxisPartition PartitionAxis(int64_t extent, int64_t parts) {
  AxisPartition partition;
  partition.parts = parts;
  partition.small = extent / parts;
  partition.num_big = extent % parts;
  partition.big = partition.small + (partition.num_big != 0 ? 1 : 0);
  return partition;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

int64_t DmaElementGranule(const DataType& dtype, int64_t dma_alignment_bytes) {
  const int64_t bytes = DTypeBytes(dtype);
  CHECK(bytes > 0 && dma_alignment_bytes > 0);
  return dma_alignment_bytes / std::gcd(dma_alignment_bytes, bytes);
}

std::vector<int64_t> PhysicalElementGranules(const VectorGraph& graph, int64_t dma_alignment_bytes) {
  size_t class_count = 0;
  for (const VectorTensor& tensor : graph.tensors) {
    CHECK(tensor.physical_shape_class != std::numeric_limits<size_t>::max())
        << "AutoTile vector graph has an unresolved physical-shape class";
    class_count = std::max(class_count, tensor.physical_shape_class + 1);
  }
  std::vector<int64_t> class_granules(class_count, 1);
  for (const VectorTensor& tensor : graph.tensors) {
    int64_t& granule = class_granules[tensor.physical_shape_class];
    granule = std::lcm(granule, DmaElementGranule(tensor.dtype, dma_alignment_bytes));
  }
  std::vector<int64_t> result;
  result.reserve(graph.tensors.size());
  for (const VectorTensor& tensor : graph.tensors) {
    result.push_back(class_granules[tensor.physical_shape_class]);
  }
  return result;
}

int64_t FrameRows(const VectorTensor& tensor, int64_t rows) {
  return tensor.rows == 1 ? 1 : std::min(tensor.rows, rows);
}

int64_t FrameCols(const VectorTensor& tensor, int64_t cols) {
  return tensor.cols == 1 ? 1 : std::min(tensor.cols, cols);
}

int64_t TensorFrameBytes(const VectorTensor& tensor, int64_t element_granule, int64_t rows, int64_t cols,
                         bool reduction_layout) {
  const int64_t bytes = DTypeBytes(tensor.dtype);
  int64_t frame_rows = FrameRows(tensor, rows);
  int64_t frame_cols = FrameCols(tensor, cols);
  // A logical extent of one is not necessarily a broadcast.  The emitter
  // pads a non-broadcast reduction tile to the DMA granule even when a
  // balanced partition happens to contain one row/column.  Key padding on
  // the source tensor's shape, exactly as SliceInput does, so the planner
  // cannot under-price the common one-row softmax region.
  if (reduction_layout && tensor.rows != 1) frame_rows = AlignUp(frame_rows, element_granule);
  if (tensor.cols != 1) frame_cols = AlignUp(frame_cols, element_granule);
  return frame_rows * frame_cols * bytes;
}

int64_t RowReductionScratchBytes(const VectorTensor& tensor, int64_t element_granule, int64_t rows,
                                 int64_t cols, bool reduction_layout) {
  const int64_t bytes = DTypeBytes(tensor.dtype);
  int64_t frame_rows = FrameRows(tensor, rows);
  int64_t frame_cols = FrameCols(tensor, cols);
  if (reduction_layout && tensor.rows != 1) frame_rows = AlignUp(frame_rows, element_granule);
  if (tensor.cols != 1) frame_cols = AlignUp(frame_cols, element_granule);
  // OpConversionRegistry::RegisterReductionOps pads the scratch's final
  // dimension to at least 128 elements for row reductions. Column reductions
  // lower directly to one-operand tile ops and allocate no scratch.
  return frame_rows * std::max<int64_t>(128, frame_cols) * bytes;
}

std::vector<size_t> AncestorsOf(const VectorGraph& graph, size_t sink,
                                const std::unordered_set<size_t>& substituted_tensors = {}) {
  std::vector<size_t> producer(graph.tensors.size(), std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < graph.ops.size(); ++i) producer[graph.ops[i].output] = i;
  std::vector<bool> needed(graph.ops.size(), false);
  std::function<void(size_t)> visit = [&](size_t op_index) {
    if (needed[op_index]) return;
    needed[op_index] = true;
    for (size_t tensor : graph.ops[op_index].inputs) {
      if (substituted_tensors.count(tensor) != 0) continue;
      const size_t p = producer[tensor];
      if (p != std::numeric_limits<size_t>::max()) visit(p);
    }
  };
  visit(sink);
  std::vector<size_t> result;
  for (size_t i = 0; i < needed.size(); ++i)
    if (needed[i]) result.push_back(i);
  return result;
}

std::vector<VectorInputLifetime> BuildInputLifetimes(const VectorGraph& graph, const std::vector<size_t>& ops,
                                                     const std::unordered_set<size_t>& substituted = {}) {
  std::map<size_t, VectorInputLifetime> lifetimes;
  for (size_t step = 0; step < ops.size(); ++step) {
    const size_t op_index = ops[step];
    const VectorOp& op = graph.ops[op_index];
    for (size_t arg = 0; arg < op.inputs.size(); ++arg) {
      const size_t tensor = op.inputs[arg];
      if (substituted.count(tensor) != 0 || !graph.tensors[tensor].boundary_input) continue;
      auto [it, inserted] =
          lifetimes.emplace(tensor, VectorInputLifetime{tensor, step, step, std::vector<VectorInputUse>{}});
      if (!inserted) it->second.last_use = step;
      it->second.uses.push_back({op_index, arg});
    }
  }
  std::vector<VectorInputLifetime> result;
  result.reserve(lifetimes.size());
  for (auto& [unused, lifetime] : lifetimes) {
    (void)unused;
    result.push_back(std::move(lifetime));
  }
  return result;
}

int64_t PeakBytes(const VectorGraph& graph, const std::vector<size_t>& ops, int64_t rows, int64_t cols,
                  const std::unordered_set<size_t>& phase_outputs,
                  const std::unordered_set<size_t>& substituted, bool reduction_layout,
                  const std::vector<int64_t>& element_granules, int bands) {
  if (ops.empty()) return 0;
  std::unordered_map<size_t, size_t> first;
  std::unordered_map<size_t, size_t> last;
  for (size_t step = 0; step < ops.size(); ++step) {
    const VectorOp& op = graph.ops[ops[step]];
    first.emplace(op.output, step);
    last[op.output] = step;
    for (size_t tensor : op.inputs) {
      first.emplace(tensor, step);
      last[tensor] = step;
    }
  }
  for (size_t tensor : phase_outputs) {
    if (first.count(tensor) != 0) last[tensor] = ops.size();
  }
  for (size_t tensor : substituted) {
    first.emplace(tensor, 0);
    last[tensor] = ops.size();
  }

  int64_t current = 0;
  int64_t peak = 0;
  std::unordered_set<size_t> live;
  auto allocate = [&](size_t tensor) {
    if (!live.insert(tensor).second) return;
    current +=
        TensorFrameBytes(graph.tensors[tensor], element_granules.at(tensor), rows, cols, reduction_layout);
    peak = std::max(peak, current);
  };
  auto release = [&](size_t tensor) {
    if (live.erase(tensor) == 0) return;
    current -=
        TensorFrameBytes(graph.tensors[tensor], element_granules.at(tensor), rows, cols, reduction_layout);
  };

  for (size_t tensor : substituted) allocate(tensor);
  for (size_t step = 0; step < ops.size(); ++step) {
    const VectorOp& op = graph.ops[ops[step]];
    for (size_t tensor : op.inputs)
      if (first[tensor] == step) allocate(tensor);
    // ConvertTensorToTileOps lowers row reductions with a scratch tile. It is
    // live together with the source and the thin result for the reduction
    // instruction, so account for its exact padded shape here rather than
    // relying on the downstream allocator to discover unplanned storage.
    int64_t implicit_scratch = 0;
    if (op.kind == VectorOpKind::RowSum || op.kind == VectorOpKind::RowMax) {
      const size_t input = op.inputs.front();
      implicit_scratch = RowReductionScratchBytes(graph.tensors[input], element_granules.at(input), rows,
                                                  cols, reduction_layout);
    } else if (IsOp(op.call, "tensor.rsqrt") && op.call->GetKwarg<bool>("high_precision", false)) {
      const size_t input = op.inputs.front();
      implicit_scratch =
          TensorFrameBytes(graph.tensors[input], element_granules.at(input), rows, cols, reduction_layout);
    }
    if (implicit_scratch != 0) {
      current += implicit_scratch;
      peak = std::max(peak, current);
    }
    allocate(op.output);
    current -= implicit_scratch;
    for (size_t tensor : op.inputs)
      if (last[tensor] == step) release(tensor);
    if (last[op.output] == step) release(op.output);
  }
  return peak * std::max(1, bands);
}

double ComputeCycles(const VectorGraph& graph, const std::vector<size_t>& ops, int64_t rows, int64_t cols,
                     int64_t iterations, int64_t work_units, int64_t vector_register_bytes,
                     bool* used_reduction_fallback = nullptr) {
  double per_frame = 0.0;
  bool stream_start = true;
  for (size_t op_index : ops) {
    const VectorOp& op = graph.ops[op_index];
    if (IsReduction(op.kind)) {
      const VectorTensor& input = graph.tensors[op.inputs.front()];
      const VectorReductionCost reduction = ReductionCycles910B(
          op.kind, input.dtype, FrameRows(input, rows), FrameCols(input, cols), vector_register_bytes);
      per_frame += reduction.cycles;
      if (used_reduction_fallback != nullptr) *used_reduction_fallback |= reduction.used_fallback;
      stream_start = true;
      continue;
    }
    const VectorTensor& output = graph.tensors[op.output];
    int64_t frame_rows = FrameRows(output, rows);
    int64_t frame_cols = FrameCols(output, cols);
    for (size_t input : op.inputs) {
      frame_rows = std::max(frame_rows, FrameRows(graph.tensors[input], rows));
      frame_cols = std::max(frame_cols, FrameCols(graph.tensors[input], cols));
    }
    per_frame += PointwiseCycles910B(op.primitive, op.geometry, output.dtype, frame_rows, frame_cols,
                                     stream_start, graph.reduced_axis != 0, vector_register_bytes);
    stream_start = false;
  }
  return per_frame * static_cast<double>(std::max<int64_t>(1, iterations)) *
         static_cast<double>(std::max<int64_t>(1, work_units));
}

double BoundaryInputBytes(const VectorGraph& graph, const VectorPhasePlan& phase, int64_t rows, int64_t cols,
                          int64_t iterations, int64_t work_units) {
  double bytes = 0.0;
  for (const VectorInputLifetime& input : phase.inputs) {
    const VectorTensor& tensor = graph.tensors[input.tensor];
    bytes +=
        static_cast<double>(FrameRows(tensor, rows) * FrameCols(tensor, cols) * DTypeBytes(tensor.dtype));
  }
  return bytes * static_cast<double>(std::max<int64_t>(1, iterations)) *
         static_cast<double>(std::max<int64_t>(1, work_units));
}

double OutputBytes(const VectorGraph& graph, const std::unordered_set<size_t>& outputs, int64_t rows,
                   int64_t cols, int64_t iterations, int64_t work_units) {
  double bytes = 0.0;
  for (size_t tensor : outputs) {
    const VectorTensor& value = graph.tensors[tensor];
    bytes += static_cast<double>(FrameRows(value, rows) * FrameCols(value, cols) * DTypeBytes(value.dtype));
  }
  return bytes * static_cast<double>(std::max<int64_t>(1, iterations)) *
         static_cast<double>(std::max<int64_t>(1, work_units));
}

double TransferCycles(double bytes, double per_core_gibps, int64_t active_cores) {
  const double bandwidth = std::min(kHbmAggregateGiBps, per_core_gibps * static_cast<double>(active_cores));
  return bytes * kCoreFrequencyHz / (kGiB * bandwidth);
}

double WaveCompute(double total, int64_t tasks, int64_t cores) {
  if (tasks <= 0) return 0.0;
  return total / static_cast<double>(std::min(tasks, cores));
}

void PopulatePhase(VectorPhasePlan* phase, const VectorGraph& graph, std::vector<size_t> ops,
                   const std::unordered_set<size_t>& substituted = {}) {
  phase->ops = std::move(ops);
  phase->inputs = BuildInputLifetimes(graph, phase->ops, substituted);
}

void ClearModeledCosts(VectorSchedulePlan* plan) {
  plan->modeled_cycles = std::numeric_limits<double>::infinity();
  plan->modeled_compute_cycles = 0.0;
  plan->modeled_transfer_cycles = 0.0;
  plan->modeled_phase_compute_cycles.fill(0.0);
  plan->modeled_phase_transfer_cycles.fill(0.0);
  plan->modeled_phase_input_bytes.fill(0.0);
  plan->modeled_phase_output_bytes.fill(0.0);
  plan->used_reduction_fallback = false;
}

bool LexicographicallyBetter(const VectorSchedulePlan& lhs, const VectorSchedulePlan& rhs) {
  if (lhs.modeled_cycles != rhs.modeled_cycles) return lhs.modeled_cycles < rhs.modeled_cycles;
  if (lhs.work_units != rhs.work_units) return lhs.work_units < rhs.work_units;
  if (lhs.tile_h != rhs.tile_h) return lhs.tile_h > rhs.tile_h;
  if (lhs.tile_w != rhs.tile_w) return lhs.tile_w > rhs.tile_w;
  return std::tie(lhs.m_partition.parts, lhs.n_partition.parts) <
         std::tie(rhs.m_partition.parts, rhs.n_partition.parts);
}

}  // namespace

const char* ScheduleKindName(VectorScheduleKind kind) {
  switch (kind) {
    case VectorScheduleKind::Materialized:
      return "materialized";
    case VectorScheduleKind::PointwiseStream:
      return "pointwise_stream";
    case VectorScheduleKind::ReductionFolded:
      return "reduction_folded";
    case VectorScheduleKind::ReductionSpanning:
      return "reduction_spanning";
    case VectorScheduleKind::Softmax:
      return "softmax";
  }
  return "unknown";
}

VectorSchedulePlan VectorPlanner910B::Plan(const VectorGraph& graph) const {
  CHECK(hardware_.vector_cores > 0 && hardware_.ub_bytes > 0 && hardware_.dma_alignment_bytes > 0 &&
        hardware_.vector_register_bytes > 0)
      << "AutoTile vector planner received an invalid Ascend 910B hardware descriptor";

  const int64_t iteration_rows = graph.iteration_rows;
  const int64_t iteration_cols = graph.iteration_cols;
  const bool reduction_layout = graph.reduced_axis != 0;
  const std::unordered_set<size_t> all_outputs(graph.required_outputs.begin(), graph.required_outputs.end());
  const std::vector<int64_t> element_granules = PhysicalElementGranules(graph, hardware_.dma_alignment_bytes);
  std::vector<size_t> all_ops(graph.ops.size());
  std::iota(all_ops.begin(), all_ops.end(), 0);

  std::vector<int64_t> task_counts;
  for (int64_t count = 1; count <= 2LL * hardware_.vector_cores; ++count)
    if (hardware_.vector_cores % count == 0 || (2LL * hardware_.vector_cores) % count == 0)
      task_counts.push_back(count);

  std::vector<std::pair<int64_t, int64_t>> grids;
  if (graph.reduced_axis == 1) {
    for (int64_t count : task_counts)
      if (count <= iteration_rows) grids.emplace_back(count, 1);
  } else if (graph.reduced_axis == 2) {
    for (int64_t count : task_counts)
      if (count <= iteration_cols) grids.emplace_back(1, count);
  } else {
    for (int64_t count : task_counts) {
      for (int64_t parts_m = 1; parts_m <= count; ++parts_m) {
        if (count % parts_m != 0) continue;
        const int64_t parts_n = count / parts_m;
        if (parts_m <= iteration_rows && parts_n <= iteration_cols) grids.emplace_back(parts_m, parts_n);
      }
    }
  }

  VectorSchedulePlan best;
  for (const auto& [parts_m, parts_n] : grids) {
    VectorSchedulePlan candidate;
    candidate.m_partition = PartitionAxis(iteration_rows, parts_m);
    candidate.n_partition = PartitionAxis(iteration_cols, parts_n);
    candidate.work_units = parts_m * parts_n;
    candidate.tile_h = candidate.m_partition.big;
    candidate.tile_w = candidate.n_partition.big;
    candidate.reduced_axis = graph.reduced_axis;
    candidate.dma_alignment_bytes = hardware_.dma_alignment_bytes;
    candidate.tensor_element_granules = element_granules;
    candidate.full_peak_ub_bytes = PeakBytes(graph, all_ops, candidate.tile_h, candidate.tile_w, all_outputs,
                                             {}, reduction_layout, element_granules, 1);

    auto record_phase = [&](VectorSchedulePlan* target, VectorPhase phase_id, int64_t rows, int64_t cols,
                            int64_t iterations, int stages, const std::unordered_set<size_t>& outputs,
                            double compute_work) {
      if (iterations <= 0) return 0.0;
      const size_t phase_index = PhaseIndex(phase_id);
      const int64_t tasks = target->work_units;
      const int64_t active = std::min<int64_t>(tasks, hardware_.vector_cores);
      const double compute = WaveCompute(compute_work, tasks, hardware_.vector_cores);
      const VectorPhasePlan& phase = target->phases[phase_index];
      const double in_bytes = BoundaryInputBytes(graph, phase, rows, cols, iterations, tasks);
      const double out_bytes = OutputBytes(graph, outputs, rows, cols, iterations, tasks);
      const double transfer =
          TransferCycles(in_bytes, kGmToUbGiBps, active) + TransferCycles(out_bytes, kUbToGmGiBps, active);
      target->modeled_compute_cycles += compute;
      target->modeled_transfer_cycles += transfer;
      target->modeled_phase_compute_cycles[phase_index] += compute;
      target->modeled_phase_transfer_cycles[phase_index] += transfer;
      target->modeled_phase_input_bytes[phase_index] += in_bytes;
      target->modeled_phase_output_bytes[phase_index] += out_bytes;
      return stages == 2 ? std::max(compute, transfer) : compute + transfer;
    };
    auto price_phase = [&](VectorSchedulePlan* target, VectorPhase phase_id, int64_t rows, int64_t cols,
                           int64_t iterations, int stages, const std::unordered_set<size_t>& outputs,
                           double extra_compute_work = 0.0) {
      bool fallback = false;
      const VectorPhasePlan& phase = target->phases[PhaseIndex(phase_id)];
      const double compute_work = ComputeCycles(graph, phase.ops, rows, cols, iterations, target->work_units,
                                                hardware_.vector_register_bytes, &fallback) +
                                  extra_compute_work;
      target->used_reduction_fallback |= fallback;
      return record_phase(target, phase_id, rows, cols, iterations, stages, outputs, compute_work);
    };

    // A recognized softmax has two legal algorithms when its complete region
    // fits in UB: replay the source DAG once with all intermediates resident,
    // or use the chunked online statistics/apply schedule below.  Preserve
    // both candidates and let the same modeled-cycle comparison used for
    // grids and chunks choose between them.
    VectorSchedulePlan materialized_softmax;
    double materialized_softmax_latency = std::numeric_limits<double>::infinity();
    if (graph.softmax.matched && candidate.full_peak_ub_bytes <= hardware_.ub_bytes) {
      materialized_softmax = candidate;
      materialized_softmax.kind = VectorScheduleKind::Materialized;
      PopulatePhase(&materialized_softmax.phases[PhaseIndex(VectorPhase::Body)], graph, all_ops);
      materialized_softmax.strip_h = materialized_softmax.tile_h;
      materialized_softmax.strip_w = materialized_softmax.tile_w;
      materialized_softmax.chunk_peak_ub_bytes = materialized_softmax.full_peak_ub_bytes;
      materialized_softmax_latency =
          price_phase(&materialized_softmax, VectorPhase::Body, materialized_softmax.tile_h,
                      materialized_softmax.tile_w, 1, 1, all_outputs);
    }

    double latency = 0.0;
    bool feasible = false;
    if (candidate.full_peak_ub_bytes <= hardware_.ub_bytes && !graph.softmax.matched) {
      candidate.kind = VectorScheduleKind::Materialized;
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Body)], graph, all_ops);
      candidate.strip_h = candidate.tile_h;
      candidate.strip_w = candidate.tile_w;
      candidate.chunk_peak_ub_bytes = candidate.full_peak_ub_bytes;
      latency =
          price_phase(&candidate, VectorPhase::Body, candidate.tile_h, candidate.tile_w, 1, 1, all_outputs);
      feasible = true;

      if (graph.reduced_axis == 0) {
        VectorSchedulePlan streamed = candidate;
        streamed.kind = VectorScheduleKind::PointwiseStream;
        double best_stream_latency = latency;
        for (int64_t row_strips = 1; row_strips <= candidate.tile_h; row_strips *= 2) {
          for (int64_t width_strips = 1; width_strips <= candidate.tile_w; width_strips *= 2) {
            if (row_strips > 1 && width_strips > 1) continue;
            const int64_t strip_h = (candidate.tile_h + row_strips - 1) / row_strips;
            const int64_t strip_w = (candidate.tile_w + width_strips - 1) / width_strips;
            const int64_t trips = row_strips * width_strips;
            if (trips < 2) continue;
            const int64_t peak =
                PeakBytes(graph, all_ops, strip_h, strip_w, all_outputs, {}, false, element_granules, 2);
            if (peak > hardware_.ub_bytes) continue;
            VectorSchedulePlan trial = candidate;
            ClearModeledCosts(&trial);
            trial.kind = VectorScheduleKind::PointwiseStream;
            trial.row_strips = row_strips;
            trial.width_strips = width_strips;
            trial.strip_h = strip_h;
            trial.strip_w = strip_w;
            trial.chunk_peak_ub_bytes = peak;
            trial.phases[PhaseIndex(VectorPhase::Body)].trip_count = trips;
            trial.phases[PhaseIndex(VectorPhase::Body)].pipeline_stages = 2;
            const double trial_latency =
                price_phase(&trial, VectorPhase::Body, strip_h, strip_w, trips, 2, all_outputs);
            if (trial_latency < best_stream_latency) {
              streamed = std::move(trial);
              best_stream_latency = trial_latency;
            }
          }
        }
        if (best_stream_latency < latency) {
          candidate = std::move(streamed);
          latency = best_stream_latency;
        }
      }
    } else if (graph.reduced_axis == 0) {
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Body)], graph, all_ops);
      double best_stream_latency = std::numeric_limits<double>::infinity();
      for (int64_t row_strips = 1; row_strips <= candidate.tile_h; row_strips *= 2) {
        for (int64_t width_strips = 1; width_strips <= candidate.tile_w; width_strips *= 2) {
          if (row_strips > 1 && width_strips > 1) continue;
          const int64_t trips = row_strips * width_strips;
          if (trips < 2) continue;
          const int64_t strip_h = (candidate.tile_h + row_strips - 1) / row_strips;
          const int64_t strip_w = (candidate.tile_w + width_strips - 1) / width_strips;
          const int64_t peak =
              PeakBytes(graph, all_ops, strip_h, strip_w, all_outputs, {}, false, element_granules, 2);
          if (peak > hardware_.ub_bytes) continue;
          VectorSchedulePlan trial = candidate;
          ClearModeledCosts(&trial);
          trial.kind = VectorScheduleKind::PointwiseStream;
          trial.row_strips = row_strips;
          trial.width_strips = width_strips;
          trial.strip_h = strip_h;
          trial.strip_w = strip_w;
          trial.chunk_peak_ub_bytes = peak;
          trial.phases[PhaseIndex(VectorPhase::Body)].trip_count = trips;
          trial.phases[PhaseIndex(VectorPhase::Body)].pipeline_stages = 2;
          const double trial_latency =
              price_phase(&trial, VectorPhase::Body, strip_h, strip_w, trips, 2, all_outputs);
          if (trial_latency >= best_stream_latency) continue;
          best_stream_latency = trial_latency;
          candidate = std::move(trial);
          latency = trial_latency;
          feasible = true;
        }
      }
    } else if (graph.reduction_count > 1 && !graph.softmax.matched) {
      // A general multi-reduction DAG is valid only when the complete
      // topological replay above fits in UB.  The streaming schedule below
      // carries one reduction state and cannot represent dependent reductions
      // such as LayerNorm's mean followed by variance.
      continue;
    } else if (graph.softmax.matched || graph.reduction_op != std::numeric_limits<size_t>::max()) {
      // Multi-live-out reduction DAGs remain eligible for a materialized plan
      // on a smaller spatial region.  They simply cannot use the two-pass
      // streaming replay because that contract carries one output buffer.
      if (graph.required_outputs.size() != 1) continue;
      const size_t reduction_op = graph.softmax.matched ? graph.softmax.max_op : graph.reduction_op;
      const size_t reduction_tensor = graph.ops[reduction_op].output;
      std::vector<size_t> stats_ops = AncestorsOf(graph, reduction_op);
      std::unordered_set<size_t> substituted{reduction_tensor};
      std::vector<size_t> apply_ops = AncestorsOf(graph, graph.required_output_ops[0], substituted);
      if (graph.required_output_ops[0] == reduction_op) apply_ops.clear();
      if (graph.softmax.matched) {
        substituted.insert(graph.ops[graph.softmax.sum_op].output);
        apply_ops = AncestorsOf(graph, graph.softmax.sink_op, substituted);
      }

      candidate.kind = graph.softmax.matched ? VectorScheduleKind::Softmax
                                             : (graph.tensors[graph.required_outputs[0]].rows == 1 ||
                                                        graph.tensors[graph.required_outputs[0]].cols == 1
                                                    ? VectorScheduleKind::ReductionFolded
                                                    : VectorScheduleKind::ReductionSpanning);
      candidate.free_tile = graph.reduced_axis == 1 ? candidate.tile_h : candidate.tile_w;
      const DataType reduction_dtype = graph.tensors[reduction_tensor].dtype;
      candidate.free_tile_alloc = AlignUp(candidate.free_tile, element_granules.at(reduction_tensor));
      candidate.reduced_extent = graph.reduced_axis == 1 ? iteration_cols : iteration_rows;
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Stats)], graph, stats_ops);
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Apply)], graph, apply_ops, substituted);
      if (graph.softmax.matched) {
        candidate.phases[PhaseIndex(VectorPhase::Stats)].ops.clear();
        candidate.phases[PhaseIndex(VectorPhase::Stats)].inputs = {
            {graph.softmax.input, 0, 0, {{graph.softmax.max_op, 0}}}};
      }

      VectorSchedulePlan best_reduction;
      double best_reduction_latency = std::numeric_limits<double>::infinity();
      int64_t largest_feasible_chunk = 0;
      int64_t feasible_chunk_candidates = 0;
      for (int64_t chunk = std::min<int64_t>(candidate.reduced_extent, 4096); chunk >= 1; --chunk) {
        if (chunk != candidate.reduced_extent && chunk % 16 != 0) continue;
        const int64_t rows = graph.reduced_axis == 1 ? candidate.free_tile : chunk;
        const int64_t cols = graph.reduced_axis == 1 ? chunk : candidate.free_tile;
        const int64_t thin_rows = graph.reduced_axis == 1 ? candidate.free_tile : 1;
        const int64_t thin_cols = graph.reduced_axis == 1 ? 1 : candidate.free_tile;
        int64_t stats_peak;
        if (graph.softmax.matched) {
          const int64_t x_bytes = TensorFrameBytes(
              graph.tensors[graph.softmax.input], element_granules.at(graph.softmax.input), rows, cols, true);
          const int64_t thin_bytes = candidate.free_tile_alloc * DTypeBytes(reduction_dtype);
          stats_peak = 2 * (6 * x_bytes + 4 * thin_bytes);
        } else {
          stats_peak =
              PeakBytes(graph, stats_ops, rows, cols, {reduction_tensor}, {}, true, element_granules, 2) +
              2 * candidate.free_tile_alloc * DTypeBytes(reduction_dtype);
        }
        int64_t apply_peak = 0;
        if (candidate.kind == VectorScheduleKind::ReductionSpanning || graph.softmax.matched) {
          apply_peak =
              PeakBytes(graph, apply_ops, rows, cols, all_outputs, substituted, true, element_granules, 2);
        } else {
          apply_peak = PeakBytes(graph, apply_ops, thin_rows, thin_cols, all_outputs, substituted, true,
                                 element_granules, 1);
        }
        const int64_t peak = std::max(stats_peak, apply_peak);
        if (peak > hardware_.ub_bytes) continue;
        if (largest_feasible_chunk == 0) largest_feasible_chunk = chunk;
        ++feasible_chunk_candidates;

        VectorSchedulePlan trial = candidate;
        ClearModeledCosts(&trial);
        trial.chunk = chunk;
        trial.full_chunks = trial.reduced_extent / chunk;
        trial.tail = trial.reduced_extent % chunk;
        trial.chunk_peak_ub_bytes = peak;
        VectorPhasePlan& stats = trial.phases[PhaseIndex(VectorPhase::Stats)];
        stats.first_chunk = 1;
        stats.trip_count = std::max<int64_t>(0, trial.full_chunks - 1);
        stats.pipeline_stages = stats.trip_count >= 2 ? 2 : 1;
        VectorPhasePlan& apply = trial.phases[PhaseIndex(VectorPhase::Apply)];
        apply.first_chunk = 0;
        apply.trip_count = trial.full_chunks;
        apply.pipeline_stages = apply.trip_count >= 2 ? 2 : 1;

        double trial_latency = 0.0;
        if (graph.softmax.matched) {
          bool fallback = false;
          const double init_compute =
              GeneratedSoftmaxCycles910B(false, trial.free_tile, trial.chunk, 1, reduction_dtype,
                                         trial.work_units, hardware_.vector_register_bytes, &fallback);
          trial.used_reduction_fallback |= fallback;
          trial_latency += record_phase(&trial, VectorPhase::Stats, rows, cols, 1, 1, {}, init_compute);
          if (stats.trip_count > 0) {
            fallback = false;
            const double update_compute = GeneratedSoftmaxCycles910B(
                true, trial.free_tile, trial.chunk, stats.trip_count, reduction_dtype, trial.work_units,
                hardware_.vector_register_bytes, &fallback);
            trial.used_reduction_fallback |= fallback;
            trial_latency += record_phase(&trial, VectorPhase::Stats, rows, cols, stats.trip_count,
                                          stats.pipeline_stages, {}, update_compute);
          }
          if (trial.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? trial.free_tile : trial.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? trial.tail : trial.free_tile;
            fallback = false;
            const double tail_compute =
                GeneratedSoftmaxCycles910B(true, trial.free_tile, trial.tail, 1, reduction_dtype,
                                           trial.work_units, hardware_.vector_register_bytes, &fallback);
            trial.used_reduction_fallback |= fallback;
            trial_latency +=
                record_phase(&trial, VectorPhase::Stats, tail_rows, tail_cols, 1, 1, {}, tail_compute);
          }
        } else {
          trial_latency += price_phase(&trial, VectorPhase::Stats, rows, cols, 1, 1, {});
          if (stats.trip_count > 0) {
            const double merge_compute = GeneratedReductionMergeCycles910B(
                graph.reduced_axis, trial.free_tile, stats.trip_count, reduction_dtype, trial.work_units,
                hardware_.vector_register_bytes);
            trial_latency += price_phase(&trial, VectorPhase::Stats, rows, cols, stats.trip_count,
                                         stats.pipeline_stages, {}, merge_compute);
          }
          if (trial.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? trial.free_tile : trial.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? trial.tail : trial.free_tile;
            const double merge_compute =
                GeneratedReductionMergeCycles910B(graph.reduced_axis, trial.free_tile, 1, reduction_dtype,
                                                  trial.work_units, hardware_.vector_register_bytes);
            trial_latency +=
                price_phase(&trial, VectorPhase::Stats, tail_rows, tail_cols, 1, 1, {}, merge_compute);
          }
        }
        if (trial.kind == VectorScheduleKind::ReductionSpanning || graph.softmax.matched) {
          trial_latency += price_phase(&trial, VectorPhase::Apply, rows, cols, trial.full_chunks,
                                       apply.pipeline_stages, all_outputs);
          if (trial.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? trial.free_tile : trial.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? trial.tail : trial.free_tile;
            trial_latency += price_phase(&trial, VectorPhase::Apply, tail_rows, tail_cols, 1, 1, all_outputs);
          }
        } else {
          trial.phases[PhaseIndex(VectorPhase::Finalize)] = apply;
          trial.phases[PhaseIndex(VectorPhase::Apply)] = {};
          const int64_t thin_rows = graph.reduced_axis == 1 ? trial.free_tile : 1;
          const int64_t thin_cols = graph.reduced_axis == 1 ? 1 : trial.free_tile;
          trial_latency +=
              price_phase(&trial, VectorPhase::Finalize, thin_rows, thin_cols, 1, 1, all_outputs);
        }

        if (trial_latency < best_reduction_latency ||
            (trial_latency == best_reduction_latency && trial.chunk > best_reduction.chunk)) {
          trial.modeled_cycles = trial_latency;
          best_reduction = std::move(trial);
          best_reduction_latency = trial_latency;
        }
      }
      if (std::isfinite(best_reduction_latency)) {
        best_reduction.largest_feasible_chunk = largest_feasible_chunk;
        best_reduction.feasible_chunk_candidates = feasible_chunk_candidates;
        candidate = std::move(best_reduction);
        latency = best_reduction_latency;
        feasible = true;
      }
    }

    if (std::isfinite(materialized_softmax_latency) &&
        (!feasible || materialized_softmax_latency < latency)) {
      candidate = std::move(materialized_softmax);
      latency = materialized_softmax_latency;
      feasible = true;
    }

    if (!feasible) continue;

    const int64_t body_tasks = candidate.work_units;
    latency += kPerTaskCycles * static_cast<double>(body_tasks);
    latency += kKernelFillCycles *
               static_cast<double>((body_tasks + hardware_.vector_cores - 1) / hardware_.vector_cores);
    candidate.modeled_cycles = latency;
    candidate.feasible = true;
    if (!best.feasible || LexicographicallyBetter(candidate, best)) best = std::move(candidate);
  }
  return best;
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
