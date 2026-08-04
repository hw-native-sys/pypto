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
#include <map>
#include <numeric>
#include <tuple>
#include <utility>

#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

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
constexpr int64_t kVectorRegisterBytes = 256;
constexpr double kCountModeFloorCycles = 16.0;
constexpr double kKernelFillCycles = 10000.0;
constexpr double kPerTaskCycles = 64.0;

struct OpDescriptor {
  VectorOpKind kind;
  VectorPrimitive primitive;
  VectorGeometry geometry;
};

std::optional<OpDescriptor> DescribeOp(const CallPtr& call) {
  if (IsOp(call, "tensor.row_sum"))
    return OpDescriptor{VectorOpKind::RowSum, VectorPrimitive::RowSum, VectorGeometry::Flat};
  if (IsOp(call, "tensor.row_max"))
    return OpDescriptor{VectorOpKind::RowMax, VectorPrimitive::RowExtrema, VectorGeometry::Flat};
  if (IsOp(call, "tensor.col_sum"))
    return OpDescriptor{VectorOpKind::ColSum, VectorPrimitive::ColSum, VectorGeometry::Flat};
  if (IsOp(call, "tensor.col_max"))
    return OpDescriptor{VectorOpKind::ColMax, VectorPrimitive::ColExtrema, VectorGeometry::Flat};

  VectorGeometry geometry = VectorGeometry::Flat;
  if (IsOp(call, "tensor.row_expand_add") || IsOp(call, "tensor.row_expand_sub") ||
      IsOp(call, "tensor.row_expand_mul") || IsOp(call, "tensor.row_expand_div") ||
      IsOp(call, "tensor.row_expand_max") || IsOp(call, "tensor.row_expand_min") ||
      IsOp(call, "tensor.row_expand_expdif")) {
    geometry = VectorGeometry::RowExpand;
  } else if (IsOp(call, "tensor.col_expand_add") || IsOp(call, "tensor.col_expand_sub") ||
             IsOp(call, "tensor.col_expand_mul") || IsOp(call, "tensor.col_expand_div") ||
             IsOp(call, "tensor.col_expand_max") || IsOp(call, "tensor.col_expand_min") ||
             IsOp(call, "tensor.col_expand_expdif")) {
    geometry = VectorGeometry::ColExpand;
  }

  VectorPrimitive primitive;
  if (IsOp(call, "tensor.add") || IsOp(call, "tensor.sub") || IsOp(call, "tensor.part_add") ||
      IsOp(call, "tensor.part_max") || IsOp(call, "tensor.part_min") || IsOp(call, "tensor.row_expand_add") ||
      IsOp(call, "tensor.row_expand_sub") || IsOp(call, "tensor.row_expand_max") ||
      IsOp(call, "tensor.row_expand_min") || IsOp(call, "tensor.col_expand_add") ||
      IsOp(call, "tensor.col_expand_sub") || IsOp(call, "tensor.col_expand_max") ||
      IsOp(call, "tensor.col_expand_min")) {
    primitive = VectorPrimitive::Add;
  } else if (IsOp(call, "tensor.maximum") || IsOp(call, "tensor.minimum")) {
    const bool scalar_rhs = call->args_.size() == 2 && As<ScalarType>(call->args_[1]->GetType()) != nullptr;
    primitive = scalar_rhs
                    ? (IsOp(call, "tensor.maximum") ? VectorPrimitive::ScalarMax : VectorPrimitive::ScalarMin)
                    : VectorPrimitive::Add;
  } else if (IsOp(call, "tensor.mul") || IsOp(call, "tensor.part_mul") ||
             IsOp(call, "tensor.row_expand_mul") || IsOp(call, "tensor.col_expand_mul")) {
    primitive = VectorPrimitive::Mul;
  } else if (IsOp(call, "tensor.div") || IsOp(call, "tensor.row_expand_div") ||
             IsOp(call, "tensor.col_expand_div") || IsOp(call, "tensor.divs")) {
    primitive =
        call->GetKwarg<bool>("high_precision", false) ? VectorPrimitive::Generic : VectorPrimitive::Div;
  } else if (IsOp(call, "tensor.exp")) {
    primitive = VectorPrimitive::Exp;
  } else if (IsOp(call, "tensor.row_expand_expdif") || IsOp(call, "tensor.col_expand_expdif")) {
    primitive = VectorPrimitive::Exp;
  } else if (IsOp(call, "tensor.log")) {
    primitive =
        call->GetKwarg<bool>("high_precision", false) ? VectorPrimitive::Generic : VectorPrimitive::Log;
  } else if (IsOp(call, "tensor.abs")) {
    primitive = VectorPrimitive::Abs;
  } else if (IsOp(call, "tensor.sqrt")) {
    primitive = VectorPrimitive::Sqrt;
  } else if (IsOp(call, "tensor.rsqrt")) {
    primitive =
        call->GetKwarg<bool>("high_precision", false) ? VectorPrimitive::Generic : VectorPrimitive::Rsqrt;
  } else if (IsOp(call, "tensor.adds") || IsOp(call, "tensor.subs")) {
    primitive = VectorPrimitive::ScalarAdd;
  } else if (IsOp(call, "tensor.muls") || IsOp(call, "tensor.neg")) {
    primitive = VectorPrimitive::ScalarMul;
  } else if (IsOp(call, "tensor.cast")) {
    primitive = VectorPrimitive::Cast;
  } else if (IsOp(call, "tensor.recip")) {
    primitive = VectorPrimitive::Recip;
  } else if (IsOp(call, "tensor.fmod") || IsOp(call, "tensor.fmods")) {
    primitive = VectorPrimitive::Generic;
  } else {
    return std::nullopt;
  }
  return OpDescriptor{VectorOpKind::Elementwise, primitive, geometry};
}

bool IsReduction(VectorOpKind kind) {
  return kind == VectorOpKind::RowSum || kind == VectorOpKind::RowMax || kind == VectorOpKind::ColSum ||
         kind == VectorOpKind::ColMax;
}

bool IsSupportedComputeDType(const DataType& dtype) {
  return dtype == DataType::FP32 || dtype == DataType::FP16 || dtype == DataType::BF16;
}

bool IsSupportedTensorDType(const DataType& dtype) {
  return IsSupportedComputeDType(dtype) || dtype == DataType::INT8;
}

bool IsUnifiedBroadcastOp(const CallPtr& call) {
  return IsOp(call, "tensor.add") || IsOp(call, "tensor.sub") || IsOp(call, "tensor.mul") ||
         IsOp(call, "tensor.div") || IsOp(call, "tensor.maximum") || IsOp(call, "tensor.minimum");
}

bool IsCommutativeBroadcastOp(const CallPtr& call) {
  return IsOp(call, "tensor.add") || IsOp(call, "tensor.mul") || IsOp(call, "tensor.maximum") ||
         IsOp(call, "tensor.minimum");
}

bool IsDivisionOp(const CallPtr& call) {
  return IsOp(call, "tensor.div") || IsOp(call, "tensor.row_expand_div") ||
         IsOp(call, "tensor.col_expand_div");
}

std::string BroadcastEmissionOp(const CallPtr& call, VectorGeometry geometry) {
  const std::string prefix =
      geometry == VectorGeometry::RowExpand ? "tensor.row_expand_" : "tensor.col_expand_";
  if (IsOp(call, "tensor.add")) return prefix + "add";
  if (IsOp(call, "tensor.sub")) return prefix + "sub";
  if (IsOp(call, "tensor.mul")) return prefix + "mul";
  if (IsOp(call, "tensor.div")) return prefix + "div";
  if (IsOp(call, "tensor.maximum")) return prefix + "max";
  if (IsOp(call, "tensor.minimum")) return prefix + "min";
  return call->op_->name_;
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

int64_t FrameRows(const VectorTensor& tensor, int64_t rows) {
  return tensor.rows == 1 ? 1 : std::min(tensor.rows, rows);
}

int64_t FrameCols(const VectorTensor& tensor, int64_t cols) {
  return tensor.cols == 1 ? 1 : std::min(tensor.cols, cols);
}

int64_t TensorFrameBytes(const VectorTensor& tensor, int64_t rows, int64_t cols, bool reduction_layout,
                         int64_t dma_alignment_bytes) {
  const int64_t bytes = DTypeBytes(tensor.dtype);
  const int64_t granule = std::max<int64_t>(1, dma_alignment_bytes / bytes);
  int64_t frame_rows = FrameRows(tensor, rows);
  int64_t frame_cols = FrameCols(tensor, cols);
  // A logical extent of one is not necessarily a broadcast.  The emitter
  // pads a non-broadcast reduction tile to the DMA granule even when a
  // balanced partition happens to contain one row/column.  Key padding on
  // the source tensor's shape, exactly as SliceInput does, so the planner
  // cannot under-price the common one-row softmax region.
  if (reduction_layout && tensor.rows != 1) frame_rows = AlignUp(frame_rows, granule);
  if (tensor.cols != 1) frame_cols = AlignUp(frame_cols, granule);
  return frame_rows * frame_cols * bytes;
}

int64_t RowReductionScratchBytes(const VectorTensor& tensor, int64_t rows, int64_t cols,
                                 bool reduction_layout, int64_t dma_alignment_bytes) {
  const int64_t bytes = DTypeBytes(tensor.dtype);
  const int64_t granule = std::max<int64_t>(1, dma_alignment_bytes / bytes);
  int64_t frame_rows = FrameRows(tensor, rows);
  int64_t frame_cols = FrameCols(tensor, cols);
  if (reduction_layout && tensor.rows != 1) frame_rows = AlignUp(frame_rows, granule);
  if (tensor.cols != 1) frame_cols = AlignUp(frame_cols, granule);
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
                  int64_t dma_alignment_bytes, int bands) {
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
    current += TensorFrameBytes(graph.tensors[tensor], rows, cols, reduction_layout, dma_alignment_bytes);
    peak = std::max(peak, current);
  };
  auto release = [&](size_t tensor) {
    if (live.erase(tensor) == 0) return;
    current -= TensorFrameBytes(graph.tensors[tensor], rows, cols, reduction_layout, dma_alignment_bytes);
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
      implicit_scratch = RowReductionScratchBytes(graph.tensors[op.inputs.front()], rows, cols,
                                                  reduction_layout, dma_alignment_bytes);
    } else if (IsOp(op.call, "tensor.rsqrt") && op.call->GetKwarg<bool>("high_precision", false)) {
      implicit_scratch = TensorFrameBytes(graph.tensors[op.inputs.front()], rows, cols, reduction_layout,
                                          dma_alignment_bytes);
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

struct PrimitiveCost {
  double slope;
  double fixed;
  bool count_mode;
};

PrimitiveCost PrimitiveCycles(VectorPrimitive primitive) {
  switch (primitive) {
    case VectorPrimitive::Generic:
      return {2.0, 32.0, false};
    case VectorPrimitive::Add:
      return {2.0, 24.0, true};
    case VectorPrimitive::Mul:
      return {2.0, 25.0, true};
    case VectorPrimitive::Div:
      return {4.0, 30.0, true};
    case VectorPrimitive::Exp:
      return {2.0, 31.0, false};
    case VectorPrimitive::Log:
      return {2.0, 33.0, false};
    case VectorPrimitive::Abs:
      return {1.0, 29.0, false};
    case VectorPrimitive::Sqrt:
      return {2.0, 39.0, false};
    case VectorPrimitive::Rsqrt:
      return {1.0, 24.0, false};
    case VectorPrimitive::ScalarAdd:
      return {1.0, 31.0, false};
    case VectorPrimitive::ScalarMul:
      return {1.0, 26.0, false};
    case VectorPrimitive::ScalarMax:
      return {1.0, 23.0, false};
    case VectorPrimitive::ScalarMin:
      return {1.0, 30.0, false};
    case VectorPrimitive::Cast:
      return {1.0, 24.0, false};
    case VectorPrimitive::Recip:
      return {2.0, 30.0, false};
    case VectorPrimitive::RowSum:
    case VectorPrimitive::RowExtrema:
    case VectorPrimitive::ColSum:
    case VectorPrimitive::ColExtrema:
      return {0.0, 0.0, false};
  }
  return {2.0, 32.0, false};
}

double ReductionCycles(const VectorGraph& graph, const VectorOp& op, int64_t rows, int64_t cols) {
  const VectorTensor& input = graph.tensors[op.inputs.front()];
  const int64_t element_bytes = DTypeBytes(input.dtype);
  const int64_t elements_per_repeat = std::max<int64_t>(1, kVectorRegisterBytes / element_bytes);
  const int64_t frame_rows = FrameRows(input, rows);
  const int64_t frame_cols = FrameCols(input, cols);
  if (op.kind == VectorOpKind::ColSum || op.kind == VectorOpKind::ColMax) {
    return 16.0 * std::max<int64_t>(0, frame_rows - 1) +
           30.0 * (frame_rows > 1 ? std::log2(static_cast<double>(frame_rows)) : 0.0);
  }
  const int64_t passes = std::max<int64_t>(1, (frame_cols + elements_per_repeat - 1) / elements_per_repeat);
  return 45.0 * static_cast<double>(passes - 1) + 51.0;
}

double ComputeCycles(const VectorGraph& graph, const std::vector<size_t>& ops, int64_t rows, int64_t cols,
                     int64_t iterations, int64_t work_units) {
  double per_frame = 0.0;
  bool stream_start = true;
  for (size_t op_index : ops) {
    const VectorOp& op = graph.ops[op_index];
    if (IsReduction(op.kind)) {
      per_frame += ReductionCycles(graph, op, rows, cols);
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
    const int64_t epr = std::max<int64_t>(1, kVectorRegisterBytes / DTypeBytes(output.dtype));
    const int64_t repeats = op.geometry == VectorGeometry::Flat ? (frame_rows * frame_cols + epr - 1) / epr
                                                                : frame_rows * ((frame_cols + epr - 1) / epr);
    PrimitiveCost cost = PrimitiveCycles(op.primitive);
    if (op.geometry == VectorGeometry::RowExpand) cost.fixed += 19.0;
    per_frame += cost.slope * static_cast<double>(repeats) + (stream_start ? cost.fixed : 0.0);
    if (cost.count_mode && frame_cols % epr != 0) per_frame += kCountModeFloorCycles;
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

bool LexicographicallyBetter(const VectorSchedulePlan& lhs, const VectorSchedulePlan& rhs) {
  if (lhs.modeled_cycles != rhs.modeled_cycles) return lhs.modeled_cycles < rhs.modeled_cycles;
  if (lhs.work_units != rhs.work_units) return lhs.work_units < rhs.work_units;
  if (lhs.tile_h != rhs.tile_h) return lhs.tile_h > rhs.tile_h;
  if (lhs.tile_w != rhs.tile_w) return lhs.tile_w > rhs.tile_w;
  if (lhs.reduction_split.factor != rhs.reduction_split.factor)
    return lhs.reduction_split.factor < rhs.reduction_split.factor;
  return std::tie(lhs.m_partition.parts, lhs.n_partition.parts) <
         std::tie(rhs.m_partition.parts, rhs.n_partition.parts);
}

}  // namespace

int64_t DTypeBytes(const DataType& dtype) { return static_cast<int64_t>(dtype.GetByte()); }

std::pair<int64_t, int64_t> StaticTensorShape(const TypePtr& type) {
  auto tensor = As<TensorType>(type);
  if (tensor == nullptr || tensor->shape_.size() != 2) return {-1, -1};
  auto rows = As<ConstInt>(tensor->shape_[0]);
  auto cols = As<ConstInt>(tensor->shape_[1]);
  if (rows == nullptr || cols == nullptr) return {-1, -1};
  return {rows->value_, cols->value_};
}

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

VectorGraph VectorGraph::Build(const FunctionPtr& function, const ProgramPtr& program) {
  (void)program;
  CHECK_SPAN(function != nullptr && function->body_ != nullptr, function ? function->span_ : Span::unknown())
      << "AutoTile requires a function body";

  VectorGraph graph;
  graph.function = function;
  std::unordered_map<const Var*, size_t> producer;
  std::unordered_map<const Var*, ParamDirection> parameter_directions;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    const ParamDirection direction =
        i < function->param_directions_.size() ? function->param_directions_[i] : ParamDirection::In;
    parameter_directions.emplace(function->params_[i].get(), direction);
  }

  auto register_tensor = [&](const VarPtr& var, bool boundary_input) -> size_t {
    auto found = graph.tensor_by_var.find(var.get());
    if (found != graph.tensor_by_var.end()) return found->second;
    auto tensor_type = As<TensorType>(var->GetType());
    CHECK_SPAN(tensor_type != nullptr, var->span_) << "AutoTile supports tensor operation values only";
    const auto [rows, cols] = StaticTensorShape(tensor_type);
    CHECK_SPAN(rows > 0 && cols > 0, var->span_)
        << "AutoTile supports positive, static rank-2 tensors; value '" << var->name_hint_
        << "' does not have that shape";
    CHECK_SPAN(IsSupportedTensorDType(tensor_type->dtype_), var->span_)
        << "AutoTile vector scheduling does not support dtype " << tensor_type->dtype_.ToString();
    const size_t id = graph.tensors.size();
    graph.tensors.push_back({var, rows, cols, tensor_type->dtype_, boundary_input, false});
    graph.tensor_by_var.emplace(var.get(), id);
    return id;
  };

  std::vector<StmtPtr> statements;
  if (auto sequence = As<SeqStmts>(function->body_))
    statements = sequence->stmts_;
  else
    statements.push_back(function->body_);

  ReturnStmtPtr return_stmt;
  for (const StmtPtr& stmt : statements) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      CHECK_SPAN(return_stmt == nullptr, ret->span_) << "AutoTile requires one top-level return";
      return_stmt = ret;
      continue;
    }
    auto assign = As<AssignStmt>(stmt);
    CHECK_SPAN(assign != nullptr, stmt->span_) << "AutoTile requires a straight-line tensor DAG; control "
                                                  "flow and effectful statements are unsupported";
    auto call = As<Call>(assign->value_);
    CHECK_SPAN(call != nullptr && call->op_ != nullptr, assign->span_)
        << "AutoTile requires every tensor definition to be a direct operation call";
    auto output_type = As<TensorType>(assign->var_->GetType());
    CHECK_SPAN(output_type != nullptr, assign->span_)
        << "AutoTile does not schedule scalar-producing statements inside the marked function";
    auto descriptor = DescribeOp(call);
    CHECK_SPAN(descriptor.has_value(), call->span_)
        << "AutoTile cannot form one vector schedule because operation '" << call->op_->name_
        << "' is unsupported";

    const size_t output = register_tensor(assign->var_, false);
    CHECK_SPAN(producer.emplace(assign->var_.get(), graph.ops.size()).second, assign->span_)
        << "AutoTile requires SSA tensor definitions";
    VectorOp op;
    op.stmt = assign;
    op.call = call;
    op.emission_op = call->op_->name_;
    op.kind = descriptor->kind;
    op.primitive = descriptor->primitive;
    op.geometry = descriptor->geometry;
    op.output = output;
    for (const ExprPtr& arg : call->args_) {
      auto tensor_type = As<TensorType>(arg->GetType());
      if (tensor_type == nullptr) continue;
      auto var = AsVarLike(arg);
      CHECK_SPAN(var != nullptr, call->span_)
          << "AutoTile requires tensor operands to be named SSA values after FlattenCallExpr";
      const bool is_boundary = producer.count(var.get()) == 0;
      if (is_boundary) {
        auto param = parameter_directions.find(var.get());
        CHECK_SPAN(param != parameter_directions.end() &&
                       (param->second == ParamDirection::In || param->second == ParamDirection::InOut),
                   call->span_)
            << "AutoTile tensor input '" << var->name_hint_ << "' is not a readable function parameter";
      }
      const size_t input = register_tensor(var, is_boundary);
      op.inputs.push_back(input);
      CHECK_SPAN(IsSupportedComputeDType(tensor_type->dtype_), call->span_)
          << "AutoTile supports INT8 only as a terminal tensor.cast output, not as vector compute input";
    }
    CHECK_SPAN(!op.inputs.empty(), call->span_) << "AutoTile vector operations require a tensor operand";
    if (op.geometry == VectorGeometry::Flat && IsUnifiedBroadcastOp(call) && op.inputs.size() == 2) {
      bool row_expand = false;
      bool col_expand = false;
      size_t narrow = 0;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        const VectorTensor& input = graph.tensors[op.inputs[i]];
        const VectorTensor& result = graph.tensors[op.output];
        if (input.rows == result.rows && input.cols == 1 && result.cols > 1) {
          row_expand = true;
          narrow = i;
        }
        if (input.rows == 1 && input.cols == result.cols && result.rows > 1) {
          col_expand = true;
          narrow = i;
        }
      }
      CHECK_SPAN(!(row_expand && col_expand), call->span_)
          << "AutoTile requires an explicit row/column expansion for an ambiguous [1,1] tensor broadcast";
      if (row_expand || col_expand) {
        CHECK_SPAN(narrow != 0 || IsCommutativeBroadcastOp(call), call->span_)
            << "AutoTile cannot reverse a broadcasted lhs for non-commutative operation '" << call->op_->name_
            << "'";
        op.geometry = row_expand ? VectorGeometry::RowExpand : VectorGeometry::ColExpand;
        op.emission_op = BroadcastEmissionOp(call, op.geometry);
        op.swap_operands = narrow == 0;
      }
    }
    CHECK_SPAN(!(IsDivisionOp(call) && call->GetKwarg<bool>("high_precision", false) &&
                 op.geometry != VectorGeometry::Flat),
               call->span_)
        << "AutoTile does not support high-precision division with a broadcast operand";
    if (IsOp(call, "tensor.cast")) {
      CHECK_SPAN(IsSupportedComputeDType(output_type->dtype_) ||
                     (graph.tensors[op.inputs.front()].dtype == DataType::FP32 &&
                      output_type->dtype_ == DataType::INT8),
                 call->span_)
          << "AutoTile supports compute-dtype casts and terminal FP32-to-INT8 casts";
    } else {
      CHECK_SPAN(IsSupportedComputeDType(output_type->dtype_), call->span_)
          << "AutoTile supports FP32, FP16, and BF16 vector compute outputs";
      for (size_t input : op.inputs) {
        CHECK_SPAN(graph.tensors[input].dtype == output_type->dtype_, call->span_)
            << "AutoTile requires explicit tensor.cast for mixed tensor dtypes";
      }
      CHECK_SPAN(!(IsDivisionOp(call) && op.geometry != VectorGeometry::Flat &&
                   output_type->dtype_ == DataType::BF16),
                 call->span_)
          << "AutoTile broadcast division supports FP16 and FP32 only";
    }
    graph.ops.push_back(std::move(op));
  }

  CHECK_SPAN(return_stmt != nullptr && !return_stmt->value_.empty(), function->span_)
      << "AutoTile requires at least one returned tensor";
  std::unordered_set<size_t> unique_outputs;
  for (const ExprPtr& value : return_stmt->value_) {
    auto var = AsVarLike(value);
    CHECK_SPAN(var != nullptr, return_stmt->span_)
        << "AutoTile requires returned operations to be named by FlattenCallExpr";
    auto tensor = graph.tensor_by_var.find(var.get());
    auto defining_op = producer.find(var.get());
    CHECK_SPAN(tensor != graph.tensor_by_var.end() && defining_op != producer.end(), return_stmt->span_)
        << "AutoTile requires every returned tensor to be produced inside the marked function";
    CHECK_SPAN(unique_outputs.insert(tensor->second).second, return_stmt->span_)
        << "AutoTile does not support duplicate returned tensors";
    graph.tensors[tensor->second].required_output = true;
    graph.required_outputs.push_back(tensor->second);
    graph.required_output_ops.push_back(defining_op->second);
  }
  CHECK_SPAN(!graph.ops.empty(), function->span_) << "AutoTile found no tensor operations to schedule";

  std::unordered_map<size_t, size_t> use_count;
  for (const VectorOp& op : graph.ops)
    for (size_t input : op.inputs) ++use_count[input];
  for (size_t tensor = 0; tensor < graph.tensors.size(); ++tensor) {
    if (graph.tensors[tensor].dtype != DataType::INT8) continue;
    CHECK_SPAN(graph.tensors[tensor].required_output && use_count[tensor] == 0,
               graph.tensors[tensor].var->span_)
        << "AutoTile supports INT8 only as an unconsumed returned cast result";
  }

  int64_t iteration_rows = 1;
  int64_t iteration_cols = 1;
  int reduction_count = 0;
  for (size_t i = 0; i < graph.ops.size(); ++i) {
    const VectorOp& op = graph.ops[i];
    for (size_t tensor : op.inputs) {
      iteration_rows = std::max(iteration_rows, graph.tensors[tensor].rows);
      iteration_cols = std::max(iteration_cols, graph.tensors[tensor].cols);
    }
    iteration_rows = std::max(iteration_rows, graph.tensors[op.output].rows);
    iteration_cols = std::max(iteration_cols, graph.tensors[op.output].cols);
    if (IsReduction(op.kind)) {
      ++reduction_count;
      const int axis = (op.kind == VectorOpKind::RowSum || op.kind == VectorOpKind::RowMax) ? 1 : 2;
      CHECK_SPAN(graph.reduced_axis == 0 || graph.reduced_axis == axis, op.stmt->span_)
          << "AutoTile does not support reductions over both axes in one vector schedule";
      graph.reduced_axis = axis;
      graph.reduction_op = i;
    }
  }

  for (const VectorOp& op : graph.ops) {
    const VectorTensor& output = graph.tensors[op.output];
    for (size_t input : op.inputs) {
      const VectorTensor& tensor = graph.tensors[input];
      CHECK_SPAN((tensor.rows == 1 || tensor.rows == iteration_rows) &&
                     (tensor.cols == 1 || tensor.cols == iteration_cols),
                 op.stmt->span_)
          << "AutoTile supports only full-frame or row/column-broadcast tensor operands";
    }
    if (op.kind == VectorOpKind::Elementwise) {
      CHECK_SPAN((output.rows == 1 || output.rows == iteration_rows) &&
                     (output.cols == 1 || output.cols == iteration_cols),
                 op.stmt->span_)
          << "AutoTile elementwise output shape is outside the common iteration frame";
    } else if (op.kind == VectorOpKind::RowSum || op.kind == VectorOpKind::RowMax) {
      CHECK_SPAN(output.rows == iteration_rows && output.cols == 1, op.stmt->span_)
          << "AutoTile row reduction must map [M,N] to [M,1]";
    } else {
      CHECK_SPAN(output.rows == 1 && output.cols == iteration_cols, op.stmt->span_)
          << "AutoTile column reduction must map [M,N] to [1,N]";
    }
  }
  graph.iteration_rows = iteration_rows;
  graph.iteration_cols = iteration_cols;

  if (graph.ops.size() == 5 && graph.required_outputs.size() == 1) {
    const VectorOp& max_op = graph.ops[0];
    const VectorOp& shift_op = graph.ops[1];
    const VectorOp& exp_op = graph.ops[2];
    const VectorOp& sum_op = graph.ops[3];
    const VectorOp& sink_op = graph.ops[4];
    const bool shift = IsOp(shift_op.call, "tensor.sub") || IsOp(shift_op.call, "tensor.row_expand_sub");
    const bool sink = IsOp(sink_op.call, "tensor.div") || IsOp(sink_op.call, "tensor.row_expand_div");
    const bool exact = max_op.kind == VectorOpKind::RowMax && shift && IsOp(exp_op.call, "tensor.exp") &&
                       sum_op.kind == VectorOpKind::RowSum && sink && max_op.inputs.size() == 1 &&
                       shift_op.inputs == std::vector<size_t>{max_op.inputs[0], max_op.output} &&
                       exp_op.inputs == std::vector<size_t>{shift_op.output} &&
                       sum_op.inputs == std::vector<size_t>{exp_op.output} &&
                       sink_op.inputs == std::vector<size_t>{exp_op.output, sum_op.output} &&
                       graph.required_outputs[0] == sink_op.output;
    if (exact) graph.softmax = {true, max_op.inputs[0], 0, 2, 3, 4};
  }
  CHECK_SPAN(reduction_count <= 1 || graph.softmax.matched, function->span_)
      << "AutoTile supports multiple reductions only for the canonical online softmax graph";

  return graph;
}

VectorSchedulePlan VectorPlanner910B::Plan(const VectorGraph& graph) const {
  CHECK(hardware_.vector_cores > 0 && hardware_.ub_bytes > 0 && hardware_.dma_alignment_bytes > 0)
      << "AutoTile vector planner received an invalid Ascend 910B hardware descriptor";

  const int64_t iteration_rows = graph.iteration_rows;
  const int64_t iteration_cols = graph.iteration_cols;
  const bool reduction_layout = graph.reduced_axis != 0;
  const std::unordered_set<size_t> all_outputs(graph.required_outputs.begin(), graph.required_outputs.end());
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
    candidate.full_peak_ub_bytes = PeakBytes(graph, all_ops, candidate.tile_h, candidate.tile_w, all_outputs,
                                             {}, reduction_layout, hardware_.dma_alignment_bytes, 1);

    auto price_phase = [&](const VectorPhasePlan& phase, int64_t rows, int64_t cols, int64_t iterations,
                           int stages, const std::unordered_set<size_t>& outputs) {
      const int64_t tasks = candidate.work_units;
      const int64_t active = std::min<int64_t>(tasks, hardware_.vector_cores);
      const double compute = WaveCompute(ComputeCycles(graph, phase.ops, rows, cols, iterations, tasks),
                                         tasks, hardware_.vector_cores);
      const double in_bytes = BoundaryInputBytes(graph, phase, rows, cols, iterations, tasks);
      const double out_bytes = OutputBytes(graph, outputs, rows, cols, iterations, tasks);
      const double transfer =
          TransferCycles(in_bytes, kGmToUbGiBps, active) + TransferCycles(out_bytes, kUbToGmGiBps, active);
      candidate.modeled_compute_cycles += compute;
      candidate.modeled_transfer_cycles += transfer;
      return stages == 2 ? std::max(compute, transfer) : compute + transfer;
    };

    double latency = 0.0;
    bool feasible = false;
    if (candidate.full_peak_ub_bytes <= hardware_.ub_bytes && !graph.softmax.matched) {
      candidate.kind = VectorScheduleKind::Materialized;
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Body)], graph, all_ops);
      candidate.strip_h = candidate.tile_h;
      candidate.strip_w = candidate.tile_w;
      candidate.chunk_peak_ub_bytes = candidate.full_peak_ub_bytes;
      latency = price_phase(candidate.phases[PhaseIndex(VectorPhase::Body)], candidate.tile_h,
                            candidate.tile_w, 1, 1, all_outputs);
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
            const int64_t peak = PeakBytes(graph, all_ops, strip_h, strip_w, all_outputs, {}, false,
                                           hardware_.dma_alignment_bytes, 2);
            if (peak > hardware_.ub_bytes) continue;
            VectorSchedulePlan trial = candidate;
            trial.kind = VectorScheduleKind::PointwiseStream;
            trial.row_strips = row_strips;
            trial.width_strips = width_strips;
            trial.strip_h = strip_h;
            trial.strip_w = strip_w;
            trial.chunk_peak_ub_bytes = peak;
            trial.phases[PhaseIndex(VectorPhase::Body)].trip_count = trips;
            trial.phases[PhaseIndex(VectorPhase::Body)].pipeline_stages = 2;
            const int64_t tasks = trial.work_units;
            const int64_t active = std::min<int64_t>(tasks, hardware_.vector_cores);
            const double compute = WaveCompute(ComputeCycles(graph, all_ops, strip_h, strip_w, trips, tasks),
                                               tasks, hardware_.vector_cores);
            const double in_bytes = BoundaryInputBytes(graph, trial.phases[PhaseIndex(VectorPhase::Body)],
                                                       strip_h, strip_w, trips, tasks);
            const double out_bytes = OutputBytes(graph, all_outputs, strip_h, strip_w, trips, tasks);
            const double transfer = TransferCycles(in_bytes, kGmToUbGiBps, active) +
                                    TransferCycles(out_bytes, kUbToGmGiBps, active);
            const double trial_latency = std::max(compute, transfer);
            if (trial_latency < best_stream_latency) {
              streamed = std::move(trial);
              streamed.modeled_compute_cycles = compute;
              streamed.modeled_transfer_cycles = transfer;
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
          const int64_t peak = PeakBytes(graph, all_ops, strip_h, strip_w, all_outputs, {}, false,
                                         hardware_.dma_alignment_bytes, 2);
          if (peak > hardware_.ub_bytes) continue;
          const int64_t active = std::min<int64_t>(candidate.work_units, hardware_.vector_cores);
          const double compute =
              WaveCompute(ComputeCycles(graph, all_ops, strip_h, strip_w, trips, candidate.work_units),
                          candidate.work_units, hardware_.vector_cores);
          const double in_bytes = BoundaryInputBytes(graph, candidate.phases[PhaseIndex(VectorPhase::Body)],
                                                     strip_h, strip_w, trips, candidate.work_units);
          const double out_bytes =
              OutputBytes(graph, all_outputs, strip_h, strip_w, trips, candidate.work_units);
          const double transfer = TransferCycles(in_bytes, kGmToUbGiBps, active) +
                                  TransferCycles(out_bytes, kUbToGmGiBps, active);
          const double trial_latency = std::max(compute, transfer);
          if (trial_latency >= best_stream_latency) continue;
          best_stream_latency = trial_latency;
          candidate.kind = VectorScheduleKind::PointwiseStream;
          candidate.row_strips = row_strips;
          candidate.width_strips = width_strips;
          candidate.strip_h = strip_h;
          candidate.strip_w = strip_w;
          candidate.chunk_peak_ub_bytes = peak;
          candidate.phases[PhaseIndex(VectorPhase::Body)].trip_count = trips;
          candidate.phases[PhaseIndex(VectorPhase::Body)].pipeline_stages = 2;
          candidate.modeled_compute_cycles = compute;
          candidate.modeled_transfer_cycles = transfer;
          latency = trial_latency;
          feasible = true;
        }
      }
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
      candidate.free_tile_alloc =
          AlignUp(candidate.free_tile, hardware_.dma_alignment_bytes / DTypeBytes(reduction_dtype));
      candidate.reduced_extent = graph.reduced_axis == 1 ? iteration_cols : iteration_rows;
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Stats)], graph, stats_ops);
      PopulatePhase(&candidate.phases[PhaseIndex(VectorPhase::Apply)], graph, apply_ops, substituted);
      if (graph.softmax.matched) {
        candidate.phases[PhaseIndex(VectorPhase::Stats)].ops.clear();
        candidate.phases[PhaseIndex(VectorPhase::Stats)].inputs = {
            {graph.softmax.input, 0, 0, {{graph.softmax.max_op, 0}}}};
      }

      int64_t best_chunk = 0;
      int64_t best_peak = 0;
      for (int64_t chunk = std::min<int64_t>(candidate.reduced_extent, 4096); chunk >= 1; --chunk) {
        if (chunk != candidate.reduced_extent && chunk % 16 != 0) continue;
        const int64_t rows = graph.reduced_axis == 1 ? candidate.free_tile : chunk;
        const int64_t cols = graph.reduced_axis == 1 ? chunk : candidate.free_tile;
        const int64_t thin_rows = graph.reduced_axis == 1 ? candidate.free_tile : 1;
        const int64_t thin_cols = graph.reduced_axis == 1 ? 1 : candidate.free_tile;
        int64_t stats_peak;
        if (graph.softmax.matched) {
          const int64_t x_bytes = TensorFrameBytes(graph.tensors[graph.softmax.input], rows, cols, true,
                                                   hardware_.dma_alignment_bytes);
          const int64_t thin_bytes = candidate.free_tile_alloc * DTypeBytes(reduction_dtype);
          stats_peak = 2 * (6 * x_bytes + 4 * thin_bytes);
        } else {
          stats_peak = PeakBytes(graph, stats_ops, rows, cols, {reduction_tensor}, {}, true,
                                 hardware_.dma_alignment_bytes, 2) +
                       2 * candidate.free_tile_alloc * DTypeBytes(reduction_dtype);
        }
        int64_t apply_peak = 0;
        if (candidate.kind == VectorScheduleKind::ReductionSpanning || graph.softmax.matched) {
          apply_peak = PeakBytes(graph, apply_ops, rows, cols, all_outputs, substituted, true,
                                 hardware_.dma_alignment_bytes, 2);
        } else {
          apply_peak = PeakBytes(graph, apply_ops, thin_rows, thin_cols, all_outputs, substituted, true,
                                 hardware_.dma_alignment_bytes, 1);
        }
        const int64_t peak = std::max(stats_peak, apply_peak);
        if (peak <= hardware_.ub_bytes) {
          best_chunk = chunk;
          best_peak = peak;
          break;
        }
      }
      if (best_chunk > 0) {
        candidate.chunk = best_chunk;
        candidate.full_chunks = candidate.reduced_extent / best_chunk;
        candidate.tail = candidate.reduced_extent % best_chunk;
        candidate.chunk_peak_ub_bytes = best_peak;
        VectorPhasePlan& stats = candidate.phases[PhaseIndex(VectorPhase::Stats)];
        stats.first_chunk = 1;
        stats.trip_count = std::max<int64_t>(0, candidate.full_chunks - 1);
        stats.pipeline_stages = stats.trip_count >= 2 ? 2 : 1;
        VectorPhasePlan& apply = candidate.phases[PhaseIndex(VectorPhase::Apply)];
        apply.first_chunk = 0;
        apply.trip_count = candidate.full_chunks;
        apply.pipeline_stages = apply.trip_count >= 2 ? 2 : 1;

        const int64_t rows = graph.reduced_axis == 1 ? candidate.free_tile : candidate.chunk;
        const int64_t cols = graph.reduced_axis == 1 ? candidate.chunk : candidate.free_tile;
        const int64_t tasks = candidate.work_units;
        const int64_t active = std::min<int64_t>(tasks, hardware_.vector_cores);
        if (graph.softmax.matched) {
          const double wide = static_cast<double>(rows * cols);
          const double epr = static_cast<double>(kVectorRegisterBytes / DTypeBytes(reduction_dtype));
          const double repeats = std::ceil(wide / epr);
          const double per_chunk = 10.0 * repeats + 180.0;
          const double total_compute = per_chunk * static_cast<double>(candidate.full_chunks) * tasks;
          const double stats_compute = WaveCompute(total_compute, tasks, hardware_.vector_cores);
          const double stats_bytes =
              static_cast<double>(rows * cols * DTypeBytes(reduction_dtype) * candidate.full_chunks * tasks);
          const double stats_transfer = TransferCycles(stats_bytes, kGmToUbGiBps, active);
          latency += stats.pipeline_stages == 2 ? std::max(stats_compute, stats_transfer)
                                                : stats_compute + stats_transfer;
          candidate.modeled_compute_cycles += stats_compute;
          candidate.modeled_transfer_cycles += stats_transfer;
          if (candidate.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? candidate.free_tile : candidate.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? candidate.tail : candidate.free_tile;
            const double tail_wide = static_cast<double>(tail_rows * tail_cols);
            const double tail_compute = WaveCompute((10.0 * std::ceil(tail_wide / epr) + 180.0) * tasks,
                                                    tasks, hardware_.vector_cores);
            const double tail_bytes = tail_wide * static_cast<double>(DTypeBytes(reduction_dtype) * tasks);
            const double tail_transfer = TransferCycles(tail_bytes, kGmToUbGiBps, active);
            latency += tail_compute + tail_transfer;
            candidate.modeled_compute_cycles += tail_compute;
            candidate.modeled_transfer_cycles += tail_transfer;
          }
        } else {
          latency += price_phase(stats, rows, cols, std::max<int64_t>(1, candidate.full_chunks),
                                 stats.pipeline_stages, {});
          if (candidate.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? candidate.free_tile : candidate.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? candidate.tail : candidate.free_tile;
            latency += price_phase(stats, tail_rows, tail_cols, 1, 1, {});
          }
          const int64_t merges = candidate.full_chunks - 1 + (candidate.tail > 0 ? 1 : 0);
          if (merges > 0) {
            const PrimitiveCost merge = PrimitiveCycles(VectorPrimitive::Add);
            const int64_t repeats =
                (candidate.free_tile + kVectorRegisterBytes / DTypeBytes(reduction_dtype) - 1) /
                (kVectorRegisterBytes / DTypeBytes(reduction_dtype));
            const double merge_compute =
                WaveCompute((merge.slope * static_cast<double>(repeats) + merge.fixed) * merges * tasks,
                            tasks, hardware_.vector_cores);
            latency += merge_compute;
            candidate.modeled_compute_cycles += merge_compute;
          }
        }
        if (candidate.kind == VectorScheduleKind::ReductionSpanning || graph.softmax.matched) {
          latency += price_phase(apply, rows, cols, std::max<int64_t>(1, candidate.full_chunks),
                                 apply.pipeline_stages, all_outputs);
          if (candidate.tail > 0) {
            const int64_t tail_rows = graph.reduced_axis == 1 ? candidate.free_tile : candidate.tail;
            const int64_t tail_cols = graph.reduced_axis == 1 ? candidate.tail : candidate.free_tile;
            latency += price_phase(apply, tail_rows, tail_cols, 1, 1, all_outputs);
          }
        } else {
          candidate.phases[PhaseIndex(VectorPhase::Finalize)] = apply;
          candidate.phases[PhaseIndex(VectorPhase::Apply)] = {};
          const int64_t thin_rows = graph.reduced_axis == 1 ? candidate.free_tile : 1;
          const int64_t thin_cols = graph.reduced_axis == 1 ? 1 : candidate.free_tile;
          latency += price_phase(candidate.phases[PhaseIndex(VectorPhase::Finalize)], thin_rows, thin_cols, 1,
                                 1, all_outputs);
        }
        feasible = true;
      }
    }

    if (!feasible) continue;

    if (candidate.kind == VectorScheduleKind::Materialized && graph.ops.size() >= 1 &&
        graph.ops.back().kind == VectorOpKind::ColSum && graph.required_outputs.size() == 1 &&
        graph.required_output_ops[0] == graph.ops.size() - 1 && iteration_cols % candidate.tile_w == 0 &&
        candidate.n_partition.num_big == 0 &&
        candidate.tile_w * DTypeBytes(graph.tensors[graph.required_outputs[0]].dtype) >=
            hardware_.dma_alignment_bytes) {
      const int64_t max_split =
          std::min<int64_t>(hardware_.vector_cores / candidate.work_units, iteration_rows / 16);
      for (int64_t split = 2; split <= max_split; ++split) {
        if (iteration_rows % (16 * split) != 0) continue;
        const int64_t split_tasks = candidate.work_units * split;
        const int64_t seed_tasks = candidate.work_units;
        const int64_t partial_rows = iteration_rows / split;
        const int64_t split_peak = PeakBytes(graph, all_ops, partial_rows, candidate.tile_w, all_outputs, {},
                                             true, hardware_.dma_alignment_bytes, 1);
        if (split_peak > hardware_.ub_bytes) continue;
        const double split_compute =
            WaveCompute(ComputeCycles(graph, all_ops, partial_rows, candidate.tile_w, 1, split_tasks),
                        split_tasks, hardware_.vector_cores);
        const double seed_compute =
            WaveCompute(kPerTaskCycles * static_cast<double>(seed_tasks), seed_tasks, hardware_.vector_cores);
        const double body_in_bytes =
            BoundaryInputBytes(graph, candidate.phases[PhaseIndex(VectorPhase::Body)], partial_rows,
                               candidate.tile_w, 1, split_tasks);
        const double body_out_bytes = OutputBytes(graph, all_outputs, 1, candidate.tile_w, 1, split_tasks);
        const double seed_out_bytes = OutputBytes(graph, all_outputs, 1, candidate.tile_w, 1, seed_tasks);
        const double body_transfer = TransferCycles(body_in_bytes, kGmToUbGiBps,
                                                    std::min<int64_t>(split_tasks, hardware_.vector_cores)) +
                                     TransferCycles(body_out_bytes, kUbToGmGiBps,
                                                    std::min<int64_t>(split_tasks, hardware_.vector_cores));
        const double seed_transfer = TransferCycles(seed_out_bytes, kUbToGmGiBps,
                                                    std::min<int64_t>(seed_tasks, hardware_.vector_cores));
        const double split_latency = split_compute + body_transfer + seed_compute + seed_transfer;
        if (split_latency < latency) {
          latency = split_latency;
          candidate.reduction_split = {true, split, partial_rows, seed_tasks};
          candidate.chunk_peak_ub_bytes = std::max(
              split_peak, candidate.tile_w * DTypeBytes(graph.tensors[graph.required_outputs[0]].dtype));
          candidate.modeled_compute_cycles = split_compute + seed_compute;
          candidate.modeled_transfer_cycles = body_transfer + seed_transfer;
        }
      }
    }

    const int64_t body_tasks = candidate.work_units * candidate.reduction_split.factor;
    const int64_t seed_tasks =
        candidate.reduction_split.present ? candidate.reduction_split.seed_work_units : 0;
    latency += kPerTaskCycles * static_cast<double>(body_tasks + seed_tasks);
    latency += kKernelFillCycles *
               static_cast<double>(
                   (body_tasks + hardware_.vector_cores - 1) / hardware_.vector_cores +
                   (seed_tasks > 0 ? (seed_tasks + hardware_.vector_cores - 1) / hardware_.vector_cores : 0));
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
