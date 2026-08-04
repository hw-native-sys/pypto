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
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

enum class VectorOpKind : uint8_t {
  Elementwise,
  RowSum,
  RowMax,
  ColSum,
  ColMax,
};

enum class VectorPrimitive : uint8_t {
  Generic,
  Add,
  Mul,
  Div,
  Exp,
  Log,
  Abs,
  Sqrt,
  Rsqrt,
  ScalarAdd,
  ScalarMul,
  ScalarMax,
  ScalarMin,
  Cast,
  Recip,
  RowSum,
  RowExtrema,
  ColSum,
  ColExtrema,
};

enum class VectorGeometry : uint8_t {
  Flat,
  RowExpand,
  ColExpand,
};

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

struct VectorTensor {
  VarPtr var;
  int64_t rows = 0;
  int64_t cols = 0;
  DataType dtype = DataType::FP32;
  bool boundary_input = false;
  bool required_output = false;
};

struct VectorOp {
  AssignStmtPtr stmt;
  CallPtr call;
  std::string emission_op;
  bool swap_operands = false;
  VectorOpKind kind = VectorOpKind::Elementwise;
  VectorPrimitive primitive = VectorPrimitive::Add;
  VectorGeometry geometry = VectorGeometry::Flat;
  std::vector<size_t> inputs;
  size_t output = 0;
};

struct SoftmaxPattern {
  bool matched = false;
  size_t input = 0;
  size_t max_op = 0;
  size_t exp_op = 0;
  size_t sum_op = 0;
  size_t sink_op = 0;
};

struct VectorGraph {
  FunctionPtr function;
  int64_t iteration_rows = 0;
  int64_t iteration_cols = 0;
  std::vector<VectorTensor> tensors;
  std::vector<VectorOp> ops;
  std::vector<size_t> required_outputs;
  std::vector<size_t> required_output_ops;
  std::unordered_map<const Var*, size_t> tensor_by_var;
  SoftmaxPattern softmax;
  int reduced_axis = 0;  // 0 = none, 1 = width, 2 = height
  size_t reduction_op = std::numeric_limits<size_t>::max();

  [[nodiscard]] static VectorGraph Build(const FunctionPtr& function, const ProgramPtr& program);
};

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

struct VectorReductionSplitPlan {
  bool present = false;
  int64_t factor = 1;
  int64_t partial_extent = 0;
  int64_t seed_work_units = 0;
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
  int64_t full_chunks = 0;
  int64_t tail = 0;
  std::array<VectorPhasePlan, 4> phases;
  VectorReductionSplitPlan reduction_split;
  double modeled_cycles = std::numeric_limits<double>::infinity();
  double modeled_compute_cycles = 0.0;
  double modeled_transfer_cycles = 0.0;
};

struct VectorHardware {
  int vector_cores = 0;
  int64_t ub_bytes = 0;
  int64_t dma_alignment_bytes = 0;
};

class VectorPlanner910B {
 public:
  explicit VectorPlanner910B(VectorHardware hardware) : hardware_(hardware) {}

  [[nodiscard]] VectorSchedulePlan Plan(const VectorGraph& graph) const;

 private:
  VectorHardware hardware_;
};

[[nodiscard]] int64_t DTypeBytes(const DataType& dtype);
[[nodiscard]] std::pair<int64_t, int64_t> StaticTensorShape(const TypePtr& type);
[[nodiscard]] const char* ScheduleKindName(VectorScheduleKind kind);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_PLAN_H_
