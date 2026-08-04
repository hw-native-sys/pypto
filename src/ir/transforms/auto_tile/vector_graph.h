/*
 * Copyright (c) PyPTO Contributors. This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_GRAPH_H_
#define PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
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
};

/** Fail-closed result of admitting one tensor-level function to vector AutoTile. */
struct VectorAdmissionResult {
  bool supported = false;
  VectorGraph graph;
  std::string reason;
  std::exception_ptr failure;
};

/** Analyze a function without mutating it. Unsupported graphs return a diagnostic instead of a partial graph.
 */
[[nodiscard]] VectorAdmissionResult AdmitVectorGraph(const FunctionPtr& function, const ProgramPtr& program);

[[nodiscard]] int64_t DTypeBytes(const DataType& dtype);
[[nodiscard]] std::pair<int64_t, int64_t> StaticTensorShape(const TypePtr& type);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_GRAPH_H_
