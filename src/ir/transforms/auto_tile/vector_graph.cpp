/*
 * Copyright (c) PyPTO Contributors. This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/ir/transforms/auto_tile/vector_graph.h"

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>

#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

struct OpDescriptor {
  VectorOpKind kind;
  VectorPrimitive primitive;
  VectorGeometry geometry;
};

struct OpAdmission {
  std::optional<OpDescriptor> descriptor;
  std::string reason;
};

OpAdmission AdmitOp(const CallPtr& call) {
  if (IsOp(call, "tensor.row_sum"))
    return {{OpDescriptor{VectorOpKind::RowSum, VectorPrimitive::RowSum, VectorGeometry::Flat}}, {}};
  if (IsOp(call, "tensor.row_max"))
    return {{OpDescriptor{VectorOpKind::RowMax, VectorPrimitive::RowExtrema, VectorGeometry::Flat}}, {}};
  if (IsOp(call, "tensor.col_sum"))
    return {{OpDescriptor{VectorOpKind::ColSum, VectorPrimitive::ColSum, VectorGeometry::Flat}}, {}};
  if (IsOp(call, "tensor.col_max"))
    return {{OpDescriptor{VectorOpKind::ColMax, VectorPrimitive::ColExtrema, VectorGeometry::Flat}}, {}};

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
  } else if (IsOp(call, "tensor.exp") || IsOp(call, "tensor.row_expand_expdif") ||
             IsOp(call, "tensor.col_expand_expdif")) {
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
    return {std::nullopt, "operation '" + call->op_->name_ + "' is unsupported"};
  }
  return {{OpDescriptor{VectorOpKind::Elementwise, primitive, geometry}}, {}};
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

VectorGraph BuildVectorGraphOrThrow(const FunctionPtr& function, const ProgramPtr& program) {
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
    const OpAdmission admission = AdmitOp(call);
    CHECK_SPAN(admission.descriptor.has_value(), call->span_)
        << "AutoTile cannot form one vector schedule because " << admission.reason;
    const OpDescriptor& descriptor = *admission.descriptor;

    const size_t output = register_tensor(assign->var_, false);
    CHECK_SPAN(producer.emplace(assign->var_.get(), graph.ops.size()).second, assign->span_)
        << "AutoTile requires SSA tensor definitions";
    VectorOp op;
    op.stmt = assign;
    op.call = call;
    op.emission_op = call->op_->name_;
    op.kind = descriptor.kind;
    op.primitive = descriptor.primitive;
    op.geometry = descriptor.geometry;
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

}  // namespace

VectorAdmissionResult AdmitVectorGraph(const FunctionPtr& function, const ProgramPtr& program) {
  VectorAdmissionResult result;
  try {
    result.graph = BuildVectorGraphOrThrow(function, program);
    result.supported = true;
  } catch (const pypto::Error& error) {
    result.reason = error.what();
    result.failure = std::current_exception();
  }
  return result;
}

int64_t DTypeBytes(const DataType& dtype) { return static_cast<int64_t>(dtype.GetByte()); }

std::pair<int64_t, int64_t> StaticTensorShape(const TypePtr& type) {
  auto tensor = As<TensorType>(type);
  if (tensor == nullptr || tensor->shape_.size() != 2) return {-1, -1};
  auto rows = As<ConstInt>(tensor->shape_[0]);
  auto cols = As<ConstInt>(tensor->shape_[1]);
  if (rows == nullptr || cols == nullptr) return {-1, -1};
  return {rows->value_, cols->value_};
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
