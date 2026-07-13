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

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/distributed_codegen.h"
#include "pypto/codegen/distributed/distributed_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::CallPtr;
using ir::ConstInt;
using ir::ExprPtr;
using ir::MakeTuple;

namespace {

std::string AllReduceOpSuffix(int reduce_op) {
  CHECK(reduce_op == static_cast<int>(ir::ReduceOp::kSum))
      << "builtin.tensor.allreduce variant mangling currently supports only ReduceOp.Sum, got " << reduce_op;
  return "sum";
}

std::string AllReduceOpCpp(int reduce_op) {
  CHECK(reduce_op == static_cast<int>(ir::ReduceOp::kSum))
      << "builtin.tensor.allreduce currently supports only ReduceOp.Sum, got " << reduce_op;
  return "ReduceOp::kSum";
}

std::string AllReduceDTypeSuffix(const DataType& dtype) {
  CHECK(dtype == DataType::FP32)
      << "builtin.tensor.allreduce variant mangling currently supports only FP32, got " << dtype.ToString();
  return "fp32";
}

std::string AllReduceDTypeCpp(const DataType& dtype) {
  CHECK(dtype == DataType::FP32)
      << "builtin.tensor.allreduce template instantiation currently supports only FP32, got "
      << dtype.ToString();
  return "float";
}

std::string MangleTensorAllReduceVariant(const std::string& op_name, int reduce_op, const DataType& dtype) {
  return op_name + "__" + AllReduceOpSuffix(reduce_op) + "__" + AllReduceDTypeSuffix(dtype);
}

std::string Fp32VariantSuffix(const DataType& dtype) {
  CHECK(dtype == DataType::FP32)
      << "builtin tensor collective variant mangling currently supports only FP32, got " << dtype.ToString();
  return "fp32";
}

std::string Fp32TypeCpp(const DataType& dtype) {
  CHECK(dtype == DataType::FP32)
      << "builtin tensor collective template instantiation currently supports only FP32, got "
      << dtype.ToString();
  return "float";
}

std::string ArgDirectionToTensorArgType(ir::ArgDirection dir) {
  switch (dir) {
    case ir::ArgDirection::Input:
      return "TensorArgType.INPUT";
    case ir::ArgDirection::InOut:
      return "TensorArgType.INOUT";
    case ir::ArgDirection::Output:
      return "TensorArgType.OUTPUT";
    case ir::ArgDirection::OutputExisting:
      return "TensorArgType.OUTPUT_EXISTING";
    default:
      throw pypto::ValueError("unsupported ArgDirection for builtin tensor collective dispatch");
  }
}

void EmitBuiltinWindowCollectiveDispatch(DistributedCodegen& codegen, const CallPtr& call,
                                         const std::string& variant) {
  INTERNAL_CHECK(call && call->op_)
      << "Internal error: builtin tensor collective dispatch needs a valid Call";

  const std::string rank_expr = codegen.ResolveRankExpr(call);
  INTERNAL_CHECK_SPAN(!rank_expr.empty(), call->span_)
      << "Internal error: builtin tensor collective dispatch must carry device= attr";

  const auto arg_directions = call->GetArgDirections();
  INTERNAL_CHECK_SPAN(arg_directions.size() == call->args_.size(), call->span_)
      << "Internal error: builtin tensor collective dispatch arg_directions length mismatch";

  std::optional<std::string> handle_var;
  for (const auto& arg : call->args_) {
    auto dist_type = ir::As<ir::DistributedTensorType>(arg->GetType());
    if (!dist_type || !dist_type->window_buffer_.has_value()) continue;
    const auto& window_buffer = dist_type->window_buffer_.value();
    const std::string arg_handle = codegen.GetCommDomainHandleVar(window_buffer);
    if (!handle_var.has_value()) {
      handle_var = arg_handle;
      continue;
    }
    INTERNAL_CHECK_SPAN(*handle_var == arg_handle, call->span_)
        << "Internal error: builtin tensor collective window args must share the same comm-domain scope";
  }
  INTERNAL_CHECK_SPAN(handle_var.has_value(), call->span_)
      << "Internal error: builtin tensor collective dispatch needs at least one window-bound arg";

  const std::string ta_var = codegen.NextTaskArgsVar();
  const std::string cfg_var = ta_var + "_config";

  codegen.Emit(ta_var + " = TaskArgs()");
  for (size_t i = 0; i < call->args_.size(); ++i) {
    const std::string tag = ArgDirectionToTensorArgType(arg_directions[i]);
    if (auto dist_type = ir::As<ir::DistributedTensorType>(call->args_[i]->GetType())) {
      INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), call->span_)
          << "Internal error: builtin tensor collective window arg must have a WindowBuffer";
      const auto& window_buffer = dist_type->window_buffer_.value();
      const std::string arg_handle = codegen.GetCommDomainHandleVar(window_buffer);
      INTERNAL_CHECK_SPAN(arg_handle == *handle_var, call->span_)
          << "Internal error: builtin tensor collective window args must share the same comm-domain scope";
      const std::string name = codegen.SanitizeName(window_buffer->name_hint_);
      const std::string shape = codegen.FormatShapeTuple(dist_type->shape_);
      const std::string dtype_enum =
          "DataType." + DistributedCodegen::DataTypeToSimplerEnum(dist_type->dtype_);
      codegen.Emit(ta_var + ".add_tensor(Tensor.make(data=" + arg_handle + "[" + rank_expr +
                   "].buffer_ptrs[\"" + name + "\"], shapes=" + shape + ", dtype=" + dtype_enum +
                   ", child_memory=True), " + tag + ")");
      continue;
    }
    if (ir::As<ir::TileType>(call->args_[i]->GetType())) {
      const std::string arg_name = codegen.GetExprAsCode(call->args_[i]);
      INTERNAL_CHECK_SPAN(!arg_name.empty(), call->span_)
          << "Internal error: builtin tensor collective tile arg must resolve to a Python name";
      codegen.Emit(ta_var + ".add_tensor(make_tensor_arg(tensors[\"" + arg_name + "\"]), " + tag + ")");
      continue;
    }
    INTERNAL_CHECK_SPAN(false, call->span_)
        << "Internal error: unsupported builtin tensor collective arg type at index " << i;
  }

  codegen.Emit(ta_var + ".add_scalar(" + *handle_var + "[" + rank_expr + "].domain_size)");
  codegen.Emit(ta_var + ".add_scalar(" + *handle_var + "[" + rank_expr + "].device_ctx)");
  codegen.Emit(cfg_var + " = CallConfig()");
  codegen.Emit(cfg_var + ".block_dim = 1");
  codegen.Emit(cfg_var + ".aicpu_thread_num = config.aicpu_thread_num");
  codegen.Emit("_keep.append(" + ta_var + ")");
  codegen.Emit("orch.submit_next_level(callables[\"" + variant + "\"], " + ta_var + ", " + cfg_var +
               ", worker=" + rank_expr + ")");
}

}  // namespace

// ============================================================================
// pld.tensor.alloc_window_buffer — host-side marker, no runtime emission.
//
// The alloc op is consumed at compile time by ``MaterializeCommDomainScopes``; the
// resulting ``WindowBuffer`` metadata is emitted by ``EmitCommDomainAllocations``
// as part of the ``orch.allocate_domain(buffers=[CommBufferSpec(...), ...])``
// spec list wrapping the host_orch body. The host_orch.py module never needs
// to reach for the IR-level alloc op again — chip dispatch reads the device
// pointer from ``__comm_d0[r].buffer_ptrs["<name>"]`` instead. Returning
// empty signals the surrounding ``AssignStmt`` visitor to drop the line.
// ============================================================================
REGISTER_DISTRIBUTED_OP(pld_tensor_alloc_window_buffer, "pld.tensor.alloc_window_buffer") {
  (void)op;
  (void)codegen;
  return "";
}

// ============================================================================
// pld.tensor.window — host-side marker, no runtime emission.
//
// ``pld.tensor.window`` materialises a window-bound view at IR construction
// time; ``MaterializeCommDomainScopes`` rewires every dispatch site so the per-rank
// device pointer is read from ``__comm_d0[r].buffer_ptrs["<name>"]`` at
// chip-arg emission time. The host_orch.py module never calls back into
// the IR window op.
// ============================================================================
REGISTER_DISTRIBUTED_OP(pld_tensor_window, "pld.tensor.window") {
  (void)op;
  (void)codegen;
  return "";
}

// ============================================================================
// builtin.tensor.allreduce: compiler-generated host collective chip dispatch.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_allreduce, "builtin.tensor.allreduce") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.allreduce codegen requires DistributedCodegen";
  const int reduce_op = op->GetAttr<int>("op");
  const auto dtype = op->GetAttr<DataType>("dtype");
  const std::string variant = MangleTensorAllReduceVariant(op->op_->name_, reduce_op, dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(
        op, variant, {{"op_cpp", AllReduceOpCpp(reduce_op)}, {"dtype_cpp", AllReduceDTypeCpp(dtype)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// builtin.tensor.barrier: compiler-generated host collective chip dispatch.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_barrier, "builtin.tensor.barrier") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.barrier codegen requires DistributedCodegen";
  const std::string variant = op->op_->name_ + "__" + Fp32VariantSuffix(DataType::FP32);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(op, variant, {{"dtype_cpp", Fp32TypeCpp(DataType::FP32)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// builtin.tensor.broadcast: compiler-generated host collective chip dispatch.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_broadcast, "builtin.tensor.broadcast") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.broadcast codegen requires DistributedCodegen";
  const int root = op->GetAttr<int>("root");
  const auto dtype = op->GetAttr<DataType>("dtype");
  const std::string variant =
      op->op_->name_ + "__root" + std::to_string(root) + "__" + Fp32VariantSuffix(dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(
        op, variant, {{"root_cpp", std::to_string(root)}, {"dtype_cpp", Fp32TypeCpp(dtype)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// builtin.tensor.reduce_scatter: compiler-generated host collective chip dispatch.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_reduce_scatter, "builtin.tensor.reduce_scatter") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.reduce_scatter codegen requires DistributedCodegen";
  const int reduce_op = op->GetAttr<int>("op");
  const auto dtype = op->GetAttr<DataType>("dtype");
  const std::string variant =
      op->op_->name_ + "__" + AllReduceOpSuffix(reduce_op) + "__" + Fp32VariantSuffix(dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(
        op, variant, {{"op_cpp", AllReduceOpCpp(reduce_op)}, {"dtype_cpp", Fp32TypeCpp(dtype)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// builtin.tensor.allgather: compiler-generated chip dispatch for pld.tensor.allgather.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_allgather, "builtin.tensor.allgather") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.allgather codegen requires DistributedCodegen";
  const auto dtype = op->GetAttr<DataType>("dtype");
  const std::string variant = op->op_->name_ + "__" + Fp32VariantSuffix(dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(op, variant, {{"dtype_cpp", Fp32TypeCpp(dtype)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// builtin.tensor.all_to_all: compiler-generated chip dispatch for pld.tensor.all_to_all.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_all_to_all, "builtin.tensor.all_to_all") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.all_to_all codegen requires DistributedCodegen";
  const auto dtype = op->GetAttr<DataType>("dtype");
  const std::string variant = op->op_->name_ + "__" + Fp32VariantSuffix(dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(op, variant, {{"dtype_cpp", Fp32TypeCpp(dtype)}});
  }
  EmitBuiltinWindowCollectiveDispatch(*dist_codegen, op, variant);
  return "";
}

// ============================================================================
// tensor.slice — emit Python tensor indexing into ``tensors[...]``.
//
// IR form:
//   t = tensor.slice(input, shape, offset, valid_shape, drop_dims)
//
// where shape / offset / valid_shape / drop_dims are MakeTuples. valid_shape
// is purely IR metadata and is ignored at the host layer. For each axis:
//   * axis ∈ drop_dims → scalar Python index ``offset[axis]`` (rank
//     reduction, must be a unit dim)
//   * otherwise        → slice ``offset[axis] : offset[axis] + shape[axis]``
//
// The result is registered into the ``tensors`` dict so downstream
// dispatch sites can ``chip_args.add_tensor(make_tensor_arg(tensors["t"]), ...)``
// without an extra binding step.
// ============================================================================
REGISTER_DISTRIBUTED_OP(tensor_slice, "tensor.slice") {
  auto& dist_codegen = dynamic_cast<DistributedCodegen&>(codegen);

  CHECK(op->args_.size() == 3 || op->args_.size() == 4 || op->args_.size() == 5)
      << "tensor.slice host_orch codegen expects 3-5 args (input, shape, offset[, valid_shape[, "
         "drop_dims]]), "
         "got "
      << op->args_.size();

  const std::string input_name = codegen.GetExprAsCode(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.slice input must resolve to a non-empty Python name";

  const std::string lhs = codegen.GetCurrentResultTarget();
  CHECK(!lhs.empty()) << "tensor.slice in host_orch must have an assignment target";

  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "tensor.slice shape must be MakeTuple";
  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tensor.slice offset must be MakeTuple";
  CHECK(offset_tuple->elements_.size() == shape_tuple->elements_.size())
      << "tensor.slice offset/shape rank mismatch";

  std::set<int64_t> drop_dims;
  if (op->args_.size() == 5) {
    auto dd_tuple = As<MakeTuple>(op->args_[4]);
    INTERNAL_CHECK_SPAN(dd_tuple, op->span_) << "tensor.slice drop_dims must be MakeTuple";
    for (const auto& e : dd_tuple->elements_) {
      auto ci = As<ConstInt>(e);
      CHECK(ci) << "tensor.slice drop_dims entries must be ConstInt";
      drop_dims.insert(ci->value_);
    }
  }

  std::ostringstream indices;
  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    if (i > 0) indices << ", ";
    const std::string offset_i = codegen.GetExprAsCode(offset_tuple->elements_[i]);
    if (drop_dims.count(static_cast<int64_t>(i)) > 0) {
      // Drop this axis via scalar indexing — torch reduces the rank by 1.
      indices << offset_i;
    } else {
      const std::string shape_i = codegen.GetExprAsCode(shape_tuple->elements_[i]);
      // Constant-fold ``0 + shape_i`` for readability when offset is 0.
      auto offset_const = As<ConstInt>(offset_tuple->elements_[i]);
      if (offset_const && offset_const->value_ == 0) {
        indices << "0:" << shape_i;
      } else {
        indices << offset_i << ":" << offset_i << " + " << shape_i;
      }
    }
  }

  std::ostringstream line;
  line << "tensors[\"" << lhs << "\"] = tensors[\"" << input_name << "\"][" << indices.str() << "]";
  codegen.Emit(line.str());
  dist_codegen.MarkDeclared(lhs);
  return "";
}

}  // namespace codegen
}  // namespace pypto
