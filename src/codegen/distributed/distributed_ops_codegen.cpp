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
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
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
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::CallPtr;
using ir::ConstInt;
using ir::ExprPtr;
using ir::MakeTuple;

namespace {

struct AllReduceOpVariant {
  const char* suffix;
  const char* cpp;
  const char* instruction;
};

AllReduceOpVariant GetAllReduceOpVariant(int reduce_op) {
  switch (static_cast<ir::ReduceOp>(reduce_op)) {
    case ir::ReduceOp::kSum:
      return {"sum", "ReduceOp::kSum", "TADD"};
    case ir::ReduceOp::kMax:
      return {"max", "ReduceOp::kMax", "TMAX"};
    case ir::ReduceOp::kMin:
      return {"min", "ReduceOp::kMin", "TMIN"};
    case ir::ReduceOp::kProd:
      return {"prod", "ReduceOp::kProd", "TMUL"};
    default:
      throw pypto::ValueError("unsupported builtin.tensor.allreduce ReduceOp " + std::to_string(reduce_op));
  }
}

struct AllReduceDTypeVariant {
  const char* suffix;
  const char* cpp;
};

AllReduceDTypeVariant GetAllReduceDTypeVariant(const DataType& dtype) {
  if (dtype == DataType::FP16) return {"fp16", "half"};
  if (dtype == DataType::FP32) return {"fp32", "float"};
  throw pypto::ValueError("unsupported builtin.tensor.allreduce dtype " + dtype.ToString());
}

std::string ReduceScatterOpSuffix(int reduce_op) {
  CHECK(reduce_op == static_cast<int>(ir::ReduceOp::kSum))
      << "builtin.tensor.reduce_scatter variant mangling currently supports only ReduceOp.Sum, got "
      << reduce_op;
  return "sum";
}

std::string ReduceScatterOpCpp(int reduce_op) {
  CHECK(reduce_op == static_cast<int>(ir::ReduceOp::kSum))
      << "builtin.tensor.reduce_scatter currently supports only ReduceOp.Sum, got " << reduce_op;
  return "ReduceOp::kSum";
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
                                         const std::string& variant,
                                         std::optional<int> core_num = std::nullopt) {
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
      codegen.Emit(ta_var + ".add_tensor(" + arg_handle + "[" + rank_expr + "].buffers[\"" + name +
                   "\"].tensor(shapes=" + shape + ", dtype=" + dtype_enum + "), " + tag + ")");
      continue;
    }
    if (ir::As<ir::TileType>(call->args_[i]->GetType())) {
      const std::string arg_name = codegen.GetExprAsCode(call->args_[i]);
      INTERNAL_CHECK_SPAN(!arg_name.empty(), call->span_)
          << "Internal error: builtin tensor collective tile arg must resolve to a Python name";
      codegen.Emit(ta_var + ".add_tensor(make_tensor_arg(orch._worker, tensors[\"" + arg_name + "\"]), " +
                   tag + ")");
      continue;
    }
    INTERNAL_CHECK_SPAN(false, call->span_)
        << "Internal error: unsupported builtin tensor collective arg type at index " << i;
  }

  codegen.Emit(ta_var + ".add_scalar(" + *handle_var + "[" + rank_expr + "].domain_size)");
  codegen.Emit(ta_var + ".add_scalar(" + *handle_var + "[" + rank_expr + "].device_ctx)");
  if (core_num.has_value()) {
    codegen.Emit(ta_var + ".add_scalar(" + std::to_string(*core_num) + ")");
  }
  codegen.Emit(cfg_var + " = CallConfig()");
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
// wire view from ``__comm_d0[r].buffers["<name>"].tensor(...)`` instead. Returning
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
// wire view is derived from ``__comm_d0[r].buffers["<name>"]`` at
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

/// Emit the chip dispatch shared by the mesh and ring host AllReduce builtins.
///
/// ``multicore`` is set only for the mesh builtin, which launches a synchronized
/// SPMD AIV grid of ``core_num`` blocks. The ring builtin stays single-block, so
/// it neither carries a ``core_num`` attr nor renders a launch-spec method.
/// ``core_num`` itself is validated by ``LowerHostTensorCollectives`` — including
/// the backend capacity bound — so codegen only forwards it.
std::string EmitAllReduceLikeDispatch(DistributedCodegen& dist_codegen, const CallPtr& op,
                                      bool include_reduce_inst, bool multicore) {
  const int reduce_op = op->GetAttr<int>("op");
  const auto dtype = op->GetAttr<DataType>("dtype");
  const auto reduce_variant = GetAllReduceOpVariant(reduce_op);
  const auto dtype_variant = GetAllReduceDTypeVariant(dtype);
  const std::string variant = op->op_->name_ + "__" + reduce_variant.suffix + "__" + dtype_variant.suffix;

  std::optional<int> core_num;
  if (multicore) {
    const int requested = op->GetAttr<int>("core_num");
    INTERNAL_CHECK_SPAN(requested > 0, op->span_)
        << "Internal error: builtin.tensor.allreduce core_num must be positive, got " << requested
        << "; LowerHostTensorCollectives should have rejected it";
    core_num = requested;
  }

  if (dist_codegen.MarkBuiltinEmitted(variant)) {
    std::map<std::string, std::string> template_vars{{"op_cpp", reduce_variant.cpp},
                                                     {"dtype_cpp", dtype_variant.cpp}};
    if (include_reduce_inst) {
      template_vars["reduce_inst"] = reduce_variant.instruction;
    }
    if (multicore) {
      template_vars["launch_core_count_method"] =
          pypto::backend::GetBackend()->GetHandler()->GetLaunchSpecCoreCountMethod();
    }
    dist_codegen.RecordBuiltinNextLevel(op, variant, std::move(template_vars));
  }
  EmitBuiltinWindowCollectiveDispatch(dist_codegen, op, variant, core_num);
  return "";
}

REGISTER_DISTRIBUTED_OP(builtin_tensor_allreduce, "builtin.tensor.allreduce") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.allreduce codegen requires DistributedCodegen";
  return EmitAllReduceLikeDispatch(*dist_codegen, op, /*include_reduce_inst=*/true,
                                   /*multicore=*/true);
}

// ============================================================================
// builtin.tensor.allreduce_ring: host ring allreduce chip dispatch.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_allreduce_ring, "builtin.tensor.allreduce_ring") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.allreduce_ring codegen requires DistributedCodegen";
  return EmitAllReduceLikeDispatch(*dist_codegen, op, /*include_reduce_inst=*/false,
                                   /*multicore=*/false);
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
      op->op_->name_ + "__" + ReduceScatterOpSuffix(reduce_op) + "__" + Fp32VariantSuffix(dtype);

  if (dist_codegen->MarkBuiltinEmitted(variant)) {
    dist_codegen->RecordBuiltinNextLevel(
        op, variant, {{"op_cpp", ReduceScatterOpCpp(reduce_op)}, {"dtype_cpp", Fp32TypeCpp(dtype)}});
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
// builtin.tensor.all_to_all_v: compiler-generated chip dispatch for pld.tensor.all_to_all_v.
// ============================================================================
REGISTER_DISTRIBUTED_OP(builtin_tensor_all_to_all_v, "builtin.tensor.all_to_all_v") {
  auto* dist_codegen = dynamic_cast<DistributedCodegen*>(&codegen);
  INTERNAL_CHECK(dist_codegen) << "builtin.tensor.all_to_all_v codegen requires DistributedCodegen";
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
// dispatch sites can ``chip_args.add_tensor(make_tensor_arg(orch._worker, tensors["t"]), ...)``
// without an extra binding step.
// ============================================================================
REGISTER_DISTRIBUTED_OP(tensor_slice, "tensor.slice") {
  auto& dist_codegen = dynamic_cast<DistributedCodegen&>(codegen);

  INTERNAL_CHECK_SPAN(op->args_.size() == 3 || op->args_.size() == 4 || op->args_.size() == 5, op->span_)
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

// ============================================================================
// tensor.assemble — write a source tensor into a target tensor slice.
//
// IR form:
//   result = tensor.assemble(target, source, offset, *, atomic)
//
// The operation is SSA-functional but writes in place: ``result`` aliases
// ``target`` after the source's valid region has been copied into the target
// window. HOST orchestrators have no atomic-combine instruction, so atomic add
// is rejected instead of being silently reduced to a plain Python assignment.
// ============================================================================
REGISTER_DISTRIBUTED_OP(tensor_assemble, "tensor.assemble") {
  auto& dist_codegen = dynamic_cast<DistributedCodegen&>(codegen);

  INTERNAL_CHECK_SPAN(op->args_.size() == 3, op->span_)
      << "Internal error: tensor.assemble expects 3 arguments";

  const int atomic = op->GetKwarg<int>("atomic", static_cast<int>(ir::AtomicType::kNone));
  CHECK_SPAN(atomic == static_cast<int>(ir::AtomicType::kNone), op->span_)
      << "pl.assemble(..., atomic=pl.AtomicType.Add) is only supported inside an InCore function. "
         "This assemble is in a HOST orchestrator, where no atomic-combine instruction exists. Move it "
         "inside the scope that produces the partial result.";

  const std::string target_name = codegen.GetExprAsCode(op->args_[0]);
  const std::string source_name = codegen.GetExprAsCode(op->args_[1]);
  const std::string lhs = codegen.GetCurrentResultTarget();
  CHECK(!target_name.empty()) << "tensor.assemble target must resolve to a non-empty Python name";
  CHECK(!source_name.empty()) << "tensor.assemble source must resolve to a non-empty Python name";
  CHECK(!lhs.empty()) << "tensor.assemble in a HOST orchestrator must have an assignment target";

  auto target_type = ir::As<ir::TensorType>(op->args_[0]->GetType());
  auto source_type = ir::As<ir::TensorType>(op->args_[1]->GetType());
  CHECK_SPAN(target_type && source_type, op->span_)
      << "tensor.assemble in a HOST orchestrator currently supports plain Tensor operands only";

  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "Internal error: tensor.assemble offset must be MakeTuple";
  const size_t target_rank = target_type->shape_.size();
  const size_t source_rank = source_type->shape_.size();
  CHECK_SPAN(offset_tuple->elements_.size() == target_rank, op->span_)
      << "tensor.assemble in a HOST orchestrator requires offset rank to match target rank";
  CHECK_SPAN(source_rank <= target_rank, op->span_)
      << "tensor.assemble in a HOST orchestrator requires source rank not to exceed target rank";

  const std::vector<ExprPtr> source_valid_shape = ir::GetValidShape(source_type);
  const size_t leading_target_rank = target_rank - source_rank;
  std::ostringstream target_indices;
  std::ostringstream source_indices;
  for (size_t i = 0; i < target_rank; ++i) {
    if (i > 0) {
      target_indices << ", ";
    }
    const std::string offset_i = codegen.GetExprAsCode(offset_tuple->elements_[i]);
    const std::string extent_i =
        i < leading_target_rank ? "1" : codegen.GetExprAsCode(source_valid_shape[i - leading_target_rank]);
    auto offset_const = As<ConstInt>(offset_tuple->elements_[i]);
    if (offset_const && offset_const->value_ == 0) {
      target_indices << "0:" << extent_i;
    } else {
      target_indices << offset_i << ":" << offset_i << " + " << extent_i;
    }
  }
  for (size_t i = 0; i < source_rank; ++i) {
    if (i > 0) {
      source_indices << ", ";
    }
    source_indices << "0:" << codegen.GetExprAsCode(source_valid_shape[i]);
  }

  std::string source_expr = "tensors[\"" + source_name + "\"]";
  if (source_rank > 0) {
    source_expr += "[" + source_indices.str() + "]";
  }
  codegen.Emit("tensors[\"" + target_name + "\"][" + target_indices.str() + "] = " + source_expr);
  if (lhs != target_name) {
    codegen.Emit("tensors[\"" + lhs + "\"] = tensors[\"" + target_name + "\"]");
  }
  dist_codegen.MarkDeclared(lhs);
  return "";
}

}  // namespace codegen
}  // namespace pypto
