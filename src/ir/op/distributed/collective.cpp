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

/**
 * @file collective.cpp
 * @brief Distributed tensor-level collective ops — pld.tensor.* composites and builtin.tensor.* host
 * dispatches.
 *
 * Composite collective ops that lower through LowerCompositeOps (pass 14)
 * into notify/wait/remote_load/store primitives (InCore path), or through
 * LowerHostTensorCollectives into builtin.tensor.* chip dispatches (HOST path).
 * Each op registers a type deducer and an op description; the actual IR
 * expansion lives in the respective lowering pass.
 *
 *   - pld.tensor.barrier(signal)                                  -> DistributedTensorType
 *   - pld.tensor.broadcast(target, signal, root)                   -> DistributedTensorType
 *   - pld.tensor.allgather(target, signal)          (2-arg HOST)   -> DistributedTensorType
 *   - pld.tensor.allgather(local_data, target, signal, out) (4-arg)-> TensorType
 *   - pld.tensor.reduce_scatter(target, signal, op)                -> DistributedTensorType
 *
 * The five builtin.tensor.* ops are internal chip-dispatch targets emitted by the
 * host-orchestrator lowering pass (LowerHostTensorCollectives):
 * builtin.tensor.{allreduce,barrier,broadcast,reduce_scatter,allgather}.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void CheckReduceOp(int op_value, const std::string& op_name) {
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << op_name << " op must be ReduceOp.Sum (got int " << op_value << ")";
}

void CheckSupportedBuiltinVariant(int op_value, DataType dtype, const std::string& op_name) {
  CheckReduceOp(op_value, op_name);
  CHECK(dtype == DataType::FP32) << op_name << " currently supports only (op=ReduceOp.Sum, dtype=FP32); got "
                                 << "(op=ReduceOp.Sum, dtype=" << dtype.ToString() << ")";
}

void CheckSupportedFp32BuiltinVariant(DataType dtype, const std::string& op_name) {
  CHECK(dtype == DataType::FP32) << op_name << " currently supports only dtype=FP32; got "
                                 << dtype.ToString();
}

void CheckSignalDistributedTensor(const DistributedTensorTypePtr& signal_type, const std::string& op_name) {
  CHECK(signal_type) << op_name << " signal must be a DistributedTensor";
  CHECK(signal_type->dtype_ == DataType::INT32)
      << op_name << " signal dtype must be INT32, got " << signal_type->dtype_.ToString();
  CHECK(signal_type->shape_.size() == 1)
      << op_name << " signal must be a rank-1 DistributedTensor, got rank " << signal_type->shape_.size();
}

// ============================================================================
// builtin.tensor.allreduce
// ============================================================================

TypePtr DeduceBuiltinTensorAllReduceType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.allreduce";
  CHECK(args.size() == 2) << kOpName << " requires exactly 2 positional arguments (src, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }

  auto src_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(src_type) << kOpName << " src must be a DistributedTensor, got " << args[0]->GetType()->TypeName();
  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << kOpName << " signal must be a DistributedTensor, got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << kOpName << " signal dtype must be INT32, got " << signal_type->dtype_.ToString();
  CHECK(signal_type->shape_.size() == 1 || signal_type->shape_.size() == 2)
      << kOpName << " signal must be rank-1 [world_size] or rank-2 [world_size, 1], got rank "
      << signal_type->shape_.size();
  if (signal_type->shape_.size() == 2) {
    auto second_extent = As<ConstInt>(signal_type->shape_[1]);
    CHECK(second_extent) << kOpName << " rank-2 signal shape[1] must be the constant 1";
    CHECK(second_extent->value_ == 1)
        << kOpName << " rank-2 signal shape[1] must be 1, got " << second_extent->value_;
  }

  auto op_value = GetRequiredKwarg<int>(kwargs, "op", kOpName);
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == src_type->dtype_) << kOpName << " dtype kwarg (" << dtype.ToString()
                                   << ") must match src dtype (" << src_type->dtype_.ToString() << ")";
  CheckSupportedBuiltinVariant(op_value, dtype, kOpName);
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.allreduce")
    .set_op_category("DistributedOp")
    .set_description("Internal chip-dispatch builtin for pld.tensor.allreduce.")
    .add_argument("src", "Window-bound DistributedTensor to reduce in place")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<int>("op")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.allreduce")
    .f_deduce_type(DeduceBuiltinTensorAllReduceType);

// ============================================================================
// pld.tensor.barrier — cross-rank barrier (notify-all + wait-all)
// ============================================================================

namespace {

TypePtr DeduceTensorBarrierType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 1) << "pld.tensor.barrier requires exactly 1 positional argument (signal), but got "
                          << args.size();
  CHECK(args[0]) << "pld.tensor.barrier positional argument #0 must not be null";

  auto signal_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(signal_type) << "pld.tensor.barrier signal must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.barrier signal must have INT32 element type (the barrier slot is an int counter), "
         "got dtype "
      << signal_type->dtype_.ToString();

  // Return signal's type — the rebind idiom lets users write
  // ``sig = pld.tensor.barrier(sig)``, matching allreduce.
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("pld.tensor.barrier")
    .set_description(
        "`signal` is a window-bound INT32 matrix used as the cross-rank synchronisation (one slot "
        "per rank). InCore path: lowered to notify-all/wait-all by LowerCompositeOps. "
        "HOST builtin path: lowered to builtin.tensor.barrier per chip by LowerHostTensorCollectives.")
    .set_op_category("DistributedOp")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorBarrierType);

// ============================================================================
// pld.tensor.broadcast — broadcast root rank's data to all ranks
// ============================================================================

namespace {

TypePtr DeduceTensorBroadcastType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "pld.tensor.broadcast requires exactly 2 positional arguments "
                             "(target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.broadcast positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.broadcast target must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();

  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "pld.tensor.broadcast signal must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.broadcast signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Validate root kwarg.
  auto root_value = GetRequiredKwarg<int>(kwargs, "root", "pld.tensor.broadcast");
  CHECK(root_value >= 0) << "pld.tensor.broadcast root rank must be non-negative, got " << root_value;

  // Result type: same as target (in-place rebind — every rank's slot now
  // holds root's data).
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("pld.tensor.broadcast")
    .set_description(
        "Broadcast: replicate root rank's window-bound data to every rank in the comm group. "
        "`target` is a window-bound DistributedTensor (each rank writes its own data before the "
        "call; root's data is read and replicated by all non-root ranks). `signal` is a "
        "window-bound INT32 matrix used as the cross-rank barrier. `root` (int kwarg) selects "
        "the source rank. InCore path: lowered to notify-all/wait-all + remote_load from root "
        "by LowerCompositeOps. HOST builtin path: lowered to builtin.tensor.broadcast per chip "
        "by LowerHostTensorCollectives.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("root")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorBroadcastType);

// ============================================================================
// pld.tensor.allgather — gather data from all ranks (2-arg HOST builtin or 4-arg InCore composite)
// ============================================================================

namespace {

TypePtr DeduceTensorAllGatherType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 2 || args.size() == 4)
      << "pld.tensor.allgather requires 2 args (target, signal) for host builtin path "
         "or 4 args (local_data, target, signal, out) for InCore composite, but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allgather positional argument #" << i << " must not be null";
  }

  if (args.size() == 2) {
    // 2-arg HOST builtin path: allgather(target, signal) — data pre-staged in window.
    auto target_type = As<DistributedTensorType>(args[0]->GetType());
    CHECK(target_type) << "pld.tensor.allgather target must be a DistributedTensor (window-bound), got "
                       << args[0]->GetType()->TypeName();
    CHECK(target_type->shape_.size() == 2)
        << "pld.tensor.allgather target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
    auto signal_type = As<DistributedTensorType>(args[1]->GetType());
    CHECK(signal_type) << "pld.tensor.allgather signal must be a DistributedTensor (window-bound), got "
                       << args[1]->GetType()->TypeName();
    CHECK(signal_type->dtype_ == DataType::INT32)
        << "pld.tensor.allgather signal must have INT32 element type, got dtype "
        << signal_type->dtype_.ToString();
    return args[0]->GetType();
  }

  // 4-arg InCore composite path.
  // arg 0: local_data — Tile (or Tensor before ConvertTensorToTileOps) with this rank's chunk
  auto local_type = args[0]->GetType();
  CHECK(As<TileType>(local_type) || As<TensorType>(local_type))
      << "pld.tensor.allgather local_data must be a Tile or Tensor, got " << local_type->TypeName();

  // arg 1: target — DistributedTensor [NR, SIZE] staging window
  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << "pld.tensor.allgather target must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << "pld.tensor.allgather target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";

  // arg 2: signal — DistributedTensor INT32
  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << "pld.tensor.allgather signal must be a DistributedTensor (window-bound), got "
                     << args[2]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.allgather signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // arg 3: out — Tensor [1, NR*SIZE] where the gathered result is written
  auto out_type = As<TensorType>(args[3]->GetType());
  CHECK(out_type) << "pld.tensor.allgather out must be a Tensor (not a DistributedTensor), got "
                  << args[3]->GetType()->TypeName();
  CHECK(out_type->shape_.size() == 2)
      << "pld.tensor.allgather out must be 2D [1, NR*SIZE], got " << out_type->shape_.size() << " dims";

  // Return the output Tensor type — the intrinsic writes directly into it.
  return out_type;
}

}  // namespace

REGISTER_OP("pld.tensor.allgather")
    .set_description(
        "All-gather: gather data from all ranks.  Two forms are supported: "
        "(2-arg) `pld.tensor.allgather(target, signal)` for HOST builtin — "
        "each rank's chunk is pre-staged in the window-bound target, lowered "
        "to builtin.tensor.barrier per chip; "
        "(4-arg) `pld.tensor.allgather(local_data, target, signal, out)` for "
        "InCore composite — `local_data` is the rank's chunk (Tile [1, SIZE]), "
        "`target` is a window-bound DistributedTensor[NR, SIZE] staging area, "
        "`signal` is a window-bound INT32 barrier tensor, `out` is a plain "
        "Tensor[1, NR*SIZE] receiving the rank-ordered concatenation. "
        "The 4-arg form is lowered by LowerCompositeOps into tile.store + "
        "notify-all/wait-all + per-peer remote_load + tile.store into out; "
        "this Call never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("local_data", "Local tile [1, SIZE] — this rank's data (Input)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .add_argument("out", "Plain Tensor[1, NR*SIZE] — receives the gathered result (Output)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllGatherType);

// ============================================================================
// pld.tensor.all_to_all — symmetric all-to-all (4-arg InCore composite)
// ============================================================================

namespace {

TypePtr DeduceTensorAllToAllType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 4)
      << "pld.tensor.all_to_all requires 4 args (input, target, signal, out) for InCore composite, but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.all_to_all positional argument #" << i << " must not be null";
  }

  // 4-arg InCore composite path
  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "pld.tensor.all_to_all input must be a Tensor, got " << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2)
      << "pld.tensor.all_to_all input must be 2D [NR, SIZE], got " << input_type->shape_.size() << " dims";

  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << "pld.tensor.all_to_all target must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << "pld.tensor.all_to_all target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  CHECK(target_type->shape_.size() == input_type->shape_.size())
      << "pld.tensor.all_to_all target rank must match input rank, got " << target_type->shape_.size()
      << " vs " << input_type->shape_.size();
  CHECK(AreExprsEqual(target_type->shape_[0], input_type->shape_[0]) &&
        AreExprsEqual(target_type->shape_[1], input_type->shape_[1]))
      << "pld.tensor.all_to_all target shape must equal input shape";
  CHECK(target_type->dtype_ == input_type->dtype_)
      << "pld.tensor.all_to_all target dtype " << target_type->dtype_.ToString() << " must match input dtype "
      << input_type->dtype_.ToString();

  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << "pld.tensor.all_to_all signal must be a DistributedTensor (window-bound), got "
                     << args[2]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.all_to_all signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();
  CHECK(signal_type->shape_.size() == 2)
      << "pld.tensor.all_to_all signal must be 2D [NR, 1], got " << signal_type->shape_.size() << " dims";
  {
    auto signal_dim1 = As<ConstInt>(signal_type->shape_[1]);
    CHECK(signal_dim1 && signal_dim1->value_ == 1)
        << "pld.tensor.all_to_all signal second dimension must be 1, got "
        << (signal_dim1 ? std::to_string(signal_dim1->value_) : "<dynamic>");
  }

  auto out_type = As<TensorType>(args[3]->GetType());
  CHECK(out_type) << "pld.tensor.all_to_all out must be a Tensor (not a DistributedTensor), got "
                  << args[3]->GetType()->TypeName();
  CHECK(out_type->shape_.size() == 2)
      << "pld.tensor.all_to_all out must be 2D [NR, SIZE], got " << out_type->shape_.size() << " dims";
  CHECK(out_type->shape_.size() == input_type->shape_.size())
      << "pld.tensor.all_to_all out rank must match input rank, got " << out_type->shape_.size() << " vs "
      << input_type->shape_.size();
  CHECK(AreExprsEqual(out_type->shape_[0], input_type->shape_[0]) &&
        AreExprsEqual(out_type->shape_[1], input_type->shape_[1]))
      << "pld.tensor.all_to_all out shape must equal input shape";
  CHECK(out_type->dtype_ == input_type->dtype_)
      << "pld.tensor.all_to_all out dtype " << out_type->dtype_.ToString() << " must match input dtype "
      << input_type->dtype_.ToString();

  return out_type;
}

}  // namespace

REGISTER_OP("pld.tensor.all_to_all")
    .set_description(
        "All-to-all: symmetric personalized exchange.  Every rank sends a distinct "
        "chunk to every other rank.  ``input`` is a Tensor [NR, SIZE] where "
        "``input[dest, :]`` is the chunk destined for rank ``dest``.  ``target`` is "
        "a window-bound DistributedTensor [NR, SIZE] staging area.  ``signal`` is a "
        "window-bound INT32 barrier tensor.  ``out`` is a plain Tensor [NR, SIZE] "
        "where ``out[src, :]`` holds the chunk received from rank ``src``.  "
        "Lowered by LowerCompositeOps into a 3-phase mesh decomposition "
        "(stage-in → barrier → peer-read via pld.tile.get).")
    .set_op_category("DistributedOp")
    .add_argument("input", "Plain Tensor [NR, SIZE] with per-destination chunks (Input)")
    .add_argument("target", "Window-bound DistributedTensor [NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .add_argument("out", "Plain Tensor [NR, SIZE] — receives the exchange result (Output)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllToAllType);

// ============================================================================
// pld.tensor.reduce_scatter — reduce + scatter chunks across ranks
// ============================================================================

namespace {

TypePtr DeduceTensorReduceScatterType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "pld.tensor.reduce_scatter requires exactly 2 positional arguments "
                             "(target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.reduce_scatter positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.reduce_scatter target must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2) << "pld.tensor.reduce_scatter target must be 2D [NR, SIZE], got "
                                         << target_type->shape_.size() << " dims";

  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "pld.tensor.reduce_scatter signal must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.reduce_scatter signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Validate op kwarg — kSum only for first version (same as allreduce).
  auto op_value = GetRequiredKwarg<int>(kwargs, "op", "pld.tensor.reduce_scatter");
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << "pld.tensor.reduce_scatter op must be ReduceOp.Sum (got int " << op_value
      << "); Max / Min / Prod lowerings are not yet implemented";

  // Result type: same as target (in-place rebind — rank r's row now holds
  // the reduced chunk r).
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("pld.tensor.reduce_scatter")
    .set_description(
        "Reduce-scatter: element-wise reduce chunks across all ranks, then scatter so each "
        "rank receives one reduced chunk. `target` has shape [NR, SIZE] — each rank stages "
        "all NR chunks before the call. After the call, rank r's row [r, 0:SIZE] holds the "
        "reduced value of chunk r. `signal` is a window-bound INT32 matrix for the cross-rank "
        "barrier. `op` selects the reduction operator (Sum only in first version). "
        "InCore path: lowered to a 5-phase decomposition by LowerCompositeOps. "
        "HOST builtin path: lowered to builtin.tensor.reduce_scatter per chip by "
        "LowerHostTensorCollectives.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("op")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorReduceScatterType);

// ============================================================================
// builtin.tensor.barrier — host dispatch for pld.tensor.barrier
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorBarrierType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  constexpr const char* kOpName = "builtin.tensor.barrier";
  CHECK(args.size() == 1) << kOpName << " requires exactly 1 positional argument (signal), but got "
                          << args.size();
  CHECK(args[0]) << kOpName << " positional argument #0 must not be null";
  CheckSignalDistributedTensor(As<DistributedTensorType>(args[0]->GetType()), kOpName);
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.barrier")
    .set_description("Internal chip-dispatch builtin for pld.tensor.barrier.")
    .set_op_category("DistributedOp")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.barrier")
    .f_deduce_type(DeduceBuiltinTensorBarrierType);

// ============================================================================
// builtin.tensor.broadcast — host dispatch for pld.tensor.broadcast
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorBroadcastType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.broadcast";
  CHECK(args.size() == 2) << kOpName << " requires exactly 2 positional arguments (target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }
  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << kOpName << " target must be a DistributedTensor, got "
                     << args[0]->GetType()->TypeName();
  CheckSignalDistributedTensor(As<DistributedTensorType>(args[1]->GetType()), kOpName);
  auto root_value = GetRequiredKwarg<int>(kwargs, "root", kOpName);
  CHECK(root_value >= 0) << kOpName << " root rank must be non-negative, got " << root_value;
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == target_type->dtype_)
      << kOpName << " dtype kwarg (" << dtype.ToString() << ") must match target dtype ("
      << target_type->dtype_.ToString() << ")";
  CheckSupportedFp32BuiltinVariant(dtype, kOpName);
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.broadcast")
    .set_description("Internal chip-dispatch builtin for pld.tensor.broadcast.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor to broadcast in place")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<int>("root")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.broadcast")
    .f_deduce_type(DeduceBuiltinTensorBroadcastType);

// ============================================================================
// builtin.tensor.reduce_scatter — host dispatch for pld.tensor.reduce_scatter
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorReduceScatterType(const std::vector<ExprPtr>& args,
                                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.reduce_scatter";
  CHECK(args.size() == 2) << kOpName << " requires exactly 2 positional arguments (target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }
  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << kOpName << " target must be a DistributedTensor, got "
                     << args[0]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << kOpName << " target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  CheckSignalDistributedTensor(As<DistributedTensorType>(args[1]->GetType()), kOpName);
  auto op_value = GetRequiredKwarg<int>(kwargs, "op", kOpName);
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == target_type->dtype_)
      << kOpName << " dtype kwarg (" << dtype.ToString() << ") must match target dtype ("
      << target_type->dtype_.ToString() << ")";
  CheckSupportedBuiltinVariant(op_value, dtype, kOpName);
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.reduce_scatter")
    .set_description("Internal chip-dispatch builtin for pld.tensor.reduce_scatter.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor to reduce-scatter in place")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<int>("op")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.reduce_scatter")
    .f_deduce_type(DeduceBuiltinTensorReduceScatterType);

// ============================================================================
// builtin.tensor.allgather — host dispatch for pld.tensor.allgather
//
// NOT YET WIRED: reserved for future concurrent-dispatch lowering.
// Currently the HOST allgather path lowers to builtin.tensor.barrier instead;
// this builtin will be used when the concurrent-dispatch lowering lands.
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorAllGatherType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.allgather";
  CHECK(args.size() == 2 || args.size() == 3)
      << kOpName << " requires 2 args (target, signal) or 3 args (local_data, target, signal), but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }
  const size_t target_idx = args.size() == 2 ? 0 : 1;
  const size_t signal_idx = args.size() == 2 ? 1 : 2;
  if (args.size() == 3) {
    auto local_type = As<TileType>(args[0]->GetType());
    CHECK(local_type) << kOpName << " local_data must be a Tile, got " << args[0]->GetType()->TypeName();
  }
  auto target_type = As<DistributedTensorType>(args[target_idx]->GetType());
  CHECK(target_type) << kOpName << " target must be a DistributedTensor, got "
                     << args[target_idx]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << kOpName << " target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  CheckSignalDistributedTensor(As<DistributedTensorType>(args[signal_idx]->GetType()), kOpName);
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == target_type->dtype_)
      << kOpName << " dtype kwarg (" << dtype.ToString() << ") must match target dtype ("
      << target_type->dtype_.ToString() << ")";
  CheckSupportedFp32BuiltinVariant(dtype, kOpName);
  if (args.size() == 2) {
    return args[target_idx]->GetType();
  }
  return std::make_shared<TileType>(target_type->shape_, target_type->dtype_);
}

}  // namespace

REGISTER_OP("builtin.tensor.allgather")
    .set_description("Internal chip-dispatch builtin for pld.tensor.allgather.")
    .set_op_category("DistributedOp")
    .add_argument("local_data", "Local tile chunk to stage")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] staging window")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.allgather")
    .f_deduce_type(DeduceBuiltinTensorAllGatherType);

}  // namespace ir
}  // namespace pypto
