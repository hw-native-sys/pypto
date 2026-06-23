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
 * @brief Distributed tensor-level collective ops — pld.tensor.{barrier,broadcast,allgather,reduce_scatter}.
 *
 * Composite collective ops that lower through LowerCompositeOps (pass 14)
 * into notify/wait/remote_load/store primitives.  Each op registers a type
 * deducer and an op description; the actual IR expansion lives in the
 * lowering pass.
 *
 *   - pld.tensor.barrier(signal)                -> DistributedTensorType
 *   - pld.tensor.broadcast(target, signal, root) -> DistributedTensorType
 *   - pld.tensor.allgather(local_data, target, signal, out) -> TensorType
 *   - pld.tensor.reduce_scatter(target, signal, op)    -> DistributedTensorType
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
  CHECK(signal_type->shape_.size() == 1)
      << kOpName << " signal must be a rank-1 DistributedTensor, got rank " << signal_type->shape_.size();

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
        "per rank). Lowered to a notify-all / wait-all sequence by LowerCompositeOps; this op "
        "never survives past that pass.")
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
        "the source rank. Lowered to notify-all / wait-all + pld.tensor.get by LowerCompositeOps; "
        "this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("root")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorBroadcastType);

// ============================================================================
// pld.tensor.allgather — gather data from every rank into every rank's window
// ============================================================================

namespace {

TypePtr DeduceTensorAllGatherType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 4) << "pld.tensor.allgather requires exactly 4 positional arguments "
                             "(local_data, target, signal, out), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allgather positional argument #" << i << " must not be null";
  }

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
        "All-gather: gather data from all ranks, writing the concatenated result into "
        "a user-provided output Tensor. `local_data` is the rank's chunk (Tensor or Tile [1, SIZE]). "
        "`target` is a window-bound DistributedTensor[NR, SIZE] used as the staging area. "
        "`signal` is a window-bound INT32 DistributedTensor used as the cross-rank barrier. "
        "`out` is a plain Tensor[1, NR*SIZE] that receives the rank-ordered concatenation. "
        "Lowered to tile.load (when local_data is a Tensor) + tile.store + notify-all / wait-all "
        "+ per-peer remote_load + tile.store into out by LowerCompositeOps; this op never "
        "survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("local_data", "Local Tensor or Tile [1, SIZE] — this rank's data (Input)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .add_argument("out", "Plain Tensor[1, NR*SIZE] — receives the gathered result (Output)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllGatherType);

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
        "barrier. `op` selects the reduction operator (Sum only in first version). Lowered to "
        "a 5-phase decomposition by LowerCompositeOps; this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("op")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorReduceScatterType);

}  // namespace ir
}  // namespace pypto
