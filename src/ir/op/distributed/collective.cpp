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
 *   - pld.tensor.allgather(local_data, target, signal) (unified 3-arg)  -> DistributedTensorType
 *   - pld.tensor.reduce_scatter(target, signal, op)                -> DistributedTensorType
 *   - pld.tensor.all_to_all(input, target, signal)                 -> DistributedTensorType
 *
 * The six builtin.tensor.* ops are internal chip-dispatch targets emitted by the
 * host-orchestrator lowering pass (LowerHostTensorCollectives):
 * builtin.tensor.{allreduce,barrier,broadcast,reduce_scatter,allgather}.
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
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
// pld.tensor.allgather — gather data from all ranks (unified 3-arg: HOST builtin or InCore composite)
// ============================================================================

namespace {

// Dimension pairs that must agree at runtime (e.g. two windows' NR extents)
// may be two structurally distinct IR nodes for the same value — each
// commonly sourced from its own pld.world_size() call in the HOST builtin
// path. Enforce equality only when both are statically known (ConstInt);
// otherwise trust the runtime rather than comparing structurally.
void CheckDimAgreesIfStatic(const ExprPtr& lhs, const ExprPtr& rhs, const std::string& op_name,
                            const char* lhs_desc, const char* rhs_desc) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (lhs_const && rhs_const) {
    CHECK(lhs_const->value_ == rhs_const->value_)
        << op_name << " " << lhs_desc << " first dimension (" << lhs_const->value_ << ") must equal "
        << rhs_desc << " first dimension (" << rhs_const->value_ << ")";
  }
}

TypePtr DeduceTensorAllGatherType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 3) << "pld.tensor.allgather requires exactly 3 args (input, target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allgather positional argument #" << i << " must not be null";
  }

  // Unified 3-arg contract for both paths: allgather(input, target, signal).
  //   arg[0] input  — this rank's single chunk, always [1, SIZE]. InCore: plain
  //                   Tensor. HOST: a [1, SIZE] DistributedTensor staging window.
  //                   HOST vs InCore is a function-context property, not an
  //                   arg[0]-type property, so the deducer accepts either kind
  //                   and defers path-specific validation to the lowering passes.
  //   arg[1] target — DistributedTensor [NR, SIZE] window (push target + result).
  //   arg[2] signal — DistributedTensor INT32 cross-rank barrier.
  // input and target must be different buffers — aliasing them is a
  // cross-process data race (same constraint as all_to_all).
  CHECK(args[0].get() != args[1].get())
      << "pld.tensor.allgather input and target must be different buffers, but the same "
         "expression was passed for both";

  auto input_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(input_type) << "pld.tensor.allgather input must be a Tensor or DistributedTensor, got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2)
      << "pld.tensor.allgather input must be 2D [1, SIZE] (this rank's single chunk), got "
      << input_type->shape_.size() << " dims";
  if (auto input_rows = As<ConstInt>(input_type->shape_[0])) {
    CHECK(input_rows->value_ == 1)
        << "pld.tensor.allgather input must be [1, SIZE] (this rank's single chunk), got first dim "
        << input_rows->value_;
  }

  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << "pld.tensor.allgather target must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << "pld.tensor.allgather target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  CHECK(target_type->dtype_ == input_type->dtype_)
      << "pld.tensor.allgather target dtype " << target_type->dtype_.ToString() << " must match input dtype "
      << input_type->dtype_.ToString();
  // Dim 1 (SIZE) is always a plain literal shared by input and target.
  CHECK(AreExprsEqual(target_type->shape_[1], input_type->shape_[1]))
      << "pld.tensor.allgather target SIZE must equal input SIZE";

  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << "pld.tensor.allgather signal must be a DistributedTensor (window-bound), got "
                     << args[2]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.allgather signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();
  CHECK(signal_type->shape_.size() == 1 || signal_type->shape_.size() == 2)
      << "pld.tensor.allgather signal must be 1D [NR] or 2D [NR, 1], got " << signal_type->shape_.size()
      << " dims";
  if (signal_type->shape_.size() == 2) {
    auto signal_dim1 = As<ConstInt>(signal_type->shape_[1]);
    CHECK(signal_dim1 && signal_dim1->value_ == 1)
        << "pld.tensor.allgather signal second dimension must be 1, got "
        << (signal_dim1 ? std::to_string(signal_dim1->value_) : "<dynamic>");
  }
  // The notify/wait barrier indexes `signal` per rank (0..NR-1), so its first
  // dimension must equal target's NR.  Two windows commonly source NR from
  // separate pld.world_size() nodes, so only enforce when both are static.
  CheckDimAgreesIfStatic(signal_type->shape_[0], target_type->shape_[0], "pld.tensor.allgather", "signal",
                         "target");

  // Return target in-place (window-as-result).
  return target_type;
}

}  // namespace

REGISTER_OP("pld.tensor.allgather")
    .set_description(
        "All-gather: gather data from all ranks.  Unified 3-arg push-based API "
        "`pld.tensor.allgather(input, target, signal)`.  `input` is this rank's "
        "single [1, SIZE] chunk (plain Tensor on the InCore path, a [1, SIZE] "
        "staging window on the HOST path); `target` is a window-bound "
        "DistributedTensor[NR, SIZE] that receives the gathered result in-place "
        "— after the barrier `target[src, :]` holds the chunk from rank `src`; "
        "`signal` is a window-bound INT32 barrier tensor.  "
        "InCore is lowered by LowerCompositeOps into a push decomposition "
        "(pld.tile.put this rank's chunk into every peer's `target` + "
        "notify-all/wait-all); HOST is lowered by LowerHostTensorCollectives to "
        "builtin.tensor.allgather per chip (in-kernel TPUT push + barrier).  "
        "Returns `target` in-place; the composite Call never survives lowering.")
    .set_op_category("DistributedOp")
    .add_argument("input",
                  "This rank's single chunk — [1, SIZE] Tensor (InCore) or [1, SIZE] staging "
                  "window (HOST) (Input)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] — gathered result in-place (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllGatherType);

// ============================================================================
// pld.tensor.all_to_all — symmetric all-to-all (3-arg push-based InCore composite)
// ============================================================================

namespace {

TypePtr DeduceTensorAllToAllType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 3)
      << "pld.tensor.all_to_all requires 3 args (input, target, signal) for InCore composite, but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.all_to_all positional argument #" << i << " must not be null";
  }
  // input and target must be different windows — aliasing them is a
  // cross-process data race (see kernel.cpp.in for the full explanation).
  // This catches the same-expression case; two distinct pld.window(...)
  // calls over the same underlying alloc aren't detectable here (window
  // identity isn't materialized until MaterializeCommDomainScopes).
  CHECK(args[0].get() != args[1].get())
      << "pld.tensor.all_to_all input and target must be different windows, but the same "
         "expression was passed for both";

  // 3-arg push-based composite or HOST builtin path.
  // input may be plain Tensor (InCore) or DistributedTensor (HOST window-sourced).
  auto input_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(input_type) << "pld.tensor.all_to_all input must be a Tensor or DistributedTensor, got "
                    << args[0]->GetType()->TypeName();
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
  // Dim 0 (NR): input and target are two separate windows (the HOST builtin
  // path requires different buffers — see kernel.cpp.in), so their NR extent
  // is commonly two distinct pld.world_size() IR nodes. Dim 1 (SIZE) is
  // always a plain literal, so keep it a strict structural check.
  CheckDimAgreesIfStatic(target_type->shape_[0], input_type->shape_[0], "pld.tensor.all_to_all", "target",
                         "input");
  CHECK(AreExprsEqual(target_type->shape_[1], input_type->shape_[1]))
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
  CHECK(signal_type->shape_.size() == 1 || signal_type->shape_.size() == 2)
      << "pld.tensor.all_to_all signal must be 1D [NR] or 2D [NR, 1], got " << signal_type->shape_.size()
      << " dims";
  if (signal_type->shape_.size() == 2) {
    auto signal_dim1 = As<ConstInt>(signal_type->shape_[1]);
    CHECK(signal_dim1 && signal_dim1->value_ == 1)
        << "pld.tensor.all_to_all signal second dimension must be 1, got "
        << (signal_dim1 ? std::to_string(signal_dim1->value_) : "<dynamic>");
  }
  CheckDimAgreesIfStatic(signal_type->shape_[0], input_type->shape_[0], "pld.tensor.all_to_all", "signal",
                         "input");

  // Return target in-place (window-as-result, same idiom as reduce_scatter / broadcast).
  return target_type;
}

}  // namespace

REGISTER_OP("pld.tensor.all_to_all")
    .set_description(
        "All-to-all: symmetric personalized exchange.  Every rank pushes its "
        "per-destination chunks directly to every peer's window via "
        "``pld.tensor.put`` (TPUT), then synchronises with a notify/wait "
        "barrier.  ``input`` is a Tensor [NR, SIZE] where ``input[dest, :]`` "
        "is the chunk destined for rank ``dest``.  ``target`` is a "
        "window-bound DistributedTensor [NR, SIZE] that receives the result "
        "in-place — after the barrier ``target[src, :]`` holds the chunk "
        "received from rank ``src``.  ``signal`` is a window-bound INT32 "
        "barrier tensor.  Lowered by LowerCompositeOps into a 2-phase push "
        "decomposition (push → barrier → return target).")
    .set_op_category("DistributedOp")
    .add_argument("input", "Plain Tensor [NR, SIZE] with per-destination chunks (Input)")
    .add_argument("target",
                  "Window-bound DistributedTensor [NR, SIZE] — receives the result in-place (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
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
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorAllGatherType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.allgather";
  CHECK(args.size() == 3) << kOpName << " requires 3 args (input, target, signal), but got " << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }
  CHECK(args[0].get() != args[1].get())
      << kOpName
      << " input and target must be different windows, but the same expression was passed for both";
  auto input_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(input_type) << kOpName << " input must be a DistributedTensor (window-bound), got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2)
      << kOpName << " input must be 2D [1, SIZE] (this rank's single chunk), got "
      << input_type->shape_.size() << " dims";
  if (auto input_rows = As<ConstInt>(input_type->shape_[0])) {
    CHECK(input_rows->value_ == 1) << kOpName
                                   << " input must be [1, SIZE] (this rank's single chunk), got first dim "
                                   << input_rows->value_;
  }
  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << kOpName << " target must be a DistributedTensor, got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << kOpName << " target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  CHECK(target_type->dtype_ == input_type->dtype_)
      << kOpName << " target dtype " << target_type->dtype_.ToString() << " must match input dtype "
      << input_type->dtype_.ToString();
  // input is [1, SIZE]; only the SIZE dimension (dim 1) must match target.
  CHECK(AreExprsEqual(target_type->shape_[1], input_type->shape_[1]))
      << kOpName << " input SIZE must equal target SIZE";
  // Accept both 1D and 2D signals (matching DeduceTensorAllGatherType / all_to_all)
  // so the builtin deducer is consistent with the user-facing op.
  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << kOpName << " signal must be a DistributedTensor, got "
                     << args[2]->GetType()->TypeName();
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
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == target_type->dtype_)
      << kOpName << " dtype kwarg (" << dtype.ToString() << ") must match target dtype ("
      << target_type->dtype_.ToString() << ")";
  CheckSupportedFp32BuiltinVariant(dtype, kOpName);
  // 3-arg HOST builtin (input, target, signal): return target in-place.
  return args[1]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.allgather")
    .set_description("Internal chip-dispatch builtin for pld.tensor.allgather.")
    .set_op_category("DistributedOp")
    .add_argument("input",
                  "Window-bound DistributedTensor[1, SIZE] — this rank's staging window (TPUT source)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] result window (TPUT destination)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.allgather")
    .f_deduce_type(DeduceBuiltinTensorAllGatherType);

// ============================================================================
// builtin.tensor.all_to_all — host dispatch for pld.tensor.all_to_all
// ============================================================================

namespace {

TypePtr DeduceBuiltinTensorAllToAllType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.all_to_all";
  CHECK(args.size() == 3) << kOpName
                          << " requires exactly 3 positional arguments (input, target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }
  // input: a SEPARATE window, distinct from target, holding this rank's
  // per-destination outgoing chunks. Never a destination for any incoming
  // TPUT. This catches the same-expression case; two distinct windows over
  // the same underlying alloc aren't detectable here.
  CHECK(args[0].get() != args[1].get())
      << kOpName
      << " input and target must be different windows, but the same expression was "
         "passed for both";
  auto input_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(input_type) << kOpName << " input must be a DistributedTensor (window-bound), got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2)
      << kOpName << " input must be 2D [NR, SIZE], got " << input_type->shape_.size() << " dims";

  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << kOpName << " target must be a DistributedTensor, got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << kOpName << " target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";
  // Dim 0 (NR): input and target are two separate windows, each typically
  // sourced from its own pld.world_size() call. Dim 1 (SIZE) is always a
  // plain literal, so keep it a strict structural check.
  CheckDimAgreesIfStatic(target_type->shape_[0], input_type->shape_[0], kOpName, "target", "input");
  CHECK(AreExprsEqual(target_type->shape_[1], input_type->shape_[1]))
      << kOpName << " target shape must equal input shape";
  CHECK(target_type->dtype_ == input_type->dtype_)
      << kOpName << " target dtype " << target_type->dtype_.ToString() << " must match input dtype "
      << input_type->dtype_.ToString();
  // Accept both 1D and 2D signals (matching DeduceTensorAllToAllType) so that
  // the builtin deducer is consistent with the user-facing op.  The lowering
  // pass (LowerHostTensorCollectives::CheckStaticSignalCapacity) also handles
  // both shapes.
  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << kOpName << " signal must be a DistributedTensor, got "
                     << args[2]->GetType()->TypeName();
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
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == target_type->dtype_)
      << kOpName << " dtype kwarg (" << dtype.ToString() << ") must match target dtype ("
      << target_type->dtype_.ToString() << ")";
  CheckSupportedFp32BuiltinVariant(dtype, kOpName);
  return args[1]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.all_to_all")
    .set_description("Internal chip-dispatch builtin for pld.tensor.all_to_all.")
    .set_op_category("DistributedOp")
    .add_argument("input",
                  "Window-bound DistributedTensor[NR, SIZE] — this rank's outgoing staging window "
                  "(TPUT source only, never an incoming-push destination)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] result window (TPUT destination)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.all_to_all")
    .f_deduce_type(DeduceBuiltinTensorAllToAllType);

}  // namespace ir
}  // namespace pypto
