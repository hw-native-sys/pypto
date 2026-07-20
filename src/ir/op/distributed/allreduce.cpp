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
 * @file allreduce.cpp
 * @brief Distributed tensor-level allreduce — pld.tensor.allreduce.
 *
 * Composite collective op: in-place allreduce of a window-bound
 * :class:`DistributedTensorType` across every rank of its comm group, using a
 * window-bound INT32 ``signal`` matrix for the cross-rank barrier. Sibling of
 * ``pld.system.notify`` / ``pld.system.wait`` / ``pld.tile.remote_load``.
 * Explicit-signal InCore mesh allreduce uses a ready barrier followed by
 * UB-sized reduce/barrier/store chunks in
 * ``src/ir/transforms/lower_composite_ops_pass.cpp``; host-level allreduce is
 * lowered later by ``LowerHostTensorCollectives``.
 * Explicit signal buffers are single-shot: callers issuing multiple allreduces
 * must provide a fresh signal for each call.
 *
 * IR signature:
 *
 *     pld.tensor.allreduce(target, *, op: int)          -> DistributedTensorType
 *     pld.tensor.allreduce(target, signal, *, op: int)  -> DistributedTensorType
 *
 * The ``op`` integer is the underlying value of :enum:`ReduceOp` (see
 * ``include/pypto/ir/comm.h``); the deducer rejects unsupported variants so
 * the lowering rule can dispatch without a separate guard. First-version
 * lowering implements only ``ReduceOp::kSum``.
 *
 * Result type: same as ``target`` (the call is in-place — semantically the
 * returned :class:`DistributedTensor` *is* ``target``, post-reduce). User code
 * therefore writes ``pub = pld.tensor.allreduce(pub, sig, op=...)``, matching
 * the established :func:`pl.store` rebind idiom.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``target`` must have :class:`DistributedTensorType`. The actual
 *   window-buffer binding (``window_buffer_``) is supplied by the host
 *   orchestrator's ``pld.window(...)`` call and may be ``nullopt`` on
 *   InCore parameters that flow through; matching the sibling
 *   ``pld.system.notify`` / ``pld.tile.remote_load`` deducers, this op
 *   relies on the kind check alone here.
 * * ``signal`` must have :class:`DistributedTensorType` with element type
 *   INT32 — the lowering uses ``pld.system.notify`` / ``pld.system.wait``
 *   against this slot.
 * * ``op`` kwarg must be a known :enum:`ReduceOp` value; first-version
 *   lowering accepts only ``ReduceOp::kSum``.
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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceTensorAllReduceType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1 || args.size() == 2)
      << "pld.tensor.allreduce requires 1 or 2 positional arguments (target[, signal]), but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allreduce positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.allreduce target must be a DistributedTensor (window-bound), "
                        "got "
                     << args[0]->GetType()->TypeName();

  if (args.size() == 2) {
    auto signal_type = As<DistributedTensorType>(args[1]->GetType());
    CHECK(signal_type) << "pld.tensor.allreduce signal must be a DistributedTensor (window-bound), "
                          "got "
                       << args[1]->GetType()->TypeName();
    CHECK(signal_type->dtype_ == DataType::INT32)
        << "pld.tensor.allreduce signal must have INT32 element type "
           "(the barrier slot is an int counter), got dtype "
        << signal_type->dtype_.ToString();
  }

  // Validate `op` kwarg falls in the ReduceOp range — first version supports
  // kSum only; the other enum values are accepted by the parser binding but
  // rejected here so users get a clear error rather than silently wrong
  // codegen.
  auto op_value = GetRequiredKwarg<int>(kwargs, "op", "pld.tensor.allreduce");
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << "pld.tensor.allreduce op must be ReduceOp.Sum (got int " << op_value
      << "); Max / Min / Prod lowerings are not yet implemented";

  // Result type: same DistributedTensorType as the input target (in-place
  // reduce — the same view holds the reduced value on every rank). Preserve
  // the window_buffer_ back-reference so downstream passes still see the
  // comm-domain binding.
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.allreduce — in-place all-reduce of a window-bound DistributedTensor
// ============================================================================

REGISTER_OP("pld.tensor.allreduce")
    .set_description(
        "In-place all-reduce of a window-bound DistributedTensor across every rank of its comm "
        "group. After the call, every rank's slice of `target` holds the reduced value. "
        "`signal`, when present, is a window-bound INT32 matrix used as the cross-rank barrier "
        "(one slot per rank). Host one-argument calls synthesize a private signal before lowering. "
        "`op` selects the reduction operator. Explicit-signal InCore calls are lowered to a ready "
        "barrier plus UB-sized remote_load + accumulate + guarded store chunks by LowerCompositeOps; "
        "host-level calls are lowered later by LowerHostTensorCollectives.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (InOut)")
    .add_argument("signal",
                  "Optional window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("op")
    .set_attr<std::string>("mode")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllReduceType);

}  // namespace ir
}  // namespace pypto
