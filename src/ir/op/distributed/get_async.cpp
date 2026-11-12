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
 * @file get_async.cpp
 * @brief Asynchronous cross-rank tensor read - ``pld.tensor.get_async``.
 *
 * V1 of task 25.3 exposes an asynchronous URMA read. The op is the async
 * sibling of ``pld.tensor.get`` with the following contract:
 *
 * * It returns ``dst`` (the result aliases argument 0); returning from the op
 *   does **not** mean the data has arrived.
 * * Completion is owned by the enclosing task and rides the producer ``InOut``
 *   -> consumer ``In`` task dependency. There is no user-visible session,
 *   event, or wait/test in V1.
 * * ``engine`` is a per-op compile-time kwarg. V1 accepts URMA only and rejects
 *   any other engine by name rather than silently falling back.
 *
 * IR signature::
 *
 *     pld.tensor.get_async(dst, peer, src, *, engine: int) -> dst
 *     pld.tensor.get_async(dst, peer, src, dst_offsets, src_offsets, shape,
 *                          *, engine: int) -> dst
 *
 * The op is deliberately **not** lowered in this slice: ``ConvertTensorToTileOps``
 * learns the async form in the following pass PR. Until then, tests exercise the
 * IR contract only.
 *
 * Verifier:
 *
 * * ``dst`` must be a :class:`DistributedTensorType` — both endpoints are
 *   registered-window views. A plain tensor would require automatic MR
 *   registration, which is outside the V1 interface.
 * * ``src`` must be a :class:`DistributedTensorType` — the peer needs a window
 *   slot to read from.
 * * ``peer`` must be a scalar integer rank expression; constants must be non-negative.
 * * ``dst`` / ``src`` share element type and rank; static dims are positive.
 *   Full-slice transfers require matching shapes; subregions supply
 *   ``dst_offsets``, ``src_offsets`` and ``shape`` together.
 * * Leading singleton transfer dimensions may select partial inner regions.
 *   After the first non-singleton transfer dimension, every inner dimension
 *   must be full on both endpoints so the selected GM range is contiguous.
 * * ``engine`` must be :enum:`CommEngine::kURMA`.
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "src/ir/op/distributed/comm_op_utils.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kOpName = "pld.tensor.get_async";

TypePtr DeduceGetAsyncType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6) << kOpName
                                              << " requires 3 positional arguments (dst, peer, src) or 6 "
                                                 "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
                                              << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dst_type) << kOpName << " dst must be a DistributedTensor (window-bound), got "
                  << args[0]->GetType()->TypeName();
  comm_op::ValidateAsyncPeer(args[1], kOpName);
  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(src_type) << kOpName << " src must be a DistributedTensor (window-bound), got "
                  << args[2]->GetType()->TypeName();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << kOpName << " dst and src must have the same element type, got dst " << args[0]->GetType()->TypeName()
      << " vs src " << args[2]->GetType()->TypeName();

  // Both endpoints are registered-window views, and V1 provides no packing or
  // staging, so the physical views must be dense row-major as well.
  comm_op::ValidateAsyncEndpointView(dst_type, kOpName, "dst");
  comm_op::ValidateAsyncEndpointView(src_type, kOpName, "src");

  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, kOpName,
                                         /*require_same_shape=*/args.size() == 3);
  if (args.size() == 6) {
    comm_op::ValidateAsyncIntegerRegionArgs(args, 3, kOpName);
    const auto transfer_shape =
        comm_op::ValidateRegionArgs(args, 3, dst_type->shape_, src_type->shape_, kOpName);
    comm_op::ValidateAsyncContiguousTransfer(transfer_shape, dst_type->shape_, kOpName);
    comm_op::ValidateAsyncContiguousTransfer(transfer_shape, src_type->shape_, kOpName);
  }
  // Capacity is established upstream: MaterializeCommDomainScopes rejects a
  // window view larger than its backing allocation, and the transfer-region
  // validators above bound this request to the endpoint view. Symbolic view or
  // slot sizes are checked by the generated host guard before dispatch.
  comm_op::ValidateAsyncEngine(kwargs, kOpName);

  // The result is the local destination buffer itself; no fresh allocation.
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.get_async - asynchronous URMA read from a peer rank's slice
// ============================================================================
//
// Core placement: VECTOR, matching the synchronous TGET path. The fetch is
// issued from the AIV task that owns the destination; completion is registered
// against that task, so the consumer's In dependency is released only after the
// task's in-core work and its async requests complete.

REGISTER_OP("pld.tensor.get_async")
    .set_description(
        "Asynchronous cross-rank get: issue a URMA read of the `peer` rank's slice of the "
        "window-bound DistributedTensor `src` into local `dst` and return `dst`. Returning "
        "does not mean the data has arrived; the enclosing task owns completion and the "
        "producer InOut -> consumer In dependency carries it. `engine` is a compile-time "
        "kwarg; V1 accepts CommEngine.URMA only.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local destination — window-bound DistributedTensor (same dtype as src)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Remote (peer) window-bound DistributedTensor source (same dtype as dst)")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the local dst; present only in the "
                  "subregion form (all three region args supplied together)")
    .add_argument("src_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's src slice; present only in the "
                  "subregion form")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); present only in the subregion form")
    .set_attr<int>("engine")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    // The local destination is written in place and remains live as the
    // producer's InOut output; the remote source is only read.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_arg_effect(2, ArgEffect::Read)
    .set_output_reuses_input(0)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type(DeduceGetAsyncType);

}  // namespace ir
}  // namespace pypto
