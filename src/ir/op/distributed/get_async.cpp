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
 * @brief Cross-rank URMA GET with either task-owned or explicit event completion.
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
 * * ``src`` must be a window-bound :class:`DistributedTensorType`.
 * * ``dst`` may be a plain :class:`TensorType` or a distributed tensor.
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
#include <memory>
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

// All async GET forms have the same local and remote GM endpoint contract.
void ValidateGetEndpoints(const std::vector<ExprPtr>& args, const std::string& op_name,
                          bool require_same_shape) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " positional argument #" << i << " must not be null";
  }
  auto dst_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(dst_type) << op_name << " dst must be a Tensor, got " << args[0]->GetType()->TypeName();
  comm_op::ValidateAsyncPeer(args[1], op_name);
  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(src_type) << op_name << " src must be a DistributedTensor (window-bound), got "
                  << args[2]->GetType()->TypeName();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must have the same element type, got dst " << args[0]->GetType()->TypeName()
      << " vs src " << args[2]->GetType()->TypeName();

  // No packing or staging is provided, so both physical views must be dense row-major.
  comm_op::ValidateAsyncEndpointView(dst_type, op_name, "dst");
  comm_op::ValidateAsyncEndpointView(src_type, op_name, "src");

  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, op_name, require_same_shape);
}

TypePtr DeduceGetAsyncType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6) << kOpName
                                              << " requires 3 positional arguments (dst, peer, src) or 6 "
                                                 "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
                                              << args.size();

  ValidateGetEndpoints(args, kOpName, args.size() == 3);
  if (args.size() == 6) {
    auto dst_type = AsTensorTypeLike(args[0]->GetType());
    auto src_type = As<DistributedTensorType>(args[2]->GetType());
    comm_op::ValidateAsyncIntegerRegionArgs(args, 3, kOpName);
    const auto transfer_shape =
        comm_op::ValidateRegionArgs(args, 3, dst_type->shape_, src_type->shape_, kOpName);
    comm_op::ValidateAsyncContiguousTransfer(transfer_shape, dst_type->shape_, kOpName);
    comm_op::ValidateAsyncContiguousTransfer(transfer_shape, src_type->shape_, kOpName);
  }
  // Capacity is established upstream: MaterializeCommDomainScopes rejects a
  // window view larger than its backing allocation, and the transfer-region
  // validators above bound this request to the endpoint view. Symbolic view or
  // slot sizes are checked by the generated host guard at the window declaration.
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
    .add_argument("dst", "Local destination — Tensor or DistributedTensor (same dtype as src)")
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

REGISTER_OP("pld.tensor.get_async_event")
    .set_description("Submit a single-peer URMA GET and return (destination, completion event).")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local GM destination tensor")
    .add_argument("peer", "Remote rank")
    .add_argument("src", "Remote window-bound source tensor")
    .set_attr<int>("engine")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    .set_output_arity(2)
    .set_tuple_result_alias(0, 0)
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_arg_effect(1, ArgEffect::Read)
    .set_arg_effect(2, ArgEffect::Read)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const comm_op::Kwargs& kwargs) -> TypePtr {
      const std::string name = "pld.tensor.get_async_event";
      CHECK(args.size() == 3) << name << " requires (dst, peer, src)";
      ValidateGetEndpoints(args, name, true);
      comm_op::ValidateAsyncEngine(kwargs, name);
      return std::make_shared<TupleType>(std::vector<TypePtr>{args[0]->GetType(), GetAsyncEventType()});
    });

REGISTER_OP("pld.tile.get_async")
    .set_description(
        "Submit a single-peer GET using an existing URMA session and return (destination, completion event).")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local GM destination tensor")
    .add_argument("peer", "Remote rank")
    .add_argument("src", "Remote window-bound source tensor")
    .add_argument("session", "URMA session for the source communication domain")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    .set_output_arity(2)
    .set_tuple_result_alias(0, 0)
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_arg_effect(1, ArgEffect::Read)
    .set_arg_effect(2, ArgEffect::Read)
    .set_mutable_session_arg(3)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const comm_op::Kwargs& kwargs) -> TypePtr {
      const std::string name = "pld.tile.get_async";
      CHECK(args.size() == 4) << name << " requires (dst, peer, src, session)";
      CHECK(kwargs.empty()) << name << " takes no kwargs";
      ValidateGetEndpoints(args, name, true);
      CHECK(IsA<AsyncSessionType>(args[3]->GetType())) << name << " expects an AsyncSession";
      return std::make_shared<TupleType>(std::vector<TypePtr>{args[0]->GetType(), GetAsyncEventType()});
    });

}  // namespace ir
}  // namespace pypto
