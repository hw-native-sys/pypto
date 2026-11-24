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
 * @file async_session.cpp
 * @brief Peer-independent URMA sessions, explicit PUT, and event completion.
 *
 * Sessions carry mutable AIV-local state. Each submission supplies the peer
 * for its remote view and produces one event. Waiting observes completion on
 * the issuing core; deferring hands the completion condition to the task.
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

constexpr const char* kMakeSessionOp = "pld.system.make_session";
constexpr const char* kPutAsyncOp = "pld.tile.put_async";
constexpr const char* kWaitEventOp = "pld.system.wait_event";

/// Validate positional arity and null operands for a V2 session/event op.
void CheckArity(const char* op_name, const char* arg_roles, size_t expected,
                const std::vector<ExprPtr>& args) {
  CHECK(args.size() == expected) << op_name << " requires exactly " << expected << " positional argument(s) ("
                                 << arg_roles << "), but got " << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " positional argument #" << i << " must not be null";
  }
}

/// Validate that `arg` carries the singleton handle type `HandleType`.
template <typename HandleType>
void CheckHandle(const char* op_name, const char* arg_role, const ExprPtr& arg, const char* producer) {
  CHECK_SPAN(IsA<HandleType>(arg->GetType()), arg->span_)
      << op_name << " expects " << arg_role << " (output of " << producer << "), got "
      << arg->GetType()->TypeName();
}

// --------------------------------------------------------------------------
// pld.system.make_session
// --------------------------------------------------------------------------

TypePtr DeduceMakeSessionType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckArity(kMakeSessionOp, "comm_ctx", 1, args);
  CheckHandle<CommCtxType>(kMakeSessionOp, "a CommCtx", args[0], "pld.system.get_comm_ctx");
  comm_op::ValidateAsyncEngine(kwargs, kMakeSessionOp);
  return GetAsyncSessionType();
}

// --------------------------------------------------------------------------
// pld.tile.put_async
// --------------------------------------------------------------------------

TypePtr DeducePutAsyncType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckArity(kPutAsyncOp, "dst, peer, src, session", 4, args);
  CHECK_SPAN(kwargs.empty(), args[0]->span_) << kPutAsyncOp << " takes no kwargs";
  comm_op::ValidateAsyncPeer(args[1], kPutAsyncOp);
  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK_SPAN(dst_type, args[0]->span_) << kPutAsyncOp << " dst must be a window-bound DistributedTensor, got "
                                       << args[0]->GetType()->TypeName();
  auto src_type = AsTensorTypeLike(args[2]->GetType());
  CHECK_SPAN(src_type, args[2]->span_)
      << kPutAsyncOp << " src must be a Tensor, got " << args[2]->GetType()->TypeName();
  CHECK_SPAN(dst_type->dtype_ == src_type->dtype_, args[0]->span_)
      << kPutAsyncOp << " dst and src must have the same element type, got dst "
      << args[0]->GetType()->TypeName() << " vs src " << args[2]->GetType()->TypeName();

  CheckHandle<AsyncSessionType>(kPutAsyncOp, "session to be an AsyncSession", args[3],
                                "pld.system.make_session");

  // No packing or staging is provided, so both physical views must be dense
  // row-major - the Gather descriptor describes one contiguous byte range per
  // segment.
  comm_op::ValidateAsyncEndpointView(dst_type, kPutAsyncOp, "dst");
  comm_op::ValidateAsyncEndpointView(src_type, kPutAsyncOp, "src");
  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, kPutAsyncOp,
                                         /*require_same_shape=*/true);

  // One submission yields exactly one event covering the whole logical
  // transfer; PyPTO never splits or aggregates.
  return GetAsyncEventType();
}

// --------------------------------------------------------------------------
// pld.system.wait_event
// --------------------------------------------------------------------------

// The verifier distinguishes tensor events from tile events after SSA conversion.
void CheckCompletionOperands(const char* name, const std::vector<ExprPtr>& args,
                             const comm_op::Kwargs& kwargs) {
  CHECK(args.size() == 1 || args.size() == 2) << name << " requires (event[, session])";
  CHECK(kwargs.empty()) << name << " takes no kwargs";
  CHECK(args[0] && IsA<AsyncEventType>(args[0]->GetType())) << name << " expects an AsyncEvent";
  if (args.size() == 2) {
    CHECK(args[1] && IsA<AsyncSessionType>(args[1]->GetType())) << name << " expects an AsyncSession";
  }
}

}  // namespace

// ============================================================================
// pld.system.make_session — build an async submission session
// ============================================================================
//
// VECTOR, matching the transfer ops it feeds: the session is constructed and
// posted to from the AIV task that owns the transfer.
//
// `set_no_duplicate()` is load-bearing, not stylistic. Copying the construction
// onto another lane would give two lanes what looks like one queue but is two,
// so an event posted on one would be waited on the other.

REGISTER_OP("pld.system.make_session")
    .set_description("Construct a mutable AIV-local URMA session for a communication workspace.")
    .set_op_category("DistributedOp")
    .add_argument("comm_ctx", "Communication domain context from pld.system.get_comm_ctx")
    .set_attr<int>("engine")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_no_duplicate()
    .no_memory_spec()
    // Reads the context to derive the workspace; writes no user tensor.
    .no_arg_writes()
    .f_deduce_type(DeduceMakeSessionType);

// ============================================================================
// pld.tile.put_async — submit one URMA write, return its completion event
// ============================================================================
//
// VECTOR, as for the V1 async put.
//
// This is a separate op rather than an overload of `pld.tensor.put_async`
// because it returns an event instead of `dst`: an overload would make one call
// change its return type by context. `dst` stays ReadWrite so the write is
// visible to dependence analysis, but there is no `set_output_reuses_input` —
// the result is a handle, not a buffer.

REGISTER_OP("pld.tile.put_async")
    .set_description("Submit one URMA write to the given peer and return its completion event.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote window-bound GM destination")
    .add_argument("peer", "Remote rank for this submission")
    .add_argument("src", "Local GM source with the destination's shape and dtype")
    .add_argument("session", "URMA session for the destination communication domain")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    // The destination is written in place; the source is only read. The IR keeps
    // the real payload operands and their effects - descriptors are a backend
    // detail and never appear here.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_arg_effect(2, ArgEffect::Read)
    .set_arg_effect(3, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type(DeducePutAsyncType);

// ============================================================================
// pld.system.wait_event — block until one async transfer completes
// ============================================================================

REGISTER_OP("pld.system.wait_event")
    .set_description(
        "Block until an asynchronous transfer event completes within its producing session "
        "and yield a BOOL success flag. Lowers to the ISA event.Wait(session); registers no "
        "task-completion condition. The event must come from a submission on the same "
        "session.")
    .set_op_category("DistributedOp")
    .add_argument("event", "Distributed PUT or GET completion event")
    .add_argument("session", "Optional originating session for tile events only")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    // Observes completion of an already-submitted request; writes no operand.
    .no_arg_writes()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const comm_op::Kwargs& kwargs) -> TypePtr {
      CheckCompletionOperands(kWaitEventOp, args, kwargs);
      return std::make_shared<ScalarType>(DataType::BOOL);
    });

REGISTER_OP("pld.system.defer_wait_event")
    .set_description("Hand completion of a distributed transfer to the issuing task.")
    .set_op_category("DistributedOp")
    .add_argument("event", "Distributed PUT or GET completion event")
    .add_argument("session", "Optional originating session for tile events only")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    .no_arg_writes()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const comm_op::Kwargs& kwargs) -> TypePtr {
      CheckCompletionOperands("pld.system.defer_wait_event", args, kwargs);
      return GetUnknownType();
    });

}  // namespace ir
}  // namespace pypto
