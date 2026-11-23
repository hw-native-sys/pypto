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
 * @brief Explicit session/event surface for URMA async transfers - V2 of task 25.3.
 *
 * V1 (``pld.tensor.put_async`` / ``get_async``) returns ``dst`` and lets the
 * enclosing task own completion; there is no user-visible session or event.
 * V2 is **additive**: it adds an explicit session, an explicit event, and an
 * in-kernel wait, while V1 keeps its return type and completion guarantee
 * unchanged.
 *
 * IR signatures::
 *
 *     pld.system.make_session(ctx, peer, *, engine: int) -> AsyncSession
 *     pld.tile.put_async(dst, src, session)              -> AsyncEvent
 *     pld.system.wait_event(event, session)              -> bool
 *
 * Design points, from the #3304 thread (see notes plan 144):
 *
 * * **The session owns engine and peer** (D1). The tile ops do not repeat them,
 *   so one session can back many submissions to the same peer.
 * * **The session is mutable state, not a value** (C12). ISA updates its runtime
 *   state across posts and waits, so it is never pure: it must not be CSE'd,
 *   hoisted as a pure op, copied, or substituted for another session over the
 *   same workspace.
 * * **An event is one logical transfer** (C13). PyPTO never splits a transfer or
 *   aggregates per-chunk events.
 * * **Submission registers nothing.** ``wait_event`` observes completion on this
 *   core. The deferred path (``defer_wait`` on an event) is deliberately **not**
 *   in this change - its emission site is still open upstream (simpler #3186),
 *   and the two candidate shapes emit at different places.
 *
 * Not in this change: the Gather descriptor materialization and native
 * ``pto.comm.tput_async_gather`` emission (backend), ``MaterializeAsyncSessions``
 * (pass), the event form of ``defer_wait``, and GET. GET's V2 path is the ISA
 * Scatter ``TGET_ASYNC<URMA>`` overload, but PTOAS has no Scatter GET op yet, so
 * there is nothing to lower it to.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
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
  CHECK(IsA<HandleType>(arg->GetType())) << op_name << " expects " << arg_role << " (output of " << producer
                                         << "), got " << arg->GetType()->TypeName();
}

// --------------------------------------------------------------------------
// pld.system.make_session
// --------------------------------------------------------------------------

TypePtr DeduceMakeSessionType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckArity(kMakeSessionOp, "ctx, peer", 2, args);
  CheckHandle<CommCtxType>(kMakeSessionOp, "ctx to be a CommContext", args[0], "pld.system.get_comm_ctx");
  // One session binds exactly one peer: the backend writes that peer into the
  // single remote descriptor. A dynamic peer is allowed - the materialization
  // pass keys session reuse on it - but it must be an integer rank index.
  comm_op::ValidateAsyncPeer(args[1], kMakeSessionOp);
  comm_op::ValidateAsyncEngine(kwargs, kMakeSessionOp);
  return GetAsyncSessionType();
}

// --------------------------------------------------------------------------
// pld.tile.put_async
// --------------------------------------------------------------------------

TypePtr DeducePutAsyncType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckArity(kPutAsyncOp, "dst, src, session", 3, args);
  CHECK(kwargs.empty()) << kPutAsyncOp
                        << " takes no kwargs: engine and peer live on the session, and the V2 "
                           "submission controls (cqe_policy, submit_mode, batch_size) are V3, not V2";

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dst_type) << kPutAsyncOp << " dst must be a window-bound DistributedTensor, got "
                  << args[0]->GetType()->TypeName();
  auto src_type = AsTensorTypeLike(args[1]->GetType());
  CHECK(src_type) << kPutAsyncOp << " src must be a Tensor, got " << args[1]->GetType()->TypeName();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << kPutAsyncOp << " dst and src must have the same element type, got dst "
      << args[0]->GetType()->TypeName() << " vs src " << args[1]->GetType()->TypeName();

  CheckHandle<AsyncSessionType>(kPutAsyncOp, "session to be an AsyncSession", args[2],
                                "pld.system.make_session");

  // No packing or staging is provided, so both physical views must be dense
  // row-major - the Gather descriptor describes one contiguous byte range per
  // segment.
  comm_op::ValidateAsyncEndpointView(dst_type, kPutAsyncOp, "dst");
  comm_op::ValidateAsyncEndpointView(src_type, kPutAsyncOp, "src");
  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, kPutAsyncOp,
                                         /*require_same_shape=*/true);

  // One submission yields exactly one event covering the whole logical
  // transfer; PyPTO never splits or aggregates (C13).
  return GetAsyncEventType();
}

// --------------------------------------------------------------------------
// pld.system.wait_event
// --------------------------------------------------------------------------

TypePtr DeduceWaitEventType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckArity(kWaitEventOp, "event, session", 2, args);
  CHECK(kwargs.empty()) << kWaitEventOp << " takes no kwargs, but got " << kwargs.size();
  CheckHandle<AsyncEventType>(kWaitEventOp, "event to be an AsyncEvent", args[0], "pld.tile.put_async");
  CheckHandle<AsyncSessionType>(kWaitEventOp, "session to be an AsyncSession", args[1],
                                "pld.system.make_session");
  return std::make_shared<ScalarType>(DataType::BOOL);
}

}  // namespace

// ============================================================================
// pld.system.make_session — build a per-peer async submission session
// ============================================================================
//
// Core placement: VECTOR, matching the async transfer ops it feeds. The session
// is constructed and posted to from the AIV task that owns the transfer.
//
// `set_no_duplicate()` is load-bearing and not a stylistic choice. A session
// carries mutable ISA runtime state across posts and waits (C12). Duplicating
// its construction onto another lane would hand two lanes what looks like the
// same queue but is two independent ones, and an event posted on one would be
// waited on the other.

REGISTER_OP("pld.system.make_session")
    .set_description(
        "Build an asynchronous submission session (AsyncSessionType) bound to one peer rank "
        "on a CommContext. The session owns the engine and the peer, so the tile submission "
        "ops do not repeat them, and it is reused across submissions to that peer. The "
        "session carries mutable ISA state: it is never copied, CSE'd, or substituted for "
        "another session over the same workspace. `engine` is a compile-time kwarg; V2 "
        "accepts CommEngine.URMA only.")
    .set_op_category("DistributedOp")
    .add_argument("ctx", "A CommContext handle (CommCtxType) from pld.system.get_comm_ctx")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
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
// Core placement: VECTOR, as for the V1 async put.
//
// Unlike the V1 `pld.tensor.put_async`, this op returns an **event**, not `dst`.
// That is deliberate (D2): one call never changes its return type by context,
// so the explicit-event form is a separate op rather than an overload. `dst` is
// still marked ReadWrite so the write is visible to dependence analysis, but
// there is no `set_output_reuses_input` - the result is a handle, not a buffer.
//
// Returning does **not** mean the transfer completed. Completion is observed by
// pld.system.wait_event on this core, or (later, not in this change) handed to
// the task through the event form of defer_wait.

REGISTER_OP("pld.tile.put_async")
    .set_description(
        "Submit one asynchronous URMA write of local `src` into the session's peer slice of "
        "the window-bound DistributedTensor `dst`, and return the completion event "
        "(AsyncEventType) for that single logical transfer. The engine and peer come from "
        "`session`. Returning does not mean completion: pair the event with its producing "
        "session in pld.system.wait_event. Submission registers no task condition.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("src", "Local source — Tensor or DistributedTensor (same dtype as dst)")
    .add_argument("session", "The async session (AsyncSessionType) from pld.system.make_session")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    // The destination is written in place; the source is only read. The IR keeps
    // the real payload operands and their effects - descriptors are a backend
    // detail and never appear here (C16).
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_arg_effect(1, ArgEffect::Read)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type(DeducePutAsyncType);

// ============================================================================
// pld.system.wait_event — block until one async transfer completes
// ============================================================================
//
// Lowers to the ISA `event.Wait(session)` and registers **no** task-completion
// condition: an in-kernel wait is the alternative to handing completion over,
// not a supplement to it.

REGISTER_OP("pld.system.wait_event")
    .set_description(
        "Block until an asynchronous transfer event completes within its producing session "
        "and yield a BOOL success flag. Lowers to the ISA event.Wait(session); registers no "
        "task-completion condition. The event must come from a submission on the same "
        "session.")
    .set_op_category("DistributedOp")
    .add_argument("event", "A completion event (AsyncEventType) from pld.tile.put_async")
    .add_argument("session", "The matching async session (AsyncSessionType) that produced the event")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_memory_spec()
    // Observes completion of an already-submitted request; writes no operand.
    .no_arg_writes()
    .f_deduce_type(DeduceWaitEventType);

}  // namespace ir
}  // namespace pypto
