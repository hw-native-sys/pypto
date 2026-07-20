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
 * @file prefetch_async.cpp
 * @brief Asynchronous GM->L2 prefetch ops — ``prefetch.make_context``,
 *        ``prefetch.async_prefetch``, ``prefetch.session`` and ``prefetch.wait``.
 *
 * DSL surface::
 *
 *     ctx     = pl.prefetch.make_context(ws)             # PrefetchAsyncContext
 *     evt     = pl.prefetch.async_prefetch(src, ctx)     # AsyncEvent
 *     session = pl.prefetch.session(ctx)                 # AsyncSession
 *     done    = pl.prefetch.wait(evt, session)           # BOOL scalar
 *
 * These are latency-hiding cache hints: the prefetch warms L2 with ``src`` and
 * changes no tensor values, so a kernel is numerically identical with or
 * without it. Completion is explicit — unlike most PTO intrinsics
 * ``TPREFETCH_ASYNC`` carries no implicit wait-event synchronization, hence the
 * returned event/session pair.
 *
 * Codegen lowers the family to PTOAS ``pto.make_prefetch_async_context`` /
 * ``pto.tprefetch_async`` / ``pto.get_prefetch_async_session`` /
 * ``pto.comm.wait_async_event``. The underlying SDMA CMO path is only effective
 * on A3/A5; on other targets PTOAS degrades the prefetch to a functional no-op,
 * so no PyPTO-side target gating is needed.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Validate positional-arity and reject kwargs for a prefetch op. ``arg_roles``
/// describes the expected arguments in the count message (e.g. ``"src, ctx"``).
void CheckArity(const std::string& op_name, const std::string& arg_roles, size_t expected,
                const std::vector<ExprPtr>& args,
                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == expected) << op_name << " requires exactly " << expected << " positional argument(s) ("
                                 << arg_roles << "), but got " << args.size();
  CHECK(kwargs.empty()) << op_name << " takes no kwargs, but got " << kwargs.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " argument " << i << " must not be null";
  }
}

/// Validate that ``arg`` carries the singleton handle type ``HandleType``.
template <typename HandleType>
void CheckHandle(const std::string& op_name, const std::string& arg_role, const ExprPtr& arg,
                 const std::string& producer) {
  CHECK(IsA<HandleType>(arg->GetType())) << op_name << " expects " << arg_role << " (output of " << producer
                                         << "), got " << arg->GetType()->TypeName();
}

/**
 * @brief Enforce the PTOAS ``tprefetch_async`` source constraint.
 *
 * Mirrors ``verifyAsyncFlatContiguous1DGMViewLike`` in PTOAS: ``src`` must be a
 * GM tensor with a fully static shape that is *logically* 1D — every dimension
 * except the last must be 1, so the region is one flat contiguous run. Rejecting
 * this here means a shape mistake fails at PyPTO IR construction with a DSL-level
 * message instead of surfacing much later as a PTOAS verification error.
 */
void CheckFlatContiguous1DSource(const std::string& op_name, const ExprPtr& src) {
  auto tensor_type = AsTensorTypeLike(src->GetType());
  CHECK(tensor_type) << op_name << " expects src to be a GM Tensor (prefetch reads global memory), got "
                     << src->GetType()->TypeName();

  const auto& shape = tensor_type->shape_;
  CHECK(!shape.empty()) << op_name << " expects src to have rank >= 1, got a rank-0 tensor";

  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    auto dim = As<ConstInt>(shape[i]);
    CHECK(dim) << op_name << " expects src to have a fully static shape, but dimension " << i
               << " is dynamic; async prefetch needs a compile-time byte extent";
    dims.push_back(dim->value_);
  }

  for (size_t i = 0; i + 1 < dims.size(); ++i) {
    CHECK(dims[i] == 1) << op_name
                        << " expects src to be a flat contiguous logical 1D GM region (all "
                           "dimensions except the last must be 1), but dimension "
                        << i << " is " << dims[i]
                        << "; reshape the tensor to [N] or [1, ..., N] before prefetching";
  }
}

}  // namespace

// ============================================================================
// prefetch.make_context — build a PrefetchAsyncContext from a GM i8 workspace
// ============================================================================

REGISTER_OP("prefetch.make_context")
    .set_description(
        "Materialize an asynchronous-prefetch context (PrefetchAsyncContextType) from a "
        "GM INT8 scratch workspace. The workspace backs the SDMA path used by "
        "prefetch.async_prefetch; the resulting handle also carries the async session "
        "projected by prefetch.session.")
    .set_op_category("PrefetchOp")
    // AIV-only: TPREFETCH_ASYNC drives its SDMA tmpBuf from a Vec(UB) scratch
    // tile held inside PrefetchAsyncContext (pto-isa static_asserts
    // ScratchTile::Loc == TileType::Vec), and UB lives on the vector core.
    // Without this, these ops carry no tile operand, fall through to SHARED,
    // and ExpandMixedKernel duplicates them onto the cube lane too.
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("workspace", "A GM scratch Tensor with INT8 element type")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      CheckArity("prefetch.make_context", "a workspace Tensor", 1, args, kwargs);
      auto workspace_type = AsTensorTypeLike(args[0]->GetType());
      CHECK(workspace_type) << "prefetch.make_context expects workspace to be a GM Tensor, got "
                            << args[0]->GetType()->TypeName();
      CHECK(workspace_type->dtype_ == DataType::INT8)
          << "prefetch.make_context expects an INT8 workspace (raw byte scratch for the SDMA "
             "path), got "
          << workspace_type->dtype_.ToString();
      return GetPrefetchAsyncContextType();
    });

// ============================================================================
// prefetch.async_prefetch — start an async GM -> L2 prefetch, return its event
// ============================================================================

REGISTER_OP("prefetch.async_prefetch")
    .set_description(
        "Start one asynchronous prefetch of a flat contiguous 1D GM region into L2 cache "
        "via SDMA CMO and return the completion event (AsyncEventType). Does not block and "
        "does not modify any tensor value — it only warms the cache. Pair the returned "
        "event with prefetch.session(ctx) in prefetch.wait to observe completion.")
    .set_op_category("PrefetchOp")
    // AIV-only: TPREFETCH_ASYNC drives its SDMA tmpBuf from a Vec(UB) scratch
    // tile held inside PrefetchAsyncContext (pto-isa static_asserts
    // ScratchTile::Loc == TileType::Vec), and UB lives on the vector core.
    // Without this, these ops carry no tile operand, fall through to SHARED,
    // and ExpandMixedKernel duplicates them onto the cube lane too.
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("src", "A flat contiguous logical-1D GM Tensor to pull into L2")
    .add_argument("ctx", "An async-prefetch context (PrefetchAsyncContextType)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      CheckArity("prefetch.async_prefetch", "src, ctx", 2, args, kwargs);
      CheckFlatContiguous1DSource("prefetch.async_prefetch", args[0]);
      CheckHandle<PrefetchAsyncContextType>("prefetch.async_prefetch",
                                            "ctx to be a "
                                            "PrefetchAsyncContext",
                                            args[1], "prefetch.make_context");
      return GetAsyncEventType();
    });

// ============================================================================
// prefetch.session — project the async session bound to a context
// ============================================================================

REGISTER_OP("prefetch.session")
    .set_description(
        "Project the asynchronous session (AsyncSessionType) bound to an async-prefetch "
        "context. Pure projection — emits no work; the session is the second operand of "
        "prefetch.wait.")
    .set_op_category("PrefetchOp")
    // AIV-only: TPREFETCH_ASYNC drives its SDMA tmpBuf from a Vec(UB) scratch
    // tile held inside PrefetchAsyncContext (pto-isa static_asserts
    // ScratchTile::Loc == TileType::Vec), and UB lives on the vector core.
    // Without this, these ops carry no tile operand, fall through to SHARED,
    // and ExpandMixedKernel duplicates them onto the cube lane too.
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("ctx", "An async-prefetch context (PrefetchAsyncContextType)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      CheckArity("prefetch.session", "a PrefetchAsyncContext", 1, args, kwargs);
      CheckHandle<PrefetchAsyncContextType>("prefetch.session", "ctx to be a PrefetchAsyncContext", args[0],
                                            "prefetch.make_context");
      return GetAsyncSessionType();
    });

// ============================================================================
// prefetch.wait — block until an async prefetch event completes
// ============================================================================

REGISTER_OP("prefetch.wait")
    .set_description(
        "Wait for an asynchronous prefetch event to complete within its session and yield "
        "a BOOL done flag. Use this before the hot loop that consumes the prefetched region "
        "so the data is resident in L2.")
    .set_op_category("PrefetchOp")
    // AIV-only: TPREFETCH_ASYNC drives its SDMA tmpBuf from a Vec(UB) scratch
    // tile held inside PrefetchAsyncContext (pto-isa static_asserts
    // ScratchTile::Loc == TileType::Vec), and UB lives on the vector core.
    // Without this, these ops carry no tile operand, fall through to SHARED,
    // and ExpandMixedKernel duplicates them onto the cube lane too.
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("event", "A completion event (AsyncEventType) from prefetch.async_prefetch")
    .add_argument("session", "The matching async session (AsyncSessionType) from prefetch.session")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      CheckArity("prefetch.wait", "event, session", 2, args, kwargs);
      CheckHandle<AsyncEventType>("prefetch.wait", "event to be an AsyncEvent", args[0],
                                  "prefetch.async_prefetch");
      CheckHandle<AsyncSessionType>("prefetch.wait", "session to be an AsyncSession", args[1],
                                    "prefetch.session");
      return std::make_shared<ScalarType>(DataType::BOOL);
    });

}  // namespace ir
}  // namespace pypto
