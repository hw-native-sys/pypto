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
 * @file put_async.cpp
 * @brief InCore asynchronous cross-rank write - ``pld.tensor.put_async`` and
 *        its SDMA session / event-wait companions.
 *
 * Where ``pld.tensor.put`` blocks the AIV core behind a per-transfer pipe drain,
 * this family issues the transfer on the SDMA engine and hands back an
 * :class:`AsyncEventType` handle, so local compute can overlap the transfer and
 * a later explicit wait drains it::
 *
 *     sess = pld.system.async_session()
 *     evt  = pld.tensor.put_async(win, peer, chunk, session=sess)
 *     ...                                   # local compute overlaps the SDMA transfer
 *     done = pld.system.wait_async_event(evt, session=sess)
 *     # a cross-rank notify that publishes this data MUST follow the wait
 *
 * IR signatures::
 *
 *     pld.system.async_session(*, sync_id: int, block_bytes: int) -> AsyncSession
 *     pld.tensor.put_async(dst, peer, src, session) -> AsyncEvent
 *     pld.tensor.put_async(dst, peer, src, session,
 *                          dst_offsets, src_offsets, shape) -> AsyncEvent
 *     pld.system.wait_async_event(event, session) -> Scalar(BOOL)
 *
 * and the internal tile-level forms that ``ConvertTensorToTileOps`` lowers to,
 * which carry the hidden UB scratch buffer::
 *
 *     pld.tile.async_session(scratch, *, sync_id, block_bytes) -> AsyncSession
 *     pld.tile.put_async(dst, peer, src, session
 *                        [, dst_offsets, src_offsets, shape]) -> AsyncEvent
 *     pld.tile.wait_async_event(event, session, scratch) -> Scalar(BOOL)
 *
 * Why the scratch is an operand of *both* the session build and the wait: pto-isa
 * ``BuildSdmaSession`` stores ``session.tmpBufAddr = tmpBuf.addr`` and
 * ``AsyncEvent::Wait -> SdmaWaitEvent`` reads back through it, so the 256 B Vec(UB)
 * scratch must stay live from the session build through the *last* wait. Its only
 * other IR use is the build, so an allocator that ended its live range there would
 * be free to hand that address to a compute tile issued between issue and wait -
 * corrupting the completion word on hardware only, with no simulator signal.
 * Threading it into the wait pins the live range across the whole async window.
 *
 * Unlike ``pld.tile.put`` there is **no staging tile**: PTOAS ``TPutAsyncOp`` takes
 * only ``(dst, src, session)`` and no ``buf(...)`` operand group, because SDMA moves
 * GM->GM directly rather than bouncing through UB.
 *
 * Restrictions in v1 (all enforced by the deducers below):
 *
 * * ``dst`` and ``src`` regions must be **static, flat-contiguous and logically
 *   1-D** - PTOAS's ``verifyAsyncFlatContiguous1DGMViewLike`` and pto-isa's
 *   ``TPutAsyncIsFlatContiguous1D`` both require it. The scattered
 *   ``[r_route, 0]``-offset combine push is therefore *not* covered by this op.
 * * No ``atomic`` kwarg - ``pto.comm.tput_async`` has no atomicType operand at all.
 * * ``sync_id`` must be <= 7 and ``block_bytes`` positive: ``BuildSdmaSession``
 *   returns false outside those ranges, PTOAS's op discards that bool, and
 *   ``__sdma_put_async`` does not check ``session.valid`` before posting. Both are
 *   compile-time constants, so the failure is caught here where it cannot reach
 *   the device.
 */

#include <any>
#include <cstddef>
#include <cstdint>
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

using Kwargs = std::vector<std::pair<std::string, std::any>>;

/// pto-isa `BuildSdmaSession` rejects syncId > 7 (the SDMA sync-flag id is 3 bits).
constexpr int kMaxSyncId = 7;

/// Default SDMA chunking granularity, in bytes. PTOAS's lowering hardcodes 32 KB
/// when the attr is absent (PTOToEmitC.cpp), while pto-isa's own
/// `kDefaultSdmaBlockBytes` is 1 MB and the prefetch context uses 64 MB. For the
/// multi-hundred-KB chunk pushes this op targets, 32 KB would be an accident
/// rather than a choice, so PyPTO always emits the attr explicitly.
constexpr int64_t kDefaultBlockBytes = int64_t{1024} * 1024;

void CheckNonNull(const std::vector<ExprPtr>& args, const std::string& op_name) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " positional argument #" << i << " must not be null";
  }
}

/// Validate the session-build attrs. Both are compile-time constants, so an
/// out-of-range value is a build-time error here rather than a silently invalid
/// session on device (see the file header).
void ValidateSessionAttrs(const Kwargs& kwargs, const std::string& op_name) {
  const int sync_id = GetIntKwarg(kwargs, "sync_id", 0);
  CHECK(sync_id >= 0 && sync_id <= kMaxSyncId)
      << op_name << " sync_id must be in [0, " << kMaxSyncId << "], but got " << sync_id
      << ": pto-isa BuildSdmaSession rejects a larger id and returns an invalid session that "
         "PTOAS's build_async_session op silently discards.";
  // Omitting the attr means "use PyPTO's documented default", not "let PTOAS
  // pick" — PTOAS would silently apply 32 KB. Validate the effective value.
  const int64_t block_bytes =
      static_cast<int64_t>(GetIntKwarg(kwargs, "block_bytes", static_cast<int>(kDefaultBlockBytes)));
  CHECK(block_bytes > 0) << op_name << " block_bytes must be positive, got " << block_bytes
                         << ": it is the SDMA chunking granularity passed to BuildSdmaSession.";
}

/// Shape of a transfer operand: the DistributedTensor window shape, or the plain
/// Tensor shape for a local source.
std::vector<ExprPtr> OperandShape(const ExprPtr& arg, const std::string& op_name, const std::string& role) {
  auto tensor_type = AsTensorTypeLike(arg->GetType());
  CHECK(tensor_type) << op_name << " " << role << " must be a Tensor or DistributedTensor, but got "
                     << arg->GetType()->TypeName();
  return tensor_type->shape_;
}

/**
 * @brief Shared contract for both the tensor- and tile-level async put.
 *
 * `session_index` is where the AsyncSession handle sits, and `region_base` is
 * the index of `dst_offsets` (or 0 when this is a full-slice put).
 */
void ValidatePutAsyncContract(const std::vector<ExprPtr>& args, size_t session_index, size_t region_base,
                              const Kwargs& kwargs, const std::string& op_name) {
  CheckNonNull(args, op_name);
  comm_op::ValidateNoAtomic(kwargs, op_name);

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dst_type) << op_name << " dst must be a window-bound DistributedTensor, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(As<ScalarType>(args[1]->GetType()))
      << op_name << " peer must be an integer ScalarType, but got " << args[1]->GetType()->TypeName();
  auto src_type = AsTensorTypeLike(args[2]->GetType());
  CHECK(src_type) << op_name << " src must be a Tensor or DistributedTensor, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must share an element type, but got " << dst_type->dtype_.ToString()
      << " and " << src_type->dtype_.ToString();
  CHECK(IsA<AsyncSessionType>(args[session_index]->GetType()))
      << op_name << " expects session to be an AsyncSession (output of pld.system.async_session), but got "
      << args[session_index]->GetType()->TypeName();

  CHECK(dst_type->shape_.size() == src_type->shape_.size())
      << op_name << " dst rank (" << dst_type->shape_.size() << ") must match src rank ("
      << src_type->shape_.size() << ")";

  const bool full_slice = (region_base == 0);
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (!full_slice) {
    transfer_shape =
        comm_op::ValidateRegionArgs(args, region_base, dst_type->shape_, src_type->shape_, op_name);
  } else {
    comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, op_name,
                                           /*require_same_shape=*/true);
  }

  // The 1-D-static constraint applies to what actually moves. For a full-slice
  // put that is both windows; for a subregion put it is the explicit shape, and
  // the surrounding window may legitimately be 2-D.
  comm_op::ValidateAsyncFlatContiguous1D(transfer_shape, op_name, "transfer");
  if (full_slice) {
    comm_op::ValidateAsyncFlatContiguous1D(OperandShape(args[2], op_name, "src"), op_name, "src");
  }
}

}  // namespace

// ============================================================================
// pld.system.async_session / pld.tile.async_session - build the SDMA session
// ============================================================================

// AIV-only, for the same reason prefetch.make_context is: the session's tmpBuf
// is a Vec(UB) scratch tile (pto-isa static_asserts ScratchTile::Loc ==
// TileType::Vec) and UB lives on the vector core. Without this the op carries no
// tile operand at the tensor level, falls through to SHARED, and
// ExpandMixedKernel duplicates it onto the cube lane too.
REGISTER_OP("pld.system.async_session")
    .set_description(
        "Build an SDMA async session for pld.tensor.put_async. The runtime owns and injects the "
        "workspace backing the SDMA path through a hidden codegen parameter, so this operation has "
        "no user operand; ConvertTensorToTileOps materializes the required Vec(UB) scratch buffer "
        "and lowers this to pld.tile.async_session. Build one session per kernel and pass it to "
        "every put_async and wait_async_event.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_argument()
    .set_attr<int>("sync_id")
    .set_attr<int>("block_bytes")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.empty()) << "pld.system.async_session takes no positional arguments, but got "
                          << args.size();
      ValidateSessionAttrs(kwargs, "pld.system.async_session");
      return GetAsyncSessionType();
    });

REGISTER_OP("pld.tile.async_session")
    .set_description(
        "Tile-level SDMA session build (internal; produced by ConvertTensorToTileOps from "
        "pld.system.async_session). Carries the explicit Vec(UB) scratch buffer that pto-isa "
        "BuildSdmaSession bounces descriptor and completion words through, so PyPTO's memory "
        "allocator assigns its UB address before PTO codegen.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("scratch", "Vec(UB) scratch buffer (>= 8 bytes; 256 B by convention)")
    .set_attr<int>("sync_id")
    .set_attr<int>("block_bytes")
    .no_memory_spec()
    // The scratch is written by the session build (descriptor staging), and read
    // back by every wait that shares this session.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.size() == 1) << "pld.tile.async_session requires exactly 1 positional argument "
                                 "(scratch), but got "
                              << args.size();
      CheckNonNull(args, "pld.tile.async_session");
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "pld.tile.async_session scratch must be a TileType, but got "
                       << args[0]->GetType()->TypeName();
      ValidateSessionAttrs(kwargs, "pld.tile.async_session");
      return GetAsyncSessionType();
    });

// ============================================================================
// pld.tensor.put_async / pld.tile.put_async - issue the transfer, return event
// ============================================================================

REGISTER_OP("pld.tensor.put_async")
    .set_description(
        "Cross-rank asynchronous put: issue an SDMA write of the local source `src` into the "
        "`peer` rank's slice of the window-bound DistributedTensor `dst`, and return an AsyncEvent "
        "completion handle without blocking. Local compute may overlap the transfer; the event "
        "must be drained with pld.system.wait_async_event before the kernel ends or before any "
        "cross-rank notify that publishes the transferred data. Regions must be static, "
        "flat-contiguous and logically 1-D. No atomic combine - use pld.tensor.put for that.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local source - DistributedTensor (window-bound) or plain Tensor")
    .add_argument("session", "SDMA session (AsyncSessionType) from pld.system.async_session")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's dst slice; present only in "
                  "the subregion form (all three region args supplied together)")
    .add_argument("src_offsets", "Optional per-dim offsets (MakeTuple) into the local src")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); must be static and 1-D")
    .no_memory_spec()
    // A plain SDMA push overwrites the region it lands on. There is no atomic
    // form, so unlike pld.tensor.put this is unconditionally a Write.
    .set_arg_effect(0, ArgEffect::Write)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.size() == 4 || args.size() == 7)
          << "pld.tensor.put_async requires 4 positional arguments (dst, peer, src, session) or 7 "
             "(dst, peer, src, session, dst_offsets, src_offsets, shape), but got "
          << args.size();
      ValidatePutAsyncContract(args, /*session_index=*/3, /*region_base=*/args.size() == 7 ? 4 : 0, kwargs,
                               "pld.tensor.put_async");
      return GetAsyncEventType();
    });

REGISTER_OP("pld.tile.put_async")
    .set_description(
        "Tile-level asynchronous cross-rank put (internal; produced by ConvertTensorToTileOps from "
        "pld.tensor.put_async). Unlike pld.tile.put it carries no staging tile: PTOAS "
        "pto.comm.tput_async moves GM->GM on the SDMA engine and takes no buf(...) operand.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local source - DistributedTensor (window-bound) or plain Tensor")
    .add_argument("session", "SDMA session (AsyncSessionType) from pld.tile.async_session")
    .add_argument("dst_offsets", "Optional per-dim offsets (MakeTuple) into the peer's dst slice")
    .add_argument("src_offsets", "Optional per-dim offsets (MakeTuple) into the local src")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); must be static and 1-D")
    .no_memory_spec()
    .set_arg_effect(0, ArgEffect::Write)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.size() == 4 || args.size() == 7)
          << "pld.tile.put_async requires 4 positional arguments (dst, peer, src, session) or 7 "
             "(dst, peer, src, session, dst_offsets, src_offsets, shape), but got "
          << args.size();
      ValidatePutAsyncContract(args, /*session_index=*/3, /*region_base=*/args.size() == 7 ? 4 : 0, kwargs,
                               "pld.tile.put_async");
      return GetAsyncEventType();
    });

// ============================================================================
// pld.system.wait_async_event / pld.tile.wait_async_event - drain the event
// ============================================================================

REGISTER_OP("pld.system.wait_async_event")
    .set_description(
        "Block until an async transfer event completes within its session, and yield a BOOL done "
        "flag. This is the acquire half of put_async: the GM release fence and the peer-region "
        "cache invalidate for the transfer are emitted after this wait, not after the issue, so a "
        "cross-rank notify that publishes the transferred data must follow it.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("event", "A completion event (AsyncEventType) from pld.tensor.put_async")
    .add_argument("session", "The matching SDMA session (AsyncSessionType)")
    .no_memory_spec()
    .no_arg_writes()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.size() == 2) << "pld.system.wait_async_event requires exactly 2 positional arguments "
                                 "(event, session), but got "
                              << args.size();
      CHECK(kwargs.empty()) << "pld.system.wait_async_event takes no kwargs, but got " << kwargs.size();
      CheckNonNull(args, "pld.system.wait_async_event");
      CHECK(IsA<AsyncEventType>(args[0]->GetType()))
          << "pld.system.wait_async_event expects event to be an AsyncEvent (output of "
             "pld.tensor.put_async), but got "
          << args[0]->GetType()->TypeName();
      CHECK(IsA<AsyncSessionType>(args[1]->GetType()))
          << "pld.system.wait_async_event expects session to be an AsyncSession (output of "
             "pld.system.async_session), but got "
          << args[1]->GetType()->TypeName();
      return std::make_shared<ScalarType>(DataType::BOOL);
    });

REGISTER_OP("pld.tile.wait_async_event")
    .set_description(
        "Tile-level event wait (internal; produced by ConvertTensorToTileOps from "
        "pld.system.wait_async_event). Carries the session's Vec(UB) scratch buffer as an explicit "
        "operand so its live range spans the whole async window: pto-isa reads the completion word "
        "back through session.tmpBufAddr, which points into that scratch.")
    .set_op_category("DistributedOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("event", "A completion event (AsyncEventType) from pld.tile.put_async")
    .add_argument("session", "The matching SDMA session (AsyncSessionType)")
    .add_argument("scratch", "The session's Vec(UB) scratch buffer (liveness pin; see file header)")
    .no_memory_spec()
    // The wait reads the completion word back through the scratch.
    .set_arg_effect(2, ArgEffect::ReadWrite)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) -> TypePtr {
      CHECK(args.size() == 3) << "pld.tile.wait_async_event requires exactly 3 positional arguments "
                                 "(event, session, scratch), but got "
                              << args.size();
      CHECK(kwargs.empty()) << "pld.tile.wait_async_event takes no kwargs, but got " << kwargs.size();
      CheckNonNull(args, "pld.tile.wait_async_event");
      CHECK(IsA<AsyncEventType>(args[0]->GetType()))
          << "pld.tile.wait_async_event expects event to be an AsyncEvent, but got "
          << args[0]->GetType()->TypeName();
      CHECK(IsA<AsyncSessionType>(args[1]->GetType()))
          << "pld.tile.wait_async_event expects session to be an AsyncSession, but got "
          << args[1]->GetType()->TypeName();
      CHECK(As<TileType>(args[2]->GetType()))
          << "pld.tile.wait_async_event expects scratch to be a TileType, but got "
          << args[2]->GetType()->TypeName();
      return std::make_shared<ScalarType>(DataType::BOOL);
    });

}  // namespace ir
}  // namespace pypto
