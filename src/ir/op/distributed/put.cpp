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
 * @file put.cpp
 * @brief Distributed cross-rank tensor write - ``pld.tensor.put``.
 *
 * Synchronously writes the local window-bound :class:`DistributedTensorType`
 * ``src`` into the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``dst`` (HCCL TPUT). Both operands live at
 * the **tensor** (GM) level, so the user-facing op is a sibling of
 * ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` rather than the
 * tile-producing ``pld.tile.remote_load``.
 *
 * The VEC staging tile that TPUT bounces through is still intentionally hidden
 * from the public DSL surface. It is materialized by ``ConvertTensorToTileOps``
 * as ``tile.create`` and threaded into the internal ``pld.tile.put`` op so the
 * normal memory allocator assigns its UB address before PTO codegen.
 *
 * IR signatures::
 *
 *     pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
 *     pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape,
 *                    *, atomic: int) -> Unknown
 *
 * Optional ``chunk_rows`` / ``chunk_cols`` int attrs size the VEC staging tile
 * to a sub-tile of the flattened transfer; the optional ``pipeline`` bool attr
 * requests ping-pong double-buffering and is lowered by
 * ``ConvertTensorToTileOps`` into a *second* ``tile.create`` staging tile threaded
 * into ``pld.tile.put`` (``pipeline=True`` requires both chunk dims set). The
 * tile-level form therefore carries an optional second ``stage`` operand::
 *
 *     pld.tile.put(dst, peer, src, stage[, stage2]
 *                  [, dst_offsets, src_offsets, shape], *, atomic: int) -> Unknown
 *
 * The ``atomic`` integer is the underlying value of :enum:`AtomicType`
 * (``include/pypto/ir/comm.h``); the deducer validates the int against the
 * enum range so codegen can cast back without a separate guard. The DSL
 * surface (``pld.tensor.put`` in
 * ``python/pypto/language/distributed/op/tensor_ops.py``) accepts the typed
 * Python enum and packs ``int(atomic)`` into the kwarg. Side-effect-only -
 * the op produces :class:`UnknownType`, mirroring ``pld.system.notify`` /
 * ``pld.system.wait``.
 *
 * Verifier:
 *
 * * ``dst`` must have :class:`DistributedTensorType` - the destination of a
 *   cross-rank write must be window-bound (the remote peer needs a window slot
 *   to receive into).
 * * ``src`` accepts either :class:`DistributedTensorType` *or* plain
 *   :class:`TensorType` (matched via :func:`AsTensorTypeLike`). The TPUT
 *   primitive only requires src to be a readable local GM region; it does not
 *   need a window. This lets kernels TPUT directly from host-backed inputs
 *   without first staging through a window buffer.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``dst`` and ``src`` must share element type, rank, and positive dimensions
 *   (positivity checked on static dims; dims may be dynamic — a dynamic transfer
 *   extent then requires a static ``chunk_rows`` / ``chunk_cols``).
 * * Full-slice puts require matching ``dst`` / ``src`` shape (by value when
 *   static, structurally when dynamic). Subregion puts may use different per-rank
 *   slice extents; ``dst_offsets``, ``src_offsets``, and ``shape`` are
 *   rank-matched tuples (``shape`` may be dynamic) and are provided together.
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
#include "src/ir/op/distributed/comm_op_utils.h"

namespace pypto {
namespace ir {

namespace {

void ValidatePutContract(const ExprPtr& dst, const ExprPtr& peer, const ExprPtr& src,
                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                         const std::string& op_name, bool require_same_shape) {
  CHECK(dst) << op_name << " dst argument must not be null";
  CHECK(peer) << op_name << " peer argument must not be null";
  CHECK(src) << op_name << " src argument must not be null";

  auto dst_type = As<DistributedTensorType>(dst->GetType());
  CHECK(dst_type) << op_name << " dst must be a DistributedTensor (window-bound), got "
                  << dst->GetType()->TypeName();

  CHECK(IsA<ScalarType>(peer->GetType()))
      << op_name << " peer must be a scalar (rank index), got " << peer->GetType()->TypeName();

  // src accepts either DistributedTensor or plain Tensor: TPUT only needs a
  // readable local GM region for the source, no window membership required.
  auto src_type = AsTensorTypeLike(src->GetType());
  CHECK(src_type) << op_name << " src must be a Tensor or DistributedTensor, got "
                  << src->GetType()->TypeName();

  // TPUT contract: dst and src always describe same-rank window slices with
  // identical element type and static positive extents (and, for full-slice,
  // identical shapes). Subregion puts narrow via the separate `shape` arg.
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must have the same element type, got dst " << dst->GetType()->TypeName()
      << " vs src " << src->GetType()->TypeName();
  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, op_name, require_same_shape);

  auto atomic_value = GetRequiredKwarg<int>(kwargs, "atomic", op_name);
  CHECK(atomic_value == static_cast<int>(AtomicType::kNone) ||
        atomic_value == static_cast<int>(AtomicType::kAdd))
      << op_name << " atomic must be AtomicType.None_ or AtomicType.Add (got int " << atomic_value << ")";
}

TypePtr DeducePutType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6)
      << "pld.tensor.put requires 3 positional arguments (dst, peer, src) or 6 "
         "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.put positional argument #" << i << " must not be null";
  }
  ValidatePutContract(args[0], args[1], args[2], kwargs, "pld.tensor.put", args.size() == 3);
  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  // Transfer extent = the explicit subregion shape, or the full window shape for
  // a full-slice put. Either may be dynamic, which then requires a static chunk.
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (args.size() == 6) {
    auto src_type = AsTensorTypeLike(args[2]->GetType());
    transfer_shape =
        comm_op::ValidateRegionArgs(args, 3, dst_type->shape_, src_type->shape_, "pld.tensor.put");
  }
  comm_op::ValidateChunkNonNegative(kwargs, "pld.tensor.put");
  comm_op::ValidateDynamicTransferHasChunk(transfer_shape, kwargs, "pld.tensor.put");
  comm_op::ValidatePipelineHasChunk(kwargs, "pld.tensor.put");
  // Side-effect-only: no SSA result for downstream consumers.
  return GetUnknownType();
}

TypePtr DeducePutTileType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  const size_t n = args.size();
  CHECK(n == 4 || n == 5 || n == 7 || n == 8)
      << "pld.tile.put requires 4/5 (single/double stage, full-slice) or 7/8 (single/double stage, "
         "subregion) positional arguments (dst, peer, src, stage[, stage2][, dst_offsets, "
         "src_offsets, shape]), but got "
      << n;
  for (size_t i = 0; i < n; ++i) {
    CHECK(args[i]) << "pld.tile.put positional argument #" << i << " must not be null";
  }
  const bool has_stage2 = (n == 5 || n == 8);
  const bool has_region = (n == 7 || n == 8);
  const size_t region_base = 4 + (has_stage2 ? 1 : 0);
  ValidatePutContract(args[0], args[1], args[2], kwargs, "pld.tile.put", !has_region);

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  auto stage_type = comm_op::ValidateStageTile(args[3], dst_type->dtype_, "pld.tile.put stage");
  TileTypePtr stage2_type;
  if (has_stage2) {
    stage2_type = comm_op::ValidateStageTile(args[4], dst_type->dtype_, "pld.tile.put stage2");
    // pto-isa ping/pong contract: the two staging tiles must be identical in
    // shape (same dtype already enforced by ValidateStageTile against dst).
    CHECK(AreExprsEqual(stage_type->shape_[0], stage2_type->shape_[0]) &&
          AreExprsEqual(stage_type->shape_[1], stage2_type->shape_[1]))
        << "pld.tile.put ping/pong staging tiles must have identical shape";
  }

  auto src_type = AsTensorTypeLike(args[2]->GetType());
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (has_region) {
    transfer_shape =
        comm_op::ValidateRegionArgs(args, region_base, dst_type->shape_, src_type->shape_, "pld.tile.put");
  }

  // The explicit stage tile (allocated by ConvertTensorToTileOps) is the 2-D VEC
  // bounce buffer that pto-isa TPUT streams the transfer through. TPUT reads the
  // full transfer extent from the partition views and uses the stage tile's
  // rows/cols as its 2-D sliding chunk size, so the stage may be SMALLER than
  // the transfer (a single chunk). It must only not EXCEED the transfer in
  // either flattened dim (rows = prod(leading dims), cols = innermost dim). A
  // second stage (ping/pong double-buffering) shares the same extent.
  comm_op::ValidateStageFitsTransfer(stage_type->shape_, transfer_shape, args[3]->span_, "pld.tile.put");
  if (has_stage2) {
    comm_op::ValidateStageFitsTransfer(stage2_type->shape_, transfer_shape, args[4]->span_, "pld.tile.put");
  }

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.put - synchronous cross-rank bulk write into a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.put")
    .set_description(
        "Cross-rank put: synchronously write local source `src` "
        "(window-bound DistributedTensor or plain Tensor) into the `peer` rank's slice of "
        "the window-bound DistributedTensor `dst`. `atomic` selects plain-store vs atomic-add "
        "combine semantics. Lowered by ConvertTensorToTileOps to a `tile.create`-allocated "
        "VEC staging tile plus a `pld.tile.put` call, so the staging tile flows through "
        "PyPTO's memory allocator (required at --pto-level=level3). Optional `chunk_rows` / "
        "`chunk_cols` (0 = full) size that staging tile to a sub-tile of the flattened "
        "transfer [rows, cols] extent; pto-isa TPUT then auto-chunks the full transfer "
        "through it, so transfers larger than UB no longer need to fit in one staging tile.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src",
                  "Local source — DistributedTensor (window-bound) or plain Tensor (same dtype as dst)")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's dst slice; present only in the "
                  "subregion form (all three region args supplied together)")
    .add_argument(
        "src_offsets",
        "Optional per-dim offsets (MakeTuple) into the local src; present only in the subregion form")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); present only in the subregion form")
    .set_attr<int>("atomic")
    .set_attr<int>("chunk_rows")
    .set_attr<int>("chunk_cols")
    .set_attr<bool>("pipeline")
    .no_memory_spec()
    .f_deduce_type(DeducePutType);

// ============================================================================
// pld.tile.put - tile-level form with explicit VEC staging tile (post-conversion)
// ============================================================================

REGISTER_OP("pld.tile.put")
    .set_description(
        "Tile-level form of pld.tensor.put with an explicit VEC staging tile. "
        "Created by ConvertTensorToTileOps; not user-facing.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src",
                  "Local source — DistributedTensor (window-bound) or plain Tensor (same dtype as dst)")
    .add_argument("stage", "VEC staging TileType (rows x cols <= flattened transfer; auto-chunked by TPUT)")
    .add_argument("stage2",
                  "Optional second VEC staging TileType (same shape as stage); when present, TPUT "
                  "ping-pong double-buffers the chunked transfer through both tiles")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's dst slice; present only in the "
                  "subregion form (all three region args supplied together)")
    .add_argument(
        "src_offsets",
        "Optional per-dim offsets (MakeTuple) into the local src; present only in the subregion form")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); present only in the subregion form")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutTileType);

}  // namespace ir
}  // namespace pypto
