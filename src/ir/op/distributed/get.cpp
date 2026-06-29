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
 * @file get.cpp
 * @brief Distributed cross-rank tensor read - ``pld.tensor.get``.
 *
 * Synchronously reads the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``src`` into the local window-bound
 * :class:`DistributedTensorType` ``dst`` (TGET). Semantically this is the
 * tensor-level bulk form of ``remote_load + store``: it copies remote GM into
 * local GM through a VEC staging tile. ``ConvertTensorToTileOps`` materializes
 * that stage as ``tile.create`` + the internal ``pld.tile.get`` op, mirroring
 * ``pld.tensor.put``.
 *
 * IR signatures::
 *
 *     pld.tensor.get(dst, peer, src) -> Unknown
 *     pld.tensor.get(dst, peer, src, dst_offsets, src_offsets, shape)
 *         -> Unknown
 *
 * Optional ``chunk_rows`` / ``chunk_cols`` int attrs size the VEC staging tile;
 * the optional ``pipeline`` bool attr requests ping-pong double-buffering
 * and is lowered by ``ConvertTensorToTileOps`` into a *second* ``tile.create``
 * staging tile threaded into ``pld.tile.get`` (``pipeline=True`` requires both
 * chunk dims set). The tile-level form therefore carries an optional second ``stage``::
 *
 *     pld.tile.get(dst, peer, src, stage[, stage2]
 *                  [, dst_offsets, src_offsets, shape]) -> Unknown
 *
 * Side-effect-only: the op produces :class:`UnknownType`, mirroring
 * ``pld.tensor.put`` and the sync primitives.
 *
 * Verifier:
 *
 * * ``dst`` accepts either :class:`DistributedTensorType` *or* plain
 *   :class:`TensorType` (matched via :func:`AsTensorTypeLike`). The TGET
 *   primitive only requires dst to be a writable local GM region; it does not
 *   need a window. This lets kernels TGET directly into host-backed output
 *   tensors without first allocating a window buffer.
 * * ``src`` must have :class:`DistributedTensorType` - the source of a
 *   cross-rank read must be window-bound (the remote peer needs a window
 *   slot to read from).
 * * ``peer`` must be a :class:`ScalarType` expression (rank index).
 * * ``dst`` and ``src`` must share element type, rank, and positive dimensions
 *   (positivity checked on static dims; dims may be dynamic — a dynamic transfer
 *   extent then requires a static ``chunk_rows`` / ``chunk_cols``).
 * * Full-slice gets require matching ``dst`` / ``src`` shape (by value when
 *   static, structurally when dynamic). Subregion gets may use different per-rank
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
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "src/ir/op/distributed/comm_op_utils.h"

namespace pypto {
namespace ir {

namespace {

void ValidateGetContract(const ExprPtr& dst, const ExprPtr& peer, const ExprPtr& src,
                         const std::string& op_name, bool require_same_shape) {
  CHECK(dst) << op_name << " dst argument must not be null";
  CHECK(peer) << op_name << " peer argument must not be null";
  CHECK(src) << op_name << " src argument must not be null";

  auto dst_type = AsTensorTypeLike(dst->GetType());
  CHECK(dst_type) << op_name << " dst must be a Tensor or DistributedTensor, got "
                  << dst->GetType()->TypeName();

  CHECK(IsA<ScalarType>(peer->GetType()))
      << op_name << " peer must be a scalar (rank index), got " << peer->GetType()->TypeName();

  auto src_type = As<DistributedTensorType>(src->GetType());
  CHECK(src_type) << op_name << " src must be a DistributedTensor (window-bound), got "
                  << src->GetType()->TypeName();

  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must have the same element type, got dst " << dst->GetType()->TypeName()
      << " vs src " << src->GetType()->TypeName();
  comm_op::ValidateTransferShapeContract(dst_type->shape_, src_type->shape_, op_name, require_same_shape);
}

TypePtr DeduceGetType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6)
      << "pld.tensor.get requires 3 positional arguments (dst, peer, src) or 6 "
         "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
      << args.size();
  // Optional chunk_rows / chunk_cols attrs are validated by the framework's
  // ValidateKwargs against the registered attr set; no manual empty() guard.
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.get positional argument #" << i << " must not be null";
  }

  ValidateGetContract(args[0], args[1], args[2], "pld.tensor.get", args.size() == 3);
  auto dst_type = AsTensorTypeLike(args[0]->GetType());
  // Transfer extent = the explicit subregion shape, or the full dst window shape
  // for a full-slice get. Either may be dynamic, which then requires a static chunk.
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (args.size() == 6) {
    auto src_type = As<DistributedTensorType>(args[2]->GetType());
    transfer_shape =
        comm_op::ValidateRegionArgs(args, 3, dst_type->shape_, src_type->shape_, "pld.tensor.get");
  }
  comm_op::ValidateChunkNonNegative(kwargs, "pld.tensor.get");
  comm_op::ValidateDynamicTransferHasChunk(transfer_shape, kwargs, "pld.tensor.get");
  comm_op::ValidatePipelineHasChunk(kwargs, "pld.tensor.get");

  return GetUnknownType();
}

TypePtr DeduceGetTileType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  const size_t n = args.size();
  CHECK(n == 4 || n == 5 || n == 7 || n == 8)
      << "pld.tile.get requires 4/5 (single/double stage, full-slice) or 7/8 (single/double stage, "
         "subregion) positional arguments (dst, peer, src, stage[, stage2][, dst_offsets, "
         "src_offsets, shape]), but got "
      << n;
  CHECK(kwargs.empty()) << "pld.tile.get does not accept keyword attributes";
  for (size_t i = 0; i < n; ++i) {
    CHECK(args[i]) << "pld.tile.get positional argument #" << i << " must not be null";
  }
  const bool has_stage2 = (n == 5 || n == 8);
  const bool has_region = (n == 7 || n == 8);
  const size_t region_base = 4 + (has_stage2 ? 1 : 0);
  ValidateGetContract(args[0], args[1], args[2], "pld.tile.get", !has_region);

  auto dst_type = AsTensorTypeLike(args[0]->GetType());
  auto stage_type = comm_op::ValidateStageTile(args[3], dst_type->dtype_, "pld.tile.get stage");
  TileTypePtr stage2_type;
  if (has_stage2) {
    stage2_type = comm_op::ValidateStageTile(args[4], dst_type->dtype_, "pld.tile.get stage2");
    // pto-isa ping/pong contract: identical-shape staging tiles.
    CHECK(AreExprsEqual(stage_type->shape_[0], stage2_type->shape_[0]) &&
          AreExprsEqual(stage_type->shape_[1], stage2_type->shape_[1]))
        << "pld.tile.get ping/pong staging tiles must have identical shape";
  }

  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (has_region) {
    transfer_shape =
        comm_op::ValidateRegionArgs(args, region_base, dst_type->shape_, src_type->shape_, "pld.tile.get");
  }

  // The explicit stage tile is the 2-D VEC bounce buffer that pto-isa TGET
  // streams the transfer through; it may be smaller than the transfer (a single
  // chunk) but must not exceed the flattened [rows, cols] transfer extent. A
  // second stage (ping/pong double-buffering) shares the same extent.
  comm_op::ValidateStageFitsTransfer(stage_type->shape_, transfer_shape, args[0]->span_, args[3]->span_,
                                     "pld.tile.get");
  if (has_stage2) {
    comm_op::ValidateStageFitsTransfer(stage2_type->shape_, transfer_shape, args[0]->span_, args[4]->span_,
                                       "pld.tile.get");
  }

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.get - synchronous cross-rank bulk read from a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.get")
    .set_description(
        "Cross-rank get: synchronously read the `peer` rank's slice of the window-bound "
        "DistributedTensor `src` into the local destination `dst` "
        "(window-bound DistributedTensor or plain Tensor). "
        "Semantically equivalent to remote_load + store. Supports full-slice and explicit "
        "subregion forms. ConvertTensorToTileOps lowers this to tile.create + pld.tile.get; "
        "PTO emission then produces CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + "
        "partition_view (src) + partition_view (dst) + explicit VEC staging tile + TGET. "
        "Optional `chunk_rows` / `chunk_cols` (0 = full) size that staging tile to a sub-tile "
        "of the flattened transfer [rows, cols] extent; pto-isa TGET then auto-chunks the full "
        "transfer through it, so transfers larger than UB no longer need to fit in one tile.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local destination — DistributedTensor (window-bound) or plain Tensor")
    .add_argument("peer", "Peer rank index (ScalarType)")
    .add_argument("src", "Remote (peer) window-bound DistributedTensor source (same dtype as dst)")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the local dst; present only in the "
                  "subregion form (all three region args supplied together)")
    .add_argument("src_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's src slice; present only in the "
                  "subregion form")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); present only in the subregion form")
    .set_attr<int>("chunk_rows")
    .set_attr<int>("chunk_cols")
    .set_attr<bool>("pipeline")
    .no_memory_spec()
    .f_deduce_type(DeduceGetType);

// ============================================================================
// pld.tile.get - tile-level form with explicit VEC staging tile (post-conversion)
// ============================================================================

REGISTER_OP("pld.tile.get")
    .set_description(
        "Tile-level form of pld.tensor.get with an explicit VEC staging tile. "
        "Created by ConvertTensorToTileOps; not user-facing.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local destination — DistributedTensor (window-bound) or plain Tensor")
    .add_argument("peer", "Peer rank index (ScalarType)")
    .add_argument("src", "Remote (peer) window-bound DistributedTensor source (same dtype as dst)")
    .add_argument("stage", "VEC staging TileType (rows x cols <= flattened transfer; auto-chunked by TGET)")
    .add_argument("stage2",
                  "Optional second VEC staging TileType (same shape as stage); when present, TGET "
                  "ping-pong double-buffers the chunked transfer through both tiles")
    .add_argument("dst_offsets",
                  "Optional per-dim offsets (MakeTuple) into the local dst; present only in the "
                  "subregion form (all three region args supplied together)")
    .add_argument("src_offsets",
                  "Optional per-dim offsets (MakeTuple) into the peer's src slice; present only in the "
                  "subregion form")
    .add_argument("shape", "Optional per-dim transfer shape (MakeTuple); present only in the subregion form")
    .no_memory_spec()
    .f_deduce_type(DeduceGetTileType);

}  // namespace ir
}  // namespace pypto
