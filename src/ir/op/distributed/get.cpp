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
 * * ``dst`` and ``src`` must share element type, rank, and positive static
 *   dimensions.
 * * Full-slice gets require identical static shape. Subregion gets may use
 *   different per-rank slice extents, but ``dst_offsets``, ``src_offsets``,
 *   and ``shape`` must be rank-matched static tuples and are provided
 *   together.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

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

  const auto& dst_shape = dst_type->shape_;
  const auto& src_shape = src_type->shape_;
  CHECK(!dst_shape.empty()) << op_name << " requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size())
      << op_name << " dst rank (" << dst_shape.size() << ") must match src rank (" << src_shape.size() << ")";
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    auto d = As<ConstInt>(dst_shape[i]);
    auto s = As<ConstInt>(src_shape[i]);
    CHECK(d && s) << op_name << " requires static (compile-time constant) shapes on dst and src; dimension "
                  << i << " is dynamic";
    CHECK(d->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << d->value_;
    CHECK(s->value_ > 0) << op_name << " src shape dimension " << i << " must be positive, got " << s->value_;
    if (require_same_shape) {
      CHECK(d->value_ == s->value_) << op_name << " dst and src must have the same static shape; dimension "
                                    << i << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
    }
  }
}

void ValidateGetRegionArgs(const std::vector<ExprPtr>& args, size_t region_arg_base,
                           const std::vector<ExprPtr>& dst_shape, const std::vector<ExprPtr>& src_shape,
                           const std::string& op_name, std::vector<ExprPtr>* out_transfer_shape = nullptr) {
  auto dst_offsets = As<MakeTuple>(args[region_arg_base]);
  auto src_offsets = As<MakeTuple>(args[region_arg_base + 1]);
  auto transfer_shape = As<MakeTuple>(args[region_arg_base + 2]);
  CHECK(dst_offsets) << op_name << " dst_offsets must be a tuple";
  CHECK(src_offsets) << op_name << " src_offsets must be a tuple";
  CHECK(transfer_shape) << op_name << " shape must be a tuple";
  CHECK(dst_offsets->elements_.size() == dst_shape.size())
      << op_name << " dst_offsets rank must match dst rank";
  CHECK(src_offsets->elements_.size() == src_shape.size())
      << op_name << " src_offsets rank must match src rank";
  CHECK(transfer_shape->elements_.size() == dst_shape.size())
      << op_name << " shape rank must match tensor rank";
  if (out_transfer_shape) {
    *out_transfer_shape = transfer_shape->elements_;
  }

  for (size_t i = 0; i < transfer_shape->elements_.size(); ++i) {
    // The transfer extent may be dynamic (a runtime sub-extent of the fixed
    // window). A dynamic dim requires a static chunk to bound the staging tile
    // (enforced by ValidateDynamicTransferHasChunk in the deducer). The static
    // positivity / bounds checks below only apply when the dim is a ConstInt.
    auto dim = As<ConstInt>(transfer_shape->elements_[i]);
    if (dim) {
      CHECK(dim->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got "
                             << dim->value_;
    }
    auto dst_dim = As<ConstInt>(dst_shape[i]);
    auto src_dim = As<ConstInt>(src_shape[i]);
    INTERNAL_CHECK(dst_dim && src_dim) << op_name << " tensor shapes must be static before region validation";

    if (auto dst_offset = As<ConstInt>(dst_offsets->elements_[i])) {
      CHECK(dst_offset->value_ >= 0) << op_name << " dst_offsets dimension " << i
                                     << " must be non-negative, got " << dst_offset->value_;
      if (dim) {
        CHECK(dst_offset->value_ + dim->value_ <= dst_dim->value_)
            << op_name << " dst subregion dimension " << i
            << " exceeds dst shape (offset=" << dst_offset->value_ << ", shape=" << dim->value_
            << ", dst_dim=" << dst_dim->value_ << ")";
      }
    }
    if (auto src_offset = As<ConstInt>(src_offsets->elements_[i])) {
      CHECK(src_offset->value_ >= 0) << op_name << " src_offsets dimension " << i
                                     << " must be non-negative, got " << src_offset->value_;
      if (dim) {
        CHECK(src_offset->value_ + dim->value_ <= src_dim->value_)
            << op_name << " src subregion dimension " << i
            << " exceeds src shape (offset=" << src_offset->value_ << ", shape=" << dim->value_
            << ", src_dim=" << src_dim->value_ << ")";
      }
    }
  }
}

// Read an optional int attr (chunk_rows / chunk_cols), defaulting to 0 (unset).
int ReadIntKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) return AnyCast<int>(v, key);
  }
  return 0;
}

// A dynamic transfer extent (a runtime sub-extent of the fixed window) needs a
// static chunk to bound the VEC staging tile (UB allocation is static). The
// flattened transfer is [rows = prod(leading dims), cols = innermost dim]: a
// dynamic innermost requires chunk_cols, a dynamic leading dim requires
// chunk_rows. Fully-static transfers need no chunk. User-facing.
void ValidateDynamicTransferHasChunk(const std::vector<ExprPtr>& transfer_shape,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  if (transfer_shape.empty()) return;
  const bool innermost_dynamic = !As<ConstInt>(transfer_shape.back());
  bool leading_dynamic = false;
  for (size_t i = 0; i + 1 < transfer_shape.size(); ++i) {
    if (!As<ConstInt>(transfer_shape[i])) leading_dynamic = true;
  }
  if (!innermost_dynamic && !leading_dynamic) return;
  const int chunk_rows = ReadIntKwarg(kwargs, "chunk_rows");
  const int chunk_cols = ReadIntKwarg(kwargs, "chunk_cols");
  CHECK(!leading_dynamic || chunk_rows > 0)
      << op_name
      << ": a dynamic leading transfer dim needs a static chunk_rows to bound the VEC staging tile";
  CHECK(!innermost_dynamic || chunk_cols > 0)
      << op_name
      << ": a dynamic innermost transfer dim needs a static chunk_cols to bound the VEC staging tile";
}

// Flatten an N-D transfer shape to its 2-D [rows, cols] extent
// (rows = prod(leading dims), cols = innermost dim) and confirm the 2-D VEC
// stage tile fits within it. pto-isa TPUT/TGET auto-chunks the transfer through
// a smaller stage, so the stage is allowed to be smaller than the transfer; it
// must only not exceed it. Dynamic transfer dims can't be compared statically
// (the chunk bounds them at runtime), so they are skipped here.
//
// NOTE: kept identical to the copy in put.cpp (mirrors the existing per-file
// Validate*Contract / Validate*RegionArgs duplication between these two TUs).
void ValidateStageFitsTransfer(const std::vector<ExprPtr>& stage_shape,
                               const std::vector<ExprPtr>& transfer_shape, const Span& transfer_span,
                               const Span& stage_span, const std::string& op_name) {
  (void)transfer_span;
  int64_t transfer_cols = 1;
  bool cols_static = false;
  int64_t transfer_rows = 1;
  bool rows_static = true;
  for (size_t i = 0; i < transfer_shape.size(); ++i) {
    auto d = As<ConstInt>(transfer_shape[i]);
    if (i + 1 == transfer_shape.size()) {
      cols_static = static_cast<bool>(d);
      if (d) transfer_cols = d->value_;
    } else if (d) {
      transfer_rows *= d->value_;
    } else {
      rows_static = false;
    }
  }
  auto stage_rows_c = As<ConstInt>(stage_shape[0]);
  auto stage_cols_c = As<ConstInt>(stage_shape[1]);
  INTERNAL_CHECK_SPAN(stage_rows_c && stage_cols_c, stage_span)
      << "Internal error: " << op_name << " stage dims must be static ConstInt";
  const int64_t stage_rows = stage_rows_c->value_;
  const int64_t stage_cols = stage_cols_c->value_;
  INTERNAL_CHECK_SPAN(stage_rows > 0 && stage_cols > 0, stage_span)
      << "Internal error: " << op_name << " stage dims must be positive, got [" << stage_rows << ", "
      << stage_cols << "]";
  INTERNAL_CHECK_SPAN(!rows_static || stage_rows <= transfer_rows, stage_span)
      << "Internal error: " << op_name << " stage rows " << stage_rows
      << " must fit within flattened transfer rows " << transfer_rows
      << " (pto-isa auto-chunks a smaller stage)";
  INTERNAL_CHECK_SPAN(!cols_static || stage_cols <= transfer_cols, stage_span)
      << "Internal error: " << op_name << " stage cols " << stage_cols << " must fit within transfer cols "
      << transfer_cols << " (pto-isa auto-chunks a smaller stage)";
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
  if (args.size() == 6) {
    auto dst_type = AsTensorTypeLike(args[0]->GetType());
    auto src_type = As<DistributedTensorType>(args[2]->GetType());
    std::vector<ExprPtr> transfer_shape;
    ValidateGetRegionArgs(args, 3, dst_type->shape_, src_type->shape_, "pld.tensor.get", &transfer_shape);
    ValidateDynamicTransferHasChunk(transfer_shape, kwargs, "pld.tensor.get");
  }

  return GetUnknownType();
}

TypePtr DeduceGetTileType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4 || args.size() == 7)
      << "pld.tile.get requires 4 positional arguments (dst, peer, src, stage) or 7 "
         "(dst, peer, src, stage, dst_offsets, src_offsets, shape), but got "
      << args.size();
  CHECK(kwargs.empty()) << "pld.tile.get does not accept keyword attributes";
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tile.get positional argument #" << i << " must not be null";
  }
  ValidateGetContract(args[0], args[1], args[2], "pld.tile.get", args.size() == 4);

  auto stage_type = As<TileType>(args[3]->GetType());
  CHECK(stage_type) << "pld.tile.get stage must be a TileType, got " << args[3]->GetType()->TypeName();
  auto dst_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(stage_type->dtype_ == dst_type->dtype_)
      << "pld.tile.get stage dtype must match dst dtype, got stage=" << stage_type->dtype_.ToString()
      << " dst=" << dst_type->dtype_.ToString();
  CHECK(stage_type->shape_.size() == 2)
      << "pld.tile.get stage must be a 2D VEC staging tile, got rank " << stage_type->shape_.size();

  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (args.size() == 7) {
    ValidateGetRegionArgs(args, 4, dst_type->shape_, src_type->shape_, "pld.tile.get", &transfer_shape);
  }

  // The explicit stage tile is the 2-D VEC bounce buffer that pto-isa TGET
  // streams the transfer through; it may be smaller than the transfer (a single
  // chunk) but must not exceed the flattened [rows, cols] transfer extent.
  ValidateStageFitsTransfer(stage_type->shape_, transfer_shape, args[0]->span_, args[3]->span_,
                            "pld.tile.get");

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
    .set_attr<int>("chunk_rows")
    .set_attr<int>("chunk_cols")
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
    .no_memory_spec()
    .f_deduce_type(DeduceGetTileType);

}  // namespace ir
}  // namespace pypto
