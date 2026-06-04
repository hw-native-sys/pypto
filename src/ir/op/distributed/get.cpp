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
 * Verifier (strict per kind-trait rules - ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``dst`` / ``src`` must have :class:`DistributedTensorType` - refuse plain
 *   :class:`TensorType` so non-window-bound tensors cannot participate in a
 *   cross-rank read.
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

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void ValidateGetContract(const ExprPtr& dst, const ExprPtr& peer, const ExprPtr& src,
                         const std::string& op_name, bool require_same_shape) {
  CHECK(dst) << op_name << " dst argument must not be null";
  CHECK(peer) << op_name << " peer argument must not be null";
  CHECK(src) << op_name << " src argument must not be null";

  auto dst_type = As<DistributedTensorType>(dst->GetType());
  CHECK(dst_type) << op_name << " dst must be a DistributedTensor (window-bound), got "
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
    auto dim = As<ConstInt>(transfer_shape->elements_[i]);
    CHECK(dim) << op_name << " shape dimensions must be static constants";
    CHECK(dim->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << dim->value_;
    auto dst_dim = As<ConstInt>(dst_shape[i]);
    auto src_dim = As<ConstInt>(src_shape[i]);
    INTERNAL_CHECK(dst_dim && src_dim) << op_name << " tensor shapes must be static before region validation";

    if (auto dst_offset = As<ConstInt>(dst_offsets->elements_[i])) {
      CHECK(dst_offset->value_ >= 0) << op_name << " dst_offsets dimension " << i
                                     << " must be non-negative, got " << dst_offset->value_;
      CHECK(dst_offset->value_ + dim->value_ <= dst_dim->value_)
          << op_name << " dst subregion dimension " << i
          << " exceeds dst shape (offset=" << dst_offset->value_ << ", shape=" << dim->value_
          << ", dst_dim=" << dst_dim->value_ << ")";
    }
    if (auto src_offset = As<ConstInt>(src_offsets->elements_[i])) {
      CHECK(src_offset->value_ >= 0) << op_name << " src_offsets dimension " << i
                                     << " must be non-negative, got " << src_offset->value_;
      CHECK(src_offset->value_ + dim->value_ <= src_dim->value_)
          << op_name << " src subregion dimension " << i
          << " exceeds src shape (offset=" << src_offset->value_ << ", shape=" << dim->value_
          << ", src_dim=" << src_dim->value_ << ")";
    }
  }
}

TypePtr DeduceGetType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6)
      << "pld.tensor.get requires 3 positional arguments (dst, peer, src) or 6 "
         "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
      << args.size();
  CHECK(kwargs.empty()) << "pld.tensor.get does not accept keyword attributes";
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.get positional argument #" << i << " must not be null";
  }

  ValidateGetContract(args[0], args[1], args[2], "pld.tensor.get", args.size() == 3);
  if (args.size() == 6) {
    auto dst_type = As<DistributedTensorType>(args[0]->GetType());
    auto src_type = As<DistributedTensorType>(args[2]->GetType());
    ValidateGetRegionArgs(args, 3, dst_type->shape_, src_type->shape_, "pld.tensor.get");
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
  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
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

  int64_t expected_elems = 1;
  for (const auto& dim : transfer_shape) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK_SPAN(d, args[0]->span_)
        << "Internal error: pld.tile.get transfer shape was not static after validation";
    expected_elems *= d->value_;
  }
  int64_t stage_elems = 1;
  for (const auto& dim : stage_type->shape_) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK_SPAN(d, args[3]->span_) << "Internal error: pld.tile.get stage dim is not ConstInt";
    INTERNAL_CHECK_SPAN(d->value_ > 0, args[3]->span_)
        << "Internal error: pld.tile.get stage dim not positive (" << d->value_ << ")";
    stage_elems *= d->value_;
  }
  INTERNAL_CHECK_SPAN(stage_elems == expected_elems, args[3]->span_)
      << "Internal error: pld.tile.get stage holds " << stage_elems << " elements, expected "
      << expected_elems << " (prod(transfer shape))";

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.get - synchronous cross-rank bulk read from a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.get")
    .set_description(
        "Cross-rank get: synchronously read the `peer` rank's slice of the window-bound "
        "DistributedTensor `src` into the local window-bound DistributedTensor `dst`. "
        "Semantically equivalent to remote_load + store. Supports full-slice and explicit "
        "subregion forms. ConvertTensorToTileOps lowers this to tile.create + pld.tile.get; "
        "PTO emission then produces CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + "
        "partition_view (src) + partition_view (dst) + explicit VEC staging tile + TGET.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType)")
    .add_argument("src", "Remote (peer) window-bound DistributedTensor source (same dtype as dst)")
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
    .add_argument("dst", "Local window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType)")
    .add_argument("src", "Remote (peer) window-bound DistributedTensor source (same dtype as dst)")
    .add_argument("stage", "VEC staging TileType (rows x cols == prod(transfer shape))")
    .no_memory_spec()
    .f_deduce_type(DeduceGetTileType);

}  // namespace ir
}  // namespace pypto
