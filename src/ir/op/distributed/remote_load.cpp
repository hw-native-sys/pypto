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
 * @file remote_load.cpp
 * @brief Distributed cross-rank tile load — ``pld.tile.remote_load``.
 *
 * Reads a region of the ``peer`` rank's slice of a window-bound
 * :class:`DistributedTensorType` into a local tile. Mirrors ``tile.load``
 * at the IR level (positional ``offsets`` / ``shape`` tuples + TileType
 * result), but the source is a *remote* slice — the address translation
 * is realised at codegen time by ``CommRemoteOffset(ctx, peer) + addptr + make_tensor_view``.
 *
 * IR signature::
 *
 *     pld.tile.remote_load(target, peer, offsets, shape[, valid_shape])
 *         -> TileType(shape, target.dtype)
 *
 * The DSL surface (``pld.tile.remote_load`` in
 * ``python/pypto/language/distributed/op/tile_ops.py``) accepts positional or
 * keyword arguments so printer output round-trips; the underlying IR op keeps
 * them positional, matching ``tile.load``.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>``
 * does NOT match a plain :class:`TensorType`):
 *
 * * ``target`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so users cannot accidentally feed a non-window-bound
 *   tensor into a cross-rank load.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``offsets`` / ``shape`` / optional ``valid_shape`` must each be a
 *   :class:`MakeTuple`, with rank equal to ``target.shape.size()``.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

std::vector<ExprPtr> NormalizeRemoteValidShape(const MakeTuple& valid_shape, const Span& span) {
  std::vector<ExprPtr> normalized;
  normalized.reserve(valid_shape.elements_.size());
  const auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  for (size_t i = 0; i < valid_shape.elements_.size(); ++i) {
    const auto& extent = valid_shape.elements_[i];
    auto scalar_type = extent ? As<ScalarType>(extent->GetType()) : nullptr;
    CHECK_SPAN(scalar_type && scalar_type->dtype_.IsInt(), span)
        << "pld.tile.remote_load valid_shape " << i << " must be an integer scalar, but got "
        << (extent ? extent->GetType()->TypeName() : "null");

    ExprPtr index_extent;
    if (auto constant = As<ConstInt>(extent)) {
      index_extent = std::make_shared<ConstInt>(constant->value_, DataType::INDEX, extent->span_);
    } else if (scalar_type->dtype_ == DataType::INDEX) {
      index_extent = extent;
    } else {
      index_extent = std::make_shared<Cast>(extent, DataType::INDEX, extent->span_);
    }
    CHECK_SPAN(ProveValidExtentLessEqual(zero, index_extent) != ProofResult::kFalse, span)
        << "pld.tile.remote_load valid_shape " << i << " is provably negative; "
        << "a valid region cannot have a negative extent";
    normalized.push_back(index_extent);
  }
  return normalized;
}

TypePtr DeduceRemoteLoadType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
  CHECK(args.size() == 4 || args.size() == 5) << "pld.tile.remote_load requires 4 or 5 positional arguments "
                                                 "(target, peer, offsets, shape[, valid_shape]), but got "
                                              << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tile.remote_load positional argument #" << i << " must not be null";
  }

  // target must be a DistributedTensorType. As<DistributedTensorType> is an
  // exact ObjectKind match — a plain TensorType (e.g. a regular pl.Tensor
  // parameter) will not match here, which is exactly what we want.
  auto dist_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dist_type) << "pld.tile.remote_load target must be a DistributedTensor (window-bound), got "
                   << args[0]->GetType()->TypeName();

  // peer must be a scalar (integer rank index). Allow any ScalarType — dtype
  // narrowing to integer is handled at codegen time when emitting the
  // CommRemoteOffset scalar arithmetic.
  CHECK(IsA<ScalarType>(args[1]->GetType()))
      << "pld.tile.remote_load peer must be a scalar (rank index), got " << args[1]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[2]);
  CHECK(offsets_tuple) << "pld.tile.remote_load offsets must be a tuple (MakeTuple of scalars), got "
                       << args[2]->TypeName();

  auto shape_tuple = As<MakeTuple>(args[3]);
  CHECK(shape_tuple) << "pld.tile.remote_load shape must be a tuple (MakeTuple of scalars), got "
                     << args[3]->TypeName();
  const bool has_requested_valid = args.size() == 5;
  std::vector<ExprPtr> requested_valid;
  if (has_requested_valid) {
    auto valid_shape_tuple = As<MakeTuple>(args[4]);
    CHECK(valid_shape_tuple)
        << "pld.tile.remote_load valid_shape must be a tuple (MakeTuple of scalars), got "
        << args[4]->TypeName();
    requested_valid = NormalizeRemoteValidShape(*valid_shape_tuple, args[4]->span_);
  }

  const auto target_rank = dist_type->shape_.size();
  CHECK(offsets_tuple->elements_.size() == target_rank)
      << "pld.tile.remote_load offsets rank (" << offsets_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(shape_tuple->elements_.size() == target_rank)
      << "pld.tile.remote_load shape rank (" << shape_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(!has_requested_valid || requested_valid.size() == target_rank)
      << "pld.tile.remote_load valid_shape rank (" << requested_valid.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(target_rank > 0) << "pld.tile.remote_load requires at least one dimension on target";

  // Result: a local TileType with the requested physical shape and the target's
  // dtype. Its effective valid_shape always intersects the request (when
  // present) with the source's valid region, so both the four- and five-argument
  // forms enforce physical bounds and remote partitions never read padded data.
  // Layout / memory-space stay unresolved at this point; downstream passes
  // (InferTileMemorySpace etc.) pick them from consumer demand, mirroring
  // tile.load with no target_memory kwarg.
  TileView tile_view;
  tile_view.valid_shape = InferWindowReadValidShape({
      /*source_physical=*/dist_type->shape_,
      /*source_valid=*/GetEffectiveTensorValidShape(*dist_type),
      /*offsets=*/offsets_tuple->elements_,
      /*window=*/shape_tuple->elements_,
      /*requested_valid=*/requested_valid,
      /*kind=*/WindowReadKind::kClampedWindow,
      /*clamp=*/false,
      /*op_name=*/"pld.tile.remote_load",
      /*bounds_remedy=*/
      "Pass a smaller valid_shape to describe a ragged remote-load tail",
      /*span=*/args[0]->span_,
      /*materialize_symbolic_intersection=*/true,
  });

  return std::make_shared<TileType>(shape_tuple->elements_, dist_type->dtype_, std::nullopt, tile_view);
}

}  // namespace

// ============================================================================
// pld.tile.remote_load — cross-rank slice of a DistributedTensor into a tile
// ============================================================================

REGISTER_OP("pld.tile.remote_load")
    .set_description(
        "Load a region of the peer rank's slice of a window-bound DistributedTensor "
        "into a local tile. Mirrors tile.load at the IR level but the source is a "
        "remote slice — address translation is realised at codegen via "
        "CommRemoteOffset(ctx, peer) + addptr + make_tensor_view.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (DistributedTensorType)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .add_argument("shape", "Tile shape per dimension (MakeTuple of scalars)")
    .add_argument("valid_shape", "Optional valid tile extent for ragged tails (MakeTuple of scalars)")
    .no_memory_spec()
    .f_deduce_type(DeduceRemoteLoadType);

}  // namespace ir
}  // namespace pypto
