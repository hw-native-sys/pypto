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
 * is realised at codegen time by ``CommRemotePtr(ctx, local_ptr, peer)``.
 *
 * IR signature::
 *
 *     pld.tile.remote_load(target, peer, offsets, shape) -> TileType(shape, target.dtype)
 *
 * The DSL surface (``pld.tile.remote_load`` in
 * ``python/pypto/language/distributed/op/tile_ops.py``) exposes ``peer`` /
 * ``offsets`` / ``shape`` as keyword-only arguments for readability; the
 * underlying IR op keeps them positional, matching the convention used by
 * ``tile.load`` (see ``src/ir/op/tile_ops/memory.cpp``).
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>``
 * does NOT match a plain :class:`TensorType`):
 *
 * * ``target`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so users cannot accidentally feed a non-window-bound
 *   tensor into a cross-rank load.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``offsets`` / ``shape`` must each be a :class:`MakeTuple`, with rank
 *   equal to ``target.shape.size()``.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceRemoteLoadType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
  CHECK(args.size() == 4) << "pld.tile.remote_load requires exactly 4 positional arguments "
                             "(target, peer, offsets, shape), but got "
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
  // CommRemotePtr scalar arithmetic.
  CHECK(IsA<ScalarType>(args[1]->GetType()))
      << "pld.tile.remote_load peer must be a scalar (rank index), got " << args[1]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[2]);
  CHECK(offsets_tuple) << "pld.tile.remote_load offsets must be a tuple (MakeTuple of scalars), got "
                       << args[2]->TypeName();

  auto shape_tuple = As<MakeTuple>(args[3]);
  CHECK(shape_tuple) << "pld.tile.remote_load shape must be a tuple (MakeTuple of scalars), got "
                     << args[3]->TypeName();

  const auto target_rank = dist_type->shape_.size();
  CHECK(offsets_tuple->elements_.size() == target_rank)
      << "pld.tile.remote_load offsets rank (" << offsets_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(shape_tuple->elements_.size() == target_rank)
      << "pld.tile.remote_load shape rank (" << shape_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(target_rank > 0) << "pld.tile.remote_load requires at least one dimension on target";

  // Result: a local TileType with the requested shape and the target's dtype.
  // Layout / memory-space stay unresolved at this point; downstream passes
  // (InferTileMemorySpace etc.) pick them from consumer demand, mirroring
  // tile.load with no target_memory kwarg.
  return std::make_shared<TileType>(shape_tuple->elements_, dist_type->dtype_);
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
        "CommRemotePtr(ctx, local_ptr, peer).")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (DistributedTensorType)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .add_argument("shape", "Tile shape per dimension (MakeTuple of scalars)")
    .no_memory_spec()
    .f_deduce_type(DeduceRemoteLoadType);

}  // namespace ir
}  // namespace pypto
