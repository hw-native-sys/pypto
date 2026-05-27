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
 * @file remote_store.cpp
 * @brief Distributed cross-rank tile store — ``pld.tile.remote_store``.
 *
 * Writes a local tile into a region of the ``peer`` rank's slice of a
 * window-bound :class:`DistributedTensorType`. Mirrors ``tile.store`` at the
 * IR level (positional ``offsets`` tuple + side-effect-only return), but the
 * destination is a *remote* slice — the address translation is realised at
 * codegen time by ``CommRemoteOffset(ctx, peer) + addptr + make_tensor_view``.
 *
 * IR signature::
 *
 *     pld.tile.remote_store(src_tile, target, peer, offsets) -> Unknown
 *
 * The DSL surface (``pld.tile.remote_store`` in
 * ``python/pypto/language/distributed/op/tile_ops.py``) exposes ``target`` /
 * ``peer`` / ``offsets`` as keyword-only arguments for readability; the
 * underlying IR op keeps them positional, matching the convention used by
 * ``tile.store`` (see ``src/ir/op/tensor_ops/memory.cpp``).
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``src_tile`` must have :class:`TileType` — the local tile being pushed.
 * * ``target`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so users cannot accidentally feed a non-window-bound
 *   tensor into a cross-rank store.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``offsets`` must be a :class:`MakeTuple`, with rank equal to
 *   ``target.shape.size()``.
 * * ``src_tile`` dtype must match ``target`` dtype.
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

TypePtr DeduceRemoteStoreType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
  CHECK(args.size() == 4) << "pld.tile.remote_store requires exactly 4 positional arguments "
                             "(src_tile, target, peer, offsets), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tile.remote_store positional argument #" << i << " must not be null";
  }

  // src_tile must be a TileType.
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "pld.tile.remote_store src_tile must be a TileType, got "
                   << args[0]->GetType()->TypeName();

  // target must be a DistributedTensorType. As<DistributedTensorType> is an
  // exact ObjectKind match — a plain TensorType (e.g. a regular pl.Tensor
  // parameter) will not match here, which is exactly what we want.
  auto dist_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(dist_type) << "pld.tile.remote_store target must be a DistributedTensor (window-bound), got "
                   << args[1]->GetType()->TypeName();

  // peer must be a scalar (integer rank index). Allow any ScalarType — dtype
  // narrowing to integer is handled at codegen time when emitting the
  // CommRemoteOffset scalar arithmetic.
  CHECK(IsA<ScalarType>(args[2]->GetType()))
      << "pld.tile.remote_store peer must be a scalar (rank index), got " << args[2]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[3]);
  CHECK(offsets_tuple) << "pld.tile.remote_store offsets must be a tuple (MakeTuple of scalars), got "
                       << args[3]->TypeName();

  const auto target_rank = dist_type->shape_.size();
  CHECK(offsets_tuple->elements_.size() == target_rank)
      << "pld.tile.remote_store offsets rank (" << offsets_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
  CHECK(target_rank > 0) << "pld.tile.remote_store requires at least one dimension on target";

  // TPUT contract: src_tile dtype must match target dtype.
  CHECK(tile_type->dtype_ == dist_type->dtype_)
      << "pld.tile.remote_store src_tile dtype (" << tile_type->dtype_.ToString()
      << ") must match target dtype (" << dist_type->dtype_.ToString() << ")";

  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tile.remote_store — cross-rank write of a local tile into a peer's slice
// ============================================================================

REGISTER_OP("pld.tile.remote_store")
    .set_description(
        "Write a local tile into a region of the peer rank's slice of a window-bound "
        "DistributedTensor. Mirrors tile.store at the IR level but the destination is a "
        "remote slice — address translation is realised at codegen via "
        "CommRemoteOffset(ctx, peer) + addptr + make_tensor_view.")
    .set_op_category("DistributedOp")
    .add_argument("src_tile", "Local source tile (TileType, dtype must match target)")
    .add_argument("target", "Window-bound DistributedTensor destination (DistributedTensorType)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .no_memory_spec()
    .f_deduce_type(DeduceRemoteStoreType);

}  // namespace ir
}  // namespace pypto
