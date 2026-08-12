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
 * @brief Distributed cross-rank store — ``pld.tile.remote_store`` and its
 *        tensor-level sibling ``pld.tensor.remote_store``.
 *
 * Writes a local value into a region of the ``peer`` rank's slice of a
 * window-bound :class:`DistributedTensorType`. Mirrors ``tile.store`` at the
 * IR level (positional ``offsets`` tuple, optional ``atomic`` attr,
 * side-effect-only return), but the destination is a *remote* slice — the
 * address translation is realised at codegen time by inline peer-offset
 * arithmetic + ``addptr`` + ``make_tensor_view``.
 *
 * IR signatures::
 *
 *     pld.tile.remote_store(src_tile, target, peer, offsets, *, atomic: int) -> Unknown
 *     pld.tensor.remote_store(src, target, peer, offsets, *, atomic: int) -> Unknown
 *
 * The two forms differ only in the IR level of ``src``, exactly as
 * ``tensor.aiv_shard`` / ``tile.aiv_shard`` do: the namespace encodes the level
 * the op lives at (see ``docs/en/dev/distributed_ops.md``). The tensor form is
 * what a tensor-level ``@pl.jit`` kernel writes to push a computed value
 * cross-rank; ``ConvertTensorToTileOps`` lowers it 1:1 to the tile form, and a
 * ``src`` that is still GM-resident at that point is auto-bridged with a
 * natural ``tile.load`` into Vec by the conversion registry's ``InputSpaceReq``.
 * ``pld.tensor.put`` remains the GM->GM bulk (TPUT) path: it stages through a
 * VEC bounce buffer and owns the chunking / pipelining knobs that a single
 * ``pto.tstore`` has no use for.
 *
 * The DSL surface (``pld.tile.remote_store`` /``pld.tensor.remote_store`` in
 * ``python/pypto/language/distributed/op/``) exposes ``target`` / ``peer`` /
 * ``offsets`` as keyword-or-positional for readability; the underlying IR ops
 * keep them positional, matching the convention used by ``tile.store`` (see
 * ``src/ir/op/tensor_ops/memory.cpp``). ``pld.remote_store`` dispatches between
 * the two on the kind of ``src``.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``src_tile`` must have :class:`TileType` (tile form) / ``src`` must have
 *   :class:`TensorType` (tensor form) — mismatches name the sibling op so the
 *   author is pointed at the entry point for the level they are writing at.
 * * ``target`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so users cannot accidentally feed a non-window-bound
 *   tensor into a cross-rank store.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``offsets`` must be a :class:`MakeTuple`, with rank equal to
 *   ``target.shape.size()``.
 * * ``src`` dtype must match ``target`` dtype.
 * * ``target`` rank >= 2, and ``src`` must be 2-D — or N-D with every leading
 *   dim 1, since the deducer also runs *before* ``FlattenTileNdTo2D`` collapses
 *   N-D tiles. Codegen pushes the 2-D extent into the inner two dims of the peer
 *   slice, padding leading dims with 1s (see ``EffectivePushExtent``).
 * * the pushed region must fit inside ``target`` at ``offsets`` (static dims
 *   only) — see ``comm_op::ValidatePushFitsTarget``.
 * * ``atomic`` (optional, defaults to ``AtomicType::kNone``) must be a legal
 *   :enum:`AtomicType`; ``kAdd`` emits ``pto.tstore``'s ``atomicType`` attr,
 *   the same combine mode ``tile.store`` already exposes.
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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "src/ir/op/distributed/comm_op_utils.h"

namespace pypto {
namespace ir {

namespace {

// Reduce a source extent to the 2-D region the store actually writes.
//
// The deducer runs at authoring time, *before* FlattenTileNdTo2D (pass 13) has
// collapsed N-D tiles, so a legal source may still carry leading dims here. Each
// of those must describe a single element: flatten folds them into the row count
// (rows = product of the leading dims), and codegen pads the peer partition view
// with size-1 leading dims to match (see MakeRemoteStoreCodegenPTO). A leading
// extent of 1 therefore survives the round trip unchanged and is accepted; a
// static extent > 1 would be folded into the rows and overflow the target's
// inner dims, so it is rejected here rather than emitting an out-of-bounds
// partition view. A dynamic leading dim cannot be disproved statically and is
// left to run.
std::vector<ExprPtr> EffectivePushExtent(const std::vector<ExprPtr>& extent, const std::string& op_name) {
  CHECK(extent.size() >= 2) << op_name
                            << " src must be at least 2-D (it is pushed as a 2-D region), got rank "
                            << extent.size();
  for (size_t i = 0; i + 2 < extent.size(); ++i) {
    auto dim = As<ConstInt>(extent[i]);
    CHECK_SPAN(!dim || dim->value_ == 1, extent[i]->span_)
        << op_name << " src leading dimension " << i << " must be 1 (only the inner two dims are pushed, "
        << "and larger leading dims fold into the row extent and overrun the target), got " << dim->value_;
  }
  return {extent.end() - 2, extent.end()};
}

// The target / peer / offsets / dtype / bounds contract shared by the tile-level
// and tensor-level forms. `push_extent` is the source extent: the tile's
// effective valid_shape for the tile form, the source tensor's shape for the
// tensor form; it is reduced to its inner 2-D region by EffectivePushExtent.
// Returns nothing — every failure is a user error.
void ValidateRemoteStoreContract(const std::vector<ExprPtr>& args, const std::vector<ExprPtr>& push_extent,
                                 const DataType& src_dtype,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  // target must be a DistributedTensorType. As<DistributedTensorType> is an
  // exact ObjectKind match — a plain TensorType (e.g. a regular pl.Tensor
  // parameter) will not match here, which is exactly what we want.
  auto dist_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(dist_type) << op_name << " target must be a DistributedTensor (window-bound), got "
                   << args[1]->GetType()->TypeName();

  // peer must be a scalar (integer rank index). Allow any ScalarType — dtype
  // narrowing to integer is handled at codegen time when emitting the
  // peer-offset scalar arithmetic.
  CHECK(IsA<ScalarType>(args[2]->GetType()))
      << op_name << " peer must be a scalar (rank index), got " << args[2]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[3]);
  CHECK(offsets_tuple) << op_name << " offsets must be a tuple (MakeTuple of scalars), got "
                       << args[3]->TypeName();

  const auto target_rank = dist_type->shape_.size();
  CHECK(offsets_tuple->elements_.size() == target_rank)
      << op_name << " offsets rank (" << offsets_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";

  // TSTORE contract: src dtype must match target dtype. Checked before the
  // shape contract below so a dtype mismatch is reported as such rather than
  // being masked by an unrelated rank complaint.
  CHECK(src_dtype == dist_type->dtype_)
      << op_name << " src dtype (" << src_dtype.ToString() << ") must match target dtype ("
      << dist_type->dtype_.ToString() << ")";

  // Codegen pushes the 2-D extent into the inner two dims of the peer slice,
  // padding the leading (target_rank - 2) partition dims with 1s. Reject the
  // shapes that have no lowering here, as a user error at the call site, rather
  // than as an internal check deep in MakeRemoteStoreCodegenPTO.
  auto effective_extent = EffectivePushExtent(push_extent, op_name);
  CHECK(target_rank >= 2) << op_name << " target rank must be >= 2 to hold a 2-D push, got " << target_rank;

  comm_op::ValidatePushFitsTarget(effective_extent, offsets_tuple->elements_, dist_type->shape_, op_name);

  // `atomic` is optional here (unlike put, where it is always packed): the DSL
  // omits the attr entirely for a plain store so existing printed IR round-trips
  // byte-identically. Same accepted set as put / tile.store, and the same
  // hardware dtype allow-list tile.store applies -- both emit `pto.tstore`'s
  // atomicType, so a dtype the local store path rejects is equally unsupported
  // cross-rank. src and target dtypes are equal by the check above.
  const int atomic_value = GetKwargOr<int>(kwargs, "atomic", 0);
  comm_op::ValidateAtomicValue(atomic_value, op_name);
  comm_op::ValidateAtomicAddDtype(atomic_value, dist_type->dtype_, op_name);
}

void ValidateRemoteStoreArgs(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 4) << op_name
                          << " requires 4 positional arguments (src, target, peer, offsets), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " positional argument #" << i << " must not be null";
  }
}

TypePtr DeduceRemoteStoreType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  ValidateRemoteStoreArgs(args, "pld.tile.remote_store");

  // src_tile must be a TileType.
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "pld.tile.remote_store src_tile must be a TileType, got "
                   << args[0]->GetType()->TypeName()
                   << ". In a tensor-level kernel (@pl.jit) call pld.tensor.remote_store instead "
                      "(pld.remote_store dispatches between the two).";

  // Codegen sizes the peer partition view from the tile's *effective* view, so
  // the bounds check has to use the same extent (a padded tile's physical shape
  // is wider than what it actually writes).
  const auto tile_view = tile_view_semantics::GetEffectiveTileView(*tile_type);
  ValidateRemoteStoreContract(args, tile_view.valid_shape, tile_type->dtype_, kwargs,
                              "pld.tile.remote_store");

  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

TypePtr DeduceTensorRemoteStoreType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  ValidateRemoteStoreArgs(args, "pld.tensor.remote_store");

  // src is a tensor-level value: a computed value that ConvertTensorToTileOps
  // will have rewritten to a tile by lowering time, or a GM tensor the same pass
  // auto-bridges with a tile.load. As<TensorType> is an exact ObjectKind match,
  // so a DistributedTensorType src is refused — pushing one window into another
  // is pld.tensor.put's GM->GM job, not a tstore.
  auto src_type = As<TensorType>(args[0]->GetType());
  CHECK(src_type) << "pld.tensor.remote_store src must be a Tensor (tensor-level value), got "
                  << args[0]->GetType()->TypeName()
                  << ". In a tile-level kernel (@pl.jit.incore / @pl.program) call "
                     "pld.tile.remote_store instead (pld.remote_store dispatches between the two); "
                     "to push one window buffer into another, use pld.tensor.put.";

  ValidateRemoteStoreContract(args, src_type->shape_, src_type->dtype_, kwargs, "pld.tensor.remote_store");

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
        "DistributedTensor. Mirrors tile.store at the IR level (including the optional "
        "`atomic` combine mode) but the destination is a remote slice — address translation "
        "is realised at codegen via inline peer-offset arithmetic + addptr + make_tensor_view.")
    .set_op_category("DistributedOp")
    .add_argument("src_tile", "Local source tile (2-D TileType, dtype must match target)")
    .add_argument("target", "Window-bound DistributedTensor destination (DistributedTensorType)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .set_attr<int>("atomic")
    // Same source spaces tile.store accepts: both lower to pto.tstore, so an Acc
    // (fix-pipe) operand is as legal here as a Vec one. Declaring it lets
    // InferTileMemorySpace pull a producer into a legal space instead of letting
    // an illegal one reach codegen.
    .set_input_memory(0, {MemorySpace::Vec, MemorySpace::Acc})
    .f_deduce_type(DeduceRemoteStoreType);

// ============================================================================
// pld.tensor.remote_store — tensor-level form, lowered 1:1 to the tile form
// ============================================================================

REGISTER_OP("pld.tensor.remote_store")
    .set_description(
        "Tensor-level cross-rank push: write a local tensor-level value into a region of the "
        "peer rank's slice of a window-bound DistributedTensor. Lowered 1:1 by "
        "ConvertTensorToTileOps to pld.tile.remote_store (mirroring tensor.aiv_shard -> "
        "tile.aiv_shard); a src still resident in GM at that point is auto-bridged with a "
        "natural tile.load into Vec. Use pld.tensor.put instead for a GM->GM bulk transfer "
        "that needs TPUT's chunking / pipelining / window-to-window staging.")
    .set_op_category("DistributedOp")
    .add_argument("src", "Local source value (2-D TensorType, dtype must match target)")
    .add_argument("target", "Window-bound DistributedTensor destination (DistributedTensorType)")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorRemoteStoreType);

}  // namespace ir
}  // namespace pypto
