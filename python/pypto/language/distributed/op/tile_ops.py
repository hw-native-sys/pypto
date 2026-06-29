# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.tile.*`` — cross-rank tile DSL wrappers.

Each wrapper accepts DSL types, unwraps to ``ir.Expr``, and delegates to the
matching IR builder in :mod:`pypto.ir.op.distributed.tile_ops`.
"""

from collections.abc import Sequence

from pypto.ir.op.distributed import tile_ops as _ir_tile
from pypto.language.typing import IntLike, Tile
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import AtomicType, Call, Expr

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _normalize_intlike, _unwrap


def _is_region_arg(x: object) -> bool:
    """True if a positional arg in the ``stage2`` slot is actually a region tuple.

    The printer emits ``pld.tile.put`` / ``pld.tile.get`` region args
    (``dst_offsets``, ``src_offsets``, ``shape``) positionally right after the
    stage tile(s). A single-stage *subregion* call therefore lands a region tuple
    in the optional ``stage2`` slot on reparse, while a real second staging tile
    is a :class:`pl.Tile` / :class:`ir.Expr`. Region args reparse as Python
    sequences or an :class:`ir.MakeTuple`; staging tiles never do.
    """
    return isinstance(x, (list, tuple, _ir.MakeTuple))


def remote_load(
    target: DistributedTensor,
    peer: IntLike,
    offsets: Sequence[IntLike],
    shape: Sequence[IntLike],
) -> Tile:
    """Load a region of ``peer`` rank's slice of a DistributedTensor into a local tile.

    Mirrors :func:`pl.tile.load` at the user-visible surface, but the source
    is a *remote* slice of a window-bound :class:`pld.DistributedTensor`.
    Address translation happens at codegen via ``CommRemoteOffset`` + addptr + make_tensor_view.

    All arguments are positional-or-keyword (mirroring :func:`pl.tile.load`),
    so the printed IR — which emits them positionally — round-trips through
    the parser. Callers may still pass them by keyword for readability.

    Args:
        target: A window-bound :class:`pld.DistributedTensor` (any rank, any
            dtype). The C++ verifier refuses plain :class:`pl.Tensor` here
            (precise ObjectKind match on :class:`ir.DistributedTensorType`).
        peer: Peer rank index. Accepts an ``int`` literal, a DSL
            ``Scalar``, or a raw ``ir.Expr`` (e.g. ``pld.rank(ctx) + 1``).
        offsets: Offsets into the remote slice, one per ``target`` dimension.
        shape: Per-dimension shape of the tile to load. Determines the output
            :class:`pl.Tile` shape.

    Returns:
        A local :class:`pl.Tile` of the requested shape, dtype equal to
        ``target.dtype``.
    """
    target_expr = _unwrap(target)
    if not isinstance(target_expr, Expr) or not isinstance(target_expr.type, _ir.DistributedTensorType):
        got = (
            _ir.python_print_type(target_expr.type)
            if isinstance(target_expr, Expr)
            else type(target_expr).__name__
        )
        raise TypeError(f"pld.tile.remote_load expects a DistributedTensor target (window-bound); got {got}")

    call = _ir_tile.remote_load(
        target_expr, _unwrap(peer), _normalize_intlike(offsets), _normalize_intlike(shape)
    )
    return Tile(expr=call)


def remote_store(
    src_tile: Tile,
    target: DistributedTensor,
    peer: IntLike,
    offsets: Sequence[IntLike],
) -> Call:
    """Write a local tile into a region of ``peer`` rank's slice of a DistributedTensor.

    Mirrors :func:`pl.tile.store` at the user-visible surface, but the
    destination is a *remote* slice of a window-bound
    :class:`pld.DistributedTensor`. Address translation happens at codegen
    via ``CommRemoteOffset`` + addptr + make_tensor_view.

    All arguments are positional-or-keyword (mirroring :func:`pl.tile.store`),
    so the printed IR — which emits them positionally — round-trips through
    the parser. Callers may still pass them by keyword for readability.

    Args:
        src_tile: Local :class:`pl.Tile` (dtype must match ``target.dtype``).
        target: A window-bound :class:`pld.DistributedTensor` (any rank, any
            dtype). The C++ verifier refuses plain :class:`pl.Tensor` here
            (precise ObjectKind match on :class:`ir.DistributedTensorType`).
        peer: Peer rank index. Accepts an ``int`` literal, a DSL ``Scalar``,
            or a raw ``ir.Expr`` (e.g. ``pld.rank(ctx) + 1``).
        offsets: Offsets into the remote slice, one per ``target`` dimension.

    Returns:
        A side-effect-only :class:`ir.Call` (no SSA result for downstream use).
    """
    tile_expr = _unwrap(src_tile)
    target_expr = _unwrap(target)
    if not isinstance(target_expr, Expr) or not isinstance(target_expr.type, _ir.DistributedTensorType):
        got = (
            _ir.python_print_type(target_expr.type)
            if isinstance(target_expr, Expr)
            else type(target_expr).__name__
        )
        raise TypeError(f"pld.tile.remote_store expects a DistributedTensor target (window-bound); got {got}")

    return _ir_tile.remote_store(tile_expr, target_expr, _unwrap(peer), _normalize_intlike(offsets))


def put(
    dst: DistributedTensor,
    peer: IntLike,
    src: DistributedTensor | Tensor,
    stage: Tile,
    stage2: Tile | None = None,
    dst_offsets: Sequence[IntLike] | None = None,
    src_offsets: Sequence[IntLike] | None = None,
    shape: Sequence[IntLike] | None = None,
    *,
    atomic: AtomicType = AtomicType.None_,
) -> Call:
    """Tile-level form of :func:`pld.tensor.put` with an explicit VEC staging tile.

    Emitted by ``ConvertTensorToTileOps``; defined here only so the printer's
    output roundtrips through the parser. User code calls :func:`pld.tensor.put`.

    ``src`` accepts either a :class:`pld.DistributedTensor` or a plain
    :class:`pl.Tensor`; see :func:`pld.tensor.put` for the rationale. ``stage2``
    is the optional second staging tile for ping-pong double-buffering.
    """
    # The printer prints region tuples positionally after the stage(s); a
    # single-stage subregion call therefore lands a region tuple in the stage2
    # slot on reparse. Shift it back into the region args.
    if _is_region_arg(stage2):
        stage2, dst_offsets, src_offsets, shape = None, stage2, dst_offsets, src_offsets
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    stage_expr = _unwrap(stage)
    stage2_expr = _unwrap(stage2) if stage2 is not None else None
    if not isinstance(dst_expr, Expr) or not isinstance(dst_expr.type, _ir.DistributedTensorType):
        got = _ir.python_print_type(dst_expr.type) if isinstance(dst_expr, Expr) else type(dst_expr).__name__
        raise TypeError(f"pld.tile.put expects a DistributedTensor dst (window-bound); got {got}")
    if not isinstance(src_expr, Expr) or not isinstance(
        src_expr.type, (_ir.TensorType, _ir.DistributedTensorType)
    ):
        got = _ir.python_print_type(src_expr.type) if isinstance(src_expr, Expr) else type(src_expr).__name__
        raise TypeError(f"pld.tile.put expects a Tensor or DistributedTensor src; got {got}")
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tile.put dst_offsets, src_offsets, and shape must be provided together")
    if has_region:
        assert dst_offsets is not None
        assert src_offsets is not None
        assert shape is not None
        return _ir_tile.put(
            dst_expr,
            _unwrap(peer),
            src_expr,
            stage_expr,
            atomic=atomic,
            stage2=stage2_expr,
            dst_offsets=_normalize_intlike(dst_offsets),
            src_offsets=_normalize_intlike(src_offsets),
            shape=_normalize_intlike(shape),
        )
    return _ir_tile.put(dst_expr, _unwrap(peer), src_expr, stage_expr, atomic=atomic, stage2=stage2_expr)


def get(
    dst: DistributedTensor | Tensor,
    peer: IntLike,
    src: DistributedTensor,
    stage: Tile,
    stage2: Tile | None = None,
    dst_offsets: Sequence[IntLike] | None = None,
    src_offsets: Sequence[IntLike] | None = None,
    shape: Sequence[IntLike] | None = None,
) -> Call:
    """Tile-level form of :func:`pld.tensor.get` with an explicit VEC staging tile.

    Emitted by ``ConvertTensorToTileOps``; defined here only so the printer's
    output roundtrips through the parser. User code calls :func:`pld.tensor.get`.
    ``stage2`` is the optional second staging tile for ping-pong double-buffering.
    """
    # Mirror put(): a single-stage subregion call lands a region tuple in the
    # stage2 slot on reparse; shift it back into the region args.
    if _is_region_arg(stage2):
        stage2, dst_offsets, src_offsets, shape = None, stage2, dst_offsets, src_offsets
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    stage_expr = _unwrap(stage)
    stage2_expr = _unwrap(stage2) if stage2 is not None else None
    if not isinstance(dst_expr, Expr) or not isinstance(
        dst_expr.type, (_ir.TensorType, _ir.DistributedTensorType)
    ):
        got = _ir.python_print_type(dst_expr.type) if isinstance(dst_expr, Expr) else type(dst_expr).__name__
        raise TypeError(f"pld.tile.get expects a Tensor or DistributedTensor dst; got {got}")
    if not isinstance(src_expr, Expr) or not isinstance(src_expr.type, _ir.DistributedTensorType):
        got = _ir.python_print_type(src_expr.type) if isinstance(src_expr, Expr) else type(src_expr).__name__
        raise TypeError(f"pld.tile.get expects a DistributedTensor src (window-bound); got {got}")
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tile.get dst_offsets, src_offsets, and shape must be provided together")
    if has_region:
        assert dst_offsets is not None
        assert src_offsets is not None
        assert shape is not None
        return _ir_tile.get(
            dst_expr,
            _unwrap(peer),
            src_expr,
            stage_expr,
            stage2=stage2_expr,
            dst_offsets=_normalize_intlike(dst_offsets),
            src_offsets=_normalize_intlike(src_offsets),
            shape=_normalize_intlike(shape),
        )
    return _ir_tile.get(dst_expr, _unwrap(peer), src_expr, stage_expr, stage2=stage2_expr)


__all__ = ["get", "remote_load", "remote_store", "put"]
