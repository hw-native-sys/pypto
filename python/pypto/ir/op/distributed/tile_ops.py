# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for ``pld.tile.*`` distributed tile ops.

The IR op signature is positional (matching ``tile.load``); the DSL wrapper
keeps ``peer`` / ``offsets`` / ``shape`` keyword-only for readability.
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import AtomicType, Call, Expr, Span

from ...utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def remote_load(
    target: Expr,
    peer: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tile.remote_load(target, peer, offsets, shape)`` Call.

    Args:
        target: A :class:`ir.Expr` with type :class:`ir.DistributedTensorType`
            (the verifier rejects plain :class:`ir.TensorType`).
        peer: Scalar peer rank index (:class:`ir.Expr` of :class:`ir.ScalarType`).
        offsets: Per-dimension offsets into ``target``'s coordinate space —
            sequence of ints/:class:`ir.Expr`, or an existing :class:`ir.MakeTuple`.
        shape: Per-dimension tile shape — same shape conventions as ``offsets``.
        span: Optional source span (auto-captured if absent).

    Returns:
        :class:`ir.Call` with result type :class:`ir.TileType` (shape =
        ``shape``, dtype = ``target.dtype``).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    return _ir_core.create_op_call(
        "pld.tile.remote_load",
        [target, peer, offsets_tuple, shape_tuple],
        {},
        actual_span,
    )


def put(
    dst: Expr,
    peer: int | Expr,
    src: Expr,
    stage: Expr,
    atomic: AtomicType,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tile.put(dst, peer, src, stage)`` Call (post-conversion form)."""
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    return _ir_core.create_op_call(
        "pld.tile.put", [dst, peer_expr, src, stage], {"atomic": int(atomic)}, actual_span
    )


__all__ = ["remote_load", "put"]
