# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` /
``pld.tensor.get`` / ``pld.tensor.put``.

These are the raw IR-layer equivalents of :func:`pypto.ir.op.tile_ops.load`
and friends: they take ``ir.Expr`` arguments, normalize them to the shapes
the C++ deducer expects, and emit the ``Call`` via
:func:`ir.create_op_call`. The DSL layer in
:mod:`pypto.language.distributed.op.tensor_ops` wraps these to accept
DSL types and unwrap the result back to a :class:`DistributedTensor`.
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import AtomicType, Call, Expr, ReduceOp, Span

from ...utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def alloc_window_buffer(size: int | Expr, *, name: str, span: Span | None = None) -> Call:
    """Build a ``pld.tensor.alloc_window_buffer(size)`` Call.

    The op's result type is the singleton :class:`ir.PtrType` (allocation
    identity token). The ``name`` kwarg is injected by the parser from the
    assignment LHS — users never write it explicitly.

    Args:
        size: Per-rank allocation size in bytes (``int`` or scalar
            :class:`ir.Expr`).
        name: Unique buffer identifier, kwarg-only.
        span: Optional source span (auto-captured if absent).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    if isinstance(size, int):
        size_expr: Expr = _ir_core.ConstInt(size, DataType.INT64, actual_span)
    else:
        size_expr = size
    return _ir_core.create_op_call("pld.tensor.alloc_window_buffer", [size_expr], {"name": name}, actual_span)


def window(
    buf: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    dtype: DataType,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.window(buf, shape, dtype=...)`` Call.

    Args:
        buf: A :class:`ir.Expr` of type :class:`ir.PtrType` (typically the
            LHS Var bound by :func:`alloc_window_buffer`).
        shape: Per-rank shape — list / tuple of ints / Exprs, or an existing
            :class:`ir.MakeTuple`.
        dtype: Element data type (kwarg-only).
        span: Optional source span (auto-captured if absent).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    shape_tuple = _to_make_tuple(shape, actual_span)
    return _ir_core.create_op_call("pld.tensor.window", [buf, shape_tuple], {"dtype": dtype}, actual_span)


def put(
    dst: Expr,
    peer: int | Expr,
    src: Expr,
    atomic: AtomicType = AtomicType.None_,
    *,
    dst_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    src_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.put(dst, peer, src)`` Call.

    Cross-rank put: synchronously write the local window-bound DistributedTensor
    ``src`` into ``peer``'s slice of the window-bound DistributedTensor ``dst``.
    ``atomic`` (:class:`ir.AtomicType`) selects plain-store vs atomic-add and is
    packed as an ``int`` attr. Side-effect only — the result is an
    ``UnknownType`` Call. The verifier rejects a non-:class:`ir.DistributedTensorType`
    ``dst`` / ``src``. With no offsets/shape this writes the full source slice
    into the full destination slice. When offsets and shape are provided it
    writes ``src[src_offsets:src_offsets+shape]`` into the peer rank's
    ``dst[dst_offsets:dst_offsets+shape]``.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    args: list[Expr] = [dst, peer_expr, src]
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tensor.put dst_offsets, src_offsets, and shape must be provided together")
    if has_region:
        assert dst_offsets is not None
        assert src_offsets is not None
        assert shape is not None
        args.extend(
            [
                _to_make_tuple(dst_offsets, actual_span),
                _to_make_tuple(src_offsets, actual_span),
                _to_make_tuple(shape, actual_span),
            ]
        )
    return _ir_core.create_op_call("pld.tensor.put", args, {"atomic": int(atomic)}, actual_span)


def get(
    dst: Expr,
    peer: int | Expr,
    src: Expr,
    *,
    dst_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    src_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.get(dst, peer, src)`` Call.

    Cross-rank get: synchronously read ``peer``'s slice of the window-bound
    DistributedTensor ``src`` into the local window-bound DistributedTensor
    ``dst``. Side-effect only - the result is an ``UnknownType`` Call. PTO
    emission consumes the explicit VEC staging tile produced by
    ``ConvertTensorToTileOps`` as ``tile.create`` + ``pld.tile.get``, matching
    ``pld.tensor.put``. With no offsets/shape this reads the full peer source
    slice into the full local destination slice. When offsets and shape are
    provided it reads
    ``src[src_offsets:src_offsets+shape]`` from the peer rank into local
    ``dst[dst_offsets:dst_offsets+shape]``.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    args: list[Expr] = [dst, peer_expr, src]
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tensor.get dst_offsets, src_offsets, and shape must be provided together")
    if has_region:
        assert dst_offsets is not None
        assert src_offsets is not None
        assert shape is not None
        args.extend(
            [
                _to_make_tuple(dst_offsets, actual_span),
                _to_make_tuple(src_offsets, actual_span),
                _to_make_tuple(shape, actual_span),
            ]
        )
    return _ir_core.create_op_call("pld.tensor.get", args, {}, actual_span)


def allreduce(
    target: Expr,
    signal: Expr,
    op: ReduceOp = ReduceOp.Sum,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.allreduce(target, signal)`` Call.

    In-place cross-rank allreduce: after the call, every rank's slice of
    ``target`` holds the reduced value. ``signal`` is a window-bound INT32
    matrix used as the cross-rank barrier. ``op`` (:class:`ir.ReduceOp`)
    selects the reduction operator, defaults to ``ReduceOp.Sum``, and is
    packed as an ``int`` attr. The result type is ``target``'s
    :class:`ir.DistributedTensorType` (the rebind target — same semantics as
    :func:`pl.store`).

    LowerCompositeOps expands this into the 4-phase
    notify/wait/remote_load+accumulate/store decomposition; this Call never
    survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.tensor.allreduce", [target, signal], {"op": int(op)}, actual_span)


def barrier(
    signal: Expr,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.barrier(signal)`` Call.

    Cross-rank barrier: blocks until all ranks in the comm group have
    reached the barrier. ``signal`` is a window-bound INT32 matrix used
    as the cross-rank synchronisation. The result type is ``signal``'s
    :class:`ir.DistributedTensorType` (the rebind target — same semantics
    as :func:`allreduce`).

    LowerCompositeOps expands this into a notify-all / wait-all sequence;
    this Call never survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.tensor.barrier", [signal], {}, actual_span)


def broadcast(
    target: Expr,
    signal: Expr,
    root: int,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.broadcast(target, signal, root=...)`` Call.

    Broadcast: replicate root rank's data to every rank in the comm group.
    ``root`` is a static int selecting the source rank.  The result type is
    ``target``'s :class:`ir.DistributedTensorType` (in-place rebind).

    LowerCompositeOps expands this into notify-all / wait-all + remote_load;
    this Call never survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.tensor.broadcast", [target, signal], {"root": root}, actual_span)


def allgather(
    local_data: Expr,
    target: Expr,
    signal: Expr,
    out: Expr,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.allgather(local_data, target, signal, out)`` Call.

    All-gather: gather data from all ranks, writing the concatenated
    result into a user-provided output Tensor.  ``local_data`` is a
    Tensor [1, SIZE] with this rank's chunk (the intrinsic handles the
    ``tile.load`` internally).  ``target`` has shape [NR, SIZE] — used
    internally as a staging window.  ``signal`` is a window-bound INT32
    barrier tensor.  ``out`` is a plain Tensor [1, NR*SIZE] that receives
    the rank-ordered concatenation.

    LowerCompositeOps expands this into tile.load (when local_data is a
    Tensor) + tile.store + notify-all / wait-all + per-peer remote_load
    + tile.store into out; this Call never survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.tensor.allgather", [local_data, target, signal, out], {}, actual_span)


def reduce_scatter(
    target: Expr,
    signal: Expr,
    op: ReduceOp,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.reduce_scatter(target, signal, op=...)`` Call.

    Reduce-scatter: element-wise reduce chunks across all ranks, scatter
    one reduced chunk per rank.  ``op`` (:class:`ir.ReduceOp`) selects the
    reduction operator (Sum only in first version).  Result type is
    ``target``'s :class:`ir.DistributedTensorType` (in-place rebind).

    LowerCompositeOps expands this into a 5-phase decomposition matching
    allreduce; this Call never survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "pld.tensor.reduce_scatter", [target, signal], {"op": int(op)}, actual_span
    )


__all__ = [
    "alloc_window_buffer",
    "allgather",
    "allreduce",
    "barrier",
    "broadcast",
    "get",
    "put",
    "reduce_scatter",
    "window",
]
