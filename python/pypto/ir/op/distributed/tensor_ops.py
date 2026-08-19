# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` /
``pld.tensor.get`` / ``pld.tensor.put`` / ``pld.tensor.remote_store``.

These are the raw IR-layer equivalents of :func:`pypto.ir.op.tile_ops.load`
and friends: they take ``ir.Expr`` arguments, normalize them to the shapes
the C++ deducer expects, and emit the ``Call`` via
:func:`ir.create_op_call`. The DSL layer in
:mod:`pypto.language.distributed.op.tensor_ops` wraps these to accept
DSL types and unwrap the result back to a :class:`DistributedTensor`.
"""

from collections.abc import Sequence
from typing import overload

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import AtomicType, Call, Expr, ReduceOp, Span

from ...utils import _get_span_or_capture, _normalize_expr, _to_make_tuple

_ALLREDUCE_SIGNAL_MISSING = object()


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


def remote_store(
    src: Expr,
    target: Expr,
    peer: int | Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    atomic: int = 0,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.remote_store(src, target, peer, offsets)`` Call.

    Tensor-level twin of :func:`pypto.ir.op.distributed.tile_ops.remote_store`:
    same four arguments, same semantics, one IR level up.
    ``ConvertTensorToTileOps`` lowers it 1:1 to ``pld.tile.remote_store`` (and
    auto-bridges a still-GM ``src`` with a ``tile.load``), so a tensor-level
    ``@pl.jit`` kernel can push a computed value cross-rank without first
    round-tripping it through global memory.

    Args:
        src: Local :class:`ir.Expr` with 2-D :class:`ir.TensorType` (dtype must
            match ``target.dtype``). The verifier rejects a
            :class:`ir.DistributedTensorType` — a window-to-window transfer is
            :func:`put`'s GM-to-GM job.
        target: A :class:`ir.Expr` with type :class:`ir.DistributedTensorType`
            (the verifier rejects plain :class:`ir.TensorType`).
        peer: Scalar peer rank index (:class:`ir.Expr` of :class:`ir.ScalarType`).
        offsets: Per-dimension offsets into ``target``'s coordinate space —
            sequence of ints/:class:`ir.Expr`, or an existing :class:`ir.MakeTuple`.
        atomic: ``AtomicType`` underlying int — 0 (``kNone``, plain overwrite)
            or 1 (``kAdd``, atomic-add into the peer's region). Omitted from the
            kwargs entirely when 0.
        span: Optional source span (auto-captured if absent).

    Returns:
        :class:`ir.Call` with :class:`ir.UnknownType` (side-effect only).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    return _ir_core.create_op_call(
        "pld.tensor.remote_store",
        [src, target, peer_expr, offsets_tuple],
        {"atomic": int(atomic)} if atomic else {},
        actual_span,
    )


def put(  # noqa: PLR0913
    dst: Expr,
    peer: int | Expr,
    src: Expr,
    atomic: AtomicType = AtomicType.None_,
    *,
    dst_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    src_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    chunk_rows: int = 0,
    chunk_cols: int = 0,
    pipeline: bool = False,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.put(dst, peer, src)`` Call.

    Cross-rank put: synchronously write the local window-bound DistributedTensor
    ``src`` into ``peer``'s slice of the window-bound DistributedTensor ``dst``.
    ``atomic`` (:class:`ir.AtomicType`) selects plain-store vs atomic-add and is
    packed as an ``int`` attr; ``Add`` requires an fp32/bf16/fp16/int32/int16/int8
    destination (the hardware atomic-add dtypes), and a bf16 destination is
    Ascend910B-only (checked by the ``AtomicAddDtypeValid`` verifier).
    Side-effect only — the result is an
    ``UnknownType`` Call. The verifier requires ``dst`` to be a
    :class:`ir.DistributedTensorType`; ``src`` may be either that or a plain
    :class:`ir.TensorType` (TPUT only needs a readable local GM region). With
    no offsets/shape this writes the full source slice
    into the full destination slice. When offsets and shape are provided it
    writes ``src[src_offsets:src_offsets+shape]`` into the peer rank's
    ``dst[dst_offsets:dst_offsets+shape]``.

    ``chunk_rows`` / ``chunk_cols`` (``0`` = full) size the VEC staging tile that
    ``ConvertTensorToTileOps`` allocates to a sub-tile of the flattened transfer
    ``[rows, cols]`` extent (``rows`` = product of leading dims, ``cols`` =
    innermost dim); pto-isa TPUT then auto-chunks the full transfer through it.
    Packed as ``int`` attrs only when non-zero.

    ``pipeline`` requests ping-pong double-buffering: ``ConvertTensorToTileOps``
    then allocates *two* staging tiles and threads both into ``pld.tile.put``.
    Packed as the ``pipeline`` bool attr (``True``) only when True. The C++ deducer
    requires both ``chunk_rows`` and ``chunk_cols`` to be set when ``pipeline``.
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
    attrs: dict[str, int | bool] = {"atomic": int(atomic)}
    if chunk_rows:
        attrs["chunk_rows"] = int(chunk_rows)
    if chunk_cols:
        attrs["chunk_cols"] = int(chunk_cols)
    if pipeline:
        attrs["pipeline"] = True
    return _ir_core.create_op_call("pld.tensor.put", args, attrs, actual_span)


def get(
    dst: Expr,
    peer: int | Expr,
    src: Expr,
    *,
    dst_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    src_offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    chunk_rows: int = 0,
    chunk_cols: int = 0,
    pipeline: bool = False,
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

    ``chunk_rows`` / ``chunk_cols`` (``0`` = full) size the VEC staging tile to a
    sub-tile of the flattened transfer ``[rows, cols]`` extent so pto-isa TGET
    auto-chunks the full transfer through it. Packed as ``int`` attrs only when
    non-zero.

    ``pipeline`` requests ping-pong double-buffering: ``ConvertTensorToTileOps``
    then allocates *two* staging tiles and threads both into ``pld.tile.get``.
    Packed as the ``pipeline`` bool attr (``True``) only when True. The C++ deducer
    requires both ``chunk_rows`` and ``chunk_cols`` to be set when ``pipeline``.
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
    attrs: dict[str, int | bool] = {}
    if chunk_rows:
        attrs["chunk_rows"] = int(chunk_rows)
    if chunk_cols:
        attrs["chunk_cols"] = int(chunk_cols)
    if pipeline:
        attrs["pipeline"] = True
    return _ir_core.create_op_call("pld.tensor.get", args, attrs, actual_span)


@overload
def allreduce(
    target: Expr,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
    core_num: int = 1,
    span: Span | None = None,
) -> Call: ...


@overload
def allreduce(
    target: Expr,
    signal: Expr,
    op: ReduceOp = ReduceOp.Sum,
    *,
    mode: str = "mesh",
    core_num: int = 1,
    span: Span | None = None,
) -> Call: ...


def allreduce(
    target: Expr,
    signal: Expr | object = _ALLREDUCE_SIGNAL_MISSING,
    op: ReduceOp = ReduceOp.Sum,
    *,
    mode: str = "mesh",
    core_num: int = 1,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.allreduce(target[, signal])`` Call.

    In-place cross-rank allreduce: after the call, every rank's slice of
    ``target`` holds the reduced value. ``signal``, when provided, is a
    window-bound INT32 matrix used as the cross-rank barrier. Host-level calls
    may omit it; SynthesizeAllReduceSignals inserts a private signal before
    downstream lowering. The signal is self-clearing: the lowering restores
    its cells to zero after each call, so one buffer can be reused across
    back-to-back calls and inside for/while loops. Only an omitted signal is
    loop-bound on the HOST rail: synthesis cannot allocate one per dynamic
    iteration, so pass an explicit signal when calling inside a loop. ``op``
    (:class:`ir.ReduceOp`) selects the reduction operator, defaults to
    ``ReduceOp.Sum``, and is packed as an ``int`` attr. ``mode`` selects the
    lowering algorithm: ``"mesh"`` (direct exchange, O(P) windows) or
    ``"ring"`` (chunked reduce-scatter + allgather, O(1) windows). The result
    type is ``target``'s :class:`ir.DistributedTensorType` (the rebind target —
    same semantics as :func:`pl.store`).

    Explicit-signal InCore allreduce is expanded by LowerCompositeOps into a
    ready barrier followed by UB-sized remote-load/accumulate chunks, each with
    a monotonic AtomicAdd/wait before store (mesh), or the 2(P−1)-step
    reduce-scatter + allgather ring schedule (ring). Host-level
    allreduce is lowered later by LowerHostTensorCollectives after signal
    synthesis and comm-domain materialization. Any symbolic target or
    partial-valid extent that survives lowering must be runtime-bound by a
    kernel scalar, loop variable, or physical tensor-shape parameter; a
    type-metadata-only symbol is rejected during PTO codegen. A fully dynamic
    physical target dimension is bound from that tensor parameter.
    """
    if not isinstance(core_num, int) or isinstance(core_num, bool):
        raise TypeError(
            "pld.tensor.allreduce core_num must be a positive compile-time int, "
            f"got {type(core_num).__name__}"
        )
    if core_num <= 0:
        raise ValueError(f"pld.tensor.allreduce core_num must be positive, got {core_num}")

    actual_span = _get_span_or_capture(span, frame_offset=1)
    if signal is _ALLREDUCE_SIGNAL_MISSING:
        args = [target]
    elif signal is None:
        raise TypeError(
            "pld.tensor.allreduce signal cannot be None; omit the signal argument for host synthesis"
        )
    elif isinstance(signal, Expr):
        args = [target, signal]
    else:
        raise TypeError(f"pld.tensor.allreduce signal must be an Expr, got {type(signal).__name__}")
    return _ir_core.create_op_call(
        "pld.tensor.allreduce",
        args,
        {"op": int(op), "mode": mode, "core_num": core_num},
        actual_span,
    )


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

    LowerCompositeOps expands this into notify-all / wait-all + tile.create +
    pld.tile.get; this Call never survives past that pass.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.tensor.broadcast", [target, signal], {"root": root}, actual_span)


def allgather(
    local_data: Expr,
    target: Expr,
    signal: Expr,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.allgather(input, target, signal)`` Call.

    Unified 3-arg push-based form for both paths.  Arg roles:
      arg[0] = local_data — this rank's single chunk, Tensor [1, SIZE]
                            (InCore) or [1, SIZE] staging window (HOST)
      arg[1] = target     — DistributedTensor [NR, SIZE] result window
      arg[2] = signal     — DistributedTensor INT32 barrier

    **InCore composite:** each rank pushes its chunk into every peer's
    ``target`` row ``my_rank`` via ``pld.tile.put``, then notify/wait barrier;
    ``target`` becomes the gathered [NR, SIZE] result (window-as-result).
    Lowered by LowerCompositeOps into a push decomposition.

    **HOST builtin:** distinct input/target windows; lowered to
    ``builtin.tensor.allgather`` per chip (in-kernel TPUT push + barrier).

    Args:
        local_data: This rank's single chunk — Tensor [1, SIZE] (InCore) or
            [1, SIZE] DistributedTensor staging window (HOST).
        target: DistributedTensor [NR, SIZE] result window (window-as-result).
        signal: Window-bound INT32 barrier tensor.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    _args: list[Expr] = [local_data, target, signal]
    return _ir_core.create_op_call("pld.tensor.allgather", _args, {}, actual_span)


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


def all_to_all(
    input: Expr,
    target: Expr,
    signal: Expr,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.all_to_all(...)`` Call.

    3-arg push-based InCore composite: Tensor [NR, SIZE] input, DistributedTensor
    [NR, SIZE] target (window-as-result), INT32 barrier signal.  Lowered by
    LowerCompositeOps into a 2-phase push decomposition (push via
    ``pld.tensor.put`` / TPUT → barrier → return target).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    _args: list[Expr] = [input, target, signal]
    return _ir_core.create_op_call("pld.tensor.all_to_all", _args, {}, actual_span)


def all_to_all_v(
    input: Expr,
    target: Expr,
    signal: Expr,
    send_counts: Expr,
    recv_counts: Expr,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.all_to_all_v(...)`` Call.

    5-arg window-as-result intrinsic: 2D Tensor or DistributedTensor
    [NR*MAX_RECV, SIZE] input, DistributedTensor [NR*MAX_RECV, SIZE] target
    (staging window / result), INT32 barrier signal [NR, 1], INT32
    ``send_counts`` [NR] / [NR, 1] holding the number of rows to send to each
    destination, and window-bound INT32 ``recv_counts`` [NR, 1] that receives
    per-source valid-row counts after the barrier (published via
    ``pld.system.notify`` as ``min(send_counts[dest], MAX_RECV)``). Powered by
    LowerCompositeOps into a 2-phase push-based decomposition (push → barrier),
    returning the target window. Each push transfers a full MAX_RECV-row
    capacity block regardless of the runtime count; the caller reads back
    from the window with ``pl.load``, using ``recv_counts[src, 0]`` to know
    how many of the physically-transferred rows are logically valid.

    ``send_counts`` is read at runtime, so the counts may be data-dependent;
    each count is clamped to the per-peer capacity ``MAX_RECV =
    target.shape[0] // NR``. The barrier signal is self-clearing (restored to
    zero after each call) and safe to reuse inside a ``for``/``while`` loop.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    _args: list[Expr] = [input, target, signal, send_counts, recv_counts]
    return _ir_core.create_op_call("pld.tensor.all_to_all_v", _args, {}, actual_span)


__all__ = [
    "all_to_all",
    "all_to_all_v",
    "alloc_window_buffer",
    "allgather",
    "allreduce",
    "barrier",
    "broadcast",
    "get",
    "put",
    "remote_store",
    "reduce_scatter",
    "window",
]
