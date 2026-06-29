# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` /
``pld.tensor.get`` / ``pld.tensor.put`` — DSL wrappers.

Layout mirrors the ``tile.alloc`` / ``MemRef`` / ``TileType`` triple:

* ``alloc_window_buffer`` is **pure address-space allocation** — it takes a
  per-rank ``size`` in **bytes** and returns the singleton :class:`ir.PtrType`
  (allocation-identity token). The comm-collection pass later wraps the Ptr
  in an :class:`ir.WindowBuffer` Var subclass.
* ``window`` lifts that Ptr handle into a :class:`ir.DistributedTensorType`
  view by specifying the per-rank ``shape`` and ``dtype``.
* ``put`` is a synchronous cross-rank bulk write (HCCL TPUT): ``dst`` must be
  a window-bound :class:`pld.DistributedTensor` (the peer needs a window slot
  to receive into), while ``src`` may be either a window-bound
  :class:`pld.DistributedTensor` or a plain :class:`pl.Tensor` — TPUT only
  needs a readable local GM region on the source side. ``ConvertTensorToTileOps``
  rewrites it to a ``tile.create`` VEC staging tile plus a ``pld.tile.put``
  call so the stage participates in memory allocation/lowering before backend
  codegen.
* ``get`` is a synchronous cross-rank bulk read: ``dst`` may be a window-bound
  :class:`pld.DistributedTensor` or a plain :class:`pl.Tensor` — TGET only
  needs a writable local GM region on the destination side; ``src`` must be a
  window-bound :class:`pld.DistributedTensor` (the peer needs a window slot to
  read from). ``ConvertTensorToTileOps`` rewrites it to a ``tile.create`` VEC
  staging tile plus a ``pld.tile.get`` call so the stage participates in memory
  allocation/lowering before backend codegen.

``alloc_window_buffer`` is intercepted at the AssignStmt level by the parser
so the buffer's ``name`` kwarg can be derived from the LHS — the body of that
interception still funnels through this wrapper to keep the IR-construction
site singular.
"""

from collections.abc import Sequence

from pypto.ir.op.distributed import tensor_ops as _ir_tensor
from pypto.language.typing import IntLike, Ptr
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import AtomicType, Call, Expr, ReduceOp

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _normalize_intlike, _unwrap, _unwrap_distributed_tensors


def _validate_chunk(chunk_rows: int, chunk_cols: int, op_name: str) -> None:
    """Validate the put/get staging-tile chunk dims (``0`` = full, else positive int).

    ``chunk_rows`` / ``chunk_cols`` size the VEC staging tile to a sub-tile of the
    flattened transfer ``[rows, cols]`` extent so pto-isa auto-chunks the full
    transfer through it. The staging-tile shape is a compile-time constant, so the
    dims must be non-negative Python ints (``0`` meaning "full extent").
    """
    for name, value in (("chunk_rows", chunk_rows), ("chunk_cols", chunk_cols)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{op_name} {name} must be an int (static), got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{op_name} {name} must be non-negative (0 = full), got {value}")


def _validate_pipeline(pipeline: bool, chunk_rows: int, chunk_cols: int, op_name: str) -> None:
    """Validate the put/get ``pipeline`` (ping-pong double-buffering) kwarg.

    Double-buffering only helps a chunked transfer (pto-isa slides it through two
    staging tiles with overlapped TLOAD/TSTORE), so ``pipeline=True`` requires
    both ``chunk_rows`` and ``chunk_cols`` to be set. The C++ deducer enforces the
    same rule; this front check yields a clearer DSL-level error.
    """
    if not pipeline:
        return
    if not (chunk_rows > 0 and chunk_cols > 0):
        raise ValueError(
            f"{op_name} pipeline=True requires both chunk_rows>0 and chunk_cols>0 "
            f"(got chunk_rows={chunk_rows}, chunk_cols={chunk_cols})"
        )


def alloc_window_buffer(size: IntLike, *, name: str = "") -> Ptr:
    """Declare a per-rank HCCL window-buffer in a comm-domain scope slot of ``size`` bytes.

    Mirrors ``tile.alloc(memory_space, size)``: pure allocation semantics, no
    shape / dtype concept on the buffer itself. The result is the
    allocation-identity token that ``pld.tensor.window`` consumes.

    Args:
        size: Per-rank allocation size in **bytes**. Accepts an ``int``
            literal, a DSL ``Scalar``, or a raw :class:`ir.Expr`.
        name: Unique buffer identifier. The parser injects this from the LHS
            of the surrounding assignment
            (``buf = pld.tensor.alloc_window_buffer(N)``); users **must not**
            pass it explicitly.

    Returns:
        A :class:`pl.Ptr` wrapping the underlying ``ir.Call`` of result type
        :class:`ir.PtrType`. The parser unwraps it back to ``ir.Expr`` and
        binds it to the LHS as a plain :class:`ir.Var`; passing that Var
        through :func:`window` materialises a :class:`DistributedTensor`
        view.

    Raises:
        ValueError: If ``name`` is empty (the parser must have injected it).
    """
    if not name:
        raise ValueError(
            "pld.tensor.alloc_window_buffer must appear as the RHS of a simple assignment "
            "(its result must be bound to a named variable)"
        )
    if isinstance(size, (list, tuple)):
        raise ValueError(
            "pld.tensor.alloc_window_buffer size must be a scalar (int / Expr in bytes), not a list/tuple"
        )
    call = _ir_tensor.alloc_window_buffer(_unwrap(size), name=name)
    return Ptr(expr=call)


def window(
    buf: Ptr,
    shape: Sequence[IntLike],
    *,
    dtype: DataType,
) -> DistributedTensor:
    """Materialise a window-buffer Ptr handle as a DistributedTensor view.

    Shape and dtype enter the type system here; the result type
    (:class:`ir.DistributedTensorType`) carries an optional back-reference to
    the source :class:`ir.WindowBuffer` that the comm-collection pass fills
    in later.

    Args:
        buf: A :class:`pl.Ptr` produced by :func:`alloc_window_buffer` (or a
            raw :class:`ir.Expr` of type :class:`ir.PtrType`).
        shape: Per-rank shape (list / tuple of ints, DSL ``Scalar``s, or raw
            ``ir.Expr``s — anything :data:`IntLike` accepts).
        dtype: Element data type. Kwarg-only.

    Returns:
        A :class:`DistributedTensor` view of the given shape and dtype.
    """
    buf_expr = _unwrap(buf)
    if not isinstance(buf_expr, Expr):
        raise TypeError("pld.tensor.window first argument must be an IR expression")
    if not isinstance(buf_expr.type, _ir.PtrType):
        raise TypeError(
            "pld.tensor.window expects a Ptr handle (output of pld.tensor.alloc_window_buffer); "
            f"got {_ir.python_print_type(buf_expr.type)}"
        )
    shape_list = _normalize_intlike(shape)
    call = _ir_tensor.window(buf_expr, shape_list, dtype=dtype)
    return DistributedTensor(expr=call)


def put(
    dst: DistributedTensor,
    peer: IntLike,
    src: DistributedTensor | Tensor,
    dst_offsets: Sequence[IntLike] | None = None,
    src_offsets: Sequence[IntLike] | None = None,
    shape: Sequence[IntLike] | None = None,
    *,
    atomic: AtomicType = AtomicType.None_,
    chunk_rows: int = 0,
    chunk_cols: int = 0,
    pipeline: bool = False,
) -> Call:
    """Cross-rank put: write the local slice ``src`` into the peer rank's slice of ``dst``.

    Side-effect-only (the returned Call carries ``UnknownType``). Rewritten by
    ``ConvertTensorToTileOps`` to a ``tile.create``-allocated VEC staging tile plus
    a ``pld.tile.put`` call so the staging tile flows through PyPTO's memory
    allocator (required at ``--pto-level=level3``); backend codegen then emits
    ``CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + partition_view +
    TPUT`` against that pre-allocated tile. Both operands are GM/tensor-level
    window views (the staging tile is internal), so this is a ``pld.tensor`` op,
    paired with the GM-to-GM TGET rather than the tile-producing
    ``pld.tile.remote_load``.

    ``dst`` / ``peer`` / ``src`` are positional-or-keyword so the printed IR
    (which emits them positionally) round-trips through the parser; ``atomic``
    stays keyword-only because it lowers to an IR attr (printed as
    ``atomic=<int>``), mirroring ``pld.system.notify``'s ``op``.

    With no offsets/shape this writes the full local ``src`` slice to the full
    peer ``dst`` slice. Supplying ``dst_offsets``, ``src_offsets``, and
    ``shape`` narrows the transfer to matching subregions; all three must be
    provided together.

    Args:
        dst: Window-bound :class:`pld.DistributedTensor` destination (the peer
            rank's slice). The C++ verifier refuses a plain :class:`pl.Tensor`.
        peer: Peer rank index.
        src: Local source — either a :class:`pld.DistributedTensor` (window-
            bound) or a plain :class:`pl.Tensor`. Must share element type with
            ``dst``. Window membership is not required on the source side;
            TPUT only needs a readable local GM region.
        dst_offsets: Optional offsets into the peer ``dst`` slice.
        src_offsets: Optional offsets into the local ``src`` slice.
        shape: Optional static transfer shape. Required when either offset
            argument is provided.
        atomic: :class:`pld.AtomicType` selecting plain-store
            (``AtomicType.None_``, the default) vs atomic-add
            (``AtomicType.Add``) combine semantics (keyword-only).
        chunk_rows: Optional VEC staging-tile row extent (keyword-only,
            ``0`` = full). Sizes the staging tile to a sub-tile of the flattened
            transfer (``rows`` = product of leading dims), so pto-isa TPUT
            auto-chunks the full transfer through it — transfers larger than UB
            no longer need to fit in one staging tile. Oversized values are
            clamped to the transfer extent.
        chunk_cols: Optional VEC staging-tile column extent (keyword-only,
            ``0`` = full innermost dim). Pairs with ``chunk_rows``.
        pipeline: Enable ping-pong double-buffering (keyword-only). When True,
            ``ConvertTensorToTileOps`` allocates two staging tiles and pto-isa
            TPUT overlaps TLOAD/TSTORE across chunks through them. Requires both
            ``chunk_rows`` and ``chunk_cols`` to be set (> 0).
    """
    _validate_chunk(chunk_rows, chunk_cols, "pld.tensor.put")
    _validate_pipeline(pipeline, chunk_rows, chunk_cols, "pld.tensor.put")
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    if not isinstance(dst_expr, Expr) or not isinstance(dst_expr.type, _ir.DistributedTensorType):
        got = _ir.python_print_type(dst_expr.type) if isinstance(dst_expr, Expr) else type(dst_expr).__name__
        raise TypeError(f"pld.tensor.put expects a DistributedTensor dst (window-bound); got {got}")
    if not isinstance(src_expr, Expr) or not isinstance(
        src_expr.type, (_ir.TensorType, _ir.DistributedTensorType)
    ):
        got = _ir.python_print_type(src_expr.type) if isinstance(src_expr, Expr) else type(src_expr).__name__
        raise TypeError(f"pld.tensor.put expects a Tensor or DistributedTensor src; got {got}")
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tensor.put dst_offsets, src_offsets, and shape must be provided together")

    if not has_region:
        return _ir_tensor.put(
            dst_expr,
            _unwrap(peer),
            src_expr,
            atomic=atomic,
            chunk_rows=chunk_rows,
            chunk_cols=chunk_cols,
            pipeline=pipeline,
        )
    assert dst_offsets is not None
    assert src_offsets is not None
    assert shape is not None
    return _ir_tensor.put(
        dst_expr,
        _unwrap(peer),
        src_expr,
        dst_offsets=_normalize_intlike(dst_offsets),
        src_offsets=_normalize_intlike(src_offsets),
        shape=_normalize_intlike(shape),
        atomic=atomic,
        chunk_rows=chunk_rows,
        chunk_cols=chunk_cols,
        pipeline=pipeline,
    )


def get(
    dst: DistributedTensor | Tensor,
    peer: IntLike,
    src: DistributedTensor,
    dst_offsets: Sequence[IntLike] | None = None,
    src_offsets: Sequence[IntLike] | None = None,
    shape: Sequence[IntLike] | None = None,
    *,
    chunk_rows: int = 0,
    chunk_cols: int = 0,
    pipeline: bool = False,
) -> Call:
    """Cross-rank get: read the peer rank's slice of ``src`` into local ``dst``.

    Side-effect-only (the returned Call carries ``UnknownType``). Semantically
    equivalent to ``remote_load + store`` but represented as one tensor-level
    bulk communication op. Lowers to ``CommRemoteOffset(ctx, peer) + addptr +
    make_tensor_view + partition_view + a synthesised VEC staging tile + TGET``
    at codegen.

    With no offsets/shape this reads the full peer ``src`` slice into the full
    local ``dst`` slice. Supplying ``dst_offsets``, ``src_offsets``, and
    ``shape`` narrows the transfer to matching subregions; all three must be
    provided together.

    Args:
        dst: Local destination — either a window-bound
            :class:`pld.DistributedTensor` or a plain :class:`pl.Tensor`.
            TGET only needs a writable local GM region to receive into;
            window membership is not required on the destination side.
        peer: Peer rank index.
        src: Peer rank's window-bound :class:`pld.DistributedTensor` source.
        dst_offsets: Optional offsets into the local ``dst`` slice.
        src_offsets: Optional offsets into the peer ``src`` slice.
        shape: Optional static transfer shape. Required when either offset
            argument is provided.
        chunk_rows: Optional VEC staging-tile row extent (keyword-only,
            ``0`` = full) sizing the staging tile to a sub-tile of the flattened
            transfer so pto-isa TGET auto-chunks the full transfer through it.
            Oversized values are clamped to the transfer extent.
        chunk_cols: Optional VEC staging-tile column extent (keyword-only,
            ``0`` = full innermost dim). Pairs with ``chunk_rows``.
        pipeline: Enable ping-pong double-buffering (keyword-only). When True,
            ``ConvertTensorToTileOps`` allocates two staging tiles and pto-isa
            TGET overlaps TLOAD/TSTORE across chunks through them. Requires both
            ``chunk_rows`` and ``chunk_cols`` to be set (> 0).

    Returns:
        The underlying IR Call.
    """
    _validate_chunk(chunk_rows, chunk_cols, "pld.tensor.get")
    _validate_pipeline(pipeline, chunk_rows, chunk_cols, "pld.tensor.get")
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    if not isinstance(dst_expr, Expr) or not isinstance(
        dst_expr.type, (_ir.TensorType, _ir.DistributedTensorType)
    ):
        got = _ir.python_print_type(dst_expr.type) if isinstance(dst_expr, Expr) else type(dst_expr).__name__
        raise TypeError(f"pld.tensor.get expects a Tensor or DistributedTensor dst; got {got}")
    if not isinstance(src_expr, Expr) or not isinstance(src_expr.type, _ir.DistributedTensorType):
        got = _ir.python_print_type(src_expr.type) if isinstance(src_expr, Expr) else type(src_expr).__name__
        raise TypeError(f"pld.tensor.get expects a DistributedTensor src (window-bound); got {got}")
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tensor.get dst_offsets, src_offsets, and shape must be provided together")

    if not has_region:
        return _ir_tensor.get(
            dst_expr,
            _unwrap(peer),
            src_expr,
            chunk_rows=chunk_rows,
            chunk_cols=chunk_cols,
            pipeline=pipeline,
        )
    assert dst_offsets is not None
    assert src_offsets is not None
    assert shape is not None
    return _ir_tensor.get(
        dst_expr,
        _unwrap(peer),
        src_expr,
        dst_offsets=_normalize_intlike(dst_offsets),
        src_offsets=_normalize_intlike(src_offsets),
        shape=_normalize_intlike(shape),
        chunk_rows=chunk_rows,
        chunk_cols=chunk_cols,
        pipeline=pipeline,
    )


def allreduce(
    target: DistributedTensor,
    signal: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
) -> DistributedTensor:
    """In-place cross-rank allreduce of a window-bound DistributedTensor.

    After this call returns, every rank's slice of ``target`` holds the
    reduced value. Mirrors :func:`pl.store`'s rebind idiom — users assign the
    result back to the same name:

    .. code-block:: python

        pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)

    LowerCompositeOps expands this single Call into the 4-phase
    notify/wait/remote_load+accumulate/store decomposition; the kernel sees
    only the lowered primitives. ``signal`` must be a window-bound INT32
    :class:`pld.DistributedTensor` used as the cross-rank barrier (one slot
    per rank); the host orchestrator allocates and zero-initialises it via
    :func:`alloc_window_buffer` + :func:`window`.

    **Signal buffer is single-shot per call.** The lowering uses two
    barrier waves on the same cells (Set 1 → wait ≥1, then AtomicAdd 1
    → wait ≥2), so by the time the call returns every cell sits at
    ``2`` rather than its initial ``0``. **Do not reuse the same signal
    buffer for a back-to-back allreduce** — the second call's first
    wait would pass immediately on the stale ``≥1``, breaking the
    barrier and racing Phase 3 against the previous reduction's
    Phase 4. Callers issuing multiple allreduces must allocate a fresh
    signal buffer (``alloc_window_buffer`` + ``window``) for each
    call. A self-resetting variant is blocked on a runtime fix —
    PTOAS issue #797.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` holding per-rank
            data. The C++ verifier refuses a plain :class:`pl.Tensor`.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` whose shape
            is ``[nranks, 1]`` (or any shape providing one cell per rank).
            Must be **freshly allocated for this call** (see warning above).
        op: :class:`pld.ReduceOp` selecting the reduction operator
            (keyword-only). Defaults to :attr:`pld.ReduceOp.Sum`. First-version
            lowering accepts only ``Sum``; ``Max`` / ``Min`` / ``Prod`` are
            reserved enum values and will be rejected at the C++ deducer.

    Returns:
        The rebound :class:`pld.DistributedTensor` view of ``target`` —
        identical shape / dtype / window-buffer binding, post-reduce content.
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.allreduce", target=target, signal=signal
    )
    call = _ir_tensor.allreduce(target_expr, signal_expr, op)
    return DistributedTensor(expr=call)


def barrier(
    signal: DistributedTensor,
) -> DistributedTensor:
    """Cross-rank barrier synchronisation.

    Blocks until all ranks in the comm group have reached the barrier.
    Uses a window-bound INT32 ``signal`` matrix for cross-rank
    synchronisation (one slot per rank).  LowerCompositeOps expands this
    into a notify-all / wait-all sequence.

    .. code-block:: python

        sig = pld.tensor.barrier(sig)

    **Signal buffer is single-shot per call.**  The lowering uses
    ``Set(1)`` + ``Ge(1)`` — cells go from 0 to 1.  Do not reuse the
    same signal buffer for back-to-back barriers without reallocation.

    Args:
        signal: Window-bound INT32 :class:`pld.DistributedTensor` whose
            shape provides one cell per rank.  Must be freshly allocated
            for this call.

    Returns:
        The rebound :class:`pld.DistributedTensor` view of ``signal``.
    """
    signal_expr: Expr
    (signal_expr,) = _unwrap_distributed_tensors("pld.tensor.barrier", signal=signal)
    call = _ir_tensor.barrier(signal_expr)
    return DistributedTensor(expr=call)


def broadcast(
    target: DistributedTensor,
    signal: DistributedTensor,
    *,
    root: int,
) -> DistributedTensor:
    """Broadcast root rank's data to all ranks.

    After this call returns, every rank's slice of ``target`` holds
    root's data.  Uses a window-bound INT32 ``signal`` matrix for the
    cross-rank barrier.

    .. code-block:: python

        # Root stages data; non-root skip.
        if my_rank == ROOT_RANK:
            data = pl.store(local, [0, 0], data)
        data = pld.tensor.broadcast(data, sig, root=ROOT_RANK)
        # Every rank now has root's data in data[0, 0:SIZE].

    Args:
        target: Window-bound :class:`pld.DistributedTensor` holding per-rank
            data.  Root must stage its data before the call; non-root slots
            are ignored on input.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` for the
            cross-rank barrier.  Single-shot per call.
        root: Root rank index (int, keyword-only).  Must be non-negative.

    Returns:
        The rebound :class:`pld.DistributedTensor` view of ``target``.
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.broadcast", target=target, signal=signal
    )
    call = _ir_tensor.broadcast(target_expr, signal_expr, root)
    return DistributedTensor(expr=call)


def allgather(
    local_data: Tensor,
    target: DistributedTensor,
    signal: DistributedTensor,
    out: Tensor,
) -> Tensor:
    """Gather data from all ranks, writing the concatenated result into a user-provided output Tensor.

    ``local_data`` is a plain Tensor [1, SIZE] with this rank's chunk —
    the intrinsic handles the ``pl.load`` internally.  ``target`` has shape
    [NR, SIZE] — used internally as a staging window (the intrinsic handles
    the stage-in).  ``signal`` is a window-bound INT32 barrier tensor.
    ``out`` is a plain Tensor [1, NR*SIZE] that receives the rank-ordered
    concatenated result.

    Usage::

        result = pld.tensor.allgather(inp, data, sig, out)
        return out  # out now holds [1, NR*SIZE] of gathered data

    Args:
        local_data: :class:`pl.Tensor` of shape [1, SIZE] — this rank's
            chunk to gather.  The intrinsic loads it into a tile internally.
        target: Window-bound :class:`pld.DistributedTensor` of shape
            [NR, SIZE] used as the internal staging window.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` for the
            cross-rank barrier.
        out: A :class:`pl.Tensor` of shape [1, NR*SIZE] that receives the
            rank-ordered concatenation of all gathered chunks.

    Returns:
        The ``out`` :class:`pl.Tensor` containing the gathered result —
        identical on every rank.
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.allgather", target=target, signal=signal
    )
    local_data_expr = _unwrap(local_data)
    out_expr = _unwrap(out)
    call = _ir_tensor.allgather(local_data_expr, target_expr, signal_expr, out_expr)
    return Tensor(expr=call)


def reduce_scatter(
    target: DistributedTensor,
    signal: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
) -> DistributedTensor:
    """Reduce-scatter: reduce chunks across ranks, one reduced chunk per rank.

    ``target`` has shape [NR, SIZE] — one row per chunk.  Each rank must
    stage all NR chunks before calling::

        for j in range(nranks):
            data = pl.store(chunk_j, [j, 0], data)
        data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
        # data[my_rank, 0:SIZE] now holds this rank's reduced chunk.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` of shape
            [NR, SIZE].  Each rank stages all NR chunks, one per row.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` for
            the cross-rank barrier.  Single-shot per call.
        op: :class:`pld.ReduceOp` (keyword-only).  ``Sum`` only in
            first version; ``Max`` / ``Min`` / ``Prod`` reserved.

    Returns:
        The rebound :class:`pld.DistributedTensor` — rank r's row
        [r, 0:SIZE] holds the reduced chunk r.
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.reduce_scatter", target=target, signal=signal
    )
    call = _ir_tensor.reduce_scatter(target_expr, signal_expr, op)
    return DistributedTensor(expr=call)


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
