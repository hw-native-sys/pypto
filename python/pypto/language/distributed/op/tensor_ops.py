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
from typing import overload

from pypto.ir.op.distributed import tensor_ops as _ir_tensor
from pypto.language.typing import IntLike, Ptr
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import AtomicType, Call, Expr, ReduceOp

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _normalize_intlike, _unwrap, _unwrap_distributed_tensors

_ALLREDUCE_SIGNAL_MISSING = object()


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


@overload
def allreduce(
    target: DistributedTensor, *, op: ReduceOp = ReduceOp.Sum, mode: str = "mesh"
) -> DistributedTensor: ...


@overload
def allreduce(
    target: DistributedTensor,
    signal: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
) -> DistributedTensor: ...


def allreduce(
    target: DistributedTensor,
    signal: DistributedTensor | object = _ALLREDUCE_SIGNAL_MISSING,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
) -> DistributedTensor:
    """In-place cross-rank allreduce of a window-bound DistributedTensor.

    After this call returns, every rank's slice of ``target`` holds the
    reduced value. Mirrors :func:`pl.store`'s rebind idiom — users assign the
    result back to the same name:

    .. code-block:: python

        pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
        pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum, mode="ring")

    LowerCompositeOps expands the explicit-signal InCore form into either:
    (a) the 4-phase notify/wait/remote_load+accumulate/store decomposition
    for ``mode="mesh"`` (default); or (b) the NCCL-style 2(P−1)-step
    chunked reduce-scatter + allgather ring schedule for ``mode="ring"``.
    In both modes the kernel sees only the lowered primitives.
    Host-orchestrator code can omit ``signal``
    outside ``for`` and ``while`` loops; the compiler synthesizes a private
    INT32 signal window of shape ``[pld.world_size(), 1]`` for that call
    (mesh mode only — ring mode on the HOST rail is delivered by a
    subsequent host builtin).

    Mesh signal shape is ``[NR, 1]``; ring signal shape is
    ``[2 * (NR − 1), NR]`` (one row per ring round). Both are single-shot
    per call — the lowering uses ``AtomicAdd(0→1)`` / ``WaitGe(1)`` per
    barrier wave, and cells end the call at non-zero values.
    **Do not reuse the same signal buffer for a back-to-back allreduce**
    — allocate a fresh signal buffer (``alloc_window_buffer`` + ``window``)
    for each allreduce call. All allreduce calls in ``for`` and ``while``
    loops are rejected because the current signal protocol cannot provide a
    fresh signal for every dynamic iteration. A self-resetting variant is
    blocked on a runtime fix — PTOAS issue #797.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` holding per-rank
            data. The C++ verifier refuses a plain :class:`pl.Tensor`.
        signal: Optional window-bound INT32 :class:`pld.DistributedTensor`.
            In InCore code this remains required. In host-orchestrator code,
            omitting it outside ``for`` and ``while`` loops lets the compiler
            synthesize a private signal of shape ``[pld.world_size(), 1]``.
            Allreduce calls in those loops are rejected for both signatures.
        op: :class:`pld.ReduceOp` selecting the reduction operator
            (keyword-only). Defaults to :attr:`pld.ReduceOp.Sum`. First-version
            lowering accepts only ``Sum``; ``Max`` / ``Min`` / ``Prod`` are
            reserved enum values and will be rejected at the C++ deducer.
        mode: Algorithm selector (keyword-only). ``"mesh"`` (default) for
            direct all-to-all exchange; ``"ring"`` for the NCCL-style
            chunked reduce-scatter + allgather ring schedule. ``"ring"``
            requires an explicit ``signal`` — host signal synthesis is
            mesh-only, so omitting the signal with ``mode="ring"`` is
            rejected.

    Returns:
        The rebound :class:`pld.DistributedTensor` view of ``target`` —
        identical shape / dtype / window-buffer binding, post-reduce content.
    """
    if signal is _ALLREDUCE_SIGNAL_MISSING:
        # Host signal synthesis only produces a mesh-shaped [world_size, 1]
        # signal. Ring mode needs a [2*(NR-1), NR] signal, so it must be
        # passed explicitly — reject the synthesized-signal path for it.
        if mode != "mesh":
            raise ValueError(
                f'pld.tensor.allreduce mode="{mode}" requires an explicit signal; '
                "host signal synthesis only supports mesh mode. Pass a window-bound "
                'signal, e.g. pld.tensor.allreduce(target, signal, mode="ring").'
            )
        (target_expr,) = _unwrap_distributed_tensors("pld.tensor.allreduce", target=target)
        call = _ir_tensor.allreduce(target_expr, op=op)
        return DistributedTensor(expr=call)
    if signal is None:
        raise TypeError(
            "pld.tensor.allreduce signal cannot be None; omit the signal argument for host synthesis"
        )

    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.allreduce", target=target, signal=signal
    )
    call = _ir_tensor.allreduce(target_expr, signal_expr, op, mode=mode)
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
    local_data: Tensor | DistributedTensor,
    target: DistributedTensor | None = None,
    signal: DistributedTensor | None = None,
    out: Tensor | None = None,
) -> Tensor | DistributedTensor:
    """Gather data from all ranks, either as an InCore composite or HOST builtin.

        **InCore composite (4 args):** ``pld.tensor.allgather(local_data, target, signal, out)`` —
        ``local_data`` is a plain Tensor [1, SIZE] with this rank's chunk.
        The intrinsic handles ``pl.load``, stage-in, notify/wait, and per-peer
        ``remote_load`` into ``out``.

        **HOST builtin (2 args):** ``pld.tensor.allgather(data, signal)`` —
        each rank's chunk is already staged in ``data[my_rank, :]`` via a prior
    publish step.  The host lowering emits ``builtin.tensor.barrier`` per chip
            (the allgather AIV kernel requires concurrent cross-chip dispatch;
            a barrier synchronises pre-staged window data).

        Args:
            local_data: For InCore: :class:`pl.Tensor` [1, SIZE].  For HOST:
                window-bound :class:`pld.DistributedTensor` [NR, SIZE] with
                pre-staged chunks.
            target: InCore only: :class:`pld.DistributedTensor` [NR, SIZE] staging window.
            signal: Window-bound INT32 :class:`pld.DistributedTensor` barrier tensor.
            out: InCore only: :class:`pl.Tensor` [1, NR*SIZE] output.

        Returns:
            InCore: the ``out`` :class:`pl.Tensor`.  HOST: the rebound
            :class:`pld.DistributedTensor`.
    """
    if isinstance(target, DistributedTensor) and signal is None and out is None:
        # 2-arg HOST builtin path: allgather(data, signal)
        # Positional mapping: data→local_data, signal→target
        data_expr, signal_expr = _unwrap_distributed_tensors(
            "pld.tensor.allgather", target=local_data, signal=target
        )
        call = _ir_tensor.allgather(data_expr, signal_expr)
        return DistributedTensor(expr=call)
    # 4-arg InCore composite path
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


def all_to_all(
    input: Tensor,
    target: DistributedTensor,
    signal: DistributedTensor,
) -> DistributedTensor:
    """All-to-all: symmetric personalized exchange (push-based).

    3-arg InCore composite: ``pld.tensor.all_to_all(input, target, signal)``
    Every rank pushes its per-destination chunks directly to every peer's
    window via ``pld.tensor.put`` (TPUT), then synchronises with a notify/wait
    barrier.  Returns ``target`` in-place (window-as-result — same idiom as
    ``reduce_scatter`` / ``broadcast``).

    Args:
        input: :class:`pl.Tensor` [NR, SIZE] with per-destination chunks.
            ``input[dest, :]`` is the chunk destined for rank ``dest``.
        target: :class:`pld.DistributedTensor` [NR, SIZE] window that receives
            the result in-place.  After the call,
            ``target[src, :]`` holds the chunk received from rank ``src``.
        signal: :class:`pld.DistributedTensor` [NR, 1] INT32 barrier.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.all_to_all", target=target, signal=signal
    )
    input_expr = _unwrap(input)
    call = _ir_tensor.all_to_all(input_expr, target_expr, signal_expr)
    return DistributedTensor(expr=call)


__all__ = [
    "all_to_all",
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
