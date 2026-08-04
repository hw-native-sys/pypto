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


def _validate_window_buffer_shape(shape: list[int | Expr]) -> None:
    """Validate that a window-buffer shape is non-empty and static dims are positive.

    Args:
        shape: Normalized shape — elements are ``int`` or :class:`ir.Expr`.

    Raises:
        ValueError: If ``shape`` is empty or any statically-known dimension
            is non-positive.
    """
    if not shape:
        raise ValueError("pld.tensor.alloc_window_buffer shape must be non-empty")
    for i, d in enumerate(shape):
        if isinstance(d, int):
            if d <= 0:
                raise ValueError(
                    f"pld.tensor.alloc_window_buffer shape dimension[{i}] is {d}; "
                    "all dimensions must be positive"
                )
        elif isinstance(d, _ir.ConstInt):
            if d.value <= 0:
                raise ValueError(
                    f"pld.tensor.alloc_window_buffer shape dimension[{i}] is {d.value}; "
                    "all dimensions must be positive"
                )


def _compute_window_buffer_bytes(shape: list[int | Expr], dtype: DataType) -> int | Expr:
    """Compute byte size for a shape+dtype ``alloc_window_buffer`` call.

    When all shape dimensions are statically known (Python ``int`` or
    :class:`ir.ConstInt`) the product is computed eagerly as a Python
    ``int`` (no IR nodes emitted).  Otherwise (any dimension is a dynamic
    :class:`ir.Expr`) a chain of :class:`ir.Mul` nodes is emitted.

    Args:
        shape: Normalized shape — elements are ``int`` or :class:`ir.Expr`.
        dtype: Element data type (byte size via ``dtype.get_byte()``).

    Returns:
        Byte size as ``int`` (static) or ``ir.Expr`` (dynamic).
    """
    _validate_window_buffer_shape(shape)

    # Fold pure-static: Python ints AND ir.ConstInt — both are compile-time
    # constants.  Per the parser, integer literals arrive as ConstInt(INDEX),
    # not raw Python int, so we must check both.
    static_total = 1
    for d in shape:
        if isinstance(d, int):
            static_total *= d
        elif isinstance(d, _ir.ConstInt):
            static_total *= d.value
        else:
            break
    else:
        # All dims folded — return an eager Python int.
        return static_total * dtype.get_byte()

    # Dynamic path: at least one genuinely dynamic dim.
    span = _ir.Span.unknown()
    # Seed from the first dim instead of ConstInt(1) to avoid a redundant
    # Mul(1, dim0) identity in the emitted IR.
    first = shape[0]
    if isinstance(first, int):
        result: Expr = _ir.ConstInt(first, DataType.INT64, span)
    elif isinstance(first, _ir.ConstInt):
        result = _ir.ConstInt(first.value, DataType.INT64, span)
    else:
        result = first
    for dim in shape[1:]:
        if isinstance(dim, int):
            dim_expr = _ir.ConstInt(dim, DataType.INT64, span)
        elif isinstance(dim, _ir.ConstInt):
            dim_expr = _ir.ConstInt(dim.value, DataType.INT64, span)
        else:
            dim_expr = dim
        result = _ir.Mul(result, dim_expr, DataType.INT64, span)
    byte_expr: Expr = _ir.ConstInt(dtype.get_byte(), DataType.INT64, span)
    return _ir.Mul(result, byte_expr, DataType.INT64, span)


@overload
def alloc_window_buffer(size: IntLike, *, name: str = "") -> Ptr: ...


@overload
def alloc_window_buffer(shape: Sequence[IntLike], *, dtype: DataType, name: str = "") -> Ptr: ...


def alloc_window_buffer(  # type: ignore[no-redef]
    size: IntLike | Sequence[IntLike], *, dtype: DataType | None = None, name: str = ""
) -> Ptr:
    """Declare a per-rank HCCL window-buffer in a comm-domain scope slot.

    Two forms:

    * **Canonical byte form:**
      ``alloc_window_buffer(size, *, name=...)`` — ``size`` is the per-rank
      allocation size in **bytes**.  Accepts an ``int`` literal, a DSL
      ``Scalar``, or a raw :class:`ir.Expr`.

    * **Shape+dtype convenience overload:**
      ``alloc_window_buffer(shape, *, dtype=..., name=...)`` — ``shape`` is a
      list / tuple of per-rank dimensions.  The byte size is computed
      automatically as ``product(shape) x dtype.get_byte()`` and the call
      normalizes to the canonical byte form.  ``dtype`` is required when
      ``shape`` is a sequence and rejected otherwise.

    Both forms return the singleton :class:`ir.PtrType` allocation-identity
    token that ``pld.tensor.window`` consumes.  The two-phase
    ``alloc → window`` design is preserved — a single buffer can back multiple
    ``window()`` views across loop iterations.

    Args:
        size: (Canonical form) Per-rank allocation size in **bytes**.
        shape: (Convenience form) Per-rank shape as a sequence of per-dimension
            sizes — ``int``, DSL ``Scalar``, or raw :class:`ir.Expr`.
        dtype: (Convenience form, required) Element data type. Must be
            ``None`` (default) in the canonical byte form.
        name: Unique buffer identifier. The parser injects this from the LHS
            of the surrounding assignment
            (``buf = pld.tensor.alloc_window_buffer(...)``); users **must not**
            pass it explicitly.

    Returns:
        A :class:`pl.Ptr` wrapping the underlying ``ir.Call`` of result type
        :class:`ir.PtrType`. The parser unwraps it back to ``ir.Expr`` and
        binds it to the LHS as a plain :class:`ir.Var`; passing that Var
        through :func:`window` materialises a :class:`DistributedTensor`
        view.

    .. note::

       This function is callable only inside HOST-level orchestration
       (``level=pl.Level.HOST, role=pl.Role.Orchestrator``). Calling it
       inside InCore (``type=pl.FunctionType.InCore``) raises a parser error.

    .. seealso::

       :func:`window` for creating typed DistributedTensor views over an
       allocated buffer.

    Raises:
        ValueError: If ``name`` is empty (the parser must have injected it).
        ValueError: If ``shape`` is a sequence but ``dtype`` is not provided.
        ValueError: If ``size`` is scalar but ``dtype`` is provided.
    """
    if not name:
        raise ValueError(
            "pld.tensor.alloc_window_buffer must appear as the RHS of a simple assignment "
            "(its result must be bound to a named variable)"
        )

    if isinstance(size, (list, tuple)):
        if dtype is None:
            raise ValueError(
                "pld.tensor.alloc_window_buffer requires dtype= when the first argument is a shape "
                "(list / tuple of per-rank dimensions)"
            )
        shape_normalized: list[int | Expr] = _normalize_intlike(size)
        byte_size: int | Expr = _compute_window_buffer_bytes(shape_normalized, dtype)
        call = _ir_tensor.alloc_window_buffer(_unwrap(byte_size), name=name)
        return Ptr(expr=call)

    if dtype is not None:
        raise ValueError(
            "pld.tensor.alloc_window_buffer dtype= is only valid when the first argument is a shape "
            "(list / tuple), not a scalar byte size"
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

    .. seealso::

       :func:`alloc_window_buffer` for the two-phase ``alloc → window`` pattern.

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
    target: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
    core_num: int = 1,
) -> DistributedTensor: ...


@overload
def allreduce(
    target: DistributedTensor,
    signal: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
    core_num: int = 1,
) -> DistributedTensor: ...


def allreduce(
    target: DistributedTensor,
    signal: DistributedTensor | object = _ALLREDUCE_SIGNAL_MISSING,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
    core_num: int = 1,
) -> DistributedTensor:
    """In-place cross-rank allreduce of a window-bound DistributedTensor.

    After this call returns, every rank's slice of ``target`` holds the
    reduced value. Mirrors :func:`pl.store`'s rebind idiom — users assign the
    result back to the same name:

    .. code-block:: python

        pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
        pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum, mode="ring")

        # Mesh mode on the host orchestrator can omit the signal argument:
        pub = pld.tensor.allreduce(pub, op=pld.ReduceOp.Sum)

    **Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
    ``[world_size, 1]`` (the compiler-synthesized signal is rank-2). InCore
    composites take rank-2 ``[nranks, 1]`` for mesh -- the rank count may be
    dynamic -- and ``[2*(NR-1), NR]`` for ring, where ``NR`` must be a
    compile-time constant.

    **Signal reuse:** InCore composites use the self-clearing credit-barrier
    protocol (see below), so their signal is reusable across back-to-back
    calls — including inside ``for`` / ``while`` / ``if``. The HOST builtin
    allreduce is not yet self-clearing, so a HOST allreduce (explicit or
    synthesized signal) is still rejected inside ``for`` / ``while`` loops.

    .. seealso::

        :func:`alloc_window_buffer` and :func:`window` for buffer allocation
        and view creation.

    .. rubric:: Implementation Notes

    LowerCompositeOps expands the explicit-signal InCore form into either:
    (a) a notify/wait ready barrier followed by UB-sized
    remote_load+accumulate/store chunks for ``mode="mesh"`` (default); or
    (b) the NCCL-style 2(P-1)-step
    chunked reduce-scatter + allgather ring schedule for ``mode="ring"``.
    In both modes the kernel sees only the lowered primitives.
    Host-orchestrator code can omit ``signal`` outside ``for`` / ``while``
    loops; the compiler synthesizes a private INT32 signal window of shape
    ``[pld.world_size(), 1]`` for that call (mesh mode only — ring mode on the
    HOST rail is delivered by a subsequent host builtin).

    Mesh signal shape is ``[NR, 1]``. The InCore ring schedule uses
    ``[2 * (NR − 1), NR]`` (one row per ring round). The host builtin ring
    schedule uses ``[2 * (NR − 1) + 1, NR]`` (one extra row for the return
    barrier). The mesh and ring shapes address cells differently, so one
    buffer must not be shared between the two conventions.

    .. note::

        The host-builtin ring schedule currently supports only
        ``ReduceOp.Sum`` with ``dtype=FP32``.  ``ReduceOp.Max``,
        ``ReduceOp.Min``, ``ReduceOp.Prod``, and ``FP16`` are not yet
        available with ``mode="ring"``.

    Fully-valid packed mesh targets are viewed as one logical 1D stream and
    processed in chunks of at most 16 KiB. For statically known smaller targets,
    the physical chunk width shrinks to the smallest 32-byte-aligned width that
    covers the target. The final chunk uses ``valid_shape`` so arbitrary element
    counts do not read or store past the end.

    See ``docs/en/dev/distributed_ops.md`` and
    ``docs/en/dev/passes/13-lower_composite_ops.md`` for the mesh partial-valid /
    symbolic-extent target constraints (unchanged by this PR).

    **Barrier protocol (self-clearing credit barrier):** every call's
    barriers count a call-local generation ``g`` starting at 1 —
    ``AtomicAdd(1) → WaitGe(g)`` — and a trailing epilogue subtracts the
    call's total credit ``N`` back out of every non-self cell with a single
    ``AtomicAdd(-N)``. Adds and subtracts commute, so the signal is provably
    all-zero again once every rank finishes its epilogue: **the next call on
    the same signal also starts at generation 1**, with no cross-call state
    to go stale. ``N`` may be a runtime scalar, so a mesh allreduce's
    per-chunk credit count does not need to be a compile-time constant.

    In mesh mode, the ready barrier is generation 1; every reduced chunk then
    barriers on the next generation before storing that chunk (``1 +
    chunk_count`` credits total for a fully-valid call; the partial-valid
    rectangle path issues exactly 2). In ring mode, every subchunk of every
    round barriers on its own call-local ready + read-complete generation
    pair; the epilogue subtracts ``2 * chunk_count`` (uniform across rounds)
    from every row of the ``[2*(NR-1), NR]`` signal.

    Ring mode first views a packed ND target as one linear stream, then
    traverses each segment in physical subchunks of at most 16 KiB. FP32 uses
    balanced logical segment boundaries. FP16 aligns every non-empty segment
    start and remote tail span to 32 bytes while restoring the logical
    ``valid_shape`` before reduction and store. This also covers ``SIZE < NR``
    without changing the packed public layout or dropping elements.

    For InCore composites, a signal buffer is safely reusable across
    back-to-back allreduce calls — including inside ``for`` / ``while`` /
    ``if`` — since every call is a stateless cycle starting from all-zero.
    A call aborted mid-flight (error or timeout) leaves credits on the signal;
    recover with a host-side reset (``reset_persistent_windows``) before the
    next dispatch.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` holding per-rank
            FP16 or FP32 data. The C++ verifier refuses a plain
            :class:`pl.Tensor` and unsupported dtypes.
        signal: Optional window-bound INT32 :class:`pld.DistributedTensor`.
            In InCore code this remains required. In host-orchestrator code
            outside ``for`` / ``while`` loops, omitting it lets the compiler
            synthesize a private signal of shape
            ``[pld.world_size(), core_num]``.
        op: :class:`pld.ReduceOp` selecting element-wise ``Sum``, ``Max``,
            ``Min``, or ``Prod`` (keyword-only). Defaults to
            :attr:`pld.ReduceOp.Sum`.
        mode: Algorithm selector (keyword-only). ``"mesh"`` (default) for
            direct all-to-all exchange; ``"ring"`` for the NCCL-style
            chunked reduce-scatter + allgather ring schedule. ``"ring"``
            requires an explicit ``signal`` — host signal synthesis is
            mesh-only, so omitting the signal with ``mode="ring"`` is
            rejected.
        core_num: Number of AIV blocks used by a HOST AllReduce builtin
            (keyword-only). Must be a positive compile-time Python integer and
            may not exceed the configured backend's AIV core count. Defaults to
            1. Multicore is ``mode="mesh"`` only — ``mode="ring"`` requires
            ``core_num=1``. InCore calls must keep this value at 1 and use an
            enclosing :func:`pl.spmd` for multi-core execution.

    Returns:
        The rebound :class:`pld.DistributedTensor` view of ``target`` —
        identical shape / dtype / window-buffer binding, post-reduce content.
    """
    if not isinstance(core_num, int) or isinstance(core_num, bool):
        raise TypeError(
            "pld.tensor.allreduce core_num must be a positive compile-time int, "
            f"got {type(core_num).__name__}"
        )
    if core_num <= 0:
        raise ValueError(f"pld.tensor.allreduce core_num must be positive, got {core_num}")

    if signal is _ALLREDUCE_SIGNAL_MISSING:
        # Host signal synthesis produces a mesh-shaped [world_size, core_num]
        # signal. Ring mode needs a [2*(NR-1), NR] signal, so it must be
        # passed explicitly — reject the synthesized-signal path for it.
        if mode != "mesh":
            raise ValueError(
                f'pld.tensor.allreduce mode="{mode}" requires an explicit signal; '
                "host signal synthesis only supports mesh mode. Pass a window-bound "
                'signal, e.g. pld.tensor.allreduce(target, signal, mode="ring").'
            )
        (target_expr,) = _unwrap_distributed_tensors("pld.tensor.allreduce", target=target)
        call = _ir_tensor.allreduce(target_expr, op=op, core_num=core_num)
        return DistributedTensor(expr=call)
    if signal is None:
        raise TypeError(
            "pld.tensor.allreduce signal cannot be None; omit the signal argument for host synthesis"
        )

    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.allreduce", target=target, signal=signal
    )
    call = _ir_tensor.allreduce(target_expr, signal_expr, op, mode=mode, core_num=core_num)
    return DistributedTensor(expr=call)


def barrier(
    signal: DistributedTensor,
) -> DistributedTensor:
    """Cross-rank barrier synchronisation.

    Blocks until all ranks in the comm group have reached the barrier.
    Uses a window-bound INT32 ``signal`` tensor for cross-rank
    synchronisation.  LowerCompositeOps expands this
    into a notify-all / wait-all sequence.

    .. code-block:: python

        sig = pld.tensor.barrier(sig)

    **Reusable across calls** — see :func:`allreduce` for the shared
    self-clearing credit-barrier protocol. Each call is a stateless cycle
    that restarts at generation 1, so ``sig`` may be reused for back-to-back
    barriers, including inside ``for`` / ``while`` / ``if``.

    Args:
        signal: Window-bound INT32 :class:`pld.DistributedTensor` whose
            shape provides one cell per rank.

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
    root's data.  Uses a window-bound INT32 ``signal`` tensor for the
    cross-rank barrier.

    **Signal shape:** host builtins require rank-1 ``[world_size]``. InCore
    composites take rank-2 ``[nranks, 1]`` -- the rank count may be dynamic.

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
            cross-rank barrier.  Reusable across calls — see
            :func:`allreduce` for the shared barrier protocol.
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
    target: DistributedTensor,
    signal: DistributedTensor,
) -> DistributedTensor:
    """All-gather: gather data from all ranks (push-based).

    Unified 3-arg form: ``pld.tensor.allgather(local_data, target, signal)``.
    ``local_data`` is this rank's single ``[1, SIZE]`` chunk; every rank pushes it
    into every peer's ``target`` row ``my_rank`` via TPUT, then synchronises
    with a notify/wait barrier.  Returns ``target`` in-place (window-as-result
    — same idiom as ``all_to_all`` / ``reduce_scatter`` / ``broadcast``).

    ``local_data`` must be a DIFFERENT buffer from ``target`` — never pass the same
    window for both.  On the HOST path ``local_data`` is a ``[1, SIZE]`` staging
    window populated by an earlier publish step; on the InCore path it is a
    plain :class:`pl.Tensor` ``[1, SIZE]`` — both are accepted.  HOST vs InCore
    is a function-context property resolved by the lowering passes.

    **Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
    ``[world_size, 1]``. InCore composites take rank-2 ``[nranks, 1]`` -- the
    rank count may be dynamic.

    Args:
        local_data: This rank's single chunk — ``[1, SIZE]`` :class:`pl.Tensor`
            (InCore) or ``[1, SIZE]`` :class:`pld.DistributedTensor` staging
            window (HOST).  Must differ from ``target``.
        target: :class:`pld.DistributedTensor` ``[NR, SIZE]`` result window.
            After the call, ``target[src, :]`` holds the chunk from rank
            ``src``.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` barrier
            tensor. Reusable across calls — see :func:`allreduce` for the
            shared barrier protocol.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.allgather", target=target, signal=signal
    )
    input_expr = _unwrap(local_data)
    call = _ir_tensor.allgather(input_expr, target_expr, signal_expr)
    return DistributedTensor(expr=call)


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

    **Signal shape:** host builtins require rank-1 ``[world_size]``. InCore
    composites take rank-2 ``[nranks, 1]`` -- the rank count may be dynamic.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` of shape
            [NR, SIZE].  Each rank stages all NR chunks, one per row.
        signal: Window-bound INT32 :class:`pld.DistributedTensor` for
            the cross-rank barrier. Reusable across calls (2 credits per
            call — ready + post-reduce) — see :func:`allreduce` for the
            shared barrier protocol.
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
    input: Tensor | DistributedTensor,
    target: DistributedTensor,
    signal: DistributedTensor,
) -> DistributedTensor:
    """All-to-all: symmetric personalized exchange (push-based).

    3-arg form: ``pld.tensor.all_to_all(input, target, signal)``.
    Every rank pushes its per-destination chunks directly to every peer's
    ``target`` window via TPUT, then synchronises with a notify/wait
    barrier.  Returns ``target`` in-place (window-as-result — same idiom as
    ``reduce_scatter`` / ``broadcast``).

    ``input`` must be a DIFFERENT buffer from ``target`` — never pass the same
    window for both.  For the HOST-level builtin dispatch, ``input`` is
    typically a second window (e.g. a :class:`pld.DistributedTensor` staged
    by an earlier InCore step) rather than the InCore composite's plain
    :class:`pl.Tensor` — both are accepted.

    **Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
    ``[world_size, 1]``. InCore composites take rank-2 ``[nranks, 1]`` -- the
    rank count may be dynamic.

    Args:
        input: [NR, SIZE] Tensor or DistributedTensor with per-destination
            chunks, distinct from ``target``.  ``input[dest, :]`` is the
            chunk destined for rank ``dest``.
        target: :class:`pld.DistributedTensor` [NR, SIZE] window that receives
            the result in-place.  After the call,
            ``target[src, :]`` holds the chunk received from rank ``src``.
        signal: :class:`pld.DistributedTensor` [NR, 1] INT32 barrier.
            Reusable across calls — see :func:`allreduce` for the shared
            barrier protocol.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    target_expr, signal_expr = _unwrap_distributed_tensors(
        "pld.tensor.all_to_all", target=target, signal=signal
    )
    input_expr = _unwrap(input)
    call = _ir_tensor.all_to_all(input_expr, target_expr, signal_expr)
    return DistributedTensor(expr=call)


def all_to_all_v(
    input: Tensor | DistributedTensor,
    target: DistributedTensor,
    signal: DistributedTensor,
    send_counts: Tensor | DistributedTensor,
    recv_counts: DistributedTensor,
) -> DistributedTensor:
    """All-to-all: variable-size personalized exchange (push-based, window-as-result).

    5-arg form: ``pld.tensor.all_to_all_v(input, target, signal, send_counts,
    recv_counts)``.

    Each rank pushes a full ``MAX_RECV``-row capacity block to peer ``dest``;
    only ``send_counts[dest]`` of those rows are logically valid — the counts
    are read at runtime, so they may be data-dependent (e.g. MoE tokens per
    expert), but they do not change the transfer size.  Mirrors the symmetric
    ``pld.tensor.all_to_all`` otherwise: rows are pushed into a flat 2D staging
    window via ``pld.tile.put``, and the window is returned so the caller can
    read back via ``pl.load``.  There is no built-in read-back phase — the user
    writes the read-back loop in the InCore function.

    ``input`` is a flat 2D [NR*MAX_RECV, SIZE] Tensor or DistributedTensor whose
    rows ``dest*MAX_RECV .. dest*MAX_RECV + send_counts[dest] - 1`` hold the
    chunk for peer ``dest``.  ``target`` is a flat 2D DistributedTensor
    [NR*MAX_RECV, SIZE] — the staging window that doubles as the result;
    rank ``src``'s rows land at ``src*MAX_RECV ...``.

    ``MAX_RECV = target.shape[0] // NR`` is both the compile-time per-peer
    *capacity* and the fixed transfer size: it fixes the row-index arithmetic
    so a receiver can locate each sender's block without knowing that
    sender's count, and every push transfers exactly ``MAX_RECV`` rows
    regardless of the runtime count.  Counts are clamped to ``MAX_RECV``.
    Rows beyond a sender's count still physically cross the wire but are
    logically invalid — as with ``MPI_Alltoallv``, the receiver must not read
    past its published ``recv_counts`` entry; the tail is not zeroed, so
    treat it as containing stale/undefined data, not zeros.

    During the same push, each rank also publishes
    ``min(send_counts[dest], MAX_RECV)`` into peer ``dest``'s
    ``recv_counts[my_rank, 0]`` via ``pld.system.notify`` (Set). After the
    barrier, ``recv_counts[src, 0]`` tells this rank how many of the
    physically-transferred rows from ``src`` are logically valid — use that
    count to know where to stop reading. This is the MPI_Alltoallv recvcounts
    side (published value is the clamped logical count, not the physical
    transfer size).

    The barrier ``signal`` is single-use (same Set(1)/wait≥1 protocol as
    allreduce) and must not be reused inside a ``for``/``while`` loop.

    Args:
        input: Flat 2D Tensor or DistributedTensor [NR*MAX_RECV, SIZE] with
            per-destination chunks (e.g. a window published by a preceding
            exchange).
        target: :class:`pld.DistributedTensor` [NR*MAX_RECV, SIZE] —
            flat 2D staging window (InOut); returned as the result.
        signal: :class:`pld.DistributedTensor` [NR, 1] INT32 barrier (InOut).
        send_counts: INT32 [NR] or [NR, 1] rows-per-destination counts (Input).
            A plain :class:`pl.Tensor` or a window-bound
            :class:`pld.DistributedTensor` (e.g. counts published by a
            preceding exchange).
        recv_counts: :class:`pld.DistributedTensor` INT32 [NR, 1] — after the
            call, ``recv_counts[src, 0]`` holds how many rows ``src`` actually
            sent here (clamped to ``MAX_RECV``) (InOut).

    Returns:
        The ``target`` :class:`pld.DistributedTensor` with received chunks.
    """
    target_expr, signal_expr, recv_expr = _unwrap_distributed_tensors(
        "pld.tensor.all_to_all_v",
        target=target,
        signal=signal,
        recv_counts=recv_counts,
    )
    input_expr = _unwrap(input)
    counts_expr = _unwrap(send_counts)
    call = _ir_tensor.all_to_all_v(input_expr, target_expr, signal_expr, counts_expr, recv_expr)
    return DistributedTensor(expr=call)


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
    "reduce_scatter",
    "window",
]
