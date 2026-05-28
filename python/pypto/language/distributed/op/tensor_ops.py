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
* ``put`` is a synchronous cross-rank bulk write (HCCL TPUT): both ``dst`` and
  ``src`` are window-bound :class:`pld.DistributedTensor` (GM/tensor-level)
  views. ``ConvertTensorToTileOps`` rewrites it to a ``tile.create`` VEC
  staging tile plus a ``pld.tile.put`` call so the stage participates in
  memory allocation/lowering before backend codegen.
* ``get`` is a synchronous cross-rank bulk read: both ``dst`` and ``src`` are
  window-bound :class:`pld.DistributedTensor` (GM/tensor-level) views — the
  VEC staging tile that TGET bounces through is synthesised at codegen, so it
  stays a tensor-level op rather than a tile-level one.

``alloc_window_buffer`` is intercepted at the AssignStmt level by the parser
so the buffer's ``name`` kwarg can be derived from the LHS — the body of that
interception still funnels through this wrapper to keep the IR-construction
site singular.
"""

from collections.abc import Sequence

from pypto.ir.op.distributed import tensor_ops as _ir_tensor
from pypto.language.typing import IntLike, Ptr
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import AtomicType, Call, Expr

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _normalize_intlike, _unwrap


def alloc_window_buffer(size: IntLike, *, name: str = "") -> Ptr:
    """Declare a per-rank CommGroup window-buffer slot of ``size`` bytes.

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
    src: DistributedTensor,
    dst_offsets: Sequence[IntLike] | None = None,
    src_offsets: Sequence[IntLike] | None = None,
    shape: Sequence[IntLike] | None = None,
    *,
    atomic: AtomicType = AtomicType.None_,
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
        src: Window-bound :class:`pld.DistributedTensor` source (the local
            rank's slice); must share element type with ``dst``.
        dst_offsets: Optional offsets into the peer ``dst`` slice.
        src_offsets: Optional offsets into the local ``src`` slice.
        shape: Optional static transfer shape. Required when either offset
            argument is provided.
        atomic: :class:`pld.AtomicType` selecting plain-store
            (``AtomicType.None_``, the default) vs atomic-add
            (``AtomicType.Add``) combine semantics (keyword-only).
    """
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    for role, expr in (("dst", dst_expr), ("src", src_expr)):
        if not isinstance(expr, Expr) or not isinstance(expr.type, _ir.DistributedTensorType):
            got = _ir.python_print_type(expr.type) if isinstance(expr, Expr) else type(expr).__name__
            raise TypeError(f"pld.tensor.put expects a DistributedTensor {role} (window-bound); got {got}")
    has_region = dst_offsets is not None or src_offsets is not None or shape is not None
    if has_region and (dst_offsets is None or src_offsets is None or shape is None):
        raise ValueError("pld.tensor.put dst_offsets, src_offsets, and shape must be provided together")

    if not has_region:
        return _ir_tensor.put(dst_expr, _unwrap(peer), src_expr, atomic=atomic)
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
    )


def get(
    dst: DistributedTensor,
    peer: IntLike,
    src: DistributedTensor,
) -> Call:
    """Cross-rank get: read the peer rank's slice of ``src`` into local ``dst``.

    Side-effect-only (the returned Call carries ``UnknownType``). Semantically
    equivalent to ``remote_load + store`` but represented as one tensor-level
    bulk communication op. Lowers to ``CommRemoteOffset(ctx, peer) + addptr +
    make_tensor_view + partition_view + a synthesised VEC staging tile + TGET``
    at codegen.

    Args:
        dst: Local window-bound :class:`pld.DistributedTensor` destination.
        peer: Peer rank index.
        src: Peer rank's window-bound :class:`pld.DistributedTensor` source.

    Returns:
        The underlying IR Call.
    """
    dst_expr = _unwrap(dst)
    src_expr = _unwrap(src)
    for role, expr in (("dst", dst_expr), ("src", src_expr)):
        if not isinstance(expr, Expr) or not isinstance(expr.type, _ir.DistributedTensorType):
            got = _ir.python_print_type(expr.type) if isinstance(expr, Expr) else type(expr).__name__
            raise TypeError(f"pld.tensor.get expects a DistributedTensor {role} (window-bound); got {got}")
    return _ir_tensor.get(dst_expr, _unwrap(peer), src_expr)


__all__ = ["alloc_window_buffer", "get", "put", "window"]
