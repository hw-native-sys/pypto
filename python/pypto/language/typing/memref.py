# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MemRef wrapper type for PyPTO Language DSL.

Thin subclass of ``ir.MemRef`` that widens the accepted ``base`` and
``byte_offset`` parameters so that pyright accepts the ``pl.MemRef(...)``
forms emitted by the IR printer inside ``@pl.program`` code:

* ``base`` also accepts ``PtrType`` — for ``pl.MemRef(ptr_var, offset, size)``
  where ``ptr_var`` is annotated as ``pl.Ptr``.
* ``byte_offset`` also accepts ``Scalar`` — the printer renders a non-constant
  offset as a ``Scalar`` arithmetic expression (e.g. ``pos * 128 * 4``), and a
  constant offset as ``pl.const(0, pl.INT64)``, which is statically a ``Scalar``.
"""

from typing import Any, overload

from pypto.pypto_core.ir import (
    Expr,
    MemorySpace,
    PtrType,
    Span,
    Var,
)
from pypto.pypto_core.ir import (
    MemRef as _IrMemRef,
)

from .scalar import Scalar

# A printed MemRef byte offset is either a constant (rendered by the printer as
# ``pl.const(...)``, statically a ``Scalar``), a DSL ``Scalar`` arithmetic
# expression, or a raw IR ``Expr`` when the MemRef is built programmatically.
# ``int`` stays for MemRefs built programmatically with a plain offset literal.
_ByteOffset = int | Expr | Scalar


class MemRef(_IrMemRef):
    """DSL-level memory reference accepting PtrType bases and Scalar offsets.

    Identical to ``ir.MemRef`` at runtime. The overloads only widen what
    pyright accepts so that printed IR — which uses ``pl.Ptr``-annotated
    base variables and ``Scalar`` arithmetic byte offsets — type-checks
    cleanly when re-loaded as a ``@pl.program``.

    Called with **no offset and size**, it declares an allocation of your own
    rather than describing an existing one: size and address are left for
    ``InitMemRef`` to derive, and the compiler's opportunistic reuse never packs
    anything else into it. The allocation takes the name of the variable it is
    bound to, so the name is written once::

        scratch = pl.MemRef()

        t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(t0)

    Pass ``slots=N`` for N equally-sized slots of one allocation, then pick one by
    subscript. The slots are contiguous and identically sized, so rotating through
    them is a ping-pong the packer cannot collapse::

        l0c = pl.MemRef(slots=2)

        ping: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Acc] = pl.tile.matmul(q, b0)
        pong: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Acc] = pl.tile.matmul(q, b1)

    The subscript is an ordinary index expression, so it may be a **runtime**
    value — a rotation needs no unrolling::

        for i in pl.range(N):
            t: pl.Tile[[64, 64], pl.FP32, l0c[i % 2], pl.Mem.Acc] = pl.tile.matmul(q, b)

    A constant index folds into a static address; a runtime one becomes the tile's
    address at run time. Nothing here is tied to a particular loop form — the
    index is just an expression, so `pl.range`, a `pl.pipeline` sub-index, or a
    function parameter all work.

    Reference it by variable, so a misspelling is a ``NameError`` rather than a
    second allocation. Since the variable *is* the name, one declaration may not
    be reached through two names (``b = a``) and two declarations may not claim
    one name; both are rejected. ``pl.MemRef("other")`` names it explicitly,
    overriding the variable — that is the form the IR printer emits, so a dumped
    program reparses without a surrounding Python scope.

    Co-liveness is checked **per slot**: two tiles on different slots are meant to
    be live together (that is the ping-pong), and only two tiles landing on the
    *same* slot can corrupt each other. A runtime index has no static slot to
    attribute a tile to, so the check is skipped there — the rotation is yours to
    get right — while isolation from every other allocation still holds. Tiles
    sharing one declared allocation must also agree on memory space.

    Declaring an allocation inside a ``pl.pipeline(stage=2)`` body is rejected —
    the cloned stages would make a tile co-live with itself. Declaring slots and
    asking the compiler to multi-buffer are alternatives, not layers: to
    hand-manage a level, drive it with ``pl.range`` and give it its own slots.

    Note: ``pl.MemRef(...)`` calls inside a ``@pl.program`` body are resolved
    by the parser (``parser/type_resolver.py``), not dispatched through this
    ``__init__``. A ``Scalar`` byte offset is therefore only ever seen by
    pyright; it never reaches the underlying ``ir.MemRef`` constructor.
    """

    def __getitem__(self, slot: "int | Scalar") -> "MemRef":  # type: ignore[override]
        """Select one slot of a multi-slot declared allocation.

        Widened over ``ir.MemRef.__getitem__`` for the same reason as
        ``_ByteOffset``: inside a ``@pl.program`` body the subscript is resolved
        from the AST by the parser and may be a runtime index expression, which
        pyright sees as a ``Scalar``. It never reaches the runtime binding.

        Args:
            slot: Slot index — a constant, or an index expression in a program body

        Returns:
            The same declaration bound to that slot
        """
        return super().__getitem__(slot)  # type: ignore[arg-type]

    @overload
    def __init__(self, span: Span = ..., slots: int = ...) -> None: ...
    @overload
    def __init__(self, name: str, span: Span = ..., slots: int = ...) -> None: ...
    # The resolved forms take ``slots`` / ``slot`` too: a MemRef stays slot k of an
    # N-slot allocation after InitMemRef has folded the index into the offset, and
    # `pl.MemRef(base, offset, size, slots=N)[k]` is what the printer emits for one.
    @overload
    def __init__(
        self,
        base: Var,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        base: str,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        base: PtrType,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(self, addr: int, size: int, id: int, span: Span = ...) -> None: ...
    @overload
    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, id: int, span: Span = ...
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["MemRef"]
