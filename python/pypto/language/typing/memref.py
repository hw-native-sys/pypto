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

    The **one-argument** form declares an allocation of your own rather than
    describing an existing one: size and address are left for ``InitMemRef`` to
    derive, and the compiler's opportunistic reuse never packs anything else into
    it. Declare it once and reference it, so a misspelling is a ``NameError``
    rather than a second allocation::

        scratch = pl.MemRef("scratch")

        t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(t0)

    Tiles sharing one declared allocation must not be live at the same time, and
    must agree on memory space; both are checked. Declaring an allocation inside a
    ``pl.pipeline(stage=2)`` body is rejected — the cloned stages would make a
    tile co-live with itself — so to hand-manage a level, drive it with
    ``pl.range`` and declare one allocation per slot.

    Note: ``pl.MemRef(...)`` calls inside a ``@pl.program`` body are resolved
    by the parser (``parser/type_resolver.py``), not dispatched through this
    ``__init__``. A ``Scalar`` byte offset is therefore only ever seen by
    pyright; it never reaches the underlying ``ir.MemRef`` constructor.
    """

    @overload
    def __init__(self, name: str, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: Var, byte_offset: _ByteOffset, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: str, byte_offset: _ByteOffset, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: PtrType, byte_offset: _ByteOffset, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, addr: int, size: int, id: int, span: Span = ...) -> None: ...
    @overload
    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, id: int, span: Span = ...
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["MemRef"]
