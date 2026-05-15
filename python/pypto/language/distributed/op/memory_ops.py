# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.alloc_window_buffer`` / ``pld.window`` — DSL wrappers for CommGroup windows.

These thin wrappers mirror the layering used for ``pl.tile.*`` / ``pl.tensor.*``:
they accept DSL types, unwrap to ``ir.Expr``, delegate to
:mod:`pypto.ir.op.distributed.memory_ops` for the actual ``Call`` construction,
and wrap the result back to the DSL surface type.

The parser routes ``pld.<op>(...)`` calls through ``_dispatch_op`` (the same
helper used for the rest of the DSL surface), which calls the wrappers below
through :func:`invoke_dsl`. ``alloc_window_buffer`` is special-cased in
:meth:`parse_assignment` because the buffer name has to be derived from the
LHS and validated for global uniqueness — but the body of that interception
also funnels through this wrapper, keeping the IR-construction site singular.

Layout mirrors the ``tile.alloc`` / ``MemRef`` / ``TileType`` triple:

* ``alloc_window_buffer`` is **pure address-space allocation** — it takes a
  per-rank ``size`` in **bytes** and returns the singleton :class:`ir.PtrType`
  (allocation-identity token). At parse time the LHS is a plain
  ``Var(PtrType)``; the comm-collection pass later wraps the Ptr in an
  :class:`ir.WindowBuffer` Var subclass.
* ``window`` lifts that Ptr handle into a :class:`ir.DistributedTensorType`
  view by specifying the per-rank ``shape`` and ``dtype``.
"""

from collections.abc import Sequence
from typing import Any

from pypto.ir.op.distributed import memory_ops as _ir_memory
from pypto.language.typing import IntLike, Ptr
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Expr

from ..typing.distributed_tensor import DistributedTensor


def _unwrap(value: Any) -> Any:
    """Unwrap a DSL wrapper (Tensor / Tile / Scalar / Array) to ``ir.Expr``.

    Falls through unchanged for raw ``ir.Expr`` and primitive ``int`` /
    ``float`` values (which the IR layer normalises to ``ConstInt`` /
    ``ConstFloat``).
    """
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def alloc_window_buffer(size: IntLike, *, name: str = "") -> Ptr:
    """Declare a per-rank CommGroup window-buffer slot of ``size`` bytes.

    Mirrors ``tile.alloc(memory_space, size)``: pure allocation semantics, no
    shape / dtype concept on the buffer itself. The result is the
    allocation-identity token that ``pld.window`` consumes.

    Args:
        size: Per-rank allocation size in **bytes**. Accepts an ``int``
            literal, a DSL ``Scalar``, or a raw :class:`ir.Expr`.
        name: Unique buffer identifier. The parser injects this from the LHS
            of the surrounding assignment (``buf = pld.alloc_window_buffer(N)``);
            users **must not** pass it explicitly.

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
            "pld.alloc_window_buffer must appear as the RHS of a simple assignment "
            "(its result must be bound to a named variable)"
        )
    if isinstance(size, (list, tuple)):
        raise ValueError(
            "pld.alloc_window_buffer size must be a scalar (int / Expr in bytes), not a list/tuple"
        )
    call = _ir_memory.alloc_window_buffer(_unwrap(size), name=name)
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
        raise TypeError("pld.window first argument must be an IR expression")
    if not isinstance(buf_expr.type, _ir.PtrType):
        raise TypeError(
            "pld.window expects a Ptr handle (output of pld.alloc_window_buffer); "
            f"got {_ir.python_print_type(buf_expr.type)}"
        )
    shape_list = [_unwrap(s) for s in shape]
    call = _ir_memory.window(buf_expr, shape_list, dtype=dtype)
    return DistributedTensor(expr=call)


__all__ = ["alloc_window_buffer", "window"]
