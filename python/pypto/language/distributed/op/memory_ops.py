# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.alloc_window_buffer`` / ``pld.window`` — DSL sentinels for CommGroup windows.

These functions are parser sentinels — calling them at Python runtime always
raises. They exist so that source code like::

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self):
        buf = pld.alloc_window_buffer(256 * 4)            # 256 FP32 elements
        data = pld.window(buf, [256], dtype=pl.FP32)
        ...

is syntactically valid Python that the AST parser can intercept and lift into
``ir.OpExpr(pld.alloc_window_buffer)`` / ``ir.OpExpr(pld.window)`` IR nodes.

The layout mirrors the ``tile.alloc`` / ``MemRef`` / ``TileType`` triple:

* ``alloc_window_buffer`` is **pure address-space allocation** — it takes a
  per-rank ``size`` in **bytes** (matching ``tile.alloc(memspace, size)``)
  and returns the singleton :class:`ir.PtrType` (allocation-identity token).
  At parse time the LHS is a plain ``Var(PtrType)``; the comm-collection
  pass later wraps the Ptr in an :class:`ir.WindowBuffer` Var subclass and
  registers it on the program's CommGroup metadata.
* ``window`` lifts that Ptr handle into a :class:`ir.DistributedTensorType`
  view by specifying the per-rank ``shape`` and ``dtype`` at materialisation
  time. The result type's ``window_buffer`` back-reference is filled in by
  the same comm-collection pass.
"""

from collections.abc import Sequence

from pypto.language.typing import IntLike
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Var

from ..typing.distributed_tensor import DistributedTensor


def alloc_window_buffer(size: IntLike) -> Var:  # noqa: ARG001
    """Declare a per-rank CommGroup window-buffer slot of ``size`` bytes.

    Mirrors ``tile.alloc(memory_space, size)``: pure allocation semantics,
    no shape / dtype concept on the buffer itself. The result is the
    allocation-identity token that ``pld.window`` consumes.

    Args:
        size: Per-rank allocation size in **bytes**. Accepts an ``int``
            literal, a DSL ``Scalar``, or a raw ``ir.Expr`` (e.g. a
            symbolic expression containing ``pld.world_size()``).

    Returns:
        A plain :class:`ir.Var` of type :class:`ir.PtrType` bound at the
        assignment LHS (the buffer's runtime-unique name comes from the
        LHS variable identifier, captured automatically by the parser).
        Pass the result through :func:`window` to materialise a
        :class:`DistributedTensor` view.

    Raises:
        RuntimeError: Always — this function is a parser sentinel. The parser
            intercepts the call before Python ever invokes the body.
    """
    raise RuntimeError(
        "pld.alloc_window_buffer must be called inside a @pl.function "
        "(level=Level.HOST, role=Role.Orchestrator)"
    )


def window(
    buf: Var,  # noqa: ARG001
    shape: Sequence[IntLike],  # noqa: ARG001
    *,
    dtype: DataType,  # noqa: ARG001
) -> DistributedTensor:
    """Materialise a window-buffer Ptr handle as a DistributedTensor view.

    Shape and dtype enter the type system here; the result type
    (:class:`ir.DistributedTensorType`) carries an optional back-reference
    to the source :class:`ir.WindowBuffer` that the comm-collection pass
    fills in later.

    Args:
        buf: A :class:`ir.Var` of type :class:`ir.PtrType` produced by
            :func:`alloc_window_buffer`.
        shape: Per-rank shape (list / tuple of ints, DSL ``Scalar``s, or raw
            ``ir.Expr``s — anything :data:`IntLike` accepts).
        dtype: Element data type. Kwarg-only.

    Returns:
        A :class:`DistributedTensor` view (IR-level
        :class:`ir.DistributedTensorType`) of the given shape and dtype that
        represents the local rank's slice of the window.

    Raises:
        RuntimeError: Always — this function is a parser sentinel.
    """
    raise RuntimeError(
        "pld.window must be called inside a @pl.function (level=Level.HOST, role=Role.Orchestrator)"
    )


__all__ = ["alloc_window_buffer", "window"]
