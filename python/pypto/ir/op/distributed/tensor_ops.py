# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` /
``pld.tensor.put``.

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
from pypto.pypto_core.ir import AtomicType, Call, Expr, Span

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
    atomic: AtomicType,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.tensor.put(dst, peer, src)`` Call.

    Cross-rank put: synchronously write the local window-bound DistributedTensor
    ``src`` into ``peer``'s slice of the window-bound DistributedTensor ``dst``.
    ``atomic`` (:class:`ir.AtomicType`) selects plain-store vs atomic-add and is
    packed as an ``int`` attr. Side-effect only — the result is an
    ``UnknownType`` Call. The verifier rejects a non-:class:`ir.DistributedTensorType`
    ``dst`` / ``src`` and requires both to share element type and static shape.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    return _ir_core.create_op_call(
        "pld.tensor.put", [dst, peer_expr, src], {"atomic": int(atomic)}, actual_span
    )


__all__ = ["alloc_window_buffer", "put", "window"]
