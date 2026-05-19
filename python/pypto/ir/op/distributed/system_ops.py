# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for distributed system-level ops (``pld.system.world_size``,
``pld.system.get_comm_ctx``, ``pld.system.rank`` / ``pld.system.nranks``,
``pld.system.notify`` / ``pld.system.wait``).

Mirror of :mod:`pypto.ir.op.system_ops` for the distributed namespace —
exposes the registered C++ ops as Python builders. The DSL layer in
:mod:`pypto.language.distributed.op.system_ops` wraps these for the
``pld.system.*`` surface and re-exports the short form via
``pld.<op>`` unified dispatch.
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, NotifyOp, Span, WaitCmp

from ...utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def world_size(*, span: Span | None = None) -> Call:
    """Build a ``pld.system.world_size()`` Call returning ``ScalarType(INT64)``.

    Host-only — the parser already validates the call site, so this builder
    is unconditional.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.system.world_size", [], {}, actual_span)


def get_comm_ctx(dist_tensor: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.system.get_comm_ctx(dist_tensor)`` Call returning ``CommCtxType``.

    Type verifier enforces that ``dist_tensor`` has
    :class:`ir.DistributedTensorType` (precise ObjectKind match — plain
    :class:`ir.TensorType` is refused).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.system.get_comm_ctx", [dist_tensor], {}, actual_span)


def rank(ctx: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.system.rank(ctx)`` Call returning ``ScalarType(INT32)``.

    Type verifier enforces that ``ctx`` has :class:`ir.CommCtxType`.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.system.rank", [ctx], {}, actual_span)


def nranks(ctx: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.system.nranks(ctx)`` Call returning ``ScalarType(INT32)``.

    Type verifier enforces that ``ctx`` has :class:`ir.CommCtxType`.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.system.nranks", [ctx], {}, actual_span)


def notify(
    target: _ir_core.Expr,
    peer: int | _ir_core.Expr,
    offsets: Sequence[int | _ir_core.Expr] | _ir_core.MakeTuple,
    value: int | _ir_core.Expr,
    op: NotifyOp,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.system.notify(target, peer, offsets, value)`` Call.

    Cross-rank notify: deposit ``value`` at ``peer``'s slot of the
    window-bound signal matrix ``target``. ``op`` (:class:`ir.NotifyOp`)
    selects atomic-add vs set semantics and is packed as an ``int`` attr.
    Side-effect only — the result is an ``UnknownType`` Call. The verifier
    rejects a non-:class:`ir.DistributedTensorType` ``target``.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    peer_expr = _normalize_expr(peer, actual_span, int_dtype=DataType.INT32)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    value_expr = _normalize_expr(value, actual_span, int_dtype=DataType.INT32)
    return _ir_core.create_op_call(
        "pld.system.notify", [target, peer_expr, offsets_tuple, value_expr], {"op": int(op)}, actual_span
    )


def wait(
    signal: _ir_core.Expr,
    offsets: Sequence[int | _ir_core.Expr] | _ir_core.MakeTuple,
    expected: int | _ir_core.Expr,
    cmp: WaitCmp,
    *,
    span: Span | None = None,
) -> Call:
    """Build a ``pld.system.wait(signal, offsets, expected)`` Call.

    Cross-rank wait: block until the local slot of ``signal`` satisfies
    ``cmp`` against ``expected``. ``cmp`` (:class:`ir.WaitCmp`) selects
    eq vs ge and is packed as an ``int`` attr. Side-effect only — the
    result is an ``UnknownType`` Call. The verifier rejects a
    non-:class:`ir.DistributedTensorType` ``signal``.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    expected_expr = _normalize_expr(expected, actual_span, int_dtype=DataType.INT32)
    return _ir_core.create_op_call(
        "pld.system.wait", [signal, offsets_tuple, expected_expr], {"cmp": int(cmp)}, actual_span
    )


__all__ = ["get_comm_ctx", "notify", "nranks", "rank", "wait", "world_size"]
