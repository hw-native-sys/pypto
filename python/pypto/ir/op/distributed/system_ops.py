# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for distributed system-level ops (``pld.system.world_size``,
``pld.system.get_comm_ctx``, ``pld.system.rank`` / ``pld.system.nranks``).

Mirror of :mod:`pypto.ir.op.system_ops` for the distributed namespace —
exposes the registered C++ ops as Python builders. The DSL layer in
:mod:`pypto.language.distributed.op.system_ops` wraps these for the
``pld.system.*`` surface and re-exports the short form via
``pld.<op>`` unified dispatch.
"""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Span

from ...utils import _get_span_or_capture


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


__all__ = ["get_comm_ctx", "nranks", "rank", "world_size"]
