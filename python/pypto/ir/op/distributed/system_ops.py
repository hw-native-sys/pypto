# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for distributed system-level ops (``pld.world_size``,
``pld.get_comm_ctx``, ``pld.comm_ctx.rank`` / ``.nranks``).

Mirror of :mod:`pypto.ir.op.system_ops` for the distributed namespace —
exposes the registered C++ ops as Python builders. The DSL layer in
:mod:`pypto.language.distributed.op.system_ops` wraps :func:`world_size`
for symmetry with the rest of the ``pld.*`` surface; the comm-ctx
builders are consumed directly by the parser's ``data.comm.rank`` /
``data.comm.nranks`` attribute desugar (no user-facing DSL surface).
"""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Span

from ...utils import _get_span_or_capture


def world_size(*, span: Span | None = None) -> Call:
    """Build a ``pld.world_size()`` Call returning ``ScalarType(INT64)``.

    Host-only — the parser already validates the call site, so this builder
    is unconditional.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.world_size", [], {}, actual_span)


def get_comm_ctx(dist_tensor: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.get_comm_ctx(dist_tensor)`` Call returning ``CommCtxType``.

    Type verifier enforces that ``dist_tensor`` has
    :class:`ir.DistributedTensorType` (precise ObjectKind match — plain
    :class:`ir.TensorType` is refused).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.get_comm_ctx", [dist_tensor], {}, actual_span)


def comm_ctx_rank(ctx: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.comm_ctx.rank(ctx)`` Call returning ``ScalarType(INT32)``.

    Type verifier enforces that ``ctx`` has :class:`ir.CommCtxType`.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.comm_ctx.rank", [ctx], {}, actual_span)


def comm_ctx_nranks(ctx: _ir_core.Expr, *, span: Span | None = None) -> Call:
    """Build a ``pld.comm_ctx.nranks(ctx)`` Call returning ``ScalarType(INT32)``.

    Type verifier enforces that ``ctx`` has :class:`ir.CommCtxType`.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.comm_ctx.nranks", [ctx], {}, actual_span)


__all__ = ["comm_ctx_nranks", "comm_ctx_rank", "get_comm_ctx", "world_size"]
