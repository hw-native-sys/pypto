# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Asynchronous GM->L2 prefetch operations for PyPTO IR.

These ops express a latency-hiding cache hint: they warm L2 with a global-memory
region while unrelated compute proceeds, and change no tensor values.

- make_context: build a prefetch context from a GM INT8 scratch workspace
- async_prefetch: start one async GM -> L2 prefetch, returning a completion event
- session: project the async session bound to a context
- wait: wait for a prefetch event to complete within its session

Unlike most PTO intrinsics the underlying ``TPREFETCH_ASYNC`` carries no implicit
wait-event synchronization, so completion is explicit via the event/session pair.
"""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span

from ..utils import _get_span_or_capture


def make_context(workspace: Expr, span: Span | None = None) -> Call:
    """Build an asynchronous-prefetch context from a GM scratch workspace.

    Args:
        workspace: A GM scratch Tensor with INT8 element type, backing the SDMA path
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression producing a ``PrefetchAsyncContextType`` handle
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.make_context", [workspace], {}, actual_span)


def async_prefetch(src: Expr, ctx: Expr, span: Span | None = None) -> Call:
    """Start one asynchronous prefetch of a GM region into L2 cache.

    Does not block and does not modify ``src``.

    Args:
        src: A flat contiguous logical-1D GM Tensor to pull into L2
        ctx: An async-prefetch context from :func:`make_context`
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression producing an ``AsyncEventType`` completion event
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.async_prefetch", [src, ctx], {}, actual_span)


def session(ctx: Expr, span: Span | None = None) -> Call:
    """Project the asynchronous session bound to a prefetch context.

    Args:
        ctx: An async-prefetch context from :func:`make_context`
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression producing an ``AsyncSessionType`` handle
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.session", [ctx], {}, actual_span)


def wait(event: Expr, session_handle: Expr, span: Span | None = None) -> Call:
    """Wait for an asynchronous prefetch event to complete within its session.

    Args:
        event: A completion event from :func:`async_prefetch`
        session_handle: The matching async session from :func:`session`
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression producing a BOOL scalar done flag
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.wait", [event, session_handle], {}, actual_span)


__all__ = [
    "async_prefetch",
    "make_context",
    "session",
    "wait",
]
