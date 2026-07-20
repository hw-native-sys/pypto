# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.prefetch.*`` — asynchronous GM->L2 prefetch operations.

A latency-hiding cache hint: :func:`async_prefetch` starts an SDMA-backed pull of
a global-memory region into L2 while unrelated compute proceeds, and
:func:`wait` blocks until it lands. The prefetch changes no tensor values, so a
kernel is numerically identical with or without it — only performance differs.

Typical usage::

    ctx = pl.prefetch.make_context(ws)             # ws: pl.Tensor[[K], pl.INT8]
    evt = pl.prefetch.async_prefetch(x, ctx)
    session = pl.prefetch.session(ctx)
    ...                                            # unrelated compute overlaps
    pl.prefetch.wait(evt, session)                 # x is now resident in L2

The SDMA CMO path is only effective on A3/A5; on other targets PTOAS degrades
the prefetch to a functional no-op, so kernels using these ops stay portable.
"""

from typing import Any

from pypto.ir.op import prefetch_ops as _ir_prefetch

from ..typing.prefetch_handle import AsyncEvent, AsyncSession, PrefetchAsyncContext
from ..typing.scalar import Scalar
from ..typing.tensor import Tensor


def _unwrap(value: Any) -> Any:
    """Unwrap a DSL wrapper (Tensor / handle / ...) to ``ir.Expr``.

    Falls through unchanged for raw ``ir.Expr`` values, which the parser may
    already have unwrapped at the DSL boundary.
    """
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def make_context(workspace: Tensor) -> PrefetchAsyncContext:
    """Build an asynchronous-prefetch context from a GM scratch workspace.

    Args:
        workspace: A GM scratch :class:`pl.Tensor` with ``INT8`` element type.
            The op verifier (C++) rejects any other element type — the SDMA path
            needs raw bytes.

    Returns:
        A :class:`PrefetchAsyncContext` handle to pass to :func:`async_prefetch`
        and :func:`session`.
    """
    return PrefetchAsyncContext(expr=_ir_prefetch.make_context(_unwrap(workspace)))


def async_prefetch(src: Tensor, ctx: PrefetchAsyncContext) -> AsyncEvent:
    """Start one asynchronous prefetch of a GM region into L2 cache.

    Does not block and does not modify ``src``.

    Args:
        src: A flat contiguous logical-1D GM :class:`pl.Tensor` to pull into L2.
            The op verifier (C++) requires a fully static shape whose dimensions
            are all 1 except the last — e.g. ``[N]`` or ``[1, N]``.
        ctx: A :class:`PrefetchAsyncContext` from :func:`make_context`.

    Returns:
        An :class:`AsyncEvent` to pass to :func:`wait` along with the session.
    """
    return AsyncEvent(expr=_ir_prefetch.async_prefetch(_unwrap(src), _unwrap(ctx)))


def session(ctx: PrefetchAsyncContext) -> AsyncSession:
    """Project the asynchronous session bound to a prefetch context.

    Args:
        ctx: A :class:`PrefetchAsyncContext` from :func:`make_context`.

    Returns:
        An :class:`AsyncSession` to pass to :func:`wait`.
    """
    return AsyncSession(expr=_ir_prefetch.session(_unwrap(ctx)))


def wait(event: AsyncEvent, session_handle: AsyncSession) -> Scalar:
    """Wait for an asynchronous prefetch event to complete within its session.

    Call this before the hot loop that consumes the prefetched region so the
    data is resident in L2.

    Args:
        event: An :class:`AsyncEvent` from :func:`async_prefetch`.
        session_handle: The matching :class:`AsyncSession` from :func:`session`.

    Returns:
        A ``BOOL`` :class:`Scalar` done flag.
    """
    return Scalar(expr=_ir_prefetch.wait(_unwrap(event), _unwrap(session_handle)))


__all__ = [
    "async_prefetch",
    "make_context",
    "session",
    "wait",
]
