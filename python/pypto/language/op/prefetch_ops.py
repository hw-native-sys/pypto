# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.prefetch.*`` — asynchronous GM->L2 prefetch operations.

A latency-hiding cache hint: [`async_prefetch`][pypto.language.prefetch.async_prefetch] starts an SDMA-backed
pull of a global-memory region into L2 while unrelated compute proceeds, and
[`wait`][pypto.language.prefetch.wait] blocks until it lands. The prefetch changes no tensor values, so a
kernel is numerically identical with or without it — only performance differs.

Typical usage::

    ctx = pl.prefetch.make_context()
    evt = pl.prefetch.async_prefetch(x, ctx)
    session = pl.prefetch.session(ctx)
    ...                                            # unrelated compute overlaps
    pl.prefetch.wait(evt, session)                 # x is now resident in L2

The runtime owns and injects the SDMA scratch workspace. One-shot execution
enables SDMA automatically from generated artifact metadata; an explicitly
reused worker must be created with ``ChipWorker(enable_sdma=True)``. The current
runtime-provisioned path is covered on onboard a2a3. A runtime without an SDMA
provider fails during worker initialization; there is no fallback or no-op path.
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


def make_context() -> PrefetchAsyncContext:
    """Build an asynchronous-prefetch context with a runtime-injected workspace.

    The caller supplies no workspace. Codegen and the runtime bind the returned
    context to a hidden runtime-owned SDMA allocation.

    Returns:
        A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] handle to pass to
        [`async_prefetch`][pypto.language.prefetch.async_prefetch]
        and [`session`][pypto.language.prefetch.session].
    """
    return PrefetchAsyncContext(expr=_ir_prefetch.make_context())


def async_prefetch(src: Tensor, ctx: PrefetchAsyncContext) -> AsyncEvent:
    """Start one asynchronous prefetch of a GM region into L2 cache.

    Does not block and does not modify ``src``.

    Args:
        src: A flat contiguous logical-1D GM [`pl.Tensor`][pypto.language.Tensor] to pull into L2.
            The op verifier (C++) requires a fully static shape whose dimensions
            are all 1 except the last — e.g. ``[N]`` or ``[1, N]``.
        ctx: A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] from
            [`make_context`][pypto.language.prefetch.make_context].

    Returns:
        An [`AsyncEvent`][pypto.language.AsyncEvent] to pass to [`wait`][pypto.language.prefetch.wait] along
        with the session.
    """
    return AsyncEvent(expr=_ir_prefetch.async_prefetch(_unwrap(src), _unwrap(ctx)))


def session(ctx: PrefetchAsyncContext) -> AsyncSession:
    """Project the asynchronous session bound to a prefetch context.

    Args:
        ctx: A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] from
            [`make_context`][pypto.language.prefetch.make_context].

    Returns:
        An [`AsyncSession`][pypto.language.AsyncSession] to pass to [`wait`][pypto.language.prefetch.wait].
    """
    return AsyncSession(expr=_ir_prefetch.session(_unwrap(ctx)))


def wait(event: AsyncEvent, session_handle: AsyncSession) -> Scalar:
    """Wait for an asynchronous prefetch event to complete within its session.

    Call this before the hot loop that consumes the prefetched region so the
    data is resident in L2.

    Args:
        event: An [`AsyncEvent`][pypto.language.AsyncEvent] from
            [`async_prefetch`][pypto.language.prefetch.async_prefetch].
        session_handle: The matching [`AsyncSession`][pypto.language.AsyncSession] from
            [`session`][pypto.language.prefetch.session].

    Returns:
        A ``BOOL`` [`Scalar`][pypto.language.Scalar] done flag.
    """
    return Scalar(expr=_ir_prefetch.wait(_unwrap(event), _unwrap(session_handle)))


__all__ = [
    "async_prefetch",
    "make_context",
    "session",
    "wait",
]
