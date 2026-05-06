# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO IR.

System operations handle hardware synchronization and cross-core communication:
- sync_src / sync_dst: Set/Wait flag-based synchronization between pipes
- bar_v / bar_m / bar_all: Barrier synchronization for vector, matrix, or all units
- tpush_to_aiv / tpush_to_aic: Push tile data across cores
- tpop_from_aic / tpop_from_aiv: Pop tile data from cross-core pipe
- aic_initialize_pipe / aiv_initialize_pipe: Initialize cross-core pipes
- reserve_buffer / import_peer_buffer: Cross-core buffer management (i32 SSA results)
"""

from typing import Protocol, runtime_checkable

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, PipeType, Span

from ..utils import _get_span_or_capture
from .tile_ops import (  # noqa: F401
    tpop_from_aic,
    tpop_from_aiv,
    tpush_to_aic,
    tpush_to_aiv,
)


@runtime_checkable
class _UnwrapsToExpr(Protocol):
    """Language wrappers (e.g. ``pl.Scalar[dtype]``) that expose ``unwrap() -> Expr``."""

    def unwrap(self) -> Expr: ...


def _create_sync_op(
    op_name: str,
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None,
) -> Call:
    """Create a flag-based synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.sync_src")
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}
    return _ir_core.create_op_call(op_name, [], kwargs, actual_span)


def _create_barrier_op(op_name: str, *, span: Span | None) -> Call:
    """Create a barrier synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.bar_v")
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, [], {}, actual_span)


def sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None = None,
) -> Call:
    """Send a synchronization signal (Set Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that will wait on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_src
    """
    return _create_sync_op(
        "system.sync_src", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


def sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Wait Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_dst
    """
    return _create_sync_op(
        "system.sync_dst", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


def bar_v(*, span: Span | None = None) -> Call:
    """Vector unit barrier."""
    return _create_barrier_op("system.bar_v", span=span)


def bar_m(*, span: Span | None = None) -> Call:
    """Matrix unit barrier."""
    return _create_barrier_op("system.bar_m", span=span)


def bar_all(*, span: Span | None = None) -> Call:
    """Global barrier synchronization."""
    return _create_barrier_op("system.bar_all", span=span)


# Sentinel value: compiler auto-assigns the buffer base address
AUTO: int = -1

PipeBufOperand = Expr | int | float | _UnwrapsToExpr


def _consumer_buf_operand(buf: PipeBufOperand, span: Span) -> Expr:
    """Build positional operand for pipe init: Expr passthrough; int (incl. 0 / ``AUTO``) -> ConstInt."""
    if isinstance(buf, Expr):
        return buf
    if isinstance(buf, _UnwrapsToExpr):
        return buf.unwrap()
    if isinstance(buf, float):
        return ConstInt(int(buf), DataType.INT32, span)
    return ConstInt(buf, DataType.INT32, span)


def _build_pipe_init_args(
    c2v_consumer_buf: PipeBufOperand,
    v2c_consumer_buf: PipeBufOperand,
    span: Span,
) -> list[Expr]:
    """Positional args (c2v_consumer_buf, v2c_consumer_buf) for aic/aiv_initialize_pipe."""
    return [
        _consumer_buf_operand(c2v_consumer_buf, span),
        _consumer_buf_operand(v2c_consumer_buf, span),
    ]


def aic_initialize_pipe(
    c2v_consumer_buf: PipeBufOperand = 0,
    v2c_consumer_buf: PipeBufOperand = 0,
    *,
    dir_mask: int,
    slot_size: int,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIC side.

    Args:
        c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {"dir_mask": dir_mask, "slot_size": slot_size}
    if id is not None:
        kwargs["id"] = id
    args = _build_pipe_init_args(c2v_consumer_buf, v2c_consumer_buf, actual_span)
    return _ir_core.create_op_call("system.aic_initialize_pipe", args, kwargs, actual_span)


def aiv_initialize_pipe(
    c2v_consumer_buf: PipeBufOperand = 0,
    v2c_consumer_buf: PipeBufOperand = 0,
    *,
    dir_mask: int,
    slot_size: int,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIV side.

    Args:
        c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {"dir_mask": dir_mask, "slot_size": slot_size}
    if id is not None:
        kwargs["id"] = id
    args = _build_pipe_init_args(c2v_consumer_buf, v2c_consumer_buf, actual_span)
    return _ir_core.create_op_call("system.aiv_initialize_pipe", args, kwargs, actual_span)


def reserve_buffer(*, name: str, size: int, base: int = AUTO, span: Span | None = None) -> Call:
    """Reserve a named buffer for cross-core communication.

    Result type is ``ScalarType(INT32)`` (PTO ``pto.reserve_buffer ... -> i32``).

    Args:
        name: Buffer name
        size: Buffer size in bytes
        base: Base address in local SRAM. Use AUTO (-1) to let the compiler
              pick a non-conflicting address, or an explicit integer for
              manual kernels.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.reserve_buffer", [], {"name": name, "size": size, "base": base}, actual_span
    )


def import_peer_buffer(*, name: str, peer_func: str, span: Span | None = None) -> Call:
    """Import a buffer from a peer function in the same group.

    Result type is ``ScalarType(INT32)`` (PTO ``pto.import_reserved_buffer ... -> i32``).

    Args:
        name: Buffer name to import
        peer_func: Name of the peer function that owns the buffer
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.import_peer_buffer", [], {"name": name, "peer_func": peer_func}, actual_span
    )


# ============================================================================
# Slot release operations (split consumer protocol)
# ============================================================================


def tfree_to_aic(tile: Expr, span: Span | None = None, *, id: int | None = None) -> Call:
    """Release ring buffer slot back to AIC producer.

    Called by AIV consumer after finishing with data from tpop_from_aic.

    Args:
        tile: Tile expression obtained from tpop_from_aic to release
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {}
    if id is not None:
        kwargs["id"] = id
    return _ir_core.create_op_call("system.tfree_to_aic", [tile], kwargs, actual_span)


def tfree_to_aiv(tile: Expr, span: Span | None = None, *, id: int | None = None) -> Call:
    """Release ring buffer slot back to AIV producer.

    Called by AIC consumer after finishing with data from tpop_from_aiv.

    Args:
        tile: Tile expression obtained from tpop_from_aiv to release
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {}
    if id is not None:
        kwargs["id"] = id
    return _ir_core.create_op_call("system.tfree_to_aiv", [tile], kwargs, actual_span)
