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
- reserve_buffer / import_peer_buffer: Cross-core buffer management
"""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, PipeType, Span

from ..utils import _get_span_or_capture


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


# ============================================================================
# Cross-core communication operations
# ============================================================================


def tpush_to_aiv(tile: Expr, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe.

    Args:
        tile: Tile data to push
        aiv_idx: Target AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tpush_to_aiv", [tile], {"aiv_idx": aiv_idx}, actual_span)


def tpush_to_aic(tile: Expr, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe.

    Args:
        tile: Tile data to push
        aiv_idx: Source AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tpush_to_aic", [tile], {"aiv_idx": aiv_idx}, actual_span)


def tpop_from_aic(tile: Expr, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        tile: A tile expression used as a template for the shape and type of the data to pop.
        aiv_idx: Target AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tpop_from_aic", [tile], {"aiv_idx": aiv_idx}, actual_span)


def tpop_from_aiv(tile: Expr, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Pop tile data from AIV cross-core pipe into AIC.

    Args:
        tile: A tile expression used as a template for the shape and type of the data to pop.
        aiv_idx: Source AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tpop_from_aiv", [tile], {"aiv_idx": aiv_idx}, actual_span)


def aic_initialize_pipe(*, dir_mask: int, slot_size: int, span: Span | None = None) -> Call:
    """Initialize cross-core pipe on AIC side.

    Args:
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.aic_initialize_pipe", [], {"dir_mask": dir_mask, "slot_size": slot_size}, actual_span
    )


def aiv_initialize_pipe(*, dir_mask: int, slot_size: int, span: Span | None = None) -> Call:
    """Initialize cross-core pipe on AIV side.

    Args:
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.aiv_initialize_pipe", [], {"dir_mask": dir_mask, "slot_size": slot_size}, actual_span
    )


def reserve_buffer(*, name: str, size: int, span: Span | None = None) -> Call:
    """Reserve a named buffer for cross-core communication.

    Args:
        name: Buffer name
        size: Buffer size in bytes
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.reserve_buffer", [], {"name": name, "size": size}, actual_span)


def import_peer_buffer(*, name: str, peer_func: str, span: Span | None = None) -> Call:
    """Import a buffer from a peer function in the same group.

    Args:
        name: Buffer name to import
        peer_func: Name of the peer function that owns the buffer
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.import_peer_buffer", [], {"name": name, "peer_func": peer_func}, actual_span
    )
