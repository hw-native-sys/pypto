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

from pypto.pypto_core import DataType
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


def _resolve_tpop_type(
    result_type: _ir_core.Type | None,
    shape: list[int] | None,
    dtype: DataType | None,
    memory_space: _ir_core.MemorySpace | None = None,
) -> _ir_core.Type | None:
    """Resolve the result type for a tpop op from explicit type or shape/dtype."""
    if result_type is not None and (shape is not None or dtype is not None):
        raise ValueError("result_type is mutually exclusive with shape/dtype")
    if (shape is None) != (dtype is None):
        raise ValueError("shape and dtype must both be provided or both omitted")
    if result_type is not None:
        return result_type
    if shape is not None and dtype is not None:
        return _ir_core.TileType(shape, dtype, None, None, memory_space)
    return None


def tpop_from_aic(
    *,
    result_type: _ir_core.Type | None = None,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    aiv_idx: int,
    span: Span | None = None,
) -> Call:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        result_type: Explicit result type (e.g. TileType). Mutually exclusive with shape/dtype.
        shape: Shape of the tile to receive (alternative to result_type).
        dtype: Data type of the tile to receive (alternative to result_type).
        aiv_idx: Target AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    resolved_type = _resolve_tpop_type(result_type, shape, dtype, _ir_core.MemorySpace.Vec)
    if resolved_type is not None:
        op = _ir_core.get_op("system.tpop_from_aic")
        return _ir_core.Call(op, [], {"aiv_idx": aiv_idx}, resolved_type, actual_span)
    return _ir_core.create_op_call("system.tpop_from_aic", [], {"aiv_idx": aiv_idx}, actual_span)


def tpop_from_aiv(
    *,
    result_type: _ir_core.Type | None = None,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    aiv_idx: int,
    span: Span | None = None,
) -> Call:
    """Pop tile data from AIV cross-core pipe into AIC.

    Args:
        result_type: Explicit result type (e.g. TileType). Mutually exclusive with shape/dtype.
        shape: Shape of the tile to receive (alternative to result_type).
        dtype: Data type of the tile to receive (alternative to result_type).
        aiv_idx: Source AIV core index
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    resolved_type = _resolve_tpop_type(result_type, shape, dtype, _ir_core.MemorySpace.Mat)
    if resolved_type is not None:
        op = _ir_core.get_op("system.tpop_from_aiv")
        return _ir_core.Call(op, [], {"aiv_idx": aiv_idx}, resolved_type, actual_span)
    return _ir_core.create_op_call("system.tpop_from_aiv", [], {"aiv_idx": aiv_idx}, actual_span)


# Sentinel value: compiler auto-assigns the buffer base address
AUTO: int = -1


def _build_pipe_kwargs(
    dir_mask: int,
    slot_size: int,
    c2v_consumer_buf: int,
    v2c_consumer_buf: int,
) -> dict[str, int]:
    """Build kwargs dict for pipe initialization, omitting AUTO (-1) consumer bufs."""
    kwargs: dict[str, int] = {"dir_mask": dir_mask, "slot_size": slot_size}
    if c2v_consumer_buf != AUTO:
        kwargs["c2v_consumer_buf"] = c2v_consumer_buf
    if v2c_consumer_buf != AUTO:
        kwargs["v2c_consumer_buf"] = v2c_consumer_buf
    return kwargs


def aic_initialize_pipe(
    *,
    dir_mask: int,
    slot_size: int,
    c2v_consumer_buf: int = AUTO,
    v2c_consumer_buf: int = AUTO,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIC side.

    Args:
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        c2v_consumer_buf: C2V consumer buffer base address (AUTO = not used)
        v2c_consumer_buf: V2C consumer buffer base address (AUTO = not used)
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = _build_pipe_kwargs(dir_mask, slot_size, c2v_consumer_buf, v2c_consumer_buf)
    return _ir_core.create_op_call("system.aic_initialize_pipe", [], kwargs, actual_span)


def aiv_initialize_pipe(
    *,
    dir_mask: int,
    slot_size: int,
    c2v_consumer_buf: int = AUTO,
    v2c_consumer_buf: int = AUTO,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIV side.

    Args:
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        c2v_consumer_buf: C2V consumer buffer base address (AUTO = not used)
        v2c_consumer_buf: V2C consumer buffer base address (AUTO = not used)
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = _build_pipe_kwargs(dir_mask, slot_size, c2v_consumer_buf, v2c_consumer_buf)
    return _ir_core.create_op_call("system.aiv_initialize_pipe", [], kwargs, actual_span)


def reserve_buffer(*, name: str, size: int, base: int = AUTO, span: Span | None = None) -> Call:
    """Reserve a named buffer for cross-core communication.

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


def tfree_to_aic(*, aiv_idx: int, span: Span | None = None) -> Call:
    """Release ring buffer slot back to AIC producer.

    Called by AIV consumer after finishing with data from tpop_from_aic.

    Args:
        aiv_idx: AIV core index releasing the slot
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tfree_to_aic", [], {"aiv_idx": aiv_idx}, actual_span)


def tfree_to_aiv(*, aiv_idx: int, span: Span | None = None) -> Call:
    """Release ring buffer slot back to AIV producer.

    Called by AIC consumer after finishing with data from tpop_from_aiv.

    Args:
        aiv_idx: AIV core index whose slot is being released
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.tfree_to_aiv", [], {"aiv_idx": aiv_idx}, actual_span)
