# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile block queries and cross-core transfers."""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    Expr,
    MemorySpace,
    Span,
)

from ..utils import (
    _get_span_or_capture,
)


def get_block_idx(span: Span | None = None) -> Call:
    """Get the current block index.

    This operation returns the index of the current compute tile. It is typically
    used in tile-level programming to identify which block of data is being processed.

    Args:
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns an INDEX scalar representing the block index

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.get_block_idx", [], {}, actual_span)


def get_subblock_idx(span: Span | None = None) -> Call:
    """Get the current sub-block (vector core) index.

    Returns the index of the current vector core within a split execution.

    Args:
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns an INDEX scalar representing the sub-block index
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.get_subblock_idx", [], {}, actual_span)


def get_block_num(span: Span | None = None) -> Call:
    """Get the total number of blocks in the current SPMD task.

    Args:
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns an INDEX scalar representing the total block count
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.get_block_num", [], {}, actual_span)


# ============================================================================
# Cross-core tpush / tpop operations
# ============================================================================


def _resolve_tpop_type(
    result_type: _ir_core.Type | None,
    shape: list[int] | None,
    dtype: DataType | None,
    memory_space: MemorySpace | None = None,
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


def tpush_to_aiv(
    tile: Expr,
    *,
    split: int,
    lane_stride: int | None = None,
    id: int | None = None,
    fixpipe_bf16: bool | None = None,
    fixpipe_relu: bool | None = None,
    span: Span | None = None,
) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe.

    Args:
        tile: Tile data to push
        split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
            axes over an odd extent)
        lane_stride: Partition stride carried when a ragged boundary was
            balanced across the two AIV lanes; omit for the box partition
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        fixpipe_bf16: Compiler-generated A5 BF16 C2V payload; must match the pipe initializers.
        fixpipe_relu: Compiler-generated ReLU before BF16 narrowing.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs: dict[str, Any] = {"split": split}
    if lane_stride is not None:
        kwargs["lane_stride"] = lane_stride
    if id is not None:
        kwargs["id"] = id
    if fixpipe_bf16 is not None:
        kwargs["fixpipe_bf16"] = fixpipe_bf16
    if fixpipe_relu is not None:
        kwargs["fixpipe_relu"] = fixpipe_relu
    return _ir_core.create_op_call("tile.tpush_to_aiv", [tile], kwargs, actual_span)


def tpush_to_aic(tile: Expr, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe.

    Args:
        tile: Tile data to push
        split: Split mode (0=none, 1=up-down, 2=left-right)
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {"split": split}
    if id is not None:
        kwargs["id"] = id
    return _ir_core.create_op_call("tile.tpush_to_aic", [tile], kwargs, actual_span)


def aiv_shard(tile: Expr, *, split: int, lane_stride: int | None = None, span: Span | None = None) -> Call:
    """Cross the AIC -> AIV boundary, halving on the split axis (full -> half).

    ``split=1`` / ``2`` halve the named axis. ``split=0`` (a task-parallel
    ``mode=NONE`` region) has no split axis: the op still marks the crossing and
    the result type preserves the operand's shape.

    Args:
        tile: Input tile (TileType; 2D unless split=0)
        split: Split mode (0=no split axis, 1=up-down/axis0, 2=left-right/axis1)
        lane_stride: Partition stride stamped by LowerAutoVectorSplit when it
            balances a ragged boundary across the two AIV lanes; omit for the
            default box partition
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs: dict[str, Any] = {"split": split}
    if lane_stride is not None:
        kwargs["lane_stride"] = lane_stride
    return _ir_core.create_op_call("tile.aiv_shard", [tile], kwargs, actual_span)


def aic_gather(tile: Expr, *, split: int, span: Span | None = None) -> Call:
    """Cross the AIV -> AIC boundary, rejoining on the split axis (half -> full).

    Inverse of :func:`aiv_shard`: ``split=1`` / ``2`` double the named axis, while
    ``split=0`` (a task-parallel ``mode=NONE`` region) marks the crossing and
    preserves the operand's shape.

    Args:
        tile: Input tile (TileType; 2D unless split=0)
        split: Split mode (0=no split axis, 1=up-down/axis0, 2=left-right/axis1)
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("tile.aic_gather", [tile], {"split": split}, actual_span)


def tpop_from_aic(
    *,
    result_type: _ir_core.Type | None = None,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    lane_stride: int | None = None,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        result_type: Explicit result type (e.g. TileType). Mutually exclusive with shape/dtype.
        shape: Shape of the tile to receive (alternative to result_type).
        dtype: Data type of the tile to receive (alternative to result_type).
        split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
            axes over an odd extent)
        lane_stride: Partition stride carried when a ragged boundary was
            balanced across the two AIV lanes; omit for the box partition
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    resolved_type = _resolve_tpop_type(result_type, shape, dtype, MemorySpace.Vec)
    kwargs: dict[str, Any] = {"split": split}
    if lane_stride is not None:
        kwargs["lane_stride"] = lane_stride
    if id is not None:
        kwargs["id"] = id
    if resolved_type is not None:
        op = _ir_core.get_op("tile.tpop_from_aic")
        return _ir_core.Call(op, [], kwargs, resolved_type, actual_span)
    return _ir_core.create_op_call("tile.tpop_from_aic", [], kwargs, actual_span)


def tpop_from_aiv(
    *,
    result_type: _ir_core.Type | None = None,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Pop tile data from AIV cross-core pipe into AIC.

    Args:
        result_type: Explicit result type (e.g. TileType). Mutually exclusive with shape/dtype.
        shape: Shape of the tile to receive (alternative to result_type).
        dtype: Data type of the tile to receive (alternative to result_type).
        split: Split mode (0=none, 1=up-down, 2=left-right)
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    resolved_type = _resolve_tpop_type(result_type, shape, dtype, MemorySpace.Mat)
    kwargs = {"split": split}
    if id is not None:
        kwargs["id"] = id
    if resolved_type is not None:
        op = _ir_core.get_op("tile.tpop_from_aiv")
        return _ir_core.Call(op, [], kwargs, resolved_type, actual_span)
    return _ir_core.create_op_call("tile.tpop_from_aiv", [], kwargs, actual_span)
