# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile row/column reductions and broadcasts."""

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    Expr,
    Span,
)

from ..utils import (
    _get_span_or_capture,
    _normalize_scalar_operand,
)

# ============================================================================
# Row Broadcast Operations
# ============================================================================


def row_expand(target: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Expand row vector [rows, 1] to target shape [rows, cols].

    Args:
        target: Target tile defining output shape (TileType [M, N])
        row_vec: Row vector to expand (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise expansion
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand", [target, row_vec], {}, actual_span)


def row_expand_sub(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast subtraction.

    Subtracts a row vector from each row of the tile.
    tile[i, :] - row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_sub", [tile, row_vec], {}, actual_span)


def row_expand_div(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast division.

    Divides each row of the tile by the corresponding row vector value.
    tile[i, :] / row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_div", [tile, row_vec], {}, actual_span)


def row_expand_mul(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast multiplication.

    Multiplies each row of the tile by the corresponding row vector value.
    tile[i, :] * row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_mul", [tile, row_vec], {}, actual_span)


def row_expand_add(
    tile: Expr,
    row_vec: Expr,
    span: Span | None = None,
    *,
    tmp: Expr | None = None,
) -> Call:
    """Row-wise broadcast addition.

    A non-row-major ``[M, 1]`` carrier broadcasts one scalar per row. A
    row-major carrier holds one 32-byte lane block per row and repeats that
    block across the destination columns.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: DN row scalar carrier or row-major packed 32-byte carrier
        span: Optional source span for debugging (auto-captured if not provided)
        tmp: Optional PTOAS scratch tile

    Returns:
        Call expression for row-wise broadcast addition
    """
    actual_span = _get_span_or_capture(span)
    args = [tile, row_vec] if tmp is None else [tile, row_vec, tmp]
    return _ir_core.create_op_call("tile.row_expand_add", args, {}, actual_span)


def row_expand_max(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast maximum.

    Takes the element-wise maximum of each row and the row vector value.
    max(tile[i, :], row_vec[i, 0]) for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_max", [tile, row_vec], {}, actual_span)


def row_expand_min(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast minimum.

    Takes the element-wise minimum of each row and the row vector value.
    min(tile[i, :], row_vec[i, 0]) for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast minimum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_min", [tile, row_vec], {}, actual_span)


def row_expand_expdif(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise exp-diff with per-row scalar.

    Computes exp(tile[i, :] - row_vec[i, 0]) for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector providing per-row scalar (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise exp-diff
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_expand_expdif", [tile, row_vec], {}, actual_span)


def col_expand(target: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector [1, cols] to target shape [rows, cols].

    Args:
        target: Target tile defining output shape (TileType [M, N])
        col_vec: Column vector to expand (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise expansion
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand", [target, col_vec], {}, actual_span)


def col_expand_mul(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and multiply with target tile.

    Multiplies each column of the tile by the corresponding column vector value.
    tile[:, j] * col_vec[0, j] for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_mul", [tile, col_vec], {}, actual_span)


def col_expand_div(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and divide target tile by it.

    Divides each column of the tile by the corresponding column vector value.
    tile[:, j] / col_vec[0, j] for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_div", [tile, col_vec], {}, actual_span)


def col_expand_sub(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and subtract from target tile.

    Subtracts a column vector from each column of the tile.
    tile[:, j] - col_vec[0, j] for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_sub", [tile, col_vec], {}, actual_span)


def col_expand_max(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and take element-wise maximum with target tile.

    max(tile[:, j], col_vec[0, j]) for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_max", [tile, col_vec], {}, actual_span)


def col_expand_min(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and take element-wise minimum with target tile.

    min(tile[:, j], col_vec[0, j]) for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast minimum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_min", [tile, col_vec], {}, actual_span)


def col_expand_expdif(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and compute exp-diff with per-column scalar.

    Computes exp(tile[:, j] - col_vec[0, j]) for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector providing per-column scalar (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise exp-diff
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_expdif", [tile, col_vec], {}, actual_span)


def col_expand_add(tile: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Expand column vector and add to target tile.

    Adds a column vector to each column of the tile.
    tile[:, j] + col_vec[0, j] for all j.

    Args:
        tile: Input tile (TileType [M, N])
        col_vec: Column vector (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_expand_add", [tile, col_vec], {}, actual_span)


def expands(target: Expr, scalar: int | float | Expr, span: Span | None = None) -> Call:
    """Expand scalar to target tile shape.

    Broadcasts a scalar value to match the shape of the target tile.

    Args:
        target: Target tile defining output shape (TileType)
        scalar: Scalar value to expand (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for scalar expansion
    """
    actual_span = _get_span_or_capture(span)
    scalar_expr = _normalize_scalar_operand(target, scalar, actual_span, fallback_int_dtype=DataType.FP32)
    return _ir_core.create_op_call("tile.expands", [target, scalar_expr], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


def row_max(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise max reduction of a tile (reduces along the last axis, maps to TROWMAX).

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile (TileType)
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise max reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_max", [tile, tmp_tile], {}, actual_span)


def row_sum(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise sum reduction of a tile (reduces along the last axis, maps to TROWSUM).

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile (TileType)
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_sum", [tile, tmp_tile], {}, actual_span)


def row_min(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise min reduction (reduces along the last axis, maps to TROWMIN).

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile (TileType, e.g. [M, N])
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise min reduction (TileType, e.g. [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_min", [tile, tmp_tile], {}, actual_span)


def row_prod(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise product reduction (reduces along the last axis, maps to TROWPROD).

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile (TileType, e.g. [M, N])
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise product reduction (TileType, e.g. [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_prod", [tile, tmp_tile], {}, actual_span)


def col_sum(tile: Expr, tmp_tile: Expr | None = None, span: Span | None = None) -> Call:
    """Column-wise sum reduction of a tile (reduces along axis=0, maps to TCOLSUM).

    Output shape is [1, N] for an [M, N] input.

    Passing ``tmp_tile`` activates the binary-tree reduction path (O(log M) depth,
    better precision). Omitting ``tmp_tile`` emits the sequential reduction path.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Optional scratch tile (TileType, same shape/dtype as input) that
            activates binary-tree reduction. Unlike the arg reductions, this is
            not enforced by type deduction -- pass the input's shape and dtype.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise sum reduction (TileType [1, N])
    """
    actual_span = _get_span_or_capture(span)
    args = [tile] if tmp_tile is None else [tile, tmp_tile]
    return _ir_core.create_op_call("tile.col_sum", args, {}, actual_span)


def col_max(tile: Expr, span: Span | None = None) -> Call:
    """Column-wise max reduction of a tile (reduces along axis=0, maps to TCOLMAX).

    Output shape is [1, N] for an [M, N] input.

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise max reduction (TileType [1, N])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_max", [tile], {}, actual_span)


def col_min(tile: Expr, span: Span | None = None) -> Call:
    """Column-wise min reduction of a tile (reduces along axis=0, maps to TCOLMIN).

    Output shape is [1, N] for an [M, N] input.

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise min reduction (TileType [1, N])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_min", [tile], {}, actual_span)


def col_prod(tile: Expr, span: Span | None = None) -> Call:
    """Column-wise product reduction (reduces along axis=0, maps to TCOLPROD).

    Output shape is [1, N] for an [M, N] input.

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise product reduction (TileType [1, N])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_prod", [tile], {}, actual_span)


def row_argmax(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise argmax (column index of the per-row maximum, maps to TROWARGMAX).

    Output shape is [rows, 1] with int32 index dtype.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise argmax (TileType [M, 1], int32)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_argmax", [tile, tmp_tile], {}, actual_span)


def row_argmin(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise argmin (column index of the per-row minimum, maps to TROWARGMIN).

    Output shape is [rows, 1] with int32 index dtype.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise argmin (TileType [M, 1], int32)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.row_argmin", [tile, tmp_tile], {}, actual_span)


def col_argmax(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Column-wise argmax (row index of the per-column maximum, maps to TCOLARGMAX).

    Output shape is [1, N] with int32 index dtype. Unlike col_max, the column
    argmax requires a tmp scratch tile.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Scratch tile (TileType) with exactly the same shape and dtype as
            ``tile`` -- the kernel reads the column count from the tmp/src extent
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise argmax (TileType [1, N], int32)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_argmax", [tile, tmp_tile], {}, actual_span)


def col_argmin(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Column-wise argmin (row index of the per-column minimum, maps to TCOLARGMIN).

    Output shape is [1, N] with int32 index dtype. Unlike col_min, the column
    argmin requires a tmp scratch tile.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Scratch tile (TileType) with exactly the same shape and dtype as
            ``tile`` -- the kernel reads the column count from the tmp/src extent
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise argmin (TileType [1, N], int32)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.col_argmin", [tile, tmp_tile], {}, actual_span)
