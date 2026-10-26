# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile row/column reductions and broadcasts."""

from pypto.ir.op import tile_ops as _ir_ops
from pypto.pypto_core.ir import (
    Expr,
)

from ..typing import Scalar, Tile


def row_max(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise max reduction.

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension

    Returns:
        Tile wrapping the row_max operation
    """
    call_expr = _ir_ops.row_max(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_sum(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise sum reduction.

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension

    Returns:
        Tile wrapping the row_sum operation
    """
    call_expr = _ir_ops.row_sum(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_min(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise min reduction.

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension

    Returns:
        Tile wrapping the row_min operation
    """
    call_expr = _ir_ops.row_min(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_prod(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise product reduction.

    Reduces the last axis with keepdim, producing output shape
    ``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
            every dimension at least as large as the corresponding input dimension

    Returns:
        Tile wrapping the row_prod operation
    """
    call_expr = _ir_ops.row_prod(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def col_sum(tile: Tile, tmp_tile: Tile | None = None) -> Tile:
    """Column-wise sum reduction.

    Output shape is ``[1, N]`` for an ``[M, N]`` input.

    Passing ``tmp_tile`` activates the binary-tree reduction path (O(log M) depth,
    better precision); omitting it uses the sequential path.

    Args:
        tile: Input tile
        tmp_tile: Optional scratch tile (same shape/dtype as input) that selects
            the binary-tree reduction path. Unlike the arg reductions, this is not
            enforced by type deduction -- pass the input's shape and dtype.

    Returns:
        Tile wrapping the col_sum operation
    """
    tmp_expr = None if tmp_tile is None else tmp_tile.unwrap()
    call_expr = _ir_ops.col_sum(tile.unwrap(), tmp_expr)
    return Tile(expr=call_expr)


def col_max(tile: Tile) -> Tile:
    """Column-wise max reduction.

    Output shape is ``[1, N]`` for an ``[M, N]`` input.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the col_max operation
    """
    call_expr = _ir_ops.col_max(tile.unwrap())
    return Tile(expr=call_expr)


def col_min(tile: Tile) -> Tile:
    """Column-wise min reduction.

    Output shape is ``[1, N]`` for an ``[M, N]`` input.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the col_min operation
    """
    call_expr = _ir_ops.col_min(tile.unwrap())
    return Tile(expr=call_expr)


def col_prod(tile: Tile) -> Tile:
    """Column-wise product reduction.

    Output shape is ``[1, N]`` for an ``[M, N]`` input.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the col_prod operation
    """
    call_expr = _ir_ops.col_prod(tile.unwrap())
    return Tile(expr=call_expr)


def row_argmax(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise argmax (column index of the per-row maximum, int32 output).

    Output shape is ``[rows, 1]`` with INT32 index dtype.

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

    Returns:
        Tile wrapping the row_argmax operation
    """
    call_expr = _ir_ops.row_argmax(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_argmin(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise argmin (column index of the per-row minimum, int32 output).

    Output shape is ``[rows, 1]`` with INT32 index dtype.

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

    Returns:
        Tile wrapping the row_argmin operation
    """
    call_expr = _ir_ops.row_argmin(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def col_argmax(tile: Tile, tmp_tile: Tile) -> Tile:
    """Column-wise argmax (row index of the per-column maximum, int32 output).

    Output shape is ``[1, N]`` with INT32 index dtype. Unlike [`col_max`][pypto.language.tile.col_max], the
    column argmax requires a ``tmp_tile`` scratch buffer.

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

    Returns:
        Tile wrapping the col_argmax operation
    """
    call_expr = _ir_ops.col_argmax(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def col_argmin(tile: Tile, tmp_tile: Tile) -> Tile:
    """Column-wise argmin (row index of the per-column minimum, int32 output).

    Output shape is ``[1, N]`` with INT32 index dtype. Unlike [`col_min`][pypto.language.tile.col_min], the
    column argmin requires a ``tmp_tile`` scratch buffer.

    Args:
        tile: Input tile
        tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

    Returns:
        Tile wrapping the col_argmin operation
    """
    call_expr = _ir_ops.col_argmin(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_expand(target: Tile, row_vec: Tile) -> Tile:
    """Expand row vector to target shape.

    Args:
        target: Target tile defining output shape [M, N]
        row_vec: Row vector to expand [M, 1]

    Returns:
        Tile wrapping the row_expand operation
    """
    call_expr = _ir_ops.row_expand(target.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_sub(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast subtraction.

    Subtracts a row vector from each row of the tile:
    ``tile[i, :] - row_vec[i, 0]`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_sub operation
    """
    call_expr = _ir_ops.row_expand_sub(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_div(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast division.

    Divides each row of the tile by the corresponding row vector value:
    ``tile[i, :] / row_vec[i, 0]`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_div operation
    """
    call_expr = _ir_ops.row_expand_div(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_mul(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast multiplication.

    Multiplies each row of the tile by the corresponding row vector value:
    ``tile[i, :] * row_vec[i, 0]`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_mul operation
    """
    call_expr = _ir_ops.row_expand_mul(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_add(tile: Tile, row_vec: Tile, tmp: Tile | None = None) -> Tile:
    """Row-wise scalar or packed-block expansion addition.

    A non-row-major ``[M, 1]`` carrier broadcasts one scalar per row:
    ``tile[i, :] + row_vec[i, 0]`` for all ``i``. A row-major carrier instead holds
    one 32-byte lane block per row and repeats that block across the destination
    columns.

    Args:
        tile: Input tile [M, N]
        row_vec: DN ``[M, 1]`` scalar carrier or row-major packed 32-byte carrier
        tmp: Optional PTOAS scratch tile

    Returns:
        Tile wrapping the row_expand_add operation
    """
    tmp_expr = None if tmp is None else tmp.unwrap()
    call_expr = _ir_ops.row_expand_add(tile.unwrap(), row_vec.unwrap(), tmp=tmp_expr)
    return Tile(expr=call_expr)


def col_expand(target: Tile, col_vec: Tile) -> Tile:
    """Expand column vector to target shape.

    Args:
        target: Target tile defining output shape [M, N]
        col_vec: Column vector to expand [1, N]

    Returns:
        Tile wrapping the col_expand operation
    """
    call_expr = _ir_ops.col_expand(target.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_mul(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and multiply with tile.

    Multiplies each column of the tile by the corresponding column vector value:
    ``tile[:, j] * col_vec[0, j]`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_mul operation
    """
    call_expr = _ir_ops.col_expand_mul(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_div(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and divide tile by it.

    Divides each column of the tile by the corresponding column vector value:
    ``tile[:, j] / col_vec[0, j]`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_div operation
    """
    call_expr = _ir_ops.col_expand_div(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_sub(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and subtract from tile.

    Subtracts a column vector from each column of the tile:
    ``tile[:, j] - col_vec[0, j]`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_sub operation
    """
    call_expr = _ir_ops.col_expand_sub(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_add(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and add to tile.

    Adds a column vector to each column of the tile:
    ``tile[:, j] + col_vec[0, j]`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_add operation
    """
    call_expr = _ir_ops.col_expand_add(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_max(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast maximum: max(tile, row_vec broadcasted).

    Takes the element-wise maximum of each row and the row vector value:
    ``max(tile[i, :], row_vec[i, 0])`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_max operation
    """
    call_expr = _ir_ops.row_expand_max(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_min(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast minimum: min(tile, row_vec broadcasted).

    Takes the element-wise minimum of each row and the row vector value:
    ``min(tile[i, :], row_vec[i, 0])`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_min operation
    """
    call_expr = _ir_ops.row_expand_min(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_expdif(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise exp-diff: exp(tile - row_vec) with per-row scalar.

    Computes ``exp(tile[i, :] - row_vec[i, 0])`` for all ``i``.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector providing per-row scalar [M, 1]

    Returns:
        Tile wrapping the row_expand_expdif operation
    """
    call_expr = _ir_ops.row_expand_expdif(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_max(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and take element-wise maximum with tile.

    Computes ``max(tile[:, j], col_vec[0, j])`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_max operation
    """
    call_expr = _ir_ops.col_expand_max(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_min(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and take element-wise minimum with tile.

    Computes ``min(tile[:, j], col_vec[0, j])`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_min operation
    """
    call_expr = _ir_ops.col_expand_min(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_expdif(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and compute exp-diff with per-column scalar.

    Computes ``exp(tile[:, j] - col_vec[0, j])`` for all ``j``.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector providing per-column scalar [1, N]

    Returns:
        Tile wrapping the col_expand_expdif operation
    """
    call_expr = _ir_ops.col_expand_expdif(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def expands(target: Tile, scalar: int | float | Expr | Scalar) -> Tile:
    """Expand scalar to target tile shape.

    Broadcasts a scalar value to match the shape of the target tile.

    Args:
        target: Target tile defining output shape
        scalar: Scalar value to expand

    Returns:
        Tile wrapping the expands operation
    """
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_ops.expands(target.unwrap(), scalar_expr)
    return Tile(expr=call_expr)
