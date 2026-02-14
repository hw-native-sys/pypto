# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Block operations for PyPTO Language DSL.

This module provides type-safe wrappers around pypto.ir.op.block operations
that accept and return Tile types instead of raw Expr/Call objects.
"""

from typing import Literal, Sequence, Union

__all__ = [
    "create_tile",
    "load",
    "store",
    "l0c_store",
    "move",
    "ub_copy",
    "full",
    "get_block_idx",
    "add",
    "sub",
    "mul",
    "div",
    "adds",
    "subs",
    "muls",
    "divs",
    "neg",
    "exp",
    "sqrt",
    "rsqrt",
    "recip",
    "log",
    "abs",
    "relu",
    "cast",
    "matmul",
    "matmul_acc",
    "row_max",
    "row_sum",
    "row_min",
    "maximum",
    "row_expand_sub",
    "row_expand_div",
    "row_expand_mul",
    "row_expand_add",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "expands",
    "minimum",
    "cmp",
    "cmps",
    "sum",
    "max",
    "min",
    "view",
    "reshape",
    "transpose",
]

from pypto.ir.op import block_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..typing import Scalar, Tensor, Tile


def create_tile(
    shape: list[int],
    dtype: DataType,
    target_memory: int = 1,
) -> Tile:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        target_memory: Target memory level (1=UB, 2=L1, 3=L0A, 4=L0B)

    Returns:
        Tile wrapping the create_tile operation
    """
    call_expr = _ir_ops.create_tile(shape, dtype, target_memory)
    return Tile(expr=call_expr)


def load(
    tensor: Tensor,
    offsets: Sequence[int | Expr],
    shapes: Sequence[int | Expr],
    target_memory: int = 1,
) -> Tile:
    """Copy data from tensor to unified buffer (tile).

    Args:
        tensor: Source tensor
        offsets: Offsets in each dimension
        sizes: Shape of the tile in each dimension
        target_memory: Target memory space (1=UB default, 2=L1)

    Returns:
        Tile wrapping the load operation

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # 3D load
        >>> tile = load(tensor, offsets=[0, 0, 0], shapes=[8, 16, 32])
    """
    call_expr = _ir_ops.load(tensor.unwrap(), offsets, shapes, target_memory)
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    offsets: Sequence[int | Expr],
    shapes: Sequence[int | Expr],
    output_tensor: Tensor,
) -> Tensor:
    """Copy data from tile back to tensor.

    Args:
        tile: Source tile
        offsets: Offsets in each dimension
        sizes: Shape of the tile in each dimension
        output_tensor: Output tensor

    Returns:
        Tensor wrapping the store operation

    Example:
        >>> # 2D store
        >>> result = store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=tensor)
        >>> # 3D store
        >>> result = store(tile, offsets=[0, 0, 0], shapes=[8, 16, 32], output_tensor=tensor)
    """
    call_expr = _ir_ops.store(tile.unwrap(), offsets, shapes, output_tensor.unwrap())
    return Tensor(expr=call_expr)


def l0c_store(
    tile: Tile,
    offsets: Union[list[Union[int, Expr]], tuple[Union[int, Expr], ...]],
    shapes: Union[list[Union[int, Expr]], tuple[Union[int, Expr], ...]],
    output_tensor: Tensor,
) -> Tensor:
    """Copy data from L0C tile to GM tensor.

    Args:
        tile: Source tile
        offsets: Offsets in each dimension
        sizes: Shape of the tile in each dimension
        output_tensor: Output tensor

    Returns:
        Tensor wrapping the l0c_store operation

    Example:
        >>> # 2D l0c_store
        >>> result = l0c_store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=tensor)
        >>> # 3D l0c_store
        >>> result = l0c_store(tile, offsets=[0, 0, 0], shapes=[8, 16, 32], output_tensor=tensor)
    """
    call_expr = _ir_ops.l0c_store(tile.unwrap(), offsets, shapes, output_tensor.unwrap())
    return Tensor(expr=call_expr)


def move(tile: Tile, target_memory: int, transpose: bool = False) -> Tile:
    """Move tile between memory levels with optional transpose.

    Args:
        tile: Input tile
        target_memory: Target memory space (1=UB, 2=L1, 3=L0A, 4=L0B)
        transpose: Whether to transpose the tile

    Returns:
        Tile wrapping the move operation
    """
    call_expr = _ir_ops.move(tile.unwrap(), target_memory, transpose)
    return Tile(expr=call_expr)


def ub_copy(tile: Tile) -> Tile:
    """Copy tile within UB (Unified Buffer) memory.

    This is a specialized operation for copying tiles within UB memory (UB→UB).
    Both source and destination must be on UB. For other memory transfers,
    use move() with the target_memory parameter.

    Args:
        tile: Input tile (must be in UB memory)

    Returns:
        Tile wrapping the ub_copy operation (result is in UB memory)
    """
    call_expr = _ir_ops.ub_copy(tile.unwrap())
    return Tile(expr=call_expr)


def full(shape: list[int], dtype: DataType, value: Union[int, float]) -> Tile:
    """Create a tile from a shape and fill with value in UB.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        value: filling scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Tile wrapping the full operation
    """
    call_expr = _ir_ops.full(shape, dtype, value)
    return Tile(expr=call_expr)


def get_block_idx() -> Scalar:
    """Get the current block index.

    This operation returns the index of the current compute block. It is typically
    used in block-level programming to identify which block of data is being processed.

    Returns:
        Scalar wrapping the get_block_idx operation (UINT64 type)

    Example:
        >>> block_idx = pl.block.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    call_expr = _ir_ops.get_block_idx()
    return Scalar(expr=call_expr)


def add(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise addition of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the add operation
    """
    call_expr = _ir_ops.add(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def sub(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise subtraction of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the sub operation
    """
    call_expr = _ir_ops.sub(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def mul(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise multiplication of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the mul operation
    """
    call_expr = _ir_ops.mul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def div(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise division of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the div operation
    """
    call_expr = _ir_ops.div(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def adds(lhs: Tile, rhs: Union[int, float, Expr, Scalar]) -> Tile:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the adds operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.adds(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def subs(lhs: Tile, rhs: Union[int, float, Expr, Scalar]) -> Tile:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the subs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.subs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def muls(lhs: Tile, rhs: Union[int, float, Expr, Scalar]) -> Tile:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the muls operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.muls(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def divs(lhs: Tile, rhs: Union[int, float, Expr, Scalar]) -> Tile:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the divs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.divs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def neg(tile: Tile) -> Tile:
    """Element-wise negation.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the neg operation
    """
    call_expr = _ir_ops.neg(tile.unwrap())
    return Tile(expr=call_expr)


def exp(tile: Tile) -> Tile:
    """Element-wise exponential.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the exp operation
    """
    call_expr = _ir_ops.exp(tile.unwrap())
    return Tile(expr=call_expr)


def sqrt(tile: Tile) -> Tile:
    """Element-wise square root.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the sqrt operation
    """
    call_expr = _ir_ops.sqrt(tile.unwrap())
    return Tile(expr=call_expr)


def rsqrt(tile: Tile) -> Tile:
    """Element-wise reciprocal square root.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the rsqrt operation
    """
    call_expr = _ir_ops.rsqrt(tile.unwrap())
    return Tile(expr=call_expr)


def recip(tile: Tile) -> Tile:
    """Element-wise reciprocal.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the recip operation
    """
    call_expr = _ir_ops.recip(tile.unwrap())
    return Tile(expr=call_expr)


def log(tile: Tile) -> Tile:
    """Element-wise natural logarithm.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the log operation
    """
    call_expr = _ir_ops.log(tile.unwrap())
    return Tile(expr=call_expr)


def abs(tile: Tile) -> Tile:
    """Element-wise absolute value.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the abs operation
    """
    call_expr = _ir_ops.abs(tile.unwrap())
    return Tile(expr=call_expr)


def relu(tile: Tile) -> Tile:
    """Element-wise ReLU activation (max(0, x)).

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the relu operation
    """
    call_expr = _ir_ops.relu(tile.unwrap())
    return Tile(expr=call_expr)


def cast(
    tile: Tile,
    target_type: Union[int, DataType],
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
):
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Round Mode: None(0), RINT(1), ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6)

    Returns:
        Tile wrapping the cast operation
    """
    call_expr = _ir_ops.cast(tile.unwrap(), target_type, mode)
    return Tile(expr=call_expr)


def matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the matmul operation
    """
    call_expr = _ir_ops.matmul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def matmul_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """Matrix multiplication with accumulation: acc += lhs @ rhs.

    Args:
        acc: Accumulator tile
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the matmul_acc operation
    """
    call_expr = _ir_ops.matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def row_max(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise max reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_max operation
    """
    call_expr = _ir_ops.row_max(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_sum(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise sum reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_sum operation
    """
    call_expr = _ir_ops.row_sum(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_min(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise min reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_min operation
    """
    call_expr = _ir_ops.row_min(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def maximum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise maximum of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the maximum operation
    """
    call_expr = _ir_ops.maximum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def row_expand_sub(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast subtraction.

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

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_mul operation
    """
    call_expr = _ir_ops.row_expand_mul(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_add(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast addition.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_add operation
    """
    call_expr = _ir_ops.row_expand_add(tile.unwrap(), row_vec.unwrap())
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

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_sub operation
    """
    call_expr = _ir_ops.col_expand_sub(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def expands(target: Tile, scalar: Union[int, float, Expr, Scalar]) -> Tile:
    """Expand scalar to target tile shape.

    Args:
        target: Target tile defining output shape
        scalar: Scalar value to expand

    Returns:
        Tile wrapping the expands operation
    """
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_ops.expands(target.unwrap(), scalar_expr)
    return Tile(expr=call_expr)


def minimum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise minimum of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the minimum operation
    """
    call_expr = _ir_ops.minimum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def cmp(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile:
    """Element-wise comparison of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

    Returns:
        Tile wrapping the cmp operation
    """
    call_expr = _ir_ops.cmp(lhs.unwrap(), rhs.unwrap(), cmp_type)
    return Tile(expr=call_expr)


def cmps(lhs: Tile, rhs: Union[int, float, Expr, Scalar], cmp_type: int = 0) -> Tile:
    """Element-wise comparison of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value
        cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

    Returns:
        Tile wrapping the cmps operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.cmps(lhs.unwrap(), rhs_expr, cmp_type)
    return Tile(expr=call_expr)


def sum(tile: Tile, axis: int, keepdim: bool = False) -> Tile:
    """Sum reduction along specified axis.

    Args:
        tile: Input tile
        axis: Reduction axis (0 for rows, 1 for columns, -1 for last)
        keepdim: Whether to keep the reduced dimension as 1

    Returns:
        Tile wrapping the sum operation
    """
    call_expr = _ir_ops.sum(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


def max(tile: Tile, axis: int, keepdim: bool = False) -> Tile:
    """Max reduction along specified axis.

    Args:
        tile: Input tile
        axis: Reduction axis (0 for rows, 1 for columns, -1 for last)
        keepdim: Whether to keep the reduced dimension as 1

    Returns:
        Tile wrapping the max operation
    """
    call_expr = _ir_ops.max(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


def min(tile: Tile, axis: int, keepdim: bool = False) -> Tile:
    """Min reduction along specified axis.

    Args:
        tile: Input tile
        axis: Reduction axis (0 for rows, 1 for columns, -1 for last)
        keepdim: Whether to keep the reduced dimension as 1

    Returns:
        Tile wrapping the min operation
    """
    call_expr = _ir_ops.min(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


def view(tile: Tile, shape: list[Union[int, Expr]], offset: list[Union[int, Expr]]) -> Tile:
    """Create a view/slice of a tile with new shape and offset.

    Args:
        tile: Input tile
        shape: New shape dimensions (at most 2 for TileType)
        offset: Offset dimensions for the view

    Returns:
        Tile wrapping the view operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.view(tile_expr, shape, offset)
    return Tile(expr=call_expr)


def reshape(tile: Tile, shape: list[Union[int, Expr]]) -> Tile:
    """Reshape tile to new shape.

    Args:
        tile: Input tile
        shape: New shape dimensions (at most 2 for TileType)

    Returns:
        Tile wrapping the reshape operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.reshape(tile_expr, shape)
    return Tile(expr=call_expr)


def transpose(tile: Tile, axis1: int, axis2: int) -> Tile:
    """Transpose tile by swapping two axes.

    Args:
        tile: Input tile
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)

    Returns:
        Tile wrapping the transpose operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.transpose(tile_expr, axis1, axis2)
    return Tile(expr=call_expr)
