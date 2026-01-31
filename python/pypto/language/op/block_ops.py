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

from typing import Union

from pypto.ir.op import block_ops as _ir_ops
from pypto.pypto_core.ir import Expr

from ..tensor import Tensor
from ..tile import Tile


def load(
    tensor: Tensor,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    target_memory: int = 1,
) -> Tile:
    """Copy data from tensor to unified buffer (tile).

    Args:
        tensor: Source tensor
        row_offset: Row offset in the tensor
        col_offset: Column offset in the tensor
        height: Height of the tile to copy
        width: Width of the tile to copy
        target_memory: Target memory space (1=UB default, 2=L1)

    Returns:
        Tile wrapping the load operation
    """
    call_expr = _ir_ops.load(tensor.unwrap(), row_offset, col_offset, height, width, target_memory)
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    output_tensor: Tensor,
) -> Tensor:
    """Copy data from tile back to tensor.

    Args:
        tile: Source tile
        row_offset: Row offset in the output tensor
        col_offset: Column offset in the output tensor
        height: Height of the tile to copy
        width: Width of the tile to copy
        output_tensor: Output tensor

    Returns:
        Tensor wrapping the store operation
    """
    call_expr = _ir_ops.store(tile.unwrap(), row_offset, col_offset, height, width, output_tensor.unwrap())
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


def adds(lhs: Tile, rhs: Union[int, float, Expr]) -> Tile:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the adds operation
    """
    call_expr = _ir_ops.adds(lhs.unwrap(), rhs)
    return Tile(expr=call_expr)


def subs(lhs: Tile, rhs: Union[int, float, Expr]) -> Tile:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the subs operation
    """
    call_expr = _ir_ops.subs(lhs.unwrap(), rhs)
    return Tile(expr=call_expr)


def muls(lhs: Tile, rhs: Union[int, float, Expr]) -> Tile:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the muls operation
    """
    call_expr = _ir_ops.muls(lhs.unwrap(), rhs)
    return Tile(expr=call_expr)


def divs(lhs: Tile, rhs: Union[int, float, Expr]) -> Tile:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the divs operation
    """
    call_expr = _ir_ops.divs(lhs.unwrap(), rhs)
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


def matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile (2D)
        rhs: Right-hand side tile (2D)

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


def row_max(tile: Tile) -> Tile:
    """Row-wise max reduction.

    Args:
        tile: Input tile (2D)

    Returns:
        Tile wrapping the row_max operation
    """
    call_expr = _ir_ops.row_max(tile.unwrap())
    return Tile(expr=call_expr)


def row_sum(tile: Tile) -> Tile:
    """Row-wise sum reduction.

    Args:
        tile: Input tile (2D)

    Returns:
        Tile wrapping the row_sum operation
    """
    call_expr = _ir_ops.row_sum(tile.unwrap())
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
