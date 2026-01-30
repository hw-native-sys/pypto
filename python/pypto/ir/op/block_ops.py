# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Block operations for PyPTO IR.

Block operations work on TileType (unified buffer) and support block-level programming.
These operations include memory operations (load, store), element-wise operations,
unary operations, and reduction operations.
"""

from typing import Any, Dict, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span

from ..utils import _normalize_expr

# ============================================================================
# Memory Operations
# ============================================================================


def load(
    tensor: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    target_memory: int = 1,
) -> Call:
    """Copy data from tensor to specified memory level.

    Args:
        tensor: Source tensor (TensorType)
        row_offset: Row offset in the tensor (scalar)
        col_offset: Column offset in the tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        target_memory: Target memory space for the output tile.
                     1=UB (UB, default), 2=L1.

    Returns:
        Call expression that returns a TileType with the copied data
    """
    # Validate target_memory: only UB(1) and L1(2) are allowed for load
    if target_memory not in (1, 2):
        raise ValueError(f"target_memory for block.load must be 1 (UB) or 2 (L1), got {target_memory}")

    span = Span.unknown()
    args = [
        tensor,
        _normalize_expr(row_offset, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, int_dtype=DataType.INT32),
        _normalize_expr(height, int_dtype=DataType.INT32),
        _normalize_expr(width, int_dtype=DataType.INT32),
    ]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {"target_memory": target_memory}

    return _ir_core.create_op_call("block.load", args, kwargs, span)


def store(
    tile: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    output_tensor: Expr,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        row_offset: Row offset in the output tensor (scalar)
        col_offset: Column offset in the output tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        output_tensor: Output tensor (TensorType)

    Returns:
        Call expression that returns the output tensor
    """
    span = Span.unknown()
    args = [
        tile,
        _normalize_expr(row_offset, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, int_dtype=DataType.INT32),
        _normalize_expr(height, int_dtype=DataType.INT32),
        _normalize_expr(width, int_dtype=DataType.INT32),
        output_tensor,
    ]
    return _ir_core.create_op_call("block.store", args, {}, span)


def move(
    tile: Expr,
    target_memory: int,
    transpose: bool = False,
) -> Call:
    """Move tile between memory levels with optional transpose.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (1=UB, 2=L1, 3=L0A, 4=L0B)
        transpose: Whether to transpose the tile (default: False)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    span = Span.unknown()
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "target_memory": target_memory,
        "transpose": transpose,
    }

    return _ir_core.create_op_call("block.move", args, kwargs, span)


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise multiplication of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)

    Returns:
        Call expression for element-wise multiplication
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.mul", [lhs, rhs], {}, span)


def add(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise addition of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)

    Returns:
        Call expression for element-wise addition
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.add", [lhs, rhs], {}, span)


def div(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise division of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)

    Returns:
        Call expression for element-wise division
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.div", [lhs, rhs], {}, span)


def sub(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise subtraction of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)

    Returns:
        Call expression for element-wise subtraction
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.sub", [lhs, rhs], {}, span)


def muls(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.muls", [lhs, rhs_expr], {}, span)


def adds(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise addition with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.adds", [lhs, rhs_expr], {}, span)


def divs(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise division with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.divs", [lhs, rhs_expr], {}, span)


def subs(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.subs", [lhs, rhs_expr], {}, span)


# ============================================================================
# Unary Operations
# ============================================================================


def neg(tile: Expr) -> Call:
    """Element-wise negation of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise negation
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.neg", [tile], {}, span)


def exp(tile: Expr) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise exponential
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.exp", [tile], {}, span)


def recip(tile: Expr) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise reciprocal
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.recip", [tile], {}, span)


def sqrt(tile: Expr) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise square root
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.sqrt", [tile], {}, span)


def rsqrt(tile: Expr) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.rsqrt", [tile], {}, span)


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(lhs: Expr, rhs: Expr) -> Call:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile (TileType, 2D)
        rhs: Right-hand side tile (TileType, 2D)

    Returns:
        Call expression for matrix multiplication
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.matmul", [lhs, rhs], {}, span)


def matmul_acc(acc: Expr, lhs: Expr, rhs: Expr) -> Call:
    """Matrix multiplication with accumulation.

    Performs matrix multiplication and accumulates the result: acc = acc + lhs @ rhs.
    This is commonly used in loop-based matrix multiplication where results are
    accumulated over the K dimension.

    Args:
        acc: Accumulator tile (TileType, 2D) to accumulate into
        lhs: Left-hand side tile (TileType, 2D)
        rhs: Right-hand side tile (TileType, 2D)

    Returns:
        Call expression for matrix multiplication with accumulation
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.matmul_acc", [acc, lhs, rhs], {}, span)


# ============================================================================
# Reduction Operations
# ============================================================================


def max(tile: Expr, axis: int, keepdim: bool = False) -> Call:
    """Max reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)

    Returns:
        Call expression for max reduction
    """
    span = Span.unknown()
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.max", args, kwargs, span)


def row_max(tile: Expr) -> Call:
    """Row-wise max reduction of a 2D tile.

    This is a convenience function equivalent to max(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType, 2D)

    Returns:
        Call expression for row-wise max reduction
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.row_max", [tile], {}, span)


def row_sum(tile: Expr) -> Call:
    """Row-wise sum reduction of a 2D tile.

    This is a convenience function equivalent to sum(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType, 2D)

    Returns:
        Call expression for row-wise sum reduction
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.row_sum", [tile], {}, span)


# ============================================================================
# Row Broadcast Operations
# ============================================================================


def row_expand_sub(tile: Expr, row_vec: Expr) -> Call:
    """Row-wise broadcast subtraction.

    Subtracts a row vector from each row of the tile.
    tile[i, :] - row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType, 2D [M, N])
        row_vec: Row vector (TileType, 2D [M, 1])

    Returns:
        Call expression for row-wise broadcast subtraction
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.row_expand_sub", [tile, row_vec], {}, span)


def row_expand_div(tile: Expr, row_vec: Expr) -> Call:
    """Row-wise broadcast division.

    Divides each row of the tile by the corresponding row vector value.
    tile[i, :] / row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType, 2D [M, N])
        row_vec: Row vector (TileType, 2D [M, 1])

    Returns:
        Call expression for row-wise broadcast division
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.row_expand_div", [tile, row_vec], {}, span)


def row_expand_mul(tile: Expr, row_vec: Expr) -> Call:
    """Row-wise broadcast multiplication.

    Multiplies each row of the tile by the corresponding row vector value.
    tile[i, :] * row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType, 2D [M, N])
        row_vec: Row vector (TileType, 2D [M, 1])

    Returns:
        Call expression for row-wise broadcast multiplication
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.row_expand_mul", [tile, row_vec], {}, span)


def maximum(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise maximum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)

    Returns:
        Call expression for element-wise maximum
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.maximum", [lhs, rhs], {}, span)


# ============================================================================
# Reduction Operations (continued)
# ============================================================================


def sum(tile: Expr, axis: int, keepdim: bool = False) -> Call:
    """Sum reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)

    Returns:
        Call expression for sum reduction
    """

    span = Span.unknown()
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.sum", args, kwargs, span)
