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

from typing import Any, Optional, Sequence, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, Span

from ..utils import _get_span_or_capture, _normalize_expr

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
    span: Optional[Span] = None,
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
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the copied data
    """
    # Validate target_memory: only UB(1) and L1(2) are allowed for load
    if target_memory not in (1, 2):
        raise ValueError(f"target_memory for block.load must be 1 (UB) or 2 (L1), got {target_memory}")

    actual_span = _get_span_or_capture(span)
    args = [
        tensor,
        _normalize_expr(row_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(height, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(width, actual_span, int_dtype=DataType.INT32),
    ]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {"target_memory": target_memory}

    return _ir_core.create_op_call("block.load", args, kwargs, actual_span)


def store(
    tile: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    output_tensor: Expr,
    span: Optional[Span] = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        row_offset: Row offset in the output tensor (scalar)
        col_offset: Column offset in the output tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        output_tensor: Output tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    args = [
        tile,
        _normalize_expr(row_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(height, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(width, actual_span, int_dtype=DataType.INT32),
        output_tensor,
    ]
    return _ir_core.create_op_call("block.store", args, {}, actual_span)


def l0c_store(
    tile: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    output_tensor: Expr,
    span: Optional[Span] = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        row_offset: Row offset in the output tensor (scalar)
        col_offset: Column offset in the output tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        output_tensor: Output tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    args = [
        tile,
        _normalize_expr(row_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(height, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(width, actual_span, int_dtype=DataType.INT32),
        output_tensor,
    ]
    return _ir_core.create_op_call("block.l0c_store", args, {}, actual_span)


def move(
    tile: Expr,
    target_memory: int,
    transpose: bool = False,
    span: Optional[Span] = None,
) -> Call:
    """Move tile between memory levels with optional transpose.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (1=UB, 2=L1, 3=L0A, 4=L0B)
        transpose: Whether to transpose the tile (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {
        "target_memory": target_memory,
        "transpose": transpose,
    }

    return _ir_core.create_op_call("block.move", args, kwargs, actual_span)


def zeros(
    shape: Sequence[Union[int, Expr]],
    dtype: DataType,
    span: Optional[Span] = None,
) -> Call:
    """Create a zero-initialized tile with specified shape and dtype.

    Args:
        shape: Shape of the tile (sequence of ints or scalar Exprs)
               Can be 1D, 2D, 3D, or higher dimensional
        dtype: Data type of the tile (DataType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a zero-initialized TileType

    Example:
        >>> # 2D tile
        >>> tile_2d = block.zeros([32, 128], DataType.FP32)
        >>> # Creates a [32, 128] tile filled with zeros

        >>> # 3D tile
        >>> tile_3d = block.zeros([8, 16, 32], DataType.FP16)
        >>> # Creates a [8, 16, 32] tile filled with zeros

        >>> # 1D tile
        >>> tile_1d = block.zeros([256], DataType.INT32)
        >>> # Creates a [256] tile filled with zeros
    """
    actual_span = _get_span_or_capture(span)

    # Normalize shape dimensions to Expr
    args = [_normalize_expr(dim, actual_span, int_dtype=DataType.INT32) for dim in shape]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {"dtype": dtype.code()}

    return _ir_core.create_op_call("block.zeros", args, kwargs, actual_span)


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.mul", [lhs, rhs], {}, actual_span)


def add(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise addition of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.add", [lhs, rhs], {}, actual_span)


def div(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise division of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.div", [lhs, rhs], {}, actual_span)


def sub(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sub", [lhs, rhs], {}, actual_span)


def muls(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.muls", [lhs, rhs_expr], {}, actual_span)


def adds(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.adds", [lhs, rhs_expr], {}, actual_span)


def divs(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.divs", [lhs, rhs_expr], {}, actual_span)


def subs(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.subs", [lhs, rhs_expr], {}, actual_span)


def cmp(lhs: Expr, rhs: Expr, cmp_type: int = 0, span: Optional[Span] = None) -> Call:
    """Element-wise comparison of two tiles (returns boolean tile).

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        cmp_type: Comparison type (int):
                  0 = GT (>), 1 = LT (<), 2 = GE (>=), 3 = LE (<=), 4 = EQ (==), 5 = NE (!=)
                  Default: 0 (greater than)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise comparison

    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("block.cmp", [lhs, rhs], kwargs, actual_span)


def cmps(
    lhs: Expr,
    rhs: Union[int, float, Expr],
    cmp_type: int = 0,
    span: Optional[Span] = None,
) -> Call:
    """Element-wise comparison of tile and scalar (returns boolean tile).

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        cmp_type: Comparison type (int):
                  0 = GT (>), 1 = LT (<), 2 = GE (>=), 3 = LE (<=), 4 = EQ (==), 5 = NE (!=)
                  Default: 0 (greater than)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise comparison with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("block.cmps", [lhs, rhs_expr], kwargs, actual_span)


# ============================================================================
# Unary Operations
# ============================================================================


def neg(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise negation of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise negation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.neg", [tile], {}, actual_span)


def exp(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.exp", [tile], {}, actual_span)


def recip(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.recip", [tile], {}, actual_span)


def sqrt(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sqrt", [tile], {}, actual_span)


def rsqrt(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.rsqrt", [tile], {}, actual_span)


def cast(tile: Expr, target_dtype: DataType, span: Optional[Span] = None) -> Call:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_dtype: Target data type (DataType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cast to target dtype

    Example:
        >>> tile_bf16 = ...  # TileType with BF16 dtype
        >>> tile_fp32 = block.cast(tile_bf16, DataType.FP32)
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"target_dtype": target_dtype.code()}
    return _ir_core.create_op_call("block.cast", [tile], kwargs, actual_span)


def log(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise natural logarithm of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise natural logarithm
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.log", [tile], {}, actual_span)


def abs(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise absolute value of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise absolute value
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.abs", [tile], {}, actual_span)


def relu(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise ReLU activation function (max(0, x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ReLU activation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.relu", [tile], {}, actual_span)


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.matmul", [lhs, rhs], {}, actual_span)


def matmul_acc(acc: Expr, lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Matrix multiplication with accumulation.

    Performs matrix multiplication and accumulates the result: acc = acc + lhs @ rhs.
    This is commonly used in loop-based matrix multiplication where results are
    accumulated over the K dimension.

    Args:
        acc: Accumulator tile (TileType) to accumulate into
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.matmul_acc", [acc, lhs, rhs], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


# ============================================================================
# Row Broadcast Operations
# ============================================================================


def row_expand_sub(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.row_expand_sub", [tile, row_vec], {}, actual_span)


def row_expand_div(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.row_expand_div", [tile, row_vec], {}, actual_span)


def row_expand_mul(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.row_expand_mul", [tile, row_vec], {}, actual_span)


def row_expand_add(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise broadcast addition.

    Adds a row vector to each row of the tile.
    tile[i, :] + row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_expand_add", [tile, row_vec], {}, actual_span)


def col_expand(target: Expr, col_vec: Expr, span: Optional[Span] = None) -> Call:
    """Expand column vector [1, cols] to target shape [rows, cols].

    Args:
        target: Target tile defining output shape (TileType [M, N])
        col_vec: Column vector to expand (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise expansion
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.col_expand", [target, col_vec], {}, actual_span)


def col_expand_mul(tile: Expr, col_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.col_expand_mul", [tile, col_vec], {}, actual_span)


def col_expand_div(tile: Expr, col_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.col_expand_div", [tile, col_vec], {}, actual_span)


def col_expand_sub(tile: Expr, col_vec: Expr, span: Optional[Span] = None) -> Call:
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
    return _ir_core.create_op_call("block.col_expand_sub", [tile, col_vec], {}, actual_span)


def expands(target: Expr, scalar: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
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
    scalar_expr = (
        _normalize_expr(scalar, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(scalar, Expr)
        else scalar
    )
    return _ir_core.create_op_call("block.expands", [target, scalar_expr], {}, actual_span)


def maximum(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise maximum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.maximum", [lhs, rhs], {}, actual_span)


def minimum(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise minimum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise minimum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.minimum", [lhs, rhs], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


def sum(tile: Expr, axis: int, keepdim: bool = False, span: Optional[Span] = None) -> Call:
    """Sum reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for sum reduction
    """

    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.sum", args, kwargs, actual_span)


def max(tile: Expr, axis: int, keepdim: bool = False, span: Optional[Span] = None) -> Call:
    """Max reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for max reduction
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.max", args, kwargs, actual_span)


def min(tile: Expr, axis: int, keepdim: bool = False, span: Optional[Span] = None) -> Call:
    """Min reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for min reduction
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.min", args, kwargs, actual_span)


def row_sum(tile: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise sum reduction (reduces along axis=1, maps to TROWSUM).

    Reduces each row to a single value, producing output shape [rows, 1].

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction (TileType [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_sum", [tile], {}, actual_span)


def row_max(tile: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise max reduction (reduces along axis=1, maps to TROWMAX).

    Reduces each row to a single value, producing output shape [rows, 1].

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise max reduction (TileType [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_max", [tile], {}, actual_span)


def row_min(tile: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise min reduction (reduces along axis=1, maps to TROWMIN).

    Reduces each row to a single value, producing output shape [rows, 1].

    Args:
        tile: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise min reduction (TileType [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_min", [tile], {}, actual_span)


# ============================================================================
# Transform Operations
# ============================================================================


def view(
    tile: Expr,
    shape: Sequence[Union[int, Expr]],
    offset: Sequence[Union[int, Expr]],
    span: Optional[Span] = None,
) -> Call:
    """Create a view/slice of a tile with new shape and offset.

    Args:
        tile: Input tile expression
        shape: New shape dimensions
        offset: Offset dimensions for the view
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tile view
    """
    actual_span = _get_span_or_capture(span)

    # Convert shape to MakeTuple
    shape_elements = [_normalize_expr(dim, actual_span, int_dtype=DataType.UINT64) for dim in shape]
    shape_tuple = _ir_core.MakeTuple(shape_elements, actual_span)

    # Convert offset to MakeTuple
    offset_elements = [_normalize_expr(off, actual_span, int_dtype=DataType.UINT64) for off in offset]
    offset_tuple = _ir_core.MakeTuple(offset_elements, actual_span)

    args = [tile, shape_tuple, offset_tuple]
    return _ir_core.create_op_call("block.view", args, {}, actual_span)


def reshape(tile: Expr, shape: Sequence[Union[int, Expr]], span: Optional[Span] = None) -> Call:
    """Reshape tile to new shape.

    Args:
        tile: Input tile expression
        shape: New shape dimensions
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile reshape
    """
    actual_span = _get_span_or_capture(span)

    # Convert shape to MakeTuple
    shape_elements = [_normalize_expr(dim, actual_span, int_dtype=DataType.UINT64) for dim in shape]
    shape_tuple = _ir_core.MakeTuple(shape_elements, actual_span)

    args = [tile, shape_tuple]
    return _ir_core.create_op_call("block.reshape", args, {}, actual_span)


def transpose(tile: Expr, axis1: int, axis2: int, span: Optional[Span] = None) -> Call:
    """Transpose tile by swapping two axes.

    Args:
        tile: Input tile expression
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile transpose
    """
    actual_span = _get_span_or_capture(span)

    # Create ConstInt for axis indices
    axis1_expr = ConstInt(axis1, DataType.INT32, actual_span)
    axis2_expr = ConstInt(axis2, DataType.INT32, actual_span)

    args = [tile, axis1_expr, axis2_expr]

    return _ir_core.create_op_call("block.transpose", args, {}, actual_span)
