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
) -> Call:
    """Copy data from tensor to unified buffer (tile).

    Args:
        tensor: Source tensor (TensorType)
        row_offset: Row offset in the tensor (scalar)
        col_offset: Column offset in the tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)

    Returns:
        Call expression that returns a TileType with the copied data
    """
    span = Span.unknown()
    args = [
        tensor,
        _normalize_expr(row_offset, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, int_dtype=DataType.INT32),
        _normalize_expr(height, int_dtype=DataType.INT32),
        _normalize_expr(width, int_dtype=DataType.INT32),
    ]
    return _ir_core.create_op_call("block.load", args, {}, span)


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


def sqrt(tile: Expr) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)

    Returns:
        Call expression for element-wise square root
    """
    span = Span.unknown()
    return _ir_core.create_op_call("block.sqrt", [tile], {}, span)


# ============================================================================
# Reduction Operations
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
