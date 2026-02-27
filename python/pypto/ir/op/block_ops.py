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

from collections.abc import Sequence
from typing import Any, Literal

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstFloat, ConstInt, Expr, MemorySpace, Span

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def _validate_offsets_shapes(offsets_tuple: _ir_core.MakeTuple, shapes_tuple: _ir_core.MakeTuple) -> None:
    """Validate that offsets and shapes have matching, non-zero dimensions.

    Args:
        offsets_tuple: MakeTuple of offset expressions
        shapes_tuple: MakeTuple of shape expressions

    Raises:
        ValueError: If dimensions don't match or are empty
    """
    if len(offsets_tuple.elements) != len(shapes_tuple.elements):
        raise ValueError(
            f"offsets and shapes must have same number of dimensions, "
            f"got {len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )
    if len(offsets_tuple.elements) == 0:
        raise ValueError("offsets and shapes must have at least one dimension")


# ============================================================================
# Memory Operations
# ============================================================================


def create_tile(
    shape: Sequence[int] | _ir_core.MakeTuple,
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
    span: Span | None = None,
) -> Call:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile, or a MakeTuple
        dtype: Data type of the tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the created tile
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype, "target_memory": target_memory}
    return _ir_core.create_op_call("block.create_tile", [shape_tuple], kwargs, actual_span)


def load(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    target_memory: MemorySpace = MemorySpace.Vec,
    span: Span | None = None,
) -> Call:
    """Copy data from tensor to specified memory level.

    Args:
        tensor: Source tensor (TensorType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple
        shapes: Shape of the tile in each dimension (sequence of scalars), or a MakeTuple
        target_memory: Target memory space (MemorySpace.Vec default, or MemorySpace.Mat)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the copied data

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # 3D load
        >>> tile = load(tensor, offsets=[0, 0, 0], shapes=[8, 16, 32])
    """
    # Validate target_memory: only Vec and Mat are allowed for load
    if target_memory not in (MemorySpace.Vec, MemorySpace.Mat):
        raise ValueError(
            f"target_memory for block.load must be MemorySpace.Vec or MemorySpace.Mat, got {target_memory}"
        )

    actual_span = _get_span_or_capture(span)

    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_offsets_shapes(offsets_tuple, shapes_tuple)

    kwargs: dict[str, Any] = {"target_memory": target_memory}
    return _ir_core.create_op_call("block.load", [tensor, offsets_tuple, shapes_tuple], kwargs, actual_span)


def store(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    output_tensor: Expr,
    span: Span | None = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple
        shapes: Shape of the tile in each dimension (sequence of scalars), or a MakeTuple
        output_tensor: Output tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor

    Example:
        >>> # 2D store
        >>> result = store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=tensor)
        >>> # 3D store
        >>> result = store(tile, offsets=[0, 0, 0], shapes=[8, 16, 32], output_tensor=tensor)
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_offsets_shapes(offsets_tuple, shapes_tuple)

    return _ir_core.create_op_call(
        "block.store", [tile, offsets_tuple, shapes_tuple, output_tensor], {}, actual_span
    )


def l0c_store(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    output_tensor: Expr,
    span: Span | None = None,
) -> Call:
    """Copy data from L0C tile to GM tensor.

    Args:
        tile: Source tile (TileType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple
        shapes: Shape of the tile in each dimension (sequence of scalars), or a MakeTuple
        output_tensor: Output tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor

    Example:
        >>> # 2D l0c_store
        >>> result = l0c_store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=tensor)
        >>> # 3D l0c_store
        >>> result = l0c_store(tile, offsets=[0, 0, 0], shapes=[8, 16, 32], output_tensor=tensor)
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_offsets_shapes(offsets_tuple, shapes_tuple)

    return _ir_core.create_op_call(
        "block.l0c_store", [tile, offsets_tuple, shapes_tuple, output_tensor], {}, actual_span
    )


def move(
    tile: Expr,
    target_memory: MemorySpace,
    transpose: bool = False,
    span: Span | None = None,
) -> Call:
    """Move tile between memory levels with optional transpose.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)
        transpose: Whether to transpose the tile (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    kwargs: dict[str, Any] = {
        "target_memory": target_memory,
        "transpose": transpose,
    }

    return _ir_core.create_op_call("block.move", args, kwargs, actual_span)


def vec_move(
    tile: Expr,
    span: Span | None = None,
) -> Call:
    """Copy tile within Vec (vector/unified buffer) memory.

    This operation is specifically for Vec→Vec copies. Both source and destination
    must be on Vec memory. For other memory transfer patterns, use move().

    Args:
        tile: Input tile (TileType) in Vec memory
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in Vec memory space
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.vec_move", [tile], {}, actual_span)


def get_block_idx(span: Span | None = None) -> Call:
    """Get the current block index.

    This operation returns the index of the current compute block. It is typically
    used in block-level programming to identify which block of data is being processed.

    Args:
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a UINT64 scalar representing the block index

    Example:
        >>> block_idx = pl.block.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.get_block_idx", [], {}, actual_span)


def full(
    shape: Sequence[int] | _ir_core.MakeTuple,
    dtype: DataType,
    value: int | float,
    span: Span | None = None,
) -> Call:
    """Create a tile from a shape and fill with value in UB.

    Args:
        shape: Shape of the tile, or a MakeTuple
        dtype: Data type of the tile
        value: filling scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the created tile
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    if isinstance(value, int):
        value_expr = ConstInt(value, dtype, actual_span)
    else:
        value_expr = ConstFloat(value, dtype, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype}
    return _ir_core.create_op_call("block.full", [shape_tuple, value_expr], kwargs, actual_span)


def fillpad(tile: Expr, span: Span | None = None) -> Call:
    """Fill tile with padding for remaining elements.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the filled and padded tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.fillpad", [tile], {}, actual_span)


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def add(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def div(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def sub(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def rem(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise remainder (modulo) of two tiles.

    Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise remainder
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.rem", [lhs, rhs], {}, actual_span)


def rems(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise remainder (modulo) of tile and scalar.

    Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise remainder with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.rems", [lhs, rhs_expr], {}, actual_span)


def shl(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise left shift of two tiles.

    Computes lhs << rhs element-wise. Maps to the TSHL hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise left shift
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.shl", [lhs, rhs], {}, actual_span)


def shls(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise left shift of tile and scalar.

    Computes lhs << rhs element-wise. Maps to the TSHLS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar shift amount (int/Expr with INT32 ScalarType); must be >= 0
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise left shift with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32) if not isinstance(rhs, Expr) else rhs
    )
    return _ir_core.create_op_call("block.shls", [lhs, rhs_expr], {}, actual_span)


def shr(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise right shift of two tiles.

    Computes lhs >> rhs element-wise. Maps to the TSHR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise right shift
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.shr", [lhs, rhs], {}, actual_span)


def shrs(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise right shift of tile and scalar.

    Computes lhs >> rhs element-wise. Maps to the TSHRS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar shift amount (int/Expr with INT32 ScalarType); must be >= 0
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise right shift with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32) if not isinstance(rhs, Expr) else rhs
    )
    return _ir_core.create_op_call("block.shrs", [lhs, rhs_expr], {}, actual_span)


def and_(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise AND of two tiles.

    Computes lhs & rhs element-wise. Maps to the TAND hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise AND
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.and", [lhs, rhs], {}, actual_span)


def ands(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise AND of tile and scalar.

    Computes lhs & rhs element-wise. Maps to the TANDS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/Expr with INT32 ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise AND with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32) if not isinstance(rhs, Expr) else rhs
    )
    return _ir_core.create_op_call("block.ands", [lhs, rhs_expr], {}, actual_span)


def or_(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise OR of two tiles.

    Computes lhs | rhs element-wise. Maps to the TOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise OR
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.or", [lhs, rhs], {}, actual_span)


def ors(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise OR of tile and scalar.

    Computes lhs | rhs element-wise. Maps to the TORS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/Expr with INT32 ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise OR with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32) if not isinstance(rhs, Expr) else rhs
    )
    return _ir_core.create_op_call("block.ors", [lhs, rhs_expr], {}, actual_span)


def xor(lhs: Expr, rhs: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise XOR of two tiles.

    Computes lhs ^ rhs element-wise. Maps to the TXOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise XOR
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.xor", [lhs, rhs, tmp], {}, actual_span)


def xors(lhs: Expr, rhs: int | Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise XOR of tile and scalar.

    Computes lhs ^ rhs element-wise. Maps to the TXORS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/Expr with INT32 ScalarType)
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise XOR with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32) if not isinstance(rhs, Expr) else rhs
    )
    return _ir_core.create_op_call("block.xors", [lhs, rhs_expr, tmp], {}, actual_span)


def prelu(tile: Expr, slope: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise parametric ReLU of a tile.

    Computes prelu(tile, slope) element-wise. Maps to the TPRELU hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        slope: Slope tile (TileType) used for negative values
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise parametric ReLU
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.prelu", [tile, slope, tmp], {}, actual_span)


def addc(lhs: Expr, rhs: Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise addition of three tiles.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDC hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        rhs2: Third tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ternary addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.addc", [lhs, rhs, rhs2], {}, actual_span)


def subc(lhs: Expr, rhs: Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of three tiles.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBC hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        rhs2: Third tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ternary subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.subc", [lhs, rhs, rhs2], {}, actual_span)


def addsc(lhs: Expr, rhs: int | float | Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise addition of tile, scalar, and tile.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        rhs2: Third tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise tile-scalar-tile addition
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.addsc", [lhs, rhs_expr, rhs2], {}, actual_span)


def subsc(lhs: Expr, rhs: int | float | Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of tile, scalar, and tile.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        rhs2: Third tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise tile-scalar-tile subtraction
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.subsc", [lhs, rhs_expr, rhs2], {}, actual_span)


def lrelu(tile: Expr, slope: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise leaky ReLU of a tile with scalar slope.

    Computes max(x, slope * x) element-wise. Maps to the TLRELU hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        slope: Scalar slope for negative values (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise leaky ReLU
    """
    actual_span = _get_span_or_capture(span)
    slope_expr = (
        _normalize_expr(slope, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(slope, Expr)
        else slope
    )
    return _ir_core.create_op_call("block.lrelu", [tile, slope_expr], {}, actual_span)


def sel(mask: Expr, lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Per-element selection between two tiles using a predicate mask tile.

    For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
    Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

    Args:
        mask: Predicate mask tile (TileType); encoding is target-defined
        lhs: Source tile 0, selected where mask is true (TileType)
        rhs: Source tile 1, selected where mask is false (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for per-element tile selection
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sel", [mask, lhs, rhs], {}, actual_span)


def sels(lhs: Expr, rhs: Expr, select_mode: int | float | Expr, span: Span | None = None) -> Call:
    """Select between two tiles based on a scalar mode.

    Maps to the TSELS hardware intrinsic. The interpretation of select_mode values
    is target-dependent and enforced by codegen.

    Args:
        lhs: Source tile 0 (TileType)
        rhs: Source tile 1 (TileType)
        select_mode: Scalar select mode
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile select
    """
    actual_span = _get_span_or_capture(span)
    select_mode_expr = (
        _normalize_expr(select_mode, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(select_mode, Expr)
        else select_mode
    )
    return _ir_core.create_op_call("block.sels", [lhs, rhs, select_mode_expr], {}, actual_span)


def muls(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.muls", [lhs, rhs_expr], {}, actual_span)


def adds(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.adds", [lhs, rhs_expr], {}, actual_span)


def divs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.divs", [lhs, rhs_expr], {}, actual_span)


def subs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.subs", [lhs, rhs_expr], {}, actual_span)


def cmp(lhs: Expr, rhs: Expr, cmp_type: int = 0, span: Span | None = None) -> Call:
    """Element-wise comparison of two tiles (returns boolean tile).

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise comparison

    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("block.cmp", [lhs, rhs], kwargs, actual_span)


def cmps(
    lhs: Expr,
    rhs: int | float | Expr,
    cmp_type: int = 0,
    span: Span | None = None,
) -> Call:
    """Element-wise comparison of tile and scalar (returns boolean tile).

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise comparison with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("block.cmps", [lhs, rhs_expr], kwargs, actual_span)


# ============================================================================
# Unary Operations
# ============================================================================


def neg(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise negation of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise negation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.neg", [tile], {}, actual_span)


def exp(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.exp", [tile], {}, actual_span)


def recip(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.recip", [tile], {}, actual_span)


def sqrt(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sqrt", [tile], {}, actual_span)


def rsqrt(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.rsqrt", [tile], {}, actual_span)


def cast(
    tile: Expr,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
    span: Span | None = None,
) -> Call:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Round Mode: None(0), RINT(1), ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cast to target dtype

    Example:
        >>> tile_bf16 = ...  # TileType with BF16 dtype
        >>> tile_fp32 = block.cast(tile_bf16, DataType.FP32)
    """
    modes = {"none": 0, "rint": 1, "round": 2, "floor": 3, "ceil": 4, "trunc": 5, "odd": 6}
    mode_val = modes.get(mode)
    if mode_val is None:
        raise ValueError(f"Invalid rounding mode '{mode}'. Expected one of {list(modes.keys())}.")

    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"target_type": target_type, "mode": mode_val}
    return _ir_core.create_op_call("block.cast", [tile], kwargs, actual_span)


def log(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise natural logarithm of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise natural logarithm
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.log", [tile], {}, actual_span)


def abs(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise absolute value of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise absolute value
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.abs", [tile], {}, actual_span)


def relu(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise ReLU activation function (max(0, x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ReLU activation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.relu", [tile], {}, actual_span)


def not_(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise NOT of a tile.

    Computes ~tile element-wise. Maps to the TNOT hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise NOT
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.not", [tile], {}, actual_span)


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def matmul_acc(acc: Expr, lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def matmul_bias(lhs: Expr, rhs: Expr, bias: Expr, span: Span | None = None) -> Call:
    """Matrix multiplication with bias add: C = lhs @ rhs + bias.

    Args:
        lhs: Left-hand side tile (TileType [M, K])
        rhs: Right-hand side tile (TileType [K, N])
        bias: Bias tile (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication with bias
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.matmul_bias", [lhs, rhs, bias], {}, actual_span)


def gemv(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

    Args:
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for GEMV
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.gemv", [lhs, rhs], {}, actual_span)


def gemv_acc(acc: Expr, lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

    Args:
        acc: Accumulator tile (TileType [1, N])
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for GEMV with accumulation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.gemv_acc", [acc, lhs, rhs], {}, actual_span)


def gemv_bias(lhs: Expr, rhs: Expr, bias: Expr, span: Span | None = None) -> Call:
    """GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

    Args:
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        bias: Bias tile (TileType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for GEMV with bias
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.gemv_bias", [lhs, rhs, bias], {}, actual_span)


# ============================================================================
# Row Broadcast Operations
# ============================================================================


def row_expand(src: Expr, span: Span | None = None) -> Call:
    """Broadcast the first element of each source row across the destination row.

    For each element (i, j) in the valid region: dst[i, j] = src[i, 0].

    Args:
        src: Input tile (TileType [M, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise first-element broadcast
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_expand", [src], {}, actual_span)


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
    return _ir_core.create_op_call("block.row_expand_sub", [tile, row_vec], {}, actual_span)


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
    return _ir_core.create_op_call("block.row_expand_div", [tile, row_vec], {}, actual_span)


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
    return _ir_core.create_op_call("block.row_expand_mul", [tile, row_vec], {}, actual_span)


def row_expand_add(tile: Expr, row_vec: Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("block.col_expand", [target, col_vec], {}, actual_span)


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
    return _ir_core.create_op_call("block.col_expand_mul", [tile, col_vec], {}, actual_span)


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
    return _ir_core.create_op_call("block.col_expand_div", [tile, col_vec], {}, actual_span)


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
    return _ir_core.create_op_call("block.col_expand_sub", [tile, col_vec], {}, actual_span)


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
    scalar_expr = (
        _normalize_expr(scalar, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(scalar, Expr)
        else scalar
    )
    return _ir_core.create_op_call("block.expands", [target, scalar_expr], {}, actual_span)


def maximum(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def minimum(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def maxs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise maximum of tile and scalar.

    Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.maxs", [lhs, rhs_expr], {}, actual_span)


def mins(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise minimum of tile and scalar.

    Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise minimum with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.mins", [lhs, rhs_expr], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


def sum(tile: Expr, tmp_tile: Expr, axis: int, keepdim: bool = False, span: Span | None = None) -> Call:
    """Sum reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        tmp_tile: Temporary tile for reduction workspace
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for sum reduction
    """

    actual_span = _get_span_or_capture(span)
    args = [tile, tmp_tile]

    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.sum", args, kwargs, actual_span)


def max(tile: Expr, axis: int, keepdim: bool = False, span: Span | None = None) -> Call:
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

    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.max", args, kwargs, actual_span)


def min(tile: Expr, axis: int, keepdim: bool = False, span: Span | None = None) -> Call:
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

    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.min", args, kwargs, actual_span)


def row_max(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise max reduction of a tile.

    This is a convenience function equivalent to max(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType)
        tmp_tile: Temporary tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise max reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_max", [tile, tmp_tile], {}, actual_span)


def row_sum(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise sum reduction of a tile.

    This is a convenience function equivalent to sum(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType)
        tmp_tile: Temporary tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_sum", [tile, tmp_tile], {}, actual_span)


def row_min(tile: Expr, tmp_tile: Expr, span: Span | None = None) -> Call:
    """Row-wise min reduction (reduces along axis=1, maps to TROWMIN).

    Reduces each row to a single value, producing output shape [rows, 1].

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Temporary tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise min reduction (TileType [M, 1])
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_min", [tile, tmp_tile], {}, actual_span)


# ============================================================================
# Transform Operations
# ============================================================================


def view(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Create a view/slice of a tile with new shape and offset.

    Args:
        tile: Input tile expression
        shape: New shape dimensions, or a MakeTuple
        offset: Offset dimensions for the view, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tile view
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)
    offset_tuple = _to_make_tuple(offset, actual_span)

    args = [tile, shape_tuple, offset_tuple]
    return _ir_core.create_op_call("block.view", args, {}, actual_span)


def reshape(tile: Expr, shape: Sequence[int | Expr] | _ir_core.MakeTuple, span: Span | None = None) -> Call:
    """Reshape tile to new shape.

    Args:
        tile: Input tile expression
        shape: New shape dimensions, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile reshape
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [tile, shape_tuple]
    return _ir_core.create_op_call("block.reshape", args, {}, actual_span)


def transpose(tile: Expr, axis1: int, axis2: int, span: Span | None = None) -> Call:
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
    axis1_expr = ConstInt(axis1, DataType.INDEX, actual_span)
    axis2_expr = ConstInt(axis2, DataType.INDEX, actual_span)

    args = [tile, axis1_expr, axis2_expr]

    return _ir_core.create_op_call("block.transpose", args, {}, actual_span)
