# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile operations for PyPTO IR.

Tile operations work on TileType (unified buffer) and support tile-level programming.
These operations include memory operations (load, store), element-wise operations,
unary operations, and reduction operations.
"""

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    ConstFloat,
    ConstInt,
    Expr,
    MemorySpace,
    PadValue,
    ScalarType,
    Span,
    TileLayout,
)

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple, resolve_cast_mode
from ._pad_value import normalize_pad_value


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


def _normalize_tile_binary_rhs(rhs: int | float | Expr, span: Span) -> Expr:
    """Normalize a tile binary-op rhs into an IR expression."""
    return (
        _normalize_expr(rhs, span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )


def _create_tile_binary_call(
    tile_op_name: str,
    scalar_op_name: str,
    lhs: Expr,
    rhs: int | float | Expr,
    span: Span,
) -> Call:
    """Create a tile binary call with scalar auto-dispatch."""
    rhs_expr = _normalize_tile_binary_rhs(rhs, span)
    if isinstance(rhs_expr.type, ScalarType):
        return _ir_core.create_op_call(scalar_op_name, [lhs, rhs_expr], {}, span)
    return _ir_core.create_op_call(tile_op_name, [lhs, rhs_expr], {}, span)


# ============================================================================
# Memory Operations
# ============================================================================


def alloc(
    memory_space: int | Expr,
    addr: int | Expr,
    size: int | Expr,
    alloc_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Allocate memory for a MemRef object.

    Internal op emitted by InitMemRef / AllocateMemoryAddr passes.

    Args:
        memory_space: Memory space enum value
        addr: Starting address
        size: Size in bytes
        alloc_id: MemRef identifier
        span: Optional source span

    Returns:
        Call node representing the tile.alloc operation
    """
    actual_span = _get_span_or_capture(span)
    args = [
        _normalize_expr(memory_space, actual_span),
        _normalize_expr(addr, actual_span),
        _normalize_expr(size, actual_span),
        _normalize_expr(alloc_id, actual_span),
    ]
    return _ir_core.create_op_call("tile.alloc", args, {}, actual_span)


def create(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
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
    return _ir_core.create_op_call("tile.create", [shape_tuple], kwargs, actual_span)


create_tile = create


def load(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    target_memory: MemorySpace = MemorySpace.Vec,
    transpose: bool = False,
    span: Span | None = None,
) -> Call:
    """Copy data from tensor to specified memory level.

    Args:
        tensor: Source tensor (TensorType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple.
            Always in the source tensor's coordinate system.
        shapes: Shape of the region to load in each dimension (sequence of scalars),
            or a MakeTuple. Always in the source tensor's coordinate system, even
            when transpose=True. The output TileType shape will be transposed
            automatically by the type deduction layer.
        valid_shapes: Valid shape of the tile in each dimension (sequence of scalars), or a
            MakeTuple. When provided, sets TileView.valid_shape in the output TileType.
            When omitted, shapes is used as valid_shape. Useful for dynamic shapes where
            the actual valid data region differs from the allocated tile size.
            Uses the same coordinate convention as shapes.
        target_memory: Target memory space (MemorySpace.Vec default, or MemorySpace.Mat)
        transpose: Whether to transpose the tile during load (default: False).
            Only supported when target_memory is MemorySpace.Mat (L1).
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the copied data

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # 2D load with transpose to L1 (tensor is [N, K], output tile is [K, N])
        >>> tile = load(tensor, offsets=[0, 0], shapes=[N, K],
        ...             target_memory=MemorySpace.Mat, transpose=True)
    """
    # Validate target_memory: only Vec and Mat are allowed for load
    if target_memory not in (MemorySpace.Vec, MemorySpace.Mat):
        raise ValueError(
            f"target_memory for tile.load must be MemorySpace.Vec or MemorySpace.Mat, got {target_memory}"
        )

    if transpose and target_memory != MemorySpace.Mat:
        raise ValueError(
            f"transpose=True is only supported when target_memory is MemorySpace.Mat (L1), "
            f"got target_memory={target_memory}"
        )

    actual_span = _get_span_or_capture(span)

    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_offsets_shapes(offsets_tuple, shapes_tuple)

    kwargs: dict[str, Any] = {"target_memory": target_memory, "transpose": transpose}

    valid_shapes_tuple = shapes_tuple
    if valid_shapes is not None:
        valid_shapes_tuple = _to_make_tuple(valid_shapes, actual_span)
        if len(valid_shapes_tuple.elements) != len(shapes_tuple.elements):
            raise ValueError(
                f"valid_shapes and shapes must have same number of dimensions, "
                f"got {len(valid_shapes_tuple.elements)} valid_shapes and {len(shapes_tuple.elements)} shapes"
            )

    return _ir_core.create_op_call(
        "tile.load", [tensor, offsets_tuple, shapes_tuple, valid_shapes_tuple], kwargs, actual_span
    )


def store(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    output_tensor: Expr,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple
        output_tensor: Output tensor (TensorType)
        shapes: ND partition shape (sequence of ints), or None for 2D tiles. Normally
            injected automatically by FlattenTileNdTo2D for ND tensors.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    if shapes is not None:
        args: list[Expr] = [tile, offsets_tuple, output_tensor, _to_make_tuple(shapes, actual_span)]
    else:
        args = [tile, offsets_tuple, output_tensor]

    return _ir_core.create_op_call("tile.store", args, {}, actual_span)


def assemble(
    target: Expr,
    source: Expr,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Write source tile data into target tile at specified offset.

    Args:
        target: Target tile (TileType)
        source: Source tile to write (TileType)
        offset: Offset dimensions for where to write, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the same shape/dtype as target
    """
    actual_span = _get_span_or_capture(span)
    offset_tuple = _to_make_tuple(offset, actual_span)

    return _ir_core.create_op_call("tile.assemble", [target, source, offset_tuple], {}, actual_span)


def scatter_update(
    input: Expr,
    *args: Expr | int,
    dim: int | Expr | None = None,
    index: Expr | None = None,
    src: Expr | None = None,
    scratch: Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Update tile rows at positions specified by 2D index tile with values from src.

    Supports two variants based on input/src rank:
    - 2D: input [rows, d], src [b*s, d], index [b, s]
    - 4D: input [blockNum, blockSize, 1, d], src [b, s, 1, d], index [b, s]

    Accepts both call forms:
    - scatter_update(input, dim, index, src, scratch)
    - scatter_update(input, index, src, scratch, dim=-2)

    Args:
        input: Destination tile (TileType, 2D or 4D)
        dim: Dimension to scatter along (currently only -2 is supported)
        index: 2D index tile [b, s] of integer dtype
        src: Source tile (same rank as input)
        scratch: [1, d] scratch row tile (Vec memory) used as the per-row staging buffer
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning a TileType with the same shape/dtype as input
    """
    if len(args) == 4 and dim is None and index is None and src is None and scratch is None:
        dim, index, src, scratch = args
    elif len(args) == 3 and dim is not None and index is None and src is None and scratch is None:
        index, src, scratch = args
    elif len(args) == 1 and dim is None and index is not None and src is not None and scratch is not None:
        # (input, dim, index=..., src=..., scratch=...) — dim passed positionally
        dim = args[0]
    elif len(args) != 0:
        raise TypeError(
            "scatter_update expects (input, dim, index, src, scratch), "
            "(input, index, src, scratch, dim=...), or (input, dim, index=..., src=..., scratch=...)"
        )

    if dim is None or index is None or src is None or scratch is None:
        raise TypeError("scatter_update requires input, dim, index, src, and scratch")

    actual_span = _get_span_or_capture(span)
    if isinstance(dim, ConstInt):
        dim_val = int(dim.value)
    elif isinstance(dim, int):
        dim_val = dim
    else:
        raise TypeError(f"dim must be int or ConstInt, got {type(dim)}")

    if not isinstance(index, Expr):
        raise TypeError(f"index must be Expr, got {type(index)}")
    if not isinstance(src, Expr):
        raise TypeError(f"src must be Expr, got {type(src)}")
    if not isinstance(scratch, Expr):
        raise TypeError(f"scratch must be Expr, got {type(scratch)}")
    op_args: list[Expr] = [input, index, src, scratch]
    kwargs: dict[str, Any] = {"dim": dim_val}
    return _ir_core.create_op_call("tile.scatter_update", op_args, kwargs, actual_span)


def mscatter(
    src: Expr,
    idx: Expr,
    output_tensor: Expr,
    span: Span | None = None,
) -> Call:
    """Scatter-store elements from src tile to output_tensor at per-element indices.

    Semantics: ``output_tensor[idx[i, j]] = src[i, j]``

    Maps to the PTOAS ``pto.mscatter`` instruction.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        idx: Index tile (INT32, same rank as src)
        output_tensor: Output tensor (TensorType, same dtype as src)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.mscatter", [src, idx, output_tensor], {}, actual_span)


def concat(
    src0: Expr,
    src1: Expr,
    span: Span | None = None,
) -> Call:
    """Concatenate two tiles along the column dimension.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise concatenation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.concat", [src0, src1], {}, actual_span)


def move(
    tile: Expr,
    target_memory: MemorySpace,
    blayout: TileLayout | None = None,
    slayout: TileLayout | None = None,
    span: Span | None = None,
) -> Call:
    """Move tile between memory levels.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)
        blayout: Optional block layout for the destination tile
        slayout: Optional scatter layout for the destination tile
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    kwargs: dict[str, Any] = {
        "target_memory": target_memory,
    }
    if blayout is not None:
        kwargs["blayout"] = blayout
    if slayout is not None:
        kwargs["slayout"] = slayout

    return _ir_core.create_op_call("tile.move", args, kwargs, actual_span)


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


def full(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
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
    return _ir_core.create_op_call("tile.full", [shape_tuple, value_expr], kwargs, actual_span)


def ci(
    start: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType = DataType.INT32,
    descending: bool = False,
    span: Span | None = None,
) -> Call:
    """Generate a contiguous integer sequence into a tile (pto.tci).

    For a column index ``k`` in the first row of the destination tile:
    - Ascending: ``dst[0, k] = start + k``
    - Descending: ``dst[0, k] = start - k``

    Note:
        ``pto.tci`` uses the destination's valid-column count as the sequence
        length and does NOT populate additional rows. Leading dimensions must
        be 1 — prefer shapes of the form ``[1, N]``.

    Args:
        start: Starting integer (plain int or a scalar Expr). Its dtype must match ``dtype``.
        shape: Destination tile shape (static, leading dims must be 1, innermost dim != 1).
        dtype: Destination dtype. Must be one of {INT16, INT32}.
        descending: If True, generate a descending sequence.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression that returns a TileType with the generated sequence.
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(start, Expr):
        if isinstance(start, ConstInt) and start.dtype != dtype:
            start_expr = ConstInt(start.value, dtype, actual_span)
        else:
            start_expr = start
    else:
        start_expr = ConstInt(start, dtype, actual_span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype, "descending": descending}
    return _ir_core.create_op_call("tile.ci", [start_expr, shape_tuple], kwargs, actual_span)


arange = ci


def fillpad(tile: Expr, pad_value: PadValue | int | float = PadValue.zero, span: Span | None = None) -> Call:
    """Fill remaining tile elements with specified padding value.

    Args:
        tile: Input tile (TileType)
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the filled and padded tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.fillpad", [tile], {"pad_value": normalize_pad_value(pad_value)}, actual_span
    )


def fillpad_inplace(
    tile: Expr, pad_value: PadValue | int | float = PadValue.zero, span: Span | None = None
) -> Call:
    """Fill padding elements of input tile in place with specified pad value.

    Unlike fillpad which returns a new tile, this operation mutates the input
    tile in place. The valid data region is unchanged; only out-of-bounds
    (padding) elements are written.

    Args:
        tile: Input tile (TileType)
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression (result typically discarded since op is in-place)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.fillpad_inplace", [tile], {"pad_value": normalize_pad_value(pad_value)}, actual_span
    )


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise multiplication of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.muls.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.mul", "tile.muls", lhs, rhs, actual_span)


def add(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise addition of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.adds.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.add", "tile.adds", lhs, rhs, actual_span)


def div(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise division of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.divs.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.div", "tile.divs", lhs, rhs, actual_span)


def sub(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.subs.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.sub", "tile.subs", lhs, rhs, actual_span)


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
    return _ir_core.create_op_call("tile.rem", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.rems", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.shl", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.shls", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.shr", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.shrs", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.and", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.ands", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.or", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.ors", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.xor", [lhs, rhs, tmp], {}, actual_span)


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
    return _ir_core.create_op_call("tile.xors", [lhs, rhs_expr, tmp], {}, actual_span)


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
    return _ir_core.create_op_call("tile.prelu", [tile, slope, tmp], {}, actual_span)


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
    return _ir_core.create_op_call("tile.addc", [lhs, rhs, rhs2], {}, actual_span)


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
    return _ir_core.create_op_call("tile.subc", [lhs, rhs, rhs2], {}, actual_span)


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
    return _ir_core.create_op_call("tile.addsc", [lhs, rhs_expr, rhs2], {}, actual_span)


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
    return _ir_core.create_op_call("tile.subsc", [lhs, rhs_expr, rhs2], {}, actual_span)


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
    return _ir_core.create_op_call("tile.lrelu", [tile, slope_expr], {}, actual_span)


def sel(mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Per-element selection between two tiles using a predicate mask tile.

    For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
    Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

    Args:
        mask: Predicate mask tile (TileType); encoding is target-defined
        lhs: Source tile 0, selected where mask is true (TileType)
        rhs: Source tile 1, selected where mask is false (TileType)
        tmp: Scratch tile required by TSEL (TileType UINT8 [1, 32] on A2/A3)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for per-element tile selection
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sel", [mask, lhs, rhs, tmp], {}, actual_span)


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
    return _ir_core.create_op_call("tile.sels", [lhs, rhs, select_mode_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.muls", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.adds", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.divs", [lhs, rhs_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.subs", [lhs, rhs_expr], {}, actual_span)


def cmp(lhs: Expr, rhs: Expr, cmp_type: int = 0, span: Span | None = None) -> Call:
    """Element-wise comparison of two tiles (returns a packed predicate mask tile).

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for a packed predicate mask tile.
        Use tile.sel with an explicit tmp tile to materialize values.

    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("tile.cmp", [lhs, rhs], kwargs, actual_span)


def cmps(
    lhs: Expr,
    rhs: int | float | Expr,
    cmp_type: int = 0,
    span: Span | None = None,
) -> Call:
    """Element-wise comparison of tile and scalar (returns a packed predicate mask tile).

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for a packed predicate mask tile.
        Use tile.sel with an explicit tmp tile to materialize values.
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.INT32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("tile.cmps", [lhs, rhs_expr], kwargs, actual_span)


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
    return _ir_core.create_op_call("tile.neg", [tile], {}, actual_span)


def exp(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.exp", [tile], {}, actual_span)


def sin(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise sine of a tile (radians).

    FP32-only: non-FP32 inputs are rejected. Cast explicitly via
    ``pl.cast(tile, pl.FP32)`` before applying.

    Args:
        tile: Input tile (TileType, FP32)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise sine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sin", [tile], {}, actual_span)


def cos(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise cosine of a tile (radians).

    FP32-only: non-FP32 inputs are rejected. Cast explicitly via
    ``pl.cast(tile, pl.FP32)`` before applying.

    Args:
        tile: Input tile (TileType, FP32)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cosine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.cos", [tile], {}, actual_span)


def recip(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.recip", [tile], {}, actual_span)


def sqrt(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sqrt", [tile], {}, actual_span)


def rsqrt(tile: Expr, tmp: Expr | None = None, span: Span | None = None) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)
        tmp: Optional scratch tile (TileType, same shape/dtype as ``tile``).
            Passing it selects the high-precision PTO lowering.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    args: list[Expr] = [tile] if tmp is None else [tile, tmp]
    return _ir_core.create_op_call("tile.rsqrt", args, {}, actual_span)


def cast(
    tile: Expr,
    target_type: int | DataType,
    mode: str | int = "round",
    span: Span | None = None,
) -> Call:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cast to target dtype

    Example:
        >>> tile_bf16 = ...  # TileType with BF16 dtype
        >>> tile_fp32 = tile.cast(tile_bf16, DataType.FP32)
    """
    mode_val = resolve_cast_mode(mode)

    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"target_type": target_type, "mode": mode_val}
    return _ir_core.create_op_call("tile.cast", [tile], kwargs, actual_span)


def log(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise natural logarithm of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise natural logarithm
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.log", [tile], {}, actual_span)


def abs(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise absolute value of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise absolute value
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.abs", [tile], {}, actual_span)


def relu(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise ReLU activation function (max(0, x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ReLU activation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.relu", [tile], {}, actual_span)


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
    return _ir_core.create_op_call("tile.not", [tile], {}, actual_span)


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
    return _ir_core.create_op_call("tile.matmul", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.matmul_acc", [acc, lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.matmul_bias", [lhs, rhs, bias], {}, actual_span)


def batch_matmul(
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
) -> Call:
    """Batch matrix multiplication of two tiles with broadcasting.

    For inputs with shape [...batch_dims, M, K] and [...batch_dims, K, N],
    the output has shape [...broadcast_batch_dims, M, N].

    Args:
        lhs: Left-hand side tile (TileType, at least 2D)
        rhs: Right-hand side tile (TileType, at least 2D)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for batch matrix multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.batch_matmul", [lhs, rhs], {}, actual_span)


def batch_matmul_acc(
    acc: Expr,
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
) -> Call:
    """Batch matrix multiplication with accumulation.

    Performs ``acc = acc + lhs @ rhs`` with batch-dim broadcasting between lhs and
    rhs. The broadcast batch shape must equal acc's batch shape (acc is the in-place
    accumulation target and is not broadcast).

    Args:
        acc: Accumulator tile (TileType, at least 2D)
        lhs: Left-hand side tile (TileType, at least 2D)
        rhs: Right-hand side tile (TileType, at least 2D)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for batch matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.batch_matmul_acc", [acc, lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.gemv", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.gemv_acc", [acc, lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.gemv_bias", [lhs, rhs, bias], {}, actual_span)


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
    return _ir_core.create_op_call("tile.row_expand_add", [tile, row_vec], {}, actual_span)


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
    return _ir_core.create_op_call("tile.expands", [target, scalar_expr], {}, actual_span)


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
    return _ir_core.create_op_call("tile.maximum", [lhs, rhs], {}, actual_span)


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
    return _ir_core.create_op_call("tile.minimum", [lhs, rhs], {}, actual_span)


def maximums(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tile.maximums", [lhs, rhs_expr], {}, actual_span)


def minimums(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tile.minimums", [lhs, rhs_expr], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


def sum(tile: Expr, axis: int, keepdim: bool = False, span: Span | None = None) -> Call:
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

    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("tile.sum", args, kwargs, actual_span)


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

    return _ir_core.create_op_call("tile.max", args, kwargs, actual_span)


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

    return _ir_core.create_op_call("tile.min", args, kwargs, actual_span)


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
    return _ir_core.create_op_call("tile.row_max", [tile, tmp_tile], {}, actual_span)


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
    return _ir_core.create_op_call("tile.row_sum", [tile, tmp_tile], {}, actual_span)


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
    return _ir_core.create_op_call("tile.row_min", [tile, tmp_tile], {}, actual_span)


def col_sum(tile: Expr, tmp_tile: Expr | None = None, span: Span | None = None) -> Call:
    """Column-wise sum reduction of a tile (reduces along axis=0, maps to TCOLSUM).

    Output shape is [1, N] for an [M, N] input.

    Passing ``tmp_tile`` activates the binary-tree reduction path (O(log M) depth,
    better precision). Omitting ``tmp_tile`` emits the sequential reduction path.

    Args:
        tile: Input tile (TileType [M, N])
        tmp_tile: Optional scratch tile (TileType, same shape/dtype as input) that
            activates binary-tree reduction.
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


def read(tile: Expr, indices: Expr | list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None) -> Call:
    """Read a scalar value from a tile at given indices.

    Args:
        tile: Input tile expression
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tile dimension), or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression reading a scalar from the tile
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tile, indices_tuple]
    return _ir_core.create_op_call("tile.read", args, {}, actual_span)


def write(
    tile: Expr,
    indices: Expr | list[int | Expr] | _ir_core.MakeTuple,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Write a scalar value into a tile at given indices.

    Args:
        tile: Destination tile expression (TileType)
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tile dimension), or a MakeTuple
        value: Scalar value to write (ScalarType, must match tile dtype)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the tile (for chaining)
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tile, indices_tuple, value]
    return _ir_core.create_op_call("tile.write", args, {}, actual_span)


# ============================================================================
# Transform Operations
# ============================================================================


def slice(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    pad_value: PadValue | int | float | None = None,
    span: Span | None = None,
) -> Call:
    """Create a slice of a tile with static shape and optional valid shape.

    Args:
        tile: Input tile expression
        shape: Static shape dimensions, or a MakeTuple
        offset: Offset dimensions for the slice, or a MakeTuple
        valid_shape: Valid shape dimensions, or a MakeTuple. When omitted, shape
            is reused as the valid shape.
        pad_value: Optional padding mode for out-of-valid-shape elements.
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (normalized
            via :func:`normalize_pad_value`). ``PadValue.null`` is passed
            through unchanged and means "no padding". When omitted (``None``),
            the kwarg is not forwarded — the deducer defaults to
            ``PadValue.null``.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tile slice
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)
    offset_tuple = _to_make_tuple(offset, actual_span)
    args = [tile, shape_tuple, offset_tuple]

    if valid_shape is not None:
        valid_shape_tuple = _to_make_tuple(valid_shape, actual_span)
        if len(valid_shape_tuple.elements) != len(shape_tuple.elements):
            raise ValueError(
                f"valid_shape and shape must have same number of dimensions, "
                "got "
                f"{len(valid_shape_tuple.elements)} valid_shape dims and "
                f"{len(shape_tuple.elements)} shape dims"
            )
        args.append(valid_shape_tuple)

    kwargs: dict[str, Any] = {}
    if pad_value is not None:
        # PadValue.null is a legal "no padding" signal for slice (unlike
        # fillpad, which requires a real padding mode). Pass it through;
        # normalize the rest via the shared helper so numeric sugar and
        # validation match tile.fillpad exactly.
        kwargs["pad_value"] = pad_value if pad_value is PadValue.null else normalize_pad_value(pad_value)

    return _ir_core.create_op_call("tile.slice", args, kwargs, actual_span)


def extract(
    src: Expr,
    index_row: int | Expr,
    index_col: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    target_memory: MemorySpace,
    span: Span | None = None,
) -> Call:
    """Extract a sub-tile from src at (index_row, index_col).

    Maps to ISA TEXTRACT Variant 1 (Standard Extract). The result tile has the
    given static ``shape`` and lives in ``target_memory``.

    Args:
        src: Source tile expression (TileType, 2D)
        index_row: Starting row offset (int or Expr)
        index_col: Starting col offset (int or Expr)
        shape: Static destination shape (2-element sequence or MakeTuple of ConstInt)
        target_memory: Destination memory space (Left/Right for Mat sources, Mat for Acc sources)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression with TileType[shape, src.dtype] in target_memory
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    row_expr = _normalize_expr(index_row, actual_span, int_dtype=DataType.INDEX)
    col_expr = _normalize_expr(index_col, actual_span, int_dtype=DataType.INDEX)
    return _ir_core.create_op_call(
        "tile.extract",
        [src, row_expr, col_expr, shape_tuple],
        {"target_memory": target_memory},
        actual_span,
    )


def reshape(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
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
    return _ir_core.create_op_call("tile.reshape", args, {}, actual_span)


def transpose(tile: Expr, axis1: int | ConstInt, axis2: int | ConstInt, span: Span | None = None) -> Call:
    """Transpose tile by swapping two axes.

    Args:
        tile: Input tile expression
        axis1: First axis to swap as an int or ConstInt (supports negative indexing)
        axis2: Second axis to swap as an int or ConstInt (supports negative indexing)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile transpose
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(axis1, ConstInt):
        axis1_expr = axis1
    elif isinstance(axis1, int):
        axis1_expr = ConstInt(axis1, DataType.INDEX, actual_span)
    else:
        raise TypeError(f"axis1 must be int or ConstInt, got {type(axis1)}")

    if isinstance(axis2, ConstInt):
        axis2_expr = axis2
    elif isinstance(axis2, int):
        axis2_expr = ConstInt(axis2, DataType.INDEX, actual_span)
    else:
        raise TypeError(f"axis2 must be int or ConstInt, got {type(axis2)}")

    args = [tile, axis1_expr, axis2_expr]

    return _ir_core.create_op_call("tile.transpose", args, {}, actual_span)


def set_validshape(
    tile: Expr,
    valid_rows: int | Expr,
    valid_cols: int | Expr,
    span: Span | None = None,
) -> Call:
    """Update valid-shape metadata of a tile without data movement.

    .. note::
        Internal API — this op is intended for compiler-generated code only
        and should not be exposed to end users in future releases.

    Args:
        tile: Input tile expression (must be 2D TileType)
        valid_rows: Number of valid rows (int or Scalar INDEX expression)
        valid_cols: Number of valid columns (int or Scalar INDEX expression)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile.set_validshape
    """
    actual_span = _get_span_or_capture(span)
    vr_expr = (
        valid_rows if isinstance(valid_rows, Expr) else ConstInt(valid_rows, DataType.INDEX, actual_span)
    )
    vc_expr = (
        valid_cols if isinstance(valid_cols, Expr) else ConstInt(valid_cols, DataType.INDEX, actual_span)
    )
    return _ir_core.create_op_call("tile.set_validshape", [tile, vr_expr, vc_expr], {}, actual_span)


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


def tpush_to_aiv(tile: Expr, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe.

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


def tpop_from_aic(
    *,
    result_type: _ir_core.Type | None = None,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        result_type: Explicit result type (e.g. TileType). Mutually exclusive with shape/dtype.
        shape: Shape of the tile to receive (alternative to result_type).
        dtype: Data type of the tile to receive (alternative to result_type).
        split: Split mode (0=none, 1=up-down, 2=left-right)
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    resolved_type = _resolve_tpop_type(result_type, shape, dtype, MemorySpace.Vec)
    kwargs = {"split": split}
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


# ============================================================================
# Sorting Operations
# ============================================================================


def sort32(src: Expr, idx: Expr, span: Span | None = None) -> Call:
    """Sort fixed 32-element blocks with explicit index tile.

    Sorts 32-element blocks in src and permutes idx accordingly.
    Output tile stores sorted value-index pairs with doubled last dimension.

    Args:
        src: Input value tile (TileType, FP16 or FP32, Vec memory)
        idx: Input index tile (TileType, Vec memory) with sequential offsets
        span: Optional source span for debugging

    Returns:
        Call expression returning sorted tile with doubled last dimension
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sort32", [src, idx], {}, actual_span)


# ============================================================================
# Gather Operations
# ============================================================================


def gather(
    src: Expr,
    indices: Expr,
    tmp: Expr,
    span: Span | None = None,
) -> Call:
    """Gather elements from src by per-element indices (index form).

    Computes ``dst[i, j] = src[indices[i, j]]``. Maps to PTOAS ``pto.tgather`` index form.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        indices: Index tile (INT32)
        tmp: Temporary workspace tile (INT32)
        span: Optional source span

    Returns:
        Call expression returning gathered tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.gather", [src, indices, tmp], {}, actual_span)


def gather_mask(
    src: Expr,
    mask_pattern: int,
    *,
    output_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Gather elements from src using a fixed hardware mask pattern (mask form).

    Maps to PTOAS ``pto.tgather`` mask form.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        mask_pattern: Mask pattern selector (1-7).
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        output_dtype: Optional output dtype (keyword-only). When provided, the result
            tile has this dtype instead of src's dtype. The hardware only requires
            sizeof(dst_dtype) == sizeof(src_dtype). Useful for extracting UINT32 index
            bits from FP32 sort32 output (bit reinterpretation).
        span: Optional source span

    Returns:
        Call expression returning gathered tile
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"mask_pattern": mask_pattern}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return _ir_core.create_op_call("tile.gather_mask", [src], kwargs, actual_span)


# Compare modes accepted by tile.gather_compare / tensor.gather_compare.
# Maps to PTOAS pto.tgather cmpMode: eq/ne/lt/le/gt/ge.
_GATHER_COMPARE_CMP_MODES = ("eq", "ne", "lt", "le", "gt", "ge")


def resolve_gather_compare_cmp_mode(cmp_mode: str | int) -> int:
    """Normalize a gather_compare cmp_mode to its integer enum value.

    Accepts either a string in ``{"eq", "ne", "lt", "le", "gt", "ge"}`` or
    an int in ``[0, 5]``. Raises ``ValueError`` on anything else.
    """
    if isinstance(cmp_mode, str):
        try:
            return _GATHER_COMPARE_CMP_MODES.index(cmp_mode)
        except ValueError as e:
            raise ValueError(
                f"Invalid cmp_mode {cmp_mode!r}: expected one of {list(_GATHER_COMPARE_CMP_MODES)} or int 0-5"
            ) from e
    if isinstance(cmp_mode, int) and not isinstance(cmp_mode, bool):
        if 0 <= cmp_mode < len(_GATHER_COMPARE_CMP_MODES):
            return cmp_mode
        raise ValueError(
            f"Invalid cmp_mode {cmp_mode}: int must be in [0, {len(_GATHER_COMPARE_CMP_MODES) - 1}]"
        )
    raise ValueError(f"Invalid cmp_mode {cmp_mode!r}: expected str or int 0-5")


def gather_compare(
    src: Expr,
    kvalue: Expr,
    tmp: Expr,
    *,
    cmp_mode: str | int,
    offset: int = 0,
    out_cols: int,
    count_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Compare-form gather (tile-level): scan ``src`` per-row against ``kvalue``.

    Maps to PTOAS ``pto.tgather`` compare-form. Returns a single
    :class:`Call` whose result type is a ``TupleType{dst, cdst}``::

        dst  : TileType, [rows, out_cols], INT32  — gathered indices
        cdst : TileType, [rows],           count_dtype — per-row match count

    The DSL form ``d, c = pl.tile.gather_compare(src, kvalue, tmp, ...)`` is
    desugared by the parser into ``_tuple = call; d = _tuple[0]; c = _tuple[1]``.

    Args:
        src: Source tile (FP16/FP32/INT16/INT32, 2D).
        kvalue: Scalar threshold (ScalarType; dtype must match ``src``; applied to every row).
        tmp: Workspace tile (UINT8) sized for the codegen kernel.
        cmp_mode: Compare mode — one of ``"eq"`` / ``"ne"`` / ``"lt"`` /
            ``"le"`` / ``"gt"`` / ``"ge"`` or int ``0..5``.
        offset: Starting index offset (default 0).
        out_cols: Output column count per row for ``dst`` (positive int).
        count_dtype: Per-row count dtype, INT32 or UINT32; defaults to INT32.
        span: Optional source span (auto-captured if omitted).
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {
        "cmp_mode": resolve_gather_compare_cmp_mode(cmp_mode),
        "offset": offset,
        "out_cols": out_cols,
    }
    if count_dtype is not None:
        kwargs["count_dtype"] = count_dtype
    return _ir_core.create_op_call("tile.gather_compare", [src, kvalue, tmp], kwargs, actual_span)


# ============================================================================
# Merge Sort Operations
# ============================================================================


def mrgsort(
    src0: Expr,
    src1: Expr | None = None,
    src2: Expr | None = None,
    src3: Expr | None = None,
    tmp: Expr | None = None,
    executed: Expr | None = None,
    exhausted: bool = False,
    *,
    block_len: int | Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge).

    Format1 (block_len form): sorts a tile containing multiple pre-sorted runs.
    Format2 (2-4 way form): merges 2, 3, or 4 pre-sorted input tiles.

    Args:
        src0: For format1: input tile with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tile.
        src1: (format2) Second sorted input tile.
        src2: (format2, optional) Third sorted input tile (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tile (4-way only).
        tmp: (format2) Temporary workspace tile, must be passed as keyword arg for 2/3-way.
        executed: (format2) Exhaustion status tile (written by hardware), keyword arg for 2/3-way.
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.
        span: Optional source span for debugging.

    Returns:
        Call expression returning merged sorted tile.
    """
    actual_span = _get_span_or_capture(span)
    if block_len is not None:
        # format1: single-list merge sort (pto.tmrgsort format1)
        if any(arg is not None for arg in (src1, src2, src3, tmp, executed)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ..., tmp, executed) "
                "are mutually exclusive; do not pass format2 arguments with block_len"
            )
        # PTO ISA requires block_len as i32. The parser may emit ConstInt with INDEX dtype,
        # so always extract the integer value and create a fresh INT32 constant.
        if isinstance(block_len, _ir_core.ConstInt):
            block_len_expr = _ir_core.ConstInt(block_len.value, DataType.INT32, actual_span)
        elif isinstance(block_len, Expr):
            block_len_expr = block_len
        else:
            block_len_expr = _ir_core.ConstInt(block_len, DataType.INT32, actual_span)
        return _ir_core.create_op_call("tile.mrgsort_format1", [src0, block_len_expr], {}, actual_span)
    # format2: 2-4 way merge (pto.tmrgsort format2)
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, "
            "or at least (src0, src1, tmp=<tile>, executed=<tile>) for format2"
        )
    if src2 is None and src3 is not None:
        raise ValueError("mrgsort() format2 requires src2 when src3 is provided")
    if tmp is None or executed is None:
        raise ValueError(
            "mrgsort() format2 requires tmp and executed to be provided as keyword arguments; "
            "use mrgsort(src0, src1[, src2[, src3]], tmp=<tile>, executed=<tile>)"
        )
    kwargs: dict[str, Any] = {"exhausted": exhausted}
    if src2 is None:
        # 2-way merge
        args = [src0, src1, tmp, executed]
    elif src3 is None:
        # 3-way merge
        args = [src0, src1, src2, tmp, executed]
    else:
        # 4-way merge
        args = [src0, src1, src2, src3, tmp, executed]
    return _ir_core.create_op_call("tile.mrgsort_format2", args, kwargs, actual_span)


def mrgsort_format1(src0: Expr, block_len: int | Expr, span: Span | None = None) -> Call:
    """Single-list merge sort (format1). Used by the parser for roundtrip fidelity.

    Prefer ``mrgsort(src, block_len=...)`` in user code.
    """
    return mrgsort(src0, block_len=block_len, span=span)


def mrgsort_format2(*args: Expr, exhausted: bool = False, span: Span | None = None) -> Call:
    """2-4 way merge sort (format2). Used by the parser for roundtrip fidelity.

    Positional args: ``(src0, src1[, src2[, src3]], tmp, executed)``
    The last 2 positional args are always ``tmp`` and ``executed``.

    Prefer ``mrgsort(src0, src1[, src2[, src3]], tmp=<tile>, executed=<tile>)`` in user code.
    """
    if len(args) < 4 or len(args) > 6:
        raise ValueError(
            f"mrgsort_format2() requires 4-6 positional arguments "
            f"(src0, src1[, src2[, src3]], tmp, executed), got {len(args)}"
        )
    srcs = args[:-2]
    tmp = args[-2]
    executed = args[-1]
    src0 = srcs[0]
    src1 = srcs[1]
    src2 = srcs[2] if len(srcs) > 2 else None
    src3 = srcs[3] if len(srcs) > 3 else None
    return mrgsort(src0, src1, src2, src3, tmp=tmp, executed=executed, exhausted=exhausted, span=span)
