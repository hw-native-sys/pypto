# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile operations for PyPTO Language DSL.

This module provides type-safe wrappers around pypto.ir.op.tile operations
that accept and return Tile types instead of raw Expr/Call objects.

Accessed as ``pl.tile.*``
"""

from collections.abc import Sequence
from typing import overload

__all__ = [
    "create_tile",
    "create",
    "read",
    "write",
    "load",
    "store",
    "assemble",
    "scatter_update",
    "concat",
    "move",
    "full",
    "fillpad",
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
    "matmul_bias",
    "gemv",
    "gemv_acc",
    "gemv_bias",
    "row_max",
    "row_sum",
    "row_min",
    "maximum",
    "row_expand",
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
    "slice",
    "reshape",
    "transpose",
    "rem",
    "rems",
    "and_",
    "ands",
    "or_",
    "ors",
    "xor",
    "xors",
    "shl",
    "shls",
    "shr",
    "shrs",
    "maxs",
    "mins",
    "prelu",
    "not_",
    "addc",
    "subc",
    "addsc",
    "subsc",
    "lrelu",
    "sel",
    "sels",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
]

from pypto.ir.op import tile_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Expr, MemorySpace, PadValue

from ..typing import IntLike, Scalar, Tensor, Tile
from .system_ops import (  # noqa: F401
    tpop_from_aic,
    tpop_from_aiv,
    tpush_to_aic,
    tpush_to_aiv,
)


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def create(
    shape: Sequence[IntLike],
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
) -> Tile:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)

    Returns:
        Tile wrapping the create operation
    """
    # create C++ binding accepts Sequence[int]; Expr elements from Scalar
    # unwrapping are valid at DSL parse time (parser reads the AST).
    call_expr = _ir_ops.create(
        _normalize_intlike(shape),  # type: ignore[reportArgumentType]
        dtype,
        target_memory,
    )
    return Tile(expr=call_expr)


create_tile = create


def read(tile: Tile, indices: IntLike | Sequence[IntLike]) -> Scalar:
    """Read a scalar value from a tile at given indices.

    Args:
        tile: Input tile
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tile dimension)

    Returns:
        Scalar wrapping the read operation
    """
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    call_expr = _ir_ops.read(tile.unwrap(), _normalize_intlike(indices_seq))
    return Scalar(expr=call_expr)


def write(tile: Tile, indices: IntLike | Sequence[IntLike], value: Scalar) -> None:
    """Write a scalar value into a tile at given indices.

    Args:
        tile: Destination tile
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tile dimension)
        value: Scalar value to write
    """
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    call_expr = _ir_ops.write(tile.unwrap(), _normalize_intlike(indices_seq), value.unwrap())
    _ = call_expr  # result is the tile itself; discarded here


def load(
    tensor: Tensor,
    offsets: Sequence[IntLike],
    shapes: Sequence[IntLike],
    target_memory: MemorySpace = MemorySpace.Vec,
    valid_shapes: Sequence[IntLike] | None = None,
    transpose: bool = False,
) -> Tile:
    """Copy data from tensor to unified buffer (tile).

    Args:
        tensor: Source tensor
        offsets: Offsets in each dimension
        shapes: Shape of the tile in each dimension
        target_memory: Target memory space (MemorySpace.Vec default, or MemorySpace.Mat)
        valid_shapes: Valid shape of the tile in each dimension. When provided, sets
            TileView.valid_shape in the output TileType. When omitted, shapes is used
            as valid_shape.
        transpose: Whether to transpose the tile during load (default: False).
            Only supported when target_memory is MemorySpace.Mat (L1).

    Returns:
        Tile wrapping the load operation

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # 2D load with transpose to L1
        >>> tile = load(tensor, offsets=[0, 0], shapes=[K, N],
        ...             target_memory=pl.MemorySpace.Mat, transpose=True)
    """
    if valid_shapes is None:
        valid_shapes = shapes
    call_expr = _ir_ops.load(
        tensor.unwrap(),
        _normalize_intlike(offsets),
        _normalize_intlike(shapes),
        _normalize_intlike(valid_shapes),
        target_memory,
        transpose,
    )
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    offsets: Sequence[IntLike],
    output_tensor: Tensor,
    shapes: Sequence[IntLike] | None = None,
) -> Tensor:
    """Copy data from tile back to tensor.

    Args:
        tile: Source tile
        offsets: Offsets in each dimension
        output_tensor: Output tensor
        shapes: Optional ND partition shape. Injected by FlattenTileNdTo2D for ND tensors.

    Returns:
        Tensor wrapping the store operation

    Example:
        >>> # 2D store
        >>> result = store(tile, [0, 0], tensor)
        >>> # 3D store
        >>> result = store(tile, [0, 0, 0], tensor)
    """
    normalized_offsets = _normalize_intlike(offsets)
    normalized_shapes = _normalize_intlike(shapes) if shapes is not None else None
    call_expr = _ir_ops.store(tile.unwrap(), normalized_offsets, output_tensor.unwrap(), normalized_shapes)
    return Tensor(expr=call_expr)


def assemble(target: Tile, source: Tile, offset: Sequence[IntLike]) -> Tile:
    """Write source tile data into target tile at specified offset.

    Args:
        target: Target tile to update
        source: Source tile to write
        offset: Offset dimensions for where to write

    Returns:
        Tile wrapping the assemble operation
    """
    call_expr = _ir_ops.assemble(target.unwrap(), source.unwrap(), _normalize_intlike(offset))
    return Tile(expr=call_expr)


def scatter_update(
    input: Tile,
    dim: int,
    index: Tile,
    src: Tile,
) -> Tile:
    """Update tile rows at positions specified by 2D index tile with values from src.

    Args:
        input: Destination tile (2D or 4D)
        dim: Dimension to scatter along (currently only -2 is supported)
        index: 2D index tile [b, s] of integer dtype
        src: Source tile (same rank as input)

    Returns:
        Tile wrapping the scatter_update operation
    """
    call_expr = _ir_ops.scatter_update(input.unwrap(), dim, index.unwrap(), src.unwrap())
    return Tile(expr=call_expr)


def concat(src0: Tile, src1: Tile) -> Tile:
    """Concatenate two tiles along the column dimension.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile with concatenated columns
    """
    call_expr = _ir_ops.concat(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def move(tile: Tile, target_memory: MemorySpace) -> Tile:
    """Move tile between memory levels.

    Args:
        tile: Input tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)

    Returns:
        Tile wrapping the move operation
    """
    call_expr = _ir_ops.move(tile.unwrap(), target_memory)
    return Tile(expr=call_expr)


def full(shape: list[int], dtype: DataType, value: int | float) -> Tile:
    """Create a tile from a shape and fill with value in Vec.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        value: filling scalar

    Returns:
        Tile wrapping the full operation
    """
    call_expr = _ir_ops.full(shape, dtype, value)
    return Tile(expr=call_expr)


def fillpad(tile: Tile, pad_value: PadValue = PadValue.zero) -> Tile:
    """Fill remaining tile elements with specified padding value.

    Args:
        tile: Input tile
        pad_value: Padding mode (PadValue.zero, PadValue.max, or PadValue.min). Default is zero.

    Returns:
        Tile wrapping the fillpad operation
    """
    call_expr = _ir_ops.fillpad(tile.unwrap(), pad_value=pad_value)
    return Tile(expr=call_expr)


def get_block_idx() -> Scalar:
    """Get the current block index.

    This operation returns the index of the current compute tile. It is typically
    used in tile-level programming to identify which block of data is being processed.

    Returns:
        Scalar wrapping the get_block_idx operation (UINT64 type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
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


def adds(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
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


def subs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
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


def muls(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
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


def divs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
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
    target_type: int | DataType,
    mode: str | int = "round",
) -> Tile:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)

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


def matmul_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    """Matrix multiplication with bias add: C = lhs @ rhs + bias.

    Args:
        lhs: Left-hand side tile [M, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N]

    Returns:
        Tile wrapping the matmul_bias operation
    """
    call_expr = _ir_ops.matmul_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
    return Tile(expr=call_expr)


def gemv(lhs: Tile, rhs: Tile) -> Tile:
    """General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]

    Returns:
        Tile wrapping the gemv operation
    """
    call_expr = _ir_ops.gemv(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

    Args:
        acc: Accumulator tile [1, N]
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]

    Returns:
        Tile wrapping the gemv_acc operation
    """
    call_expr = _ir_ops.gemv_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    """GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N]

    Returns:
        Tile wrapping the gemv_bias operation
    """
    call_expr = _ir_ops.gemv_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
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


def row_expand(src: Tile) -> Tile:
    """Broadcast the first element of each source row across the destination row.

    For each element (i, j): dst[i, j] = src[i, 0].

    Args:
        src: Input tile [M, N]

    Returns:
        Tile wrapping the row_expand operation
    """
    call_expr = _ir_ops.row_expand(src.unwrap())
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


def expands(target: Tile, scalar: int | float | Expr | Scalar) -> Tile:
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


def cmps(lhs: Tile, rhs: int | float | Expr | Scalar, cmp_type: int = 0) -> Tile:
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


@overload
def max(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...


@overload
def max(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


def max(tile: Tile | Scalar, axis: int | Scalar = 0, keepdim: bool = False) -> Tile | Scalar:
    """Max reduction along specified axis, or scalar max of two values.

    Args:
        tile: Input tile or first scalar operand
        axis: Reduction axis (for tiles) or second scalar operand
        keepdim: Whether to keep the reduced dimension as 1 (tiles only)

    Returns:
        Tile or Scalar wrapping the max operation
    """
    if isinstance(tile, Scalar):
        rhs: Expr = (
            axis.unwrap()
            if isinstance(axis, Scalar)
            else _ir_core.ConstInt(axis, DataType.INT32, _ir_core.Span.unknown())
        )
        return Scalar(expr=_ir_core.max_(tile.unwrap(), rhs))
    assert isinstance(axis, int)
    call_expr = _ir_ops.max(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


@overload
def min(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...


@overload
def min(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


@overload
def min(tile: int, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


def min(tile: Tile | Scalar | int, axis: int | Scalar = 0, keepdim: bool = False) -> Tile | Scalar:
    """Min reduction along specified axis, or scalar min of two values.

    Args:
        tile: Input tile or first scalar operand
        axis: Reduction axis (for tiles) or second scalar operand
        keepdim: Whether to keep the reduced dimension as 1 (tiles only)

    Returns:
        Tile or Scalar wrapping the min operation
    """
    if isinstance(tile, (Scalar, int)):
        lhs: Expr = (
            tile.unwrap()
            if isinstance(tile, Scalar)
            else _ir_core.ConstInt(tile, DataType.INT32, _ir_core.Span.unknown())
        )
        rhs: Expr = (
            axis.unwrap()
            if isinstance(axis, Scalar)
            else _ir_core.ConstInt(axis, DataType.INT32, _ir_core.Span.unknown())
        )
        return Scalar(expr=_ir_core.min_(lhs, rhs))
    assert isinstance(axis, int)
    call_expr = _ir_ops.min(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


def slice(
    tile: Tile,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
) -> Tile:
    """Create a slice of a tile with static shape and optional valid shape.

    Args:
        tile: Input tile
        shape: Static shape dimensions (at most 2 for TileType)
        offset: Offset dimensions for the slice
        valid_shape: Valid shape dimensions. When omitted, shape is reused as the
            logical valid shape.

    Returns:
        Tile wrapping the slice operation
    """
    tile_expr = tile.unwrap()
    normalized_valid_shape = None if valid_shape is None else _normalize_intlike(valid_shape)
    call_expr = _ir_ops.slice(
        tile_expr,
        _normalize_intlike(shape),
        _normalize_intlike(offset),
        normalized_valid_shape,
    )
    return Tile(expr=call_expr)


def reshape(tile: Tile, shape: Sequence[IntLike]) -> Tile:
    """Reshape tile to new shape.

    Args:
        tile: Input tile
        shape: New shape dimensions (at most 2 for TileType)

    Returns:
        Tile wrapping the reshape operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.reshape(tile_expr, _normalize_intlike(shape))
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


def rem(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise remainder (modulo) of two tiles.

    Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the rem operation
    """
    call_expr = _ir_ops.rem(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def rems(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise remainder (modulo) of tile and scalar.

    Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the rems operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.rems(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def and_(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise AND of two tiles.

    Computes lhs & rhs element-wise. Maps to the TAND hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the and operation
    """
    call_expr = _ir_ops.and_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ands(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise AND of tile and scalar.

    Computes lhs & rhs element-wise. Maps to the TANDS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the ands operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.ands(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def or_(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise OR of two tiles.

    Computes lhs | rhs element-wise. Maps to the TOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the or operation
    """
    call_expr = _ir_ops.or_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ors(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise OR of tile and scalar.

    Computes lhs | rhs element-wise. Maps to the TORS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the ors operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.ors(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def xor(lhs: Tile, rhs: Tile, tmp: Tile) -> Tile:
    """Element-wise bitwise XOR of two tiles.

    Computes lhs ^ rhs element-wise. Maps to the TXOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the xor operation
    """
    call_expr = _ir_ops.xor(lhs.unwrap(), rhs.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def xors(lhs: Tile, rhs: int | Expr | Scalar, tmp: Tile) -> Tile:
    """Element-wise bitwise XOR of tile and scalar.

    Computes lhs ^ rhs element-wise. Maps to the TXORS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the xors operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.xors(lhs.unwrap(), rhs_expr, tmp.unwrap())
    return Tile(expr=call_expr)


def shl(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise left shift of two tiles.

    Computes lhs << rhs element-wise. Maps to the TSHL hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the shl operation
    """
    call_expr = _ir_ops.shl(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shls(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise left shift of tile and scalar.

    Computes lhs << rhs element-wise. Maps to the TSHLS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile
        rhs: Scalar shift amount; must be >= 0

    Returns:
        Tile wrapping the shls operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.shls(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def shr(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise right shift of two tiles.

    Computes lhs >> rhs element-wise. Maps to the TSHR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the shr operation
    """
    call_expr = _ir_ops.shr(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shrs(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise right shift of tile and scalar.

    Computes lhs >> rhs element-wise. Maps to the TSHRS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile
        rhs: Scalar shift amount; must be >= 0

    Returns:
        Tile wrapping the shrs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.shrs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def maxs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise maximum of tile and scalar.

    Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the maxs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.maxs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def mins(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise minimum of tile and scalar.

    Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the mins operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.mins(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def prelu(tile: Tile, slope: Tile, tmp: Tile) -> Tile:
    """Element-wise parametric ReLU of a tile.

    Computes prelu(tile, slope) element-wise. Maps to the TPRELU hardware intrinsic.

    Args:
        tile: Input tile
        slope: Slope tile used for negative values
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the prelu operation
    """
    call_expr = _ir_ops.prelu(tile.unwrap(), slope.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def not_(tile: Tile) -> Tile:
    """Element-wise bitwise NOT of a tile.

    Computes ~tile element-wise. Maps to the TNOT hardware intrinsic.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the not operation
    """
    call_expr = _ir_ops.not_(tile.unwrap())
    return Tile(expr=call_expr)


def addc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    """Element-wise addition of three tiles.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        rhs2: Third tile

    Returns:
        Tile wrapping the addc operation
    """
    call_expr = _ir_ops.addc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def subc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    """Element-wise subtraction of three tiles.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        rhs2: Third tile

    Returns:
        Tile wrapping the subc operation
    """
    call_expr = _ir_ops.subc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def addsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise addition of tile, scalar, and tile.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Scalar value
        rhs2: Third tile

    Returns:
        Tile wrapping the addsc operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.addsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def subsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise subtraction of tile, scalar, and tile.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Scalar value
        rhs2: Third tile

    Returns:
        Tile wrapping the subsc operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.subsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def lrelu(tile: Tile, slope: int | float | Expr | Scalar) -> Tile:
    """Element-wise leaky ReLU with scalar slope.

    Computes max(tile, slope * tile) element-wise. Maps to the TLRELU hardware intrinsic.

    Args:
        tile: Input tile
        slope: Scalar slope for negative values

    Returns:
        Tile wrapping the lrelu operation
    """
    slope_expr = slope.unwrap() if isinstance(slope, Scalar) else slope
    call_expr = _ir_ops.lrelu(tile.unwrap(), slope_expr)
    return Tile(expr=call_expr)


def sel(mask: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """Per-element selection between two tiles using a predicate mask tile.

    For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
    Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

    Args:
        mask: Predicate mask tile; encoding is target-defined
        lhs: Source tile 0, selected where mask is true
        rhs: Source tile 1, selected where mask is false

    Returns:
        Tile wrapping the sel operation
    """
    call_expr = _ir_ops.sel(mask.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def sels(lhs: Tile, rhs: Tile, select_mode: int | float | Expr | Scalar) -> Tile:
    """Select between two tiles based on a scalar mode.

    Maps to the TSELS hardware intrinsic. The interpretation of select_mode values
    is target-dependent and enforced by codegen.

    Args:
        lhs: Source tile 0
        rhs: Source tile 1
        select_mode: Scalar select mode

    Returns:
        Tile wrapping the sels operation
    """
    select_mode_expr = select_mode.unwrap() if isinstance(select_mode, Scalar) else select_mode
    call_expr = _ir_ops.sels(lhs.unwrap(), rhs.unwrap(), select_mode_expr)
    return Tile(expr=call_expr)
