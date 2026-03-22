# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unified operation dispatch for PyPTO Language DSL.

Provides type-dispatched wrappers that auto-select between tensor and tile
operations based on the input type (Tensor vs Tile). Users can write
``pl.add(a, b)`` instead of explicitly choosing ``pl.tensor.add``
or ``pl.tile.add``.
"""

from collections.abc import Sequence
from typing import TypeVar, overload

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "exp",
    "neg",
    "recip",
    "sqrt",
    "rsqrt",
    "row_expand",
    "row_expand_mul",
    "row_expand_div",
    "row_expand_add",
    "row_expand_sub",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "concat",
    "expands",
    "reshape",
    "transpose",
    "slice",
    "fillpad",
    "matmul",
    "row_max",
    "row_sum",
    "row_min",
    "cast",
    "create_tile",
    "read",
    "write",
]

from pypto.ir.utils import resolve_cast_mode
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import MemorySpace, PadValue

from ..typing import IntLike, Scalar, Tensor, Tile
from . import tensor_ops as _tensor
from . import tile_ops as _tile

# ---------------------------------------------------------------------------
# TypeVar
# ---------------------------------------------------------------------------

T = TypeVar("T", Tensor, Tile)

# ---------------------------------------------------------------------------
# Binary arithmetic with scalar auto-dispatch
# ---------------------------------------------------------------------------

# --- add ---


def add(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _tile.adds(lhs, rhs)
    raise TypeError(f"add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- sub ---


def sub(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _tile.subs(lhs, rhs)
    raise TypeError(f"sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- mul ---


def mul(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _tile.muls(lhs, rhs)
    raise TypeError(f"mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- div ---


def div(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _tile.divs(lhs, rhs)
    raise TypeError(f"div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# ---------------------------------------------------------------------------
# Simple overlapping ops (dispatch on first arg type)
# ---------------------------------------------------------------------------


def maximum(lhs: T, rhs: T) -> T:
    """Element-wise maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.maximum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.maximum(lhs, rhs)
    raise TypeError(f"maximum: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def exp(input: T) -> T:
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.exp(input)
    if isinstance(input, Tile):
        return _tile.exp(input)
    raise TypeError(f"exp: expected Tensor or Tile, got {type(input).__name__}")


def neg(input: T) -> T:
    """Element-wise negation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.neg(input)
    if isinstance(input, Tile):
        return _tile.neg(input)
    raise TypeError(f"neg: expected Tensor or Tile, got {type(input).__name__}")


def recip(input: T) -> T:
    """Element-wise reciprocal (1/x), dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.recip(input)
    if isinstance(input, Tile):
        return _tile.recip(input)
    raise TypeError(f"recip: expected Tensor or Tile, got {type(input).__name__}")


def sqrt(input: T) -> T:
    """Element-wise square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.sqrt(input)
    if isinstance(input, Tile):
        return _tile.sqrt(input)
    raise TypeError(f"sqrt: expected Tensor or Tile, got {type(input).__name__}")


def rsqrt(input: T) -> T:
    """Element-wise reciprocal square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.rsqrt(input)
    if isinstance(input, Tile):
        return _tile.rsqrt(input)
    raise TypeError(f"rsqrt: expected Tensor or Tile, got {type(input).__name__}")


def row_expand_mul(lhs: T, rhs: T) -> T:
    """Row-wise broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_mul(lhs, rhs)
    raise TypeError(
        "row_expand_mul: expected both operands to be Tensor or both to be Tile, "
        f"got lhs={type(lhs).__name__}, rhs={type(rhs).__name__}"
    )


def row_expand_div(lhs: T, rhs: T) -> T:
    """Row-wise broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_div(lhs, rhs)
    raise TypeError(
        "row_expand_div: expected both operands to be Tensor or both to be Tile, "
        f"got lhs={type(lhs).__name__}, rhs={type(rhs).__name__}"
    )


def col_expand_mul(lhs: T, rhs: T) -> T:
    """Column-wise broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_mul(lhs, rhs)
    raise TypeError(
        "col_expand_mul: expected (Tensor, Tensor) or (Tile, Tile), "
        f"got ({type(lhs).__name__}, {type(rhs).__name__})"
    )


def row_expand(input: T) -> T:
    """Row-wise broadcast: dst[i, j] = src[i, 0], dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.row_expand(input)
    if isinstance(input, Tile):
        return _tile.row_expand(input)
    raise TypeError(f"row_expand: expected Tensor or Tile, got {type(input).__name__}")


def row_expand_add(lhs: T, rhs: T) -> T:
    """Row-wise broadcast addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_add(lhs, rhs)
    raise TypeError(
        "row_expand_add: expected both operands to be Tensor or both to be Tile, "
        f"got lhs={type(lhs).__name__}, rhs={type(rhs).__name__}"
    )


def row_expand_sub(lhs: T, rhs: T) -> T:
    """Row-wise broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_sub(lhs, rhs)
    raise TypeError(
        "row_expand_sub: expected both operands to be Tensor or both to be Tile, "
        f"got lhs={type(lhs).__name__}, rhs={type(rhs).__name__}"
    )


def col_expand(lhs: T, rhs: T) -> T:
    """Column-wise expansion, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand(lhs, rhs)
    raise TypeError(
        "col_expand: expected (Tensor, Tensor) or (Tile, Tile), "
        f"got ({type(lhs).__name__}, {type(rhs).__name__})"
    )


def col_expand_div(lhs: T, rhs: T) -> T:
    """Column-wise broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_div(lhs, rhs)
    raise TypeError(
        "col_expand_div: expected (Tensor, Tensor) or (Tile, Tile), "
        f"got ({type(lhs).__name__}, {type(rhs).__name__})"
    )


def col_expand_sub(lhs: T, rhs: T) -> T:
    """Column-wise broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_sub(lhs, rhs)
    raise TypeError(
        "col_expand_sub: expected (Tensor, Tensor) or (Tile, Tile), "
        f"got ({type(lhs).__name__}, {type(rhs).__name__})"
    )


def expands(target: Tensor | Tile, scalar: int | float | Scalar) -> Tensor | Tile:
    """Expand scalar to target shape, dispatched by target type."""
    if isinstance(target, Tensor):
        return _tensor.expands(target, scalar)
    if isinstance(target, Tile):
        return _tile.expands(target, scalar)
    raise TypeError(f"expands: expected Tensor or Tile, got {type(target).__name__}")


def reshape(input: T, shape: Sequence[IntLike]) -> T:
    """Reshape operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.reshape(input, shape)
    if isinstance(input, Tile):
        return _tile.reshape(input, shape)
    raise TypeError(f"reshape: expected Tensor or Tile, got {type(input).__name__}")


def transpose(input: T, axis1: int, axis2: int) -> T:
    """Transpose operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.transpose(input, axis1, axis2)
    if isinstance(input, Tile):
        return _tile.transpose(input, axis1, axis2)
    raise TypeError(f"transpose: expected Tensor or Tile, got {type(input).__name__}")


def concat(src0: T, src1: T) -> T:
    """Column-wise concatenation, dispatched by input type."""
    if isinstance(src0, Tensor) and isinstance(src1, Tensor):
        return _tensor.concat(src0, src1)
    if isinstance(src0, Tile) and isinstance(src1, Tile):
        return _tile.concat(src0, src1)
    raise TypeError(
        f"concat: src0 and src1 must be the same type (both Tensor or both Tile),"
        f" got {type(src0).__name__} and {type(src1).__name__}"
    )


def slice(
    input: T,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
) -> T:
    """Slice operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.slice(input, shape, offset, valid_shape)
    if isinstance(input, Tile):
        return _tile.slice(input, shape, offset, valid_shape)
    raise TypeError(f"slice: expected Tensor or Tile, got {type(input).__name__}")


def fillpad(value: T, pad_value: PadValue = PadValue.zero) -> T:
    """Fill invalid elements, dispatched by input type."""
    if isinstance(value, Tensor):
        return _tensor.fillpad(value, pad_value)
    if isinstance(value, Tile):
        return _tile.fillpad(value, pad_value)
    raise TypeError(f"fillpad: expected Tensor or Tile, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Different-signature ops (accept superset of kwargs)
# ---------------------------------------------------------------------------


@overload
def matmul(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = ...,
    a_trans: bool = ...,
    b_trans: bool = ...,
    c_matrix_nz: bool = ...,
) -> Tensor: ...
@overload
def matmul(lhs: Tile, rhs: Tile) -> Tile: ...


def matmul(
    lhs: T,
    rhs: T,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> T:
    """Matrix multiplication, dispatched by input type.

    Tensor path accepts extra kwargs (out_dtype, a_trans, b_trans, c_matrix_nz).
    Tile path ignores them.
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.matmul(lhs, rhs, out_dtype, a_trans, b_trans, c_matrix_nz)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.matmul(lhs, rhs)
    raise TypeError(f"matmul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def row_max(input: T, tmp_tile: Tile | None = None) -> T:
    """Row-wise max reduction, dispatched by input type.

    For Tile inputs, tmp_tile is required as a temporary buffer.
    For Tensor inputs, tmp_tile is ignored.
    """
    if isinstance(input, Tensor):
        return _tensor.row_max(input)
    if isinstance(input, Tile):
        if tmp_tile is None:
            raise ValueError("row_max on Tile requires tmp_tile argument")
        return _tile.row_max(input, tmp_tile)
    raise TypeError(f"row_max: expected Tensor or Tile, got {type(input).__name__}")


def row_sum(input: T, tmp_tile: Tile | None = None) -> T:
    """Row-wise sum reduction, dispatched by input type.

    For Tile inputs, tmp_tile is required as a temporary buffer.
    For Tensor inputs, tmp_tile is ignored.
    """
    if isinstance(input, Tensor):
        return _tensor.row_sum(input)
    if isinstance(input, Tile):
        if tmp_tile is None:
            raise ValueError("row_sum on Tile requires tmp_tile argument")
        return _tile.row_sum(input, tmp_tile)
    raise TypeError(f"row_sum: expected Tensor or Tile, got {type(input).__name__}")


def row_min(input: T, tmp_tile: Tile | None = None) -> T:
    """Row-wise min reduction, dispatched by input type.

    For Tile inputs, tmp_tile is required as a temporary buffer.
    For Tensor inputs, tmp_tile is ignored.
    """
    if isinstance(input, Tensor):
        return _tensor.row_min(input)
    if isinstance(input, Tile):
        if tmp_tile is None:
            raise ValueError("row_min on Tile requires tmp_tile argument")
        return _tile.row_min(input, tmp_tile)
    raise TypeError(f"row_min: expected Tensor or Tile, got {type(input).__name__}")


@overload
def cast(
    input: Tensor,
    target_type: int | DataType,
    mode: str | int = "round",
) -> Tensor: ...


@overload
def cast(
    input: Tile,
    target_type: int | DataType,
    mode: str | int = "round",
) -> Tile: ...


@overload
def cast(
    input: Scalar,
    target_type: int | DataType,
    mode: str | int = "round",
) -> Scalar: ...


def cast(
    input: Tensor | Tile | Scalar,
    target_type: int | DataType,
    mode: str | int = "round",
) -> Tensor | Tile | Scalar:
    """Type casting, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.cast(input, target_type, mode)
    if isinstance(input, Tile):
        return _tile.cast(input, target_type, mode)
    if isinstance(input, Scalar):
        if resolve_cast_mode(mode) != 2:
            raise ValueError(f"cast: Scalar inputs do not support non-default mode, got mode={mode!r}")
        dtype = DataType(target_type) if isinstance(target_type, int) else target_type
        return Scalar(expr=_ir_core.cast(input.unwrap(), dtype))
    raise TypeError(f"cast: expected Tensor, Tile, or Scalar, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Tile-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def create_tile(shape: list[int], dtype: DataType, target_memory: MemorySpace) -> Tile:
    """Create a tile at specific memory space."""
    return _tile.create(shape, dtype, target_memory)


# ---------------------------------------------------------------------------
# Scalar read/write with type dispatch
# ---------------------------------------------------------------------------


def read(src: Tensor | Tile, offset: IntLike | Sequence[IntLike]) -> Scalar:
    """Read a scalar value at given indices, dispatched by source type.

    Args:
        src: Source tensor (global memory) or tile (unified buffer)
        offset: A single index expression (for 1-D flat access) or index list
            (one per dimension) into the source

    Returns:
        Scalar wrapping the read value
    """
    if isinstance(src, Tensor):
        return _tensor.read(src, offset)
    if isinstance(src, Tile):
        return _tile.read(src, offset)
    raise TypeError(f"read: expected Tensor or Tile, got {type(src).__name__}")


def write(dst: Tensor | Tile, offset: IntLike | Sequence[IntLike], value: Scalar) -> None:
    """Write a scalar value to a tensor or tile at given indices.

    Args:
        dst: Destination tensor (global memory) or tile (unified buffer)
        offset: A single index expression (for 1-D flat access) or index list
            (one per dimension) into the destination
        value: Scalar value to write
    """
    if isinstance(dst, Tensor):
        return _tensor.write(dst, offset, value)
    if isinstance(dst, Tile):
        return _tile.write(dst, offset, value)
    raise TypeError(f"write: expected Tensor or Tile, got {type(dst).__name__}")
