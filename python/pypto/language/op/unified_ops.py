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
    "reshape",
    "transpose",
    "slice",
    "matmul",
    "row_max",
    "row_sum",
    "cast",
    "create_tile",
    "read",
    "write",
]

from pypto.ir.utils import resolve_cast_mode
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace

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


def slice(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T:
    """Slice operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.slice(input, shape, offset)
    if isinstance(input, Tile):
        return _tile.slice(input, shape, offset)
    raise TypeError(f"slice: expected Tensor or Tile, got {type(input).__name__}")


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
        from pypto.pypto_core import ir as _ir_core  # noqa: PLC0415

        dtype = DataType(target_type) if isinstance(target_type, int) else target_type
        return Scalar(expr=_ir_core.cast(input.unwrap(), dtype))
    raise TypeError(f"cast: expected Tensor, Tile, or Scalar, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Tile-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def create_tile(shape: list[int], dtype: DataType, target_memory: MemorySpace) -> Tile:
    """Create a tile at specific memory space."""
    return _tile.create_tile(shape, dtype, target_memory)


# ---------------------------------------------------------------------------
# Scalar read/write with type dispatch
# ---------------------------------------------------------------------------


def read(src: Tensor | Tile, offset: IntLike) -> Scalar:
    """Read a scalar value at a flat offset, dispatched by source type.

    Args:
        src: Source tensor (global memory) or tile (unified buffer)
        offset: Flat offset into the source

    Returns:
        Scalar wrapping the read value
    """
    if isinstance(src, Tensor):
        return _tensor.load_scalar(src, offset)
    if isinstance(src, Tile):
        return _tile.getval(src, offset)
    raise TypeError(f"read: expected Tensor or Tile, got {type(src).__name__}")


def write(dst: Tensor | Tile, offset: IntLike, value: Scalar) -> None:
    """Write a scalar value to a tensor or tile at a flat offset.

    Args:
        dst: Destination tensor (global memory) or tile (unified buffer)
        offset: Flat offset into the destination
        value: Scalar value to write
    """
    if isinstance(dst, Tensor):
        return _tensor.store_scalar(dst, offset, value)
    if isinstance(dst, Tile):
        return _tile.setval(dst, offset, value)
    raise TypeError(f"write: expected Tensor or Tile, got {type(dst).__name__}")
