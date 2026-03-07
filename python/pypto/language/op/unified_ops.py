# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unified operation dispatch for PyPTO Language DSL.

Provides type-dispatched wrappers that auto-select between tensor and block
operations based on the input type (Tensor vs Tile). Users can write
``pl.add(a, b)`` instead of explicitly choosing ``pl.tensor.add``
or ``pl.block.add``.
"""

from collections.abc import Sequence
from typing import Literal, TypeVar, overload

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "exp",
    "neg",
    "recip",
    "sqrt",
    "rsqrt",
    "log",
    "abs",
    "relu",
    "row_expand",
    "row_expand_sub",
    "row_expand_div",
    "row_expand_mul",
    "row_expand_add",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "reshape",
    "transpose",
    "view",
    "matmul",
    "row_max",
    "row_sum",
    "cast",
    "create_tile",
]

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace

from ..typing import IntLike, Scalar, Tensor, Tile
from . import block_ops as _block
from . import tensor_ops as _tensor

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
        return _block.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.adds(lhs, rhs)
    raise TypeError(f"add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- sub ---


def sub(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.subs(lhs, rhs)
    raise TypeError(f"sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- mul ---


def mul(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.muls(lhs, rhs)
    raise TypeError(f"mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- div ---


def div(lhs: T, rhs: T | int | float | Scalar) -> T:
    """Element-wise division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.divs(lhs, rhs)
    raise TypeError(f"div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# ---------------------------------------------------------------------------
# Simple overlapping ops (dispatch on first arg type)
# ---------------------------------------------------------------------------


def maximum(lhs: T, rhs: T) -> T:
    """Element-wise maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.maximum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.maximum(lhs, rhs)
    raise TypeError(f"maximum: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def minimum(lhs: T, rhs: T) -> T:
    """Element-wise minimum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.minimum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.minimum(lhs, rhs)
    raise TypeError(f"minimum: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def exp(input: T) -> T:
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.exp(input)
    if isinstance(input, Tile):
        return _block.exp(input)
    raise TypeError(f"exp: expected Tensor or Tile, got {type(input).__name__}")


def neg(input: T) -> T:
    """Element-wise negation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.neg(input)
    if isinstance(input, Tile):
        return _block.neg(input)
    raise TypeError(f"neg: expected Tensor or Tile, got {type(input).__name__}")


def recip(input: T) -> T:
    """Element-wise reciprocal, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.recip(input)
    if isinstance(input, Tile):
        return _block.recip(input)
    raise TypeError(f"recip: expected Tensor or Tile, got {type(input).__name__}")


def sqrt(input: T) -> T:
    """Element-wise square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.sqrt(input)
    if isinstance(input, Tile):
        return _block.sqrt(input)
    raise TypeError(f"sqrt: expected Tensor or Tile, got {type(input).__name__}")


def rsqrt(input: T) -> T:
    """Element-wise reciprocal square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.rsqrt(input)
    if isinstance(input, Tile):
        return _block.rsqrt(input)
    raise TypeError(f"rsqrt: expected Tensor or Tile, got {type(input).__name__}")


def log(input: T) -> T:
    """Element-wise natural logarithm, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.log(input)
    if isinstance(input, Tile):
        return _block.log(input)
    raise TypeError(f"log: expected Tensor or Tile, got {type(input).__name__}")


def abs(input: T) -> T:
    """Element-wise absolute value, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.abs(input)
    if isinstance(input, Tile):
        return _block.abs(input)
    raise TypeError(f"abs: expected Tensor or Tile, got {type(input).__name__}")


def relu(input: T) -> T:
    """Element-wise relu, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.relu(input)
    if isinstance(input, Tile):
        return _block.relu(input)
    raise TypeError(f"relu: expected Tensor or Tile, got {type(input).__name__}")


def reshape(input: T, shape: Sequence[IntLike]) -> T:
    """Reshape operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.reshape(input, shape)
    if isinstance(input, Tile):
        return _block.reshape(input, shape)
    raise TypeError(f"reshape: expected Tensor or Tile, got {type(input).__name__}")


def transpose(input: T, axis1: int, axis2: int) -> T:
    """Transpose operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.transpose(input, axis1, axis2)
    if isinstance(input, Tile):
        return _block.transpose(input, axis1, axis2)
    raise TypeError(f"transpose: expected Tensor or Tile, got {type(input).__name__}")


def view(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T:
    """View/slice operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.view(input, shape, offset)
    if isinstance(input, Tile):
        return _block.view(input, shape, offset)
    raise TypeError(f"view: expected Tensor or Tile, got {type(input).__name__}")


def row_expand(input: T) -> T:
    """Row broadcast, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.row_expand(input)
    if isinstance(input, Tile):
        return _block.row_expand(input)
    raise TypeError(f"row_expand: expected Tensor or Tile, got {type(input).__name__}")


def row_expand_sub(lhs: T, rhs: T) -> T:
    """Row broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.row_expand_sub(lhs, rhs)
    raise TypeError(f"row_expand_sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def row_expand_div(lhs: T, rhs: T) -> T:
    """Row broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.row_expand_div(lhs, rhs)
    raise TypeError(f"row_expand_div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def row_expand_mul(lhs: T, rhs: T) -> T:
    """Row broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.row_expand_mul(lhs, rhs)
    raise TypeError(f"row_expand_mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def row_expand_add(lhs: T, rhs: T) -> T:
    """Row broadcast addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.row_expand_add(lhs, rhs)
    raise TypeError(f"row_expand_add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def col_expand(lhs: T, rhs: T) -> T:
    """Column broadcast, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.col_expand(lhs, rhs)
    raise TypeError(f"col_expand: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def col_expand_mul(lhs: T, rhs: T) -> T:
    """Column broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.col_expand_mul(lhs, rhs)
    raise TypeError(f"col_expand_mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def col_expand_div(lhs: T, rhs: T) -> T:
    """Column broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.col_expand_div(lhs, rhs)
    raise TypeError(f"col_expand_div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def col_expand_sub(lhs: T, rhs: T) -> T:
    """Column broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.col_expand_sub(lhs, rhs)
    raise TypeError(f"col_expand_sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


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
        return _block.matmul(lhs, rhs)
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
        return _block.row_max(input, tmp_tile)
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
        return _block.row_sum(input, tmp_tile)
    raise TypeError(f"row_sum: expected Tensor or Tile, got {type(input).__name__}")


def cast(
    input: T,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> T:
    """Type casting, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.cast(input, target_type, mode)
    if isinstance(input, Tile):
        return _block.cast(input, target_type, mode)
    raise TypeError(f"cast: expected Tensor or Tile, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Tile-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def create_tile(shape: list[int], dtype: DataType, target_memory: MemorySpace) -> Tile:
    """Create a tile at specific memory space."""
    return _block.create_tile(shape, dtype, target_memory)
