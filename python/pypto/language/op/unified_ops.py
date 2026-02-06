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
``pl.op.add(a, b)`` instead of explicitly choosing ``pl.op.tensor.add``
or ``pl.op.block.add``.
"""

from typing import Literal, Optional, TypeVar, Union, overload

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..scalar import Scalar
from ..tensor import Tensor
from ..tile import Tile
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


def add(lhs: T, rhs: Union[T, int, float, Scalar]) -> T:
    """Element-wise addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.adds(lhs, rhs)
    raise TypeError(f"add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- sub ---


def sub(lhs: T, rhs: Union[T, int, float, Scalar]) -> T:
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.subs(lhs, rhs)
    raise TypeError(f"sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- mul ---


def mul(lhs: T, rhs: Union[T, int, float, Scalar]) -> T:
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return _block.muls(lhs, rhs)
    raise TypeError(f"mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- div ---


def div(lhs: T, rhs: Union[T, int, float, Scalar]) -> T:
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


def exp(input: T) -> T:
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.exp(input)
    if isinstance(input, Tile):
        return _block.exp(input)
    raise TypeError(f"exp: expected Tensor or Tile, got {type(input).__name__}")


def reshape(input: T, shape: list[Union[int, Expr]]) -> T:
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


def view(input: T, shape: list[Union[int, Expr]], offset: list[Union[int, Expr]]) -> T:
    """View/slice operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.view(input, shape, offset)
    if isinstance(input, Tile):
        return _block.view(input, shape, offset)
    raise TypeError(f"view: expected Tensor or Tile, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Different-signature ops (accept superset of kwargs)
# ---------------------------------------------------------------------------


@overload
def matmul(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: Optional[Union[int, DataType]] = ...,
    a_trans: bool = ...,
    b_trans: bool = ...,
    c_matrix_nz: bool = ...,
) -> Tensor: ...
@overload
def matmul(lhs: Tile, rhs: Tile) -> Tile: ...


def matmul(
    lhs: T,
    rhs: T,
    out_dtype: Optional[Union[int, DataType]] = None,
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


def row_max(input: T, tmp_tile: Optional[Tile] = None) -> T:
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


def row_sum(input: T, tmp_tile: Optional[Tile] = None) -> T:
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
    target_type: Union[int, DataType],
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> T:
    """Type casting."""
    if isinstance(input, Tensor):
        return _tensor.cast(input, target_type, mode)
    if isinstance(input, Tile):
        return _block.cast(input, target_type, mode)


# ---------------------------------------------------------------------------
# Tensor-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tile-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def create_tile(shape: list[int], dtype: DataType, target_memory: int) -> Tile:
    """Create a tile at specific memoryspace.
    target_memory: (1=UB, 2=L1, 3=L0A, 4=L0B)
    """
    return _block.create_tile(shape, dtype, target_memory)
