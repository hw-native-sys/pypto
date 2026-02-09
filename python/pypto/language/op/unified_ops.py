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

from typing import Literal, Optional, Union, overload

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..scalar import Scalar
from ..tensor import Tensor
from ..tile import Tile
from . import block_ops as _block
from . import tensor_ops as _tensor

# ---------------------------------------------------------------------------
# Binary arithmetic with scalar auto-dispatch
# ---------------------------------------------------------------------------

# --- add ---


@overload
def add(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor: ...
@overload
def add(lhs: Tile, rhs: Union[int, float, Tile, Scalar]) -> Tile: ...


def add(lhs, rhs):  # noqa: F811
    """Element-wise addition, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.add(lhs, rhs)
    if isinstance(lhs, Tile):
        if isinstance(rhs, (Tile,)):
            return _block.add(lhs, rhs)
        return _block.adds(lhs, rhs)
    raise TypeError(f"add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- sub ---


@overload
def sub(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor: ...
@overload
def sub(lhs: Tile, rhs: Union[int, float, Tile, Scalar]) -> Tile: ...


def sub(lhs, rhs):  # noqa: F811
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.sub(lhs, rhs)
    if isinstance(lhs, Tile):
        if isinstance(rhs, (Tile,)):
            return _block.sub(lhs, rhs)
        return _block.subs(lhs, rhs)
    raise TypeError(f"sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- mul ---


@overload
def mul(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor: ...
@overload
def mul(lhs: Tile, rhs: Union[int, float, Tile, Scalar]) -> Tile: ...


def mul(lhs, rhs):  # noqa: F811
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.mul(lhs, rhs)
    if isinstance(lhs, Tile):
        if isinstance(rhs, (Tile,)):
            return _block.mul(lhs, rhs)
        return _block.muls(lhs, rhs)
    raise TypeError(f"mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# --- div ---


@overload
def div(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor: ...
@overload
def div(lhs: Tile, rhs: Union[int, float, Tile, Scalar]) -> Tile: ...


def div(lhs, rhs):  # noqa: F811
    """Element-wise division, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.div(lhs, rhs)
    if isinstance(lhs, Tile):
        if isinstance(rhs, (Tile,)):
            return _block.div(lhs, rhs)
        return _block.divs(lhs, rhs)
    raise TypeError(f"div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


# ---------------------------------------------------------------------------
# Simple overlapping ops (dispatch on first arg type)
# ---------------------------------------------------------------------------


@overload
def maximum(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def maximum(lhs: Tile, rhs: Tile) -> Tile: ...


def maximum(lhs, rhs):  # noqa: F811
    """Element-wise maximum, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.maximum(lhs, rhs)
    if isinstance(lhs, Tile):
        return _block.maximum(lhs, rhs)
    raise TypeError(f"maximum: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


@overload
def exp(input: Tensor) -> Tensor: ...
@overload
def exp(input: Tile) -> Tile: ...


def exp(input):  # noqa: F811
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.exp(input)
    if isinstance(input, Tile):
        return _block.exp(input)
    raise TypeError(f"exp: expected Tensor or Tile, got {type(input).__name__}")


@overload
def reshape(input: Tensor, shape: list[Union[int, Expr]]) -> Tensor: ...
@overload
def reshape(input: Tile, shape: list[Union[int, Expr]]) -> Tile: ...


def reshape(input, shape):  # noqa: F811
    """Reshape operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.reshape(input, shape)
    if isinstance(input, Tile):
        return _block.reshape(input, shape)
    raise TypeError(f"reshape: expected Tensor or Tile, got {type(input).__name__}")


@overload
def transpose(input: Tensor, axis1: int, axis2: int) -> Tensor: ...
@overload
def transpose(input: Tile, axis1: int, axis2: int) -> Tile: ...


def transpose(input, axis1, axis2):  # noqa: F811
    """Transpose operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.transpose(input, axis1, axis2)
    if isinstance(input, Tile):
        return _block.transpose(input, axis1, axis2)
    raise TypeError(f"transpose: expected Tensor or Tile, got {type(input).__name__}")


@overload
def view(input: Tensor, shape: list[Union[int, Expr]], offset: list[Union[int, Expr]]) -> Tensor: ...
@overload
def view(input: Tile, shape: list[Union[int, Expr]], offset: list[Union[int, Expr]]) -> Tile: ...


def view(input, shape, offset):  # noqa: F811
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


def matmul(  # noqa: F811
    lhs,
    rhs,
    out_dtype=None,
    a_trans=False,
    b_trans=False,
    c_matrix_nz=False,
):
    """Matrix multiplication, dispatched by input type.

    Tensor path accepts extra kwargs (out_dtype, a_trans, b_trans, c_matrix_nz).
    Tile path ignores them.
    """
    if isinstance(lhs, Tensor):
        return _tensor.matmul(lhs, rhs, out_dtype, a_trans, b_trans, c_matrix_nz)
    if isinstance(lhs, Tile):
        return _block.matmul(lhs, rhs)
    raise TypeError(f"matmul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


@overload
def row_max(input: Tensor, axis: int = ..., keep_dim: Union[int, bool] = ...) -> Tensor: ...
@overload
def row_max(input: Tile) -> Tile: ...


def row_max(input, axis=-1, keep_dim=1):  # noqa: F811
    """Row-wise max reduction, dispatched by input type.

    Tensor path accepts axis and keep_dim kwargs.
    Tile path ignores them.
    """
    if isinstance(input, Tensor):
        return _tensor.row_max(input, axis, keep_dim)
    if isinstance(input, Tile):
        return _block.row_max(input)
    raise TypeError(f"row_max: expected Tensor or Tile, got {type(input).__name__}")


@overload
def row_sum(input: Tensor, axis: int = ..., keep_dim: Union[int, bool] = ...) -> Tensor: ...
@overload
def row_sum(input: Tile) -> Tile: ...


def row_sum(input, axis=-1, keep_dim=1):  # noqa: F811
    """Row-wise sum reduction, dispatched by input type.

    Tensor path accepts axis and keep_dim kwargs.
    Tile path ignores them.
    """
    if isinstance(input, Tensor):
        return _tensor.row_sum(input, axis, keep_dim)
    if isinstance(input, Tile):
        return _block.row_sum(input)
    raise TypeError(f"row_sum: expected Tensor or Tile, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Tensor-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def cast(
    input: Tensor,
    target_type: Union[int, DataType],
    mode: Literal["round", "floor", "ceil"] = "round",
) -> Tensor:
    """Type casting (tensor-only at language level)."""
    return _tensor.cast(input, target_type, mode)
