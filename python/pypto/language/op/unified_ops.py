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
from typing import Any, NoReturn, TypeVar, overload

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "exp",
    "neg",
    "abs",
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
    "batch_matmul",
    "matmul_acc",
    "row_max",
    "row_sum",
    "row_min",
    "col_sum",
    "col_max",
    "col_min",
    "cast",
    "cmp",
    "set_validshape",
    "create_tile",
    "read",
    "write",
]

from pypto.ir.utils import _get_span_or_capture, resolve_cast_mode
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


def _raise_type_dispatch_error(op_name: str, *args: object) -> NoReturn:
    """Raise TypeError for mixed Tensor/Tile or unsupported argument types.

    Op-name prefix is auto-normalized to ``pl.<op>`` so the message reads
    consistently whether the user invoked the wrapper directly or the DSL
    parser surfaced the error.
    """
    qualified = op_name if op_name.startswith("pl.") else f"pl.{op_name}"
    has_tensor = any(isinstance(a, Tensor) for a in args)
    has_tile = any(isinstance(a, Tile) for a in args)
    types = ", ".join(type(a).__name__ for a in args)
    if has_tensor and has_tile:
        raise TypeError(
            f"{qualified}: cannot mix Tensor and Tile arguments "
            f"({types}). All operands must be the same type "
            f"level — either all Tensor or all Tile"
        )
    raise TypeError(f"{qualified}: expected Tensor or Tile operands, got ({types})")


def _is_scalar_like(v: object) -> bool:
    """True for Scalar, Python int/float, or raw Expr with ScalarType.

    Used by the unified arithmetic wrappers so parser-shaped operands
    (raw ``ConstInt`` / ``ConstFloat`` literals, IR scalar Vars, etc.)
    flow through the scalar branch alongside DSL ``Scalar`` and Python
    literals.
    """
    if isinstance(v, (Scalar, int, float)):
        return True
    return isinstance(v, _ir_core.Expr) and isinstance(v.type, _ir_core.ScalarType)


def _to_scalar_expr(v: Any) -> _ir_core.Expr:
    """Coerce a scalar-like value to an ``Expr``.

    Caller must have already passed :func:`_is_scalar_like`. ``Scalar`` is
    unwrapped, raw ``Expr`` is returned as-is, and Python ``int`` / ``float``
    are materialized as ``ConstInt`` / ``ConstFloat`` with the parser-pinned
    span (or frame-captured fallback).
    """
    if isinstance(v, Scalar):
        return v.unwrap()
    if isinstance(v, _ir_core.Expr):
        return v
    if isinstance(v, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"scalar arithmetic does not accept bool, got {v!r}")
    if isinstance(v, int):
        return _ir_core.ConstInt(v, DataType.INDEX, _get_span_or_capture())
    return _ir_core.ConstFloat(float(v), DataType.DEFAULT_CONST_FLOAT, _get_span_or_capture())


# ---------------------------------------------------------------------------
# Binary arithmetic with scalar auto-dispatch
# ---------------------------------------------------------------------------

# --- add ---


@overload
def add(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def add(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
@overload
def add(lhs: Scalar, rhs: Scalar | int | float) -> Scalar: ...
def add(lhs, rhs):
    """Element-wise addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.adds(lhs, rhs)
    if _is_scalar_like(lhs) and _is_scalar_like(rhs):
        return Scalar(expr=_to_scalar_expr(lhs) + _to_scalar_expr(rhs))
    _raise_type_dispatch_error("add", lhs, rhs)


# --- sub ---


@overload
def sub(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def sub(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
@overload
def sub(lhs: Scalar, rhs: Scalar | int | float) -> Scalar: ...
def sub(lhs, rhs):
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.subs(lhs, rhs)
    if _is_scalar_like(lhs) and _is_scalar_like(rhs):
        return Scalar(expr=_to_scalar_expr(lhs) - _to_scalar_expr(rhs))
    _raise_type_dispatch_error("sub", lhs, rhs)


# --- mul ---


@overload
def mul(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def mul(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
@overload
def mul(lhs: Scalar, rhs: Scalar | int | float) -> Scalar: ...
def mul(lhs, rhs):
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.muls(lhs, rhs)
    if _is_scalar_like(lhs) and _is_scalar_like(rhs):
        return Scalar(expr=_to_scalar_expr(lhs) * _to_scalar_expr(rhs))
    _raise_type_dispatch_error("mul", lhs, rhs)


# --- div ---


@overload
def div(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def div(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
@overload
def div(lhs: Scalar, rhs: Scalar | int | float) -> Scalar: ...
def div(lhs, rhs):
    """Element-wise division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.divs(lhs, rhs)
    if _is_scalar_like(lhs) and _is_scalar_like(rhs):
        return Scalar(expr=_to_scalar_expr(lhs) / _to_scalar_expr(rhs))
    _raise_type_dispatch_error("div", lhs, rhs)


# ---------------------------------------------------------------------------
# Simple overlapping ops (dispatch on first arg type)
# ---------------------------------------------------------------------------


@overload
def maximum(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def maximum(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
def maximum(lhs, rhs):
    """Element-wise maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.maximum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.maximum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.maximums(lhs, rhs)
    _raise_type_dispatch_error("maximum", lhs, rhs)


@overload
def minimum(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def minimum(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
def minimum(lhs, rhs):
    """Element-wise minimum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.minimum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.minimum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.minimums(lhs, rhs)
    _raise_type_dispatch_error("minimum", lhs, rhs)


def exp(input: T) -> T:
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.exp(input)
    if isinstance(input, Tile):
        return _tile.exp(input)
    raise TypeError(f"pl.exp: expected Tensor or Tile, got {type(input).__name__}")


def neg(input: T) -> T:
    """Element-wise negation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.neg(input)
    if isinstance(input, Tile):
        return _tile.neg(input)
    raise TypeError(f"pl.neg: expected Tensor or Tile, got {type(input).__name__}")


def abs(input: T) -> T:
    """Element-wise absolute value, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.abs(input)
    if isinstance(input, Tile):
        return _tile.abs(input)
    raise TypeError(f"pl.abs: expected Tensor or Tile, got {type(input).__name__}")


def recip(input: T) -> T:
    """Element-wise reciprocal (1/x), dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.recip(input)
    if isinstance(input, Tile):
        return _tile.recip(input)
    raise TypeError(f"pl.recip: expected Tensor or Tile, got {type(input).__name__}")


def sqrt(input: T) -> T:
    """Element-wise square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.sqrt(input)
    if isinstance(input, Tile):
        return _tile.sqrt(input)
    raise TypeError(f"pl.sqrt: expected Tensor or Tile, got {type(input).__name__}")


def rsqrt(input: T, high_precision: bool = False) -> T:
    """Element-wise reciprocal square root, dispatched by input type.

    ``high_precision`` applies to the tensor path where the compiler inserts
    the scratch allocation. At the tile level, callers that need the high-
    precision path must call ``pl.tile.rsqrt(src, tmp=...)`` directly since
    buffer lifetimes are user-managed there.
    """
    if isinstance(input, Tensor):
        return _tensor.rsqrt(input, high_precision=high_precision)
    if isinstance(input, Tile):
        return _tile.rsqrt(input)
    raise TypeError(f"pl.rsqrt: expected Tensor or Tile, got {type(input).__name__}")


def row_expand_mul(lhs: T, rhs: T) -> T:
    """Row-wise broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_mul(lhs, rhs)
    _raise_type_dispatch_error("row_expand_mul", lhs, rhs)


def row_expand_div(lhs: T, rhs: T) -> T:
    """Row-wise broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_div(lhs, rhs)
    _raise_type_dispatch_error("row_expand_div", lhs, rhs)


def col_expand_mul(lhs: T, rhs: T) -> T:
    """Column-wise broadcast multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_mul(lhs, rhs)
    _raise_type_dispatch_error("col_expand_mul", lhs, rhs)


def row_expand(lhs: T, rhs: T) -> T:
    """Row-wise expansion, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand(lhs, rhs)
    _raise_type_dispatch_error("row_expand", lhs, rhs)


def row_expand_add(lhs: T, rhs: T) -> T:
    """Row-wise broadcast addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_add(lhs, rhs)
    _raise_type_dispatch_error("row_expand_add", lhs, rhs)


def row_expand_sub(lhs: T, rhs: T) -> T:
    """Row-wise broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_sub(lhs, rhs)
    _raise_type_dispatch_error("row_expand_sub", lhs, rhs)


def col_expand(lhs: T, rhs: T) -> T:
    """Column-wise expansion, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand(lhs, rhs)
    _raise_type_dispatch_error("col_expand", lhs, rhs)


def col_expand_div(lhs: T, rhs: T) -> T:
    """Column-wise broadcast division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_div(lhs, rhs)
    _raise_type_dispatch_error("col_expand_div", lhs, rhs)


def col_expand_sub(lhs: T, rhs: T) -> T:
    """Column-wise broadcast subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_sub(lhs, rhs)
    _raise_type_dispatch_error("col_expand_sub", lhs, rhs)


def expands(target: Tensor | Tile, scalar: int | float | Scalar) -> Tensor | Tile:
    """Expand scalar to target shape, dispatched by target type."""
    if isinstance(target, Tensor):
        return _tensor.expands(target, scalar)
    if isinstance(target, Tile):
        return _tile.expands(target, scalar)
    raise TypeError(f"pl.expands: expected Tensor or Tile, got {type(target).__name__}")


def reshape(input: T, shape: Sequence[IntLike]) -> T:
    """Reshape operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.reshape(input, shape)
    if isinstance(input, Tile):
        return _tile.reshape(input, shape)
    raise TypeError(f"pl.reshape: expected Tensor or Tile, got {type(input).__name__}")


def transpose(input: T, axis1: int, axis2: int) -> T:
    """Transpose operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.transpose(input, axis1, axis2)
    if isinstance(input, Tile):
        return _tile.transpose(input, axis1, axis2)
    raise TypeError(f"pl.transpose: expected Tensor or Tile, got {type(input).__name__}")


def concat(src0: T, src1: T) -> T:
    """Column-wise concatenation, dispatched by input type."""
    if isinstance(src0, Tensor) and isinstance(src1, Tensor):
        return _tensor.concat(src0, src1)
    if isinstance(src0, Tile) and isinstance(src1, Tile):
        return _tile.concat(src0, src1)
    _raise_type_dispatch_error("concat", src0, src1)


def slice(
    input: T,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
    drop_dims: Sequence[int | _ir_core.Expr] | None = None,
) -> T:
    """Slice operation, dispatched by input type.

    ``drop_dims`` lists axes to erase from the result type (numpy-style rank
    reduction); each must be a static unit dim of ``shape``. ``None`` / ``[]``
    drops nothing.
    """
    if isinstance(input, Tensor):
        return _tensor.slice(input, shape, offset, valid_shape, drop_dims)
    if isinstance(input, Tile):
        return _tile.slice(input, shape, offset, valid_shape, drop_dims)
    raise TypeError(f"pl.slice: expected Tensor or Tile, got {type(input).__name__}")


def fillpad(value: T, pad_value: PadValue | int | float = PadValue.zero) -> T:
    """Fill invalid elements, dispatched by input type.

    ``pad_value`` accepts the ``PadValue`` enum or the literal sugars ``0``,
    ``math.inf``, ``-math.inf``. Other values raise — the hardware only
    supports the three padding modes.
    """
    if isinstance(value, Tensor):
        return _tensor.fillpad(value, pad_value)
    if isinstance(value, Tile):
        return _tile.fillpad(value, pad_value)
    raise TypeError(f"pl.fillpad: expected Tensor or Tile, got {type(value).__name__}")


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

    For Tensor inputs with rank > 2 on either operand, the call is lowered to
    ``tile.batch_matmul`` (with batch broadcasting) by ``ConvertTensorToTileOps``
    and then unrolled to per-batch ``tile.matmul`` by ``FlattenTileNdTo2D``.
    Use this entry point (rather than ``pl.batch_matmul``) for tensor-level ND
    matmul.
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.matmul(lhs, rhs, out_dtype, a_trans, b_trans, c_matrix_nz)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.matmul(lhs, rhs)
    _raise_type_dispatch_error("matmul", lhs, rhs)


def batch_matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Tile-only batched matrix multiplication.

    Tensor batched matmul is handled by ``pl.matmul`` / ``pl.tensor.matmul``:
    when any operand has rank > 2, ``ConvertTensorToTileOps`` automatically
    dispatches to ``tile.batch_matmul`` (and ``FlattenTileNdTo2D`` later
    unrolls it). Use this op only when you are working at the tile level.
    """
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.batch_matmul(lhs, rhs)
    _raise_type_dispatch_error("batch_matmul", lhs, rhs)


# ---------------------------------------------------------------------------
# matmul_acc (Tensor or Tile)
# ---------------------------------------------------------------------------


@overload
def matmul_acc(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    a_trans: bool = ...,
    b_trans: bool = ...,
) -> Tensor: ...
@overload
def matmul_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile: ...


def matmul_acc(
    acc: T,
    lhs: T,
    rhs: T,
    a_trans: bool = False,
    b_trans: bool = False,
) -> T:
    """Matrix multiplication with accumulation, dispatched by input type.

    Tensor path accepts extra kwargs (a_trans, b_trans).
    Tile path ignores them.

    For Tensor inputs with rank > 2 on any of acc/lhs/rhs, the call is lowered
    to ``tile.batch_matmul_acc`` (with batch broadcasting on lhs/rhs vs the
    fixed acc batch) by ``ConvertTensorToTileOps`` and then unrolled to
    per-batch ``tile.matmul_acc`` by ``FlattenTileNdTo2D``.
    """
    if isinstance(acc, Tensor) and isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.matmul_acc(acc, lhs, rhs, a_trans, b_trans)
    if isinstance(acc, Tile) and isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.matmul_acc(acc, lhs, rhs)
    _raise_type_dispatch_error("matmul_acc", acc, lhs, rhs)


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
    raise TypeError(f"pl.row_max: expected Tensor or Tile, got {type(input).__name__}")


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
    raise TypeError(f"pl.row_sum: expected Tensor or Tile, got {type(input).__name__}")


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
    raise TypeError(f"pl.row_min: expected Tensor or Tile, got {type(input).__name__}")


def col_sum(input: T, tmp_tile: Tile | None = None) -> T:
    """Column-wise sum reduction, dispatched by input type.

    For Tile inputs, passing ``tmp_tile`` activates the binary-tree reduction
    path; omitting it uses the sequential path.
    """
    if isinstance(input, Tile):
        return _tile.col_sum(input, tmp_tile)
    _raise_type_dispatch_error("col_sum", input)


def col_max(input: T) -> T:
    """Column-wise max reduction, dispatched by input type."""
    if isinstance(input, Tile):
        return _tile.col_max(input)
    _raise_type_dispatch_error("col_max", input)


def col_min(input: T) -> T:
    """Column-wise min reduction, dispatched by input type."""
    if isinstance(input, Tile):
        return _tile.col_min(input)
    _raise_type_dispatch_error("col_min", input)


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
    if _is_scalar_like(input):
        if resolve_cast_mode(mode) != 2:
            raise ValueError(f"cast: Scalar inputs do not support non-default mode, got mode={mode!r}")
        dtype = DataType(target_type) if isinstance(target_type, int) else target_type
        return Scalar(expr=_ir_core.cast(_to_scalar_expr(input), dtype))
    raise TypeError(f"pl.cast: expected Tensor, Tile, or Scalar, got {type(input).__name__}")


@overload
def cmp(lhs: Tensor, rhs: Tensor | int | float | Scalar, cmp_type: int = 0) -> Tensor: ...
@overload
def cmp(lhs: Tile, rhs: Tile | int | float | Scalar, cmp_type: int = 0) -> Tile: ...
def cmp(lhs, rhs, cmp_type: int = 0):
    """Element-wise comparison, dispatched by input type.

    Comparison type codes: ``0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge``. For Tile
    inputs with a scalar ``rhs``, dispatches to ``tile.cmps`` automatically.
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.cmp(lhs, rhs, cmp_type=cmp_type)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.cmp(lhs, rhs, cmp_type=cmp_type)
    if isinstance(lhs, Tile) and _is_scalar_like(rhs):
        return _tile.cmps(lhs, rhs, cmp_type=cmp_type)
    _raise_type_dispatch_error("cmp", lhs, rhs)


@overload
def set_validshape(input: Tensor, valid_rows: IntLike, valid_cols: IntLike) -> Tensor: ...
@overload
def set_validshape(input: Tile, valid_rows: IntLike, valid_cols: IntLike) -> Tile: ...
def set_validshape(input, valid_rows, valid_cols):
    """Update valid-shape metadata without data movement, dispatched by input type.

    .. note::
        Internal API — intended for compiler-generated code. End users should
        prefer ``pl.load(..., valid_shapes=...)`` plus ``pl.tile.fillpad``.
    """
    if isinstance(input, Tensor):
        return _tensor.set_validshape(input, valid_rows, valid_cols)
    if isinstance(input, Tile):
        return _tile.set_validshape(input, valid_rows, valid_cols)
    raise TypeError(f"pl.set_validshape: expected Tensor or Tile, got {type(input).__name__}")


# ---------------------------------------------------------------------------
# Tile-only ops promoted to unified namespace
# ---------------------------------------------------------------------------


def create_tile(
    shape: list[int],
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
) -> Tile:
    """Create a tile at specific memory space.

    ``target_memory`` defaults to ``Vec`` to match the underlying
    ``tile.create`` wrapper — direct callers like
    ``pl.create_tile(shape, dtype)`` (omitting target_memory) keep working.
    """
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
    raise TypeError(f"pl.read: expected Tensor or Tile, got {type(src).__name__}")


def write(
    dst: Tensor | Tile,
    offset: IntLike | Sequence[IntLike],
    value: Scalar,
) -> _ir_core.Expr:
    """Write a scalar value to a tensor or tile at given indices.

    Args:
        dst: Destination tensor (global memory) or tile (unified buffer)
        offset: A single index expression (for 1-D flat access) or index list
            (one per dimension) into the destination
        value: Scalar value to write

    Returns:
        Underlying ``tensor.write`` / ``tile.write`` call expression. Direct
        callers ignore it; the DSL parser surfaces it as an ``EvalStmt``.
    """
    if isinstance(dst, Tensor):
        return _tensor.write(dst, offset, value)
    if isinstance(dst, Tile):
        return _tile.write(dst, offset, value)
    raise TypeError(f"pl.write: expected Tensor or Tile, got {type(dst).__name__}")
