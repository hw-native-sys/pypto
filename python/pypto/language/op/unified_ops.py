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
from typing import Any, Literal, NoReturn, TypeVar, overload

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "part_add",
    "part_mul",
    "part_max",
    "part_min",
    "fmod",
    "fmods",
    "maximum",
    "minimum",
    "exp",
    "log",
    "sin",
    "cos",
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
    "row_expand_max",
    "row_expand_min",
    "row_expand_expdif",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "col_expand_add",
    "col_expand_max",
    "col_expand_min",
    "col_expand_expdif",
    "concat",
    "expands",
    "reshape",
    "reinterpret_view",
    "transpose",
    "slice",
    "fillpad",
    "fillpad_expand",
    "matmul",
    "batch_matmul",
    "matmul_acc",
    "row_max",
    "row_sum",
    "row_min",
    "row_prod",
    "col_sum",
    "col_max",
    "col_min",
    "col_prod",
    "row_argmax",
    "row_argmin",
    "col_argmax",
    "col_argmin",
    "cast",
    "cmp",
    "and_",
    "ands",
    "or_",
    "ors",
    "xor",
    "xors",
    "not_",
    "shl",
    "shls",
    "shr",
    "shrs",
    "set_validshape",
    "read",
    "write",
    "assemble",
    "gather_row",
    "scatter_update",
    "sort32",
    "mrgsort",
]

from pypto.ir.utils import _elem_dtype, _get_span_or_capture, resolve_cast_mode
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import AtomicType, PadValue

from ..typing import BoolLike, IntLike, Scalar, Tensor, Tile
from . import tensor_ops as _tensor
from . import tile_ops as _tile

# ---------------------------------------------------------------------------
# TypeVar
# ---------------------------------------------------------------------------

# Bound (not constrained) so concrete subclasses propagate through ``T -> T``
# signatures — ``pl.slice(dist_tensor, ...)`` should return ``DistributedTensor``,
# not get downgraded to plain ``Tensor`` by a constrained ``TypeVar("T", Tensor,
# Tile)``. Mixed-kind two-arg uses (e.g. ``pl.matmul(tensor, tile)``) are still
# caught at runtime by each function's ``isinstance`` dispatch — the runtime
# error is more informative than a constrained-TypeVar type-check failure.
T = TypeVar("T", bound="Tensor | Tile")


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


# ---------------------------------------------------------------------------
# Cross-path kwarg guards
#
# These wrappers accept the union of both levels' kwargs, so a kwarg that only
# the *other* dispatch path can honour must raise rather than be dropped: a
# silently discarded ``b_trans`` compiles wrong math, and a discarded scratch
# tile leaves the caller's buffer dead while it still consumes UB budget.
# Only a non-default value raises — spelling out the documented default keeps
# working.
#
# **Every guard in this module raises ``TypeError``**, in both directions: an
# argument the dispatched path cannot honour, and one it requires but did not
# get. Both are "these arguments do not match this overload" — the class CPython
# itself raises for an unexpected keyword or a missing required argument — not a
# bad *value*. Deeper validation reached through these wrappers still raises
# ``ValueError`` (``pypto::ValueError`` from a C++ ``CHECK`` is registered as a
# Python ``ValueError`` subclass), so a direct-API caller guarding a whole call
# should catch both; the split here is only about which layer rejected it.
# The DSL path is unaffected either way: ``ast_parser`` catches ``(TypeError,
# ValueError)`` and re-raises ``InvalidOperationError`` with a span.
# ---------------------------------------------------------------------------

# The ``@overload`` declarations mirror that rule with ``Literal[False]`` /
# ``None`` defaults on the path that cannot honour a kwarg: the documented
# default still type-checks, while a non-default value is rejected statically as
# well as at runtime.

# Remedies for kwargs the Tile dispatch path cannot honour. Module constants so
# the guarded call sites stay one line per kwarg.
_TILE_TRANSPOSE_REMEDY = (
    "At tile level a transposed operand is an explicit zero-copy view, not an op flag: "
    "wrap the operand with pl.tile.transpose_view(...) and pass it directly."
)
_TILE_C_MATRIX_NZ_REMEDY = (
    "The tile matmul result layout is fixed by its Acc tile type; there is no "
    "tile-level equivalent of this tensor-level flag."
)
_TILE_RSQRT_PRECISION_REMEDY = (
    "The tile form selects precision by taking a scratch tile: pl.tile.rsqrt(tile, tmp)."
)


def _reject_tmp_for_tensor(op_name: str, tmp: Any, param: str = "tmp") -> None:
    """Guard the Tensor path of an op whose Tile form carries a scratch operand."""
    if tmp is not None:
        raise TypeError(
            f"pl.{op_name}: Tensor inputs must not pass {param} — the scratch tile is "
            f"allocated during Tensor-to-Tile lowering"
        )


def _require_tmp_for_tile(op_name: str, tmp: Tile | None, requirement: str) -> Tile:
    """Guard the Tile path of an op whose Tile form *requires* a scratch operand.

    The mirror image of ``_reject_tmp_for_tensor``: tile buffer lifetimes are
    user-managed, so the operand the Tensor path must omit is the same one the
    Tile path cannot synthesize. Both directions raise ``TypeError`` — this is a
    wrong-arguments-for-this-overload error, the same class CPython raises for a
    missing required argument, not a bad *value*.

    ``requirement`` completes the sentence "Tile inputs require ..." and carries
    the per-op constraint on the scratch operand. Returns the operand so the call
    site keeps the non-``None`` narrowing the inline ``is None`` check gave it.
    """
    if tmp is None:
        raise TypeError(f"pl.{op_name}: Tile inputs require {requirement}")
    return tmp


# Scratch-operand requirements, shared by the ops that impose the same one.
_TMP_ROW_REDUCTION_REQUIREMENT = (
    "tmp_tile with the same dtype and rank as the input, and every dimension at least as large "
    "as the corresponding input dimension"
)
_TMP_ROW_ARG_REDUCTION_REQUIREMENT = "tmp_tile with exactly the same shape and dtype as the input"
_TMP_COL_ARG_REDUCTION_REQUIREMENT = (
    "tmp_tile — the tile form takes caller-owned scratch, unlike pl.col_max / pl.col_min"
)


def _tmp_scratch_requirement(op_name: str) -> str:
    """Requirement text for the bitwise ops, whose scratch operand is positional."""
    return f"an explicit scratch tile — call pl.{op_name}(lhs, rhs, tmp) or pl.tile.{op_name}(lhs, rhs, tmp)"


def _reject_tile_unsupported(op_name: str, /, **flags: tuple[bool, str]) -> None:
    """Guard the Tile path against Tensor-only flags it cannot honour.

    Each entry maps a kwarg name to ``(is_non_default, remedy)``. ``op_name`` is
    positional-only so it cannot collide with a guarded kwarg of the same name.
    """
    for name, (given, remedy) in flags.items():
        if given:
            raise TypeError(f"pl.{op_name}: '{name}' is not supported for Tile operands. {remedy}")


def _check_tile_matmul_out_dtype(result: Tile, out_dtype: int | DataType | None) -> None:
    """Accept a Tile-path ``out_dtype`` only when it matches the deduced dtype.

    ``tile.matmul``'s result dtype is fixed by the Cube accumulator, so the only
    honourable request is the one already satisfied. The deduced dtype is read
    off the built call rather than re-deriving the C++ rule here.

    The Tile ``@overload`` narrows ``out_dtype`` to ``DataType | None``, so a raw
    ``int`` dtype code is already a static error; this still accepts one at
    runtime and rejects it, because the DSL parser reaches this wrapper
    dynamically. ``DataType`` exposes no Python int conversion, so an int cannot
    be verified against the deduction — and skipping verification is the very
    defect this guard exists to prevent.
    """
    if out_dtype is None:
        return
    # ``deduced`` is None only if the built call were not tile/tensor-typed, which
    # tile.matmul never produces — the guard just avoids a nonsense "deduced as
    # None" message if that ever changes.
    deduced = _elem_dtype(result.unwrap())
    if deduced is not None and (not isinstance(out_dtype, DataType) or out_dtype != deduced):
        raise TypeError(
            f"pl.matmul: out_dtype={out_dtype} is not supported for Tile operands — the Cube "
            f"accumulator fixes the result dtype, deduced as {deduced} here. Convert the result "
            f"explicitly with pl.cast(result, <dtype>), or let pl.tile.store narrow it on the "
            f"way to GM."
        )


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

    Caller must have already passed ``_is_scalar_like``. ``Scalar`` is
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
def div(
    lhs: Tensor,
    rhs: Tensor | int | float | Scalar | _ir_core.Expr,
    high_precision: bool = False,
) -> Tensor: ...
@overload
def div(
    lhs: Tile,
    rhs: Tile | int | float | Scalar | _ir_core.Expr,
    high_precision: bool = False,
) -> Tile: ...
@overload
def div(
    lhs: Scalar,
    rhs: Scalar | int | float,
    high_precision: bool = False,
) -> Scalar: ...
def div(lhs, rhs, high_precision: bool = False):
    """Element-wise division, dispatched by input type.

    A scalar ``rhs`` against a Tile dispatches to ``tile.divs``.

    Args:
        high_precision: Select PTOAS's high-precision divide. Available for
            Tensor/Tensor and Tile/Tile only -- a scalar divisor has no
            high-precision form, so passing it there raises rather than silently
            falling back.
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.div(lhs, rhs, high_precision=high_precision)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.div(lhs, rhs, high_precision=high_precision)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        if high_precision:
            raise TypeError("pl.div: high_precision requires a Tile rhs")
        return _tile.divs(lhs, rhs)
    if _is_scalar_like(lhs) and _is_scalar_like(rhs):
        if high_precision:
            raise TypeError("pl.div: high_precision is only supported for Tensor or Tile division")
        return Scalar(expr=_to_scalar_expr(lhs) / _to_scalar_expr(rhs))
    _raise_type_dispatch_error("div", lhs, rhs)


# --- part_add / part_mul / part_max / part_min ---
# Partial-combine binary ops: tensor-tensor or tile-tile only (no scalar form).


@overload
def part_add(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def part_add(lhs: Tile, rhs: Tile) -> Tile: ...
def part_add(lhs, rhs):
    """Partial element-wise add, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.part_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.part_add(lhs, rhs)
    _raise_type_dispatch_error("part_add", lhs, rhs)


@overload
def part_mul(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def part_mul(lhs: Tile, rhs: Tile) -> Tile: ...
def part_mul(lhs, rhs):
    """Partial element-wise multiply, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.part_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.part_mul(lhs, rhs)
    _raise_type_dispatch_error("part_mul", lhs, rhs)


@overload
def part_max(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def part_max(lhs: Tile, rhs: Tile) -> Tile: ...
def part_max(lhs, rhs):
    """Partial element-wise max, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.part_max(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.part_max(lhs, rhs)
    _raise_type_dispatch_error("part_max", lhs, rhs)


@overload
def part_min(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def part_min(lhs: Tile, rhs: Tile) -> Tile: ...
def part_min(lhs, rhs):
    """Partial element-wise min, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.part_min(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.part_min(lhs, rhs)
    _raise_type_dispatch_error("part_min", lhs, rhs)


# --- fmod ---


@overload
def fmod(lhs: Tensor, rhs: Tensor | int | float | Scalar) -> Tensor: ...
@overload
def fmod(lhs: Tile, rhs: Tile | int | float | Scalar) -> Tile: ...
def fmod(lhs, rhs):
    """Element-wise floating-point remainder, dispatched by input type.

    Matches ``torch.fmod`` (the remainder takes the sign of the dividend).
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar, _ir_core.Expr)):
        return _tensor.fmod(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.fmod(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar, _ir_core.Expr)):
        return _tile.fmods(lhs, rhs)
    _raise_type_dispatch_error("fmod", lhs, rhs)


# --- fmods ---


@overload
def fmods(lhs: Tensor, rhs: int | float | Scalar) -> Tensor: ...
@overload
def fmods(lhs: Tile, rhs: int | float | Scalar) -> Tile: ...
def fmods(lhs, rhs):
    """Element-wise floating-point remainder with a scalar, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.fmods(lhs, rhs)
    if isinstance(lhs, Tile):
        return _tile.fmods(lhs, rhs)
    _raise_type_dispatch_error("fmods", lhs, rhs)


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


def log(input: T, high_precision: bool = False) -> T:
    """Element-wise natural logarithm, dispatched by input type.

    Args:
        input: Input tensor or tile.
        high_precision: Select PTOAS's high-precision logarithm mode.
    """
    if isinstance(input, Tensor):
        return _tensor.log(input, high_precision=high_precision)
    if isinstance(input, Tile):
        return _tile.log(input, high_precision=high_precision)
    raise TypeError(f"pl.log: expected Tensor or Tile, got {type(input).__name__}")


def sin(input: T) -> T:
    """Element-wise sine (input in radians), dispatched by input type. FP32 only."""
    if isinstance(input, Tensor):
        return _tensor.sin(input)
    if isinstance(input, Tile):
        return _tile.sin(input)
    raise TypeError(f"pl.sin: expected Tensor or Tile, got {type(input).__name__}")


def cos(input: T) -> T:
    """Element-wise cosine (input in radians), dispatched by input type. FP32 only."""
    if isinstance(input, Tensor):
        return _tensor.cos(input)
    if isinstance(input, Tile):
        return _tile.cos(input)
    raise TypeError(f"pl.cos: expected Tensor or Tile, got {type(input).__name__}")


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


def recip(input: T, high_precision: bool = False) -> T:
    """Element-wise reciprocal (1/x), dispatched by input type.

    Args:
        input: Input tensor or tile
        high_precision: Whether to select PTOAS's high-precision reciprocal mode (FP16/FP32 only)
    """
    if isinstance(input, Tensor):
        return _tensor.recip(input, high_precision=high_precision)
    if isinstance(input, Tile):
        return _tile.recip(input, high_precision=high_precision)
    raise TypeError(f"pl.recip: expected Tensor or Tile, got {type(input).__name__}")


def sqrt(input: T) -> T:
    """Element-wise square root, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor.sqrt(input)
    if isinstance(input, Tile):
        return _tile.sqrt(input)
    raise TypeError(f"pl.sqrt: expected Tensor or Tile, got {type(input).__name__}")


@overload
def rsqrt(input: Tensor, high_precision: bool = ...) -> Tensor: ...
@overload
def rsqrt(input: Tile, high_precision: Literal[False] = ...) -> Tile: ...
def rsqrt(input, high_precision: bool = False):
    """Element-wise reciprocal square root, dispatched by input type.

    ``high_precision`` is Tensor-only: the compiler allocates the scratch tile
    during Tensor-to-Tile lowering. ``tile.rsqrt`` carries no such attribute —
    precision is selected purely by *passing* that scratch tile — so tile
    callers use ``pl.tile.rsqrt(tile, tmp)`` directly and passing
    ``high_precision=True`` with a Tile raises rather than silently yielding the
    low-precision path.
    """
    if isinstance(input, Tensor):
        return _tensor.rsqrt(input, high_precision=high_precision)
    if isinstance(input, Tile):
        _reject_tile_unsupported("rsqrt", high_precision=(high_precision, _TILE_RSQRT_PRECISION_REMEDY))
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


@overload
def row_expand_add(lhs: Tensor, rhs: Tensor) -> Tensor: ...
@overload
def row_expand_add(lhs: Tile, rhs: Tile, tmp: Tile | None = None) -> Tile: ...
def row_expand_add(lhs, rhs, tmp: Tile | None = None):
    """Row-wise broadcast addition; ``tmp`` is available only for Tile inputs."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        _reject_tmp_for_tensor("row_expand_add", tmp)
        return _tensor.row_expand_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_add(lhs, rhs, tmp)
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


def col_expand_add(lhs: T, rhs: T) -> T:
    """Column-wise broadcast addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_add(lhs, rhs)
    _raise_type_dispatch_error("col_expand_add", lhs, rhs)


def row_expand_max(lhs: T, rhs: T) -> T:
    """Row-wise broadcast maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_max(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_max(lhs, rhs)
    _raise_type_dispatch_error("row_expand_max", lhs, rhs)


def row_expand_min(lhs: T, rhs: T) -> T:
    """Row-wise broadcast minimum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_min(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_min(lhs, rhs)
    _raise_type_dispatch_error("row_expand_min", lhs, rhs)


def row_expand_expdif(lhs: T, rhs: T) -> T:
    """Row-wise exp-diff (exp(lhs - rhs) with per-row scalar), dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.row_expand_expdif(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.row_expand_expdif(lhs, rhs)
    _raise_type_dispatch_error("row_expand_expdif", lhs, rhs)


def col_expand_max(lhs: T, rhs: T) -> T:
    """Column-wise broadcast maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_max(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_max(lhs, rhs)
    _raise_type_dispatch_error("col_expand_max", lhs, rhs)


def col_expand_min(lhs: T, rhs: T) -> T:
    """Column-wise broadcast minimum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_min(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_min(lhs, rhs)
    _raise_type_dispatch_error("col_expand_min", lhs, rhs)


def col_expand_expdif(lhs: T, rhs: T) -> T:
    """Column-wise exp-diff (exp(lhs - rhs) with per-column scalar), dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.col_expand_expdif(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.col_expand_expdif(lhs, rhs)
    _raise_type_dispatch_error("col_expand_expdif", lhs, rhs)


def expands(target: Tensor | Tile, scalar: int | float | Scalar) -> Tensor | Tile:
    """Expand scalar to target shape, dispatched by target type.

    Note the argument order: the value being broadcast is the *second* argument.
    ``target`` supplies the shape and dtype; it is not read.

    Args:
        target: Value whose shape the scalar is broadcast to.
        scalar: Value to broadcast into every element.
    """
    if isinstance(target, Tensor):
        return _tensor.expands(target, scalar)
    if isinstance(target, Tile):
        return _tile.expands(target, scalar)
    raise TypeError(f"pl.expands: expected Tensor or Tile, got {type(target).__name__}")


def reshape(input: T, shape: Sequence[IntLike]) -> T:
    """Reshape operation, dispatched by input type.

    A reshape is a zero-copy view, so it never widens the valid region: the
    result holds real data in exactly the cells the input did, re-expressed in
    ``shape``. Because a valid region is an origin-anchored box, not every input
    region survives a repartition — reshaping a region that no box of ``shape``
    can describe is rejected rather than silently rounded up to fully valid.
    Reshapes that only add or drop fully-valid unit axes always work, as does a
    region occupying a contiguous prefix of the buffer.
    """
    if isinstance(input, Tensor):
        return _tensor.reshape(input, shape)
    if isinstance(input, Tile):
        return _tile.reshape(input, shape)
    raise TypeError(f"pl.reshape: expected Tensor or Tile, got {type(input).__name__}")


def reinterpret_view(
    data: T,
    dtype: DataType,
    *,
    shape: Sequence[IntLike] | None = None,
) -> T:
    """Reinterpret the same bytes with a different dtype.

    Args:
        data: Input tensor or tile.
        dtype: Target element dtype, which must differ from the source dtype.
        shape: Optional byte-equivalent target shape. When omitted, the
            physically contiguous dimension is scaled according to the
            source/target dtype byte ratio.

    Returns:
        A zero-copy view of the same kind as ``data``.
    """
    if isinstance(data, Tensor):
        return _tensor.reinterpret_view(data, dtype, shape=shape)
    if isinstance(data, Tile):
        return _tile.reinterpret_view(data, dtype, shape=shape)
    raise TypeError(f"pl.reinterpret_view: expected Tensor or Tile, got {type(data).__name__}")


def transpose(input: T, axis1: int, axis2: int) -> T:
    """Transpose operation, dispatched by input type.

    Args:
        input: Value to transpose.
        axis1: First axis to exchange. Must be a compile-time constant; negative
            indexing is supported.
        axis2: Second axis to exchange. Must differ from ``axis1`` after negative
            indexing is resolved -- naming the same axis twice is rejected, not
            treated as a no-op.
    """
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
    pad_value: PadValue | int | float | None = None,
    clamp: bool = False,
) -> T:
    """Slice operation, dispatched by input type.

    The slice is never valid where the source is not: the source's valid region,
    shifted by ``offset`` and cut to the window, bounds the result.

    ``drop_dims`` lists axes to erase from the result type (numpy-style rank
    reduction); each must be a static unit dim of ``shape`` that is still fully
    valid after that intersection. ``None`` / ``[]`` drops nothing.

    ``pad_value`` sets the padding mode for elements outside the effective valid
    region, on either path. ``None`` carries the source's mode through. Accepts
    ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or the literal
    sugars ``0``, ``math.inf``, ``-math.inf`` (same spelling as [`fillpad`][pypto.language.fillpad]).
    It only bites when the valid region is smaller than ``shape`` — which an
    explicit ``valid_shape``, a partially-valid source, or (Tensor-only)
    ``clamp=True`` can each bring about; passing it otherwise warns.

    ``clamp`` sanctions a window that runs off the end of the source: by default
    the slice asserts ``offset + shape`` stays inside the source and is rejected
    when that provably fails, whereas ``clamp=True`` lets the window overhang and
    cuts the valid region back to the source edge. It is only available on a
    Tensor — an on-chip tile window has nothing that could clamp it.
    """
    if isinstance(input, Tensor):
        return _tensor.slice(input, shape, offset, valid_shape, drop_dims, pad_value, clamp=clamp)
    if isinstance(input, Tile):
        if clamp:
            raise TypeError(
                "pl.slice: clamp=True is not supported for a Tile. An on-chip window has no "
                "clamping mechanism, so offset + shape must stay inside the source tile. "
                "Clamp the read at the tensor boundary instead — pl.load(..., clamp=True) or "
                "pl.slice(tensor, ..., clamp=True) — and slice the resulting tile in bounds."
            )
        return _tile.slice(input, shape, offset, valid_shape, drop_dims, pad_value)
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


def fillpad_expand(
    value: T, shape: Sequence[IntLike], pad_value: PadValue | int | float = PadValue.zero
) -> T:
    """Copy a smaller source into a larger destination, padding the rest.

    Dispatched by input type. The destination ``shape`` may be larger than the
    source in either dimension; the source's valid region is copied into the
    top-left of the destination and every other element is filled with
    ``pad_value`` (``PadValue`` enum or the literal sugars ``0``, ``math.inf``,
    ``-math.inf``).
    """
    if isinstance(value, Tensor):
        return _tensor.fillpad_expand(value, shape, pad_value)
    if isinstance(value, Tile):
        return _tile.fillpad_expand(value, shape, pad_value)
    raise TypeError(f"pl.fillpad_expand: expected Tensor or Tile, got {type(value).__name__}")


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
def matmul(
    lhs: Tile,
    rhs: Tile,
    out_dtype: DataType | None = ...,
    a_trans: Literal[False] = ...,
    b_trans: Literal[False] = ...,
    c_matrix_nz: Literal[False] = ...,
) -> Tile: ...


def matmul(
    lhs: T,
    rhs: T,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> T:
    """Matrix multiplication, dispatched by input type.

    ``a_trans`` / ``b_trans`` / ``c_matrix_nz`` are Tensor-only: a tensor value
    carries no layout, so a flag is the only place the information can live. At
    tile level transposition is a *type* property, so passing any of them with a
    Tile operand raises rather than being dropped.

    A transpose flag swaps its own operand's two trailing axes, so that operand
    must be at least 2D — ``a_trans`` with a 1D ``lhs`` (or ``b_trans`` with a 1D
    ``rhs``) raises rather than being ignored. On the mixed mat-vec / vec-mat
    forms the flag applies to the matrix side: a ``lhs`` stored ``[K, M]`` with
    ``a_trans=True`` against a ``[K]`` ``rhs`` deduces ``[M]``.

    ``out_dtype`` is likewise Tensor-only. ``tile.matmul``'s result dtype is
    fixed by the Cube accumulator (FP32 for float operands, INT32 for int), so
    the Tile path accepts ``out_dtype`` only when it already agrees with that
    deduction and raises otherwise.

    For Tensor inputs with rank > 2 on either operand, the call is lowered to
    ``tile.batch_matmul`` (with batch broadcasting) by ``ConvertTensorToTileOps``
    and then unrolled to per-batch ``tile.matmul`` by ``FlattenTileNdTo2D``.
    Use this entry point (rather than ``pl.batch_matmul``) for tensor-level ND
    matmul.
    """
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.matmul(lhs, rhs, out_dtype, a_trans, b_trans, c_matrix_nz)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        _reject_tile_unsupported(
            "matmul",
            a_trans=(a_trans, _TILE_TRANSPOSE_REMEDY),
            b_trans=(b_trans, _TILE_TRANSPOSE_REMEDY),
            c_matrix_nz=(c_matrix_nz, _TILE_C_MATRIX_NZ_REMEDY),
        )
        result = _tile.matmul(lhs, rhs)
        _check_tile_matmul_out_dtype(result, out_dtype)
        return result
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
    init_cond: BoolLike | None = ...,
) -> Tensor: ...
@overload
def matmul_acc(
    acc: Tile,
    lhs: Tile,
    rhs: Tile,
    a_trans: Literal[False] = ...,
    b_trans: Literal[False] = ...,
    init_cond: BoolLike | None = ...,
) -> Tile: ...


def matmul_acc(
    acc: T,
    lhs: T,
    rhs: T,
    a_trans: bool = False,
    b_trans: bool = False,
    init_cond: BoolLike | None = None,
) -> T:
    """Matrix multiplication with accumulation, dispatched by input type.

    ``a_trans`` / ``b_trans`` are Tensor-only for the same reason as in
    [`matmul`][pypto.language.matmul] — at tile level transposition is a type property, not an op
    flag — so passing either with Tile operands raises rather than being
    dropped.

    ``init_cond`` makes the accumulator's initial value conditional: on the steps
    where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
    accumulated into, which is the split-K ``k == 0`` idiom. It applies to 2D
    operands only.

    For Tensor inputs with rank > 2 on any of acc/lhs/rhs, the call is lowered
    to ``tile.batch_matmul_acc`` (with batch broadcasting on lhs/rhs vs the
    fixed acc batch) by ``ConvertTensorToTileOps`` and then unrolled to
    per-batch ``tile.matmul_acc`` by ``FlattenTileNdTo2D``.
    """
    if isinstance(acc, Tensor) and isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor.matmul_acc(acc, lhs, rhs, a_trans, b_trans, init_cond)
    if isinstance(acc, Tile) and isinstance(lhs, Tile) and isinstance(rhs, Tile):
        _reject_tile_unsupported(
            "matmul_acc",
            a_trans=(a_trans, _TILE_TRANSPOSE_REMEDY),
            b_trans=(b_trans, _TILE_TRANSPOSE_REMEDY),
        )
        return _tile.matmul_acc(acc, lhs, rhs, init_cond)
    _raise_type_dispatch_error("matmul_acc", acc, lhs, rhs)


@overload
def row_max(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_max(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_max(input, tmp_tile: Tile | None = None):
    """Row-wise max reduction, dispatched by input type.

    For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
    rank as the input, with every dimension at least as large as the input dimension.
    Tensor inputs must omit it — the scratch tile is allocated during
    Tensor-to-Tile lowering — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_max", tmp_tile, "tmp_tile")
        return _tensor.row_max(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_max", tmp_tile, _TMP_ROW_REDUCTION_REQUIREMENT)
        return _tile.row_max(input, tmp_tile)
    raise TypeError(f"pl.row_max: expected Tensor or Tile, got {type(input).__name__}")


@overload
def row_sum(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_sum(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_sum(input, tmp_tile: Tile | None = None):
    """Row-wise sum reduction, dispatched by input type.

    For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
    rank as the input, with every dimension at least as large as the input dimension.
    Tensor inputs must omit it — the scratch tile is allocated during
    Tensor-to-Tile lowering — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_sum", tmp_tile, "tmp_tile")
        return _tensor.row_sum(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_sum", tmp_tile, _TMP_ROW_REDUCTION_REQUIREMENT)
        return _tile.row_sum(input, tmp_tile)
    raise TypeError(f"pl.row_sum: expected Tensor or Tile, got {type(input).__name__}")


@overload
def row_min(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_min(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_min(input, tmp_tile: Tile | None = None):
    """Row-wise min reduction, dispatched by input type.

    For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
    rank as the input, with every dimension at least as large as the input dimension.
    Tensor inputs must omit it — the scratch tile is allocated during
    Tensor-to-Tile lowering — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_min", tmp_tile, "tmp_tile")
        return _tensor.row_min(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_min", tmp_tile, _TMP_ROW_REDUCTION_REQUIREMENT)
        return _tile.row_min(input, tmp_tile)
    raise TypeError(f"pl.row_min: expected Tensor or Tile, got {type(input).__name__}")


@overload
def row_prod(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_prod(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_prod(input, tmp_tile: Tile | None = None):
    """Row-wise product reduction, dispatched by input type.

    For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
    rank as the input, with every dimension at least as large as the input dimension.
    Tensor inputs must omit it — the scratch tile is allocated during
    Tensor-to-Tile lowering — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_prod", tmp_tile, "tmp_tile")
        return _tensor.row_prod(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_prod", tmp_tile, _TMP_ROW_REDUCTION_REQUIREMENT)
        return _tile.row_prod(input, tmp_tile)
    raise TypeError(f"pl.row_prod: expected Tensor or Tile, got {type(input).__name__}")


@overload
def col_sum(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def col_sum(input: Tile, tmp_tile: Tile | None = ...) -> Tile: ...
def col_sum(input, tmp_tile: Tile | None = None):
    """Column-wise sum reduction, dispatched by input type.

    For Tile inputs, passing ``tmp_tile`` activates the binary-tree reduction
    path; omitting it uses the sequential path. Tensor inputs must omit it: the
    tensor-to-tile conversion always lowers to the sequential path and allocates
    its own scratch, so a ``tmp_tile`` there could not select the requested
    strategy and raises instead.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("col_sum", tmp_tile, "tmp_tile")
        return _tensor.col_sum(input)
    if isinstance(input, Tile):
        return _tile.col_sum(input, tmp_tile)
    _raise_type_dispatch_error("col_sum", input)


def col_max(input: T) -> T:
    """Column-wise max reduction, dispatched by input type.

    For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_max``.
    """
    if isinstance(input, Tensor):
        return _tensor.col_max(input)
    if isinstance(input, Tile):
        return _tile.col_max(input)
    _raise_type_dispatch_error("col_max", input)


def col_min(input: T) -> T:
    """Column-wise min reduction, dispatched by input type.

    For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_min``.
    """
    if isinstance(input, Tensor):
        return _tensor.col_min(input)
    if isinstance(input, Tile):
        return _tile.col_min(input)
    _raise_type_dispatch_error("col_min", input)


def col_prod(input: T) -> T:
    """Column-wise product reduction, dispatched by input type.

    For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_prod``.
    """
    if isinstance(input, Tensor):
        return _tensor.col_prod(input)
    if isinstance(input, Tile):
        return _tile.col_prod(input)
    _raise_type_dispatch_error("col_prod", input)


@overload
def row_argmax(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_argmax(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_argmax(input, tmp_tile: Tile | None = None):
    """Row-wise argmax (per-row max index, int32), dispatched by input type.

    For Tile inputs, tmp_tile is required with exactly the same shape and dtype.
    Tensor inputs must omit it — the conversion injects the scratch tile — and
    passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_argmax", tmp_tile, "tmp_tile")
        return _tensor.row_argmax(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_argmax", tmp_tile, _TMP_ROW_ARG_REDUCTION_REQUIREMENT)
        return _tile.row_argmax(input, tmp_tile)
    raise TypeError(f"pl.row_argmax: expected Tensor or Tile, got {type(input).__name__}")


@overload
def row_argmin(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def row_argmin(input: Tile, tmp_tile: Tile) -> Tile: ...
def row_argmin(input, tmp_tile: Tile | None = None):
    """Row-wise argmin (per-row min index, int32), dispatched by input type.

    For Tile inputs, tmp_tile is required with exactly the same shape and dtype.
    Tensor inputs must omit it — the conversion injects the scratch tile — and
    passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("row_argmin", tmp_tile, "tmp_tile")
        return _tensor.row_argmin(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("row_argmin", tmp_tile, _TMP_ROW_ARG_REDUCTION_REQUIREMENT)
        return _tile.row_argmin(input, tmp_tile)
    raise TypeError(f"pl.row_argmin: expected Tensor or Tile, got {type(input).__name__}")


@overload
def col_argmax(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def col_argmax(input: Tile, tmp_tile: Tile) -> Tile: ...
def col_argmax(input, tmp_tile: Tile | None = None):
    """Column-wise argmax (per-column max index, int32), dispatched by input type.

    For Tile inputs, tmp_tile is required (unlike col_max). Tensor inputs must
    omit it — the conversion injects the tmp tile — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("col_argmax", tmp_tile, "tmp_tile")
        return _tensor.col_argmax(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("col_argmax", tmp_tile, _TMP_COL_ARG_REDUCTION_REQUIREMENT)
        return _tile.col_argmax(input, tmp_tile)
    raise TypeError(f"pl.col_argmax: expected Tensor or Tile, got {type(input).__name__}")


@overload
def col_argmin(input: Tensor, tmp_tile: None = ...) -> Tensor: ...
@overload
def col_argmin(input: Tile, tmp_tile: Tile) -> Tile: ...
def col_argmin(input, tmp_tile: Tile | None = None):
    """Column-wise argmin (per-column min index, int32), dispatched by input type.

    For Tile inputs, tmp_tile is required (unlike col_min). Tensor inputs must
    omit it — the conversion injects the tmp tile — and passing one raises.
    """
    if isinstance(input, Tensor):
        _reject_tmp_for_tensor("col_argmin", tmp_tile, "tmp_tile")
        return _tensor.col_argmin(input)
    if isinstance(input, Tile):
        tmp_tile = _require_tmp_for_tile("col_argmin", tmp_tile, _TMP_COL_ARG_REDUCTION_REQUIREMENT)
        return _tile.col_argmin(input, tmp_tile)
    raise TypeError(f"pl.col_argmin: expected Tensor or Tile, got {type(input).__name__}")


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
    """Type casting, dispatched by input type.

    Args:
        input: Value to convert.
        target_type: Destination dtype.
        mode: Rounding mode, as a name or its int code -- ``"none"`` (0),
            ``"rint"`` (1), ``"round"`` (2, the default), ``"floor"`` (3),
            ``"ceil"`` (4), ``"trunc"`` (5), ``"odd"`` (6). A ``Scalar`` input
            supports the default only and raises for any other mode.
    """
    if isinstance(input, Tensor):
        return _tensor.cast(input, target_type, mode)
    if isinstance(input, Tile):
        return _tile.cast(input, target_type, mode)
    if _is_scalar_like(input):
        # ``resolve_cast_mode`` runs first, so an invalid mode is still a ValueError;
        # only a *valid* mode this path cannot honour reaches the TypeError below.
        if resolve_cast_mode(mode) != 2:
            raise TypeError(f"pl.cast: Scalar inputs do not support non-default mode, got mode={mode!r}")
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


# ---------------------------------------------------------------------------
# Bitwise / shift ops
#
# The tile forms of xor/xors take an explicit ``tmp`` scratch tile because buffer
# lifetimes are user-managed there; the tensor forms do not, since the conversion
# pass allocates it. That Tile-only trailing operand follows ``row_expand_add`` and
# the ``row_*`` / ``col_*`` reduction family above, which take the same
# ``tmp_tile: Tile | None = None`` and reject it on the Tensor path via the shared
# ``_reject_tmp_for_tensor`` guard.
# ---------------------------------------------------------------------------


@overload
def and_(lhs: Tensor, rhs: Tensor | int | Scalar) -> Tensor: ...
@overload
def and_(lhs: Tile, rhs: Tile | int | Scalar) -> Tile: ...
def and_(lhs, rhs):
    """Element-wise bitwise AND, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, Scalar, _ir_core.Expr)):
        return _tensor.and_(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.and_(lhs, rhs)
    if isinstance(lhs, Tile) and _is_scalar_like(rhs):
        return _tile.ands(lhs, rhs)
    _raise_type_dispatch_error("and_", lhs, rhs)


@overload
def ands(lhs: Tensor, rhs: int | Scalar) -> Tensor: ...
@overload
def ands(lhs: Tile, rhs: int | Scalar) -> Tile: ...
def ands(lhs, rhs):
    """Element-wise bitwise AND with a scalar, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.ands(lhs, rhs)
    if isinstance(lhs, Tile):
        return _tile.ands(lhs, rhs)
    _raise_type_dispatch_error("ands", lhs, rhs)


@overload
def or_(lhs: Tensor, rhs: Tensor | int | Scalar) -> Tensor: ...
@overload
def or_(lhs: Tile, rhs: Tile | int | Scalar) -> Tile: ...
def or_(lhs, rhs):
    """Element-wise bitwise OR, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, Scalar, _ir_core.Expr)):
        return _tensor.or_(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.or_(lhs, rhs)
    if isinstance(lhs, Tile) and _is_scalar_like(rhs):
        return _tile.ors(lhs, rhs)
    _raise_type_dispatch_error("or_", lhs, rhs)


@overload
def ors(lhs: Tensor, rhs: int | Scalar) -> Tensor: ...
@overload
def ors(lhs: Tile, rhs: int | Scalar) -> Tile: ...
def ors(lhs, rhs):
    """Element-wise bitwise OR with a scalar, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.ors(lhs, rhs)
    if isinstance(lhs, Tile):
        return _tile.ors(lhs, rhs)
    _raise_type_dispatch_error("ors", lhs, rhs)


@overload
def xor(lhs: Tensor, rhs: Tensor | int | Scalar) -> Tensor: ...
@overload
def xor(lhs: Tile, rhs: Tile | int | Scalar, tmp: Tile) -> Tile: ...
def xor(lhs, rhs, tmp=None):
    """Element-wise bitwise XOR, dispatched by input type.

    ``pto.txor`` needs a scratch buffer. Tile buffer lifetimes are user-managed,
    so the tile path takes it as ``tmp``; the tensor path omits it because
    ConvertTensorToTileOps allocates it — the same asymmetry ``rsqrt`` carries.
    """
    if isinstance(lhs, Tensor):
        _reject_tmp_for_tensor("xor", tmp)
        return _tensor.xor(lhs, rhs)
    if isinstance(lhs, Tile):
        tmp = _require_tmp_for_tile("xor", tmp, _tmp_scratch_requirement("xor"))
        if isinstance(rhs, Tile):
            return _tile.xor(lhs, rhs, tmp)
        if _is_scalar_like(rhs):
            return _tile.xors(lhs, rhs, tmp)
    _raise_type_dispatch_error("xor", lhs, rhs)


@overload
def xors(lhs: Tensor, rhs: int | Scalar) -> Tensor: ...
@overload
def xors(lhs: Tile, rhs: int | Scalar, tmp: Tile) -> Tile: ...
def xors(lhs, rhs, tmp=None):
    """Element-wise bitwise XOR with a scalar, dispatched by input type.

    See [`xor`][pypto.language.xor] for why only the tile path takes ``tmp``.
    """
    if isinstance(lhs, Tensor):
        _reject_tmp_for_tensor("xors", tmp)
        return _tensor.xors(lhs, rhs)
    if isinstance(lhs, Tile):
        tmp = _require_tmp_for_tile("xors", tmp, _tmp_scratch_requirement("xors"))
        return _tile.xors(lhs, rhs, tmp)
    _raise_type_dispatch_error("xors", lhs, rhs)


def not_(input: T) -> T:
    """Element-wise bitwise NOT, dispatched by input type (int16/uint16 only)."""
    if isinstance(input, Tensor):
        return _tensor.not_(input)
    if isinstance(input, Tile):
        return _tile.not_(input)
    raise TypeError(f"pl.not_: expected Tensor or Tile, got {type(input).__name__}")


@overload
def shl(lhs: Tensor, rhs: Tensor | int | Scalar) -> Tensor: ...
@overload
def shl(lhs: Tile, rhs: Tile | int | Scalar) -> Tile: ...
def shl(lhs, rhs):
    """Element-wise bitwise left shift, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, Scalar, _ir_core.Expr)):
        return _tensor.shl(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.shl(lhs, rhs)
    if isinstance(lhs, Tile) and _is_scalar_like(rhs):
        return _tile.shls(lhs, rhs)
    _raise_type_dispatch_error("shl", lhs, rhs)


@overload
def shls(lhs: Tensor, rhs: int | Scalar) -> Tensor: ...
@overload
def shls(lhs: Tile, rhs: int | Scalar) -> Tile: ...
def shls(lhs, rhs):
    """Element-wise bitwise left shift by a scalar, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.shls(lhs, rhs)
    if isinstance(lhs, Tile):
        return _tile.shls(lhs, rhs)
    _raise_type_dispatch_error("shls", lhs, rhs)


@overload
def shr(lhs: Tensor, rhs: Tensor | int | Scalar) -> Tensor: ...
@overload
def shr(lhs: Tile, rhs: Tile | int | Scalar) -> Tile: ...
def shr(lhs, rhs):
    """Element-wise bitwise right shift, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, Scalar, _ir_core.Expr)):
        return _tensor.shr(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _tile.shr(lhs, rhs)
    if isinstance(lhs, Tile) and _is_scalar_like(rhs):
        return _tile.shrs(lhs, rhs)
    _raise_type_dispatch_error("shr", lhs, rhs)


@overload
def shrs(lhs: Tensor, rhs: int | Scalar) -> Tensor: ...
@overload
def shrs(lhs: Tile, rhs: int | Scalar) -> Tile: ...
def shrs(lhs, rhs):
    """Element-wise bitwise right shift by a scalar, dispatched by input type."""
    if isinstance(lhs, Tensor):
        return _tensor.shrs(lhs, rhs)
    if isinstance(lhs, Tile):
        return _tile.shrs(lhs, rhs)
    _raise_type_dispatch_error("shrs", lhs, rhs)


@overload
def set_validshape(input: Tensor, valid_rows: IntLike, valid_cols: IntLike) -> Tensor: ...
@overload
def set_validshape(input: Tile, valid_rows: IntLike, valid_cols: IntLike) -> Tile: ...
def set_validshape(input, valid_rows, valid_cols):
    """Update valid-shape metadata without data movement, dispatched by input type.

    .. note::
        Prefer expressing the extent at its source where possible —
        ``pl.load(..., valid_shape=...)`` or a slice's ``valid_shape=``. A tile
        view (slice / reshape result) is rejected: it carries its valid extent in
        its type, so there are no runtime operands to update.
    """
    if isinstance(input, Tensor):
        return _tensor.set_validshape(input, valid_rows, valid_cols)
    if isinstance(input, Tile):
        return _tile.set_validshape(input, valid_rows, valid_cols)
    raise TypeError(f"pl.set_validshape: expected Tensor or Tile, got {type(input).__name__}")


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


# ---------------------------------------------------------------------------
# Sub-region write / gather / scatter / sort with type dispatch
# ---------------------------------------------------------------------------

# Remedy for the Tensor-only ``atomic`` combine mode on ``assemble``.
_TILE_ATOMIC_REMEDY = (
    "An atomic combine needs a global-memory destination, and a tile-to-tile assemble has "
    "none. Write the accumulation into a Tensor target instead — inside an InCore function "
    "that is pl.assemble(out_tensor, tile, offset, atomic=...)."
)


@overload
def assemble(
    target: Tensor, source: Tensor, offset: Sequence[IntLike], *, atomic: AtomicType = ...
) -> Tensor: ...
@overload
def assemble(
    target: Tile, source: Tile, offset: Sequence[IntLike], *, atomic: Literal[AtomicType.None_] = ...
) -> Tile: ...
def assemble(target, source, offset, *, atomic: AtomicType = AtomicType.None_):
    """Write ``source`` into ``target`` at ``offset``, dispatched by target type.

    ``atomic`` is Tensor-only: the combine lowers to an atomic-add store into
    global memory, which a tile-to-tile assemble has no destination for. Passing
    the documented default keeps working on both paths; any other value with a
    Tile target raises.
    """
    if isinstance(target, Tensor) and isinstance(source, Tensor):
        return _tensor.assemble(target, source, offset, atomic=atomic)
    if isinstance(target, Tile) and isinstance(source, Tile):
        _reject_tile_unsupported("assemble", atomic=(atomic != AtomicType.None_, _TILE_ATOMIC_REMEDY))
        return _tile.assemble(target, source, offset)
    _raise_type_dispatch_error("assemble", target, source)


def gather_row(  # noqa: PLR0913
    dst: T,
    src: Tensor,
    dst_offset: Sequence[IntLike],
    src_offset: Sequence[IntLike],
    shapes: Sequence[IntLike],
    transpose: bool = False,
    *,
    valid_shape: Sequence[IntLike] | None = None,
) -> T:
    """Gather one GM row into a sub-region of an on-chip accumulator (DPS).

    Dispatched on ``dst`` — the destination accumulator. ``src`` is a ``Tensor``
    on both paths: this op always reads from global memory, so it is the
    destination, not the source, that names the level.
    """
    args = (src, dst_offset, src_offset, shapes, transpose)
    if isinstance(dst, Tensor):
        return _tensor.gather_row(dst, *args, valid_shape=valid_shape)
    if isinstance(dst, Tile):
        return _tile.gather_row(dst, *args, valid_shape=valid_shape)
    raise TypeError(f"pl.gather_row: expected Tensor or Tile destination, got {type(dst).__name__}")


def scatter_update(input: T, *args: Any, **kwargs: Any) -> T:
    """Update rows at positions given by a 2D index, dispatched by input type.

    Accepts the same flexible call shapes as either level's wrapper — the
    positional/keyword forms are identical on both, so the arguments are
    forwarded unchanged.
    """
    if isinstance(input, Tensor):
        return _tensor.scatter_update(input, *args, **kwargs)
    if isinstance(input, Tile):
        return _tile.scatter_update(input, *args, **kwargs)
    raise TypeError(f"pl.scatter_update: expected Tensor or Tile, got {type(input).__name__}")


def sort32(src: T, idx: T) -> T:
    """Sort fixed 32-element blocks, permuting ``idx`` alongside ``src``.

    Dispatched by input type. Returns 8-byte value-index pairs; the last
    dimension is 2x the input width for FP32 and 4x for FP16.
    """
    if isinstance(src, Tensor) and isinstance(idx, Tensor):
        return _tensor.sort32(src, idx)
    if isinstance(src, Tile) and isinstance(idx, Tile):
        return _tile.sort32(src, idx)
    _raise_type_dispatch_error("sort32", src, idx)


def mrgsort(  # noqa: PLR0913
    src0: T,
    src1: T | None = None,
    src2: T | None = None,
    src3: T | None = None,
    tmp: T | None = None,
    *,
    exhausted: bool = False,
    block_len: int | Scalar | None = None,
) -> T:
    """Merge sort — format1 (single-list) or format2 (2-4 way), dispatched by input type.

    ``tmp`` is Tile-only: at tensor level the scratch buffer is synthesized
    during Tensor-to-Tile lowering, so passing one is rejected rather than
    silently dropped. The tile path's format2 requires it, and ``pl.tile.mrgsort``
    raises with the per-format guidance when it is missing.

    ``exhausted`` is keyword-only here. The tile wrapper also accepts it as a
    sixth positional argument; that spelling stays available as
    ``pl.tile.mrgsort(...)``.
    """
    if isinstance(src0, Tensor):
        _reject_tmp_for_tensor("mrgsort", tmp)
        return _tensor.mrgsort(src0, src1, src2, src3, exhausted=exhausted, block_len=block_len)
    if isinstance(src0, Tile):
        return _tile.mrgsort(src0, src1, src2, src3, tmp, exhausted, block_len=block_len)
    raise TypeError(f"pl.mrgsort: expected Tensor or Tile, got {type(src0).__name__}")
