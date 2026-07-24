# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Utility functions for IR construction."""

import inspect
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

# Span pinned by the DSL parser while invoking a wrapper. When set, IR
# builders that fall back via ``_get_span_or_capture`` use this in preference
# to frame-capture, so nodes constructed inside DSL wrappers carry the
# call-site span rather than the wrapper file's own line.
_PARSER_SPAN: ContextVar[_ir.Span | None] = ContextVar("_PARSER_SPAN", default=None)


@contextmanager
def use_parser_span(span: _ir.Span) -> Iterator[None]:
    """Temporarily pin the parser span seen by ``_get_span_or_capture``."""
    token = _PARSER_SPAN.set(span)
    try:
        yield
    finally:
        _PARSER_SPAN.reset(token)


def _get_span_or_capture(span: _ir.Span | None = None, frame_offset: int = 1) -> _ir.Span:
    """Get explicit span, parser-pinned span, or captured frame span.

    Resolution order:
      1. Explicit ``span`` argument when provided.
      2. ``_PARSER_SPAN`` contextvar (set by the DSL parser).
      3. Frame capture from ``frame_offset`` levels up the Python stack.

    Args:
        span: Explicit span if provided
        frame_offset: Additional frames to skip beyond immediate caller

    Returns:
        Provided span, parser-pinned span, or captured span from call site
    """
    if span is not None:
        return span

    parser_span = _PARSER_SPAN.get()
    if parser_span is not None:
        return parser_span

    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back

    for _ in range(frame_offset):
        if frame is None:
            break
        frame = frame.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, -1)

    return _ir.Span.unknown()


def _normalize_expr(
    value: int | float | _ir.Expr,
    span: _ir.Span | None = None,
    int_dtype: DataType = DataType.INDEX,
    float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Convert Python values to IR expressions.

    Args:
        value: Python int/float or existing Expr
        span: Optional span for created constants
        int_dtype: Data type to use for integer constants (default: INDEX)
        float_dtype: Data type to use for float constants (default: FP32)

    Returns:
        IR expression node

    Raises:
        TypeError: If value is not int, float, or ir.Expr
    """
    if isinstance(value, _ir.Expr):
        return value

    actual_span = span if span is not None else _ir.Span.unknown()

    if isinstance(value, int):
        return _ir.ConstInt(value, int_dtype, actual_span)
    elif isinstance(value, float):
        return _ir.ConstFloat(value, float_dtype, actual_span)
    else:
        raise TypeError(f"Cannot convert {type(value)} to IR expression")


def _normalize_shape(
    shape: Sequence[int | _ir.Expr],
    span: _ir.Span | None = None,
) -> list[_ir.Expr]:
    """Convert shape dimensions to IR expressions.

    Args:
        shape: Sequence of integers or Expr nodes representing shape dimensions
        span: Optional span for created constants

    Returns:
        List of IR expression nodes

    Raises:
        TypeError: If shape contains non-int, non-Expr values
    """
    return [_normalize_expr(dim, span, int_dtype=DataType.INDEX) for dim in shape]


def _to_make_tuple(
    value: _ir.MakeTuple | Sequence[int | float | _ir.Expr],
    span: _ir.Span | None = None,
) -> _ir.MakeTuple:
    """Normalize a sequence or MakeTuple into a MakeTuple IR node.

    Args:
        value: Either an existing MakeTuple (returned as-is) or a sequence
            of ints/floats/Exprs to wrap
        span: Optional span for created constants

    Returns:
        MakeTuple IR expression
    """
    if isinstance(value, _ir.MakeTuple):
        return value
    actual_span = span if span is not None else _ir.Span.unknown()
    elements = [_normalize_expr(v, actual_span) for v in value]
    return _ir.MakeTuple(elements, actual_span)


CAST_MODE_NAMES: dict[str, int] = {
    "none": 0,
    "rint": 1,
    "round": 2,
    "floor": 3,
    "ceil": 4,
    "trunc": 5,
    "odd": 6,
}


def resolve_cast_mode(mode: str | int) -> int:
    """Resolve cast mode to int, accepting both string names and int values.

    Args:
        mode: String name ("none", "rint", "round", "floor", "ceil", "trunc",
              "odd") or int (0-6)

    Returns:
        Integer mode value

    Raises:
        ValueError: If mode is not a valid name or is out of range [0, 6]
    """
    if isinstance(mode, bool):
        raise ValueError(f"Invalid rounding mode {mode!r}. Expected str name or int in range [0, 6].")
    if isinstance(mode, int):
        max_mode = max(CAST_MODE_NAMES.values())
        if not 0 <= mode <= max_mode:
            raise ValueError(f"Invalid rounding mode {mode}. Expected int in range [0, {max_mode}].")
        return mode
    mode_val = CAST_MODE_NAMES.get(mode)
    if mode_val is None:
        raise ValueError(f"Invalid rounding mode '{mode}'. Expected one of {list(CAST_MODE_NAMES.keys())}.")
    return mode_val


def has_partial_valid_region(expr: _ir.Expr) -> bool:
    """Whether a tensor/tile value already declares less valid data than it can hold.

    An explicit ``valid_shape`` survives type canonicalization only when it really
    differs from the physical shape, so a non-empty one means the value carries
    padding that a reader has to respect.

    Args:
        expr: A tensor- or tile-typed expression

    Returns:
        True when the expression's view narrows it below its physical shape
    """
    expr_type = expr.type
    view = getattr(expr_type, "tensor_view", None)
    if view is None:
        view = getattr(expr_type, "tile_view", None)
    return view is not None and bool(view.valid_shape)


def _to_int32_scalar(value: int | _ir.Expr, span: _ir.Span) -> _ir.Expr:
    """Normalize a seed value to an INT32 scalar expression.

    Shared by the counter-based ``random`` ops (tensor and tile), which coerce
    every key/counter word to an INT32 scalar before building the call.
    """
    if isinstance(value, _ir.Expr):
        if isinstance(value, _ir.ConstInt) and value.dtype != DataType.INT32:
            return _ir.ConstInt(value.value, DataType.INT32, span)
        return value
    return _ir.ConstInt(value, DataType.INT32, span)


def _elem_dtype(operand: _ir.Expr) -> DataType | None:
    """Element dtype of a tile/tensor operand, or None if not statically known."""
    operand_type = operand.type
    if isinstance(operand_type, (_ir.TileType, _ir.TensorType, _ir.DistributedTensorType)):
        return operand_type.dtype
    return None


def _check_not_index_scalar(scalar: _ir.Expr, target: DataType | None) -> None:
    """Reject a non-constant ``index`` scalar operand with an actionable hint.

    ``index`` is never a legal operand type for a ``pto.t*s`` instruction. A
    *constant* carrying it is merely the parser's placeholder and is re-typed
    silently, but a *value* -- a loop variable, ``pl.dim(...)``, an offset -- needs
    a real conversion whose target only the user can choose.

    Raises:
        ValueError: If ``scalar`` is an ``index``-typed scalar expression.
    """
    scalar_type = scalar.type
    if not isinstance(scalar_type, _ir.ScalarType) or scalar_type.dtype != DataType.INDEX:
        return

    # Codegen lowers index<->int via arith.index_cast but rejects index<->float
    # outright (pto_scalar_expr_codegen.cpp), so a float operand needs two steps.
    if target is not None and target.is_float():
        hint = f"pl.cast(pl.cast(<value>, pl.INT32), pl.{str(target).upper()})"
        why = "; `index` cannot convert to a float dtype in one step"
    else:
        name = str(target).upper() if target is not None else "INT32"
        hint = f"pl.cast(<value>, pl.{name})"
        why = ""
    raise ValueError(
        f"Scalar operand has dtype `index`, which tile/tensor scalar instructions do "
        f"not accept. Convert it explicitly: {hint}{why}."
    )


def _const_at_dtype(value: int | float, dtype: DataType, span: _ir.Span) -> _ir.Expr:
    """Build a scalar constant carrying exactly ``dtype``.

    The node kind follows ``dtype``, not the Python type of ``value``: codegen
    dispatches on ConstInt vs ConstFloat, so an int paired with a float dtype
    must become a ConstFloat or MLIR receives ``arith.constant 5 : f32``.
    """
    if dtype.is_float():
        return _ir.ConstFloat(float(value), dtype, span)
    return _ir.ConstInt(int(value), dtype, span)


def _placeholder_value(scalar: int | float | _ir.Expr, hint_dtype: DataType | None) -> int | float | None:
    """Numeric value of a re-typable scalar placeholder, or ``None`` to pass through.

    A scalar is a re-typable placeholder when it is a raw Python literal or the
    parser's ``ConstInt(v, INDEX)`` -- these carry no deliberate dtype, so a caller
    may stamp one on. Any other expression already declares its dtype and is
    signalled with ``None`` so the caller keeps it unchanged; the sole exception is
    a non-constant ``index`` value, rejected here via ``_check_not_index_scalar``
    (``hint_dtype`` shapes the ``pl.cast`` hint).

    Returns:
        The literal/placeholder value to re-stamp, or ``None`` for an already-typed
        expression that must be left as-is.
    """
    if not isinstance(scalar, _ir.Expr):
        return scalar  # raw Python literal
    if isinstance(scalar, _ir.ConstInt) and scalar.dtype == DataType.INDEX:
        return scalar.value  # parser's INDEX placeholder
    _check_not_index_scalar(scalar, hint_dtype)
    return None  # already-typed expr -- leave untouched


def _normalize_scalar_operand(
    operand: _ir.Expr,
    scalar: int | float | _ir.Expr,
    span: _ir.Span,
    *,
    fallback_int_dtype: DataType = DataType.INT32,
    fallback_float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Normalize an untyped scalar constant to the paired tile/tensor element dtype.

    The DSL parser turns every bare int literal into ``ConstInt(v, INDEX)``
    (``ast_parser.parse_constant``). ``index`` is not a legal operand type for any
    ``pto.t*s`` instruction, and at the tensor level it also propagates through
    ``PromoteDataTypes`` into the *result* tensor dtype. ``INDEX`` is therefore
    treated as "dtype not yet decided" and re-stamped to the ``operand`` element
    dtype, alongside raw Python literals which carry no dtype at all.

    Any constant that already carries a real dtype is left untouched -- an explicit
    ``pl.const(42, pl.INT32)`` is a deliberate user annotation, not a placeholder.
    A float literal paired with an integer operand keeps ``fallback_float_dtype``
    so existing promotion semantics (``int32_tensor * 2.5 -> fp32``) are preserved.

    Args:
        operand: The tile/tensor the scalar is paired with.
        scalar: Python int/float, or an existing IR expression.
        span: Span for any constant created here.
        fallback_int_dtype: Int dtype used when ``operand`` is not statically typed.
        fallback_float_dtype: Float dtype used when ``operand`` is not statically typed.

    Returns:
        An expression whose dtype matches the operand element dtype where the rule
        above applies; otherwise ``scalar`` unchanged.

    Raises:
        ValueError: If ``scalar`` is a non-constant ``index`` value (see
            ``_check_not_index_scalar``) -- convert it with ``pl.cast``.
    """
    target = _elem_dtype(operand)
    value = _placeholder_value(scalar, target)
    if value is None:
        assert isinstance(scalar, _ir.Expr)  # _placeholder_value returns None only for exprs
        return scalar  # already-typed expr, kept as-is

    # Unknown operand type, or a float constant on an integer operand: fall back
    # to the literal-kind default so promotion behaviour is unchanged.
    if target is None or (isinstance(value, float) and target.is_int()):
        target = fallback_float_dtype if isinstance(value, float) else fallback_int_dtype

    if target.is_float() or target.is_int():
        return _const_at_dtype(value, target, span)

    # Neither int nor float (e.g. BOOL): keep prior behaviour.
    return _normalize_expr(scalar, span, int_dtype=fallback_int_dtype, float_dtype=fallback_float_dtype)


def _normalize_const_to_dtype(
    scalar: int | float | _ir.Expr,
    dtype: DataType,
    span: _ir.Span,
    *,
    float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Re-stamp an untyped scalar constant whose dtype is fixed by the operator.

    For operands that do *not* follow a tile/tensor element dtype -- a mode flag
    or float coefficient. Shares the placeholder rule via ``_placeholder_value``
    (only the parser's ``INDEX`` constants and raw Python literals are re-typed),
    and a float literal paired with an integer ``dtype`` keeps ``float_dtype``.

    Raises:
        ValueError: If ``scalar`` is a non-constant ``index`` value (see
            ``_check_not_index_scalar``) -- convert it with ``pl.cast``.
    """
    value = _placeholder_value(scalar, dtype)
    if value is None:
        assert isinstance(scalar, _ir.Expr)  # _placeholder_value returns None only for exprs
        return scalar  # already-typed expr, kept as-is

    target = float_dtype if (isinstance(value, float) and dtype.is_int()) else dtype
    return _const_at_dtype(value, target, span)


__all__ = [
    "CAST_MODE_NAMES",
    "_get_span_or_capture",
    "_normalize_const_to_dtype",
    "_normalize_expr",
    "_normalize_scalar_operand",
    "_normalize_shape",
    "_to_int32_scalar",
    "_to_make_tuple",
    "has_partial_valid_region",
    "resolve_cast_mode",
    "use_parser_span",
]
