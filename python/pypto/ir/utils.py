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


__all__ = [
    "CAST_MODE_NAMES",
    "_get_span_or_capture",
    "_normalize_expr",
    "_normalize_shape",
    "_to_make_tuple",
    "resolve_cast_mode",
    "use_parser_span",
]
