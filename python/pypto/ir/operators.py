# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Operator overloading with automatic span capture and expression normalization."""

import inspect

from pypto.pypto_core import ir as _ir

from .utils import _normalize_expr


def _capture_call_span() -> _ir.Span:
    """Capture span from the caller's location.

    Goes back through the call stack to find the actual user code that invoked the operator.

    Returns:
        Span: Source location of the caller
    """
    # Go back through frames to find user code:
    # frame 0 = _capture_call_span
    # frame 1 = our wrapper (e.g., __add__)
    # frame 2 = user's code (what we want)
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        frame = frame.f_back.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, 0)

    return _ir.Span.unknown()


def _make_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self, other):
        span = _capture_call_span()
        other_expr = _normalize_expr(other, span)
        return getattr(_ir, op_name)(self, other_expr, span)

    return wrapper


def _make_reverse_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self, other):
        span = _capture_call_span()
        other_expr = _normalize_expr(other, span)
        return getattr(_ir, op_name)(other_expr, self, span)

    return wrapper


def _make_unary_op(op_name: str):
    """Create a unary operator wrapper with span capture.

    Args:
        op_name: Name of the operator (e.g., "neg")

    Returns:
        Wrapper function
    """

    def wrapper(self) -> _ir.Expr:
        span = _capture_call_span()
        return getattr(_ir, op_name)(self, span)

    return wrapper


def _patch_operators():
    """Patch Expr and Var classes with operator overloads."""

    # Binary operators for Expr
    _ir.Expr.__add__ = _make_binary_op("add")
    _ir.Expr.__sub__ = _make_binary_op("sub")
    _ir.Expr.__mul__ = _make_binary_op("mul")
    _ir.Expr.__truediv__ = _make_binary_op("truediv")
    _ir.Expr.__floordiv__ = _make_binary_op("floordiv")
    _ir.Expr.__mod__ = _make_binary_op("mod")
    _ir.Expr.__pow__ = _make_binary_op("pow")

    # Comparison operators for Expr
    _ir.Expr.__eq__ = _make_binary_op("eq")
    _ir.Expr.__ne__ = _make_binary_op("ne")
    _ir.Expr.__lt__ = _make_binary_op("lt")
    _ir.Expr.__le__ = _make_binary_op("le")
    _ir.Expr.__gt__ = _make_binary_op("gt")
    _ir.Expr.__ge__ = _make_binary_op("ge")

    # Unary operators for Expr
    _ir.Expr.__neg__ = _make_unary_op("neg")

    # Reverse operators for Expr (when Expr is on right side)
    _ir.Expr.__radd__ = _make_reverse_binary_op("add")
    _ir.Expr.__rmul__ = _make_reverse_binary_op("mul")
    _ir.Expr.__rsub__ = _make_reverse_binary_op("sub")
    _ir.Expr.__rtruediv__ = _make_reverse_binary_op("truediv")
    _ir.Expr.__rfloordiv__ = _make_reverse_binary_op("floordiv")
    _ir.Expr.__rmod__ = _make_reverse_binary_op("mod")
    _ir.Expr.__rpow__ = _make_reverse_binary_op("pow")

    # Binary operators for Var
    _ir.Var.__add__ = _make_binary_op("add")
    _ir.Var.__sub__ = _make_binary_op("sub")
    _ir.Var.__mul__ = _make_binary_op("mul")
    _ir.Var.__truediv__ = _make_binary_op("truediv")
    _ir.Var.__floordiv__ = _make_binary_op("floordiv")
    _ir.Var.__mod__ = _make_binary_op("mod")
    _ir.Var.__pow__ = _make_binary_op("pow")

    # Comparison operators for Var
    _ir.Var.__eq__ = _make_binary_op("eq")
    _ir.Var.__ne__ = _make_binary_op("ne")
    _ir.Var.__lt__ = _make_binary_op("lt")
    _ir.Var.__le__ = _make_binary_op("le")
    _ir.Var.__gt__ = _make_binary_op("gt")
    _ir.Var.__ge__ = _make_binary_op("ge")

    # Unary operators for Var
    _ir.Var.__neg__ = _make_unary_op("neg")

    # Reverse operators for Var
    _ir.Var.__radd__ = _make_reverse_binary_op("add")
    _ir.Var.__rmul__ = _make_reverse_binary_op("mul")
    _ir.Var.__rsub__ = _make_reverse_binary_op("sub")
    _ir.Var.__rtruediv__ = _make_reverse_binary_op("truediv")
    _ir.Var.__rfloordiv__ = _make_reverse_binary_op("floordiv")
    _ir.Var.__rmod__ = _make_reverse_binary_op("mod")
    _ir.Var.__rpow__ = _make_reverse_binary_op("pow")


# Automatically patch operators when this module is imported
_patch_operators()


__all__ = []
