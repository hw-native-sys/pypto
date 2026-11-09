# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Evaluate Python AST expressions over known frontend constants."""

import ast
import operator
from collections.abc import Callable, Sequence
from typing import Any, Final, cast


class _Unknown:
    """The expression needs a runtime value or uses unsupported syntax."""


UNKNOWN: Final = _Unknown()

_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}
_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_SCALARS = (type(None), bool, int, float, str)


def _is_constant(value: object) -> bool:
    if type(value) in _SCALARS:
        return True
    if type(value) in (tuple, list):
        return all(_is_constant(item) for item in cast(Sequence[object], value))
    return False


def evaluate(node: ast.expr, lookup: Callable[[str], object] | None = None) -> Any:
    """Return a known primitive value (including None), or UNKNOWN.

    Only built-in values and explicit AST operators are evaluated. Calls,
    attributes and user-defined operator overloads are outside this contract.
    """

    def visit(expr: ast.expr) -> Any:
        if isinstance(expr, ast.Constant):
            return expr.value if _is_constant(expr.value) else UNKNOWN
        if isinstance(expr, ast.Name) and lookup is not None:
            value = lookup(expr.id)
            return value if _is_constant(value) else UNKNOWN
        if isinstance(expr, (ast.List, ast.Tuple)):
            items = [visit(item) for item in expr.elts]
            if any(item is UNKNOWN for item in items):
                return UNKNOWN
            return items if isinstance(expr, ast.List) else tuple(items)
        if isinstance(expr, ast.UnaryOp) and (unary := _UNARY.get(type(expr.op))) is not None:
            value = visit(expr.operand)
            return UNKNOWN if value is UNKNOWN else unary(value)
        if isinstance(expr, ast.BinOp) and (binary := _BINARY.get(type(expr.op))) is not None:
            left, right = visit(expr.left), visit(expr.right)
            return UNKNOWN if left is UNKNOWN or right is UNKNOWN else binary(left, right)
        if isinstance(expr, ast.BoolOp):
            for item in expr.values[:-1]:
                value = visit(item)
                if value is UNKNOWN or bool(value) == isinstance(expr.op, ast.Or):
                    return value
            return visit(expr.values[-1])
        if isinstance(expr, ast.Compare):
            left = visit(expr.left)
            for op, rhs in zip(expr.ops, expr.comparators, strict=True):
                compare = _COMPARE.get(type(op))
                if left is UNKNOWN or compare is None:
                    return UNKNOWN
                right = visit(rhs)
                if right is UNKNOWN:
                    return UNKNOWN
                if not compare(left, right):
                    return False
                left = right
            return True
        return UNKNOWN

    try:
        value = visit(node)
        return value if _is_constant(value) else UNKNOWN
    except (ArithmeticError, TypeError, ValueError, RecursionError):
        return UNKNOWN
