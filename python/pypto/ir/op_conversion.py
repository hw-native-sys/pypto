# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Op conversion utilities for tensor-to-block op mapping.

Provides:
- ConversionContext: Lightweight builder for custom conversion rules
- op_conversion: Decorator for registering custom conversion functions
- register_op_conversion: Simple name mapping registration
"""

from collections.abc import Callable
from typing import Any

from pypto.pypto_core.ir import (
    AssignStmt,
    Expr,
    Span,
    Stmt,
    Var,
)
from pypto.pypto_core.ir import (
    register_op_conversion as _register_simple,
)
from pypto.pypto_core.ir import (
    register_op_conversion_custom as _register_custom,
)


class ConversionContext:
    """Lightweight builder for conversion rules. Accumulates prologue statements."""

    def __init__(self, span: Span) -> None:
        self._stmts: list[Stmt] = []
        self._span = span

    def let(self, name: str, value: Expr) -> Var:
        """Create variable, assign value, emit AssignStmt. Returns the Var."""
        var = Var(name, value.type, self._span)
        self._stmts.append(AssignStmt(var, value, self._span))
        return var

    def emit(self, stmt: Stmt) -> None:
        """Emit a raw statement into the prologue."""
        self._stmts.append(stmt)

    @property
    def stmts(self) -> list[Stmt]:
        return list(self._stmts)


def register_op_conversion(from_op: str, to_op: str) -> None:
    """Register a simple tensor-to-block op name mapping.

    Args:
        from_op: Source op name (e.g., 'tensor.add')
        to_op: Target op name (e.g., 'block.add')
    """
    _register_simple(from_op, to_op)


def op_conversion(from_op: str) -> Callable:
    """Decorator for registering custom conversion functions.

    The decorated function receives (ctx, args, kwargs, span) where:
    - ctx: ConversionContext for accumulating prologue statements
    - args: list[Expr] — substituted positional arguments
    - kwargs: list[tuple[str, Any]] — keyword arguments
    - span: Span — source location

    It should return an Expr (the result expression).

    Example::

        @op_conversion("tensor.matmul")
        def convert_matmul(ctx, args, kwargs, span):
            lhs_l0a = ctx.let("lhs_l0a", block_ops.move(args[0], target_memory=MemorySpace.L0A))
            rhs_l0b = ctx.let("rhs_l0b", block_ops.move(args[1], target_memory=MemorySpace.L0B))
            return block_ops.matmul(lhs_l0a, rhs_l0b)
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(
            args: list[Expr], kwargs: list[tuple[str, Any]], span: Span
        ) -> Expr | tuple[list[Stmt], Expr]:
            ctx = ConversionContext(span)
            result = func(ctx, args, kwargs, span)
            if ctx.stmts:
                return (ctx.stmts, result)
            return result

        _register_custom(from_op, wrapper)
        return func

    return decorator
