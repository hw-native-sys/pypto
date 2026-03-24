# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for the arith submodule (arithmetic simplification utilities)."""

from collections.abc import Callable
from types import TracebackType
from typing import ClassVar, overload

from pypto.pypto_core.ir import Expr, Var

def fold_const(expr: Expr) -> Expr | None:
    """Try to constant-fold an expression."""
    ...

def floordiv(x: int, y: int) -> int:
    """Floor division."""
    ...

def floormod(x: int, y: int) -> int:
    """Floor modulo."""
    ...

def gcd(a: int, b: int) -> int:
    """GCD (treats 0 as identity)."""
    ...

def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    ...

def extended_euclidean(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean: returns (gcd, x, y) where a*x + b*y = gcd."""
    ...

class RewriteSimplifier:
    """Simplifies integer/index expressions by applying algebraic rewrite rules.

    Float-typed expressions are returned unchanged.
    """

    def __init__(self) -> None:
        """Create a standalone RewriteSimplifier."""
        ...

    def __call__(self, expr: Expr) -> Expr:
        """Simplify an expression by applying rewrite rules."""
        ...

    def update(self, var: Var, new_expr: Expr | None) -> None:
        """Register a variable substitution: replace var with new_expr during simplification.

        Pass None to remove a previous substitution.
        """
        ...

    def enter_constraint(self, constraint: Expr) -> Callable[[], None]:
        """Enter a constraint scope. Returns a recovery function that restores original state."""
        ...

class ConstIntBound:
    """Inclusive integer bounds [min_value, max_value] for an expression."""

    def __init__(self, min_value: int, max_value: int) -> None:
        """Create inclusive integer bounds [min_value, max_value]."""
        ...

    min_value: int
    max_value: int
    kPosInf: ClassVar[int]
    kNegInf: ClassVar[int]

    def is_const(self) -> bool:
        """Check if min == max (constant)."""
        ...

    def is_non_negative(self) -> bool:
        """Check if min >= 0."""
        ...

    def is_positive(self) -> bool:
        """Check if min > 0."""
        ...

    def is_everything(self) -> bool:
        """Check if bounds are [-inf, +inf] (no information)."""
        ...

class ConstIntBoundAnalyzer:
    """Propagates constant integer bounds through expression trees."""

    def __init__(self) -> None:
        """Create a standalone ConstIntBoundAnalyzer."""
        ...

    def __call__(self, expr: Expr) -> ConstIntBound:
        """Compute bounds for an expression."""
        ...

    def bind(self, var: Var, min_val: int, max_val_exclusive: int) -> None:
        """Bind a variable to the half-open range [min_val, max_val_exclusive)."""
        ...

    def update(self, var: Var, bound: ConstIntBound) -> None:
        """Update a variable's bound (inclusive on both ends)."""
        ...

class ModularSet:
    """Modular arithmetic properties: value = coeff * k + base."""

    def __init__(self, coeff: int, base: int) -> None:
        """Create a modular set with given coeff and base."""
        ...

    coeff: int
    base: int

    def is_exact(self) -> bool:
        """Check if exact value is known (coeff == 0)."""
        ...

    def is_everything(self) -> bool:
        """Check if no useful modular info (coeff == 1, base == 0)."""
        ...

class ModularSetAnalyzer:
    """Tracks modular arithmetic properties through expression trees."""

    def __init__(self) -> None:
        """Create a standalone ModularSetAnalyzer."""
        ...

    def __call__(self, expr: Expr) -> ModularSet:
        """Compute modular set for an expression."""
        ...

    def update(self, var: Var, info: ModularSet) -> None:
        """Update a variable's modular set information."""
        ...

    def enter_constraint(self, constraint: Expr) -> Callable[[], None] | None:
        """Enter a constraint scope. Returns a recovery function, or None."""
        ...

class Analyzer:
    """Coordinates all sub-analyzers for expression analysis and simplification."""

    def __init__(self) -> None:
        """Create an Analyzer with all sub-analyzers."""
        ...

    const_int_bound: ConstIntBoundAnalyzer
    modular_set: ModularSetAnalyzer
    rewrite_simplify: RewriteSimplifier

    @overload
    def bind(self, var: Var, expr: Expr, allow_override: bool = False) -> None: ...
    @overload
    def bind(self, var: Var, min_val: int, max_val_exclusive: int, allow_override: bool = False) -> None: ...
    def bind(self, var: Var, *args, **kwargs) -> None:
        """Bind a variable to an expression or half-open range [min_val, max_val_exclusive)."""
        ...

    def simplify(self, expr: Expr, steps: int = 2) -> Expr:
        """Simplify an expression by iterative rewrite simplification."""
        ...

    def can_prove_greater_equal(self, expr: Expr, lower_bound: int) -> bool:
        """Prove that expr >= lower_bound for all possible variable values."""
        ...

    def can_prove_less(self, expr: Expr, upper_bound: int) -> bool:
        """Prove that expr < upper_bound for all possible variable values."""
        ...

    def can_prove_equal(self, lhs: Expr, rhs: Expr) -> bool:
        """Prove that lhs and rhs are always equal."""
        ...

    def can_prove(self, cond: Expr) -> bool:
        """Prove that a boolean condition is always true."""
        ...

    def constraint_context(self, constraint: Expr) -> ConstraintContext:
        """Create a constraint scope (context manager).

        Usage::

            with analyzer.constraint_context(x >= 0):
                ...  # x >= 0 is assumed true here
        """
        ...

class ConstraintContext:
    """RAII guard that enters a constraint scope on all sub-analyzers.

    Prefer ``Analyzer.constraint_context()`` over direct construction.
    """

    def exit_scope(self) -> None:
        """Explicitly exit the constraint scope (idempotent)."""
        ...

    def __enter__(self) -> ConstraintContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
