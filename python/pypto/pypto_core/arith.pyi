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
from typing import ClassVar

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
