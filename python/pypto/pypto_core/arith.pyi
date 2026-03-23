# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for the arith submodule (arithmetic simplification utilities)."""

from pypto.pypto_core.ir import Expr

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
