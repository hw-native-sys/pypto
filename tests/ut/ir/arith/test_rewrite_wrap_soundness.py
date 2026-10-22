# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Differential oracle: ``Analyzer.simplify`` must preserve INT8 *wrapping* semantics.

Codegen evaluates each node at its own dtype, so `+`, `-`, `*` wrap. A rewrite rule is an identity over the
integers, which is not the same thing: for INT8 `c` in [120, 127], `c + 1 < 0` is true at `c == 127` (the
sum is -128) but the integer-identity rewrite `c < -1` is false everywhere. This file evaluates an expression
and its simplified form over every assignment in a window, with INT8 wrap-around, and requires them to agree.

One template per rule shape. A new rewrite rule that moves a value across `<`, `<=`, `==`, `min`, `max`,
`floordiv` or `floormod` should add its template here so a wrap-unsound rule cannot land again.
"""

import itertools

import pytest
from pypto import DataType, ir
from pypto.arith import Analyzer

S = ir.Span.unknown()
I8 = DataType.INT8
B = DataType.BOOL


def wrap(v: int) -> int:
    """Wrap to INT8, as the emitted `arith.*i : i8` does."""
    return ((v + 128) % 256) - 128


def var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(I8), S)


def c(v: int) -> ir.ConstInt:
    return ir.ConstInt(v, I8, S)


X, Y, Z = var("x"), var("y"), var("z")

_BIN = {
    "Add": lambda a, b: wrap(a + b),
    "Sub": lambda a, b: wrap(a - b),
    "Mul": lambda a, b: wrap(a * b),
    "FloorDiv": lambda a, b: a // b,
    "FloorMod": lambda a, b: a % b,
    "Min": min,
    "Max": max,
    "Lt": lambda a, b: a < b,
    "Le": lambda a, b: a <= b,
    "Gt": lambda a, b: a > b,
    "Ge": lambda a, b: a >= b,
    "Eq": lambda a, b: a == b,
    "Ne": lambda a, b: a != b,
    "And": lambda a, b: bool(a) and bool(b),
    "Or": lambda a, b: bool(a) or bool(b),
}


def evaluate(e: ir.Expr, env: dict[str, int]):
    """Evaluate `e` with INT8 wrap-around; a literal is materialised at INT8 (so `-200` is `56`)."""
    if isinstance(e, ir.ConstInt):
        return wrap(e.value)
    if isinstance(e, ir.ConstBool):
        return e.value
    if isinstance(e, ir.Var):
        return env[e.name_hint]
    if isinstance(e, ir.Neg):
        return wrap(-evaluate(e.operand, env))
    if isinstance(e, ir.Not):
        return not evaluate(e.operand, env)
    assert isinstance(e, ir.BinaryExpr), f"unsupported node in simplified expression: {type(e).__name__}"
    return _BIN[type(e).__name__](evaluate(e.left, env), evaluate(e.right, env))


# Operand windows as (lo, hi_exclusive): the whole dtype, and the two edges where a sum or difference wraps.
WINDOWS = {"full": (-128, 128), "high": (120, 128), "low": (-128, -120)}
SMALL = range(-4, 4)  # y and z, so three-variable templates stay at <= 256 * 64 assignments

TEMPLATES = {
    # --- comparisons: cancellation and constant moves -------------------------------------------------
    "lt-cancel-add": lambda: ir.Lt(ir.Add(X, Y, I8, S), ir.Add(X, Z, I8, S), B, S),
    "le-cancel-add": lambda: ir.Le(ir.Add(X, Y, I8, S), ir.Add(X, Z, I8, S), B, S),
    "lt-cancel-sub-rhs": lambda: ir.Lt(ir.Sub(Y, X, I8, S), ir.Sub(Z, X, I8, S), B, S),
    "lt-cancel-sub-lhs": lambda: ir.Lt(ir.Sub(X, Y, I8, S), ir.Sub(X, Z, I8, S), B, S),
    "lt-x-below-x-plus-z": lambda: ir.Lt(X, ir.Add(X, Z, I8, S), B, S),
    "lt-x-below-x-minus-z": lambda: ir.Lt(X, ir.Sub(X, Z, I8, S), B, S),
    "lt-add-const": lambda: ir.Lt(ir.Add(X, c(1), I8, S), c(0), B, S),
    "lt-sub-const": lambda: ir.Lt(ir.Sub(X, c(1), I8, S), c(0), B, S),
    "lt-const-sub-var": lambda: ir.Lt(ir.Sub(c(0), X, I8, S), c(5), B, S),
    "lt-const-below-add": lambda: ir.Lt(c(0), ir.Add(X, c(1), I8, S), B, S),
    "lt-const-below-sub": lambda: ir.Lt(c(-5), ir.Sub(X, c(1), I8, S), B, S),
    "le-add-const": lambda: ir.Le(ir.Add(X, c(1), I8, S), c(-1), B, S),
    "le-sub-const": lambda: ir.Le(ir.Sub(X, c(1), I8, S), c(0), B, S),
    "le-const-sub-var": lambda: ir.Le(ir.Sub(c(0), X, I8, S), c(5), B, S),
    "gt-add-const": lambda: ir.Gt(c(0), ir.Add(X, c(1), I8, S), B, S),
    "ge-add-const": lambda: ir.Ge(ir.Add(X, c(1), I8, S), c(0), B, S),
    "lt-mul-const": lambda: ir.Lt(ir.Mul(X, c(2), I8, S), ir.Mul(Y, c(2), I8, S), B, S),
    "le-mul-const": lambda: ir.Le(ir.Mul(X, c(2), I8, S), ir.Mul(Y, c(2), I8, S), B, S),
    "lt-floordiv-const": lambda: ir.Lt(ir.FloorDiv(X, c(16), I8, S), c(10), B, S),
    "lt-const-below-floordiv": lambda: ir.Lt(c(3), ir.FloorDiv(X, c(3), I8, S), B, S),
    # --- equality: a moved constant can leave the dtype -----------------------------------------------
    "eq-add-const": lambda: ir.Eq(ir.Add(X, c(100), I8, S), c(-100), B, S),
    "eq-sub-const": lambda: ir.Eq(ir.Sub(X, c(100), I8, S), c(100), B, S),
    "eq-const-sub-var": lambda: ir.Eq(ir.Sub(c(100), X, I8, S), c(-100), B, S),
    "ne-add-const": lambda: ir.Ne(ir.Add(X, c(100), I8, S), c(-100), B, S),
    "eq-cancel-add": lambda: ir.Eq(ir.Add(X, Y, I8, S), ir.Add(X, Z, I8, S), B, S),
    # --- min / max with offsets -----------------------------------------------------------------------
    "min-add-consts": lambda: ir.Min(ir.Add(X, c(1), I8, S), ir.Add(X, c(3), I8, S), I8, S),
    "min-add-vars": lambda: ir.Min(ir.Add(X, Y, I8, S), ir.Add(X, Z, I8, S), I8, S),
    "min-sub-vars": lambda: ir.Min(ir.Sub(Y, X, I8, S), ir.Sub(Z, X, I8, S), I8, S),
    "min-const-sub": lambda: ir.Min(ir.Sub(c(1), X, I8, S), ir.Sub(c(3), X, I8, S), I8, S),
    "min-self-offset": lambda: ir.Min(ir.Add(X, c(5), I8, S), X, I8, S),
    "max-add-consts": lambda: ir.Max(ir.Add(X, c(1), I8, S), ir.Add(X, c(3), I8, S), I8, S),
    "max-add-vars": lambda: ir.Max(ir.Add(X, Y, I8, S), ir.Add(X, Z, I8, S), I8, S),
    "min-minus-x": lambda: ir.Sub(ir.Min(ir.Add(X, Y, I8, S), Z, I8, S), X, I8, S),
    "add-min-offset": lambda: ir.Add(ir.Min(X, ir.Sub(Y, Z, I8, S), I8, S), Z, I8, S),
    "sub-min-var": lambda: ir.Sub(ir.Min(X, Y, I8, S), X, I8, S),
    "sub-var-min-offset": lambda: ir.Sub(X, ir.Min(ir.Add(X, Y, I8, S), Z, I8, S), I8, S),
    # --- floordiv / floormod with offsets -------------------------------------------------------------
    "floordiv-add-const": lambda: ir.FloorDiv(ir.Add(X, c(4), I8, S), c(4), I8, S),
    "floormod-add-const": lambda: ir.FloorMod(ir.Add(X, c(3), I8, S), c(3), I8, S),
    "floordiv-mul-add": lambda: ir.FloorDiv(ir.Add(ir.Mul(X, c(4), I8, S), Y, I8, S), c(4), I8, S),
    "floormod-mul-add": lambda: ir.FloorMod(ir.Add(ir.Mul(X, c(4), I8, S), Y, I8, S), c(4), I8, S),
    "sub-floordiv-sum": lambda: ir.Sub(
        X, ir.Mul(ir.FloorDiv(ir.Add(X, Y, I8, S), c(4), I8, S), c(4), I8, S), I8, S
    ),
    # The remainder workaround `(x % m + k) % m` may only be dropped when `x % m + k` cannot wrap at INT8:
    # 24 + 23 fits (reduces), 100 + 99 does not (must stay correct), and an unbounded x + 24 wraps above 103.
    "floormod-workaround-fits": lambda: ir.FloorMod(
        ir.Add(ir.FloorMod(X, c(24), I8, S), c(24), I8, S), c(24), I8, S
    ),
    "floormod-workaround-wraps": lambda: ir.FloorMod(
        ir.Add(ir.FloorMod(X, c(100), I8, S), c(100), I8, S), c(100), I8, S
    ),
    "floormod-unbounded-offset": lambda: ir.FloorMod(ir.Add(c(24), X, I8, S), c(24), I8, S),
    # --- controls: ring identities and wrap-free rules must stay sound -------------------------------
    "ring-distribute": lambda: ir.Mul(ir.Add(X, c(3), I8, S), c(5), I8, S),
    "ring-double": lambda: ir.Add(X, X, I8, S),
    "lt-self": lambda: ir.Lt(X, X, B, S),
    "sub-floordiv-self": lambda: ir.Sub(X, ir.Mul(ir.FloorDiv(X, c(4), I8, S), c(4), I8, S), I8, S),
}


def _assignments(expr: ir.Expr, window: tuple[int, int]):
    names = sorted({n for n in ("x", "y", "z") if _mentions(expr, n)})
    ranges = [range(*window) if n == "x" else SMALL for n in names]
    for values in itertools.product(*ranges):
        yield dict(zip(names, values))


def _mentions(e: ir.Expr, name: str) -> bool:
    if isinstance(e, ir.Var):
        return e.name_hint == name
    if isinstance(e, ir.BinaryExpr):
        return _mentions(e.left, name) or _mentions(e.right, name)
    if isinstance(e, ir.UnaryExpr):
        return _mentions(e.operand, name)
    return False


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_simplify_preserves_int8_wrapping_semantics(name, window):
    """The simplified expression equals the original at every INT8 assignment in the window."""
    expr = TEMPLATES[name]()
    lo, hi = WINDOWS[window]
    analyzer = Analyzer()
    analyzer.bind(X, lo, hi)
    for v in (Y, Z):
        analyzer.bind(v, SMALL.start, SMALL.stop)
    simplified = analyzer.simplify(expr)
    for env in _assignments(expr, (lo, hi)):
        assert evaluate(simplified, env) == evaluate(expr, env), (
            f"{name}: simplify changed the INT8 value at {env}: {expr} -> {simplified}"
        )


def test_ring_identities_still_simplify():
    """The gate is not a blanket ban: wrap-safe ring identities keep simplifying at INT8."""
    simplified = Analyzer().simplify(ir.Add(X, X, I8, S))
    assert type(simplified).__name__ == "Mul", f"x + x was not rewritten: {simplified}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
