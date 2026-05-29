# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser/printer tests for the unified ``with pl.scope(mode=...):`` DSL construct.

``pl.scope()`` is the single block-level runtime-scope primitive (``RuntimeScopeStmt``):
``pl.scope()`` is AUTO, ``pl.scope(mode=pl.ScopeMode.MANUAL)`` is MANUAL (the former
``pl.manual_scope()``, kept as an alias). Hand-placed AUTO scopes require
``@pl.function(auto_scope=False)``; MANUAL scopes are allowed in either mode.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.printer import python_print


def _first_runtime_scope(stmt):
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_runtime_scope(s)
            if r is not None:
                return r
    return None


def _first_if(stmt):
    if isinstance(stmt, ir.IfStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_if(s)
            if r is not None:
                return r
    if isinstance(stmt, (ir.ForStmt, ir.RuntimeScopeStmt)):
        return _first_if(stmt.body)
    return None


def test_scope_auto_requires_opt_out_and_round_trips():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                a = self.k1(x)
            return a

    fn = Prog.get_function("main")
    assert fn is not None
    scope = _first_runtime_scope(fn.body)
    assert scope is not None and scope.manual is False

    printed = python_print(Prog, format=False)
    assert "pl.scope()" in printed
    assert "auto_scope=False" in printed
    ir.assert_structural_equal(Prog, pl.parse(printed))


def test_scope_manual_mode_round_trips():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                a = self.k1(x)
            return a

    fn = Prog.get_function("main")
    assert fn is not None
    scope = _first_runtime_scope(fn.body)
    assert scope is not None and scope.manual is True

    printed = python_print(Prog, format=False)
    assert "pl.scope(mode=pl.ScopeMode.MANUAL)" in printed
    ir.assert_structural_equal(Prog, pl.parse(printed))


def test_manual_scope_alias_parses_to_same_ir():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.manual_scope():
                a = self.k1(x)
            return a

    fn = Prog.get_function("main")
    assert fn is not None
    scope = _first_runtime_scope(fn.body)
    assert scope is not None and scope.manual is True


def test_scope_rejects_positional_args():
    with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(1):  # type: ignore[arg-type]  # deliberate: positional arg rejected
                    a = x
                return a


def test_auto_scope_rejected_in_default_mode():
    with pytest.raises(Exception):  # noqa: B017 — AUTO scope requires auto_scope=False

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope():
                    a = self.k1(x)
                return a


def test_loop_carried_yield_outside_scope_ok():
    # Supported pattern: scope wraps the per-iteration work, the loop-carried
    # yield is a direct for-body child.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            for i, (acc,) in pl.range(4, init_values=(out,)):
                with pl.scope():
                    nxt: pl.Tensor[[16, 16], pl.FP32] = self.kernel(a, acc)
                acc = pl.yield_(nxt)
            return acc

    assert Prog.get_function("orch") is not None


def test_manual_scope_in_if_branch_registers_yield_var():
    # Regression: a `pl.yield_` wrapped in `with pl.manual_scope():` inside an
    # if branch must still register as the if's return-var. _scan_for_yields
    # treats manual_scope (an alias for `pl.scope(mode=MANUAL)`) as transparent,
    # exactly like `pl.scope()` — otherwise the enclosing if drops `return_var`.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(4, init_values=(x,)):
                if i == 0:
                    with pl.manual_scope():
                        a: pl.Tensor[[64], pl.FP32] = self.k1(acc)
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(a)
                else:
                    with pl.manual_scope():
                        b: pl.Tensor[[64], pl.FP32] = self.k1(acc)
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(b)
                acc = pl.yield_(val)
            return acc

    fn = Prog.get_function("main")
    assert fn is not None
    if_stmt = _first_if(fn.body)
    assert if_stmt is not None
    names = {v.name_hint for v in if_stmt.return_vars}
    assert "val" in names


def test_auto_scope_rejected_inside_manual_scope():
    with pytest.raises(Exception):  # noqa: B017 — runtime forbids AUTO nested in MANUAL

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.MANUAL):
                    with pl.scope():
                        a = self.k1(x)
                return a


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
