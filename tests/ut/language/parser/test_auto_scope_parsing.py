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
from pypto.language.parser.diagnostics import ParserSyntaxError


def _flatten_seq(stmt) -> list:
    """Return a body's statement list, treating a bare Stmt as a 1-element body."""
    return list(stmt.stmts) if isinstance(stmt, ir.SeqStmts) else [stmt]


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


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("name_hint", ["", "named", 'phase "one"\\tail\n\t\r\x00\x7f \u9636\u6bb5'])
def test_hand_built_runtime_scope_name_roundtrip(manual, name_hint):
    """Compare the original Program, including its name, with parsed printer output."""
    span = ir.Span.unknown()
    scope = ir.RuntimeScopeStmt(manual, name_hint, ir.ReturnStmt([], span), span)
    function = ir.Function(
        "main", [], [], scope, span, ir.FunctionType.Orchestration, attrs={"auto_scope": False}
    )
    original = ir.Program([function], "P", span)

    restored = pl.parse_program(python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    restored_function = restored.get_function("main")
    assert restored_function is not None
    restored_scope = _first_runtime_scope(restored_function.body)
    assert restored_scope is not None
    assert restored_scope.name_hint == name_hint
    assert restored_scope.manual == manual


def test_runtime_scope_name_is_structural():
    span = ir.Span.unknown()
    body = ir.ReturnStmt([], span)
    named = ir.RuntimeScopeStmt(True, "named", body, span)
    unnamed = ir.RuntimeScopeStmt(True, "", body, span)
    assert not ir.structural_equal(named, unnamed)


@pytest.mark.parametrize("header", ["if flag:", "for i in pl.range(4):", "while flag:"])
def test_named_auto_scope_in_control_flow_body_roundtrip(header):
    """Preserve a named AUTO scope when it is the only control-flow body statement."""
    original = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, flag: pl.Scalar[pl.BOOL]):
        {header}
            with pl.scope(name_hint="control body"):
                value: pl.Scalar[pl.INT32] = 1
        return
""")

    restored = pl.parse_program(python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)


def test_named_auto_scopes_preserve_nested_control_flow_yields():
    """Keep scope names and result bindings through nested loop/if body printing."""

    @pl.program
    class Original:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(4, init_values=(x,)):
                with pl.scope(name_hint="iteration"):
                    if i == 0:
                        with pl.scope(name_hint="first iteration"):
                            value = pl.yield_(acc)
                    else:
                        with pl.scope(name_hint="later iteration"):
                            value = pl.yield_(acc)
                    acc = pl.yield_(value)
            return acc

    restored = pl.parse_program(python_print(Original, format=False))
    ir.assert_structural_equal(Original, restored)
    assert ir.structural_hash(Original) == ir.structural_hash(restored)


@pytest.mark.parametrize(
    "header, manual",
    [("pl.scope(", False), ("pl.scope(mode=pl.ScopeMode.MANUAL, ", True), ("pl.manual_scope(", True)],
)
def test_runtime_scope_name_dsl_spellings(header, manual):
    original = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self):
        with {header}name_hint="phase one"):
            return
""")
    function = original.get_function("main")
    assert function is not None
    scope = _first_runtime_scope(function.body)
    assert scope is not None
    assert scope.name_hint == "phase one"
    assert scope.manual == manual
    ir.assert_structural_equal(original, pl.parse_program(python_print(original)))


@pytest.mark.parametrize("construct", ["pl.scope", "pl.manual_scope"])
@pytest.mark.parametrize("value", ["42", "None", '"phase" + "one"'])
def test_runtime_scope_name_requires_a_string_literal(construct, value):
    with pytest.raises(ParserSyntaxError, match="'name_hint' argument must be a string literal"):
        pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self):
        with {construct}(name_hint={value}):
            return
""")


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
    with pytest.raises(ParserSyntaxError):

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(1):  # type: ignore[arg-type]  # deliberate: positional arg rejected
                    a = x
                return a


def test_auto_scope_rejected_in_default_mode():
    with pytest.raises(ParserSyntaxError):  # AUTO scope requires auto_scope=False

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

    orch = Prog.get_function("orch")
    assert orch is not None

    # The scope wraps the per-iteration work; the yield stays a direct for-body
    # child (not swallowed into the scope).
    for_stmt = next(s for s in _flatten_seq(orch.body) if isinstance(s, ir.ForStmt))
    body_stmts = _flatten_seq(for_stmt.body)
    assert isinstance(body_stmts[0], ir.ScopeStmt)
    assert isinstance(body_stmts[-1], ir.YieldStmt)
    assert len(for_stmt.iter_args) == 1


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
    with pytest.raises(ParserSyntaxError):  # runtime forbids AUTO nested in MANUAL

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
