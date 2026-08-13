# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing ScopeStmt with pl.at(level=pl.Level.CORE_GROUP): syntax."""

import ast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError
from pypto.language.parser.text_parser import parse_program

_OP_SYSTEM_TASK_INVALID = ir.get_op("system.task_invalid").name
_OP_TILE_GET_BLOCK_IDX = ir.get_op("tile.get_block_idx").name


def _descendants(node, cls):
    """Collect every descendant of ``node`` (inclusive) that is an instance of ``cls``."""
    found = []

    def walk(n):
        if isinstance(n, cls):
            found.append(n)
        if isinstance(n, ir.SeqStmts):
            for s in n.stmts:
                walk(s)
        elif hasattr(n, "body") and n.body is not None:
            walk(n.body)

    walk(node)
    return found


def _unique_descendant(node, cls):
    """Return the single descendant of ``node`` that is an instance of ``cls``."""
    found = _descendants(node, cls)
    assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
    return found[0]


@pl.function(type=pl.FunctionType.InCore)
def _external_worker(
    a: pl.Tensor[[64], pl.FP32],
    out: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    """Standalone kernel used to exercise bare-name dispatch inside ``pl.spmd``."""
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.add(a, a)
    return out


class TestScopeParsing:
    """Test parsing of with pl.at(level=pl.Level.CORE_GROUP): syntax."""

    def test_parse_simple_incore_scope(self):
        """Test parsing a simple InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Verify the program was parsed successfully
        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

        # Get the main function
        main_func = list(TestProgram.functions.values())[0]
        assert main_func.name == "main"

        # Verify the body contains a ScopeStmt
        # The body should be SeqStmts containing ScopeStmt
        assert isinstance(main_func.body, ir.SeqStmts)

    def test_parse_nested_operations_in_scope(self):
        """Test parsing multiple operations inside InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        # Verify the program was parsed successfully
        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

    def test_parse_multiple_incore_scopes(self):
        """Test parsing multiple InCore scopes in one function."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        # Verify the program was parsed successfully
        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

    def test_parse_scope_with_surrounding_code(self):
        """Test parsing InCore scope with code before and after."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                return c

        # Verify the program was parsed successfully
        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

    def test_print_and_reparse_scope(self):
        """Test that printed ScopeStmt can be reparsed."""

        @pl.program
        class Original:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Print the program
        printed = Original.as_python()

        # Verify it contains the scope syntax
        assert "with pl.at(level=pl.Level.CORE_GROUP):" in printed


class TestScopeNameParsing:
    """Test parsing of scope name parameter."""

    def test_parse_named_incore_scope(self):
        """Test parsing with pl.at(level=..., name='my_kernel')."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="my_kernel"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        assert TestProgram is not None
        main_func = list(TestProgram.functions.values())[0]
        # Find the ScopeStmt and verify name field
        body = main_func.body
        if isinstance(body, ir.SeqStmts):
            scope_stmt = body.stmts[0]
        else:
            scope_stmt = body
        assert isinstance(scope_stmt, ir.ScopeStmt)
        assert scope_stmt.name_hint == "my_kernel"
        assert scope_stmt.scope_kind == ir.ScopeKind.InCore

    def test_parse_unnamed_scope_has_empty_name(self):
        """Test that unnamed scopes have empty name."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(TestProgram.functions.values())[0]
        body = main_func.body
        if isinstance(body, ir.SeqStmts):
            scope_stmt = body.stmts[0]
        else:
            scope_stmt = body
        assert isinstance(scope_stmt, ir.ScopeStmt)
        assert scope_stmt.name_hint == ""

    def test_parse_invalid_name_raises_error(self):
        """Test that invalid identifier names raise ParserSyntaxError."""
        with pytest.raises(ParserSyntaxError, match="valid non-keyword identifier"):

            @pl.program
            class TestProgram:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="has space"):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_named_scope_printer_roundtrip(self):
        """Test that named scopes roundtrip through the printer."""

        @pl.program
        class Original:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="my_kernel"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        printed = Original.as_python()
        assert 'name_hint="my_kernel"' in printed

    def test_parse_named_hierarchy_scope(self):
        """Test parsing with pl.at(level=HOST, name='host_func')."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, name_hint="host_func"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(TestProgram.functions.values())[0]
        body = main_func.body
        if isinstance(body, ir.SeqStmts):
            scope_stmt = body.stmts[0]
        else:
            scope_stmt = body
        assert isinstance(scope_stmt, ir.ScopeStmt)
        assert scope_stmt.name_hint == "host_func"
        assert scope_stmt.scope_kind == ir.ScopeKind.Hierarchy


class TestSpmdForLoop:
    """Test parsing of ``for i in pl.spmd(...):`` loop form.

    The loop form is syntactic sugar that expands to
    ``SpmdScopeStmt(body=InCoreScopeStmt(body=<i = tile.get_block_idx(); ...>))``
    so inline tile/tensor ops have direct access to the per-block index
    without a separate ``@pl.function(type=InCore)`` declaration.
    """

    def test_for_spmd_builds_spmd_scope_wrapping_incore(self):
        """Loop form emits SpmdScopeStmt containing an InCoreScopeStmt whose
        first statement binds the loop var to pl.tile.get_block_idx().

        ``core_num`` is positional — mirroring ``range(n)``.
        """

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    offset = i * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 4
        assert spmd.sync_start is False
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)

        body = incore.body
        first_stmt = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first_stmt, ir.AssignStmt)
        call = first_stmt.value
        assert isinstance(call, ir.Call)
        assert call.op.name == _OP_TILE_GET_BLOCK_IDX
        assert first_stmt.var.name_hint == "i"

    def test_for_spmd_accepts_core_num_kwarg(self):
        """Backward-compat: ``pl.spmd(core_num=N)`` keyword form still parses."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(core_num=4):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 4

    def test_for_spmd_accepts_closure_int_variable(self):
        """Closure-captured Python ints resolve to ConstInt via parse_name.

        Regression test for issue #1125 — parameterized builder functions
        need to pass ``core_num`` as a Python variable.
        """
        max_ctx_blocks = 64  # Plain Python int in the enclosing scope.

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(core_num=max_ctx_blocks):
                    offset = i * 8
                    t: pl.Tile[[8, 128], pl.FP32] = pl.load(a, [offset, 0], [8, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 64

    def test_for_spmd_accepts_closure_binop(self):
        """Closure arithmetic folds to ConstInt via parse_binop's fold path."""
        MAX_CTX_BLOCKS = 128
        SB_BATCH = 2

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(core_num=MAX_CTX_BLOCKS // SB_BATCH):
                    offset = i * 8
                    t: pl.Tile[[8, 128], pl.FP32] = pl.load(a, [offset, 0], [8, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 64

    def test_for_spmd_sync_start_and_name_hint(self):
        """sync_start= and name_hint= pass through to SpmdScopeStmt."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(8, sync_start=True, name_hint="my_kernel"):
                    offset = i * 64
                    t: pl.Tile[[64, 128], pl.FP32] = pl.load(a, [offset, 0], [64, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 8
        assert spmd.sync_start is True
        assert spmd.name_hint == "my_kernel_spmd"
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.name_hint == "my_kernel"

    def test_for_spmd_name_hint_split_base_and_spmd_suffix(self):
        """``name_hint`` on for-spmd splits between outer Spmd and inner InCore."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, name_hint="q_proj"):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert spmd.name_hint == "q_proj_spmd"
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.name_hint == "q_proj"

    def test_for_spmd_name_hint_already_has_spmd_suffix(self):
        """A user-provided ``*_spmd`` hint is kept on Spmd; InCore drops the suffix."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, name_hint="gate_proj_spmd"):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert spmd.name_hint == "gate_proj_spmd"
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.name_hint == "gate_proj"

    def test_with_spmd_single_call_still_supported(self):
        """Regression: the existing ``with pl.spmd(...):`` single-call form
        still builds a direct SpmdScopeStmt(body=Call), no InCore wrapping."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, out)
                return out

        main_func = TestProgram.functions[list(TestProgram.functions.keys())[-1]]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        # Walk body — should NOT contain an InCoreScopeStmt (no implicit wrap).
        found_incore = []

        def walk(n):
            if isinstance(n, ir.InCoreScopeStmt):
                found_incore.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(spmd.body)
        assert not found_incore, "with-form should not insert an implicit InCoreScopeStmt"

    def test_for_spmd_rejects_tuple_target(self):
        """A tuple target on for-spmd is rejected (single loop var only)."""
        with pytest.raises(ParserSyntaxError, match="single loop variable"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, j in pl.spmd(4):  # type: ignore[misc]
                    _ = i + j
                return a

    def test_for_spmd_rejects_chunk_kwarg(self):
        """chunk= is not a valid kwarg on pl.spmd loop forms."""
        with pytest.raises(ParserSyntaxError, match=r"does not accept 'chunk='"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, chunk=2):  # type: ignore[call-arg]
                    _ = i
                return a

    def test_for_spmd_rejects_init_values(self):
        """init_values= implies loop-carried state, which SPMD has no notion of."""
        with pytest.raises(ParserSyntaxError, match=r"does not accept 'init_values='"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, init_values=(0,)):  # type: ignore[call-arg]
                    _ = i
                return a

    def test_for_spmd_requires_core_num(self):
        """Missing core_num raises a targeted diagnostic."""
        with pytest.raises(ParserSyntaxError, match="requires core_num"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd():  # type: ignore[call-arg]
                    _ = i
                return a

    def test_for_spmd_rejects_zero_core_num(self):
        """core_num must be a positive integer."""
        with pytest.raises(ParserSyntaxError, match="must be a positive integer"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(0):
                    _ = i
                return a

    def test_for_spmd_rejects_float_core_num(self):
        """core_num must resolve to an integer-typed expression."""
        with pytest.raises(ParserSyntaxError, match="must be an integer expression"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(1.5):  # type: ignore[arg-type]
                    _ = i
                return a

    def test_for_spmd_rejects_bool_core_num(self):
        """A boolean literal is not an acceptable core_num."""
        with pytest.raises(ParserSyntaxError, match="must be an integer expression"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(True):  # type: ignore[arg-type]
                    _ = i
                return a

    def test_for_spmd_rejects_duplicate_core_num(self):
        """Supplying ``core_num`` positionally *and* as a kwarg is rejected."""
        with pytest.raises(ParserSyntaxError, match="multiple values for argument 'core_num'"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, core_num=4):  # type: ignore[misc]
                    _ = i
                return a

    def test_for_spmd_rejects_extra_positional(self):
        """``pl.spmd`` takes a single positional ``core_num``; a second one is an error."""
        with pytest.raises(ParserSyntaxError, match="at most one positional argument"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, 2):  # type: ignore[misc]
                    _ = i
                return a

    def test_for_spmd_print_reparse_roundtrip(self):
        """Printing the for-spmd IR emits the loop form so it reparses cleanly.

        The printer detects the SpmdScopeStmt(InCoreScopeStmt(i = get_block_idx; ...))
        pattern and emits ``for i in pl.spmd(N):`` (positional). Emitting the
        with-form here would fail because the body has multiple statements.
        """

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        printed = Original.as_python()
        assert "for i in pl.spmd(4):" in printed

        reparsed = parse_program(printed)
        main_fn = next(f for f in reparsed.functions.values() if f.name == "main")
        ir.assert_structural_equal(main_fn, list(Original.functions.values())[0])

    def test_for_spmd_rejects_non_bool_sync_start(self):
        """sync_start must be a boolean literal (True/False)."""
        with pytest.raises(ParserSyntaxError, match="sync_start must be a boolean literal"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, sync_start=1):  # type: ignore[arg-type]
                    _ = i
                return a

    def test_for_spmd_rejects_kwargs_unpacking(self):
        """``pl.spmd(**cfg)`` raises a targeted diagnostic rather than the
        confusing default error that tries to format ``kw.arg=None``.

        The parser's kwarg walk sees ``ast.keyword(arg=None, value=...)``
        for ``**`` unpacking; our handler rejects it before ever attempting
        to evaluate the unpacked expression, so the value need not be a
        supported expression kind.
        """
        with pytest.raises(ParserSyntaxError, match=r"does not accept \*\*kwargs"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(**a):  # type: ignore[misc]
                    _ = i
                return a

    def test_for_spmd_loop_var_survives_ssa_shadowing_in_printer(self):
        """Regression: when the outer scope already defines ``i``, SSA renames
        the inner loop variable (e.g., ``i_1``). The printer must emit the
        renamed name in the ``for ... in`` header so the header matches the
        body."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                # Outer `i` shadows the loop var; the printer must rename.
                i = 0
                for i in pl.spmd(4):  # type: ignore[assignment]
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        printed = Original.as_python()
        # Extract the `for <var> in pl.spmd(4):` header and verify `<var>` is
        # referenced in the body (e.g. `<var> * 128`).
        for line in printed.splitlines():
            stripped = line.strip()
            if stripped.startswith("for ") and "pl.spmd(" in stripped:
                header_var = stripped.split()[1]
                break
        else:
            raise AssertionError(f"no for-spmd header in printed output:\n{printed}")
        assert f"{header_var} * 128" in printed, (
            f"loop var {header_var!r} from header not referenced in body; "
            f"printer likely printed a stale raw name_hint:\n{printed}"
        )
        parse_program(printed)  # round-trips cleanly


class TestSpmdOptimizations:
    """Test ``pl.spmd(..., optimizations=[pl.split(...)])`` lowering.

    Only ``pl.split(mode)`` is supported on ``pl.spmd``.
    """

    def test_for_spmd_split_sets_inner_incore_split(self):
        """``optimizations=[pl.split(mode)]`` on the for-form sets ``split_``
        on the auto-generated inner ``InCoreScopeStmt``."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN
        body = incore.body
        first_stmt = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first_stmt, ir.AssignStmt)
        call = first_stmt.value
        assert isinstance(call, ir.Call) and isinstance(call.op, ir.Op)
        assert call.op.name == _OP_TILE_GET_BLOCK_IDX

    def test_for_spmd_qualified_split_sets_inner_incore_split(self):
        """``pl.optimizations.split(...)`` is accepted on the for-form."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(
                    4,
                    optimizations=[pl.optimizations.split(pl.SplitMode.LEFT_RIGHT)],
                ):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.LEFT_RIGHT

    def test_for_spmd_empty_optimizations_matches_no_kwarg(self):
        """``optimizations=[]`` is equivalent to omitting the kwarg — inner
        scope is plain ``InCoreScopeStmt`` with ``split_=None``."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, optimizations=[]):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE

    def test_for_spmd_cross_core_slot_sets_scope_attr(self):
        """``pl.cross_core_slot(slot_num=N)`` records ``slot_num`` on the inner
        ``InCoreScopeStmt`` attrs alongside an independent split mode."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(
                    4,
                    optimizations=[
                        pl.split(pl.SplitMode.UP_DOWN),
                        pl.cross_core_slot(slot_num=16),
                    ],
                ):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN
        assert incore.attrs.get("slot_num") == 16

    def test_for_spmd_cross_core_slot_roundtrips(self):
        """``slot_num`` survives a print -> reparse cycle on the for-spmd form.

        The two entries print as independent list elements, so the split mode
        and the slot count each round-trip on their own.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(
                    4,
                    optimizations=[
                        pl.split(pl.SplitMode.LEFT_RIGHT),
                        pl.cross_core_slot(slot_num=12),
                    ],
                ):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        printed = Prog.as_python()
        assert "optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT), pl.cross_core_slot(slot_num=12)]" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_for_spmd_cross_core_slot_alone_roundtrips(self):
        """A bare slot count needs no split mode at all on the for-spmd form.

        The printer emits only the ``pl.cross_core_slot`` entry — it must not
        fabricate a ``pl.split(pl.SplitMode.NONE)``, which the parser rejects on a
        scope holding ``pl.split_aiv`` regions.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=8)]):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "optimizations=[pl.cross_core_slot(slot_num=8)]" in printed
        assert "pl.split" not in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_deprecated_split_slot_num_prints_as_cross_core_slot(self):
        """The deprecated ``pl.split(mode, slot_num=N)`` spelling still parses,
        warns, and normalises to the dedicated entry when printed back.

        With ``SplitMode.NONE`` the mode carries no information, so the printed
        form drops it entirely and keeps only the slot count.
        """
        with pytest.warns(DeprecationWarning, match="pl.cross_core_slot"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    for i in pl.spmd(4, optimizations=[pl.split(pl.SplitMode.NONE, slot_num=8)]):
                        offset = i * 128
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                        out = pl.store(t, [offset, 0], out)
                    return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "optimizations=[pl.cross_core_slot(slot_num=8)]" in printed
        assert "pl.split" not in printed
        # Reparsing the normalised form is a fixpoint (split_ settles to None).
        assert parse_program(printed).as_python() == printed

    def test_at_incore_cross_core_slot_roundtrips(self):
        """``slot_num`` survives a print -> reparse cycle on the pl.at form."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[
                        pl.split(pl.SplitMode.UP_DOWN),
                        pl.cross_core_slot(slot_num=16),
                    ],
                ):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(Prog.functions.values())[0]
        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN
        assert incore.attrs.get("slot_num") == 16
        printed = Prog.as_python()
        assert "optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=16)]" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_at_incore_cross_core_slot_alone_roundtrips(self):
        """A bare slot count on the pl.at form needs no split mode.

        This is the shape a manual ``pl.split_aiv`` kernel uses to pin a custom
        ring depth without tripping the mutual-exclusion guard in
        ``OutlineIncoreScopes``.
        """

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.cross_core_slot(slot_num=8)],
                ):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(Prog.functions.values())[0]
        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "optimizations=[pl.cross_core_slot(slot_num=8)]" in printed
        assert "pl.split" not in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_cross_core_slot_must_be_positive(self):
        """A non-positive ``slot_num`` literal is rejected."""
        src = (
            "import pypto.language as pl\n\n"
            "@pl.program\n"
            "class P:\n"
            "    @pl.function\n"
            "    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:\n"
            "        with pl.at(level=pl.Level.CORE_GROUP, "
            "optimizations=[pl.cross_core_slot(slot_num=0)]):\n"
            "            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)\n"
            "        return y\n"
        )
        with pytest.raises(ParserSyntaxError, match="must be positive"):
            parse_program(src)

    def test_split_slot_num_must_be_positive(self):
        """A non-positive ``slot_num`` literal is rejected on the deprecated
        ``pl.split(slot_num=)`` spelling too — validation runs before the
        deprecation warning, so the error wins."""
        src = (
            "import pypto.language as pl\n\n"
            "@pl.program\n"
            "class P:\n"
            "    @pl.function\n"
            "    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:\n"
            "        with pl.at(level=pl.Level.CORE_GROUP, "
            "optimizations=[pl.split(pl.SplitMode.UP_DOWN, slot_num=0)]):\n"
            "            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)\n"
            "        return y\n"
        )
        with pytest.raises(ParserSyntaxError, match="must be positive"):
            parse_program(src)

    def test_split_rejects_unknown_kwarg(self):
        """``pl.split`` rejects keywords other than ``slot_num``."""
        src = (
            "import pypto.language as pl\n\n"
            "@pl.program\n"
            "class P:\n"
            "    @pl.function\n"
            "    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:\n"
            "        with pl.at(level=pl.Level.CORE_GROUP, "
            "optimizations=[pl.split(pl.SplitMode.UP_DOWN, foo=1)]):\n"
            "            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)\n"
            "        return y\n"
        )
        with pytest.raises(ParserSyntaxError, match="Unknown keyword argument 'foo'"):
            parse_program(src)

    def test_with_spmd_split_wraps_call_in_incore(self):
        """``with pl.spmd(N, optimizations=[pl.split(mode)]):`` wraps the
        single call in an ``InCoreScopeStmt(split_=mode)`` under the spmd."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN

    def test_with_spmd_cross_core_slot_alone_wraps_call_in_incore(self):
        """``with pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)]):``
        wraps the single call in an ``InCoreScopeStmt`` carrying ``slot_num``.

        Regression: the wrapper-less direct-dispatch fast path keys on "no split
        mode". The slot count lands on the *InCore* scope (OutlineIncoreScopes
        reads ``slot_num`` off ``InCoreScopeStmt`` only), so a slot-count-only
        scope must still get the wrapper or the value is silently discarded and
        the kernel falls back to the default ring depth.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=4)]):
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 4
        # No round-trip assertion here: the plain with-form's printer emits the
        # wrapper as a nested `with pl.at(...)`, which no longer reparses as a
        # single call. That gap is pre-existing and identical for
        # optimizations=[pl.split(MODE)] (see
        # test_with_spmd_split_wraps_call_in_incore, which likewise asserts only
        # the IR shape). The `as tid` form does round-trip — covered below.

    def test_with_spmd_as_tid_cross_core_slot_alone_wraps_call_in_incore(self):
        """Same regression on the ``as tid`` with-form, which shares
        ``_emit_spmd_body``'s dispatch."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=8)]) as tid:
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "optimizations=[pl.cross_core_slot(slot_num=8)]" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_with_spmd_split_splits_name_hint(self):
        """``with pl.spmd(..., name_hint=, optimizations=[pl.split]):`` routes hints like for-form."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(
                    4,
                    name_hint="my_kernel",
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                ):
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert spmd.name_hint == "my_kernel_spmd"
        assert incore.name_hint == "my_kernel"
        assert incore.split == ir.SplitMode.UP_DOWN

    def test_with_spmd_no_optimizations_preserves_ir_shape(self):
        """Regression: omitting optimizations keeps the historical IR shape
        (no implicit InCore wrapper around the single call)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        found_incore = []

        def walk(n):
            if isinstance(n, ir.InCoreScopeStmt):
                found_incore.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(spmd.body)
        assert not found_incore, "with-form without optimizations must not insert an InCoreScopeStmt"

    def test_spmd_rejects_unknown_optimization_entry(self):
        """Entries other than ``pl.split(...)`` are rejected."""
        with pytest.raises(ParserSyntaxError, match="Unsupported entry"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.range]):  # type: ignore[list-item]
                    _ = i
                return a

    def test_spmd_rejects_duplicate_split(self):
        """Duplicate ``pl.split(...)`` in the list is rejected."""
        with pytest.raises(ParserSyntaxError, match=r"Duplicate 'pl\.split"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(
                    4,
                    optimizations=[
                        pl.split(pl.SplitMode.UP_DOWN),
                        pl.split(pl.SplitMode.LEFT_RIGHT),
                    ],
                ):
                    _ = i
                return a

    def test_spmd_rejects_non_list_optimizations(self):
        """``optimizations=`` must be a list literal."""
        with pytest.raises(ParserSyntaxError, match="must be a list literal"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, optimizations=pl.split(pl.SplitMode.NONE)):  # type: ignore[arg-type]
                    _ = i
                return a

    def test_spmd_non_list_optimizations_error_names_api(self):
        """Invalid ``pl.spmd`` optimizations errors mention ``pl.spmd``, not ``pl.at``."""
        with pytest.raises(ParserSyntaxError, match=r"pl\.spmd\(optimizations"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, optimizations=pl.split(pl.SplitMode.NONE)):  # type: ignore[arg-type]
                    _ = i
                return a

    def test_spmd_unsupported_entry_error_names_api(self):
        """Unknown ``pl.spmd`` optimization entries mention ``pl.spmd``."""
        with pytest.raises(ParserSyntaxError, match=r"Unsupported entry in pl\.spmd"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, optimizations=[42]):  # type: ignore[list-item]
                    _ = i
                return a


class TestSpmdScopeTaskId:
    """Test ``with pl.spmd(...) as tid:`` — capturing the grid dispatch's producer TaskId.

    Mirrors ``with pl.at(...) as tid:``: the parser allocates a fresh
    ``Scalar[TASK_ID]`` Var, records it as the ``task_id_var`` attr on the
    ``SpmdScopeStmt``, and emits a transient ``AssignStmt(tid,
    system.task_invalid())`` placeholder before the scope (for ConvertToSSA).
    Unlike the plain ``with pl.spmd(...):`` form, the ``as tid`` form accepts an
    inline multi-statement body (auto-outlined into an InCore kernel), so the
    per-block index is read via ``pl.tile.get_block_idx()``.
    """

    @staticmethod
    def _top_level_stmts(func):
        body = func.body
        return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]

    @staticmethod
    def _is_task_invalid_placeholder(stmt):
        return (
            isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == _OP_SYSTEM_TASK_INVALID
        )

    def _build(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid:
                    i = pl.tile.get_block_idx()
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [offset, 0], out)
                return out

        return list(Prog.functions.values())[0]

    def test_as_tid_sets_task_id_var_on_spmd_scope(self):
        """``as tid`` records a Scalar[TASK_ID] Var as the SpmdScopeStmt task_id_var attr."""
        main_func = self._build()
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert "task_id_var" in spmd.attrs
        tid_var = spmd.attrs["task_id_var"]
        assert isinstance(tid_var, ir.Var)
        assert tid_var.name_hint == "tid"
        # The inline body is auto-wrapped in an InCoreScopeStmt (like the for-form).
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore is not None

    def test_as_tid_emits_task_invalid_placeholder_before_scope(self):
        """A transient ``AssignStmt(tid, system.task_invalid())`` precedes the scope."""
        main_func = self._build()
        stmts = self._top_level_stmts(main_func)
        spmd_idx = next(i for i, s in enumerate(stmts) if isinstance(s, ir.SpmdScopeStmt))
        assert spmd_idx > 0, "expected a placeholder statement before the SpmdScopeStmt"
        placeholder = stmts[spmd_idx - 1]
        spmd_scope = stmts[spmd_idx]
        assert self._is_task_invalid_placeholder(placeholder)
        assert isinstance(placeholder, ir.AssignStmt)
        assert isinstance(spmd_scope, ir.SpmdScopeStmt)
        # The placeholder defines the SAME Var carried by the scope's task_id_var attr.
        assert placeholder.var is spmd_scope.attrs["task_id_var"]

    def test_as_tid_accepts_inline_multi_statement_body(self):
        """The ``as tid`` form lifts the single-call guard — inline ops are allowed."""
        main_func = self._build()  # body has get_block_idx + load + add + store
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        body = incore.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        # User-written get_block_idx is the first body stmt (NOT a synthesized loop var).
        first = stmts[0]
        assert isinstance(first, ir.AssignStmt)
        assert isinstance(first.value, ir.Call) and first.value.op.name == _OP_TILE_GET_BLOCK_IDX
        assert len(stmts) > 1, "inline body should carry multiple statements"

    def test_as_tid_deps_sets_manual_dep_edges(self):
        """``with pl.spmd(n, deps=[tid0]) as tid1:`` records manual_dep_edges referencing tid0."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid0:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                with pl.spmd(4, name_hint="stage2", deps=[tid0]) as tid1:
                    j = pl.tile.get_block_idx()
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(out, [j * 128, 0], [128, 128])
                    out = pl.store(pl.add(u, u), [j * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmds = _descendants(main_func.body, ir.SpmdScopeStmt)
        assert len(spmds) == 2
        first_tid = spmds[0].attrs["task_id_var"]
        edges = spmds[1].attrs["manual_dep_edges"]
        assert isinstance(edges, (list, tuple)) and len(edges) == 1
        assert edges[0] is first_tid, "deps=[tid0] must reference the first scope's task_id_var"

    def test_as_tid_split_optimizations_on_inner_incore(self):
        """``optimizations=[pl.split(...)]`` sets split_ on the inner InCore wrapper."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]) as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN

    def test_plain_with_spmd_has_no_task_id_var(self):
        """Regression: the plain ``with pl.spmd(n):`` single-call form carries no tid attr."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, out)
                return out

        main_func = list(Prog.functions.values())[-1]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert "task_id_var" not in spmd.attrs
        assert "manual_dep_edges" not in spmd.attrs

    def test_as_tid_round_trip(self):
        """``with pl.spmd(...) as tid:`` survives print -> parse round-trip."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        printed = Original.as_python()
        assert ".spmd(" in printed and " as tid:" in printed
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Original, Reparsed)

    def test_as_tid_deps_round_trip(self):
        """``deps=[tid0]`` on a captured spmd survives print -> parse round-trip."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid0:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                with pl.spmd(4, name_hint="stage2", deps=[tid0]) as tid1:
                    j = pl.tile.get_block_idx()
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(out, [j * 128, 0], [128, 128])
                    out = pl.store(pl.add(u, u), [j * 128, 0], out)
                return out

        printed = Original.as_python()
        assert "deps=[tid0]" in printed
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Original, Reparsed)

    # ── Rejections ──────────────────────────────────────────────────────────

    def test_deps_without_tid_rejected(self):
        """``deps=`` on the plain ``with pl.spmd(n):`` form (no ``as tid``) is rejected."""
        with pytest.raises(ParserSyntaxError, match="does not accept 'deps='"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.spmd(4, name_hint="s1") as tid0:
                        i = pl.tile.get_block_idx()
                        out = pl.store(pl.load(a, [i * 128, 0], [128, 128]), [i * 128, 0], out)
                    with pl.spmd(4, deps=[tid0]):  # type: ignore[call-arg]  # deps without `as tid`
                        j = pl.tile.get_block_idx()
                        out = pl.store(pl.load(out, [j * 128, 0], [128, 128]), [j * 128, 0], out)
                    return out

    def test_empty_deps_without_tid_rejected(self):
        """``deps=[]`` (empty / normalized to []) without ``as tid`` is rejected too.

        Gating is by keyword *presence* (allow_deps=optional_vars is not None), not by
        the resolved dep list being non-empty — so even an empty/None-only ``deps=``
        on the non-capturing with-form surfaces a clear error rather than silently
        passing.
        """
        with pytest.raises(ParserSyntaxError, match="does not accept 'deps='"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    out = pl.store(pl.load(a, [0, 0], [512, 128]), [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.spmd(4, deps=[]):  # type: ignore[call-arg]  # empty deps, no `as tid`
                        out = self.kernel(a, out)
                    return out

    def test_for_spmd_deps_rejected(self):
        """The for-form does not accept ``deps=`` — steer to the ``as tid`` with-form."""
        with pytest.raises(ParserSyntaxError, match="does not accept 'deps='"):

            @pl.function
            def bad(a: pl.Tensor[[512, 128], pl.FP32]) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, deps=[]):  # type: ignore[call-arg]
                    _ = i
                return a

    def test_as_tid_tuple_target_rejected(self):
        """The ``as`` target must be a plain name, not a tuple."""
        with pytest.raises(ParserSyntaxError, match="must be a plain variable name"):

            @pl.function
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4) as (x, y):  # type: ignore[misc]
                    i = pl.tile.get_block_idx()
                    out = pl.store(pl.load(a, [i * 128, 0], [128, 128]), [i * 128, 0], out)
                return out

    def test_as_tid_nested_in_cluster_rejected(self):
        """A captured spmd cannot nest inside pl.cluster() (it is unwrapped, losing the tid)."""
        with pytest.raises(ParserSyntaxError, match="cannot capture a TaskId when nested"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.cluster():
                        with pl.spmd(4) as tid:
                            i = pl.tile.get_block_idx()
                            out = pl.store(pl.load(a, [i * 128, 0], [128, 128]), [i * 128, 0], out)
                    return out

    def test_other_scope_as_tid_still_rejected(self):
        """``as`` on a non-at/non-spmd scope is still rejected (mentions both supported forms)."""
        with pytest.raises(ParserSyntaxError, match="only applies to"):

            @pl.function
            def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster() as tid:  # type: ignore[misc]
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y


class TestSpmdInlineWithForm:
    """``with pl.spmd(n):`` (no ``as tid``) with an inline multi-statement body.

    Decouples inline-body support from TaskId capture: the plain with-form now
    auto-outlines an inline body into a synthetic InCore kernel — exactly like the
    ``as tid`` form and the for-form — WITHOUT capturing a producer TaskId. The two
    concerns are orthogonal (TaskId capture is opt-in via ``as tid``), but an inline
    body must still read the per-block index via ``pl.tile.get_block_idx()``.
    """

    def test_inline_with_spmd_no_tid_wraps_incore(self):
        """An inline body (no ``as tid``) is auto-outlined into an InCore wrapper and
        carries NO task_id_var / manual_dep_edges — the TaskId is not captured."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1"):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        # No TaskId captured — the decoupled feature under test.
        assert "task_id_var" not in spmd.attrs
        assert "manual_dep_edges" not in spmd.attrs
        # The inline body is wrapped in an InCoreScopeStmt for outlining (like the
        # for-form / as-tid form), not left as a bare Call.
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        body = incore.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        # The user-written get_block_idx is the first body stmt (NOT synthesized).
        first = stmts[0]
        assert isinstance(first, ir.AssignStmt)
        assert isinstance(first.value, ir.Call)
        assert first.value.op.name == _OP_TILE_GET_BLOCK_IDX
        assert len(stmts) > 1, "inline body should carry multiple statements"

    def test_inline_with_spmd_no_placeholder_before_scope(self):
        """Unlike the ``as tid`` form, the plain inline form emits NO
        ``AssignStmt(tid, system.task_invalid())`` placeholder before the scope."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        placeholders = [
            s
            for s in _descendants(main_func.body, ir.AssignStmt)
            if isinstance(s.value, ir.Call) and s.value.op.name == _OP_SYSTEM_TASK_INVALID
        ]
        assert not placeholders, "plain inline form must not emit a task_invalid placeholder"
        # ...but the scope itself must still be there — an empty body would also
        # trivially have no placeholders.
        assert _descendants(main_func.body, ir.SpmdScopeStmt)

    def test_inline_with_spmd_split_wraps_incore_with_split(self):
        """``optimizations=[pl.split(...)]`` on the inline plain form sets split_ on the
        inner InCore wrapper (same as the for-form / as-tid form)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert "task_id_var" not in spmd.attrs
        incore = _unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN

    def test_inline_with_spmd_round_trip(self):
        """The inline plain form survives print -> parse round-trip (no ``as tid``)."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1"):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        printed = Original.as_python()
        assert ".spmd(" in printed and " as tid:" not in printed
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Original, Reparsed)

    def test_inline_with_spmd_missing_block_idx_rejected(self):
        """An inline body that never reads the per-block index is rejected — without
        ``get_block_idx()`` every block runs identical work, so it is almost always a
        bug. A body that dispatches a ``self.<kernel>(...)`` call is exempt (the callee
        reads the index internally) — see the regression test
        ``test_with_spmd_single_call_still_supported``."""
        with pytest.raises(ParserSyntaxError, match="neither reads the per-block index"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.spmd(4):
                        # No pl.tile.get_block_idx() anywhere — every block would run
                        # identical work writing the same output region.
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                        out = pl.store(pl.add(t, t), [0, 0], out)
                    return out

    def test_inline_as_tid_missing_block_idx_rejected(self):
        """The same block-index requirement applies to the ``as tid`` inline form —
        the check lives in the shared body-emit path, so both with-forms enforce it."""
        with pytest.raises(ParserSyntaxError, match="neither reads the per-block index"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.spmd(4) as tid:  # noqa: F841
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                        out = pl.store(pl.add(t, t), [0, 0], out)
                    return out

    def test_inline_with_spmd_accepts_top_level_get_block_idx(self):
        """Regression (qwen3 decode / pypto-lib-model CI): an inline body that reads
        the block index via the top-level ``pl.get_block_idx()`` alias (not the
        qualified ``pl.tile.get_block_idx()``) is accepted, not rejected by the guard."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="fa_fused"):
                    i = pl.get_block_idx()  # top-level alias, as real models use
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = _unique_descendant(main_func.body, ir.SpmdScopeStmt)
        # Outlined (InCore wrapper present), not rejected by the block-index guard.
        _unique_descendant(spmd.body, ir.InCoreScopeStmt)

    def test_block_idx_guard_matches_every_get_block_idx_spelling(self):
        """The block-index guard matches ``get_block_idx()`` by name, across every
        valid spelling regardless of receiver. Regression: the top-level
        ``pl.get_block_idx()`` alias (used by real models, e.g. qwen3 decode) must be
        accepted — a receiver-restricted match wrongly rejected it."""
        reads = ASTParser._spmd_body_reads_block_idx
        # Top-level alias (the regression case), qualified forms, and bare import.
        assert reads(ast.parse("x = pl.get_block_idx()").body)
        assert reads(ast.parse("x = pl.tile.get_block_idx()").body)
        assert reads(ast.parse("x = tile.get_block_idx()").body)
        assert reads(ast.parse("x = get_block_idx()").body)
        # A nested use (inside an expression argument) still counts.
        assert reads(ast.parse("t = pl.load(a, [pl.get_block_idx() * 8, 0], [8, 8])").body)
        # A body with no block-index read at all is rejected.
        assert not reads(ast.parse("x = pl.load(a, [0, 0], [8, 8])").body)
        assert not reads(ast.parse("x = foo.get_subblock_idx()").body)


class TestSpmdAllowEarlyResolve:
    """``pl.spmd(..., allow_early_resolve=True)`` — speculative early-dispatch hint.

    Mirrors ``pl.submit(..., allow_early_resolve=True)`` / ``pl.at(...,
    allow_early_resolve=True)``: the flag is recorded as an ``allow_early_resolve``
    attr on the ``SpmdScopeStmt`` and the Spmd outliner threads it onto the
    synthesised ``ir.Submit`` (proven in ``test_outline_cluster_scopes.py``).
    Accepted on all three dispatch forms (plain with-form, ``as tid`` with-form,
    and the ``for`` loop form); rejected on a ``pl.cluster()``-nested ``pl.spmd``.
    """

    @staticmethod
    def _spmd_scopes(node):
        found = []

        def walk(n):
            if isinstance(n, ir.SpmdScopeStmt):
                found.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(node)
        return found

    def _unique_spmd(self, prog):
        main_func = list(prog.functions.values())[-1]
        scopes = self._spmd_scopes(main_func.body)
        assert len(scopes) == 1, f"expected exactly one SpmdScopeStmt, got {len(scopes)}"
        return scopes[0]

    def test_dsl_forwards_flag_onto_context(self):
        """pl.spmd(..., allow_early_resolve=True) reaches SpmdContext (kwarg-forwarding guard)."""
        assert pl.spmd(4, allow_early_resolve=True).allow_early_resolve is True
        assert pl.spmd(4).allow_early_resolve is False

    def test_as_tid_records_flag(self):
        """``with pl.spmd(n, allow_early_resolve=True) as tid:`` records the scope attr."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1", allow_early_resolve=True) as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        spmd = self._unique_spmd(Prog)
        assert spmd.attrs.get("allow_early_resolve") is True
        # Coexists with the captured producer TaskId.
        assert "task_id_var" in spmd.attrs

    def test_for_form_records_flag(self):
        """``for i in pl.spmd(n, allow_early_resolve=True):`` records the scope attr."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, allow_early_resolve=True):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        spmd = self._unique_spmd(Prog)
        assert spmd.attrs.get("allow_early_resolve") is True

    def test_plain_with_form_records_flag(self):
        """``with pl.spmd(n, allow_early_resolve=True):`` (single call) records the attr."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, allow_early_resolve=True):
                    out = self.kernel(a, out)
                return out

        spmd = self._unique_spmd(Prog)
        assert spmd.attrs.get("allow_early_resolve") is True
        # No `as tid`, so no captured producer TaskId — the outliner synthesises one.
        assert "task_id_var" not in spmd.attrs

    def test_default_false_omitted_from_attrs(self):
        """Omitting the kwarg leaves no allow_early_resolve attr on the scope."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                return out

        spmd = self._unique_spmd(Prog)
        assert "allow_early_resolve" not in spmd.attrs

    def test_as_tid_round_trip(self):
        """The ``as tid`` form survives print -> reparse with the flag preserved."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1", allow_early_resolve=True) as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        printed = Original.as_python()
        assert "allow_early_resolve=True" in printed
        ir.assert_structural_equal(Original, parse_program(printed))

    def test_for_form_round_trip(self):
        """The for-loop form survives print -> reparse with the flag preserved."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, allow_early_resolve=True):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        printed = Original.as_python()
        assert "allow_early_resolve=True" in printed
        ir.assert_structural_equal(Original, parse_program(printed))

    def test_plain_with_form_round_trip(self):
        """The plain single-call with-form survives print -> reparse with the flag preserved."""

        @pl.program
        class Original:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, allow_early_resolve=True):
                    out = self.kernel(a, out)
                return out

        printed = Original.as_python()
        assert "allow_early_resolve=True" in printed
        ir.assert_structural_equal(Original, parse_program(printed))

    def test_default_omitted_from_print(self):
        """A scope without the hint never prints ``allow_early_resolve``."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                return out

        assert "allow_early_resolve" not in Prog.as_python()

    def test_cluster_nested_plain_with_form_rejected(self):
        """``allow_early_resolve=True`` on a cluster-nested plain ``pl.spmd`` is rejected."""
        with pytest.raises(ParserSyntaxError, match="cannot be nested inside"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    t = pl.load(a, [0, 0], [512, 128])
                    out = pl.store(t, [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.cluster():
                        with pl.spmd(4, allow_early_resolve=True):  # type: ignore[call-arg]
                            out = self.kernel(a, out)
                    return out

    def test_cluster_nested_for_form_rejected(self):
        """``allow_early_resolve=True`` on a cluster-nested ``for ... in pl.spmd`` is rejected."""
        with pytest.raises(ParserSyntaxError, match="cannot be nested inside"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.cluster():
                        for i in pl.spmd(4, allow_early_resolve=True):  # type: ignore[call-arg]
                            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                            out = pl.store(t, [i * 128, 0], out)
                    return out

    def test_cluster_nested_as_tid_form_rejected(self):
        """A cluster-nested ``as tid`` form with the hint is rejected.

        The ``as tid`` capture is already illegal inside ``pl.cluster()`` (the
        scope is unwrapped into the Group function and produces no Submit), so
        the as-tid cluster guard fires first regardless of ``allow_early_resolve``
        — the combination is never silently accepted.
        """
        with pytest.raises(ParserSyntaxError, match="nested inside"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.cluster():
                        with pl.spmd(4, allow_early_resolve=True) as tid:  # type: ignore[call-arg]
                            i = pl.tile.get_block_idx()
                            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                            out = pl.store(t, [i * 128, 0], out)
                    return out

    def test_non_bool_literal_rejected_for_form(self):
        """A non-bool ``allow_early_resolve`` literal is rejected at parse time (for-form)."""
        with pytest.raises(ParserSyntaxError, match="allow_early_resolve must be a boolean literal"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    for i in pl.spmd(4, allow_early_resolve=1):  # type: ignore[arg-type]
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                        out = pl.store(t, [i * 128, 0], out)
                    return out

    def test_non_bool_literal_rejected_as_tid_form(self):
        """A non-bool ``allow_early_resolve`` literal is rejected at parse time (as-tid form)."""
        with pytest.raises(ParserSyntaxError, match="allow_early_resolve must be a boolean literal"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.spmd(4, allow_early_resolve=1) as tid:  # type: ignore[arg-type]
                        i = pl.tile.get_block_idx()
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                        out = pl.store(t, [i * 128, 0], out)
                    return out


class TestSpmdDispatchBodyRoundTrip:
    """Round-trip coverage for ``with pl.spmd(...)`` bodies that are not a lone call.

    The body-shape test in ``_emit_spmd_body`` is semantic — does the body do
    per-block work itself (reads ``get_block_idx``), or dispatch it to a kernel?
    — rather than a statement count. A *dispatch* body may legally hold several
    statements: ``FlattenCallExpr`` hoists a nested call arg beside the call, and
    the multi-output tuple desugar splits one assign into a temp plus projections.
    Each shape below is emitted by the printer and must re-parse to the same IR.
    """

    @staticmethod
    def _incore_count(prog, func_name="main"):
        """Number of InCore scopes in ``func_name`` — 0 unwrapped, 1 carrier, 2+ double-wrapped."""
        func = prog.get_function(func_name)
        assert func is not None, f"function {func_name!r} not found in program"
        return len(_descendants(func.body, ir.InCoreScopeStmt))

    @staticmethod
    def _assert_round_trips(prog):
        """Print ``prog``, re-parse, and require structural equality."""
        printed = ir.python_print(prog)
        reparsed = parse_program(printed)
        ir.assert_structural_equal(reparsed, prog)
        return printed, reparsed

    def test_single_call_with_split_round_trips(self):
        """``with pl.spmd(N, optimizations=[...]):`` around a single call round-trips.

        The parser wraps the call in an ``InCoreScopeStmt`` carrier so the split has
        somewhere to live; the printer spells that carrier as a nested
        ``pl.at(level=pl.Level.CORE_GROUP, optimizations=[...])``. Re-parsing must
        recognise it as the carrier instead of rejecting it.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                    out = self.kernel(a, out)
                return out

        printed, reparsed = self._assert_round_trips(Prog)
        assert "pl.split(pl.SplitMode.UP_DOWN)" in printed
        assert self._incore_count(reparsed) == 1

    def test_repeated_round_trips_do_not_accumulate_carriers(self):
        """Print -> parse is idempotent for a body whose carrier only appears once printed.

        The source below has no explicit ``pl.at``; the parser synthesises the
        carrier and the printer then spells it out. Re-parsing must recognise that
        as the carrier rather than synthesising a second ``InCoreScopeStmt`` around
        it — which would violate ``NoNestedInCore`` and add one level per cycle, so
        a single cycle would not reveal it.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, a)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, out)
                    i = pl.tile.get_block_idx()  # noqa: F841 — not the first statement
                return out

        # Three cycles: a double-wrap bug compounds, so a single cycle can hide it.
        printed = ir.python_print(Prog)
        for _ in range(3):
            reparsed = parse_program(printed)
            printed = ir.python_print(reparsed)
        final = parse_program(printed)
        ir.assert_structural_equal(final, Prog)
        assert self._incore_count(final) == 1

    def test_flattened_dispatch_body_round_trips(self):
        """A dispatch body with a temp hoisted beside the call round-trips.

        ``FlattenCallExpr`` extracts a nested call arg into a temporary and keeps it
        *inside* the scope (preserving the execution-context boundary, same as
        ``pl.at`` / ``pl.cluster``), leaving the spmd body with two statements.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def worker(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(x, x)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(4):
                    out = self.worker(pl.add(x, 1.0), out)
                return out

        flattened = passes.flatten_call_expr()(Prog)
        printed, reparsed = self._assert_round_trips(flattened)
        assert "t__tmp_v0" in printed
        assert self._incore_count(reparsed) == 0

    def test_multi_output_dispatch_body_round_trips(self):
        """A multi-output dispatch (temp + tuple projections) round-trips.

        ``o0, o1 = self.kernel(...)`` desugars into three IR statements; the printer
        emits them verbatim, so the parser must accept that shape back.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64], pl.FP32],
                o0: pl.Out[pl.Tensor[[64], pl.FP32]],
                o1: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    o0 = pl.add(a, a)
                    o1 = pl.mul(a, a)
                return o0, o1

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64], pl.FP32],
                o0: pl.Out[pl.Tensor[[64], pl.FP32]],
                o1: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                with pl.spmd(4):
                    o0, o1 = self.kernel(a, o0, o1)
                return o0, o1

        printed, reparsed = self._assert_round_trips(Prog)
        assert "_tuple_tmp[0]" in printed and "_tuple_tmp[1]" in printed
        assert self._incore_count(reparsed) == 0

    def test_optimizations_specified_on_both_scopes_rejected(self):
        """``optimizations=`` on both the spmd and its explicit carrier is rejected.

        Either entry — a split mode or a ``cross_core_slot`` count — lands on the
        InCore scope, so specifying one on each scope is ambiguous.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.add(a, a)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            with pl.at(level=pl.Level.CORE_GROUP,
                       optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
                out = self.kernel(a, out)
        return out
"""
        with pytest.raises(ParserSyntaxError, match="`optimizations=` is specified twice"):
            parse_program(src)

    def test_multiple_kernel_dispatches_rejected(self):
        """An unwrapped dispatch body may launch only one kernel.

        ``FindFirstInnerCall`` stops at the first call, so orchestration codegen
        would emit a launch for that callee and silently drop the rest.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.add(a, a)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             b: pl.Out[pl.Tensor[[64], pl.FP32]],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4):
            b = self.kernel(a, b)
            out = self.kernel(a, out)
        return out
"""
        with pytest.raises(ParserSyntaxError, match="launches 2 kernels"):
            parse_program(src)

    def test_optimizations_on_outer_scope_only_with_carrier_rejected(self):
        """``optimizations=`` on the spmd alone, with a bare carrier body, is rejected.

        The entry has to land on the InCore scope, and the body already provides one
        by hand — pushing the outer setting into it would be silent, and dropping it
        would be worse. The message distinguishes this from the both-scopes case.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.add(a, a)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = self.kernel(a, out)
        return out
"""
        with pytest.raises(ParserSyntaxError, match="belongs on the InCore scope"):
            parse_program(src)

    def test_cross_core_slot_alone_forces_carrier_on_dispatch_body(self):
        """A bare ``cross_core_slot`` count still requires an InCore carrier.

        ``slot_num`` lands on the InCore scope (``OutlineIncoreScopes`` reads it off
        ``InCoreScopeStmt`` only), so a dispatch body carrying one must be wrapped
        even though it does not read the per-block index — otherwise the count is
        silently discarded.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.add(a, a)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=8)]):
            out = self.kernel(a, out)
        return out
"""
        prog = parse_program(src)
        assert self._incore_count(prog) == 1
        main_func = prog.get_function("main")
        assert main_func is not None
        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.attrs.get("slot_num") == 8
        self._assert_round_trips(prog)

    @pytest.mark.parametrize(
        ("label", "at_args", "capture"),
        [
            ("keyword level", "level=pl.Level.CORE_GROUP", ""),
            ("positional level", "pl.Level.CORE_GROUP", ""),
            ("with a TaskId capture", "level=pl.Level.CORE_GROUP", " as tid"),
            ("with a name hint", 'level=pl.Level.CORE_GROUP, name_hint="k"', ""),
        ],
    )
    def test_carrier_recognised_across_spellings(self, label, at_args, capture):
        """Every spelling of the carrier is recognised — none double-wraps.

        The carrier level is resolved through ``_parse_at_kwargs``, so the
        positional form counts too, and an ``as tid`` capture or a ``name_hint=``
        rides on the nested scope without disqualifying it. Missing any of these
        would synthesise a second ``InCoreScopeStmt`` around one that already
        exists — a ``NoNestedInCore`` violation. The body reads the per-block
        index, which is what makes the double-wrap reachable.
        """
        src = f"""
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[512, 128], pl.FP32],
             out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(4):
            with pl.at({at_args}){capture}:
                i = pl.tile.get_block_idx()
                tl: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(tl, [i * 128, 0], out)
        return out
"""
        prog = parse_program(src)
        assert self._incore_count(prog) == 1, f"{label} double-wrapped the InCore carrier"

    def test_non_core_group_scope_is_not_a_carrier(self):
        """A nested non-CORE_GROUP ``pl.at`` builds a Hierarchy scope, not a carrier.

        It must NOT suppress the InCore wrapper the inline body still needs.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[512, 128], pl.FP32],
             out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(4):
            with pl.at(level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker):
                i = pl.tile.get_block_idx()
                tl: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(tl, [i * 128, 0], out)
        return out
"""
        prog = parse_program(src)
        main_func = prog.get_function("main")
        assert main_func is not None
        assert self._incore_count(prog) == 1
        assert len(_descendants(main_func.body, ir.HierarchyScopeStmt)) == 1

    def test_bare_name_external_kernel_dispatch_accepted(self):
        """A bare-name call to an external ``@pl.function`` is a dispatch body.

        ``parse_call`` accepts both ``self.<kernel>(...)`` and a bare name resolving
        through ``closure_vars`` to an external ``@pl.function`` / ``@pl.inline``, so
        the dispatch-body test must accept both — otherwise the bare-name form is
        rejected for failing a per-block-differentiation check its callee satisfies
        internally. Uses the decorator form because the bare name has to resolve
        against a real module closure.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(4):
                    out = _external_worker(a, out)
                return out

        assert self._incore_count(Prog) == 0, "a dispatch body must not be wrapped"
        self._assert_round_trips(Prog)

    def test_as_tid_carrier_round_trips(self):
        """``as tid`` + explicit carrier + dispatch body preserves the carrier.

        The ``as tid`` printer branch normally inlines the InCore's statements and
        hoists its split onto the ``pl.spmd(...)`` line. That sugar is lossless only
        when re-parsing rebuilds the carrier — i.e. the body carries a split or reads
        the per-block index. For a dispatch body with neither, the carrier must be
        printed explicitly or the round-trip drops it.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.add(a, a)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4) as tid:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = self.kernel(a, out)
        return out
"""
        prog = parse_program(src)
        assert self._incore_count(prog) == 1
        printed, reparsed = self._assert_round_trips(prog)
        assert " as tid:" in printed
        assert self._incore_count(reparsed) == 1

    def test_explicit_carrier_exempt_from_block_differentiation_check(self):
        """An explicit carrier opts out of the "every block runs identical work" check.

        The user wrote the InCore scope themselves, so the body is taken as
        deliberate even when it neither reads the per-block index nor dispatches a
        kernel. Pinned so the exemption is a decision rather than an accident.
        """
        src = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.add(a, a)
        return out
"""
        prog = parse_program(src)
        assert self._incore_count(prog) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
