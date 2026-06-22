# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing ScopeStmt with pl.at(level=pl.Level.CORE_GROUP): syntax."""

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError
from pypto.language.parser.text_parser import parse_program

from pypto import ir


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

    @staticmethod
    def _unique_descendant(node, cls):
        """Return the single descendant of ``node`` that is an instance of ``cls``."""
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
        assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
        return found[0]

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 4
        assert spmd.sync_start is False
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)

        body = incore.body
        first_stmt = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first_stmt, ir.AssignStmt)
        call = first_stmt.value
        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.get_block_idx"
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert isinstance(spmd.core_num, ir.ConstInt)
        assert spmd.core_num.value == 8
        assert spmd.sync_start is True
        assert spmd.name_hint == "my_kernel_spmd"
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert spmd.name_hint == "q_proj_spmd"
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        assert spmd.name_hint == "gate_proj_spmd"
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
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
        """chunk= is a pl.parallel/pl.range kwarg and not valid on pl.spmd."""
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

    Only ``pl.split(mode)`` is supported on ``pl.spmd``; ``pl.auto_chunk`` must
    be used on nested ``pl.at(level=CORE_GROUP, ...)`` instead.
    """

    @staticmethod
    def _unique_descendant(node, cls):
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
        assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
        return found[0]

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN
        body = incore.body
        first_stmt = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first_stmt, ir.AssignStmt)
        call = first_stmt.value
        assert isinstance(call, ir.Call) and isinstance(call.op, ir.Op)
        assert call.op.name == "tile.get_block_idx"

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split is None

    def test_for_spmd_split_slot_num_sets_scope_attr(self):
        """``pl.split(mode, slot_num=N)`` records ``slot_num`` on the inner
        ``InCoreScopeStmt`` attrs alongside the split mode."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN, slot_num=16)]):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(Prog.functions.values())[0]
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN
        assert incore.attrs.get("slot_num") == 16

    def test_for_spmd_split_slot_num_roundtrips(self):
        """``slot_num`` survives a print -> reparse cycle on the for-spmd form."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT, slot_num=12)]):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        printed = Prog.as_python()
        assert "slot_num=12" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_for_spmd_split_none_slot_num_roundtrips(self):
        """``slot_num`` is valid with ``SplitMode.NONE`` on the for-spmd form and
        survives a print -> reparse cycle (NONE mixed kernel still drives a
        cube->vector pipe)."""

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "pl.split(pl.SplitMode.NONE, slot_num=8)" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_at_incore_split_slot_num_roundtrips(self):
        """``slot_num`` survives a print -> reparse cycle on the pl.at form."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN, slot_num=16)],
                ):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(Prog.functions.values())[0]
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.attrs.get("slot_num") == 16
        printed = Prog.as_python()
        assert "slot_num=16" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_split_slot_num_allowed_with_none_mode(self):
        """``slot_num`` is valid with ``SplitMode.NONE``: a NONE mixed kernel
        still drives a cube->vector pipe (a2a3 dual-AIV dispatch), so the
        scope records ``slot_num`` alongside ``split=SplitMode.NONE`` and the
        attr survives a print -> reparse cycle."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.split(pl.SplitMode.NONE, slot_num=8)],
                ):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main_func = list(Prog.functions.values())[0]
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.NONE
        assert incore.attrs.get("slot_num") == 8
        printed = Prog.as_python()
        assert "slot_num=8" in printed
        assert Prog.as_python() == parse_program(printed).as_python()

    def test_split_slot_num_must_be_positive(self):
        """A non-positive ``slot_num`` literal is rejected."""
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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.UP_DOWN

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique_descendant(spmd.body, ir.InCoreScopeStmt)
        assert spmd.name_hint == "my_kernel_spmd"
        assert incore.name_hint == "my_kernel"
        assert incore.split == ir.SplitMode.UP_DOWN

    def test_spmd_rejects_auto_chunk_on_with_form(self):
        """``pl.auto_chunk`` is not supported on ``pl.spmd`` (either form)."""
        with pytest.raises(ParserSyntaxError, match="not supported in pl.spmd"):

            @pl.program
            class _Prog:
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
                    with pl.spmd(4, optimizations=[pl.auto_chunk]):
                        out = self.kernel(a, out)
                    return out

    def test_spmd_rejects_auto_chunk_on_for_form(self):
        with pytest.raises(ParserSyntaxError, match="not supported in pl.spmd"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.spmd(4, optimizations=[pl.auto_chunk]):
                    _ = i
                return a

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
        spmd = self._unique_descendant(main_func.body, ir.SpmdScopeStmt)
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
    def _descendants(node, cls):
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

    @classmethod
    def _unique(cls, node, klass):
        found = cls._descendants(node, klass)
        assert len(found) == 1, f"expected exactly one {klass.__name__}, got {len(found)}"
        return found[0]

    @staticmethod
    def _top_level_stmts(func):
        body = func.body
        return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]

    @staticmethod
    def _is_task_invalid_placeholder(stmt):
        return (
            isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "system.task_invalid"
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
        spmd = self._unique(main_func.body, ir.SpmdScopeStmt)
        assert "task_id_var" in spmd.attrs
        tid_var = spmd.attrs["task_id_var"]
        assert isinstance(tid_var, ir.Var)
        assert tid_var.name_hint == "tid"
        # The inline body is auto-wrapped in an InCoreScopeStmt (like the for-form).
        incore = self._unique(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique(spmd.body, ir.InCoreScopeStmt)
        body = incore.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        # User-written get_block_idx is the first body stmt (NOT a synthesized loop var).
        first = stmts[0]
        assert isinstance(first, ir.AssignStmt)
        assert isinstance(first.value, ir.Call) and first.value.op.name == "tile.get_block_idx"
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
        spmds = self._descendants(main_func.body, ir.SpmdScopeStmt)
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
        spmd = self._unique(main_func.body, ir.SpmdScopeStmt)
        incore = self._unique(spmd.body, ir.InCoreScopeStmt)
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
        spmd = self._unique(main_func.body, ir.SpmdScopeStmt)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
