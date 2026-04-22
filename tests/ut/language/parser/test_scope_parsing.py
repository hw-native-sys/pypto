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
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError
from pypto.language.parser.text_parser import parse_program


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
        assert spmd.core_num == 4
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
        assert spmd.core_num == 4

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
        assert spmd.core_num == 8
        assert spmd.sync_start is True
        assert spmd.name_hint == "my_kernel"

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
                i = 0  # noqa: F841
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
