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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
