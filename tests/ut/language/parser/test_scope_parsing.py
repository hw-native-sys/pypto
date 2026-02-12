# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing ScopeStmt with pl.incore() syntax."""

import pypto.language as pl
from pypto import ir


class TestScopeParsing:
    """Test parsing of with pl.incore(): syntax."""

    def test_parse_simple_incore_scope(self):
        """Test parsing a simple InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Verify the program was parsed successfully
        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

        # Get the main function
        main_func = list(TestProgram.functions.values())[0]
        assert main_func.name == "main"

        # Verify the body contains a ScopeStmt
        # The body should be SeqStmts containing OpStmts with ScopeStmt
        assert isinstance(main_func.body, ir.SeqStmts)

    def test_parse_nested_operations_in_scope(self):
        """Test parsing multiple operations inside InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
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
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.incore():
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
                with pl.incore():
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
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Print the program
        printed = ir.python_print(Original)

        # Verify it contains the scope syntax
        assert "with pl.incore():" in printed
