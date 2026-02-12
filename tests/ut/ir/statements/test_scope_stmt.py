# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ScopeStmt class."""

import pypto.language as pl
from pypto import DataType, ir


class TestScopeStmt:
    """Test ScopeStmt construction, fields, and operations."""

    def test_scope_stmt_construction(self):
        """Test basic ScopeStmt construction."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        # Create a simple assignment as body
        body = ir.AssignStmt(var_y, var_x, span)

        # Create ScopeStmt
        scope = ir.ScopeStmt(ir.ScopeKind.InCore, body, span)

        assert scope.scope_kind == ir.ScopeKind.InCore
        assert isinstance(scope.body, ir.AssignStmt)

    def test_scope_stmt_structural_equality(self):
        """Test structural equality for ScopeStmt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body1 = ir.AssignStmt(var_y, var_x, span)
        scope1 = ir.ScopeStmt(ir.ScopeKind.InCore, body1, span)

        body2 = ir.AssignStmt(var_y, var_x, span)
        scope2 = ir.ScopeStmt(ir.ScopeKind.InCore, body2, span)

        # Should be structurally equal
        assert ir.structural_equal(scope1, scope2)

    def test_scope_stmt_printing(self):
        """Test Python printer output for ScopeStmt."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Print and verify it contains "with pl.incore():"
        printed = ir.python_print(TestProgram)
        assert "with pl.incore():" in printed
