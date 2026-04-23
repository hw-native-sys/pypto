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
import pytest
from pypto import DataType, ir


class TestScopeStmt:
    """Test ScopeStmt construction, fields, and operations."""

    def test_scope_stmt_construction(self):
        """Test basic InCoreScopeStmt construction."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body = ir.AssignStmt(var_y, var_x, span)
        scope = ir.InCoreScopeStmt(body=body, span=span)

        assert scope.scope_kind == ir.ScopeKind.InCore
        assert isinstance(scope, ir.ScopeStmt)
        assert isinstance(scope.body, ir.AssignStmt)

    def test_scope_stmt_structural_equality(self):
        """Test structural equality for InCoreScopeStmt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body1 = ir.AssignStmt(var_y, var_x, span)
        scope1 = ir.InCoreScopeStmt(body=body1, span=span)

        body2 = ir.AssignStmt(var_y, var_x, span)
        scope2 = ir.InCoreScopeStmt(body=body2, span=span)

        assert ir.structural_equal(scope1, scope2)

    def test_scope_stmt_printing(self):
        """Test Python printer output for ScopeStmt."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        printed = TestProgram.as_python()
        assert "with pl.at(level=pl.Level.CORE_GROUP):" in printed

    def test_scope_stmt_with_name(self):
        """Test InCoreScopeStmt construction with a user-provided name."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.InCoreScopeStmt(name_hint="my_kernel", body=body, span=span)
        assert scope.name_hint == "my_kernel"
        assert scope.scope_kind == ir.ScopeKind.InCore

    def test_scope_stmt_default_name_is_empty(self):
        """Test that default name is empty string."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.InCoreScopeStmt(body=body, span=span)
        assert scope.name_hint == ""

    def test_spmd_scope_core_num_is_expr(self):
        """SpmdScopeStmt stores core_num as an IR expression (typically ConstInt)."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        core_num_expr = ir.ConstInt(4, DataType.INDEX, span)
        scope = ir.SpmdScopeStmt(core_num=core_num_expr, body=body, span=span)
        assert isinstance(scope.core_num, ir.ConstInt)
        assert scope.core_num.value == 4
        assert scope.scope_kind == ir.ScopeKind.Spmd

    def test_spmd_scope_core_num_int_overload(self):
        """The nanobind ctor accepts a Python int and auto-wraps it as ConstInt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.SpmdScopeStmt(core_num=8, body=body, span=span)
        assert isinstance(scope.core_num, ir.ConstInt)
        assert scope.core_num.value == 8

    def test_hierarchy_scope_typed_fields(self):
        """HierarchyScopeStmt exposes level (required) and role (optional)."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=body, span=span)
        assert scope.level == ir.Level.HOST
        assert scope.role == ir.Role.Worker
        assert scope.scope_kind == ir.ScopeKind.Hierarchy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
