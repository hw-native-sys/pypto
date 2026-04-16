# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ScopeStmt class hierarchy."""

import pypto.language as pl
import pytest
from pypto import DataType, ir


class TestScopeStmt:
    """Test ScopeStmt construction, fields, and operations."""

    def test_hierarchy_scope_construction(self):
        """Test basic HierarchyScopeStmt construction at CORE_GROUP (replaces InCore scope)."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body = ir.AssignStmt(var_y, var_x, span)
        scope = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, body=body, span=span)

        assert scope.scope_kind == ir.ScopeKind.Hierarchy
        assert scope.level == ir.Level.CORE_GROUP
        assert isinstance(scope, ir.ScopeStmt)
        assert isinstance(scope.body, ir.AssignStmt)

    def test_hierarchy_scope_structural_equality(self):
        """Test structural equality for HierarchyScopeStmt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body1 = ir.AssignStmt(var_y, var_x, span)
        scope1 = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, body=body1, span=span)

        body2 = ir.AssignStmt(var_y, var_x, span)
        scope2 = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, body=body2, span=span)

        assert ir.structural_equal(scope1, scope2)

    def test_scope_stmt_printing(self):
        """Test Python printer output for HierarchyScopeStmt at CORE_GROUP."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        printed = TestProgram.as_python()
        assert "with pl.at(level=pl.Level.CORE_GROUP):" in printed

    def test_hierarchy_scope_with_name(self):
        """Test HierarchyScopeStmt construction with a user-provided name."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, name_hint="my_kernel", body=body, span=span)
        assert scope.name_hint == "my_kernel"
        assert scope.scope_kind == ir.ScopeKind.Hierarchy

    def test_hierarchy_scope_default_name_is_empty(self):
        """Test that default name is empty string."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, body=body, span=span)
        assert scope.name_hint == ""

    def test_spmd_scope_requires_positive_core_num(self):
        """SpmdScopeStmt enforces core_num > 0 at construction time."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        with pytest.raises(ValueError, match="core_num"):
            ir.SpmdScopeStmt(core_num=0, body=body, span=span)

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

    def test_hierarchy_scope_split_at_core_group(self):
        """HierarchyScopeStmt accepts split at CORE_GROUP."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        scope = ir.HierarchyScopeStmt(
            level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=body, span=span
        )
        assert scope.split == ir.SplitMode.UP_DOWN

    def test_hierarchy_scope_split_rejected_at_non_core_group(self):
        """HierarchyScopeStmt rejects split at levels other than CORE_GROUP."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
        body = ir.AssignStmt(var_y, var_x, span)

        with pytest.raises(ValueError, match="split is only valid at Level::CORE_GROUP"):
            ir.HierarchyScopeStmt(level=ir.Level.HOST, split=ir.SplitMode.UP_DOWN, body=body, span=span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
