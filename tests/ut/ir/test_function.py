# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Function class."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestFunction:
    """Test Function class."""

    def test_function_creation(self):
        """Test creating a Function instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        assert func is not None
        assert func.span.filename == "test.py"
        assert len(func.params) == 1
        assert len(func.return_types) == 1
        assert func.body is not None

    def test_function_has_attributes(self):
        """Test that Function has params, return_types, and body attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(a, b, span)
        assign2 = ir.AssignStmt(b, ir.ConstInt(0, dtype, span), span)
        body = ir.SeqStmts([assign1, assign2], span)
        func = ir.Function("my_func", [a, b], [ir.ScalarType(dtype)], body, span)

        assert len(func.params) == 2
        assert len(func.return_types) == 1
        assert func.body is not None
        assert cast(ir.Var, func.params[0]).name == "a"
        assert cast(ir.Var, func.params[1]).name == "b"
        assert isinstance(func.return_types[0], ir.ScalarType)
        assert isinstance(func.body, ir.SeqStmts)

    def test_function_is_irnode(self):
        """Test that Function is an instance of IRNode."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        assert isinstance(func, ir.IRNode)

    def test_function_immutability(self):
        """Test that Function attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            func.params = []  # type: ignore
        with pytest.raises(AttributeError):
            func.return_types = []  # type: ignore
        with pytest.raises(AttributeError):
            func.body = assign  # type: ignore

    def test_function_with_empty_params(self):
        """Test Function with empty parameter list."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        func = ir.Function("no_params", [], [ir.ScalarType(dtype)], assign, span)

        assert len(func.params) == 0
        assert len(func.return_types) == 1
        assert func.body is not None

    def test_function_with_empty_return_types(self):
        """Test Function with empty return types list."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        func = ir.Function("no_return", [x], [], assign, span)

        assert len(func.params) == 1
        assert len(func.return_types) == 0
        assert func.body is not None

    def test_function_with_seqstmts_body(self):
        """Test Function with SeqStmts body containing multiple statements."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        body = ir.SeqStmts([assign1, assign2], span)
        func = ir.Function("multi_stmt", [x], [ir.ScalarType(dtype)], body, span)

        assert len(func.params) == 1
        assert len(func.return_types) == 1
        assert isinstance(func.body, ir.SeqStmts)
        assert len(cast(ir.SeqStmts, func.body).stmts) == 2

    def test_function_with_multiple_params(self):
        """Test Function with multiple parameters."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        c = ir.Var("c", ir.ScalarType(dtype), span)
        add_expr = ir.Add(a, b, dtype, span)
        assign = ir.AssignStmt(c, add_expr, span)
        func = ir.Function("add_func", [a, b], [ir.ScalarType(dtype)], assign, span)

        assert len(func.params) == 2
        assert cast(ir.Var, func.params[0]).name == "a"
        assert cast(ir.Var, func.params[1]).name == "b"

    def test_function_with_multiple_return_types(self):
        """Test Function with multiple return types."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(2, dtype, span), span)
        body = ir.SeqStmts([assign1, assign2], span)
        func = ir.Function("multi_return", [z], [ir.ScalarType(dtype), ir.ScalarType(dtype)], body, span)

        assert len(func.return_types) == 2
        assert isinstance(func.return_types[0], ir.ScalarType)
        assert isinstance(func.return_types[1], ir.ScalarType)

    def test_function_string_representation(self):
        """Test Function string representation."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        # Function with single param and single return
        func1 = ir.Function("simple_func", [x], [ir.ScalarType(dtype)], assign, span)
        str_repr = str(func1)
        assert isinstance(str_repr, str)

        # Function with multiple statements using SeqStmts
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        body = ir.SeqStmts([assign, assign2], span)
        func2 = ir.Function("multi_stmt", [x], [ir.ScalarType(dtype)], body, span)
        str_repr2 = str(func2)
        assert isinstance(str_repr2, str)


class TestFunctionHash:
    """Tests for Function hash function."""

    def test_function_same_structure_hash(self):
        """Test Function nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        func1 = ir.Function("test_func", [x1], [ir.ScalarType(dtype)], assign1, span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        func2 = ir.Function("test_func", [x2], [ir.ScalarType(dtype)], assign2, span)

        hash1 = ir.structural_hash(func1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(func2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_function_different_name_hash(self):
        """Test Function nodes with different names hash the same (name is IgnoreField)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("func1", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("func2", [x], [ir.ScalarType(dtype)], assign, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 == hash2  # name is IgnoreField, so should hash the same

    def test_function_different_params_hash(self):
        """Test Function nodes with different parameters hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x, z], [ir.ScalarType(dtype)], assign, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2

    def test_function_different_return_types_hash(self):
        """Test Function nodes with different return types hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(DataType.INT32)], assign, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2

    def test_function_different_body_hash(self):
        """Test Function nodes with different body hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign2, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2

    def test_function_empty_vs_non_empty_params_hash(self):
        """Test Function nodes with empty vs non-empty params hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2

    def test_function_empty_vs_non_empty_return_types_hash(self):
        """Test Function nodes with empty vs non-empty return_types hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [x], [], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2

    def test_function_different_body_types_hash(self):
        """Test Function nodes with different body types hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        body2 = ir.SeqStmts([assign1, assign2], span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], body2, span)

        hash1 = ir.structural_hash(func1)
        hash2 = ir.structural_hash(func2)
        assert hash1 != hash2


class TestFunctionStructuralEqual:
    """Tests for Function structural equality function."""

    def test_function_structural_equal(self):
        """Test Function nodes with same structure are equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        func1 = ir.Function("test_func", [x1], [ir.ScalarType(dtype)], assign1, span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        func2 = ir.Function("test_func", [x2], [ir.ScalarType(dtype)], assign2, span)

        assert ir.structural_equal(func1, func2, enable_auto_mapping=True)

    def test_function_different_name_equal(self):
        """Test Function nodes with different names are equal (name is IgnoreField)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("func1", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("func2", [x], [ir.ScalarType(dtype)], assign, span)

        assert ir.structural_equal(func1, func2)  # name is IgnoreField

    def test_function_different_params_not_equal(self):
        """Test Function nodes with different parameters are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x, z], [ir.ScalarType(dtype)], assign, span)

        assert not ir.structural_equal(func1, func2)

    def test_function_different_return_types_not_equal(self):
        """Test Function nodes with different return types are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(DataType.INT32)], assign, span)

        assert not ir.structural_equal(func1, func2)

    def test_function_different_body_not_equal(self):
        """Test Function nodes with different body are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign2, span)

        assert not ir.structural_equal(func1, func2)

    def test_function_different_from_base_irnode_not_equal(self):
        """Test Function is not equal to a different IRNode type."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        # Compare with a Var (different IRNode type)
        assert not ir.structural_equal(func, x)

    def test_function_empty_vs_non_empty_params_not_equal(self):
        """Test Function nodes with empty vs non-empty params are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        assert not ir.structural_equal(func1, func2)

    def test_function_empty_vs_non_empty_return_types_not_equal(self):
        """Test Function nodes with empty vs non-empty return_types are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)

        func1 = ir.Function("test_func", [x], [], assign, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        assert not ir.structural_equal(func1, func2)

    def test_function_different_body_types_not_equal(self):
        """Test Function nodes with different body types are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        body2 = ir.SeqStmts([assign1, assign2], span)

        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], body2, span)

        assert not ir.structural_equal(func1, func2)
