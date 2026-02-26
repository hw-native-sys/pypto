# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Function class and ParamDirection."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.language.parser.diagnostics import ParserTypeError


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
        assert func.params[0].name == "a"
        assert func.params[1].name == "b"
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
        assert func.params[0].name == "a"
        assert func.params[1].name == "b"

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
        func = ir.Function(
            "multi_return",
            [z],
            [ir.ScalarType(dtype), ir.ScalarType(dtype)],
            body,
            span,
            ir.FunctionType.InCore,
        )

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

    def test_function_default_param_directions(self):
        """Test that plain Var params default to ParamDirection.In."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        func = ir.Function("f", [x], [ir.ScalarType(dtype)], assign, span)

        assert len(func.param_directions) == 1
        assert func.param_directions[0] == ir.ParamDirection.In

    def test_function_with_explicit_directions(self):
        """Test Function with explicit (Var, ParamDirection) tuples."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x = ir.Var("x", ir.TensorType([64], dtype), span)
        assign = ir.AssignStmt(x, x, span)

        func = ir.Function(
            "f",
            [(x, ir.ParamDirection.Out)],
            [ir.TensorType([64], dtype)],
            assign,
            span,
        )

        assert len(func.params) == 1
        assert func.params[0].name == "x"
        assert func.param_directions[0] == ir.ParamDirection.Out

    def test_function_with_mixed_directions(self):
        """Test Function with mixed plain Var and (Var, ParamDirection) params."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        a = ir.Var("a", ir.TensorType([64], dtype), span)
        b = ir.Var("b", ir.TensorType([64], dtype), span)
        c = ir.Var("c", ir.TensorType([64], dtype), span)
        assign = ir.AssignStmt(a, a, span)

        func = ir.Function(
            "f",
            [a, (b, ir.ParamDirection.InOut), (c, ir.ParamDirection.Out)],
            [],
            assign,
            span,
        )

        assert len(func.params) == 3
        assert func.param_directions[0] == ir.ParamDirection.In
        assert func.param_directions[1] == ir.ParamDirection.InOut
        assert func.param_directions[2] == ir.ParamDirection.Out

    def test_function_param_directions_immutable(self):
        """Test that param_directions is immutable."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        func = ir.Function("f", [x], [ir.ScalarType(dtype)], assign, span)

        with pytest.raises(AttributeError):
            func.param_directions = []  # type: ignore

    def test_function_empty_params_empty_directions(self):
        """Test that empty params produces empty param_directions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        func = ir.Function("f", [], [], assign, span)

        assert len(func.param_directions) == 0


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

    def test_function_different_directions_hash(self):
        """Test Function nodes with different param directions hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x = ir.Var("x", ir.TensorType([64], dtype), span)
        assign = ir.AssignStmt(x, x, span)

        func_in = ir.Function("f", [x], [], assign, span)
        func_out = ir.Function("f", [(x, ir.ParamDirection.Out)], [], assign, span)
        func_inout = ir.Function("f", [(x, ir.ParamDirection.InOut)], [], assign, span)

        hash_in = ir.structural_hash(func_in)
        hash_out = ir.structural_hash(func_out)
        hash_inout = ir.structural_hash(func_inout)

        assert hash_in != hash_out
        assert hash_in != hash_inout
        assert hash_out != hash_inout

    def test_function_same_directions_same_hash(self):
        """Test Function nodes with same param directions hash the same."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x1 = ir.Var("x", ir.TensorType([64], dtype), span)
        x2 = ir.Var("x", ir.TensorType([64], dtype), span)
        assign1 = ir.AssignStmt(x1, x1, span)
        assign2 = ir.AssignStmt(x2, x2, span)

        func1 = ir.Function("f", [(x1, ir.ParamDirection.Out)], [], assign1, span)
        func2 = ir.Function("f", [(x2, ir.ParamDirection.Out)], [], assign2, span)

        hash1 = ir.structural_hash(func1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(func2, enable_auto_mapping=True)
        assert hash1 == hash2


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

        ir.assert_structural_equal(func1, func2, enable_auto_mapping=True)

    def test_function_different_name_equal(self):
        """Test Function nodes with different names are equal (name is IgnoreField)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        func1 = ir.Function("func1", [x], [ir.ScalarType(dtype)], assign, span)
        func2 = ir.Function("func2", [x], [ir.ScalarType(dtype)], assign, span)

        ir.assert_structural_equal(func1, func2)  # name is IgnoreField

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

    def test_function_same_directions_equal(self):
        """Test Function nodes with same param directions are structurally equal."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x1 = ir.Var("x", ir.TensorType([64], dtype), span)
        x2 = ir.Var("x", ir.TensorType([64], dtype), span)
        assign1 = ir.AssignStmt(x1, x1, span)
        assign2 = ir.AssignStmt(x2, x2, span)

        func1 = ir.Function("f", [(x1, ir.ParamDirection.InOut)], [], assign1, span)
        func2 = ir.Function("f", [(x2, ir.ParamDirection.InOut)], [], assign2, span)

        ir.assert_structural_equal(func1, func2, enable_auto_mapping=True)

    def test_function_different_directions_not_equal(self):
        """Test Function nodes with different param directions are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x = ir.Var("x", ir.TensorType([64], dtype), span)
        assign = ir.AssignStmt(x, x, span)

        func_in = ir.Function("f", [x], [], assign, span)
        func_out = ir.Function("f", [(x, ir.ParamDirection.Out)], [], assign, span)
        func_inout = ir.Function("f", [(x, ir.ParamDirection.InOut)], [], assign, span)

        assert not ir.structural_equal(func_in, func_out)
        assert not ir.structural_equal(func_in, func_inout)
        assert not ir.structural_equal(func_out, func_inout)


class TestFunctionSerialization:
    """Tests for Function msgpack serialization with param directions."""

    def test_serialize_function_default_directions(self):
        """Test serialization preserves default (In) param directions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        body = ir.AssignStmt(x, y, span)
        func = ir.Function("f", [x, y], [ir.ScalarType(dtype)], body, span)

        data = ir.serialize(func)
        restored = cast(ir.Function, ir.deserialize(data))

        ir.assert_structural_equal(func, restored, enable_auto_mapping=True)
        assert len(restored.param_directions) == 2
        assert restored.param_directions[0] == ir.ParamDirection.In
        assert restored.param_directions[1] == ir.ParamDirection.In

    def test_serialize_function_with_out_direction(self):
        """Test serialization preserves Out param direction."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x = ir.Var("x", ir.TensorType([64], dtype), span)
        body = ir.AssignStmt(x, x, span)
        func = ir.Function("f", [(x, ir.ParamDirection.Out)], [], body, span)

        data = ir.serialize(func)
        restored = cast(ir.Function, ir.deserialize(data))

        ir.assert_structural_equal(func, restored, enable_auto_mapping=True)
        assert restored.param_directions[0] == ir.ParamDirection.Out

    def test_serialize_function_with_inout_direction(self):
        """Test serialization preserves InOut param direction."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        x = ir.Var("x", ir.TensorType([64], dtype), span)
        body = ir.AssignStmt(x, x, span)
        func = ir.Function("f", [(x, ir.ParamDirection.InOut)], [], body, span)

        data = ir.serialize(func)
        restored = cast(ir.Function, ir.deserialize(data))

        ir.assert_structural_equal(func, restored, enable_auto_mapping=True)
        assert restored.param_directions[0] == ir.ParamDirection.InOut

    def test_serialize_function_mixed_directions(self):
        """Test serialization preserves mixed param directions."""
        span = ir.Span.unknown()
        dtype = DataType.FP32
        a = ir.Var("a", ir.TensorType([64], dtype), span)
        b = ir.Var("b", ir.TensorType([64], dtype), span)
        c = ir.Var("c", ir.TensorType([64], dtype), span)
        body = ir.AssignStmt(a, a, span)
        func = ir.Function(
            "kernel",
            [a, (b, ir.ParamDirection.InOut), (c, ir.ParamDirection.Out)],
            [ir.TensorType([64], dtype)],
            body,
            span,
            ir.FunctionType.InCore,
        )

        data = ir.serialize(func)
        restored = cast(ir.Function, ir.deserialize(data))

        ir.assert_structural_equal(func, restored, enable_auto_mapping=True)
        assert restored.param_directions[0] == ir.ParamDirection.In
        assert restored.param_directions[1] == ir.ParamDirection.InOut
        assert restored.param_directions[2] == ir.ParamDirection.Out


class TestParamDirectionDSL:
    """Tests for ParamDirection through the Python DSL (@pl.function)."""

    def test_default_direction_is_in(self):
        """Default parameter direction is In."""

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(f, ir.Function)
        assert len(f.params) == 1
        assert f.param_directions[0] == ir.ParamDirection.In

    def test_inout_tensor_param(self):
        """InOut wrapper sets direction to InOut."""

        @pl.function
        def f(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(f, ir.Function)
        assert f.param_directions[0] == ir.ParamDirection.InOut
        assert isinstance(f.params[0].type, ir.TensorType)

    def test_out_tensor_param(self):
        """Out wrapper sets direction to Out."""

        @pl.function
        def f(x: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(f, ir.Function)
        assert f.param_directions[0] == ir.ParamDirection.Out
        assert isinstance(f.params[0].type, ir.TensorType)

    def test_mixed_directions(self):
        """Multiple params with different directions."""

        @pl.function
        def kernel(
            qi: pl.Tensor[[16, 128], pl.BF16],
            output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],
            result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            scale: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            return qi

        assert len(kernel.params) == 4
        assert kernel.param_directions[0] == ir.ParamDirection.In
        assert kernel.param_directions[1] == ir.ParamDirection.InOut
        assert kernel.param_directions[2] == ir.ParamDirection.Out
        assert kernel.param_directions[3] == ir.ParamDirection.In

    def test_scalar_inout_rejected(self):
        """Scalar with InOut direction raises error."""
        with pytest.raises(ParserTypeError, match="Scalar.*InOut"):

            @pl.function
            def f(x: pl.InOut[pl.Scalar[pl.FP32]]) -> pl.Scalar[pl.FP32]:
                return x

    def test_scalar_out_allowed(self):
        """Scalar with Out direction is allowed."""

        @pl.function
        def f(x: pl.Out[pl.Scalar[pl.FP32]]) -> pl.Scalar[pl.FP32]:
            return x

        assert f.param_directions[0] == ir.ParamDirection.Out
        assert isinstance(f.params[0].type, ir.ScalarType)

    def test_inout_tile_param(self):
        """InOut works with Tile types."""

        @pl.function(type=pl.FunctionType.InCore)
        def f(x: pl.InOut[pl.Tile[[64, 64], pl.FP32]]) -> pl.Tile[[64, 64], pl.FP32]:
            return x

        assert f.param_directions[0] == ir.ParamDirection.InOut
        assert isinstance(f.params[0].type, ir.TileType)

    def test_dsl_same_directions_structural_equal(self):
        """DSL functions with same directions are structurally equal."""

        @pl.function
        def f1(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function
        def f2(y: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return y

        ir.assert_structural_equal(f1, f2)

    def test_dsl_different_directions_not_equal(self):
        """DSL functions with different directions are not structurally equal."""

        @pl.function
        def f1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function
        def f2(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert not ir.structural_equal(f1, f2)

    def test_dsl_different_directions_different_hash(self):
        """DSL functions with different directions produce different hashes."""

        @pl.function
        def f1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function
        def f2(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert ir.structural_hash(f1) != ir.structural_hash(f2)


class TestParamDirectionPrinting:
    """Tests for printing functions with param directions."""

    def test_print_in_direction_no_wrapper(self):
        """In direction prints without wrapper."""

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(f)
        assert "x: pl.Tensor[[64], pl.FP32]" in printed
        assert "InOut" not in printed
        assert "Out" not in printed

    def test_print_inout_direction(self):
        """InOut direction prints with pl.InOut[...] wrapper."""

        @pl.function
        def f(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(f)
        assert "pl.InOut[pl.Tensor[[64], pl.FP32]]" in printed

    def test_print_out_direction(self):
        """Out direction prints with pl.Out[...] wrapper."""

        @pl.function
        def f(x: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(f)
        assert "pl.Out[pl.Tensor[[64], pl.FP32]]" in printed


class TestParamDirectionRoundTrip:
    """Tests for round-trip (print -> parse) of param directions."""

    def test_roundtrip_default_direction(self):
        """Parse -> print -> parse preserves default In direction."""

        @pl.function
        def original(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)

    def test_roundtrip_inout_direction(self):
        """Parse -> print -> parse preserves InOut direction."""

        @pl.function
        def original(x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)

    def test_roundtrip_out_direction(self):
        """Parse -> print -> parse preserves Out direction."""

        @pl.function
        def original(x: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return x

        printed = str(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)

    def test_roundtrip_mixed_directions(self):
        """Parse -> print -> parse preserves mixed directions."""

        @pl.function
        def original(
            a: pl.Tensor[[64], pl.FP32],
            b: pl.InOut[pl.Tensor[[64], pl.FP32]],
            c: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return a

        printed = str(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
