# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for automatic span capture in operators."""

import pytest
from pypto import DataType, ir


class TestVarOperatorSpans:
    """Test span capture for Var operators."""

    def test_var_binary_operators_capture_span(self):
        """Test that binary operators on Var capture span correctly."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Test each binary operator
        add_result = x + y
        sub_result = x - y
        mul_result = x * y
        div_result = x / y
        floordiv_result = x // y
        mod_result = x % y
        pow_result = x**y

        base_line = add_result.span.begin_line
        base_column = add_result.span.begin_column
        # All results should have valid spans pointing to this file
        for result, line_offset in [
            (add_result, 0),
            (sub_result, 1),
            (mul_result, 2),
            (div_result, 3),
            (floordiv_result, 4),
            (mod_result, 5),
            (pow_result, 6),
        ]:
            assert result.span.filename.endswith("test_operator_spans.py")
            assert result.span.begin_line == base_line + line_offset
            assert result.span.begin_column == base_column

    def test_var_comparison_operators_capture_span(self):
        """Test that comparison operators on Var capture span correctly."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Test each comparison operator
        eq_result = x == y
        ne_result = x != y
        lt_result = x < y
        le_result = x <= y
        gt_result = x > y
        ge_result = x >= y

        base_line = eq_result.span.begin_line
        base_column = eq_result.span.begin_column
        # All results should have valid spans
        for result, line_offset in [
            (eq_result, 0),
            (ne_result, 1),
            (lt_result, 2),
            (le_result, 3),
            (gt_result, 4),
            (ge_result, 5),
        ]:
            assert result.span.filename.endswith("test_operator_spans.py")
            assert result.span.begin_line == base_line + line_offset
            assert result.span.begin_column == base_column

    def test_var_unary_operators_capture_span(self):
        """Test that unary operators on Var capture span correctly."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        neg_result = -x

        assert neg_result.span.filename.endswith("test_operator_spans.py")
        assert neg_result.span.begin_line > 0

    def test_var_reverse_operators_capture_span(self):
        """Test that reverse operators (int on left) capture span correctly."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Test reverse operators with Python int on left
        radd_result = 5 + x
        rsub_result = 5 - x
        rmul_result = 5 * x
        rdiv_result = 5 / x
        rfloordiv_result = 5 // x
        rmod_result = 5 % x
        rpow_result = 5**x

        # All results should have valid spans
        for result in [
            radd_result,
            rsub_result,
            rmul_result,
            rdiv_result,
            rfloordiv_result,
            rmod_result,
            rpow_result,
        ]:
            assert result.span.filename.endswith("test_operator_spans.py")
            assert result.span.begin_line > 0

    def test_var_with_tensortype_raises_error(self):
        """Test that operators on Var with TensorType raise appropriate error."""
        tensor_var = ir.Var("t", ir.TensorType([128, 256], DataType.FP32), ir.Span.unknown())
        scalar_var = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        with pytest.raises(TypeError, match="ScalarType"):
            _ = tensor_var + scalar_var


class TestScalarExprOperatorSpans:
    """Test span capture for ScalarExpr operators."""

    def test_constint_binary_operators_capture_span(self):
        """Test that binary operators on ConstInt capture span correctly."""
        x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())
        y = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())

        # Test binary operators
        add_result = x + y
        sub_result = x - y
        mul_result = x * y
        div_result = x / y

        # All results should have valid spans
        for result in [add_result, sub_result, mul_result, div_result]:
            assert result.span.filename.endswith("test_operator_spans.py")
            assert result.span.begin_line > 0

    def test_constfloat_operators_capture_span(self):
        """Test that operators on ConstFloat capture span correctly."""
        x = ir.ConstFloat(3.14, DataType.FP32, ir.Span.unknown())
        y = ir.ConstFloat(2.0, DataType.FP32, ir.Span.unknown())

        result = x * y

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_scalarexpr_comparison_operators_capture_span(self):
        """Test that comparison operators on ScalarExpr capture span correctly."""
        x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())
        y = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())

        eq_result = x == y
        lt_result = x < y
        ge_result = x >= y

        for result in [eq_result, lt_result, ge_result]:
            assert result.span.filename.endswith("test_operator_spans.py")
            assert result.span.begin_line > 0

    def test_scalarexpr_unary_operator_captures_span(self):
        """Test that unary operator on ScalarExpr captures span correctly."""
        x = ir.ConstInt(42, DataType.INT32, ir.Span.unknown())

        neg_result = -x

        assert neg_result.span.filename.endswith("test_operator_spans.py")
        assert neg_result.span.begin_line > 0


class TestMixedOperatorSpans:
    """Test span capture for mixed Var and ScalarExpr operators."""

    def test_var_with_constint_captures_span(self):
        """Test operators between Var and ConstInt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        five = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())

        result = x + five

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_var_with_python_int_captures_span(self):
        """Test operators between Var and Python int (auto-normalized)."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Python int should be auto-converted to ConstInt
        result = x * 2

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_constint_with_python_int_captures_span(self):
        """Test operators between ConstInt and Python int."""
        x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())

        result = x + 5

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0


class TestComplexExpressionSpans:
    """Test span capture for complex expressions with multiple operations."""

    def test_nested_operations_capture_different_spans(self):
        """Test that each operation in a complex expression captures its own span."""
        x = ir.Var("x", ir.ScalarType(DataType.FP32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.FP32), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.FP32), ir.Span.unknown())

        # Each operation should capture its own span
        temp1 = x + y  # Line 206
        temp2 = temp1 * z  # Line 207
        result = temp2 - x  # Line 208

        # All should have valid spans from this file
        assert temp1.span.filename.endswith("test_operator_spans.py")
        assert temp2.span.filename.endswith("test_operator_spans.py")
        assert result.span.filename.endswith("test_operator_spans.py")

        # Each should have different line numbers (they're on different lines)
        assert temp1.span.begin_line != temp2.span.begin_line
        assert temp2.span.begin_line != result.span.begin_line

    def test_complex_arithmetic_expression(self):
        """Test complex arithmetic expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Complex expression: (x + y) * 2 - x
        result = (x + y) * 2 - x

        # The outermost operation (subtraction) should have a valid span
        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_comparison_in_complex_expression(self):
        """Test comparison operators in complex expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        # Expression with comparison: (x + 5) < (y * 2)
        result = (x + 5) < (y * 2)

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0


class TestSpanWithDifferentDataTypes:
    """Test span capture works correctly with different data types."""

    def test_int_operators_capture_span(self):
        """Test operators on INT32 variables."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        result = x + y

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_float_operators_capture_span(self):
        """Test operators on FP32 variables."""
        x = ir.Var("x", ir.ScalarType(DataType.FP32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.FP32), ir.Span.unknown())

        result = x / y

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0

    def test_bf16_operators_capture_span(self):
        """Test operators on BF16 variables."""
        x = ir.Var("x", ir.ScalarType(DataType.BF16), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.BF16), ir.Span.unknown())

        result = x * y

        assert result.span.filename.endswith("test_operator_spans.py")
        assert result.span.begin_line > 0


class TestAllOperators:
    """Comprehensive test covering all operators."""

    def test_all_binary_operators(self):
        """Test all binary operators capture spans."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        operators_and_results = [
            ("+", x + y),
            ("-", x - y),
            ("*", x * y),
            ("/", x / y),
            ("//", x // y),
            ("%", x % y),
            ("**", x**y),
            ("==", x == y),
            ("!=", x != y),
            ("<", x < y),
            ("<=", x <= y),
            (">", x > y),
            (">=", x >= y),
        ]

        for op_name, result in operators_and_results:
            assert result.span.filename.endswith("test_operator_spans.py"), f"Operator {op_name} failed"
            assert result.span.begin_line > 0, f"Operator {op_name} has invalid line number"

    def test_all_reverse_operators(self):
        """Test all reverse operators capture spans."""
        x = ir.Var("x", ir.ScalarType(DataType.INT32), ir.Span.unknown())

        reverse_operators = [
            ("radd", 5 + x),
            ("rsub", 5 - x),
            ("rmul", 5 * x),
            ("rtruediv", 5 / x),
            ("rfloordiv", 5 // x),
            ("rmod", 5 % x),
            ("rpow", 5**x),
        ]

        for op_name, result in reverse_operators:
            assert result.span.filename.endswith("test_operator_spans.py"), (
                f"Reverse operator {op_name} failed"
            )
            assert result.span.begin_line > 0, f"Reverse operator {op_name} has invalid line number"
