# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for parser passing span information to operations."""

import inspect

import pypto.language as pl
import pytest
from pypto.pypto_core import ir


def get_current_line():
    """Get the current line number in the calling code."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return -1


class TestParserSpanPassing:
    """Test that parser passes accurate span information to operations."""

    def test_parser_passes_span_to_tensor_add(self):
        """Parser should pass AST span to tensor.add operation."""

        current_line = get_current_line()

        @pl.function
        def test_func(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)  # Line current_line + 7
            return z

        # Function should be created successfully
        assert isinstance(test_func, ir.Function)
        assert test_func.name == "test_func"

        # Check that the function body contains statements with valid spans
        body = test_func.body
        assert isinstance(body, ir.SeqStmts)
        assert len(body.stmts) > 0

        # Find the assign statement containing the add operation
        for stmt in body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.add":
                    # Verify span is valid and not unknown
                    assert value.span.is_valid()
                    # Check line number points to the operation call (line current_line + 7)
                    assert value.span.begin_line == current_line + 7
                    # Check column points to the operation (after "= ")
                    assert value.span.begin_column > 0
                    # Verify end position is set (parser should provide range)
                    # End line should be same line or later
                    assert value.span.end_line >= value.span.begin_line
                    # If same line, end column should be after begin
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column
                    break

    def test_parser_passes_span_to_tensor_mul(self):
        """Parser should pass AST span to tensor.mul operation."""

        current_line = get_current_line()

        @pl.function
        def test_mul(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)  # Line current_line + 4
            return y

        assert isinstance(test_mul, ir.Function)
        assert isinstance(test_mul.body, ir.SeqStmts)

        # Check that operations have valid spans
        for stmt in test_mul.body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.mul_scalar":
                    assert value.span.is_valid()
                    assert value.span.begin_line == current_line + 4
                    assert value.span.begin_column > 0
                    assert value.span.end_line >= value.span.begin_line
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column

    def test_parser_passes_span_to_tensor_create(self):
        """Parser should pass AST span to tensor.create operation."""

        current_line = get_current_line()

        @pl.function
        def test_create() -> pl.Tensor[[64, 32], pl.FP32]:
            x: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)  # current_line + 4
            return x

        assert isinstance(test_create, ir.Function)
        assert isinstance(test_create.body, ir.SeqStmts)

        for stmt in test_create.body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.create":
                    assert value.span.is_valid()
                    assert value.span.begin_line == current_line + 4
                    assert value.span.begin_column == 46
                    assert value.span.end_line >= value.span.begin_line
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column

    def test_parser_span_accuracy_multiple_operations(self):
        """Test that parser assigns different spans to different operations."""

        current_line = get_current_line()

        @pl.function
        def test_multi(x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            y: pl.Tensor[[32], pl.FP32] = pl.mul(x, 2.0)  # current_line + 4
            z: pl.Tensor[[32], pl.FP32] = pl.add(y, 1.0)  # current_line + 5
            return z

        assert isinstance(test_multi, ir.Function)
        assert isinstance(test_multi.body, ir.SeqStmts)

        # Collect all Call spans
        call_spans = []
        for stmt in test_multi.body.stmts:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                call_spans.append((stmt.value.op.name, stmt.value.span))

        # Should have at least 2 operations
        assert len(call_spans) >= 2

        # All spans should be valid with proper line/column info
        for (op_name, span), offset in zip(call_spans, [4, 5]):
            assert span.is_valid(), f"Invalid span for {op_name}"
            assert span.begin_line == current_line + offset, f"Invalid line number for {op_name}"
            assert span.begin_column == 42, f"Invalid column number for {op_name}"
            assert span.end_line >= span.begin_line, f"Invalid end line for {op_name}"
            if span.end_line == span.begin_line:
                assert span.end_column >= span.begin_column, f"Invalid end column for {op_name}"

    def test_parser_passes_span_to_matmul(self):
        """Parser should pass AST span to tensor.matmul operation."""

        current_line = get_current_line()

        @pl.function
        def test_matmul(
            a: pl.Tensor[[64, 32], pl.FP32],
            b: pl.Tensor[[32, 16], pl.FP32],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            c: pl.Tensor[[64, 16], pl.FP32] = pl.matmul(a, b)  # current_line + 7
            return c

        assert isinstance(test_matmul, ir.Function)
        assert isinstance(test_matmul.body, ir.SeqStmts)

        for stmt in test_matmul.body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.matmul":
                    assert value.span.is_valid()
                    assert value.span.begin_line == current_line + 7
                    assert value.span.begin_column > 0
                    assert value.span.end_line >= value.span.begin_line
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column

    def test_parser_passes_span_to_cast(self):
        """Parser should pass AST span to tensor.cast operation."""

        current_line = get_current_line()

        @pl.function
        def test_cast(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP16]:
            y: pl.Tensor[[64], pl.FP16] = pl.cast(x, target_type=pl.FP16)  # current_line + 4
            return y

        assert isinstance(test_cast, ir.Function)
        assert isinstance(test_cast.body, ir.SeqStmts)

        for stmt in test_cast.body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.cast":
                    assert value.span.is_valid()
                    assert value.span.begin_line == current_line + 4
                    assert value.span.begin_column > 0
                    assert value.span.end_line >= value.span.begin_line
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column

    def test_parser_passes_span_to_exp(self):
        """Parser should pass AST span to tensor.exp operation."""

        current_line = get_current_line()

        @pl.function
        def test_exp(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.exp(x)  # current_line + 4
            return y

        assert isinstance(test_exp, ir.Function)
        assert isinstance(test_exp.body, ir.SeqStmts)

        for stmt in test_exp.body.stmts:
            if isinstance(stmt, ir.AssignStmt):
                value = stmt.value
                if isinstance(value, ir.Call) and value.op.name == "tensor.exp":
                    assert value.span.is_valid()
                    assert value.span.begin_line == current_line + 4
                    assert value.span.begin_column > 0
                    assert value.span.end_line >= value.span.begin_line
                    if value.span.end_line == value.span.begin_line:
                        assert value.span.end_column >= value.span.begin_column

    def test_all_operations_have_valid_spans(self):
        """Comprehensive test that all operations get valid spans from parser."""

        current_line = get_current_line()

        @pl.function
        def test_comprehensive(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)  # current_line + 7
            b: pl.Tensor[[64], pl.FP32] = pl.sub(a, 1.0)  # current_line + 8
            c: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)  # current_line + 9
            d: pl.Tensor[[64], pl.FP32] = pl.div(c, 3.0)  # current_line + 10
            e: pl.Tensor[[64], pl.FP32] = pl.exp(d)  # current_line + 11
            return e

        assert isinstance(test_comprehensive, ir.Function)
        assert isinstance(test_comprehensive.body, ir.SeqStmts)

        # Collect all operations and verify spans
        operations_checked = 0
        expected_lines = [7, 8, 9, 10, 11]
        for stmt in test_comprehensive.body.stmts:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                span = stmt.value.span
                op_name = stmt.value.op.name

                # Verify span validity
                assert span.is_valid(), f"Invalid span for {op_name}"

                # Verify line number
                assert span.begin_line == current_line + expected_lines[operations_checked]

                # Verify column number
                assert span.begin_column > 0, f"Invalid column for {op_name}"

                # Verify end position
                assert span.end_line >= span.begin_line, f"Invalid end line for {op_name}"
                if span.end_line == span.begin_line:
                    assert span.end_column >= span.begin_column, f"Invalid end column for {op_name}"

                operations_checked += 1

        # Should have checked exactly 5 operations
        assert operations_checked == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
