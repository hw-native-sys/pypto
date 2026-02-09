# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for automatic span capture in IR operations."""

import inspect

import pytest
from pypto import DataType, ir
from pypto.ir.op import block as block_ops
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.utils import _get_span_or_capture


def get_current_line():
    """Get the current line number in the calling code."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return -1


class TestTensorOperationSpanCapture:
    """Test span capture for tensor operations."""

    def test_tensor_add_captures_span(self):
        """Tensor operations should capture caller span automatically."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())
        y = ir.Var("y", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        # Call operation - span should be captured
        line_before = get_current_line()
        result = tensor_ops.add(x, y)

        # Verify span points to this file and line
        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()
        assert result.span.begin_line == line_before + 1

    def test_tensor_mul_captures_span(self):
        """Test mul operation span capture."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        # Test with scalar
        line_before = get_current_line()
        result = tensor_ops.mul(x, 2.0)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()
        assert result.span.begin_line == line_before + 1

    def test_tensor_create_captures_span(self):
        """Test create operation span capture."""
        line_before = get_current_line()
        result = tensor_ops.create([64, 32], DataType.FP32)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()
        assert result.span.begin_line == line_before + 1

    def test_tensor_matmul_captures_span(self):
        """Test matmul operation span capture."""
        lhs = ir.Var("lhs", ir.TensorType([64, 32], DataType.FP32), ir.Span.unknown())
        rhs = ir.Var("rhs", ir.TensorType([32, 16], DataType.FP32), ir.Span.unknown())

        result = tensor_ops.matmul(lhs, rhs)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_explicit_span_overrides_capture(self):
        """Explicit span should override automatic capture."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())
        y = ir.Var("y", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        explicit_span = ir.Span("custom.py", 100, 20)
        result = tensor_ops.add(x, y, span=explicit_span)

        # Should use explicit span, not captured span
        assert result.span.filename == "custom.py"
        assert result.span.begin_line == 100
        assert result.span.begin_column == 20

    def test_nested_operations_each_capture_own_span(self):
        """Each nested operation should capture its own span."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())
        y = ir.Var("y", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        line_before_add = get_current_line()
        add_result = tensor_ops.add(x, y)
        line_before_mul = get_current_line()
        mul_result = tensor_ops.mul(add_result, 2.0)

        # Each should have different line numbers
        assert add_result.span.begin_line == line_before_add + 1
        assert mul_result.span.begin_line == line_before_mul + 1
        assert add_result.span.begin_line != mul_result.span.begin_line
        assert add_result.span.is_valid()
        assert mul_result.span.is_valid()

    def test_tensor_view_captures_span(self):
        """Test view operation span capture."""
        tensor = ir.Var("tensor", ir.TensorType([64, 32], DataType.FP32), ir.Span.unknown())

        result = tensor_ops.view(tensor, [32, 16], [0, 0])

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_tensor_cast_captures_span(self):
        """Test cast operation span capture."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        result = tensor_ops.cast(x, DataType.FP16)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_tensor_exp_captures_span(self):
        """Test exp operation span capture."""
        x = ir.Var("x", ir.TensorType([64], DataType.FP32), ir.Span.unknown())

        result = tensor_ops.exp(x)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_tensor_row_max_captures_span(self):
        """Test row_max operation span capture."""
        x = ir.Var("x", ir.TensorType([64, 32], DataType.FP32), ir.Span.unknown())

        result = tensor_ops.row_max(x)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()


class TestBlockOperationSpanCapture:
    """Test span capture for block operations."""

    def test_block_matmul_captures_span(self):
        """Block operations should also capture span."""
        tile_type = ir.TileType([16, 16], DataType.FP16)
        a = ir.Var("a", tile_type, ir.Span.unknown())
        b = ir.Var("b", tile_type, ir.Span.unknown())

        result = block_ops.matmul(a, b)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_block_add_captures_span(self):
        """Test block add operation span capture."""
        tile_type = ir.TileType([16, 16], DataType.FP16)
        a = ir.Var("a", tile_type, ir.Span.unknown())
        b = ir.Var("b", tile_type, ir.Span.unknown())

        result = block_ops.add(a, b)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_block_load_captures_span(self):
        """Test block load operation span capture."""
        tensor = ir.Var("tensor", ir.TensorType([64, 64], DataType.FP16), ir.Span.unknown())

        result = block_ops.load(tensor, 0, 0, 16, 16)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_block_exp_captures_span(self):
        """Test block exp operation span capture."""
        tile_type = ir.TileType([16, 16], DataType.FP16)
        tile = ir.Var("tile", tile_type, ir.Span.unknown())

        result = block_ops.exp(tile)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_block_row_max_captures_span(self):
        """Test block row_max operation span capture."""
        tile_type = ir.TileType([16, 16], DataType.FP16)
        tile = ir.Var("tile", tile_type, ir.Span.unknown())

        result = block_ops.row_max(tile)

        assert result.span.filename.endswith("test_operation_span_capture.py")
        assert result.span.is_valid()

    def test_block_explicit_span_override(self):
        """Explicit span should override automatic capture for block ops."""
        tile_type = ir.TileType([16, 16], DataType.FP16)
        a = ir.Var("a", tile_type, ir.Span.unknown())
        b = ir.Var("b", tile_type, ir.Span.unknown())

        explicit_span = ir.Span("block_ops.py", 42, 5)
        result = block_ops.add(a, b, span=explicit_span)

        assert result.span.filename == "block_ops.py"
        assert result.span.begin_line == 42
        assert result.span.begin_column == 5


class TestUtilityFunction:
    """Test the _get_span_or_capture utility function."""

    def test_get_span_or_capture_returns_explicit_span(self):
        """When span provided, should return it unchanged."""

        explicit = ir.Span("test.py", 42, 10)
        result = _get_span_or_capture(span=explicit)

        assert result.filename == "test.py"
        assert result.begin_line == 42
        assert result.begin_column == 10

    def test_get_span_or_capture_auto_captures(self):
        """When span not provided, should capture from caller."""

        line_before = get_current_line()
        result = _get_span_or_capture(frame_offset=0)

        # Should capture this test file
        assert result.filename.endswith("test_operation_span_capture.py")
        assert result.is_valid()
        assert result.begin_line == line_before + 1

    def test_get_span_or_capture_with_frame_offset(self):
        """Test frame offset parameter works correctly."""

        def wrapper():
            # With offset=1, should skip this wrapper and capture caller
            return _get_span_or_capture(frame_offset=1)

        line_before = get_current_line()
        result = wrapper()

        # Should capture the line where wrapper() is called, not inside wrapper
        assert result.filename.endswith("test_operation_span_capture.py")
        assert result.is_valid()
        assert result.begin_line == line_before + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
