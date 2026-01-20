# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Span class."""

import pytest
from pypto import DataType, ir


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test creating a Span with valid coordinates."""
        span = ir.Span("test.py", 1, 5, 1, 10)
        assert span.filename == "test.py"
        assert span.begin_line == 1
        assert span.begin_column == 5
        assert span.end_line == 1
        assert span.end_column == 10

    def test_span_to_string(self):
        """Test Span to_string method."""
        span = ir.Span("test.py", 10, 20, 10, 30)
        assert span.to_string() == "test.py:10:20"

    def test_span_str_repr(self):
        """Test Span __str__ and __repr__."""
        span = ir.Span("example.py", 5, 1, 5, 10)
        assert str(span) == "example.py:5:1"
        assert repr(span) == "example.py:5:1"

    def test_span_is_valid(self):
        """Test Span is_valid method."""
        valid_span = ir.Span("test.py", 1, 1, 1, 10)
        assert valid_span.is_valid() is True

        invalid_span = ir.Span("test.py", -1, -1, -1, -1)
        assert invalid_span.is_valid() is False

        # Test multi-line span
        multiline_span = ir.Span("test.py", 1, 5, 3, 10)
        assert multiline_span.is_valid() is True

    def test_span_unknown(self):
        """Test Span.unknown() factory method."""
        span = ir.Span.unknown()
        assert span.filename == ""
        assert span.is_valid() is False

    def test_span_immutability(self):
        """Test that Span attributes are immutable."""
        span = ir.Span.unknown()

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            span.filename = "other.py"  # type: ignore

        with pytest.raises(AttributeError):
            span.begin_line = 5  # type: ignore


class TestSpanTracking:
    """Tests for source location tracking via Span."""

    def test_span_preserved_in_tree(self):
        """Test that spans are preserved throughout the expression tree."""
        span1 = ir.Span("file1.py", 1, 1, 1, 5)
        span2 = ir.Span("file2.py", 2, 2, 2, 5)
        span3 = ir.Span("file3.py", 3, 3, 3, 10)
        dtype = DataType.INT64

        x = ir.Var("x", ir.ScalarType(dtype), span1)
        y = ir.Var("y", ir.ScalarType(dtype), span2)
        add_expr = ir.Add(x, y, dtype, span3)

        assert x.span.filename == "file1.py"
        assert y.span.filename == "file2.py"
        assert add_expr.span.filename == "file3.py"

    def test_unknown_span(self):
        """Test creating IR nodes with unknown spans."""
        unknown_span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), unknown_span)

        assert x.span.is_valid() is False
        assert x.span.filename == ""

    def test_span_immutable_in_node(self):
        """Test that span attribute in IRNode is immutable."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            x.span = ir.Span("other.py", 2, 2, 2, 5)  # type: ignore


if __name__ == "__main__":
    pytest.main(["-v", __file__])
