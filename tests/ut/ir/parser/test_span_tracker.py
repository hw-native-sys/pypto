# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SpanTracker."""

import ast

from pypto.ir.parser.span_tracker import SpanTracker
from pypto.pypto_core import ir


class TestSpanTracker:
    """Tests for SpanTracker class."""

    def test_initialization(self):
        """Test SpanTracker initializes correctly."""
        source_file = "test.py"
        source_lines = ["line1", "line2"]

        tracker = SpanTracker(source_file, source_lines)

        assert tracker.source_file == source_file
        assert tracker.source_lines == source_lines

    def test_get_span_from_node(self):
        """Test getting span from AST node."""
        source_file = "test.py"
        source = "x = 42"
        source_lines = source.split("\n")

        tracker = SpanTracker(source_file, source_lines)

        # Parse and get AST node
        tree = ast.parse(source)
        assign_node = tree.body[0]

        span = tracker.get_span(assign_node)

        assert isinstance(span, ir.Span)
        assert span.filename == source_file
        assert span.begin_line == 1
        assert span.begin_column == 0

    def test_get_span_none_node(self):
        """Test getting span from None node returns unknown span."""
        tracker = SpanTracker("test.py", [])

        span = tracker.get_span(None)

        # Should return unknown span
        assert isinstance(span, ir.Span)

    def test_get_multiline_span(self):
        """Test getting span covering multiple lines."""
        source_file = "test.py"
        source = """def func():
    x = 1
    y = 2"""
        source_lines = source.split("\n")

        tracker = SpanTracker(source_file, source_lines)

        tree = ast.parse(source)
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)
        first_stmt = func_node.body[0]
        last_stmt = func_node.body[-1]

        span = tracker.get_multiline_span(first_stmt, last_stmt)

        assert isinstance(span, ir.Span)
        assert span.filename == source_file
        assert span.begin_line == 2  # First statement line
        assert span.end_line == 3  # Last statement line

    def test_get_multiline_span_same_line(self):
        """Test multiline span on same line."""
        tracker = SpanTracker("test.py", ["x = y + z"])

        source = "x = y + z"
        tree = ast.parse(source)
        node = tree.body[0]

        span = tracker.get_multiline_span(node, node)

        assert span.begin_line == span.end_line

    def test_span_preserves_filename(self):
        """Test that span preserves the source filename."""
        source_file = "/path/to/my_module.py"
        tracker = SpanTracker(source_file, ["code"])

        tree = ast.parse("x = 1")
        node = tree.body[0]

        span = tracker.get_span(node)

        assert span.filename == source_file
