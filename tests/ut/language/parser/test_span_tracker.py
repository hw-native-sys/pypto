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

import pytest
from pypto import ir
from pypto.language.parser.span_tracker import SpanTracker


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


class TestSpanTrackerSourceMap:
    """Source-map remapping of spans to original source (issue #1612)."""

    def test_mapped_line_is_remapped(self):
        """A node whose emitted line is in the map gets the original location."""
        # line_offset shifts node.lineno (1) -> emitted line 11.
        tracker = SpanTracker(
            "<jit:kernel>", ["code"], line_offset=10, source_map={11: ("/real/kernel.py", 5, 8)}
        )
        node = ast.parse("x = 1").body[0]

        span = tracker.get_span(node)

        assert span.filename == "/real/kernel.py"
        assert span.begin_line == 5
        assert span.begin_column == 8

    def test_unmapped_line_keeps_generated_coords(self):
        """A node whose emitted line is absent from the map keeps generated coords."""
        tracker = SpanTracker(
            "<jit:kernel>", ["code"], line_offset=10, source_map={999: ("/real/kernel.py", 5, 8)}
        )
        node = ast.parse("x = 1").body[0]

        span = tracker.get_span(node)

        assert span.filename == "<jit:kernel>"
        assert span.begin_line == 11

    def test_no_source_map_is_noop(self):
        """Without a source map, spans are unchanged (default behavior)."""
        tracker = SpanTracker("<jit:kernel>", ["code"], line_offset=10)
        node = ast.parse("x = 1").body[0]

        assert tracker.get_span(node).filename == "<jit:kernel>"

    def test_multiline_span_is_remapped(self):
        """get_multiline_span remaps on its start line too."""
        tracker = SpanTracker(
            "<jit:kernel>", ["code"], line_offset=10, source_map={11: ("/real/kernel.py", 5, 8)}
        )
        node = ast.parse("x = 1").body[0]

        span = tracker.get_multiline_span(node, node)

        assert span.filename == "/real/kernel.py"
        assert span.begin_line == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
