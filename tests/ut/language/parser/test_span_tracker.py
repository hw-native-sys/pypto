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
        # ast col_offset 0 -> Span column 1: Span columns are 1-indexed, so a
        # statement at the left margin starts at column 1, not 0.
        assert span.begin_column == 1
        assert span.is_valid()

    def test_get_span_none_node(self):
        """Test getting span from None node returns unknown span."""
        tracker = SpanTracker("test.py", [])

        span = tracker.get_span(None)

        # Should return the unknown span, not a span into "test.py"
        assert isinstance(span, ir.Span)
        assert not span.is_valid()
        assert span.filename == ""
        assert span.begin_line == -1

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


class TestSpanTrackerColumnConvention:
    """Columns are 1-indexed characters, matching ``include/pypto/ir/span.h``.

    ``ast`` nodes carry a 0-indexed ``col_offset`` counted in UTF-8 *bytes*;
    ``SpanTracker`` converts both aspects on the way into ``ir.Span``.
    """

    def test_left_margin_statement_is_column_one(self):
        """col_offset 0 becomes column 1 — and the span is therefore valid.

        Under the previous 0-indexed emission this span had column 0, which
        ``Span::is_valid()`` rejects, silently dropping the location from every
        diagnostic anchored on a left-margin node.
        """
        source = "x = 42"
        tracker = SpanTracker("test.py", source.split("\n"))

        span = tracker.get_span(ast.parse(source).body[0])

        assert span.begin_column == 1
        assert span.is_valid()

    def test_column_matches_source_offset(self):
        """The column round-trips to the token when read as a 1-indexed position."""
        source = "y = foo(1)"
        tracker = SpanTracker("test.py", source.split("\n"))

        assign = ast.parse(source).body[0]
        assert isinstance(assign, ast.Assign)
        span = tracker.get_span(assign.value)

        assert source[span.begin_column - 1 :].startswith("foo(1)")

    def test_col_offset_is_indentation_aware(self):
        """``col_offset`` (dedent compensation) is added on top of the 1-indexed column."""
        source = "x = 42"
        tracker = SpanTracker("test.py", source.split("\n"), col_offset=4)

        span = tracker.get_span(ast.parse(source).body[0])

        assert span.begin_column == 5  # 4 spaces of stripped indentation, then column 1

    def test_multibyte_prefix_yields_character_column(self):
        """A multi-byte character before the token must not shift the column.

        ``ast`` reports ``col_offset`` in UTF-8 bytes, so a non-ASCII prefix
        would otherwise push the column past the token it names.
        """
        source = "y = f('é', bar)"
        tracker = SpanTracker("test.py", source.split("\n"))

        assign = ast.parse(source).body[0]
        assert isinstance(assign, ast.Assign)
        call = assign.value
        assert isinstance(call, ast.Call)
        bar = call.args[1]
        span = tracker.get_span(bar)

        # 'é' is 2 bytes but 1 character: the byte offset would overshoot by one.
        assert bar.col_offset == source.index("bar") + 1
        assert span.begin_column == source.index("bar") + 1
        assert source[span.begin_column - 1 :].startswith("bar")


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
        # Map entries hold AST coordinates, so col 8 becomes 1-indexed column 9.
        assert span.begin_column == 9

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
