# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Span tracking for preserving source location information during parsing."""

import ast
from collections.abc import Sequence
from contextvars import ContextVar

from pypto.pypto_core import ir

# Generated-program line → (orig_file, orig_line, orig_col), set by
# ``pl.parse(..., source_map=...)`` around the exec that triggers parsing.
# ``@pl.jit`` populates it so spans (and thus parse and compile error
# diagnostics) point at the user's real source instead of the synthesized
# ``<jit:name>`` text. ``None`` (the default) ⇒ no remapping. See issue #1612.
active_source_map: ContextVar[dict[int, tuple[str, int, int]] | None] = ContextVar(
    "pypto_jit_source_map", default=None
)


def ast_column_to_span_column(line: str | None, col_offset: int) -> int:
    """Convert a 0-indexed AST ``col_offset`` to a 1-indexed ``Span`` column.

    Two coordinate systems meet here. An ``ast`` node carries a 1-indexed
    ``lineno`` but a 0-indexed ``col_offset``, and that offset counts **UTF-8
    bytes** rather than characters -- a multi-byte character earlier on the line
    shifts it right. ``ir.Span`` columns are 1-indexed *character* positions
    (``include/pypto/ir/span.h``), which is what editors, ``Span::to_string()``
    and the diagnostics renderer expect. CPython makes the same conversion when
    it turns an internal ``col_offset`` into the reported ``SyntaxError.offset``.

    Args:
        line: The source line the offset refers to, used to translate the byte
            offset into a character index. ``None`` when the line is not
            available; the offset is then used as-is, which is exact for
            ASCII-only lines.
        col_offset: 0-indexed UTF-8 byte offset taken from an AST node.

    Returns:
        The 1-indexed character column to store in an ``ir.Span``.
    """
    if line is None:
        return col_offset + 1
    # ``errors="ignore"`` guards against an offset landing mid-sequence; valid
    # AST offsets are always on a character boundary.
    return len(line.encode("utf-8")[:col_offset].decode("utf-8", errors="ignore")) + 1


class SpanTracker:
    """Tracks source locations from AST nodes to IR spans."""

    def __init__(
        self,
        source_file: str,
        source_lines: Sequence[str],
        line_offset: int = 0,
        col_offset: int = 0,
        source_map: dict[int, tuple[str, int, int]] | None = None,
    ):
        """Initialize span tracker.

        Args:
            source_file: Path to the source file
            source_lines: List of source code lines (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset in *characters* to add to AST column numbers,
                restoring the indentation stripped before ``ast.parse``
            source_map: Optional generated-line → ``(orig_file, orig_line,
                orig_col)`` map. When a node's emitted line is present, the span
                is remapped to that original location (#1612). Defaults to the
                map active on :data:`active_source_map` for the current parse.
        """
        self.source_file = source_file
        self.source_lines = source_lines
        self.line_offset = line_offset
        self.col_offset = col_offset
        self.source_map = source_map if source_map is not None else active_source_map.get()

    def get_span(self, ast_node: ast.AST | None) -> ir.Span:
        """Extract span from AST node.

        Columns are converted from the AST's 0-indexed byte offsets to the
        1-indexed character columns ``ir.Span`` documents -- see
        :func:`ast_column_to_span_column`.

        Args:
            ast_node: AST node with line/column information

        Returns:
            IR span corresponding to the AST node location
        """
        if ast_node is None or not hasattr(ast_node, "lineno"):
            return ir.Span.unknown()

        begin_lineno = getattr(ast_node, "lineno", 0)
        end_lineno = getattr(ast_node, "end_lineno", 0)
        begin_line = begin_lineno + self.line_offset
        remapped = self._remap(begin_line)
        if remapped is not None:
            return remapped

        return ir.Span(
            self.source_file,
            begin_line,
            self._span_column(begin_lineno, getattr(ast_node, "col_offset", 0)),
            end_lineno + self.line_offset,
            self._span_column(end_lineno, getattr(ast_node, "end_col_offset", 0)),
        )

    def get_multiline_span(self, start_node: ast.AST, end_node: ast.AST) -> ir.Span:
        """Get span covering multiple lines.

        Args:
            start_node: AST node at the start
            end_node: AST node at the end

        Returns:
            IR span covering the range from start to end
        """
        if not hasattr(start_node, "lineno") or not hasattr(end_node, "lineno"):
            return ir.Span.unknown()

        begin_lineno = getattr(start_node, "lineno", 0)
        end_lineno = getattr(end_node, "end_lineno", 0)
        begin_line = begin_lineno + self.line_offset
        remapped = self._remap(begin_line)
        if remapped is not None:
            return remapped

        return ir.Span(
            self.source_file,
            begin_line,
            self._span_column(begin_lineno, getattr(start_node, "col_offset", 0)),
            end_lineno + self.line_offset,
            self._span_column(end_lineno, getattr(end_node, "end_col_offset", 0)),
        )

    def _span_column(self, ast_lineno: int, col_offset: int) -> int:
        """Convert an AST ``(lineno, col_offset)`` pair to a 1-indexed Span column.

        ``self.col_offset`` (the indentation stripped before ``ast.parse``) is a
        character count, so it is added after the byte-to-character conversion.

        Args:
            ast_lineno: 1-indexed AST line number, *before* ``line_offset`` is
                applied -- it indexes into the dedented ``source_lines``.
            col_offset: 0-indexed UTF-8 byte offset from the AST node.

        Returns:
            The 1-indexed character column in the original source file.
        """
        return ast_column_to_span_column(self._line_at(ast_lineno), col_offset) + self.col_offset

    def _line_at(self, ast_lineno: int) -> str | None:
        """Return the dedented source line for a 1-indexed AST line number, if known."""
        if 1 <= ast_lineno <= len(self.source_lines):
            return self.source_lines[ast_lineno - 1]
        return None

    def _remap(self, begin_line: int) -> ir.Span | None:
        """Remap a generated begin line to an original-source span, or None.

        Returns ``None`` when no source map is active or the line is absent
        (e.g. a synthesized statement) — the caller then emits the generated
        coordinates. The mapped span underlines from the statement's original
        start column (#1612: alpha-renaming makes exact end columns unreliable).

        Map entries hold AST coordinates, so the column needs the same
        0-to-1-indexed shift as any other span. No byte-to-character correction
        is applied: an entry's column is a *statement* start, whose prefix is
        pure ASCII indentation, so its byte and character offsets coincide.
        """
        if not self.source_map:
            return None
        mapped = self.source_map.get(begin_line)
        if mapped is None:
            return None
        orig_file, orig_line, orig_col = mapped
        column = ast_column_to_span_column(None, orig_col)
        return ir.Span(orig_file, orig_line, column, orig_line, column)


__all__ = ["SpanTracker", "active_source_map", "ast_column_to_span_column"]
