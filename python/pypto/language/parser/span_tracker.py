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

from pypto.pypto_core import ir


class SpanTracker:
    """Tracks source locations from AST nodes to IR spans."""

    def __init__(
        self, source_file: str, source_lines: Sequence[str], line_offset: int = 0, col_offset: int = 0
    ):
        """Initialize span tracker.

        Args:
            source_file: Path to the source file
            source_lines: List of source code lines (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
        """
        self.source_file = source_file
        self.source_lines = source_lines
        self.line_offset = line_offset
        self.col_offset = col_offset

    def get_span(self, ast_node: ast.AST | None) -> ir.Span:
        """Extract span from AST node.

        Args:
            ast_node: AST node with line/column information

        Returns:
            IR span corresponding to the AST node location
        """
        if ast_node is None or not hasattr(ast_node, "lineno"):
            return ir.Span.unknown()

        return ir.Span(
            self.source_file,
            getattr(ast_node, "lineno", 0) + self.line_offset,
            getattr(ast_node, "col_offset", 0) + self.col_offset,
            getattr(ast_node, "end_lineno", 0) + self.line_offset,
            getattr(ast_node, "end_col_offset", 0) + self.col_offset,
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

        return ir.Span(
            self.source_file,
            getattr(start_node, "lineno", 0) + self.line_offset,
            getattr(start_node, "col_offset", 0) + self.col_offset,
            getattr(end_node, "end_lineno", 0) + self.line_offset,
            getattr(end_node, "end_col_offset", 0) + self.col_offset,
        )


__all__ = ["SpanTracker"]
