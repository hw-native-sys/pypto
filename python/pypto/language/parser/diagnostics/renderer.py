# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Error rendering and formatting for pretty error messages."""

import os
import sys

from .exceptions import ParserError, SSAViolationError


class ErrorRenderer:
    """Renders parser errors in a user-friendly format with code highlighting.

    This renderer formats errors similar to Rust/TypeScript compilers with:
    - Source code context
    - Caret (^) highlighting of error location
    - Line numbers
    - Optional color support
    - Help/hint messages
    """

    def __init__(self, use_color: bool | None = None):
        """Initialize error renderer.

        Args:
            use_color: Whether to use ANSI colors. If None, auto-detect based on terminal.
        """
        if use_color is None:
            # Auto-detect color support
            use_color = self._supports_color()
        self.use_color = use_color

    def _supports_color(self) -> bool:
        """Check if terminal supports ANSI colors.

        Returns:
            True if colors are supported
        """
        # Check if output is a TTY and TERM is set
        if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
            return False

        # Check for NO_COLOR environment variable
        if os.environ.get("NO_COLOR"):
            return False

        # Check for FORCE_COLOR
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check TERM
        term = os.environ.get("TERM", "")
        if term == "dumb":
            return False

        return True

    def _colorize(self, text: str, color_code: str) -> str:
        """Apply ANSI color to text.

        Args:
            text: Text to colorize
            color_code: ANSI color code

        Returns:
            Colorized text or plain text if colors disabled
        """
        if not self.use_color:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _red(self, text: str) -> str:
        """Make text red (for errors)."""
        return self._colorize(text, "1;31")

    def _cyan(self, text: str) -> str:
        """Make text cyan (for file paths)."""
        return self._colorize(text, "36")

    def _blue(self, text: str) -> str:
        """Make text blue (for line numbers)."""
        return self._colorize(text, "34")

    def _green(self, text: str) -> str:
        """Make text green (for hints)."""
        return self._colorize(text, "32")

    def _bold(self, text: str) -> str:
        """Make text bold."""
        return self._colorize(text, "1")

    def render(self, error: ParserError) -> str:
        """Render a parser error as a formatted string.

        Args:
            error: Parser error to render

        Returns:
            Formatted error message
        """
        lines = []

        # Error header
        lines.append(self._red(self._bold(f"Error: {error.message}")))

        # Source location and code context
        if error.span:
            location = self._format_location(error.span)
            if location:
                lines.append(self._cyan(f"  --> {location}"))

        # Code context with line numbers and caret highlighting
        if error.span and error.source_lines:
            lines.extend(self._render_code_context(error))

        # Previous definition location (for SSA violations)
        if isinstance(error, SSAViolationError) and error.previous_span:
            lines.extend(self._render_previous_definition(error))

        # Help and note messages
        if error.hint:
            lines.append(self._green(f"   = help: {error.hint}"))
        if error.note:
            lines.append(self._cyan(f"   = note: {error.note}"))

        return "\n".join(lines)

    def _extract_span_info(self, span) -> tuple[str, int, int]:
        """Extract filename, line, and column from a span (dict or object).

        Args:
            span: Span object or dictionary

        Returns:
            Tuple of (filename, line, column)
        """
        if isinstance(span, dict):
            filename = span.get("filename") or span.get("file", "")
            line = span.get("begin_line") or span.get("line", 0)
            column = span.get("begin_column") or span.get("column", 0)
        else:
            filename = getattr(span, "filename", getattr(span, "file", ""))
            line = getattr(span, "begin_line", getattr(span, "line", 0))
            column = getattr(span, "begin_column", getattr(span, "column", 0))
        return filename, line, column

    def _format_location(self, span) -> str:
        """Format a span as a file:line:column location string.

        Args:
            span: Span object or dictionary

        Returns:
            Formatted location string
        """
        filename, line, column = self._extract_span_info(span)
        if not filename:
            return ""

        location = filename
        if line > 0:
            location += f":{line}"
            if column > 0:
                location += f":{column}"
        return location

    def _render_previous_definition(self, error: SSAViolationError) -> list[str]:
        """Render the previous definition section for SSA violations.

        Args:
            error: SSA violation error with previous_span

        Returns:
            List of formatted lines
        """
        lines = []
        _, prev_line, prev_col = self._extract_span_info(error.previous_span)

        if prev_line <= 0:
            return lines

        lines.append("")
        lines.append(self._cyan("   = note: Previous definition was here:"))

        # Show previous definition location
        location = self._format_location(error.previous_span)
        if location:
            lines.append(self._cyan(f"     --> {location}"))

        # Show previous definition code context
        if error.source_lines and prev_line <= len(error.source_lines):
            lines.extend(self._render_previous_context(error.source_lines, prev_line, prev_col))

        return lines

    def _render_previous_context(self, source_lines: list[str], line_num: int, column: int) -> list[str]:
        """Render code context for previous definition.

        Args:
            source_lines: Full source code lines
            line_num: Line number of previous definition
            column: Column number of previous definition

        Returns:
            List of formatted lines
        """
        lines = []
        line_num_width = len(str(min(len(source_lines), line_num + 1)))
        prefix = "     "  # 5 spaces to align with "-->" above

        lines.append(self._blue(prefix + " " * line_num_width + " |"))

        # Show 1 line before, the definition line, and 1 line after
        start = max(1, line_num - 1)
        end = min(len(source_lines), line_num + 1)

        for i in range(start, end + 1):
            if i > len(source_lines):
                break

            line_content = source_lines[i - 1].rstrip()
            formatted_line_num = self._blue(f"{i:>{line_num_width}} |")

            if i == line_num:
                # Highlight the previous definition line
                lines.append(f"{prefix}{formatted_line_num} {line_content}")
                # Add caret line highlighting the token
                caret_line = self._render_previous_caret(line_content, column, line_num_width)
                lines.append(f"{prefix}{caret_line}")
            else:
                lines.append(f"{prefix}{formatted_line_num} {line_content}")

        lines.append(self._blue(prefix + " " * line_num_width + " |"))
        return lines

    def _render_previous_caret(self, line_content: str, column: int, line_num_width: int) -> str:
        """Render caret line for previous definition (cyan color).

        Args:
            line_content: Source line content
            column: Column position
            line_num_width: Width for line number alignment

        Returns:
            Formatted caret line
        """
        padding = " " * line_num_width
        spaces = " " * column

        # Determine token length for caret
        caret_length = self._calculate_token_length(line_content, column)
        carets = "^" * caret_length

        return f"{self._blue(padding + ' |')} {spaces}{self._cyan(carets)}"

    def _calculate_token_length(self, line_content: str, column: int) -> int:
        """Calculate the length of token at given column for caret highlighting.

        Args:
            line_content: Source line content
            column: Starting column position

        Returns:
            Length of token (minimum 1)
        """
        if column >= len(line_content):
            return 1

        token_chars = 0
        for i in range(column, len(line_content)):
            char = line_content[i]
            if char.isalnum() or char == "_":
                token_chars += 1
            else:
                break

        return token_chars if token_chars > 0 else 1

    def _render_code_context(self, error: ParserError) -> list[str]:
        """Render code context with line numbers and caret highlighting.

        Args:
            error: Parser error with span and source lines

        Returns:
            List of formatted lines
        """
        lines = []
        _, error_line, error_col = self._extract_span_info(error.span)

        if error_line <= 0 or not error.source_lines:
            return lines

        source_lines = error.source_lines
        context_before = 2
        context_after = 2

        start_line = max(1, error_line - context_before)
        end_line = min(len(source_lines), error_line + context_after)
        line_num_width = len(str(end_line))

        # Empty line before code
        lines.append(self._blue(" " * (line_num_width + 1) + "|"))

        # Render context lines
        for line_num in range(start_line, end_line + 1):
            if line_num > len(source_lines):
                break

            line_content = source_lines[line_num - 1].rstrip()
            formatted_line_num = self._blue(f"{line_num:>{line_num_width}} |")

            if line_num == error_line:
                lines.append(f"{formatted_line_num} {line_content}")
                caret_line = self._render_caret_line(line_content, error_col, line_num_width, error.message)
                lines.append(caret_line)
            else:
                lines.append(f"{formatted_line_num} {line_content}")

        # Empty line after code
        lines.append(self._blue(" " * (line_num_width + 1) + "|"))

        return lines

    def _render_caret_line(self, source_line: str, column: int, line_num_width: int, message: str) -> str:
        """Render the caret (^) line pointing to the error.

        Args:
            source_line: Source code line
            column: Column position (0-based)
            line_num_width: Width of line numbers for alignment
            message: Short error message to display

        Returns:
            Formatted caret line
        """
        # Calculate position handling tabs
        position = 0
        for i, char in enumerate(source_line):
            if i >= column:
                break
            position += 4 if char == "\t" else 1

        # Build caret line
        padding = " " * (line_num_width + 1)
        spaces = " " * position
        caret_length = self._calculate_token_length(source_line, column)
        carets = self._red("^" * caret_length)

        # Add short inline message if available
        inline_msg = ""
        if message:
            short_msg = message.split(".")[0].split("\n")[0]
            if len(short_msg) < 50:
                inline_msg = self._red(f" {short_msg}")

        return f"{self._blue(padding + '|')} {spaces}{carets}{inline_msg}"


__all__ = ["ErrorRenderer"]
