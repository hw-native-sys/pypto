# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser error exceptions with rich diagnostic information."""

from pypto.pypto_core import ir


class ParserError(Exception):
    """Base class for all parser errors with diagnostic information.

    This exception captures detailed context about parsing errors including
    source location, error message, and optional hints for fixing the error.
    """

    def __init__(
        self,
        message: str,
        span: ir.Span | None = None,
        hint: str | None = None,
        note: str | None = None,
        source_lines: list[str] | None = None,
    ):
        """Initialize parser error.

        Args:
            message: Error message describing what went wrong
            span: Source location where error occurred
            hint: Optional hint for how to fix the error
            note: Optional additional note about the error
            source_lines: Optional source code lines for context
        """
        super().__init__(message)
        self.message = message

        # Extract span information to avoid keeping C++ objects alive
        # This prevents memory leaks when exceptions are caught and held
        if span is not None:
            self.span = {
                "filename": getattr(span, "filename", None),
                "line": getattr(span, "begin_line", 0),
                "column": getattr(span, "begin_column", 0),
                "file": getattr(span, "filename", None),  # For compatibility
                "begin_line": getattr(span, "begin_line", 0),
                "begin_column": getattr(span, "begin_column", 0),
            }
        else:
            self.span = None

        self.hint = hint
        self.note = note
        self.source_lines = source_lines


class ParserSyntaxError(ParserError):
    """Raised when DSL syntax is violated."""

    pass


class ParserTypeError(ParserError):
    """Raised when type annotation is incorrect or missing."""

    pass


class UndefinedVariableError(ParserError):
    """Raised when referencing an undefined variable."""

    pass


class SSAViolationError(ParserError):
    """Raised when SSA property is violated (variable redefinition)."""

    def __init__(
        self,
        message: str,
        span: ir.Span | None = None,
        hint: str | None = None,
        note: str | None = None,
        source_lines: list[str] | None = None,
        previous_span: ir.Span | None = None,
    ):
        """Initialize SSA violation error.

        Args:
            message: Error message describing what went wrong
            span: Source location where error occurred
            hint: Optional hint for how to fix the error
            note: Optional additional note about the error
            source_lines: Optional source code lines for context
            previous_span: Optional previous definition location
        """
        super().__init__(message, span, hint, note, source_lines)

        # Extract previous span information
        if previous_span is not None:
            self.previous_span = {
                "filename": getattr(previous_span, "filename", None),
                "line": getattr(previous_span, "begin_line", 0),
                "column": getattr(previous_span, "begin_column", 0),
                "file": getattr(previous_span, "filename", None),  # For compatibility
                "begin_line": getattr(previous_span, "begin_line", 0),
                "begin_column": getattr(previous_span, "begin_column", 0),
            }
        else:
            self.previous_span = None


class UnsupportedFeatureError(ParserError):
    """Raised when using an unsupported Python feature in DSL."""

    pass


class InvalidOperationError(ParserError):
    """Raised when an operation is invalid or unknown."""

    pass


class ScopeIsolationError(ParserError):
    """Raised when scope isolation is violated."""

    pass


__all__ = [
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "UndefinedVariableError",
    "SSAViolationError",
    "UnsupportedFeatureError",
    "InvalidOperationError",
    "ScopeIsolationError",
]
