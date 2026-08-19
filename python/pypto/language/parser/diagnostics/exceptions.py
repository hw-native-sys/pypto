# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser error exceptions with rich diagnostic information."""

import re
from typing import Final

from pypto.pypto_core import InternalError, ir

# Bug-class exceptions: a failed internal invariant is a PyPTO bug, not a bad kernel.
# Re-wrapping one as a user-facing ParserError hides both its type and the C++ traceback
# that diagnoses it, so every parser handler that broadly catches `Exception` lets these
# through untouched. See `.claude/rules/error-checking.md` for the CHECK /
# INTERNAL_CHECK split this preserves.
BUG_CLASS_EXCEPTIONS: Final = (InternalError, AssertionError)


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


_CHECK_TAIL_MARKER = "Check failed: "

# What a check with no ``<<`` payload leaves behind once its tail is stripped. ``CHECK`` is
# the *user*-error macro (see .claude/rules/error-checking.md); ``INTERNAL_CHECK`` is the
# bug-class one, and this helper runs on messages from both. So the wording must not read
# as "you found a compiler bug" -- it says the check was silent and names the escape hatch.
_UNSPECIFIED_CHECK_MESSAGE = (
    "The operation was rejected by a backend check that reported no message "
    "(re-run with PTO_BACKTRACE=1 to see which check failed)"
)

# The ``[<file>:<line>:<column>]`` suffix ``FatalLogger::~FatalLogger``
# (``include/pypto/core/logging.h``) appends for the ``*_SPAN`` check macros, written
# *before* the newline that starts the ``Check failed:`` tail and therefore surviving the
# tail strip below. ``Span::is_valid()`` guarantees a positive line; the column is either
# positive or the ``-1`` sentinel. ``[^\[\]]*`` keeps the filename from eating an earlier
# bracket, and the anchor keeps the match to the very end of the payload.
_TRAILING_SPAN_RE: Final = re.compile(r"\s*\[[^\[\]]*:\d+:-?\d+\]\Z")


def concise_error_message(exc: Exception, strip_trailing_span: bool = False) -> str:
    """Extract a concise user-facing message from an exception.

    Strips C++ internal details (stack traces and CHECK macro output) that are
    useful for debugging but noisy in parser error reports. The full details
    remain accessible via PTO_BACKTRACE=1 which shows the Python traceback
    containing the original exception with all C++ information.

    A check that fired without a ``<<`` message leaves nothing behind once its tail
    is stripped; that reports as :data:`_UNSPECIFIED_CHECK_MESSAGE` rather than as an
    empty string, because an empty bold ``Error:`` header is strictly worse than the
    C++ noise it replaced. An exception that was simply raised with an empty message
    keeps its empty message -- only a stripped check tail earns the fallback.

    Args:
        exc: Exception to extract the user-facing message from.
        strip_trailing_span: Also drop the ``[<file>:<line>:<column>]`` location that a
            ``CHECK_SPAN`` / ``INTERNAL_CHECK_SPAN`` leaves at the end of the payload.
            Opt-in, because the caller must have a better place to show the location:
            pass it only when the resulting :class:`ParserError` carries a ``span=`` of
            its own, so the renderer's ``-->`` arrow and code snippet still point the
            user at a source line. Callers that raise without a span (the parse-function
            wrappers in ``decorator.py``) must leave it off -- the inline location is the
            only one the user would get.

    Returns:
        The user-facing message, free of C++ traceback and ``Check failed:`` noise.
    """
    msg = str(exc)
    # Strip "C++ Traceback ..." block appended by GetFullMessage()
    pos = msg.find("\n\nC++ Traceback")
    if pos != -1:
        msg = msg[:pos]
    # Strip "No stack trace available ..." block (debug builds without symbols)
    pos = msg.find("\n\nNo stack trace available")
    if pos != -1:
        msg = msg[:pos]
    # Strip the "Check failed: <expr> at <file>:<line>" tail that FatalLogger::~FatalLogger
    # (core/logging.h:600-606) glues onto every CHECK. It is always the last line; whatever
    # precedes it is the ``<<`` payload the op wrote for the user. FatalLogger emits the
    # "\n" unconditionally, so a message-less check puts the rfind hit at 0 -- that
    # truncation is what empties the message, handled by the fallback below.
    stripped_check_tail = False
    if msg.startswith(_CHECK_TAIL_MARKER):
        msg = ""
        stripped_check_tail = True
    else:
        pos = msg.rfind("\n" + _CHECK_TAIL_MARKER)
        if pos != -1:
            msg = msg[:pos]
            stripped_check_tail = True
    # Gated on the tail: only FatalLogger writes the inline span, and it always writes the
    # tail alongside it. A pure-Python message that happens to end in bracketed
    # colon-separated integers (a slice, say) is therefore never touched.
    if strip_trailing_span and stripped_check_tail:
        msg = _TRAILING_SPAN_RE.sub("", msg)
    msg = msg.strip()
    # Gate on the tail actually having been removed, not on "something was there before":
    # `GetFullMessage()` always appends a traceback (or the "no stack trace" note), so a
    # bare `pypto::ValueError("")` under PTO_BACKTRACE=1 also strips to empty -- and calling
    # that a silent backend check would name a check that never ran, while telling the user
    # to enable a flag they already have on.
    if not msg and stripped_check_tail:
        return _UNSPECIFIED_CHECK_MESSAGE
    return msg


__all__ = [
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "UndefinedVariableError",
    "SSAViolationError",
    "UnsupportedFeatureError",
    "InvalidOperationError",
    "ScopeIsolationError",
    "concise_error_message",
]
