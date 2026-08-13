# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint: disable=unused-argument

from enum import IntEnum

from pypto.pypto_core.ir import Span

class Error(Exception):
    """Base class for PyPTO errors that have no more specific Python counterpart.

    Raised for `pypto::Error` subclasses without a dedicated translation, notably
    `VerificationError` from the IR verifier.
    """

class InternalError(RuntimeError):
    """Exception raised when an internal system error occurs.

    Registered against `PyExc_RuntimeError`, so it is a sibling of `Error` rather than a
    subclass -- the C++ hierarchy does not carry over to Python.
    """

class LogLevel(IntEnum):
    """Enumeration of available log levels"""

    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4
    EVENT = 5
    NONE = 6

def set_log_level(level: LogLevel) -> None:
    """Set the global log level threshold. Only messages at or above this level will be logged."""

def get_log_level() -> LogLevel:
    """Get the global log level threshold.

    Ignores any thread-local override installed by `_set_thread_log_level`, so
    `get_log_level()` / `set_log_level()` form a save-restore pair.
    """

def _set_thread_log_level(level: LogLevel) -> None:
    """Override the log level threshold for the calling thread."""

def _clear_thread_log_level() -> None:
    """Remove the calling thread's log level override."""

def log_debug(message: str) -> None:
    """Log a message at the DEBUG level"""

def log_info(message: str) -> None:
    """Log a message at the INFO level"""

def log_warn(message: str) -> None:
    """Log a message at the WARN level"""

def log_error(message: str) -> None:
    """Log a message at the ERROR level"""

def log_fatal(message: str) -> None:
    """Log a message at the FATAL level"""

def log_event(message: str) -> None:
    """Log a message at the EVENT level"""

def check(condition: bool, message: str) -> None:
    """Check a condition and throw ValueError if it fails"""

def internal_check(condition: bool, message: str) -> None:
    """Check an internal invariant and throw InternalError if it fails"""

def internal_check_span(condition: bool, message: str, span: Span) -> None:
    """Check an internal invariant with IR source location and throw InternalError if it fails."""
