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

class InternalError(Exception):
    """Exception raised when an internal system error occurs"""

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
