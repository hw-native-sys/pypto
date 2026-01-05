# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint: disable=unused-argument
"""
PyPTO - Python Tensor Operations Library

This package provides Python bindings for the PyPTO C++ library.
"""

from . import ir, testing
from .logging import (
    InternalError,
    LogLevel,
    check,
    internal_check,
    log_debug,
    log_error,
    log_event,
    log_fatal,
    log_info,
    log_warn,
    set_log_level,
)

class DataType:
    """Data type representation for PyPTO tensors and operations"""

    # Static type constants
    BOOL: DataType  # Boolean (true/false)
    INT4: DataType  # 4-bit signed integer
    INT8: DataType  # 8-bit signed integer
    INT16: DataType  # 16-bit signed integer
    INT32: DataType  # 32-bit signed integer
    INT64: DataType  # 64-bit signed integer
    UINT4: DataType  # 4-bit unsigned integer
    UINT8: DataType  # 8-bit unsigned integer
    UINT16: DataType  # 16-bit unsigned integer
    UINT32: DataType  # 32-bit unsigned integer
    UINT64: DataType  # 64-bit unsigned integer
    FP4: DataType  # 4-bit floating point
    FP8: DataType  # 8-bit floating point
    FP16: DataType  # 16-bit floating point (IEEE 754 half precision)
    FP32: DataType  # 32-bit floating point (IEEE 754 single precision)
    BF16: DataType  # 16-bit brain floating point
    HF4: DataType  # 4-bit hybrid float
    HF8: DataType  # 8-bit hybrid float

    def GetBit(self) -> int:
        """
        Get the size in bits of this data type. Returns the actual bit size for sub-byte types
        (e.g., 4 bits for INT4, 8 bits for INT8, etc.).

        Returns:
            The size in bits of the data type
        """

    def ToString(self) -> str:
        """
        Get a human-readable string name for this data type.

        Returns:
            The string representation of the data type
        """

    def IsFloat(self) -> bool:
        """
        Check if this data type is a floating point type (FP4, FP8, FP16, FP32, BF16, HF4, HF8).

        Returns:
            True if the data type is a floating point type, False otherwise
        """

    def IsSignedInt(self) -> bool:
        """
        Check if this data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).

        Returns:
            True if the data type is a signed integer type, False otherwise
        """

    def IsUnsignedInt(self) -> bool:
        """
        Check if this data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).

        Returns:
            True if the data type is a signed integer type, False otherwise
        """

    def IsInt(self) -> bool:
        """
        Check if this data type is any integer type (signed or unsigned).

        Returns:
            True if the data type is any integer type, False otherwise
        """

    def Code(self) -> int:
        """
        Get the underlying type code as uint8_t.

        Returns:
            The type code as an integer
        """

    def __eq__(self, other: DataType) -> bool:
        """Equality comparison operator"""

    def __ne__(self, other: DataType) -> bool:
        """Inequality comparison operator"""

    def __repr__(self) -> str:
        """String representation for debugging"""

    def __str__(self) -> str:
        """String representation for printing"""

__all__ = [
    "testing",
    # Core IR types
    "ir",
    # Error classes
    "InternalError",
    # Logging framework
    "LogLevel",
    "set_log_level",
    "log_debug",
    "log_info",
    "log_warn",
    "log_error",
    "log_fatal",
    "log_event",
    "check",
    "internal_check",
    # DataType class
    "DataType",
]
