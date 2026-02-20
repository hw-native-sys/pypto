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

from . import codegen, ir, passes, testing
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
    FP8E4M3FN: DataType  # 8-bit floating point (IEEE 754 e4m3fn format)
    FP8E5M2: DataType  # 8-bit floating point (IEEE 754 e5m2 format)
    FP16: DataType  # 16-bit floating point (IEEE 754 half precision)
    FP32: DataType  # 32-bit floating point (IEEE 754 single precision)
    BF16: DataType  # 16-bit brain floating point
    HF4: DataType  # 4-bit Hisilicon float
    HF8: DataType  # 8-bit Hisilicon float
    INDEX: DataType  # Index type for loop variables, dimensions, offsets (alias for INT64)

    def get_bit(self) -> int:
        """
        Get the size in bits of this data type. Returns the actual bit size for sub-byte types
        (e.g., 4 bits for INT4, 8 bits for INT8, etc.).

        Returns:
            The size in bits of the data type
        """

    def to_string(self) -> str:
        """
        Get a human-readable string name for this data type.

        Returns:
            The string representation of the data type
        """

    def to_c_type_string(self) -> str:
        """
        Get C style type string for code generation (e.g., 'float', 'half', 'int32_t').

        Returns:
            C style type string

        Raises:
            ValueError: If the data type is not supported for code generation
        """

    def is_float(self) -> bool:
        """
        Check if this data type is a floating point type (FP4, FP8, FP16, FP32, BF16, HF4, HF8).

        Returns:
            True if the data type is a floating point type, False otherwise
        """

    def is_signed_int(self) -> bool:
        """
        Check if this data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).

        Returns:
            True if the data type is a signed integer type, False otherwise
        """

    def is_unsigned_int(self) -> bool:
        """
        Check if this data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).

        Returns:
            True if the data type is an unsigned integer type, False otherwise
        """

    def is_int(self) -> bool:
        """
        Check if this data type is any integer type (signed or unsigned).

        Returns:
            True if the data type is any integer type, False otherwise
        """

    def code(self) -> int:
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
    # Core IR types
    "ir",
    # Pass transformations
    "passes",
    # Testing utilities
    "testing",
    # Code generation
    "codegen",
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
