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

from enum import IntEnum

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

class DataType(IntEnum):
    """Enumeration of all supported data types in PyPTO"""

    # Boolean type
    BOOL = 0  # Boolean (true/false)

    # Signed integer types
    INT4 = 1  # 4-bit signed integer
    INT8 = 2  # 8-bit signed integer
    INT16 = 3  # 16-bit signed integer
    INT32 = 4  # 32-bit signed integer
    INT64 = 5  # 64-bit signed integer

    # Unsigned integer types
    UINT4 = 6  # 4-bit unsigned integer
    UINT8 = 7  # 8-bit unsigned integer
    UINT16 = 8  # 16-bit unsigned integer
    UINT32 = 9  # 32-bit unsigned integer
    UINT64 = 10  # 64-bit unsigned integer

    # Floating point types
    FP4 = 11  # 4-bit floating point
    FP8 = 12  # 8-bit floating point
    FP16 = 13  # 16-bit floating point (IEEE 754 half precision)
    FP32 = 14  # 32-bit floating point (IEEE 754 single precision)
    BF16 = 15  # 16-bit brain floating point

    # Hisilicon float types
    HF4 = 16  # 4-bit Hisilicon float
    HF8 = 17  # 8-bit Hisilicon float

def get_dtype_bit(dtype: DataType) -> int:
    """
    Get the size in bits of a data type. Returns the actual bit size for sub-byte types
    (e.g., 4 bits for INT4, 8 bits for INT8, etc.).

    Args:
        dtype: The data type to query
    Returns:
        The size in bits of the data type
    """

def dtype_to_string(dtype: DataType) -> str:
    """
    Get a human-readable string name for a data type.

    Args:
        dtype: The data type to convert to string
    Returns:
        The string representation of the data type
    """

def is_float(dtype: DataType) -> bool:
    """
    Check if a data type is a floating point type (FP4, FP8, FP16, FP32, BF16, HF4, HF8).

    Args:
        dtype: The data type to check
    Returns:
        True if the data type is a floating point type, False otherwise
    """

def is_signed_int(dtype: DataType) -> bool:
    """
    Check if a data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).

    Args:
        dtype: The data type to check
    Returns:
        True if the data type is a signed integer type, False otherwise
    """

def is_unsigned_int(dtype: DataType) -> bool:
    """
    Check if a data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).

    Args:
        dtype: The data type to check
    Returns:
        True if the data type is an unsigned integer type, False otherwise
    """

def is_int(dtype: DataType) -> bool:
    """
    Check if a data type is any integer type (signed or unsigned).

    Args:
        dtype: The data type to check
    Returns:
        True if the data type is any integer type, False otherwise
    """

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
    # DataType enum and utilities
    "DataType",
    "get_dtype_bit",
    "dtype_to_string",
    "is_float",
    "is_signed_int",
    "is_unsigned_int",
    "is_int",
]
