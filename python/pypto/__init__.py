# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO - Python Tensor Operations Library

This package provides Python bindings for the PyPTO C++ library.
"""

from typing import cast

# Import IR module (includes operations and core IR types)
from . import ir
from .pypto_core import (
    DataType,
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
    testing,
)

# Convenient dtype constants
DT_BOOL: DataType = cast(DataType, DataType.BOOL)
DT_INT4: DataType = cast(DataType, DataType.INT4)
DT_INT8: DataType = cast(DataType, DataType.INT8)
DT_INT16: DataType = cast(DataType, DataType.INT16)
DT_INT32: DataType = cast(DataType, DataType.INT32)
DT_INT64: DataType = cast(DataType, DataType.INT64)
DT_UINT4: DataType = cast(DataType, DataType.UINT4)
DT_UINT8: DataType = cast(DataType, DataType.UINT8)
DT_UINT16: DataType = cast(DataType, DataType.UINT16)
DT_UINT32: DataType = cast(DataType, DataType.UINT32)
DT_UINT64: DataType = cast(DataType, DataType.UINT64)
DT_FP4: DataType = cast(DataType, DataType.FP4)
DT_FP8: DataType = cast(DataType, DataType.FP8)
DT_FP16: DataType = cast(DataType, DataType.FP16)
DT_FP32: DataType = cast(DataType, DataType.FP32)
DT_BF16: DataType = cast(DataType, DataType.BF16)
DT_HF4: DataType = cast(DataType, DataType.HF4)
DT_HF8: DataType = cast(DataType, DataType.HF8)

__all__ = [
    # Modules
    "testing",
    "ir",
    # Logging framework
    "InternalError",
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
    # Dtype constants
    "DT_BOOL",
    "DT_INT4",
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
    "DT_INT64",
    "DT_UINT4",
    "DT_UINT8",
    "DT_UINT16",
    "DT_UINT32",
    "DT_UINT64",
    "DT_FP4",
    "DT_FP8",
    "DT_FP16",
    "DT_FP32",
    "DT_BF16",
    "DT_HF4",
    "DT_HF8",
]

__version__ = "0.1.0"
