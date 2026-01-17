# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO IR module with tensor operations.

This module provides:
- Re-exports of all core IR types from pypto_core.ir
- Organized operation namespaces (e.g., op.tensor.create)
- IR Builder for incremental IR construction
- Helper utilities
"""

from pypto.pypto_core import DataType  # noqa: F401

# Re-export all core IR types and functions
from pypto.pypto_core.ir import *  # noqa: F401, F403

# Import operation modules
from . import op

# Import IR Builder
from .builder import IRBuilder  # noqa: F401

# Expose common DataType constants for convenience
FP16 = DataType.FP16
FP32 = DataType.FP32
INT32 = DataType.INT32
INT64 = DataType.INT64


__all__ = [
    "op",
    "IRBuilder",
    "FP16",
    "FP32",
    "INT32",
    "INT64",
]
