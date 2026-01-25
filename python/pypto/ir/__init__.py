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
- Enhanced type constructors (e.g., TensorType with integer shape support)
"""

# Re-export all core IR types and functions from native module
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import *  # noqa: F403

# Import operation modules
from . import op

# Import IR Builder
from .builder import IRBuilder

# Import high-level API functions
from .compile import compile

# Import parser DSL APIs
from .parser import Tensor, function, load, parse, range, yield_

# Import PassManager and OptimizationStrategy
from .pass_manager import OptimizationStrategy, PassManager

# Import python_print utility
from .printer import python_print

# Import TensorType and TileType with enhanced __init__ that supports integer shapes
# This patches the native TensorType and TileType classes to accept integer shapes
from .type import TensorType, TileType

# Export common DataType values for convenience
FP4 = DataType.FP4
FP8 = DataType.FP8
FP16 = DataType.FP16
FP32 = DataType.FP32
BF16 = DataType.BF16
HF4 = DataType.HF4
HF8 = DataType.HF8
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
BOOL = DataType.BOOL


__all__ = [
    "op",
    "IRBuilder",
    "TensorType",
    "TileType",
    "python_print",
    "compile",
    "PassManager",
    "OptimizationStrategy",
    "function",
    "parse",
    "load",
    "range",
    "yield_",
    "Tensor",
]  # fmt: skip
