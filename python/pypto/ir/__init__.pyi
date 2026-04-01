# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stub for pypto.ir — re-exports all public names from pypto_core.ir plus IR-level APIs.

TensorView and TileView are re-exported as the canonical C++ types from pypto_core.ir so that
type checkers treat ir.TensorView / ir.TileView as the same type as pypto_core.ir.TensorView /
pypto_core.ir.TileView.  At runtime these names resolve to Python factory objects that also
accept integer arguments, but the public type signature is unchanged.
"""

from pypto.pypto_core import DataType as DataType
from pypto.pypto_core.ir import *  # noqa: F401, F403
from pypto.pypto_core.ir import (
    IRMutator,
    IRVisitor,
    TensorType,
    TensorView,
    TileType,
    TileView,
)
from pypto.pypto_core.passes import PassContext, VerificationLevel, VerificationMode

from . import op as op
from .builder import IRBuilder
from .compile import compile
from .instruments import make_roundtrip_instrument
from .op_conversion import ConversionContext, op_conversion, register_op_conversion
from .pass_manager import OptimizationStrategy, PassManager
from .printer import python_print

# DataType aliases (mirrors runtime __init__.py)
FP4: DataType
FP8E4M3FN: DataType
FP8E5M2: DataType
FP16: DataType
FP32: DataType
BF16: DataType
HF4: DataType
HF8: DataType
INT4: DataType
INT8: DataType
INT16: DataType
INT32: DataType
INT64: DataType
UINT4: DataType
UINT8: DataType
UINT16: DataType
UINT32: DataType
UINT64: DataType
BOOL: DataType
INDEX: DataType

__all__ = [
    "op",
    "IRBuilder",
    "IRMutator",
    "IRVisitor",
    "TensorType",
    "TensorView",
    "TileType",
    "TileView",
    "python_print",
    "compile",
    "PassManager",
    "OptimizationStrategy",
    "VerificationMode",
    "VerificationLevel",
    "PassContext",
    "ConversionContext",
    "op_conversion",
    "register_op_conversion",
    "make_roundtrip_instrument",
]
