# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Foundation layer: operator specs, shape utilities, and random op chain generation."""

from .config import ControlFlowConfig
from .fuzzer import OpChainConfig, OpFuzzer
from .op_specs import (
    BLOCK_BINARY_OPS,
    BLOCK_COL_EXPAND_OPS,
    BLOCK_MATRIX_OPS,
    BLOCK_REDUCTION_OPS,
    BLOCK_RESHAPE_OPS,
    BLOCK_ROW_EXPAND_OPS,
    BLOCK_UNARY_OPS,
    OpSpec,
    ValueRange,
)
from .shape_utils import generate_aligned_shape, is_shape_aligned

__all__ = [
    "BLOCK_BINARY_OPS",
    "BLOCK_COL_EXPAND_OPS",
    "BLOCK_MATRIX_OPS",
    "BLOCK_REDUCTION_OPS",
    "BLOCK_RESHAPE_OPS",
    "BLOCK_ROW_EXPAND_OPS",
    "BLOCK_UNARY_OPS",
    "ControlFlowConfig",
    "OpChainConfig",
    "OpFuzzer",
    "OpSpec",
    "ValueRange",
    "generate_aligned_shape",
    "is_shape_aligned",
]
