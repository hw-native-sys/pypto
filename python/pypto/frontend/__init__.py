# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO Frontend module - High-level kernel programming API.

This module provides:
- @kernel decorator: combines @pl.program + @pl.function for single-kernel use cases
- Extended Tile/Tensor types with manual MemRef specification for address control
- Re-exports of common pypto.language symbols for convenience

Typical usage:
    import pypto.frontend as fe

    @fe.kernel
    def my_kernel(x: fe.Tensor[[64, 128], fe.FP16]) -> fe.Tensor[[64, 128], fe.FP32]:
        tile = fe.load(x, [0, 0], [64, 64])
        result = fe.add(tile, tile)
        return fe.store(result, [0, 0], [64, 64], x)

    # my_kernel is an ir.Program with a single function
"""

# Import the kernel decorator
from .kernel import kernel
from .jit import jit, compile, launch

# Import MemorySpace for manual address specification
from pypto.pypto_core.ir import MemorySpace

__all__ = [
    # Primary new API
    "kernel",
    "jit",
    "compile",
    "launch"
]
