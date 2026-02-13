# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type wrappers for PyPTO Language DSL.

This module provides type annotation and runtime wrapper classes for PyPTO's language DSL:
- Scalar: Scalar values with specific data types
- Tensor: Multi-dimensional arrays in global memory
- Tile: Memory blocks in unified buffer memory for block-level programming
"""

from pypto.language.typing.scalar import Scalar
from pypto.language.typing.tensor import Tensor
from pypto.language.typing.tile import Tile
# Import extended Tile with MemRef support
from pypto.language.typing.memref_tile import MemRefTile, make_memref, bind_memref

__all__ = ["Scalar", "Tensor", "Tile",
           "MemRefTile", "make_memref", "bind_memref",]
