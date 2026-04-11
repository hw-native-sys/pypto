# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO JIT compilation module.

Provides the ``@pl.jit`` decorator for writing kernel functions that are
automatically specialized and compiled on first call based on the shapes
and dtypes of their tensor arguments.

Example::

    import pypto.language as pl

    @pl.jit
    def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.incore():
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
        return c

    # Compiles on first call, serves from cache on subsequent calls
    prog = tile_add(torch.randn(128, 128), torch.randn(128, 128), torch.empty(128, 128))
"""

from .decorator import JITFunction, jit

__all__ = ["JITFunction", "jit"]
