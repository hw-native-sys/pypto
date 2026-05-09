# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-function composition with @pl.jit.

Demonstrates that ``@pl.jit.inline`` helpers are auto-discovered as deps of a
``@pl.jit`` entry function and spliced at the call site. Each helper is a normal
DSL function; the entry composes them by calling them like Python functions.

This is the @pl.jit equivalent of the older ``@pl.program`` + ``self.method()``
cross-function-call pattern: in the JIT world, dep discovery happens through the
entry function's globals, not through a class.
"""

import pypto.language as pl


@pl.jit.inline
def add_helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Tile-wise add: c = a + 1.0."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.inline
def mul_helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Tile-wise multiply: c = a * 2.0."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, 2.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def main_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Entry: c = (a + 1.0) * 2.0, composed via two @pl.jit.inline helpers."""
    intermediate = pl.create_tensor([128, 128], dtype=pl.FP32)
    intermediate = add_helper(a, intermediate)
    c = mul_helper(intermediate, c)
    return c


if __name__ == "__main__":
    import torch

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    prog = main_kernel.compile_for_test(a, c)
    print(f"main_kernel: {len(prog.functions)} fn(s)")
    for fn in prog.functions.values():
        print(f"  {fn.name}: {fn.func_type}")
