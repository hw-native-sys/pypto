# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic valid_shape examples.

Demonstrates a DSL pattern where the valid length of a tile is a runtime
scalar (caller-provided) and used inside ``pl.load(..., valid_shapes=...)``
to bound the active region of the tile, then padded via
``pl.tile.fillpad``::

    tile = pl.load(..., valid_shapes=[rows, vlen])   # vlen is a runtime scalar
    padded = pl.tile.fillpad(tile, pad_value=PadValue.min)

JIT note
--------
The pre-JIT version of this example also showed the same pattern with
``vlen`` selected via ``if/else`` (and inside a per-block loop).  In the
@pl.jit world the specializer's alpha-renamer rewrites the rebinding of
``vlen`` in the else-branch to a distinct alias, which then fails
``ConvertToSSA`` ("used outside its defining scope").  The current
recommended workaround is to push the per-call/per-iteration choice of
``vlen`` to the *caller* and pass a single scalar parameter -- as shown
below.  Restoring the in-DSL ``if/else`` pattern requires a JIT
specializer fix (see the comments in ``examples/models/qwen3_jit/``).

Note: ``__main__`` runs ``compile_for_test`` only (no device execution).
Full end-to-end execution is exercised under
``tests/st/codegen/test_dyn_valid_shape_loop.py`` and
``test_dynamic_valid_shape_if_else.py``.

Run:  python examples/kernels/09_dyn_valid_shape.py
"""

# DSL function bodies are parsed as AST -- runtime scalars (vlen, ...)
# look undefined to pyright. pl.FP32 / pl.INDEX scalar dtype markers (used as
# annotations) are DataType values, not types -- pyright can't infer them.
# pyright: reportUndefinedVariable=false, reportInvalidTypeForm=false

import pypto.language as pl
import torch

# Tile / tensor dimensions
Q_TILE = 64
BLOCK_COL = 64


@pl.jit
def dyn_valid_shape(
    data: pl.Tensor,
    scale: pl.FP32,
    vlen: pl.INDEX,
    output: pl.Out[pl.Tensor],
):
    """Load with caller-provided valid_shape, fillpad, then scale.

    The caller passes either the partial-block length or the full-block
    length; the kernel does not need to branch internally.
    """
    with pl.incore():
        s_tile = pl.load(
            data,
            [0, 0],
            [Q_TILE, BLOCK_COL],
            valid_shapes=[Q_TILE, vlen],
            target_memory=pl.MemorySpace.Vec,
        )
        s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
        scaled = pl.mul(s_padded, scale)
        pl.store(scaled, [0, 0], output)
    return output


if __name__ == "__main__":
    # Smoke test via compile_for_test (no device execution required).
    # Same kernel, two different valid_shape values: full block (64) and
    # partial last block (32). compile_for_test caches per concrete vlen,
    # so both compile cleanly.
    data = torch.randn(Q_TILE, BLOCK_COL, dtype=torch.float32)
    out = torch.zeros(Q_TILE, BLOCK_COL, dtype=torch.float32)

    prog_full = dyn_valid_shape.compile_for_test(data, 0.5, 64, out)
    print(f"dyn_valid_shape (full): {len(prog_full.functions)} fn(s)")

    prog_partial = dyn_valid_shape.compile_for_test(data, 0.5, 32, out)
    print(f"dyn_valid_shape (partial): {len(prog_partial.functions)} fn(s)")
    print("OK")
