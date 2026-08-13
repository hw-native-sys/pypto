# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic valid_shape examples -- scalar, if/else, and loop patterns.

A tile's *valid* region may be narrower than its physical shape.  Pass the
valid extent to ``pl.load(..., valid_shape=...)`` and mask the remainder with
``pl.tile.fillpad``::

    tile   = pl.load(..., valid_shape=[rows, vlen])
    padded = pl.tile.fillpad(tile, pad_value=pl.PadValue.min)

Where ``vlen`` comes from decides how much freedom it has at runtime.  The
three kernels below cover the spectrum:

1. ``dyn_valid_shape`` -- ``vlen`` is a scalar *parameter*.
2. ``dyn_valid_shape_if_else`` -- ``vlen`` is selected by an in-DSL
   ``if``/``else`` from values read out of a config tensor.
3. ``dyn_valid_shape_loop`` -- the ragged-tail idiom: loop over blocks and
   take the partial length on the last iteration.

Scalar parameters are specialization constants
----------------------------------------------
Under ``@pl.jit`` a scalar argument (``pl.INDEX``, ``pl.FP32``, ...) is a
*specialization constant*: the specializer inlines its value at every use
site, so ``dyn_valid_shape`` compiles a separate kernel per distinct ``vlen``
and the generated ``pto.alloc_tile`` carries a constant ``valid_col``.  That
is the right trade when the caller knows ``vlen`` at dispatch time -- it is
the simplest form and it constant-folds cleanly.

It also means an ``if``/``else`` over scalar *parameters* is resolved at
specialization time, not at runtime.  For a branch that survives to the
device, the selected values must be genuinely dynamic -- read them from a
tensor, as kernels 2 and 3 do.  Those lower to a real ``scf.if`` whose result
feeds the tile's ``valid_col``.

Note: ``__main__`` runs ``lower`` only (no code generation or device
execution).  Lowering is also exercised by
``tests/st/codegen/dsl/test_dyn_valid_shape_loop.py`` and
``tests/st/codegen/dsl/test_dynamic_valid_shape_if_else.py``; the generated
PTO is asserted in ``tests/ut/codegen/test_dynamic_valid_shape_if_else.py``.

Run:  python examples/intermediate/06_dyn_valid_shape.py
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
N_ROW = 128  # sij_buf rows = Q_TILE * max_blocks(2)


@pl.jit
def dyn_valid_shape(
    data: pl.Tensor,
    scale: pl.FP32,
    vlen: pl.INDEX,
    output: pl.Out[pl.Tensor],
):
    """Load with a caller-provided valid_shape, fillpad, then scale.

    ``vlen`` is a scalar parameter and therefore a specialization constant:
    each distinct value compiles its own kernel with a constant ``valid_col``.
    Use this form when the caller already knows the valid length.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        s_tile = pl.load(
            data,
            [0, 0],
            [Q_TILE, BLOCK_COL],
            valid_shape=[Q_TILE, vlen],
            target_memory=pl.MemorySpace.Vec,
        )
        s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
        scaled = pl.mul(s_padded, scale)
        pl.store(scaled, [0, 0], output)
    return output


@pl.jit
def dyn_valid_shape_if_else(
    data: pl.Tensor,
    cfg: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    """Select the valid length with an in-DSL ``if``/``else``, then load+fillpad.

    ``cfg`` holds ``[is_last, last_valid_len, full_len]``.  Reading them makes
    them runtime values, so the branch is not folded away: it lowers to an
    ``scf.if`` whose result becomes the tile's ``valid_col``.  The tile type is
    uniform across both branches -- only the runtime valid length differs.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        is_last = pl.read(cfg, [0])
        last_valid_len = pl.read(cfg, [1])
        full_len = pl.read(cfg, [2])
        if is_last == 1:
            vlen = last_valid_len
        else:
            vlen = full_len
        s_tile = pl.load(
            data,
            [0, 0],
            [Q_TILE, BLOCK_COL],
            valid_shape=[Q_TILE, vlen],
            target_memory=pl.MemorySpace.Vec,
        )
        s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
        pl.store(s_padded, [0, 0], output)
    return output


@pl.jit
def dyn_valid_shape_loop(
    sij_buf: pl.Tensor,
    cfg: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    """Loop over blocks, taking the partial valid length on the last one.

    ``cfg`` holds ``[n_blocks, last_valid_len, block_size]``.  The trip count
    is a runtime value, so the loop lowers to an ``scf.for`` and the
    per-iteration ``if``/``else`` to an ``scf.if`` nested inside it.  This is
    the ragged-tail idiom used by the paged-attention kernels.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        n_blocks = pl.cast(pl.read(cfg, [0]), pl.INDEX)
        last_valid_len = pl.cast(pl.read(cfg, [1]), pl.INDEX)
        block_size = pl.cast(pl.read(cfg, [2]), pl.INDEX)
        for i in pl.range(n_blocks):
            if i == n_blocks - 1:
                vlen = last_valid_len
            else:
                vlen = block_size
            s_tile = pl.load(
                sij_buf,
                [i * Q_TILE, 0],
                [Q_TILE, BLOCK_COL],
                valid_shape=[Q_TILE, vlen],
                target_memory=pl.MemorySpace.Vec,
            )
            s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
            pl.store(s_padded, [i * Q_TILE, 0], output)
    return output


if __name__ == "__main__":
    # Smoke test via lower (no code generation or device execution required).
    data = torch.randn(Q_TILE, BLOCK_COL, dtype=torch.float32)
    out = torch.zeros(Q_TILE, BLOCK_COL, dtype=torch.float32)

    # Same kernel, two vlen values: full block (64) and partial last block (32).
    # Each is its own specialization and runs through the pipeline independently.
    prog_full = dyn_valid_shape.lower(data, 0.5, 64, out)
    print(f"dyn_valid_shape (full): {len(prog_full.functions)} fn(s)")
    prog_partial = dyn_valid_shape.lower(data, 0.5, 32, out)
    print(f"dyn_valid_shape (partial): {len(prog_partial.functions)} fn(s)")

    # One specialization covers both branches: is_last is read at runtime.
    cfg = torch.tensor([1, 48, BLOCK_COL], dtype=torch.int64)
    prog_if_else = dyn_valid_shape_if_else.lower(data, cfg, out)
    print(f"dyn_valid_shape_if_else: {len(prog_if_else.functions)} fn(s)")

    sij_buf = torch.randn(N_ROW, BLOCK_COL, dtype=torch.float32)
    loop_out = torch.zeros(N_ROW, BLOCK_COL, dtype=torch.float32)
    loop_cfg = torch.tensor([2, 48, BLOCK_COL], dtype=torch.int64)
    prog_loop = dyn_valid_shape_loop.lower(sij_buf, loop_cfg, loop_out)
    print(f"dyn_valid_shape_loop: {len(prog_loop.functions)} fn(s)")
    print("OK")
