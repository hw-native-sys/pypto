# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Before/after tests for folding the explicit split-reshape ops in ExpandMixedKernel.

``tile.aiv_shard`` (C->V, full -> half) and ``tile.aic_gather`` (V->C, half ->
full) are recognised directly by ExpandMixedKernel as op-driven cross-core
boundaries and folded into the same tpush/tpop machinery used for a cross-C/V
``tile.move``. These tests hand-build a minimal InCore ``split_aiv`` function at
the post-InferTileMemorySpace level (memory spaces already assigned), run
``expand_mixed_kernel`` in isolation (verification disabled), and assert the
whole expanded program via ``ir.assert_structural_equal`` against a hand-authored
``Expected`` (the genuine, per-lane expanded form).

The Ascend950 backend (where the V->C direction needs an NZ fractal adapter) is
configured by the directory-level ``conftest.py``.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out


def _tile(shape, view=None, mem=None):
    return ir.TileType(shape, FP32, None, view, mem)


def _expand(program):
    """Run ExpandMixedKernel with verification/roundtrip disabled."""
    with passes.PassContext([]):
        return passes.expand_mixed_kernel()(program)


def _assert_no_free_var(program):
    """Codegen guard: a free/dangling Var prints as a ``__FREE_VAR`` placeholder.

    ExpandMixedKernel splits a kernel into AIC/AIV lanes; a mis-routed value
    leaves a lane referencing an undefined Var, which the printer marks
    ``__FREE_VAR`` and which later crashes PTO emission. This property is not
    structurally expressible, so it is checked separately from the
    before/after structural comparison.
    """
    assert "__FREE_VAR" not in ir.python_print(program)


# ---------------------------------------------------------------------------
# CASE aiv_shard (C->V, full -> half)
# ---------------------------------------------------------------------------


def _build_aiv_shard_program():
    """qk[128,128]Vec --aiv_shard(split=1)--> half[64,128], consumed by a vector add."""
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", ir.TensorType([64, 128], FP32), span)

    shard = T.aiv_shard(qk, split=1, span=span)
    assert isinstance(shard.type, ir.TileType)
    half = ir.Var("half", _tile(shard.type.shape, shard.type.tile_view, MS.Vec), span)
    add = T.add(half, half, span)
    assert isinstance(add.type, ir.TileType)
    y = ir.Var("y", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    store = T.store(y, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half, shard, span),
            ir.AssignStmt(y, add, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(qk, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_aiv_shard", span), qk


def test_aiv_shard_folds_into_cube_to_vector_boundary():
    program, _ = _build_aiv_shard_program()
    after = _expand(program)

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.AIC,
            level=pl.Level.AIC,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aic(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            # AIC: pushes the FULL tile (the unadapted qk parameter), split == 1.
            split_aiv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_c2v_slot_buffer", peer_func="split_aiv_aiv"
            )
            pl.system.aic_initialize_pipe(
                split_aiv_c2v_slot_buffer_import, pl.const(0, pl.INT32), dir_mask=1, slot_size=65536
            )
            pl.tile.tpush_to_aiv(qk, split=1)

        @pl.function(
            type=pl.FunctionType.AIV,
            level=pl.Level.AIV,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=524288, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=65536
            )
            # AIV: pops the HALF tile [64,128] in Vec (identity / non-NZ view), split == 1.
            half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)
            y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
            pl.system.tfree_to_aic(half)
            out_store: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(y, [0, 0], out_0)
            return out_store

        @pl.function(
            type=pl.FunctionType.Group,
            level=pl.Level.CORE_GROUP,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            self.split_aiv_aic(qk, out_0)
            self.split_aiv_aiv(qk, out_0)
            return out_0

    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


# ---------------------------------------------------------------------------
# CASE aic_gather (V->C, half -> full)
# ---------------------------------------------------------------------------


def _build_aic_gather_program():
    """half2[64,128]Vec --aic_gather(split=1)--> full[128,128], move->Left, matmul."""
    span = ir.Span.unknown()
    a = ir.Var("a", _tile([64, 128], mem=MS.Vec), span)
    b = ir.Var("b", _tile([128, 128], mem=MS.Right), span)
    out_0 = ir.Var("out_0", ir.TensorType([128, 128], FP32), span)

    add = T.add(a, a, span)
    assert isinstance(add.type, ir.TileType)
    half2 = ir.Var("half2", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    gather = T.aic_gather(half2, split=1, span=span)
    assert isinstance(gather.type, ir.TileType)
    full = ir.Var("full", _tile(gather.type.shape, gather.type.tile_view, MS.Vec), span)
    move_left = T.move(full, MS.Left, span=span)
    assert isinstance(move_left.type, ir.TileType)
    full_left = ir.Var("full_left", _tile(move_left.type.shape, move_left.type.tile_view, MS.Left), span)
    matmul = T.matmul(full_left, b, span)
    assert isinstance(matmul.type, ir.TileType)
    z = ir.Var("z", _tile(matmul.type.shape, matmul.type.tile_view, MS.Acc), span)
    move_vec = T.move(z, MS.Vec, span=span)
    assert isinstance(move_vec.type, ir.TileType)
    z_vec = ir.Var("z_vec", _tile(move_vec.type.shape, move_vec.type.tile_view, MS.Vec), span)
    store = T.store(z_vec, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half2, add, span),
            ir.AssignStmt(full, gather, span),
            ir.AssignStmt(full_left, move_left, span),
            ir.AssignStmt(z, matmul, span),
            ir.AssignStmt(z_vec, move_vec, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(a, _IN), (b, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_aic_gather", span), half2


def test_aic_gather_folds_into_vector_to_cube_boundary():
    program, _ = _build_aic_gather_program()
    after = _expand(program)

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.AIC,
            level=pl.Level.AIC,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aic(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ):
            split_aiv_v2c_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_v2c_slot_buffer", size=262144, base=-1
            )
            split_aiv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_c2v_slot_buffer", peer_func="split_aiv_aiv"
            )
            pl.system.aic_initialize_pipe(
                split_aiv_c2v_slot_buffer_import, split_aiv_v2c_slot_buffer, dir_mask=3, slot_size=65536
            )
            # AIC: V->C pop yields the FULL tile [128,128] in Mat. The Mat default
            # effective tile_view is NZ (col_major), so no explicit view is needed.
            full: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=1)
            # The original follow-on move(full -> Left) survives on the AIC lane
            # (it is NOT re-detected as a second cross-core boundary).
            full_left: pl.Tile[[128, 128], pl.FP32, pl.Mem.Left] = pl.tile.move(
                full, target_memory=pl.Mem.Left
            )
            pl.system.tfree_to_aiv(full)
            z: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(full_left, b)
            pl.tile.tpush_to_aiv(z, split=0)

        @pl.function(
            type=pl.FunctionType.AIV,
            level=pl.Level.AIV,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aiv(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            split_aiv_v2c_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_v2c_slot_buffer", peer_func="split_aiv_aic"
            )
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=262144, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer, split_aiv_v2c_slot_buffer_import, dir_mask=3, slot_size=65536
            )
            half2: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(a, a)
            # Push-side fractal adapter: move the HALF [64,128] into Vec with an
            # explicit NZ (col_major) layout, then push it to the cube FIFO.
            half2_nz: pl.Tile[
                [64, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.move(
                half2,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            pl.tile.tpush_to_aic(half2_nz, split=1)
            z_vec: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.tpop_from_aic(split=0)
            out_store: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(z_vec, [0, 0], out_0)
            pl.system.tfree_to_aic(z_vec)
            return out_store

        @pl.function(
            type=pl.FunctionType.Group,
            level=pl.Level.CORE_GROUP,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            self.split_aiv_aic(a, b, out_0)
            self.split_aiv_aiv(a, b, out_0)
            return out_0

    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
