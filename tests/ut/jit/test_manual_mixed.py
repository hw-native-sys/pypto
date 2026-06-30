# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""@pl.jit support for hand-written cube/vector mixed kernels.

`@pl.jit.aic` / `@pl.jit.aiv` author the two halves of a mixed kernel directly
(explicit pipe setup + tpush/tpop), and `@pl.jit.group(split=...)` dispatches the
pair. Because the halves are hand-written, every real tile is `alloc_buffer`-
pinnable and the cross-core tpop tiles stay buffer-less — keeping the cube/vector
overlap of an auto `optimizations=[pl.split(...)]` kernel while taking full
manual control of the on-chip layout.
"""

import pypto.language as pl
import pytest
from pypto.backend import BackendType

from pypto import backend, ir


@pytest.fixture(autouse=True)
def _backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _tile_stats(prog: ir.Program) -> tuple[int, int]:
    """Count (pinned tiles, buffer-less tiles) across the program."""
    pinned = 0
    bufferless = 0

    def walk(stmt) -> None:
        nonlocal pinned, bufferless
        if stmt is None:
            return
        for s in getattr(stmt, "stmts", []):
            walk(s)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
            mr = stmt.var.type.memref
            if mr is not None and "__pinned__" in mr.base_.name_hint:
                pinned += 1
            elif mr is None:
                bufferless += 1
        for attr in ("body", "then_body", "else_body"):
            if hasattr(stmt, attr):
                walk(getattr(stmt, attr))

    for func in prog.functions.values():
        walk(func.body)
    return pinned, bufferless


def test_jit_manual_mixed_kernel_compiles_with_pins():
    """A hand-written AIC+AIV mixed kernel with alloc_buffer pins compiles via jit."""
    torch = pytest.importorskip("torch")

    @pl.jit.aic
    def fa_aic(q: pl.Tensor, k: pl.Tensor, out: pl.Out[pl.Tensor]):
        v2c = pl.reserve_buffer(name="v2c_buf", size=32768, base=-1)
        c2v = pl.import_peer_buffer(name="c2v_buf", peer_func="fa_aiv")
        pl.aic_initialize_pipe(dir_mask=3, slot_size=8192, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c)
        q_mat = pl.load(q, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
        pl.tile.alloc_buffer(q_mat, addr=0, size=4096)
        k_mat = pl.load(k, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        pl.tile.alloc_buffer(k_mat, addr=4096, size=32768)
        k_t = pl.tile.transpose_view(k_mat)
        q_left = pl.move(q_mat, target_memory=pl.Mem.Left)
        pl.tile.alloc_buffer(q_left, addr=0, size=4096)
        k_right = pl.move(k_t, target_memory=pl.Mem.Right)
        pl.tile.alloc_buffer(k_right, addr=0, size=32768)
        scores = pl.matmul(q_left, k_right)
        pl.tile.alloc_buffer(scores, addr=0, size=8192)
        pl.tpush_to_aiv(scores, split=0)
        ex: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
        pl.tfree_to_aiv(ex)
        return out

    @pl.jit.aiv
    def fa_aiv(q: pl.Tensor, k: pl.Tensor, out: pl.Out[pl.Tensor]):
        c2v = pl.reserve_buffer(name="c2v_buf", size=32768, base=-1)
        v2c = pl.import_peer_buffer(name="v2c_buf", peer_func="fa_aic")
        pl.aiv_initialize_pipe(dir_mask=3, slot_size=8192, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c)
        scores: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
        e = pl.exp(scores)
        pl.tile.alloc_buffer(e, addr=0, size=8192)
        pl.tpush_to_aic(e, split=0)
        pl.tfree_to_aic(scores)
        out = pl.store(e, [0, 0], out)
        return out

    @pl.jit.group(split=pl.SplitMode.UP_DOWN)
    def fa_group(q: pl.Tensor, k: pl.Tensor, out: pl.Out[pl.Tensor]):
        fa_aic(q, k, out)
        out = fa_aiv(q, k, out)
        return out

    @pl.jit
    def fa_main(q: pl.Tensor, k: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.spmd(2, name_hint="fa_fused"):
            out = fa_group(q, k, out)
        return out

    q = torch.zeros(16, 128, dtype=torch.bfloat16)
    k = torch.zeros(128, 128, dtype=torch.bfloat16)
    out = torch.zeros(16, 128, dtype=torch.float32)
    prog = fa_main.compile_for_test(q, k, out)

    names = sorted(gv.name for gv in prog.functions)
    assert "fa_aic" in names and "fa_aiv" in names and "fa_group" in names
    pinned, bufferless = _tile_stats(prog)
    # Every real tile (q/k loads, L0 operands, Acc, Vec) is pinned; the two tpop
    # results (cross-core C2V/V2C slot) are buffer-less.
    assert pinned >= 7, f"expected the explicit tiles pinned, got {pinned}"
    assert bufferless >= 2, f"expected tpop results buffer-less, got {bufferless}"


def test_split_kwarg_rejected_on_non_group_flavors():
    """split= is only valid on @pl.jit.group."""
    with pytest.raises(TypeError, match="split"):

        @pl.jit.aic(split=pl.SplitMode.UP_DOWN)
        def _bad(q: pl.Tensor):
            return q

    with pytest.raises(TypeError, match="split"):

        @pl.jit.incore(split=pl.SplitMode.UP_DOWN)
        def _bad2(q: pl.Tensor):
            return q


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
