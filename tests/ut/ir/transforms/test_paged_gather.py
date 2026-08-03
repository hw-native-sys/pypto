# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tensor.paged_gather (paged gather directly into L1 / UB).

The op lowers (in ConvertTensorToTileOps) into a fully-scalar per-row GM->on-chip
load loop on the Cube core:

    acc = tile.create([max_indices, size], target_memory=Mat)
    for i in range(tensor.dim(indices, 0)):
        idx  = tensor.read(indices, [i])          # scalar GM read (pto.load_scalar)
        phys = block_table[idx // bs] * bs + idx % bs   # scalar
        acc  = tile.gather_row(acc, src, [i, 0], [phys, 0], [1, size])  # GM->L1

Only the small index/page-table metadata is scalar-read from GM; the bulk KV
data goes straight GM->L1 (never UB).

Note the row is written *straight into* the accumulator sub-region by
``tile.gather_row`` (pto.subview + GM->Mat pto.tload). There is deliberately no
``tile.assemble``, which would lower to an unsupported MAT->MAT ``pto.tmov``.

Tests compare whole programs against hand-written post-lowering goldens
(``_build_expected`` / inline ``Expected``) rather than grepping printed IR, so
the absent ops are pinned as strongly as the present ones.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType, is_backend_configured, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import InvalidOperationError


def _build_program(
    *,
    space: pl.MemorySpace = pl.MemorySpace.Mat,
    is_trans: bool = False,
    rows: int = 16,
    max_indices: int = 16,
    src_dtype: DataType = pl.FP16,
):
    out_shape = [128, max_indices] if is_trans else [max_indices, 128]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[out_shape, src_dtype]:
            out = pl.paged_gather(
                src,
                idx,
                bt,
                block_size=128,
                size=128,
                max_indices=max_indices,
                space=space,
                is_trans=is_trans,
            )
            return out

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[out_shape, src_dtype]:
            r = self.kernel(src, idx, bt)
            return r

    return Program


def _build_expected(
    *,
    space: pl.MemorySpace = pl.MemorySpace.Mat,
    rows: int | pl.DynVar = 16,
    max_indices: int = 16,
    src_dtype: DataType = pl.FP16,
):
    """Hand-written post-lowering golden mirroring ``_build_program`` (non-transposed).

    Written directly in the already-lowered form -- ``tile.create`` / ``tile.gather_row``
    / ``tile.store`` -- so the pass under test never runs on this side. Building the
    golden by running ``convert_tensor_to_tile_ops`` on it would make the comparison
    self-referential: a regression would change both sides and the test would stay green.

    Local names deliberately avoid the ``__`` spelling the pass emits, purely for
    readability -- structural equality compares IR shape, not names. (Goldens that
    ARE normalized by prerequisite passes, as in ``test_optimize_orch_tensors.py``,
    must avoid ``__`` for a harder reason: auto-naming rejects it. This golden runs
    through no passes at all, so the restriction does not apply here.)
    """
    out_shape = [max_indices, 128]

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
            ret_out: pl.Out[pl.Tensor[out_shape, src_dtype]],
        ) -> pl.Tensor[out_shape, src_dtype]:
            # Loop bound is the runtime index count; the accumulator stays static.
            pg_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx, 0)
            # Tile locals are left unannotated: a Tile annotation only accepts a
            # literal ``pl.Mem.<space>``, not the ``space`` parameter. The memory
            # space still reaches the IR through the ``target_memory`` kwarg.
            acc = pl.tile.create(out_shape, dtype=src_dtype, target_memory=space)
            for pg_i, (acc_iter,) in pl.range(pg_rows, init_values=(acc,)):
                # Only index/page-table metadata is scalar-read from GM.
                idx_raw: pl.Scalar[pl.INT32] = pl.tensor.read(idx, [pg_i])
                pg_idx: pl.Scalar[pl.INDEX] = pl.cast(idx_raw, target_type=pl.INDEX)
                pg_blk: pl.Scalar[pl.INDEX] = pg_idx // 128
                pg_rem: pl.Scalar[pl.INDEX] = pg_idx % 128
                pblk_raw: pl.Scalar[pl.INT32] = pl.tensor.read(bt, [pg_blk])
                pg_pblk: pl.Scalar[pl.INDEX] = pl.cast(pblk_raw, target_type=pl.INDEX)
                pg_phys: pl.Scalar[pl.INDEX] = pg_pblk * 128 + pg_rem
                # Bulk KV goes GM->on-chip straight into the accumulator sub-region.
                pg_row = pl.tile.gather_row(acc_iter, src, [pg_i, 0], [pg_phys, 0], [1, 128], transpose=False)
                pg_res = pl.yield_(pg_row)
            out_tile = pg_res
            ret_store: pl.Tensor[out_shape, src_dtype] = pl.tile.store(out_tile, [0, 0], ret_out)
            return ret_store

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[out_shape, src_dtype]:
            ret_out: pl.Tensor[out_shape, src_dtype] = pl.tensor.create(
                out_shape, dtype=src_dtype, layout=pl.TensorLayout.ND
            )
            r: pl.Tensor[out_shape, src_dtype] = self.kernel(src, idx, bt, ret_out)
            return r

    return Expected


def _convert(program):
    """Run the pass under test -- ConvertTensorToTileOps lowers ``tensor.paged_gather``."""
    return passes.convert_tensor_to_tile_ops()(program)


def test_paged_gather_lowers_to_scalar_per_row_l1_loop():
    """space=Mat lowers to a ForStmt of scalar index math + a GM->L1 per-row gather.

    The whole program is pinned, so this also covers what must NOT appear: no
    ``tile.assemble`` (it would lower to an unsupported MAT->MAT tmov) and no Vec
    tile (the bulk KV must never be preloaded into UB).
    """
    ir.assert_structural_equal(_convert(_build_program()), _build_expected())


def test_paged_gather_transpose_swaps_output_and_load():
    """is_trans=True swaps the output dims and loads each row transposed into L1.

    Written inline rather than via ``_build_expected`` because the transposed form
    differs in three places at once -- accumulator shape [size, max_indices], the
    destination offset [0, i] instead of [i, 0], and ``transpose=True``.
    """

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[16], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
            ret_out: pl.Out[pl.Tensor[[128, 16], pl.FP16]],
        ) -> pl.Tensor[[128, 16], pl.FP16]:
            pg_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx, 0)
            # Accumulator is [size, max_indices] when transposed.
            acc: pl.Tile[[128, 16], pl.FP16, pl.Mem.Mat] = pl.tile.create(
                [128, 16], dtype=pl.FP16, target_memory=pl.Mem.Mat
            )
            for pg_i, (acc_iter,) in pl.range(pg_rows, init_values=(acc,)):
                idx_raw: pl.Scalar[pl.INT32] = pl.tensor.read(idx, [pg_i])
                pg_idx: pl.Scalar[pl.INDEX] = pl.cast(idx_raw, target_type=pl.INDEX)
                pg_blk: pl.Scalar[pl.INDEX] = pg_idx // 128
                pg_rem: pl.Scalar[pl.INDEX] = pg_idx % 128
                pblk_raw: pl.Scalar[pl.INT32] = pl.tensor.read(bt, [pg_blk])
                pg_pblk: pl.Scalar[pl.INDEX] = pl.cast(pblk_raw, target_type=pl.INDEX)
                pg_phys: pl.Scalar[pl.INDEX] = pg_pblk * 128 + pg_rem
                # Row written as a column at destination offset [0, i].
                pg_row: pl.Tile[[128, 16], pl.FP16, pl.Mem.Mat] = pl.tile.gather_row(
                    acc_iter, src, [0, pg_i], [pg_phys, 0], [1, 128], transpose=True
                )
                pg_res = pl.yield_(pg_row)
            out_tile: pl.Tile[[128, 16], pl.FP16, pl.Mem.Mat] = pg_res
            ret_store: pl.Tensor[[128, 16], pl.FP16] = pl.tile.store(out_tile, [0, 0], ret_out)
            return ret_store

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[16], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[[128, 16], pl.FP16]:
            ret_out: pl.Tensor[[128, 16], pl.FP16] = pl.tensor.create(
                [128, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND
            )
            r: pl.Tensor[[128, 16], pl.FP16] = self.kernel(src, idx, bt, ret_out)
            return r

    ir.assert_structural_equal(_convert(_build_program(is_trans=True)), Expected)


def test_paged_gather_space_vec():
    """space=Vec targets UB instead of L1 while keeping the same scalar per-row loop."""
    ir.assert_structural_equal(
        _convert(_build_program(space=pl.MemorySpace.Vec)),
        _build_expected(space=pl.MemorySpace.Vec),
    )


def test_paged_gather_dynamic_row_count():
    """A runtime (dynamic) row count drives the loop bound; the L1 tile stays static."""
    rows = pl.dynamic("ROWS")

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            out = pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=64)
            return out

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            r = self.kernel(src, idx, bt)
            return r

    # Static [64, 128] accumulator, dynamic loop bound taken from ROWS via
    # tensor.dim -- both pinned structurally by the golden.
    ir.assert_structural_equal(_convert(Program), _build_expected(rows=rows, max_indices=64))


@pytest.mark.parametrize("is_trans", [False, True])
def test_paged_gather_survives_full_pipeline(is_trans):
    """The lowered loop survives the full Default pipeline through codegen lowering."""
    # A backend may already be configured by an earlier test in the session;
    # only set it when unconfigured so real set_backend_type failures still surface.
    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)
    program = _build_program(is_trans=is_trans)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = pm.run_passes(program)
    assert result is not None


def test_paged_gather_rejects_non_2d_src():
    """src must be 2D."""
    with pytest.raises(InvalidOperationError, match="2D src"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[256], pl.FP16],
                idx: pl.Tensor[[16], pl.INT32],
                bt: pl.Tensor[[8], pl.INT32],
            ) -> pl.Tensor[[16, 128], pl.FP16]:
                return pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=16)


def test_paged_gather_rejects_non_int32_indices():
    """indices must be INT32."""
    with pytest.raises(InvalidOperationError, match="indices dtype"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[256, 128], pl.FP16],
                idx: pl.Tensor[[16], pl.FP16],
                bt: pl.Tensor[[8], pl.INT32],
            ) -> pl.Tensor[[16, 128], pl.FP16]:
                return pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
