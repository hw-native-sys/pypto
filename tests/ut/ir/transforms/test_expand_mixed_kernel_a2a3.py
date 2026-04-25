# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a2a3-specific regression tests for ExpandMixedKernel cross-core handling.

GM-pipe-buffer injection is exercised separately in
``test_inject_gm_pipe_buffer.py``; this file pins down ExpandMixedKernel's
own a2a3 boundary behaviour without running InjectGMPipeBuffer.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run SSA -> infer-memory -> expand-mixed-kernel under VerificationLevel.NONE.

    InjectGMPipeBuffer is intentionally excluded so this file's Expecteds
    pin only what ExpandMixedKernel produces.
    """
    with passes.PassContext([], ir.VerificationLevel.NONE):
        return passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(program))
        )


def _run_pipeline_from_tensor(program: ir.Program) -> ir.Program:
    """Run SSA -> tensor-to-tile -> infer-memory -> expand-mixed-kernel.

    Mirrors _run_pipeline but inserts convert_tensor_to_tile_ops between SSA
    and InferTileMemorySpace, for cases that start from tensor-level IR.
    """
    with passes.PassContext([], ir.VerificationLevel.NONE):
        return passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(
                passes.convert_tensor_to_tile_ops()(passes.convert_to_ssa()(program))
            )
        )


def test_v2c_boundary_uses_nz_layout_on_a2a3():
    """On Ascend910B, cross-core push needs no layout adaptation on the AIV side.

    Ascend910B routes push/pop through ub -> gm -> mat. The ub -> gm transfer
    uses ND layout directly, so no tile.move is needed before tpush_to_aic.
    The AIC tpop lands in Mat with NZ layout (col_major blayout), and a
    subsequent Mat -> Left tile.move resolves the final layout.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            x_tile = pl.load(x, [0, 0], [16, 128])
            # Direct Vec -> Left boundary (exercises BuildCrossCoreTransferView with Left)
            x_left = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            z_vec = pl.move(
                z_tile,
                target_memory=pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.none_box,
            )
            out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            return out_0

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0_aic(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ):
            main_incore_0_v2c_slot_buffer = pl.reserve_buffer(
                name="main_incore_0_v2c_slot_buffer", size=16384, base=-1
            )
            main_incore_0_c2v_slot_buffer_import = pl.import_peer_buffer(
                name="main_incore_0_c2v_slot_buffer", peer_func="main_incore_0_aiv"
            )
            pl.aic_initialize_pipe(
                main_incore_0_c2v_slot_buffer_import,
                main_incore_0_v2c_slot_buffer,
                dir_mask=3,
                slot_size=4096,
            )
            x_left_mat: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            x_left: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                x_left_mat,
                target_memory=pl.MemorySpace.Left,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            pl.tfree_to_aiv(x_left_mat)
            y_mat: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
            )
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            pl.tpush_to_aiv(z_tile, split=0)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0_aiv(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            main_incore_0_v2c_slot_buffer_import = pl.import_peer_buffer(
                name="main_incore_0_v2c_slot_buffer", peer_func="main_incore_0_aic"
            )
            main_incore_0_c2v_slot_buffer = pl.reserve_buffer(
                name="main_incore_0_c2v_slot_buffer", size=16384, base=-1
            )
            pl.aiv_initialize_pipe(
                main_incore_0_c2v_slot_buffer,
                main_incore_0_v2c_slot_buffer_import,
                dir_mask=3,
                slot_size=4096,
            )
            x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
            pl.tpush_to_aic(x_tile, split=0)
            z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            pl.tfree_to_aic(z_vec)
            return out_0_store

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            self.main_incore_0_aic(x, y, out_0)
            result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
            return result

    After = _run_pipeline(Before)
    ir.assert_structural_equal(After, Expected)


def test_c2v_boundary_preserves_vec_pop_layout_on_a2a3():
    """On Ascend910B, C2V Vec pops must stay in the final Vec layout.

    The A2A3 GM-backed pipe consumer materializes the popped tile through an ND
    GlobalTensor. PTO-ISA does not support loading that ND buffer into an NZ
    Vec tile, so ExpandMixedKernel must not introduce an NZ Vec bridge tile
    here.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            z_vec = pl.move(
                z_tile,
                target_memory=pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.none_box,
            )
            out_0 = pl.store(z_vec, [0, 0], out_0)
            return out_0

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0_aic(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ):
            main_incore_0_c2v_slot_buffer_import = pl.import_peer_buffer(
                name="main_incore_0_c2v_slot_buffer", peer_func="main_incore_0_aiv"
            )
            pl.aic_initialize_pipe(
                main_incore_0_c2v_slot_buffer_import,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
            )
            x_mat: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
            )
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
            )
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            pl.tpush_to_aiv(z_tile, split=0)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0_aiv(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            main_incore_0_c2v_slot_buffer = pl.reserve_buffer(
                name="main_incore_0_c2v_slot_buffer", size=32768, base=-1
            )
            pl.aiv_initialize_pipe(
                main_incore_0_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
            )
            z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            pl.tfree_to_aic(z_vec)
            return out_0_store

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            self.main_incore_0_aic(x, y, out_0)
            result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
            return result

    After = _run_pipeline(Before)
    ir.assert_structural_equal(After, Expected)


def test_accumulator_with_tile_create_classifies_as_pure_aic():
    """Regression for issue #1083.

    A CORE_GROUP scope whose only "vector" signal is a declaration-only
    ``tile.create`` feeding a matmul/matmul_acc loop used to be misclassified
    as mixed — routed through the split path and emitting broken AIC/AIV IR.
    After the fix, ``tile.create`` is SHARED in the core-affinity classifier,
    and ``InferTileMemorySpace`` back-propagates the body's Acc memory to the
    iter_arg and init, so the kernel classifies as pure AIC.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def matmul_accumulator(
            self,
            a: pl.Tensor[[16, 256], pl.BF16],
            b: pl.Tensor[[256, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            acc = pl.create_tensor([16, 128], dtype=pl.FP32)
            for k in pl.range(0, 256, 64):
                a_slice = pl.slice(a, [16, 64], [0, k])
                b_slice = pl.slice(b, [64, 128], [k, 0])
                acc = pl.matmul_acc(acc, a_slice, b_slice)
            out = pl.assemble(out, acc, [0, 0])
            return out

    After = _run_pipeline_from_tensor(Before)

    # Exactly one non-orchestration function, typed as AIC (no AIC/AIV split,
    # no Group wrapper), since the scope is semantically pure cube.
    compute_funcs = [fn for _, fn in After.functions.items() if fn.func_type != ir.FunctionType.Orchestration]
    assert len(compute_funcs) == 1, (
        f"expected a single pure-AIC function, got {[(fn.name, fn.func_type) for fn in compute_funcs]}"
    )
    assert compute_funcs[0].func_type == ir.FunctionType.AIC, (
        f"expected FunctionType.AIC, got {compute_funcs[0].func_type}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
