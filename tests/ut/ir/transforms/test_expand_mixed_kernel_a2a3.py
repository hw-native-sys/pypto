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
    """Run SSA -> infer-memory -> expand-mixed-kernel.

    InjectGMPipeBuffer is intentionally excluded so this file's Expecteds
    pin only what ExpandMixedKernel produces.
    """
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))


def _run_pipeline_from_tensor(program: ir.Program) -> ir.Program:
    """Run SSA -> tensor-to-tile -> infer-memory -> expand-mixed-kernel.

    Mirrors _run_pipeline but inserts convert_tensor_to_tile_ops between SSA
    and InferTileMemorySpace, for cases that start from tensor-level IR.
    """
    return passes.expand_mixed_kernel()(
        passes.infer_tile_memory_space()(
            passes.convert_tensor_to_tile_ops()(passes.convert_to_ssa()(program))
        )
    )


def _op_name(stmt: ir.Stmt) -> str:
    """Return the op name of an AssignStmt/EvalStmt Call, or '' for anything else."""
    call = None
    if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
        call = stmt.value
    elif isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
        call = stmt.expr
    if call is not None and isinstance(call.op, ir.Op):
        return call.op.name
    return ""


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
            self.main_incore_0_aiv(x, y, out_0)
            return out_0

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
            self.main_incore_0_aiv(x, y, out_0)
            return out_0

    After = _run_pipeline(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_mediated_cross_lane_store_load_gets_handshake_on_a2a3():
    """Regression for issue #1433.

    A mixed root that exchanges data AIC -> AIV through a GM scratch tensor
    (``tile.store`` on the cube lane, ``tile.load`` from the same tensor on the
    vector lane) used to split into AIC/AIV kernels with no cross-core fence,
    leaving them racing on the shared GM region. ExpandMixedKernel must now
    detect the GM-mediated dependency and emit a tpush/tpop handshake (plus the
    automatic pipe setup) so the AIV load happens-after the AIC store.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            # AIC lane: matmul, then store the cube result into GM scratch.
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_acc = pl.matmul(x_left, y_right)
            scratch = pl.store(z_acc, [0, 0], scratch)
            # AIV lane: read the same GM scratch back, elementwise, store out.
            chunk = pl.load(scratch, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec)
            chunk = pl.exp(chunk)
            out = pl.store(chunk, [0, 0], out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay_aic(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ):
            gm_relay_c2v_slot_buffer_import = pl.import_peer_buffer(
                name="gm_relay_c2v_slot_buffer", peer_func="gm_relay_aiv"
            )
            pl.aic_initialize_pipe(
                gm_relay_c2v_slot_buffer_import,
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
            z_acc = pl.matmul(x_left, y_right)
            # Fresh local (underscore-prefixed: unused by design, the store result
            # is consumed by no later op on the AIC lane) so the binding matches the
            # pass-emitted SSA store-result var structurally.
            _scratch_stored: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_acc, [0, 0], scratch)
            pl.tpush_to_aiv(z_acc, split=0)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay_aiv(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            gm_relay_c2v_slot_buffer = pl.reserve_buffer(name="gm_relay_c2v_slot_buffer", size=32768, base=-1)
            pl.aiv_initialize_pipe(
                gm_relay_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
            )
            sync_tile: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            pl.tfree_to_aic(sync_tile)
            chunk: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                scratch, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec
            )
            chunk_exp = pl.exp(chunk)
            out_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(chunk_exp, [0, 0], out)
            return out_store

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            self.gm_relay_aic(x, y, scratch, out)
            self.gm_relay_aiv(x, y, scratch, out)
            return out

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


def test_tpop_yielded_as_branch_result_frees_before_yield_on_a2a3():
    """Regression for issue #1413.

    When a cross-core tile's last use is the ``YieldStmt`` that carries it out
    of an ``if`` branch, ExpandMixedKernel's tpop/tfree finalizer must emit the
    ``tfree_to_aic`` *before* the yield. A ``YieldStmt`` is the mandatory
    terminator of a control-flow body with return values, so appending the
    tfree after it produced a malformed branch (``tpop; yield; tfree``) that
    later crashed Simplify's ``StripTrailingYield`` with
    "control-flow body tail must be YieldStmt when return_vars is non-empty".

    Both branches of the ``if`` do cube work whose Vec-moved result is the
    branch's carried value, so after the AIC/AIV split each AIV branch reduces
    to ``tpop_from_aic`` -> ``tfree_to_aic`` -> ``yield``.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            w: pl.Tensor[[128, 128], pl.BF16],
            flag: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            if flag == 0:
                a_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                z = pl.matmul(a_left, b_right)
                acc = pl.move(z, target_memory=pl.MemorySpace.Vec)
            else:
                a_mat2 = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left2 = pl.move(a_mat2, target_memory=pl.MemorySpace.Left)
                b_mat2 = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right2 = pl.move(b_mat2, target_memory=pl.MemorySpace.Right)
                z2 = pl.matmul(a_left2, b_right2)
                acc = pl.move(z2, target_memory=pl.MemorySpace.Vec)
            out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc, [0, 0], out_0)
            return out_0

    aiv = _run_pipeline(Before).get_function("main_incore_0_aiv")
    assert aiv is not None, "expand should produce an AIV function"

    if_stmts = [s for s in ir.flatten_to_stmts(aiv.body) if isinstance(s, ir.IfStmt)]
    assert len(if_stmts) == 1, f"expected one AIV if-statement, got {len(if_stmts)}"
    if_stmt = if_stmts[0]

    for label, branch in (("then", if_stmt.then_body), ("else", if_stmt.else_body)):
        assert branch is not None, f"{label} branch must exist"
        stmts = ir.flatten_to_stmts(branch)
        # The branch carries a result, so it must still end with the YieldStmt.
        assert isinstance(stmts[-1], ir.YieldStmt), (
            f"{label} branch must end with YieldStmt, got {type(stmts[-1]).__name__} "
            f"(tfree appended after the yield => malformed body)"
        )
        # The popped tile is freed immediately *before* that yield.
        assert _op_name(stmts[-2]) == "system.tfree_to_aic", (
            f"{label} branch must emit tfree_to_aic right before the yield, got {_op_name(stmts[-2])!r}"
        )
        # Sanity: the branch opens with the matching tpop.
        assert _op_name(stmts[0]) == "tile.tpop_from_aic", (
            f"{label} branch should open with tile.tpop_from_aic, got {_op_name(stmts[0])!r}"
        )


def test_gm_sync_hoisted_before_consumer_loop_on_a2a3():
    """Issue #1564: cube->vector GM dependency whose consumer load is nested in a
    loop.

    The cube stores the matmul result to a GM scratch at the function top level;
    the vector reads that scratch back per-row inside a ``for`` loop. The fence
    must be a single tpush (after the store) paired with a single tpop *hoisted
    before the loop* — not one tpop per iteration (which would be N tpops for one
    tpush and deadlock). Before the fix the producer/consumer lived in different
    structural bodies, so no fence was emitted at all and the lanes raced (the
    first iterations read the scratch before the store landed).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay_loop(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            # AIC lane: matmul, then store the cube result into GM scratch (top level).
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_acc = pl.matmul(x_left, y_right)
            scratch = pl.store(z_acc, [0, 0], scratch)
            # AIV lane: read the same GM scratch back per-row, inside a loop.
            for r in pl.range(16):
                row = pl.load(scratch, [r, 0], [1, 64], target_memory=pl.MemorySpace.Vec)
                row = pl.exp(row)
                out = pl.store(row, [r, 0], out)
            return out

    After = _run_pipeline(Before)
    aic = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIC)
    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)

    # Producer lane: exactly one tpush fencing the cube store.
    aic_stmts = ir.flatten_to_stmts(aic.body)
    assert sum(_op_name(s) == "tile.tpush_to_aiv" for s in aic_stmts) == 1, (
        "AIC must emit exactly one tpush_to_aiv after the cube store"
    )

    # Consumer lane: the fence tpop/tfree must sit before the scatter loop, never
    # inside it (one tpop per tpush, not one per iteration).
    aiv_top = ir.flatten_to_stmts(aiv.body)
    for_idx = next((i for i, s in enumerate(aiv_top) if isinstance(s, ir.ForStmt)), None)
    assert for_idx is not None, "AIV must still contain the per-row scatter loop"
    for_stmt = aiv_top[for_idx]
    assert isinstance(for_stmt, ir.ForStmt)
    before_loop = aiv_top[:for_idx]
    assert any(_op_name(s) == "tile.tpop_from_aic" for s in before_loop), (
        "fence tpop_from_aic must be hoisted before the consumer loop"
    )
    assert any(_op_name(s) == "system.tfree_to_aic" for s in before_loop), (
        "fence tfree_to_aic must be hoisted before the consumer loop"
    )
    loop_body = ir.flatten_to_stmts(for_stmt.body)
    assert not any(_op_name(s) == "tile.tpop_from_aic" for s in loop_body), (
        "no per-iteration tpop: N tpops for a single tpush would deadlock"
    )


def test_two_gm_syncs_hoisted_to_same_loop_on_a2a3():
    """Issue #1564 follow-up: two cube->vector GM dependencies whose consumer
    loads sit in the *same* loop must each get their own hoisted fence.

    Both fences key on the same enclosing loop stmt, so a plain
    ``map<Stmt*, GmSyncPop>`` would keep only the last one and leave the other
    scratch unsynchronized. The collection uses a multimap and emits every fence
    via ``equal_range``, so the AIV must open the loop with two tpop_from_aic.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def gm_relay_two(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            scratch_a: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            scratch_b: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            # AIC lane: two matmuls -> two GM scratch tensors (top level).
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_a = pl.matmul(x_left, y_right)
            scratch_a = pl.store(z_a, [0, 0], scratch_a)
            z_b = pl.matmul(x_left, y_right)
            scratch_b = pl.store(z_b, [0, 0], scratch_b)
            # AIV lane: read BOTH scratch tensors per-row inside ONE loop.
            for r in pl.range(16):
                a = pl.load(scratch_a, [r, 0], [1, 64], target_memory=pl.MemorySpace.Vec)
                b = pl.load(scratch_b, [r, 0], [1, 64], target_memory=pl.MemorySpace.Vec)
                s = pl.add(a, b)
                out = pl.store(s, [r, 0], out)
            return out

    After = _run_pipeline(Before)
    aic = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIC)
    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)

    aic_stmts = ir.flatten_to_stmts(aic.body)
    assert sum(_op_name(s) == "tile.tpush_to_aiv" for s in aic_stmts) == 2, (
        "two cube stores must produce two tpush_to_aiv fences"
    )

    aiv_top = ir.flatten_to_stmts(aiv.body)
    for_idx = next((i for i, s in enumerate(aiv_top) if isinstance(s, ir.ForStmt)), None)
    assert for_idx is not None, "AIV must still contain the consumer loop"
    for_stmt = aiv_top[for_idx]
    assert isinstance(for_stmt, ir.ForStmt)
    before_loop = aiv_top[:for_idx]
    n_pop = sum(_op_name(s) == "tile.tpop_from_aic" for s in before_loop)
    assert n_pop == 2, f"both fences must be hoisted before the loop, got {n_pop}"
    loop_body = ir.flatten_to_stmts(for_stmt.body)
    assert not any(_op_name(s) == "tile.tpop_from_aic" for s in loop_body), (
        "fences must not be emitted per iteration"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
