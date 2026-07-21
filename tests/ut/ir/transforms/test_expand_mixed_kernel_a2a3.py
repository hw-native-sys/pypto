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


def test_explicit_sync_core_type_routes_each_event_to_one_lane():
    """Mixed JIT kernels must not duplicate set/wait onto both core types."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def explicit_sync(
            self,
            workspace: pl.Tensor[[256], pl.INT64],
            x: pl.Tensor[[16], pl.FP32],
        ) -> pl.Tensor[[16], pl.FP32]:
            pl.system.set_ffts(workspace)
            pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type="aiv")
            pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type="aic")
            return x

    after = _run_pipeline(Before)
    aic = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIC)
    aiv = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV)
    aic_ops = [_op_name(stmt) for stmt in ir.flatten_to_stmts(aic.body)]
    aiv_ops = [_op_name(stmt) for stmt in ir.flatten_to_stmts(aiv.body)]

    assert aic_ops.count("system.set_ffts") == 1
    assert aic_ops.count("system.sync_wait") == 1
    assert "system.sync_set" not in aic_ops
    assert aiv_ops.count("system.set_ffts") == 1
    assert aiv_ops.count("system.sync_set") == 1
    assert "system.sync_wait" not in aiv_ops


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


def test_split_slot_num_override_sizes_c2v_ring_on_a2a3():
    """``pl.split(mode, slot_num=N)`` (propagated as the ``slot_num`` function
    attr) overrides the hardcoded ring depth on the automatic cube->vector pipe.

    Mirrors ``test_c2v_boundary_preserves_vec_pop_layout_on_a2a3`` but with
    ``slot_num=16`` instead of the default 8: the reserved buffer grows to
    ``slot_size * 16`` (65536) and both ``initialize_pipe`` calls carry an
    explicit ``slot_num=16`` attribute.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "slot_num": 16})
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
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN, "slot_num": 16})
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
                slot_num=16,
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

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN, "slot_num": 16})
        def main_incore_0_aiv(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            main_incore_0_c2v_slot_buffer = pl.reserve_buffer(
                name="main_incore_0_c2v_slot_buffer", size=65536, base=-1
            )
            pl.aiv_initialize_pipe(
                main_incore_0_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
                slot_num=16,
            )
            z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            pl.tfree_to_aic(z_vec)
            return out_0_store

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN, "slot_num": 16})
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


def _run_to_expand_with_flatten(program: ir.Program) -> ir.Program:
    """Run the tile-pipeline prefix needed for a tile.transpose to reach
    ExpandMixedKernel: FlattenTileNdTo2D adds the transpose scratch arg that the
    TileOps2D verifier (an ExpandMixedKernel prerequisite) requires.
    """
    p = passes.convert_to_ssa()(program)
    p = passes.lower_composite_ops()(p)
    p = passes.flatten_tile_nd_to_2d()(p)
    p = passes.infer_tile_memory_space()(p)
    return passes.expand_mixed_kernel()(p)


def _assert_actionable_split_error(excinfo, mode_name: str) -> None:
    """The error must name the split mode and surface both fix directions so the
    user can act without reading the source: drop the split, or remove the
    transpose."""
    msg = str(excinfo.value)
    assert "swaps the split axis" in msg, msg
    assert mode_name in msg, msg
    assert "pl.SplitMode.NONE" in msg, msg  # direction 1: drop the split
    assert "column slice" in msg, msg  # direction 2: remove the transpose


def test_unsplittable_transpose_raises_actionable_error():
    """A requested UP_DOWN split is rejected with a ValueError when the kernel
    contains a tile.transpose that swaps the split axis.

    tile.transpose swaps axes, so the per-lane split data migrates to the other
    dim while SplitVectorKernel still halves the original split axis — it cannot
    type such a transpose correctly. The split is a perf decision the user owns,
    so the pass fails loud instead of silently compiling it un-split. Here the
    [16, 8] matmul result is transposed under UP_DOWN (split dim 0, non-singleton).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_hazard(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 8], pl.BF16],
            out_0: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
        ) -> pl.Tensor[[8, 16], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 8], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 8] (cube result)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            zt = pl.transpose(z_vec, axis1=0, axis2=1)  # source dim0=16 non-singleton -> error
            out_0 = pl.store(zt, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_expand_with_flatten(Before)
    _assert_actionable_split_error(excinfo, "UP_DOWN")


def test_left_right_transpose_also_raises():
    """The error is mode-independent: a LEFT_RIGHT split whose transpose source is
    non-singleton on the split axis (dim 1) is also rejected, because the
    transpose migrates the column split axis just as UP_DOWN migrates rows."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def t_lr(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 16], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 16], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 16] (cube result)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            zt = pl.transpose(z_vec, axis1=0, axis2=1)  # source dim1=16 non-singleton -> error
            out_0 = pl.store(zt, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_expand_with_flatten(Before)
    _assert_actionable_split_error(excinfo, "LEFT_RIGHT")


def test_singleton_split_axis_transpose_keeps_split():
    """A transpose whose source is singleton on the split axis carries no split
    data (the no-op broadcast case), so the split is preserved. Here a [1, 16]
    source is transposed under UP_DOWN (split dim 0 == 1), so it is NOT rejected
    and the AIV keeps its split attr with no dual-AIV dispatch."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_singleton(
            self,
            x: pl.Tensor[[1, 128], pl.BF16],
            y: pl.Tensor[[128, 16], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 16], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [1, 16] (cube result)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            zt = pl.transpose(z_vec, axis1=0, axis2=1)  # source dim0=1 singleton -> kept split
            out_0 = pl.store(zt, [0, 0], out_0)
            return out_0

    After = _run_to_expand_with_flatten(Before)
    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert aiv.attrs.get("dual_aiv_dispatch") is not True, (
        f"a singleton-split-axis transpose must not be rejected, got attrs={dict(aiv.attrs)}"
    )
    assert "split" in aiv.attrs, (
        f"the requested split must be preserved when the transpose is a no-op on the split axis, "
        f"got attrs={dict(aiv.attrs)}"
    )


def test_unsplittable_int8_transpose_raises_actionable_error():
    """The error is dtype-independent: an int8 transpose that swaps a
    non-singleton split axis is rejected just like the fp/bf16 cases.

    A bf16 matmul (cube) keeps the kernel mixed; separately, an int8 [16, 32]
    tensor is loaded into Vec and transposed under UP_DOWN (source dim0=16
    non-singleton), so ExpandMixedKernel rejects the split.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_hazard_i8(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 8], pl.BF16],
            q: pl.Tensor[[16, 32], pl.INT8],
            out_0: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
            out_1: pl.Out[pl.Tensor[[32, 16], pl.INT8]],
        ) -> pl.Tensor[[16, 8], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 8], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 8] (cube result, keeps the kernel mixed)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            out_0 = pl.store(z_vec, [0, 0], out_0)
            q_vec = pl.load(q, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            qt = pl.transpose(q_vec, axis1=0, axis2=1)  # int8 source dim0=16 non-singleton -> error
            out_1 = pl.store(qt, [0, 0], out_1)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_expand_with_flatten(Before)
    _assert_actionable_split_error(excinfo, "UP_DOWN")


# ---------------------------------------------------------------------------
# Regression: GM tensor written on the AIV lane, consumed by the AIC matmul.
#
# Mirror of the GM-mediated cross-lane store/load handshake test in the OPPOSITE
# direction. The vector lane stores into a GM tensor (producing a fresh SSA
# version); the cube lane loads that version back as the matmul left operand.
# ExpandMixedKernel builds the AIC body's param/clone map from func->params_
# only, so the AIV-defined GM version is a dangling free var on the AIC lane ->
# the printer marks it __FREE_VAR and PTO codegen's GetOrCreateTensorView
# crashes. The fix repoints the cross-half GM use onto the shared base parameter
# (straight-line tile.store result, IfStmt phi, and ForStmt return_var forms).
# ---------------------------------------------------------------------------


def _expand_no_verify(program: ir.Program) -> ir.Program:
    """SSA -> infer-memory -> expand-mixed-kernel, expanding with verification off.

    Verification is disabled (empty PassContext) so a mis-routed free Var is
    observable as a returned-IR property instead of a verifier crash -- matching
    the __FREE_VAR check below.
    """
    p = passes.infer_tile_memory_space()(passes.convert_to_ssa()(program))
    with passes.PassContext([]):
        return passes.expand_mixed_kernel()(p)


def _assert_no_free_var(program: ir.Program) -> None:
    """A dangling/free Var prints with a ``__FREE_VAR`` suffix and later crashes
    PTO codegen's GetOrCreateTensorView; assert none survive the split."""
    assert "__FREE_VAR" not in ir.python_print(program)


def _assert_aic_loads_reference_params(after: ir.Program) -> None:
    """Every cube-lane ``tile.load`` source must resolve to an AIC parameter.

    The AIV lane writes a GM tensor (a fresh SSA version) that the AIC lane loads
    back for the matmul. ExpandMixedKernel must thread that cross-half reference
    onto the AIC function's shared GM parameter; otherwise the AIC body loads
    from a Var defined only on the AIV lane -- a dangling reference codegen
    cannot resolve. ``unique_id`` (not ``id(...)``) is the reliable identity key
    across binding-layer Var wrappers.
    """
    aic = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIC)
    param_ids = {p.unique_id for p in aic.params}
    dangling = []
    for s in ir.flatten_to_stmts(aic.body):
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call) and _op_name(s) == "tile.load":
            src = s.value.args[0]  # tile.load arg0 is the source tensor
            if isinstance(src, ir.Var) and src.unique_id not in param_ids:
                dangling.append(src.name_hint)
    assert not dangling, (
        f"AIC tile.load reads non-parameter Var(s) {dangling}: a cross-half GM "
        f"SSA version was not threaded into the AIC parameter map"
    )


def test_aiv_gm_write_consumed_by_cube_straightline_resolves_to_param():
    """Straight-line ``tile.store`` result (``scratch__ssa_v1``).

    AIV: load -> add -> store into GM ``scratch``. AIC: load that scratch back ->
    move Left -> matmul(scratch, b) -> move Vec. Before the fix the AIC load
    reads the AIV-defined store-result version as a free var.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def gm_aiv_to_aic(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            # AIV lane: produce a GM scratch tensor (vector store).
            a_tile = pl.load(a, [0, 0], [16, 128])
            s = pl.add(a_tile, a_tile)
            scratch = pl.store(s, [0, 0], scratch)
            # AIC lane: cube matmul consumes the AIV-written scratch.
            s_mat = pl.load(scratch, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            s_left = pl.move(s_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(s_left, b_right)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            out_0 = pl.store(z_vec, [0, 0], out_0)
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)
    _assert_aic_loads_reference_params(After)


def test_aiv_gm_write_in_if_consumed_by_cube_resolves_to_param():
    """Conditional write -> IfStmt phi return_var (``scratch__phi_v*``).

    Both branches store into GM ``scratch`` (so its post-if value is a phi whose
    yields resolve to the same ``scratch`` origin); the AIC matmul loads that
    phi. Exercises the IfStmt return_var origin propagation.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def gm_phi(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            flag: pl.Scalar[pl.INT32],
            scratch: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            # AIV lane: both branches write GM scratch -> post-if phi version.
            if flag == 0:
                a0 = pl.load(a, [0, 0], [16, 128])
                s0 = pl.add(a0, a0)
                scratch = pl.store(s0, [0, 0], scratch)
            else:
                a1 = pl.load(a, [0, 0], [16, 128])
                s1 = pl.mul(a1, a1)
                scratch = pl.store(s1, [0, 0], scratch)
            # AIC lane: cube matmul consumes the phi-versioned scratch.
            s_mat = pl.load(scratch, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            s_left = pl.move(s_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(s_left, b_right)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            out_0 = pl.store(z_vec, [0, 0], out_0)
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)
    _assert_aic_loads_reference_params(After)


def test_aiv_gm_write_in_loop_consumed_by_cube_resolves_to_param():
    """Loop write -> ForStmt return_var (``scratch__rv_v*``).

    The loop carries GM ``scratch`` as an iter_arg (init = the ``scratch`` param)
    and stores into it each iteration; its post-loop value is the return_var that
    the AIC matmul loads. Covered by origin_map (return_var -> iter_arg -> init).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def gm_loop(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            # AIV lane: store GM scratch each iteration -> post-loop return_var.
            for i in pl.range(2):  # noqa: B007 - loop index unused by design
                a_tile = pl.load(a, [0, 0], [16, 128])
                s = pl.add(a_tile, a_tile)
                scratch = pl.store(s, [0, 0], scratch)
            # AIC lane: cube matmul consumes the loop-carried scratch version.
            s_mat = pl.load(scratch, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            s_left = pl.move(s_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(s_left, b_right)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            out_0 = pl.store(z_vec, [0, 0], out_0)
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)
    _assert_aic_loads_reference_params(After)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
