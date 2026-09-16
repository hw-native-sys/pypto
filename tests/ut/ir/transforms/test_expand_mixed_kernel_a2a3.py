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
import pypto.language.distributed as pld
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


def test_automatic_mx_scale_v2c_requires_a5_backend():
    """A2A3 rejects the A5-only MX byte-preserving V2C transport."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(self, src: pl.Tensor[[16, 64], pl.FP32]):
            _quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)
            _scale_mat = pl.move(
                scale,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.row_major,
            )

    with pytest.raises(ValueError, match="requires an A5 backend"):
        _run_pipeline(Before)


def test_direct_incore_defer_wait_requires_task_level_waiter_contract():
    """A direct InCore helper must not bypass task-level waiter validation."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def direct_wait(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pld.system.defer_wait(
                signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    with pytest.raises(ValueError, match="bypasses the deferred-waiter task contract"):
        _run_pipeline(Program)


def test_aiv_defer_wait_requires_task_level_waiter_contract():
    """A specialized AIV kernel cannot bypass the InCore scope validator."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def direct_wait(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pld.system.defer_wait(
                signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    with pytest.raises(ValueError, match="bypasses the deferred-waiter task contract"):
        _run_pipeline(Program)


def test_forged_waiter_marker_without_task_level_call_site_is_rejected():
    """The printable internal attr is not accepted as provenance on its own."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def direct_wait(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(
                signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    with pytest.raises(ValueError, match="no task-level Orchestration call site"):
        _run_pipeline(Program)


def test_deferred_waiter_dispatched_from_a_graph_body_is_accepted():
    """The caller check asks "is this a task-level orchestration body".

    A Graph body is one — its call sites are task launches like any other
    orchestration body's. Requiring strictly `Orchestration` rejects a legal
    Graph that dispatches a deferred waiter, with a message telling the author
    to do what they already did.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Graph)
        def layer(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            self.waiter(signal)

    _run_pipeline(Program)


def test_forged_waiter_plain_call_early_resolve_fails_closed():
    """A plain GlobalVar call cannot smuggle an unsafe early-resolve attr."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            self.waiter(signal, attrs={"allow_early_resolve": True})

    with pytest.raises(ValueError, match="cannot use allow_early_resolve=True"):
        _run_pipeline(Program)


def test_forged_waiter_plain_call_core_num_fails_closed():
    """A plain GlobalVar call cannot turn the waiter into an SPMD launch."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            self.waiter(signal, attrs={"core_num": 2})

    with pytest.raises(ValueError, match="single-block task"):
        _run_pipeline(Program)


def test_forged_waiter_plain_call_predicate_fails_closed():
    """A printable waiter marker cannot bypass the no-predicate contract."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            gate: pl.Tensor[[1], pl.INT32],
        ):
            self.waiter(signal, attrs={"predicate": gate[0] > 0})

    with pytest.raises(ValueError, match="cannot use a dispatch predicate"):
        _run_pipeline(Program)


def test_forged_waiter_plain_call_sync_start_fails_closed():
    """A printable waiter marker cannot bypass the single-block contract."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            self.waiter(signal, attrs={"sync_start": True})

    with pytest.raises(ValueError, match="single-block task"):
        _run_pipeline(Program)


def test_valid_waiter_aiv_form_remains_valid_on_expand_rerun():
    """The audit revalidates the marked AIV form produced by the first run."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                pld.system.defer_wait(
                    signal,
                    offsets=[0, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

    outlined = passes.outline_incore_scopes()(passes.convert_to_ssa()(Program))
    expanded = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(outlined))
    waiter = expanded.get_function("waiter")
    assert waiter is not None
    assert waiter.func_type == ir.FunctionType.AIV
    assert waiter.attrs["deferred_completion_waiter"] is True

    rerun = passes.expand_mixed_kernel()(expanded)
    rerun_waiter = rerun.get_function("waiter")
    assert rerun_waiter is not None
    assert rerun_waiter.func_type == ir.FunctionType.AIV


def test_validated_waiter_accepts_trailing_empty_return():
    """Function normalization may append a no-value return terminator."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(
                signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )
            return  # noqa: PLR1711 - explicit empty terminator is the test subject

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            self.waiter(signal)

    after = _run_pipeline(Program)
    waiter = after.get_function("waiter")
    assert waiter is not None
    assert waiter.func_type == ir.FunctionType.AIV


def test_hand_written_group_members_share_canonical_abi():
    """Different AIC/AIV signatures are normalized to the Group ABI (#2097).

    ``MixedKernels`` gives every active lane the same runtime argument array.
    A hand-written Group is allowed to capture the union of its member inputs,
    so ExpandMixedKernel must make both member wrappers consume that union in
    the same order instead of letting orchestration codegen use the first call's
    shorter signature for both lanes.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            self.cube_side(q, sink, task)
            result = self.vec_side(out, task)
            return result

    after = _run_pipeline(Program)
    cube = after.get_function("cube_side")
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert cube is not None
    assert vec is not None
    assert group is not None
    expected_names = [param.name_hint for param in group.params]

    assert [param.name_hint for param in cube.params] == expected_names
    assert [param.name_hint for param in vec.params] == expected_names
    assert cube.param_directions == group.param_directions
    assert vec.param_directions == group.param_directions

    member_calls = []
    for stmt in ir.flatten_to_stmts(group.body):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            member_calls.append(stmt.value)
        elif isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            member_calls.append(stmt.expr)
    assert len(member_calls) == 2
    for call in member_calls:
        assert [arg.name_hint for arg in call.args] == expected_names


def test_hand_written_no_split_pipe_group_infers_dual_aiv_dispatch():
    """910B explicit pipe members get the same lane-1 handshake replay as auto-expanded mixed kernels."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            v2c = pl.reserve_buffer(name="probe_v2c", size=4096, base=pl.AUTO)
            c2v_peer = pl.import_peer_buffer(name="probe_c2v", peer_func="vec_side")
            pl.aic_initialize_pipe(c2v_peer, v2c, dir_mask=3, slot_size=512)
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            c2v = pl.reserve_buffer(name="probe_c2v", size=4096, base=pl.AUTO)
            v2c_peer = pl.import_peer_buffer(name="probe_v2c", peer_func="cube_side")
            pl.aiv_initialize_pipe(c2v, v2c_peer, dir_mask=3, slot_size=512)
            tile = pl.tpop_from_aic(shape=[16, 16], dtype=pl.FP32, split=0)
            pl.tfree_to_aic(tile, split=0)
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            self.cube_side(q, sink)
            return self.vec_side(out)

    after = _run_pipeline(Program)
    vec = after.get_function("vec_side")
    assert vec is not None
    assert vec.attrs.get("dual_aiv_dispatch") is True


def test_reused_hand_written_group_members_get_private_adapters():
    """A member used outside its Group keeps its original ABI."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            self.cube_side(q, sink, task)
            result = self.vec_side(out, task)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            sink = self.cube_side(q, sink, task)
            return self.mixed(q, sink, out, task)

    after = _run_pipeline(Program)
    group = after.get_function("mixed")
    aic_adapter = after.get_function("mixed_aic_adapter")
    aiv_adapter = after.get_function("mixed_aiv_adapter")
    cube = after.get_function("cube_side")
    vec = after.get_function("vec_side")
    assert group is not None
    assert aic_adapter is not None
    assert aiv_adapter is not None
    assert cube is not None
    assert vec is not None
    canonical_names = [param.name_hint for param in group.params]

    assert len(cube.params) == 3
    assert len(vec.params) == 2
    assert [param.name_hint for param in aic_adapter.params] == canonical_names
    assert [param.name_hint for param in aiv_adapter.params] == canonical_names
    group_text = group.as_python()
    assert "mixed_aic_adapter(" in group_text
    assert "mixed_aiv_adapter(" in group_text


def test_hand_written_group_submit_remaps_no_dep_override():
    """A widened Submit keeps no_dep attached to the original member argument."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return q

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            q: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            task: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            _cube_out, _cube_tid = pl.submit(self.cube_side, q, task)
            vec_out, _vec_tid = pl.submit(
                self.vec_side,
                pl.no_dep(out),
                task,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar]},
            )
            return vec_out

    after = _run_pipeline(Program)
    group = after.get_function("mixed")
    assert group is not None
    submits = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit)
    ]
    vec_submit = next(submit for submit in submits if submit.op.name == "vec_side")

    assert list(vec_submit.args) == list(group.params)
    assert vec_submit.attrs["arg_direction_overrides"] == [1]
    assert list(vec_submit.arg_directions) == []


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
            pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
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
                name="main_incore_0_v2c_slot_buffer", size=8192, base=-1
            )
            main_incore_0_c2v_slot_buffer_import = pl.import_peer_buffer(
                name="main_incore_0_c2v_slot_buffer", peer_func="main_incore_0_aiv"
            )
            pl.aic_initialize_pipe(
                main_incore_0_c2v_slot_buffer_import,
                main_incore_0_v2c_slot_buffer,
                dir_mask=3,
                slot_size=4096,
                slot_num=2,
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
                name="main_incore_0_c2v_slot_buffer", size=8192, base=-1
            )
            pl.aiv_initialize_pipe(
                main_incore_0_c2v_slot_buffer,
                main_incore_0_v2c_slot_buffer_import,
                dir_mask=3,
                slot_size=4096,
                slot_num=2,
            )
            x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
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
                slot_num=2,
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
                name="main_incore_0_c2v_slot_buffer", size=8192, base=-1
            )
            pl.aiv_initialize_pipe(
                main_incore_0_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
                slot_num=2,
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
                slot_num=2,
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
            gm_relay_c2v_slot_buffer = pl.reserve_buffer(name="gm_relay_c2v_slot_buffer", size=8192, base=-1)
            pl.aiv_initialize_pipe(
                gm_relay_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=4096,
                slot_num=2,
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
    ``slot_num=16`` instead of the default 2: the reserved buffer grows to
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


# ---------------------------------------------------------------------------
# Regression (issue #2716): a multi-entry FIFO's carries survive the split.
#
# A two-entry FIFO rotates: slot 0's next value IS slot 1's iter_arg, so slot 1
# is read nowhere but the loop's own yield. StripDeadIterArgs seeds liveness from
# the body *excluding* that yield, so it used to call slot 1 dead and drop it --
# while the positional yield filter kept slot 0's value naming the iter_arg it
# had just deleted. Two outcomes, same defect: where no repair reached the loop
# the free Var survived to PTO codegen ("no MLIR mapping for MemRef base ..."),
# and where one did the rotation was silently rewritten into a self-carry, which
# freezes the FIFO and computes the wrong answer.
# ---------------------------------------------------------------------------


def _first_carry_loop(stmt: ir.Stmt) -> ir.ForStmt:
    """The first ForStmt with iter_args in `stmt`, searched depth-first."""
    for inner in ir.flatten_to_stmts(stmt):
        if isinstance(inner, ir.ForStmt):
            if inner.iter_args:
                return inner
            return _first_carry_loop(inner.body)
    raise AssertionError("no loop with iter_args found")


def _assert_fifo_rotation_intact(after: ir.Program) -> None:
    """The AIV lane keeps all three carries and still rotates slot 1 into slot 0.

    Checking the yield operand by identity is the point: a dropped slot 1 leaves
    slot 0 yielding either a free Var (crash) or its own iter_arg (silent
    self-carry), and only an identity comparison separates the latter from the
    rotation the author wrote.
    """
    aiv = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV)
    loop = _first_carry_loop(aiv.body)
    assert len(loop.iter_args) == 3, (
        f"expected the 3 authored carries to survive, got {[a.name_hint for a in loop.iter_args]}"
    )
    tail = ir.flatten_to_stmts(loop.body)[-1]
    assert isinstance(tail, ir.YieldStmt), f"loop body must end with YieldStmt, got {type(tail).__name__}"
    assert len(tail.value) == 3, f"expected 3 yielded values, got {len(tail.value)}"
    rotated = tail.value[1]
    assert isinstance(rotated, ir.Var), f"slot 1 must yield the slot-2 carry, got {type(rotated).__name__}"
    assert rotated.unique_id == loop.iter_args[2].unique_id, (
        f"slot 1 must yield carry '{loop.iter_args[2].name_hint}' (the FIFO rotation), "
        f"got '{rotated.name_hint}'"
    )


def test_two_entry_fifo_rotation_survives_split_inside_carry_less_loop():
    """FIFO loop nested in a carry-less loop -- the form that reached codegen.

    No repair descends into a loop without iter_args, so a dropped carry leaves
    the surviving slot's yield naming a Var the loop no longer binds, and that
    free Var is only caught ~30 passes later by PTO codegen.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def fifo_nested(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            for _t in pl.range(2):  # noqa: B007 - carry-less by design, see docstring
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul(a_left, b_right)  # AIC
                acc_vec = pl.move(acc, target_memory=pl.MemorySpace.Vec)  # C->V boundary
                p0 = pl.add(acc_vec, acc_vec)  # AIV: FIFO entry 0
                p1 = pl.add(p0, p0)  # AIV: FIFO entry 1
                s = pl.add(p0, p1)
                for _i, (s_it, p0_it, p1_it) in pl.range(4, init_values=(s, p0, p1)):  # noqa: B007
                    s_next = pl.add(s_it, p0_it)  # only entry 0 is consumed
                    p_new = pl.add(acc_vec, acc_vec)
                    s, p0, p1 = pl.yield_(s_next, p1_it, p_new)  # rotate 1 -> 0
                _sink = pl.store(s, [0, 0], out_0)  # noqa: F841 - keeps the outer loop carry-less
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)
    _assert_fifo_rotation_intact(After)


def test_two_entry_fifo_rotation_not_rewritten_to_self_carry():
    """FIFO loop at function top level -- the form that compiled but was wrong.

    Here FixupDanglingYieldValues does reach the loop, and its per-slot fallback
    (yield the slot's own iter_arg) turns the rotation into a self-carry: entry 0
    never advances and the freshly computed entry is dead-coded away. No error is
    raised, so only an identity check on the yield operand catches it.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def fifo_toplevel(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(a_left, b_right)  # AIC
            acc_vec = pl.move(acc, target_memory=pl.MemorySpace.Vec)  # C->V boundary
            p0 = pl.add(acc_vec, acc_vec)
            p1 = pl.add(p0, p0)
            s = pl.add(p0, p1)
            for _i, (s_it, p0_it, p1_it) in pl.range(4, init_values=(s, p0, p1)):  # noqa: B007
                s_next = pl.add(s_it, p0_it)
                p_new = pl.add(acc_vec, acc_vec)
                s, p0, p1 = pl.yield_(s_next, p1_it, p_new)  # rotate 1 -> 0
            out_0 = pl.store(s, [0, 0], out_0)
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)
    _assert_fifo_rotation_intact(After)


def test_nested_yield_of_an_enclosing_scope_value_is_not_rewritten():
    """A nested loop may legitimately yield a value bound outside its own body.

    `FixupDanglingYieldValues` walks a block accumulating the definitions it has
    passed, and every recursion used to restart that set empty. A loop reached
    through one therefore could not see the enclosing loop's variable or
    anything defined before it, so a perfectly valid yield looked dangling and
    was replaced by the slot's own carry -- the same silent self-carry this file
    guards against above, arriving by a different route.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def outer_scope_yield(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(a_left, b_right)  # AIC
            acc_vec = pl.move(acc, target_memory=pl.MemorySpace.Vec)  # C->V boundary
            outer_val = pl.add(acc_vec, acc_vec)  # bound before the carry-less loop
            for _t in pl.range(2):  # noqa: B007 - carry-less, so the repair recurses
                seed = pl.add(outer_val, outer_val)
                for _i, (s_it,) in pl.range(4, init_values=(seed,)):  # noqa: B007
                    s = pl.yield_(outer_val)  # legitimate: bound two scopes up
                _sink = pl.store(s, [0, 0], out_0)  # noqa: F841 - keeps the outer loop carry-less
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)

    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)
    loop = _first_carry_loop(aiv.body)
    tail = ir.flatten_to_stmts(loop.body)[-1]
    assert isinstance(tail, ir.YieldStmt), f"loop body must end with YieldStmt, got {type(tail).__name__}"
    assert len(tail.value) == 1, f"expected 1 yielded value, got {len(tail.value)}"
    yielded = tail.value[0]
    assert isinstance(yielded, ir.Var), f"expected a Var operand, got {type(yielded).__name__}"
    carry_ids = {a.unique_id for a in loop.iter_args}
    assert yielded.unique_id not in carry_ids, (
        f"the enclosing-scope value was rewritten into a self-carry on '{yielded.name_hint}'"
    )


def test_nested_yield_of_a_function_parameter_is_not_rewritten():
    """A parameter is bound by the signature, so no statement in the body defines it.

    The scope threaded into the walk starts at the function body, so a yield
    naming a parameter has nothing backing it unless the parameters are seeded
    in -- and it would be replaced by the slot's own carry, dropping the update.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def param_yield(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            scratch: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(a_left, b_right)  # AIC
            acc_vec = pl.move(acc, target_memory=pl.MemorySpace.Vec)  # C->V boundary
            for _t in pl.range(2):  # noqa: B007 - carry-less, so the repair recurses
                for _i, (s_it,) in pl.range(4, init_values=(scratch,)):  # noqa: B007
                    s = pl.yield_(scratch)  # legitimate: bound by the signature
                _sink = pl.store(acc_vec, [0, 0], s)  # noqa: F841 - keeps the outer loop carry-less
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)

    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)
    loop = _first_carry_loop(aiv.body)
    tail = ir.flatten_to_stmts(loop.body)[-1]
    assert isinstance(tail, ir.YieldStmt), f"loop body must end with YieldStmt, got {type(tail).__name__}"
    assert len(tail.value) == 1, f"expected 1 yielded value, got {len(tail.value)}"
    yielded = tail.value[0]
    assert isinstance(yielded, ir.Var), f"expected a Var operand, got {type(yielded).__name__}"
    carry_ids = {c.unique_id for c in loop.iter_args}
    assert yielded.unique_id not in carry_ids, (
        f"the parameter yield was rewritten into a self-carry on '{yielded.name_hint}'"
    )
    param_ids = {p.unique_id for p in aiv.params}
    assert yielded.unique_id in param_ids, (
        f"expected the yield to keep naming the parameter, got '{yielded.name_hint}'"
    )


def test_nested_yield_of_a_cross_core_result_is_not_rewritten():
    """A boundary move's destination has no defining statement left on either lane.

    `BuildCoreBody` replaces the C->V ``tile.move`` with a ``tpop`` that defines a
    fresh Var and records the original in ``tpop_var_remap``; references to it are
    repointed later by ``apply_tpop_remap``. Between those two steps the original
    looks undefined, so a yield naming it directly must not be "repaired" -- once
    the reference is replaced by the slot's own carry there is nothing left for
    the remap to fix, and the loop silently stops updating.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def xcore_yield(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
            # An independent Vec seed, so the carry's init is not itself a
            # boundary result.
            seed = pl.cast(pl.load(a, [0, 0], [16, 128]), target_type=pl.FP32)
            for _t in pl.range(2):  # noqa: B007 - carry-less, so the repair recurses
                for _i, (v_it,) in pl.range(4, init_values=(seed,)):  # noqa: B007
                    acc = pl.matmul(a_left, b_right)  # AIC
                    acc_vec = pl.move(acc, target_memory=pl.MemorySpace.Vec)  # C->V boundary
                    v = pl.yield_(acc_vec)  # yields the transfer result directly
                _sink = pl.store(v, [0, 0], out_0)  # noqa: F841 - keeps the outer loop carry-less
            return out_0

    After = _expand_no_verify(Before)
    _assert_no_free_var(After)

    aiv = next(f for f in After.functions.values() if f.func_type == ir.FunctionType.AIV)
    loop = _first_carry_loop(aiv.body)
    body_stmts = ir.flatten_to_stmts(loop.body)
    tail = body_stmts[-1]
    assert isinstance(tail, ir.YieldStmt), f"loop body must end with YieldStmt, got {type(tail).__name__}"
    assert len(tail.value) == 1, f"expected 1 yielded value, got {len(tail.value)}"
    yielded = tail.value[0]
    assert isinstance(yielded, ir.Var), f"expected a Var operand, got {type(yielded).__name__}"
    carry_ids = {c.unique_id for c in loop.iter_args}
    assert yielded.unique_id not in carry_ids, (
        f"the cross-core transfer result was rewritten into a self-carry on '{yielded.name_hint}'"
    )
    # After apply_tpop_remap the operand must be the tpop this lane defines.
    defined_ids = {s.var.unique_id for s in body_stmts if isinstance(s, ir.AssignStmt)}
    assert yielded.unique_id in defined_ids, (
        f"yield operand '{yielded.name_hint}' is not defined in the loop body"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
