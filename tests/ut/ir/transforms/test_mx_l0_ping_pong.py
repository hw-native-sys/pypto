# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in MX K256 Mat/FIFO to ordered, independent K128 L0 halves."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def _a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _program(rows=64, k=256, n=256, with_loop=False):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[rows, k], pl.FP8E4M3FN],
            a_s: pl.Tensor[[rows, k // 32], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[k, n], pl.FP8E4M3FN],
            b_s: pl.Tensor[[k // 32, n], pl.FP8E8M0, pl.MX_B_NN],
            out: pl.Out[pl.Tensor[[rows, n], pl.FP32]],
        ) -> pl.Tensor[[rows, n], pl.FP32]:
            ta = pl.load(a, [0, 0], [rows, k], target_memory=pl.Mem.Mat)
            tas = pl.load(a_s, [0, 0], [rows, k // 32], target_memory=pl.Mem.Mat)
            tb = pl.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
            tbs = pl.load(b_s, [0, 0], [k // 32, n], target_memory=pl.Mem.Mat)
            c = pl.matmul_mx(ta, tas, tb, tbs)
            if with_loop:
                for _ki in pl.pipeline(1, 4, stage=2):
                    la = pl.load(a, [0, 0], [rows, k], target_memory=pl.Mem.Mat)
                    las = pl.load(a_s, [0, 0], [rows, k // 32], target_memory=pl.Mem.Mat)
                    lb = pl.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
                    lbs = pl.load(b_s, [0, 0], [k // 32, n], target_memory=pl.Mem.Mat)
                    c = pl.matmul_mx_acc(c, la, las, lb, lbs)
            return pl.store(c, [0, 0], out)

    return Before


def _run(program, enabled, planner=passes.MemoryPlanner.PYPTO):
    with passes.PassContext([], enable_mx_l0_ping_pong=enabled, memory_planner=planner):
        program = passes.convert_to_ssa()(program)
        program = passes.simplify()(program)
        program = passes.infer_tile_memory_space()(program)
        return passes.insert_mx_scale_addr()(program)


def _mixed_program(rows):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[rows, 256], pl.FP8E4M3FN],
            a_s: pl.Tensor[[rows, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[256, 256], pl.FP8E4M3FN],
            b_s: pl.Tensor[[8, 256], pl.FP8E8M0, pl.MX_B_NN],
            out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        ) -> pl.Tensor[[rows, 256], pl.FP32]:
            ub = pl.load(b, [0, 0], [256, 256], target_memory=pl.Mem.Vec)
            tb = pl.move(
                ub,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            ta = pl.load(a, [0, 0], [rows, 256], target_memory=pl.Mem.Mat)
            tas = pl.load(a_s, [0, 0], [rows, 8], target_memory=pl.Mem.Mat)
            tbs = pl.load(b_s, [0, 0], [8, 256], target_memory=pl.Mem.Mat)
            c = pl.matmul_mx(ta, tas, tb, tbs)
            return pl.store(c, [0, 0], out)

    return Before


def _run_to_allocate(program, planner):
    # Exercise the real mixed-kernel split and its DCE before sizing storage.
    # A capacity-only tile.create used to disappear on precisely this path.
    with passes.PassContext([], enable_mx_l0_ping_pong=True, memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        stop = manager.pass_names.index("AllocateMemoryAddr")
        for pass_obj in manager.passes[: stop + 1]:
            pipeline = passes.PassPipeline()
            pipeline.add_pass(pass_obj)
            program = pipeline.run(program)
        return program


def _assignments(program):
    found = []

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            found.append(stmt)
            super().visit_assign_stmt(stmt)

    Collect().visit_program(program)
    return found


def _named(program, name):
    op = ir.get_op(name)
    return [stmt for stmt in _assignments(program) if isinstance(stmt.value, ir.Call) and stmt.value.op == op]


def _shape(value):
    return [dim.value for dim in value.type.shape]


def _mx_ops(program):
    names = {ir.get_op("tile.matmul_mx").name, ir.get_op("tile.matmul_mx_acc").name}
    return [
        stmt
        for stmt in _assignments(program)
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name in names
    ]


@pytest.mark.parametrize("rows", [48, 64])
def test_full_k256_preserves_sources_and_ordered_fp32_chain(rows):
    after = _run(_program(rows=rows), True)
    first, second = _mx_ops(after)
    assert first.value.op == ir.get_op("tile.matmul_mx")
    assert second.value.op == ir.get_op("tile.matmul_mx_acc")
    assert second.value.args[0].unique_id == first.var.unique_id
    assert first.var.type.dtype == pl.FP32
    assert second.var.type.dtype == pl.FP32
    extracts = _named(after, "tile.extract")
    assert len(extracts) == 8
    for idx in (0, 4):
        view = extracts[idx].var.type.tile_view
        # Full CR is the implicit Left layout and may canonicalize to None.
        assert view is None or (
            view.blayout == ir.TileLayout.col_major and view.slayout == ir.TileLayout.row_major
        )
    assert [_shape(stmt.var) for stmt in extracts] == [[rows, 128], [rows, 4], [128, 256], [4, 256]] * 2
    assert [(stmt.value.args[1].value, stmt.value.args[2].value) for stmt in extracts] == [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 128),
        (0, 4),
        (128, 0),
        (4, 0),
    ]
    # Both halves read the original, unchanged Mat tiles. This also keeps one
    # K256 cross-core FIFO ownership interval around BOTH B copies.
    for i in range(4):
        assert extracts[i].value.args[0].unique_id == extracts[i + 4].value.args[0].unique_id
    assert [_shape(stmt.value.args[0]) for stmt in extracts[:4]] == [
        [rows, 256],
        [rows, 8],
        [256, 256],
        [8, 256],
    ]
    assert len(_named(after, "tile.tget_scale_addr")) == 4
    for i in range(4):
        a = extracts[i].var.type.memref
        b = extracts[i + 4].var.type.memref
        assert a.is_pinned_ and b.is_pinned_
        assert a.base_.unique_id != b.base_.unique_id


def test_head_and_stage2_loop_share_exact_slot_identities():
    after = _run(_program(with_loop=True), True)
    extracts = _named(after, "tile.extract")
    assert len(extracts) == 16
    assert len(_named(after, "tile.matmul_mx")) == 1
    assert len(_named(after, "tile.matmul_mx_acc")) == 3
    for i in range(8):
        assert extracts[i].var.type.memref.base_.unique_id == extracts[i + 8].var.type.memref.base_.unique_id
    loops = []

    class CollectLoops(ir.IRVisitor):
        def visit_for_stmt(self, loop):
            loops.append(loop)
            super().visit_for_stmt(loop)

    CollectLoops().visit_program(after)
    assert len(loops) == 1
    assert loops[0].kind == ir.ForKind.Pipeline
    assert loops[0].attrs["pipeline_stages"] == 2


def test_stage2_lowering_keeps_each_shared_slot_consumer_before_next_write():
    after = _run(_program(with_loop=True), True)
    with passes.PassContext([], enable_mx_l0_ping_pong=True):
        after = passes.lower_pipeline_loops()(after)
        after = passes.canonicalize_io_order()(after)
    # After clone expansion, the actual serialized statement order must never
    # write one pinned slot while an earlier MX still holds its SSA result.
    # Counting shapes alone would miss precisely this cross-stage overwrite.
    live = {}
    consumed = set()
    bindings = {stmt.var.unique_id: stmt.value for stmt in _named(after, "tile.tget_scale_addr")}
    for stmt in _assignments(after):
        call = stmt.value
        if not isinstance(call, ir.Call):
            continue
        if call.op == ir.get_op("tile.extract"):
            assert call.attrs.get("mx_l0_ping_pong") is True
            memref = stmt.var.type.memref
            key = memref.base_.unique_id
            if key in live:
                assert live[key] in consumed, "pipeline clone overwrote an unconsumed L0 slot"
            live[key] = stmt.var.unique_id
        elif call.op in (ir.get_op("tile.matmul_mx"), ir.get_op("tile.matmul_mx_acc")):
            first = 1 if call.op == ir.get_op("tile.matmul_mx_acc") else 0
            for arg in call.args[first:]:
                memref = arg.type.memref
                # tget_scale_addr results acquire their source MemRef only in
                # InitMemRef, so use binding provenance for scales below.
                if memref is not None:
                    assert live[memref.base_.unique_id] == arg.unique_id
                    consumed.add(arg.unique_id)
            # Scale extraction's returned value is rebound in place. Its
            # binding args identify the same live payload, not a second copy.
            for arg in call.args[first + 1 :: 2]:
                binding = bindings[arg.unique_id]
                consumed.add(binding.args[0].unique_id)
    assert len(_mx_ops(after)) == 8  # head2 + two main clones*2 + one peeled tail*2


def test_slot_ownership_and_scheduling_marker_survive_ir_roundtrip():
    before = _run(_program(rows=48, with_loop=True), True)
    after = pl.parse_program(ir.python_print(before))
    extracts = _named(after, "tile.extract")
    assert len(extracts) == 16
    assert all(stmt.value.attrs.get("mx_l0_ping_pong") is True for stmt in extracts)
    for i in range(8):
        assert extracts[i].var.type.memref.base_.unique_id == extracts[i + 8].var.type.memref.base_.unique_id
    assert extracts[0].var.type.memref.base_.unique_id != extracts[4].var.type.memref.base_.unique_id


@pytest.mark.parametrize("rows", [48, 64])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_full_mixed_pipeline_keeps_natural_a_and_32k_b_halves(rows, planner):
    after = _run_to_allocate(_mixed_program(rows), planner)
    assert any(func.func_type == pl.FunctionType.AIC for func in after.functions.values())
    assert any(func.func_type == pl.FunctionType.AIV for func in after.functions.values())
    extracts = [
        stmt
        for stmt in _named(after, "tile.extract")
        if stmt.var.type.memory_space in (pl.Mem.Left, pl.Mem.LeftScale, pl.Mem.Right, pl.Mem.RightScale)
    ]
    assert len(extracts) == 8
    assert len(_mx_ops(after)) == 2
    assert not _named(after, "tile.create")
    expected_sizes = {
        pl.Mem.Left: rows * 128,
        pl.Mem.LeftScale: rows * 4,
        pl.Mem.Right: 32768,
        pl.Mem.RightScale: 1024,
    }
    by_space = {}
    for stmt in extracts:
        space = stmt.var.type.memory_space
        memref = stmt.var.type.memref
        assert memref.size_ == expected_sizes[space]
        assert isinstance(memref.byte_offset_, ir.ConstInt)
        by_space.setdefault(space, []).append((memref.byte_offset_.value, memref.size_))
    for regions in by_space.values():
        assert len(regions) == 2
        regions.sort()
        assert regions[0][0] + regions[0][1] <= regions[1][0]
    assert {addr for addr, _size in by_space[pl.Mem.Right]} == {0, 32768}
    # PTOAS binds the MX scale address from the actual data handle, rather than
    # the provisional scale allocation. Natural A slots therefore need their
    # own data>>4 check; only B's second scale slot is necessarily at 2048.
    for data_space, scale_size in [(pl.Mem.Left, rows * 4), (pl.Mem.Right, 1024)]:
        scale_ranges = sorted((addr >> 4, scale_size) for addr, _size in by_space[data_space])
        assert scale_ranges[0][0] + scale_size <= scale_ranges[1][0]
        assert scale_ranges[1][0] + scale_size <= 4096


@pytest.mark.parametrize("rows", [48, 64])
def test_ptoas_planner_preserves_original_lowering(rows):
    before = _program(rows=rows, with_loop=True)
    after = _run(before, True, passes.MemoryPlanner.PTOAS)
    ir.assert_structural_equal(_run(before, False, passes.MemoryPlanner.PTOAS), after)
    assert not _named(after, "tile.extract")


@pytest.mark.parametrize("rows,k,n", [(32, 256, 256), (64, 128, 256), (64, 256, 128)])
def test_unproven_geometry_preserves_existing_lowering(rows, k, n):
    before = _program(rows=rows, k=k, n=n)
    ir.assert_structural_equal(_run(before, False), _run(before, True))


def test_default_flag_preserves_existing_k256_lowering():
    after = _run(_program(), False)
    assert len(_named(after, "tile.extract")) == 0
    assert len(_named(after, "tile.matmul_mx")) == 1
    assert len(_named(after, "tile.tget_scale_addr")) == 2


def test_shared_explicit_l0_snapshot_keeps_original_staging():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 256], pl.FP8E4M3FN],
            a_s: pl.Tensor[[64, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[256, 256], pl.FP8E4M3FN],
            b_s: pl.Tensor[[8, 256], pl.FP8E8M0, pl.MX_B_NN],
            out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
        ) -> pl.Tensor[[64, 256], pl.FP32]:
            ta = pl.load(a, [0, 0], [64, 256], target_memory=pl.Mem.Mat)
            tas = pl.load(a_s, [0, 0], [64, 8], target_memory=pl.Mem.Mat)
            tb = pl.load(b, [0, 0], [256, 256], target_memory=pl.Mem.Mat)
            tbs = pl.load(b_s, [0, 0], [8, 256], target_memory=pl.Mem.Mat)
            # This explicit snapshot has two consumers, so replacing its one
            # copy with deferred reads at each matmul would change its lifetime.
            left = pl.move(ta, pl.Mem.Left)
            left_scale = pl.move(tas, pl.Mem.LeftScale)
            right = pl.move(tb, pl.Mem.Right)
            right_scale = pl.move(tbs, pl.Mem.RightScale)
            c = pl.matmul_mx(left, left_scale, right, right_scale)
            c = pl.matmul_mx_acc(c, left, left_scale, right, right_scale)
            return pl.store(c, [0, 0], out)

    ir.assert_structural_equal(_run(Before, False), _run(Before, True))


def test_unrelated_l0_consumer_disables_whole_function_reservation():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 256], pl.FP8E4M3FN],
            a_s: pl.Tensor[[64, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[256, 256], pl.FP8E4M3FN],
            b_s: pl.Tensor[[8, 256], pl.FP8E8M0, pl.MX_B_NN],
            dense_a: pl.Tensor[[16, 64], pl.FP16],
            dense_b: pl.Tensor[[64, 32], pl.FP16],
            dense_out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
        ) -> pl.Tensor[[64, 256], pl.FP32]:
            ta = pl.load(a, [0, 0], [64, 256], target_memory=pl.Mem.Mat)
            tas = pl.load(a_s, [0, 0], [64, 8], target_memory=pl.Mem.Mat)
            tb = pl.load(b, [0, 0], [256, 256], target_memory=pl.Mem.Mat)
            tbs = pl.load(b_s, [0, 0], [8, 256], target_memory=pl.Mem.Mat)
            c = pl.matmul_mx(ta, tas, tb, tbs)
            da = pl.load(dense_a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
            db = pl.load(dense_b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
            dc = pl.matmul(da, db)
            dense_out = pl.store(dc, [0, 0], dense_out)
            return pl.store(c, [0, 0], out)

    ir.assert_structural_equal(_run(Before, False), _run(Before, True))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
