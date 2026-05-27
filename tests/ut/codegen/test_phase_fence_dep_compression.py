# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Focused codegen tests for manual_scope phase-fence dependency compression."""

import re

import pypto.language as pl
import pytest
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir


def _generate_orch_code(program) -> str:
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func).code
    raise ValueError("No orchestration function found in program")


def _compile_program(program_cls) -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed = pm.run_passes(program_cls)
    return _generate_orch_code(transformed)


def _assert_single_barrier_shape(code: str, *, fanin: int) -> None:
    assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
    assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{fanin}];" in code, code
    real_dep_arrays = re.findall(r"PTO2TaskId (params_t\d+)_deps\[1\];", code)
    assert real_dep_arrays, code
    assert any(
        (
            f"if (phase_fence_barrier_0_tid.is_valid()) "
            f"{task_var}_deps[{task_var}_deps_count++] = phase_fence_barrier_0_tid;"
        )
        in code
        for task_var in real_dep_arrays
    ), code
    assert not re.search(rf"PTO2TaskId params_t\d+_deps\[{fanin}\];", code), code


def _assert_ordered(code: str, *needles: str) -> None:
    positions = [code.find(needle) for needle in needles]
    assert all(pos >= 0 for pos in positions), code
    assert positions == sorted(positions), code


class TestPhaseFenceDepCompressionCodegen:
    @pytest.fixture(autouse=True)
    def _no_roundtrip_verification(self):
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_standard_submit_phase_fence(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase, (tids_iter,) in pl.range(3, init_values=(tids,)):
                        tids_next = pl.array.create(branches, pl.TASK_ID)
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        for branch in pl.parallel(branches):
                            col: pl.Scalar[pl.INDEX] = branch * tile_c
                            out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids_iter])
                            tids_next[branch] = tid
                        tids = pl.yield_(tids_next)
                return out

        code = _compile_program(Prog)
        _assert_single_barrier_shape(code, fanin=branches)

    def test_standard_pl_at_phase_fence(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase, (tids_iter, out_iter) in pl.range(3, init_values=(tids, out)):
                        tids_next = pl.array.create(branches, pl.TASK_ID)
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        for branch, (out_branch, tids_next_iter) in pl.parallel(
                            branches, init_values=(out_iter, tids_next)
                        ):
                            col: pl.Scalar[pl.INDEX] = branch * tile_c
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase_tile", deps=[tids_iter]) as tid:
                                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(
                                    x, [row, col], [tile_r, tile_c]
                                )
                                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                                out_next = pl.store(r, [row, col], out_branch)
                            tids_next_out = pl.array.update_element(tids_next_iter, branch, tid)
                            out_branch_out, tids_branch_out = pl.yield_(out_next, tids_next_out)
                        tids, out = pl.yield_(tids_branch_out, out_branch_out)
                return out

        code = _compile_program(Prog)
        _assert_single_barrier_shape(code, fanin=branches)

    def test_three_level_flattened_phase_fence(self):
        rows, cols = 384, 128
        tile_r, tile_c = 16, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for epoch, (tids_epoch,) in pl.range(2, init_values=(tids,)):
                        for phase, (tids_iter,) in pl.range(3, init_values=(tids_epoch,)):
                            tids_next = pl.array.create(branches, pl.TASK_ID)
                            base: pl.Scalar[pl.INDEX] = (epoch * 3 + phase) * branches
                            for branch in pl.parallel(branches):
                                row: pl.Scalar[pl.INDEX] = (base + branch) * tile_r
                                col: pl.Scalar[pl.INDEX] = branch * tile_c
                                out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids_iter])
                                tids_next[branch] = tid
                            tids_phase = pl.yield_(tids_next)
                        tids = pl.yield_(tids_phase)
                return out

        code = _compile_program(Prog)
        _assert_single_barrier_shape(code, fanin=branches)
        assert code.count("rt_submit_dummy_task(params_phase_fence_barrier_") == 1, code

    def test_sibling_producer_consumer_loops_emit_invariant_phase_fence(self):
        rows, cols = 256, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 1.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def k2(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 2.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for producer_branch in pl.parallel(branches):
                        col: pl.Scalar[pl.INDEX] = producer_branch * tile_c
                        out, tid = pl.submit(self.k1, x, out, 0, col)
                        tids[producer_branch] = tid
                    for consumer_branch in pl.parallel(branches):
                        col: pl.Scalar[pl.INDEX] = consumer_branch * tile_c
                        out, _ = pl.submit(self.k2, x, out, tile_r, col, deps=[tids])
                return out

        code = _compile_program(Prog)
        _assert_single_barrier_shape(code, fanin=branches)
        _assert_ordered(
            code,
            "for (int64_t producer_branch =",
            "rt_submit_dummy_task(params_phase_fence_barrier_0)",
            "for (int64_t consumer_branch =",
        )

    def test_multiloop_chain_compresses_only_stable_segments(self):
        rows, cols = 640, 128
        tile_r, tile_c = 32, 32
        branches = 4
        consumers = 4
        range_consumers = 2
        b_layers = 2

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 1.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def k2(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 2.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def k3(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 3.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    tids2 = pl.array.create(b_layers * consumers, pl.TASK_ID)
                    for r1 in pl.range(2):
                        for p1 in pl.parallel(branches):
                            row: pl.Scalar[pl.INDEX] = (r1 * branches + p1) * tile_r
                            col: pl.Scalar[pl.INDEX] = p1 * tile_c
                            out, tid = pl.submit(self.k1, x, out, row, col, deps=[tids])
                            tids[p1] = tid
                    for r2 in pl.range(b_layers):
                        for p2 in pl.parallel(consumers):
                            row: pl.Scalar[pl.INDEX] = (2 * branches + r2 * consumers + p2) * tile_r
                            col: pl.Scalar[pl.INDEX] = p2 * tile_c
                            out, tid2 = pl.submit(self.k2, x, out, row, col, deps=[tids])
                            tids2[r2 * consumers + p2] = tid2
                    for r3 in pl.range(2):
                        for p3 in pl.range(range_consumers):
                            row: pl.Scalar[pl.INDEX] = (
                                2 * branches + b_layers * consumers + r3 * range_consumers + p3
                            ) * tile_r
                            col: pl.Scalar[pl.INDEX] = p3 * tile_c
                            out, _ = pl.submit(self.k3, x, out, row, col, deps=[tids2])
                return out

        code = _compile_program(Prog)
        assert code.count("rt_submit_dummy_task(params_phase_fence_barrier_") == 1, code
        assert "PTO2TaskId params_phase_fence_barrier_0_deps[4];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[8\];", code), code
        _assert_ordered(
            code,
            "for (int64_t r1 =",
            "for (int64_t p1 =",
            "for (int64_t r2 =",
            "rt_submit_dummy_task(params_phase_fence_barrier_0)",
            "for (int64_t r3 =",
        )

    def test_dense_mixed_phase_graph_miniature(self):
        rows, cols = 736, 96
        tile_r, tile_c = 32, 32
        branches = 3

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 1.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids_a = pl.array.create(branches, pl.TASK_ID)
                    tids_b = pl.array.create(branches, pl.TASK_ID)
                    for group in pl.parallel(2):
                        tids_local = pl.array.create(branches, pl.TASK_ID)
                        group_base: pl.Scalar[pl.INDEX] = group * branches
                        for lane in pl.parallel(branches):
                            row_local: pl.Scalar[pl.INDEX] = (group_base + lane) * tile_r
                            col_local: pl.Scalar[pl.INDEX] = lane * tile_c
                            out, tid_local = pl.submit(
                                self.kern, x, out, row_local, col_local, deps=[tids_local]
                            )
                            tids_local[lane] = tid_local
                    for phase in pl.range(2):
                        for p in pl.parallel(branches):
                            row_a: pl.Scalar[pl.INDEX] = (6 + phase * 2 * branches + p) * tile_r
                            col: pl.Scalar[pl.INDEX] = p * tile_c
                            out, tid_a = pl.submit(self.kern, x, out, row_a, col, deps=[tids_a])
                            tids_a[p] = tid_a

                            row_b: pl.Scalar[pl.INDEX] = (6 + phase * 2 * branches + branches + p) * tile_r
                            out, tid_b = pl.submit(self.kern, x, out, row_b, col, deps=[tids_b])
                            tids_b[p] = tid_b

                    for p2 in pl.parallel(branches):
                        row_cross: pl.Scalar[pl.INDEX] = (18 + p2) * tile_r
                        col_cross: pl.Scalar[pl.INDEX] = p2 * tile_c
                        out, _ = pl.submit(self.kern, x, out, row_cross, col_cross, deps=[tids_a])

                    prev = tids_a[0]
                    row_scalar: pl.Scalar[pl.INDEX] = 21 * tile_r
                    row_fanin: pl.Scalar[pl.INDEX] = 22 * tile_r
                    out, _ = pl.submit(self.kern, x, out, row_scalar, 0, deps=[prev])
                    out, _ = pl.submit(self.kern, x, out, row_fanin, 0, deps=[tids_b])
                return out

        code = _compile_program(Prog)
        assert code.count("rt_submit_dummy_task(params_phase_fence_barrier_") == 1, code
        assert "PTO2TaskId params_phase_fence_barrier_0_deps[3];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[3\];", code), code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        _assert_ordered(
            code,
            "for (int64_t group =",
            "for (int64_t lane =",
            "for (int64_t phase =",
            "for (int64_t p =",
            "rt_submit_dummy_task(params_phase_fence_barrier_0)",
            "for (int64_t p2 =",
        )

    def test_parallel_range_parallel_does_not_emit_outer_dummy_barrier(self):
        rows, cols = 256, 128
        tile_r, tile_c = 16, 32
        outer_branches = 2
        phases = 2
        inner_branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(inner_branches, pl.TASK_ID)
                    for outer in pl.parallel(outer_branches):
                        for phase in pl.range(phases):
                            for inner in pl.parallel(inner_branches):
                                row: pl.Scalar[pl.INDEX] = (
                                    (outer * phases + phase) * inner_branches + inner
                                ) * tile_r
                                col: pl.Scalar[pl.INDEX] = inner * tile_c
                                out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                                tids[inner] = tid
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    def test_array_fanin_to_single_consumer_does_not_emit_dummy_barrier(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for branch in pl.parallel(branches):
                        col: pl.Scalar[pl.INDEX] = branch * tile_c
                        out, tid = pl.submit(self.kern, x, out, 0, col)
                        tids[branch] = tid
                    out, _ = pl.submit(self.kern, x, out, tile_r, 0, deps=[tids])
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    def test_two_by_two_low_benefit_phase_fence_does_not_emit_dummy_barrier(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 2

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        for branch in pl.parallel(branches):
                            col: pl.Scalar[pl.INDEX] = branch * tile_c
                            out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                            tids[branch] = tid
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[2\];", code), code

    def test_if_consumer_same_carrier_update_falls_back(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        for branch in pl.parallel(branches):
                            if branch >= 0:
                                col: pl.Scalar[pl.INDEX] = branch * tile_c
                                out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                                tids[branch] = tid
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    def test_two_same_carrier_arrays_fall_back_independently(self):
        rows, cols = 256, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern_a(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kern_b(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, 1.0)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids_a = pl.array.create(branches, pl.TASK_ID)
                    tids_b = pl.array.create(branches, pl.TASK_ID)
                    for phase in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        for branch in pl.parallel(branches):
                            col: pl.Scalar[pl.INDEX] = branch * tile_c
                            out, tid_a = pl.submit(self.kern_a, x, out, row, col, deps=[tids_a])
                            tids_a[branch] = tid_a
                            out, tid_b = pl.submit(self.kern_b, x, out, row, col, deps=[tids_b])
                            tids_b[branch] = tid_b
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert len(re.findall(r"PTO2TaskId params_t\d+_deps\[4\];", code)) >= 2, code

    def test_reset_per_outer_same_carrier_loop_falls_back_inside_batch(self):
        rows, cols = 256, 128
        tile_r, tile_c = 16, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    for batch in pl.range(2):
                        tids = pl.array.create(branches, pl.TASK_ID)
                        for phase in pl.range(2):
                            base: pl.Scalar[pl.INDEX] = (batch * 2 + phase) * branches
                            for branch in pl.parallel(branches):
                                row: pl.Scalar[pl.INDEX] = (base + branch) * tile_r
                                col: pl.Scalar[pl.INDEX] = branch * tile_c
                                out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                                tids[branch] = tid
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    @pytest.mark.parametrize(
        "case_name",
        ["scalar", "mixed_array_scalar", "two_arrays_same_call", "auto_scope"],
    )
    def test_fallback_matrix_does_not_emit_dummy_barrier(self, case_name: str):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        def make_program(case_name: str):
            if case_name == "scalar":

                @pl.program
                class Prog:
                    @pl.function(type=pl.FunctionType.InCore)
                    def kern(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                        row: pl.Scalar[pl.INDEX],
                        col: pl.Scalar[pl.INDEX],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                        r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                        ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                        return ret

                    @pl.function(type=pl.FunctionType.Orchestration)
                    def main(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        with pl.manual_scope():
                            out, tid = pl.submit(self.kern, x, out, 0, 0)
                            out, _ = pl.submit(self.kern, x, out, tile_r, 0, deps=[tid])
                        return out

                return Prog

            if case_name == "mixed_array_scalar":

                @pl.program
                class Prog:
                    @pl.function(type=pl.FunctionType.InCore)
                    def kern(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                        row: pl.Scalar[pl.INDEX],
                        col: pl.Scalar[pl.INDEX],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                        r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                        ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                        return ret

                    @pl.function(type=pl.FunctionType.Orchestration)
                    def main(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        with pl.manual_scope():
                            out, seed_tid = pl.submit(self.kern, x, out, 0, 0)
                            tids = pl.array.create(branches, pl.TASK_ID)
                            for branch in pl.parallel(branches):
                                col: pl.Scalar[pl.INDEX] = branch * tile_c
                                out, tid = pl.submit(self.kern, x, out, tile_r, col, deps=[tids, seed_tid])
                                tids[branch] = tid
                        return out

                return Prog

            if case_name == "two_arrays_same_call":

                @pl.program
                class Prog:
                    @pl.function(type=pl.FunctionType.InCore)
                    def kern(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                        row: pl.Scalar[pl.INDEX],
                        col: pl.Scalar[pl.INDEX],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                        r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                        ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                        return ret

                    @pl.function(type=pl.FunctionType.Orchestration)
                    def main(
                        self,
                        x: pl.Tensor[[rows, cols], pl.FP32],
                        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                    ) -> pl.Tensor[[rows, cols], pl.FP32]:
                        with pl.manual_scope():
                            tids_a = pl.array.create(branches, pl.TASK_ID)
                            tids_b = pl.array.create(branches, pl.TASK_ID)
                            for branch in pl.parallel(branches):
                                col: pl.Scalar[pl.INDEX] = branch * tile_c
                                out, tid = pl.submit(self.kern, x, out, tile_r, col, deps=[tids_a, tids_b])
                                tids_a[branch] = tid
                                tids_b[branch] = tid
                        return out

                return Prog

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def kern(
                    self,
                    x: pl.Tensor[[rows, cols], pl.FP32],
                    out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                    row: pl.Scalar[pl.INDEX],
                    col: pl.Scalar[pl.INDEX],
                ) -> pl.Tensor[[rows, cols], pl.FP32]:
                    t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                    r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                    ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                    return ret

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    x: pl.Tensor[[rows, cols], pl.FP32],
                    out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                ) -> pl.Tensor[[rows, cols], pl.FP32]:
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for branch in pl.parallel(branches):
                        col: pl.Scalar[pl.INDEX] = branch * tile_c
                        out, tid = pl.submit(self.kern, x, out, tile_r, col, deps=[tids])
                        tids[branch] = tid
                    return out

            return Prog

        code = _compile_program(make_program(case_name))
        assert "rt_submit_dummy_task" not in code, code

    def test_partial_slot_dep_is_scalar_fallback(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for branch in pl.parallel(branches):
                        col: pl.Scalar[pl.INDEX] = branch * tile_c
                        prev = tids[branch]
                        out, tid = pl.submit(self.kern, x, out, tile_r, col, deps=[prev])
                        tids[branch] = tid
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert re.search(
            r"PTO2TaskId (params_t\d+)_deps\[1\];\s*"
            r"uint32_t \1_deps_count = 0;\s*"
            r"if \(prev\.is_valid\(\)\) \1_deps\[\1_deps_count\+\+\] = prev;\s*"
            r"\1\.set_dependencies\(\1_deps, \1_deps_count\);",
            code,
        ), code

    def test_updated_array_dep_in_same_parallel_body_falls_back(self):
        rows, cols = 128, 128
        tile_r, tile_c = 32, 32
        branches = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                t: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.load(x, [row, col], [tile_r, tile_c])
                r: pl.Tile[[tile_r, tile_c], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[rows, cols], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[rows, cols], pl.FP32],
                out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = phase * tile_r
                        next_row: pl.Scalar[pl.INDEX] = row + tile_r
                        for branch in pl.parallel(branches):
                            col: pl.Scalar[pl.INDEX] = branch * tile_c
                            out, tid_a = pl.submit(self.kern, x, out, row, col, deps=[tids])
                            tids[branch] = tid_a
                            out, tid_b = pl.submit(self.kern, x, out, next_row, col, deps=[tids])
                            tids[branch] = tid_b
                return out

        code = _compile_program(Prog)
        assert "rt_submit_dummy_task" not in code, code
        assert len(re.findall(r"PTO2TaskId params_t\d+_deps\[4\];", code)) >= 2, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
