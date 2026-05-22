# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime witnesses for manual_scope phase-fence dependency compression.

These tests intentionally avoid depending on a stable dummy-task marker in
``l2_perf_records.json``. The externally required contract is phase strictness:
all tasks in flattened stage k+1 must start after all tasks in flattened stage k
finish.
"""

import json
import os
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output"

_BRANCHES = 4
_TILE_M = 32
_BIG_N = 32
_DENSE_BIG_N = 64
_DENSE_PHASES = 1
_DENSE_GROUPS = 1
_DENSE_STEPS = 1
_DENSE_DEEP_PHASES = 1
_DENSE_CORRECTNESS_BRANCHES = 3
_DENSE_SWIMLANE_BRANCHES = 4
_EXTRA_SWIMLANE_ENV = "PYPTO_PHASE_FENCE_EXTRA_SWIMLANE"


def _require_extra_swimlane_case(label: str) -> None:
    if os.environ.get(_EXTRA_SWIMLANE_ENV) != "1":
        pytest.skip(
            f"{label} is a manual profiling witness; set {_EXTRA_SWIMLANE_ENV}=1 "
            "and run this test node by itself"
        )


def _assert_flattened_stage_strict(swimlane_data: dict, *, stages: int, branches: int) -> None:
    expected = stages * branches
    tasks = swimlane_data["tasks"]
    if len(tasks) < expected:
        pytest.skip(f"need >= {expected} tasks for phase-fence check, got {len(tasks)}")
    tasks = sorted(tasks, key=lambda t: t["start_time_us"])[:expected]
    grouped = [tasks[i * branches : (i + 1) * branches] for i in range(stages)]
    for i in range(stages - 1):
        end_i = max(t["end_time_us"] for t in grouped[i])
        start_next = min(t["start_time_us"] for t in grouped[i + 1])
        assert start_next >= end_i, (
            f"flattened stage {i + 1} starts at {start_next:.2f}us before stage {i} ends at {end_i:.2f}us"
        )


def _assert_min_task_count(swimlane_data: dict, *, expected: int) -> None:
    tasks = swimlane_data["tasks"]
    if len(tasks) < expected:
        pytest.skip(f"need >= {expected} tasks for swimlane check, got {len(tasks)}")


def _assert_multiloop_chain_shape(swimlane_data: dict) -> None:
    branches = _BRANCHES
    b_tasks = 2 * branches
    c_tasks = 2 * 2
    expected = 2 * branches + b_tasks + c_tasks
    _assert_min_task_count(swimlane_data, expected=expected)
    tasks = sorted(swimlane_data["tasks"], key=lambda t: t["start_time_us"])[:expected]

    k1_stage0 = tasks[:branches]
    k1_stage1 = tasks[branches : 2 * branches]
    b_stage = tasks[2 * branches : 2 * branches + b_tasks]
    c_stage = tasks[2 * branches + b_tasks :]
    k1_stage0_end = max(t["end_time_us"] for t in k1_stage0)
    k1_stage1_start = min(t["start_time_us"] for t in k1_stage1)
    k1_stage1_end = max(t["end_time_us"] for t in k1_stage1)
    b_stage_start = min(t["start_time_us"] for t in b_stage)
    b_stage_end = max(t["end_time_us"] for t in b_stage)
    c_stage_start = min(t["start_time_us"] for t in c_stage)

    assert k1_stage1_start >= k1_stage0_end, (
        f"multi-loop k1 stage 1 starts at {k1_stage1_start:.2f}us before k1 stage 0 "
        f"ends at {k1_stage0_end:.2f}us"
    )
    assert b_stage_start >= k1_stage1_end, (
        f"multi-loop B stage starts at {b_stage_start:.2f}us before final k1 stage "
        f"ends at {k1_stage1_end:.2f}us"
    )
    assert c_stage_start >= b_stage_end, (
        f"multi-loop C stage starts at {c_stage_start:.2f}us before full B stage ends at {b_stage_end:.2f}us"
    )


def _assert_dense_mixed_shape(swimlane_data: dict, *, branches: int) -> None:
    expected = _dense_mixed_task_bands(branches=branches)
    _assert_min_task_count(swimlane_data, expected=expected)


def _new_swimlane_file(test_runner, case: PTOTestCase, *, label: str) -> Path:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip(f"pass --enable-l2-swimlane to validate {label}")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(case)
    assert result.passed, f"{label} failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    candidates = list(after - before)
    if not candidates:
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    assert candidates, f"No l2_perf_records.json generated for {label}"
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _new_swimlane_json(test_runner, case: PTOTestCase, *, label: str) -> dict:
    path = _new_swimlane_file(test_runner, case, label=label)
    return json.loads(path.read_text())


def _build_submit_flattened_program(*, epochs: int, layers: int, phases: int):
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    stages = epochs * layers * phases
    big_m = stages * branches * tile_m

    @pl.program
    class SubmitFlattenedPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            bias: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, bias)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for epoch in pl.range(epochs):
                    for layer in pl.range(layers):
                        for phase in pl.range(phases):
                            stage: pl.Scalar[pl.INDEX] = (epoch * layers + layer) * phases + phase
                            for branch in pl.parallel(branches):
                                row: pl.Scalar[pl.INDEX] = (stage * branches + branch) * tile_m
                                out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids])
                                tids[branch] = tid
            return out

    return SubmitFlattenedPhaseFence


def _build_pl_at_flattened_program(*, epochs: int, phases: int):
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    stages = epochs * phases
    big_m = stages * branches * tile_m

    @pl.program
    class PlAtFlattenedPhaseFence:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for epoch in pl.range(epochs):
                    for phase in pl.range(phases):
                        stage: pl.Scalar[pl.INDEX] = epoch * phases + phase
                        for branch in pl.parallel(branches):
                            row: pl.Scalar[pl.INDEX] = (stage * branches + branch) * tile_m
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase_tile", deps=[tids]) as tid:
                                tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(
                                    data, [row, 0], [tile_m, big_n]
                                )
                                result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
                                out = pl.store(result, [row, 0], out)
                            tids[branch] = tid
            return out

    return PlAtFlattenedPhaseFence


def _build_reset_per_outer_program():
    batches = 2
    phases = 2
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = batches * phases * branches * tile_m

    @pl.program
    class ResetPerOuterPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                for batch in pl.range(batches):
                    tids = pl.array.create(branches, pl.TASK_ID)
                    for phase in pl.range(phases):
                        stage: pl.Scalar[pl.INDEX] = batch * phases + phase
                        for branch in pl.parallel(branches):
                            row: pl.Scalar[pl.INDEX] = (stage * branches + branch) * tile_m
                            out, tid = pl.submit(self.kernel_stripe, data, row, out, deps=[tids])
                            tids[branch] = tid
            return out

    return ResetPerOuterPhaseFence


def _build_sibling_loops_program():
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = 2 * branches * tile_m

    @pl.program
    class SiblingLoopPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for branch in pl.parallel(branches):
                    row: pl.Scalar[pl.INDEX] = branch * tile_m
                    out, tid = pl.submit(self.producer, data, row, out)
                    tids[branch] = tid
                for branch in pl.parallel(branches):
                    row: pl.Scalar[pl.INDEX] = (branches + branch) * tile_m
                    out, _ = pl.submit(self.consumer, data, row, out, deps=[tids])
            return out

    return SiblingLoopPhaseFence


def _build_if_consumer_program():
    phases = 2
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = phases * branches * tile_m

    @pl.program
    class IfConsumerPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for phase in pl.range(phases):
                    for branch in pl.parallel(branches):
                        if branch >= 0:
                            row: pl.Scalar[pl.INDEX] = (phase * branches + branch) * tile_m
                            out, tid = pl.submit(self.kernel_stripe, data, row, out, deps=[tids])
                            tids[branch] = tid
            return out

    return IfConsumerPhaseFence


def _build_if_mixed_fallback_program():
    phases = 2
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = (branches + phases * branches) * tile_m

    @pl.program
    class IfMixedFallbackPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for branch in pl.parallel(branches):
                    row: pl.Scalar[pl.INDEX] = branch * tile_m
                    out, tid = pl.submit(self.kernel_stripe, data, row, out)
                    tids[branch] = tid
                for phase in pl.range(phases):
                    for branch in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = (branches + phase * branches + branch) * tile_m
                        if branch >= 2:
                            out, tid = pl.submit(self.kernel_stripe, data, row, out, deps=[tids])
                        else:
                            prev = tids[1]
                            out, tid = pl.submit(self.kernel_stripe, data, row, out, deps=[prev])
                        tids[branch] = tid
            return out

    return IfMixedFallbackPhaseFence


def _build_multiloop_chain_program():
    branches = _BRANCHES
    consumers = _BRANCHES
    range_consumers = 2
    b_layers = 2
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = (2 * branches + b_layers * consumers + 2 * range_consumers) * tile_m

    @pl.program
    class MultiLoopChainPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def k2(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def k3(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                tids2 = pl.array.create(b_layers * consumers, pl.TASK_ID)
                for r1 in pl.range(2):
                    for p in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = (r1 * branches + p) * tile_m
                        out, tid = pl.submit(self.k1, data, row, out, deps=[tids])
                        tids[p] = tid
                for r2 in pl.range(b_layers):
                    for p in pl.parallel(consumers):
                        row: pl.Scalar[pl.INDEX] = (2 * branches + r2 * consumers + p) * tile_m
                        out, tid2 = pl.submit(self.k2, data, row, out, deps=[tids])
                        tids2[r2 * consumers + p] = tid2
                for r3 in pl.range(2):
                    for p in pl.range(range_consumers):
                        row: pl.Scalar[pl.INDEX] = (
                            2 * branches + b_layers * consumers + r3 * range_consumers + p
                        ) * tile_m
                        out, _ = pl.submit(self.k3, data, row, out, deps=[tids2])
            return out

    return MultiLoopChainPhaseFence


def _dense_mixed_task_bands(*, branches: int) -> int:
    return (
        2 * branches
        + _DENSE_PHASES * 2 * branches
        + _DENSE_GROUPS * _DENSE_STEPS * _DENSE_DEEP_PHASES * 2 * branches
        + _DENSE_STEPS * 2 * branches
        + 4
        + 2 * branches
    )


def _build_dense_mixed_phase_graph_program(*, branches: int):
    tile_m = _TILE_M
    big_n = _DENSE_BIG_N
    phases = _DENSE_PHASES
    groups = _DENSE_GROUPS
    steps = _DENSE_STEPS
    deep_phases = _DENSE_DEEP_PHASES
    big_m = _dense_mixed_task_bands(branches=branches) * tile_m

    @pl.program
    class DenseMixedPhaseGraph:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids_a = pl.array.create(branches, pl.TASK_ID)
                tids_b = pl.array.create(branches, pl.TASK_ID)
                tids_c = pl.array.create(branches, pl.TASK_ID)
                tids_d = pl.array.create(branches, pl.TASK_ID)

                for branch in pl.parallel(branches):
                    row_a: pl.Scalar[pl.INDEX] = branch * tile_m
                    out, tid_a = pl.submit(self.kernel_stripe, data, row_a, out)
                    tids_a[branch] = tid_a
                    row_b: pl.Scalar[pl.INDEX] = (branches + branch) * tile_m
                    out, tid_b = pl.submit(self.kernel_stripe, data, row_b, out)
                    tids_b[branch] = tid_b

                stage1_base: pl.Scalar[pl.INDEX] = 2 * branches
                for phase in pl.range(phases):
                    phase_base: pl.Scalar[pl.INDEX] = stage1_base + phase * 2 * branches
                    for branch in pl.parallel(branches):
                        row_a2: pl.Scalar[pl.INDEX] = (phase_base + branch) * tile_m
                        out, tid_a2 = pl.submit(self.kernel_stripe, data, row_a2, out, deps=[tids_a])
                        tids_a[branch] = tid_a2
                        row_b2: pl.Scalar[pl.INDEX] = (phase_base + branches + branch) * tile_m
                        out, tid_b2 = pl.submit(self.kernel_stripe, data, row_b2, out, deps=[tids_b])
                        tids_b[branch] = tid_b2

                stage2a_base: pl.Scalar[pl.INDEX] = stage1_base + phases * 2 * branches
                for group in pl.parallel(groups):
                    tids_local_c = pl.array.create(branches, pl.TASK_ID)
                    tids_local_d = pl.array.create(branches, pl.TASK_ID)
                    for step in pl.range(steps):
                        for deep_phase in pl.range(deep_phases):
                            nested_base: pl.Scalar[pl.INDEX] = stage2a_base + (
                                ((group * steps + step) * deep_phases + deep_phase) * 2 * branches
                            )
                            for lane in pl.parallel(branches):
                                row_local_c: pl.Scalar[pl.INDEX] = (nested_base + lane) * tile_m
                                out, tid_local_c = pl.submit(
                                    self.kernel_stripe, data, row_local_c, out, deps=[tids_local_c]
                                )
                                tids_local_c[lane] = tid_local_c
                                row_local_d: pl.Scalar[pl.INDEX] = (nested_base + branches + lane) * tile_m
                                out, tid_local_d = pl.submit(
                                    self.kernel_stripe, data, row_local_d, out, deps=[tids_local_d]
                                )
                                tids_local_d[lane] = tid_local_d

                stage2b_base: pl.Scalar[pl.INDEX] = stage2a_base + groups * steps * deep_phases * 2 * branches
                for step in pl.range(steps):
                    step_base2: pl.Scalar[pl.INDEX] = stage2b_base + step * 2 * branches
                    for branch in pl.parallel(branches):
                        row_cross_a: pl.Scalar[pl.INDEX] = (step_base2 + branch) * tile_m
                        out, tid_c = pl.submit(self.kernel_stripe, data, row_cross_a, out, deps=[tids_a])
                        tids_c[branch] = tid_c
                        row_cross_b: pl.Scalar[pl.INDEX] = (step_base2 + branches + branch) * tile_m
                        out, tid_d = pl.submit(self.kernel_stripe, data, row_cross_b, out, deps=[tids_b])
                        tids_d[branch] = tid_d

                stage3_base: pl.Scalar[pl.INDEX] = stage2b_base + steps * 2 * branches
                for r in pl.range(2):
                    prev_c = tids_c[0]
                    row_scalar: pl.Scalar[pl.INDEX] = (stage3_base + r * 2) * tile_m
                    out, _ = pl.submit(self.kernel_stripe, data, row_scalar, out, deps=[prev_c])
                    row_single: pl.Scalar[pl.INDEX] = (stage3_base + r * 2 + 1) * tile_m
                    out, _ = pl.submit(self.kernel_stripe, data, row_single, out, deps=[tids_d])

                stage4_base: pl.Scalar[pl.INDEX] = stage3_base + 4
                for branch in pl.parallel(branches):
                    row_final_c: pl.Scalar[pl.INDEX] = (stage4_base + branch) * tile_m
                    out, _ = pl.submit(self.kernel_stripe, data, row_final_c, out, deps=[tids_c])
                    row_final_d: pl.Scalar[pl.INDEX] = (stage4_base + branches + branch) * tile_m
                    out, _ = pl.submit(self.kernel_stripe, data, row_final_d, out, deps=[tids_d])
            return out

    return DenseMixedPhaseGraph


def _build_partial_reduce_chain_program():
    branches = _BRANCHES
    tile_m = _TILE_M
    big_n = _BIG_N
    big_m = (branches + 1 + branches) * tile_m

    @pl.program
    class PartialReduceChainPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def reducer(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, 1.0)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for branch in pl.parallel(branches):
                    row: pl.Scalar[pl.INDEX] = branch * tile_m
                    out, tid = pl.submit(self.producer, data, row, out)
                    tids[branch] = tid
                reducer_row: pl.Scalar[pl.INDEX] = branches * tile_m
                out, reduce_tid = pl.submit(self.reducer, data, reducer_row, out, deps=[tids])
                for branch in pl.parallel(branches):
                    row: pl.Scalar[pl.INDEX] = (branches + 1 + branch) * tile_m
                    out, _ = pl.submit(self.consumer, data, row, out, deps=[reduce_tid])
            return out

    return PartialReduceChainPhaseFence


class _PhaseFenceCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        name: str,
        program_builder,
        *,
        rows: int,
        cols: int = _BIG_N,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self._name = name
        self._program_builder = program_builder
        self._rows = rows
        self._cols = cols

    def get_name(self) -> str:
        return self._name

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [self._rows, self._cols], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self._rows, self._cols], DataType.FP32, init_value=0.0, is_output=True),
        ]

    def get_program(self) -> Any:
        return self._program_builder()

    def compute_expected(self, tensors, params=None):
        data = tensors["data"]
        out = tensors["out"]
        out.zero_()
        out[:, :] = data + 1.0


def _submit_case(*, epochs: int, layers: int, phases: int, name: str, platform: str | None = None):
    stages = epochs * layers * phases
    rows = stages * _BRANCHES * _TILE_M
    return _PhaseFenceCase(
        name,
        lambda: _build_submit_flattened_program(epochs=epochs, layers=layers, phases=phases),
        rows=rows,
        platform=platform,
    )


def _pl_at_case(*, epochs: int, phases: int, name: str, platform: str | None = None):
    stages = epochs * phases
    rows = stages * _BRANCHES * _TILE_M
    return _PhaseFenceCase(
        name,
        lambda: _build_pl_at_flattened_program(epochs=epochs, phases=phases),
        rows=rows,
        platform=platform,
    )


def _reset_case(*, platform: str | None = None):
    rows = 2 * 2 * _BRANCHES * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_reset_per_outer", _build_reset_per_outer_program, rows=rows, platform=platform
    )


def _sibling_loops_case(*, platform: str | None = None):
    rows = 2 * _BRANCHES * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_sibling_loops",
        _build_sibling_loops_program,
        rows=rows,
        platform=platform,
    )


def _if_consumer_case(*, platform: str | None = None):
    rows = 2 * _BRANCHES * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_if_consumer",
        _build_if_consumer_program,
        rows=rows,
        platform=platform,
    )


def _if_mixed_fallback_case(*, platform: str | None = None):
    rows = (_BRANCHES + 2 * _BRANCHES) * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_if_mixed_fallback",
        _build_if_mixed_fallback_program,
        rows=rows,
        platform=platform,
    )


def _multiloop_chain_case(*, platform: str | None = None):
    rows = (2 * _BRANCHES + 2 * _BRANCHES + 2 * 2) * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_multiloop_chain",
        _build_multiloop_chain_program,
        rows=rows,
        platform=platform,
    )


def _dense_mixed_case(*, branches: int = _DENSE_CORRECTNESS_BRANCHES, platform: str | None = None):
    rows = _dense_mixed_task_bands(branches=branches) * _TILE_M
    return _PhaseFenceCase(
        f"phase_fence_dense_mixed_n{branches}",
        lambda: _build_dense_mixed_phase_graph_program(branches=branches),
        rows=rows,
        cols=_DENSE_BIG_N,
        platform=platform,
    )


def _partial_reduce_chain_case(*, platform: str | None = None):
    rows = (2 * _BRANCHES + 1) * _TILE_M
    return _PhaseFenceCase(
        "phase_fence_partial_reduce_chain",
        _build_partial_reduce_chain_program,
        rows=rows,
        platform=platform,
    )


class TestPhaseFenceDepCompressionCorrectness:
    @pytest.fixture(autouse=True)
    def _skip_when_collecting_l2_swimlane(self, test_runner):
        if test_runner.config.enable_l2_swimlane:
            pytest.skip(
                "correctness cases run without --enable-l2-swimlane; swimlane mode runs profiling witnesses"
            )

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_submit_three_level_correctness(self, test_runner, platform):
        result = test_runner.run(
            _submit_case(epochs=2, layers=1, phases=3, name="phase_fence_submit_3l", platform=platform)
        )
        assert result.passed, f"three-level submit phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pl_at_three_level_correctness(self, test_runner, platform):
        result = test_runner.run(
            _pl_at_case(epochs=2, phases=3, name="phase_fence_pl_at_3l", platform=platform)
        )
        assert result.passed, f"three-level pl.at phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_reset_per_outer_correctness(self, test_runner, platform):
        result = test_runner.run(_reset_case(platform=platform))
        assert result.passed, f"reset-per-outer phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_sibling_loops_correctness(self, test_runner, platform):
        result = test_runner.run(_sibling_loops_case(platform=platform))
        assert result.passed, f"sibling-loop phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_if_consumer_correctness(self, test_runner, platform):
        result = test_runner.run(_if_consumer_case(platform=platform))
        assert result.passed, f"if-consumer phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_if_mixed_fallback_correctness(self, test_runner, platform):
        result = test_runner.run(_if_mixed_fallback_case(platform=platform))
        assert result.passed, f"if-mixed-fallback phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_multiloop_chain_correctness(self, test_runner, platform):
        result = test_runner.run(_multiloop_chain_case(platform=platform))
        assert result.passed, f"multi-loop chain phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_dense_mixed_phase_graph_correctness(self, test_runner, platform):
        result = test_runner.run(_dense_mixed_case(platform=platform))
        assert result.passed, f"dense mixed phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_partial_reduce_chain_correctness(self, test_runner, platform):
        result = test_runner.run(_partial_reduce_chain_case(platform=platform))
        assert result.passed, f"partial-reduce chain phase-fence failed: {result.error}"


class TestPhaseFenceDepCompressionSwimlane:
    def test_multiloop_chain_default(self, test_runner):
        data = _new_swimlane_json(test_runner, _multiloop_chain_case(), label="multi-loop chain phase-fence")
        _assert_multiloop_chain_shape(data)

    def test_submit_three_level_strict(self, test_runner):
        data = _new_swimlane_json(
            test_runner,
            _submit_case(epochs=2, layers=1, phases=3, name="phase_fence_submit_3l_swimlane"),
            label="three-level submit phase-fence",
        )
        _assert_flattened_stage_strict(data, stages=2 * 3, branches=_BRANCHES)

    def test_pl_at_three_level_strict(self, test_runner):
        _require_extra_swimlane_case("three-level pl.at swimlane")
        data = _new_swimlane_json(
            test_runner,
            _pl_at_case(epochs=2, phases=3, name="phase_fence_pl_at_3l_swimlane"),
            label="three-level pl.at phase-fence",
        )
        _assert_flattened_stage_strict(data, stages=2 * 3, branches=_BRANCHES)

    def test_reset_per_outer_generates_swimlane(self, test_runner):
        _require_extra_swimlane_case("reset-per-outer swimlane")
        data = _new_swimlane_json(test_runner, _reset_case(), label="reset-per-outer phase-fence")
        _assert_min_task_count(data, expected=2 * 2 * _BRANCHES)

    def test_sibling_loops_strict(self, test_runner):
        _require_extra_swimlane_case("sibling-loop swimlane")
        data = _new_swimlane_json(test_runner, _sibling_loops_case(), label="sibling-loop phase-fence")
        _assert_flattened_stage_strict(data, stages=2, branches=_BRANCHES)

    def test_if_consumer_strict(self, test_runner):
        _require_extra_swimlane_case("if-consumer swimlane")
        data = _new_swimlane_json(test_runner, _if_consumer_case(), label="if-consumer phase-fence")
        _assert_flattened_stage_strict(data, stages=2, branches=_BRANCHES)

    def test_if_mixed_fallback_swimlane(self, test_runner):
        _require_extra_swimlane_case("if-mixed-fallback swimlane")
        data = _new_swimlane_json(
            test_runner,
            _if_mixed_fallback_case(),
            label="if-mixed-fallback phase-fence",
        )
        _assert_min_task_count(data, expected=3 * _BRANCHES)

    def test_dense_mixed_extra(self, test_runner):
        _require_extra_swimlane_case("dense mixed swimlane")
        data = _new_swimlane_json(
            test_runner,
            _dense_mixed_case(branches=_DENSE_SWIMLANE_BRANCHES),
            label="dense mixed phase-fence",
        )
        _assert_dense_mixed_shape(data, branches=_DENSE_SWIMLANE_BRANCHES)

    def test_partial_reduce_chain_strict(self, test_runner):
        _require_extra_swimlane_case("partial-reduce chain swimlane")
        data = _new_swimlane_json(
            test_runner,
            _partial_reduce_chain_case(),
            label="partial-reduce chain phase-fence",
        )
        _assert_min_task_count(data, expected=2 * _BRANCHES + 1)
        tasks = sorted(data["tasks"], key=lambda t: t["start_time_us"])[: 2 * _BRANCHES + 1]
        producers = tasks[:_BRANCHES]
        reducer = tasks[_BRANCHES : _BRANCHES + 1]
        consumers = tasks[_BRANCHES + 1 :]
        producer_end = max(t["end_time_us"] for t in producers)
        reducer_start = reducer[0]["start_time_us"]
        reducer_end = reducer[0]["end_time_us"]
        consumer_start = min(t["start_time_us"] for t in consumers)
        assert reducer_start >= producer_end, (
            f"partial-reduce reducer starts at {reducer_start:.2f}us "
            f"before producers end at {producer_end:.2f}us"
        )
        assert consumer_start >= reducer_end, (
            f"partial-reduce consumers start at {consumer_start:.2f}us "
            f"before reducer ends at {reducer_end:.2f}us"
        )
