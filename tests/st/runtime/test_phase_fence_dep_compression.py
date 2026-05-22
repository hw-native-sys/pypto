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
            f"flattened stage {i + 1} starts at {start_next:.2f}us before stage {i} "
            f"ends at {end_i:.2f}us"
        )


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


class _PhaseFenceCase(PTOTestCase):
    __test__ = False

    def __init__(self, name: str, program_builder, *, rows: int, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._name = name
        self._program_builder = program_builder
        self._rows = rows

    def get_name(self) -> str:
        return self._name

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [self._rows, _BIG_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self._rows, _BIG_N], DataType.FP32, init_value=0.0, is_output=True),
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
    return _PhaseFenceCase("phase_fence_reset_per_outer", _build_reset_per_outer_program, rows=rows, platform=platform)


class TestPhaseFenceDepCompressionCorrectness:
    @pytest.fixture(autouse=True)
    def _skip_when_collecting_l2_swimlane(self, test_runner):
        if test_runner.config.enable_l2_swimlane:
            pytest.skip("correctness cases run without --enable-l2-swimlane; swimlane mode runs profiling witnesses")

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_submit_three_level_correctness(self, test_runner, platform):
        result = test_runner.run(
            _submit_case(epochs=2, layers=1, phases=3, name="phase_fence_submit_3l", platform=platform)
        )
        assert result.passed, f"three-level submit phase-fence failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_submit_four_level_correctness(self, test_runner, platform):
        result = test_runner.run(
            _submit_case(epochs=2, layers=2, phases=2, name="phase_fence_submit_4l", platform=platform)
        )
        assert result.passed, f"four-level submit phase-fence failed: {result.error}"

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


@pytest.fixture(scope="module")
def submit_three_level_swimlane(test_runner) -> dict:
    path = _new_swimlane_file(
        test_runner,
        _submit_case(epochs=2, layers=1, phases=3, name="phase_fence_submit_3l_swimlane"),
        label="three-level submit phase-fence",
    )
    return json.loads(path.read_text())


class TestPhaseFenceDepCompressionSwimlane:
    def test_submit_three_level_strict(self, submit_three_level_swimlane: dict):
        _assert_flattened_stage_strict(submit_three_level_swimlane, stages=2 * 3, branches=_BRANCHES)

    def test_submit_four_level_strict(self, test_runner):
        _require_extra_swimlane_case("four-level submit swimlane")
        path = _new_swimlane_file(
            test_runner,
            _submit_case(epochs=2, layers=2, phases=2, name="phase_fence_submit_4l_swimlane"),
            label="four-level submit phase-fence",
        )
        _assert_flattened_stage_strict(json.loads(path.read_text()), stages=2 * 2 * 2, branches=_BRANCHES)

    def test_pl_at_three_level_strict(self, test_runner):
        _require_extra_swimlane_case("three-level pl.at swimlane")
        path = _new_swimlane_file(
            test_runner,
            _pl_at_case(epochs=2, phases=3, name="phase_fence_pl_at_3l_swimlane"),
            label="three-level pl.at phase-fence",
        )
        _assert_flattened_stage_strict(json.loads(path.read_text()), stages=2 * 3, branches=_BRANCHES)
