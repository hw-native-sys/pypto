# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST regression for windowed consumer input dependencies.

The program writes disjoint windows of an ``Out`` tensor inside ``pl.at``
CORE_GROUP tasks, then each consumer task reads a wider window from the produced
tensor. The generated orchestration must pass that read window to
``params.add_input`` so runtime TensorMap overlap can discover window-level
producer/consumer dependencies.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from harness.core.test_runner import TestRunner
from pypto import ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

_N = 64
_M = 128
_SCORE_R = 32
_SCORE_C = 32
_PROBE_R = 32
_PROBE_C = 64
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _build_program():
    """Build a compact windowed-producer/full-parent-reader program."""
    N, M = _N, _M
    SCORE_R, SCORE_C = _SCORE_R, _SCORE_C
    PROBE_R, PROBE_C = _PROBE_R, _PROBE_C

    @pl.program
    class RuntimeScopePreludeProgram:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            probe: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[N, M], pl.FP32],
            pl.Tensor[[N, M], pl.FP32],
        ]:
            score_flat: pl.Tensor[[N, M], pl.FP32] = pl.reshape(score, [N, M])
            for r0 in pl.parallel(0, N, SCORE_R):
                for c0 in pl.parallel(0, M, SCORE_C):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="produce"):
                        x_window = x[r0 : r0 + SCORE_R, c0 : c0 + SCORE_C]
                        score_flat = pl.assemble(score_flat, pl.add(x_window, x_window), [r0, c0])
            for r0 in pl.parallel(0, N, PROBE_R):
                for c0 in pl.parallel(0, M, PROBE_C):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="consume"):
                        block = score_flat[r0 : r0 + PROBE_R, c0 : c0 + PROBE_C]
                        probe = pl.assemble(probe, pl.add(block, block), [r0, c0])
            return score_flat, probe

    return RuntimeScopePreludeProgram


class _RuntimeScopePreludePTO(PTOTestCase):
    """``probe = 4*x`` through a windowed producer and a full-parent reader."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"runtime_scope_prelude_{_N}x{_M}_s{_SCORE_R}x{_SCORE_C}_p{_PROBE_R}x{_PROBE_C}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_N, _M], DataType.FP32, init_value=torch.randn),
            TensorSpec("score", [_N, _M], DataType.FP32, is_output=True),
            TensorSpec("probe", [_N, _M], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _build_program()

    def compute_expected(self, tensors, params=None):
        tensors["score"][:] = 2.0 * tensors["x"]
        tensors["probe"][:] = 4.0 * tensors["x"]


def _read_single(path: Path, pattern: str) -> str:
    matches = sorted(path.glob(pattern))
    assert matches, f"no file matching {pattern} under {path}"
    assert len(matches) == 1, f"expected one {pattern} under {path}, got {matches}"
    return matches[0].read_text()


def _latest_pass_dump(work_dir: Path, pass_name: str) -> str:
    matches = sorted((work_dir / "passes_dump").glob(f"*_after_{pass_name}.py"))
    assert matches, f"missing pass dump for {pass_name} under {work_dir / 'passes_dump'}"
    return matches[-1].read_text()


def _default_saved_parent_dir(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _PROJECT_ROOT / "build_output" / f"{prefix}_{timestamp}"


def _assert_consumer_uses_score_window_input(code: str) -> None:
    task_pos = code.index("// Task 1: consume__windowed")
    consumer_region = code[code.rfind("PTO2_SCOPE() {", 0, task_pos) : task_pos]
    consumer_task = code[task_pos:]
    score_input_match = re.search(r"Tensor\s+([A-Za-z_]\w*)\s*=\s*score_flat\.view\(", consumer_region)
    assert score_input_match is not None, code
    score_input_name = score_input_match.group(1)

    assert f"params_t1.add_input({score_input_name})" in consumer_task, code
    assert "params_t1.add_input(score_flat)" not in consumer_task, code
    assert "std::min<uint32_t>(32" in consumer_region, code
    assert "std::min<uint32_t>(64" in consumer_region, code


class TestRuntimeScopePreludeCodegen:
    """Compile-time ST that inspects pass dumps and generated orchestration."""

    def test_windowed_slice_is_runtime_scope_prelude(self, request, tmp_path, monkeypatch):
        if request.config.getoption("--save-kernels"):
            output_dir = _default_saved_parent_dir("RuntimeScopePreludeProgram")
            monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
        else:
            output_dir = None
            monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))

        compiled = ir.compile(
            _build_program(),
            output_dir=output_dir,
            backend_type=BackendType.Ascend910B,
            dump_passes=True,
        )
        work_dir = compiled.output_dir

        optimized = _latest_pass_dump(work_dir, "OptimizeOrchTensors")
        assert "pl.tensor.slice(score_flat__rv_v2, [32, 64]" in optimized, optimized
        assert "consume__windowed(" in optimized and "__window" in optimized, optimized
        assert "produce__windowed" in optimized, optimized

        code = _read_single(work_dir / "orchestration", "*.cpp")
        _assert_consumer_uses_score_window_input(code)


class TestRuntimeScopePreludeExecution:
    """Numerical correctness check for the same program."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_runtime_correctness(self, test_runner, platform):
        result = test_runner.run(_RuntimeScopePreludePTO(platform=platform))
        assert result.passed, f"runtime-scope prelude execution failed: {result.error}"


@pytest.fixture(scope="module")
def runtime_scope_prelude_swimlane_file(test_runner) -> Path:
    """Run the program with profiling and return its swimlane JSON."""
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate runtime-scope prelude swimlane")

    case = _RuntimeScopePreludePTO()
    output_root = (
        Path(test_runner.config.save_kernels_dir)
        if test_runner.config.save_kernels_dir
        else _default_saved_parent_dir("runtime_scope_prelude_swimlane")
    )
    work_dir = output_root / case.get_name()
    config = RunConfig(
        platform=test_runner.config.platform,
        device_id=test_runner.config.device_id,
        rtol=test_runner.config.rtol,
        atol=test_runner.config.atol,
        save_kernels=True,
        save_kernels_dir=str(output_root),
        pto_isa_commit=test_runner.config.pto_isa_commit,
        enable_l2_swimlane=True,
    )
    case = _RuntimeScopePreludePTO(config=config)
    result = TestRunner(config).run(case)
    assert result.passed, f"runtime-scope prelude swimlane run failed: {result.error}"

    swimlane_path = work_dir / "dfx_outputs" / "l2_perf_records.json"
    assert swimlane_path.exists(), f"No l2_perf_records.json generated at {swimlane_path}"
    return swimlane_path


@pytest.fixture(scope="module")
def runtime_scope_prelude_swimlane_data(runtime_scope_prelude_swimlane_file: Path) -> dict:
    return json.loads(runtime_scope_prelude_swimlane_file.read_text())


class TestRuntimeScopePreludeSwimlane:
    """Validate that the profiled run emits task records for producer/consumer tasks."""

    def test_swimlane_has_tasks(self, runtime_scope_prelude_swimlane_data: dict):
        tasks = runtime_scope_prelude_swimlane_data["tasks"]
        expected_tasks = (_N // _SCORE_R) * (_M // _SCORE_C) + (_N // _PROBE_R) * (_M // _PROBE_C)
        assert len(tasks) >= expected_tasks, (
            f"expected at least producer+consumer task records, got {len(tasks)}"
        )

    def test_swimlane_required_fields(self, runtime_scope_prelude_swimlane_data: dict):
        required = {"task_id", "func_id", "core_id", "start_time_us", "end_time_us", "duration_us"}
        for task in runtime_scope_prelude_swimlane_data["tasks"]:
            missing = required - set(task)
            assert not missing, f"task_id={task.get('task_id')}: missing {sorted(missing)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
