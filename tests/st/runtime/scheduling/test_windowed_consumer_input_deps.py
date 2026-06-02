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
_CACHE_BLOCKS = 4
_CACHE_BLOCK_SIZE = 16
_CACHE_ROWS = _CACHE_BLOCKS * _CACHE_BLOCK_SIZE
_CACHE_DIM = 64
_CACHE_READ_DIM = 32
_STATE_ROWS = 8
_STATE_DIM = 64
_STATE_ROW_TILE = 4
_STATE_COL_TILE = 32
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _build_program():
    """Build a compact windowed-producer/windowed-consumer program."""
    N, M = _N, _M
    SCORE_R, SCORE_C = _SCORE_R, _SCORE_C
    PROBE_R, PROBE_C = _PROBE_R, _PROBE_C

    @pl.program
    class WindowedConsumerInputDepsProgram:
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

    return WindowedConsumerInputDepsProgram


def _build_cache_program():
    """Build a v4 indexer-style cache flatten/write/readback program."""
    BLOCKS, BLOCK_SIZE = _CACHE_BLOCKS, _CACHE_BLOCK_SIZE
    ROWS, DIM, READ_DIM = _CACHE_ROWS, _CACHE_DIM, _CACHE_READ_DIM

    @pl.program
    class WindowedCacheInputDepsProgram:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[ROWS, DIM], pl.FP32],
            cache: pl.Out[pl.Tensor[[BLOCKS, BLOCK_SIZE, 1, DIM], pl.FP32]],
            probe: pl.Out[pl.Tensor[[ROWS, DIM], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[BLOCKS, BLOCK_SIZE, 1, DIM], pl.FP32],
            pl.Tensor[[ROWS, DIM], pl.FP32],
        ]:
            cache_flat: pl.Tensor[[ROWS, DIM], pl.FP32] = pl.reshape(cache, [ROWS, DIM])
            for row0 in pl.parallel(0, ROWS, BLOCK_SIZE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cache_write"):
                    block = x[row0 : row0 + BLOCK_SIZE, 0:DIM]
                    cache_flat = pl.assemble(cache_flat, pl.add(block, block), [row0, 0])

            for row0 in pl.parallel(0, ROWS, BLOCK_SIZE):
                for col0 in pl.parallel(0, DIM, READ_DIM):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cache_read"):
                        cache_block = cache_flat[row0 : row0 + BLOCK_SIZE, col0 : col0 + READ_DIM]
                        probe = pl.assemble(probe, pl.add(cache_block, cache_block), [row0, col0])

            cache = pl.reshape(cache_flat, [BLOCKS, BLOCK_SIZE, 1, DIM])
            return cache, probe

    return WindowedCacheInputDepsProgram


def _build_state_pair_program():
    """Build a prefill-compressor-style kv_state/score_state handoff."""
    ROWS, DIM = _STATE_ROWS, _STATE_DIM
    ROW_TILE, COL_TILE = _STATE_ROW_TILE, _STATE_COL_TILE

    @pl.program
    class WindowedStatePairInputDepsProgram:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[ROWS, DIM], pl.FP32],
            kv_state: pl.Out[pl.Tensor[[1, ROWS, DIM], pl.FP32]],
            score_state: pl.Out[pl.Tensor[[1, ROWS, DIM], pl.FP32]],
            probe: pl.Out[pl.Tensor[[ROWS, DIM], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[1, ROWS, DIM], pl.FP32],
            pl.Tensor[[1, ROWS, DIM], pl.FP32],
            pl.Tensor[[ROWS, DIM], pl.FP32],
        ]:
            kv_flat: pl.Tensor[[ROWS, DIM], pl.FP32] = pl.reshape(kv_state, [ROWS, DIM])
            score_flat: pl.Tensor[[ROWS, DIM], pl.FP32] = pl.reshape(score_state, [ROWS, DIM])

            for col0 in pl.parallel(0, DIM, COL_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_write"):
                    chunk = x[:, col0 : col0 + COL_TILE]
                    doubled = pl.add(chunk, chunk)
                    kv_flat = pl.assemble(kv_flat, doubled, [0, col0])
                    score_flat = pl.assemble(score_flat, doubled, [0, col0])

            for row0 in pl.parallel(0, ROWS, ROW_TILE):
                for col0 in pl.parallel(0, DIM, COL_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_read"):
                        kv_block = kv_flat[row0 : row0 + ROW_TILE, col0 : col0 + COL_TILE]
                        score_block = score_flat[row0 : row0 + ROW_TILE, col0 : col0 + COL_TILE]
                        probe = pl.assemble(probe, pl.add(kv_block, score_block), [row0, col0])

            kv_state = pl.reshape(kv_flat, [1, ROWS, DIM])
            score_state = pl.reshape(score_flat, [1, ROWS, DIM])
            return kv_state, score_state, probe

    return WindowedStatePairInputDepsProgram


class _WindowedConsumerInputDepsPTO(PTOTestCase):
    """``probe = 4*x`` through windowed producer and consumer tasks."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"windowed_consumer_input_deps_{_N}x{_M}_s{_SCORE_R}x{_SCORE_C}_p{_PROBE_R}x{_PROBE_C}"

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


class _WindowedCacheInputDepsPTO(PTOTestCase):
    """``probe = 4*x`` through a flattened 4D cache handoff."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"windowed_cache_input_deps_{_CACHE_ROWS}x{_CACHE_DIM}_r{_CACHE_BLOCK_SIZE}x{_CACHE_READ_DIM}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_CACHE_ROWS, _CACHE_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "cache",
                [_CACHE_BLOCKS, _CACHE_BLOCK_SIZE, 1, _CACHE_DIM],
                DataType.FP32,
                is_output=True,
            ),
            TensorSpec("probe", [_CACHE_ROWS, _CACHE_DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _build_cache_program()

    def compute_expected(self, tensors, params=None):
        tensors["cache"][:] = (2.0 * tensors["x"]).reshape(
            _CACHE_BLOCKS,
            _CACHE_BLOCK_SIZE,
            1,
            _CACHE_DIM,
        )
        tensors["probe"][:] = 4.0 * tensors["x"]


class _WindowedStatePairInputDepsPTO(PTOTestCase):
    """``probe = 4*x`` through paired kv_state/score_state windows."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return (
            f"windowed_state_pair_input_deps_{_STATE_ROWS}x{_STATE_DIM}_r{_STATE_ROW_TILE}x{_STATE_COL_TILE}"
        )

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_STATE_ROWS, _STATE_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("kv_state", [1, _STATE_ROWS, _STATE_DIM], DataType.FP32, is_output=True),
            TensorSpec("score_state", [1, _STATE_ROWS, _STATE_DIM], DataType.FP32, is_output=True),
            TensorSpec("probe", [_STATE_ROWS, _STATE_DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _build_state_pair_program()

    def compute_expected(self, tensors, params=None):
        state = (2.0 * tensors["x"]).reshape(1, _STATE_ROWS, _STATE_DIM)
        tensors["kv_state"][:] = state
        tensors["score_state"][:] = state
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


def _assert_task_uses_window_inputs(code: str, task_name: str, parent_names: list[str]) -> None:
    _assert_task_uses_window_args(code, task_name, parent_names, "input")


def _assert_task_uses_window_outputs(code: str, task_name: str, parent_names: list[str]) -> None:
    _assert_task_uses_window_args(code, task_name, parent_names, "output")


def _assert_task_uses_window_args(code: str, task_name: str, parent_names: list[str], arg_kind: str) -> None:
    task_match = re.search(rf"// Task \d+: {re.escape(task_name)}__windowed", code)
    assert task_match is not None, code

    task_start = task_match.start()
    next_task_match = re.search(r"\n\s*// Task \d+:", code[task_start + 1 :])
    task_end = len(code) if next_task_match is None else task_start + 1 + next_task_match.start()
    task_region = code[task_start:task_end]
    scope_region = code[code.rfind("PTO2_SCOPE() {", 0, task_start) : task_start]

    for parent_name in parent_names:
        window_match = re.search(
            rf"Tensor\s+({re.escape(parent_name)}[A-Za-z0-9_]*__window)\s*=\s*"
            rf"{re.escape(parent_name)}[A-Za-z0-9_]*\.view\(",
            scope_region,
        )
        assert window_match is not None, code
        window_name = window_match.group(1)
        assert f"add_{arg_kind}({window_name})" in task_region, code
        assert f"add_{arg_kind}({parent_name})" not in task_region, code


def _assert_pass_dump_has_window(optimized: str, parent_prefix: str, shape: str) -> None:
    assert re.search(rf"pl\.tensor\.slice\({re.escape(parent_prefix)}\w*, \[{shape}\]", optimized), optimized


class TestWindowedConsumerInputDepsCodegen:
    """Compile-time ST that inspects pass dumps and generated orchestration."""

    def test_consumer_add_input_uses_score_window(self, request, tmp_path, monkeypatch):
        if request.config.getoption("--save-kernels"):
            output_dir = _default_saved_parent_dir("WindowedConsumerInputDepsProgram")
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
        _assert_task_uses_window_outputs(code, "produce", ["score_flat"])
        _assert_consumer_uses_score_window_input(code)

    def test_flattened_cache_consumer_uses_cache_window(self, request, tmp_path, monkeypatch):
        if request.config.getoption("--save-kernels"):
            output_dir = _default_saved_parent_dir("WindowedCacheInputDepsProgram")
            monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
        else:
            output_dir = None
            monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))

        compiled = ir.compile(
            _build_cache_program(),
            output_dir=output_dir,
            backend_type=BackendType.Ascend910B,
            dump_passes=True,
        )
        work_dir = compiled.output_dir

        optimized = _latest_pass_dump(work_dir, "OptimizeOrchTensors")
        _assert_pass_dump_has_window(optimized, "cache_flat__rv_", "16, 32")
        assert "cache_read__windowed(" in optimized and "__window" in optimized, optimized

        code = _read_single(work_dir / "orchestration", "*.cpp")
        _assert_task_uses_window_outputs(code, "cache_write", ["cache_flat"])
        _assert_task_uses_window_inputs(code, "cache_read", ["cache_flat"])

    def test_state_pair_consumer_uses_both_state_windows(self, request, tmp_path, monkeypatch):
        if request.config.getoption("--save-kernels"):
            output_dir = _default_saved_parent_dir("WindowedStatePairInputDepsProgram")
            monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
        else:
            output_dir = None
            monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))

        compiled = ir.compile(
            _build_state_pair_program(),
            output_dir=output_dir,
            backend_type=BackendType.Ascend910B,
            dump_passes=True,
        )
        work_dir = compiled.output_dir

        optimized = _latest_pass_dump(work_dir, "OptimizeOrchTensors")
        _assert_pass_dump_has_window(optimized, "kv_flat__rv_", "4, 32")
        _assert_pass_dump_has_window(optimized, "score_flat__rv_", "4, 32")
        assert "state_read__windowed(" in optimized and "__window" in optimized, optimized

        code = _read_single(work_dir / "orchestration", "*.cpp")
        _assert_task_uses_window_outputs(code, "state_write", ["kv_flat", "score_flat"])
        _assert_task_uses_window_inputs(code, "state_read", ["kv_flat", "score_flat"])


class TestWindowedConsumerInputDepsExecution:
    """Numerical correctness check for the same program."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "case_cls",
        [_WindowedConsumerInputDepsPTO, _WindowedCacheInputDepsPTO, _WindowedStatePairInputDepsPTO],
    )
    def test_runtime_correctness(self, test_runner, platform, case_cls):
        result = test_runner.run(case_cls(platform=platform))
        assert result.passed, f"windowed consumer input dependency execution failed: {result.error}"


@pytest.fixture(scope="module")
def windowed_consumer_input_deps_swimlane_file(test_runner) -> Path:
    """Run the program with profiling and return its swimlane JSON."""
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate windowed consumer input dependency swimlane")

    case = _WindowedConsumerInputDepsPTO()
    output_root = (
        Path(test_runner.config.save_kernels_dir)
        if test_runner.config.save_kernels_dir
        else _default_saved_parent_dir("windowed_consumer_input_deps_swimlane")
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
    case = _WindowedConsumerInputDepsPTO(config=config)
    result = TestRunner(config).run(case)
    assert result.passed, f"windowed consumer input dependency swimlane run failed: {result.error}"

    swimlane_path = work_dir / "dfx_outputs" / "l2_perf_records.json"
    assert swimlane_path.exists(), f"No l2_perf_records.json generated at {swimlane_path}"
    return swimlane_path


@pytest.fixture(scope="module")
def windowed_consumer_input_deps_swimlane_data(windowed_consumer_input_deps_swimlane_file: Path) -> dict:
    return json.loads(windowed_consumer_input_deps_swimlane_file.read_text())


class TestWindowedConsumerInputDepsSwimlane:
    """Validate that the profiled run emits task records for producer/consumer tasks."""

    def test_swimlane_has_tasks(self, windowed_consumer_input_deps_swimlane_data: dict):
        tasks = windowed_consumer_input_deps_swimlane_data["tasks"]
        expected_tasks = (_N // _SCORE_R) * (_M // _SCORE_C) + (_N // _PROBE_R) * (_M // _PROBE_C)
        assert len(tasks) >= expected_tasks, (
            f"expected at least producer+consumer task records, got {len(tasks)}"
        )

    def test_swimlane_required_fields(self, windowed_consumer_input_deps_swimlane_data: dict):
        required = {"task_id", "func_id", "core_id", "start_time_us", "end_time_us", "duration_us"}
        for task in windowed_consumer_input_deps_swimlane_data["tasks"]:
            missing = required - set(task)
            assert not missing, f"task_id={task.get('task_id')}: missing {sorted(missing)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
