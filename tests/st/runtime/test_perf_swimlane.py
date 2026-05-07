# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Swimlane JSON output validation tests.

Runs matmul 64x64x64 (PTO backend) with profiling and validates the
generated l2_perf_records.json in build_output/<run_dir>/swimlane_data/.

Requires ``--runtime-profiling`` to be set; pass ``--platform=a2a3`` (or
``a5``) to switch the target.  All tests in this file are skipped
automatically when ``--runtime-profiling`` is not passed.
"""

import json
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output"


class _MatmulPTO(PTOTestCase):
    """Matmul 64x64x64 with PTO backend — vehicle for swimlane generation."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_pto_64x64x64"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = 64, 64, 64

        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul(a, b, out_c)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


@pytest.fixture(scope="session")
def swimlane_file(test_runner) -> Path:
    """Run matmul once with profiling and return the generated swimlane file.

    Skips the entire test session (all dependent tests) when
    --runtime-profiling is not passed.
    """
    if not test_runner.config.runtime_profiling:
        pytest.skip("pass --runtime-profiling to run swimlane tests")

    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/swimlane_data/l2_perf_records.json"))

    result = test_runner.run(_MatmulPTO())
    assert result.passed, f"Matmul execution failed: {result.error}"

    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/swimlane_data/l2_perf_records.json"))
    new_files = after - before
    assert new_files, "No l2_perf_records.json was generated in build_output/*/swimlane_data/"

    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="session")
def swimlane_data(swimlane_file: Path) -> dict:
    return json.loads(swimlane_file.read_text())


class TestSwimlaneOutput:
    """Validate the structure and content of perf_swimlane_*.json."""

    def test_file_generated(self, swimlane_file: Path):
        """A l2_perf_records.json file is created in build_output/*/swimlane_data/."""
        assert swimlane_file.exists(), f"Swimlane file not found: {swimlane_file}"

    def test_top_level_structure(self, swimlane_data: dict):
        """Top-level 'version' and 'tasks' fields are present and non-empty."""
        assert "version" in swimlane_data, "Missing top-level field: 'version'"
        assert "tasks" in swimlane_data, "Missing top-level field: 'tasks'"
        assert len(swimlane_data["tasks"]) > 0, "tasks list is empty"

    def test_task_required_fields(self, swimlane_data: dict):
        """Each task contains all required fields with the correct types."""
        required: dict[str, type | tuple] = {
            "task_id": int,
            "func_id": int,
            "core_id": int,
            "core_type": str,
            "start_time_us": (int, float),
            "end_time_us": (int, float),
            "duration_us": (int, float),
            "fanout": list,
            "fanout_count": int,
        }
        for task in swimlane_data["tasks"]:
            for field, expected_type in required.items():
                assert field in task, f"task_id={task.get('task_id')}: missing field '{field}'"
                assert isinstance(task[field], expected_type), (
                    f"task_id={task.get('task_id')}: field '{field}' has type "
                    f"{type(task[field]).__name__}, expected {expected_type}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
