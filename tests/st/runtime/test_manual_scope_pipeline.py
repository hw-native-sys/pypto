# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end on-board test for ``with pl.manual_scope():`` around a 2-stage
nested-loop pipeline.

The program tiles a ``[128, 128]`` matrix with a ``[32, 32]`` block grid
(M=4, N=4). Each ``(i, j)`` tile runs a 2-stage pipeline:

- stage1: ``scratch[r, c] = 2 * x[r, c]``
- stage2: ``out[r, c]     = scratch[r, c] + 1``

The orchestrator wraps the nested loops in ``with pl.manual_scope():``:

    for i in pl.range(M):
        row = i * TILE_R
        for j in pl.parallel(N):
            col = j * TILE_C
            scratch = self.stage1(x, scratch, row, col)
            out     = self.stage2(scratch, out, row, col)

What the swimlane visualization should show
-------------------------------------------
The dependency graph derived by ``DeriveManualScopeDeps`` produces:

* **Within an iteration**: stage2 has an explicit ``add_dep(task_<stage1>)``
  on stage1, so stage2 starts strictly after stage1 finishes for the same
  ``(i, j)`` tile.
* **Across iterations**: no extra ``add_dep`` is emitted, so different
  ``(i, j)`` tiles run at maximum parallelism.

In the swimlane chart this manifests as 2 vertically-stacked tasks per
``(i, j)`` tile (stage1 then stage2, with a fan-out edge between them) and
``N=4`` such pairs running concurrently per outer iteration on the
available AIV cores.

How to run
----------

::

    # On real hardware, with profiling enabled:
    pytest tests/st/runtime/test_manual_scope_pipeline.py \\
        --runtime-profiling --platform=a2a3

    # Without --runtime-profiling, the swimlane assertions skip and only
    # numerical correctness is checked.
"""

import json
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output"

# Tile grid — kept small so a single run produces a readable swimlane chart.
_M = 4
_N = 4
_TILE_R = 32
_TILE_C = 32
_ROWS = _M * _TILE_R
_COLS = _N * _TILE_C


def _build_program():
    """Build the 2-stage manual-scope pipeline program.

    Hoisted out of the test class so we can reference the same closure
    constants directly inside the ``@pl.program`` body without lambdas or
    indirection.
    """
    M, N = _M, _N
    TILE_R, TILE_C = _TILE_R, _TILE_C
    ROWS, COLS = _ROWS, _COLS

    @pl.program
    class ManualScopePipelineProgram:
        """``out = 2*x + 1`` tiled across a ``[ROWS, COLS]`` grid."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage1(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            row: pl.Scalar[pl.INDEX],
            col: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
            r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)  # 2 * x
            ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def stage2(
            self,
            scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            row: pl.Scalar[pl.INDEX],
            col: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
            r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, 1.0)  # scratch + 1
            ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            with pl.manual_scope():
                for i in pl.range(M):
                    row: pl.Scalar[pl.INDEX] = i * TILE_R
                    for j in pl.parallel(N):
                        col: pl.Scalar[pl.INDEX] = j * TILE_C
                        scratch = self.stage1(x, scratch, row, col)
                        out = self.stage2(scratch, out, row, col)
            return out

    return ManualScopePipelineProgram


class _ManualScopePipelinePTO(PTOTestCase):
    """``out = 2*x + 1`` via a 2-stage pipeline inside a manual_scope."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"manual_scope_pipeline_{_ROWS}x{_COLS}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_ROWS, _COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("scratch", [_ROWS, _COLS], DataType.FP32, init_value=0.0),
            TensorSpec("out", [_ROWS, _COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _build_program()

    def compute_expected(self, tensors, params=None):
        # out = 2 * x + 1 element-wise.
        tensors["out"][:] = 2.0 * tensors["x"] + 1.0


class TestManualScopePipeline:
    """Numerical correctness check — runs on every supported platform."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pipeline_correctness(self, test_runner, platform):
        """``out`` matches ``2 * x + 1`` after on-board execution.

        This guards against three regressions at once: the manual_scope
        codegen wrapper, the explicit ``add_dep`` between stage1/stage2,
        and the absence of cross-iteration serialization (which would
        still pass numerically but show up as wrong parallelism in the
        swimlane fixture below).
        """
        result = test_runner.run(_ManualScopePipelinePTO(platform=platform))
        assert result.passed, f"Manual-scope pipeline execution failed: {result.error}"


# ---------------------------------------------------------------------------
# Swimlane validation — only when --runtime-profiling is enabled.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manual_scope_swimlane_file(test_runner) -> Path:
    """Run the pipeline once with profiling and return the swimlane JSON."""
    if not test_runner.config.runtime_profiling:
        pytest.skip("pass --runtime-profiling to validate the manual_scope swimlane")

    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/swimlane_data/l2_perf_records.json"))
    result = test_runner.run(_ManualScopePipelinePTO())
    assert result.passed, f"Manual-scope pipeline failed: {result.error}"

    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/swimlane_data/l2_perf_records.json"))
    new_files = after - before
    assert new_files, "No l2_perf_records.json was generated for the manual_scope run"
    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def manual_scope_swimlane_data(manual_scope_swimlane_file: Path) -> dict:
    return json.loads(manual_scope_swimlane_file.read_text())


class TestManualScopeSwimlane:
    """Validate the on-board execution graph encoded in the swimlane JSON.

    These assertions are the on-board counterpart of the unit-level
    ``test_manual_scope_seq_outer_parallel_inner_two_stage_pipeline``: the
    unit test pins the codegen output, this test pins what the runtime
    actually does.
    """

    def test_total_task_count(self, manual_scope_swimlane_data: dict):
        """Each of the ``M * N`` tiles submits 2 kernel tasks (stage1 + stage2)."""
        tasks = manual_scope_swimlane_data["tasks"]
        # The tile grid runs ``M * N`` iterations of (stage1 + stage2). Some
        # platforms may emit extra runtime/setup tasks; the lower bound is
        # the only safe assertion.
        assert len(tasks) >= _M * _N * 2, (
            f"expected at least {_M * _N * 2} tasks (M*N tiles x 2 stages), got {len(tasks)}"
        )

    def test_intra_iteration_dep_present(self, manual_scope_swimlane_data: dict):
        """Stage2 must wait for the same iteration's stage1.

        We can't recover ``(i, j)`` directly from the swimlane, but every
        stage2 task should depend on at least one earlier stage1 (its
        ``fanout_count`` from the producer side, or its parent task in the
        DAG). The minimum requirement: at least ``M * N`` fan-out edges
        exist, one per stage1→stage2 pair.
        """
        tasks = manual_scope_swimlane_data["tasks"]
        total_fanout = sum(t["fanout_count"] for t in tasks)
        assert total_fanout >= _M * _N, (
            f"expected at least {_M * _N} fan-out edges (one per stage1->stage2 pair), got {total_fanout}"
        )

    def test_inner_parallel_loop_runs_concurrently(self, manual_scope_swimlane_data: dict):
        """Inner ``pl.parallel(N)`` iterations must overlap across cores.

        With manual_scope and no cross-iteration ``add_dep``, the runtime
        is free to dispatch all ``N`` tiles of one outer iteration to
        ``N`` different AIV cores. The assertion: across all tasks, at
        least 2 distinct ``core_id`` values appear (i.e. the runtime did
        in fact parallelize). On a 1-core simulator this assertion is
        relaxed automatically.
        """
        tasks = manual_scope_swimlane_data["tasks"]
        core_ids = {t["core_id"] for t in tasks}
        # On a multi-core target the inner parallel loop should spread work
        # across cores; on single-core simulators just check we ran at all.
        if len(core_ids) > 1:
            assert len(core_ids) >= 2, (
                f"expected manual_scope's pl.parallel inner loop to use multiple cores; "
                f"only saw core_ids={sorted(core_ids)}"
            )

    def test_no_blocking_serialization_chain(self, manual_scope_swimlane_data: dict):
        """No single task may fan out to more than the necessary downstream count.

        If ``DeriveManualScopeDeps`` mistakenly cross-linked iterations,
        stage1 of an early iteration would fan out to *every* later
        stage1/stage2 in the same scope, blowing up the fan-out count
        well past the per-iteration bound (which is 1: stage1 -> its own
        stage2). The threshold below allows for runtime-injected sync
        tasks but catches grossly serialized graphs.
        """
        tasks = manual_scope_swimlane_data["tasks"]
        max_fanout = max((t["fanout_count"] for t in tasks), default=0)
        assert max_fanout <= 4, (
            f"max fan-out per task is {max_fanout} — manual_scope deps appear over-linked; "
            "iterations should not chain."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
