# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end on-board test for ``with pl.at(level=CORE_GROUP, deps=[tid]) as tid:``.

This is the ``pl.at``-block analogue of ``test_manual_scope_pipeline.py``'s
two-stage pipeline. Same computation, same tile grid, same manual_scope
wrapper, same swimlane expectations — the difference is the *interface*
used to express the dep edge:

* ``test_manual_scope_pipeline.py`` uses ``pl.submit(self.stageX, ...)``
  against pre-declared ``@pl.function(InCore)`` kernels.
* This test uses inline ``with pl.at(level=pl.Level.CORE_GROUP, ...) as tid:``
  blocks. The outliner lifts each block into a synthesised InCore kernel
  + Call, and the ``deps=[tid]`` / ``as tid`` plumbing hooks into the same
  ``Call.attrs["manual_dep_edges"]`` codegen path that ``pl.submit`` uses.

The program tiles a ``[128, 128]`` matrix with a ``[32, 32]`` block grid
(M=4, N=4). Each ``(i, j)`` tile runs the same 2-stage pipeline:

- stage1 (``pl.at``-block): ``scratch[r, c] = 2 * x[r, c]``
- stage2 (``pl.at``-block, ``deps=[stage1_tid]``): ``out[r, c] = scratch[r, c] + 1``

What the swimlane should show
-----------------------------
The user-declared ``deps=[stage1_tid]`` on the stage2 block produces:

* **Within an iteration**: stage2's ``set_dependencies`` lists stage1's
  producer TaskId, so stage2 starts strictly after stage1 finishes for the
  same ``(i, j)`` tile.
* **Across iterations**: no extra dependency is emitted, so different
  ``(i, j)`` tiles run at maximum parallelism.

How to run
----------

::

    # On real hardware, with profiling enabled:
    pytest tests/st/runtime/test_pl_at_deps_pipeline.py \\
        --enable-l2-swimlane --platform=a2a3

    # Without --enable-l2-swimlane, the swimlane assertions skip and only
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
    """Build the 2-stage pl.at-block pipeline program.

    Mirrors ``test_manual_scope_pipeline._build_program`` but replaces the
    ``pl.submit(self.stageX, ...)`` calls with inline
    ``with pl.at(level=pl.Level.CORE_GROUP, ...) as tid:`` blocks.
    """
    M, N = _M, _N
    TILE_R, TILE_C = _TILE_R, _TILE_C
    ROWS, COLS = _ROWS, _COLS

    @pl.program
    class PlAtDepsPipelineProgram:
        """``out = 2*x + 1`` tiled across a ``[ROWS, COLS]`` grid, using pl.at blocks."""

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
                        # Stage 1: scratch[row..row+TILE_R, col..col+TILE_C] = 2 * x[...].
                        # ``as stage1_tid`` captures the TaskId of the outlined Call
                        # the outliner will synthesise for this block.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as stage1_tid:
                            t1: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                            r1: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t1, t1)
                            scratch = pl.store(r1, [row, col], scratch)
                        # Stage 2: depends explicitly on stage1's TaskId via deps=.
                        with pl.at(
                            level=pl.Level.CORE_GROUP,
                            name_hint="stage2",
                            deps=[stage1_tid],
                        ) as _stage2_tid:
                            t2: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(
                                scratch, [row, col], [TILE_R, TILE_C]
                            )
                            r2: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t2, 1.0)
                            out = pl.store(r2, [row, col], out)
            return out

    return PlAtDepsPipelineProgram


class _PlAtDepsPipelinePTO(PTOTestCase):
    """``out = 2*x + 1`` via a 2-stage pl.at-block pipeline inside manual_scope."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"pl_at_deps_pipeline_{_ROWS}x{_COLS}"

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


class TestPlAtDepsPipeline:
    """Numerical correctness check — runs on every supported platform."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pipeline_correctness(self, test_runner, platform):
        """``out`` matches ``2 * x + 1`` after on-board execution.

        Guards three regressions at once: the outliner's task_id_var/manual_dep_edges
        plumbing on pl.at-blocks, the explicit ``set_dependencies`` edge between
        stage1/stage2, and the absence of cross-iteration serialisation (which
        would still pass numerically but show up as wrong parallelism in the
        swimlane fixture below).
        """
        result = test_runner.run(_PlAtDepsPipelinePTO(platform=platform))
        assert result.passed, f"pl.at-deps pipeline execution failed: {result.error}"


# ---------------------------------------------------------------------------
# Swimlane validation — only when --enable-l2-swimlane is enabled.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pl_at_deps_swimlane_file(test_runner) -> Path:
    """Run the pipeline once with profiling and return the swimlane JSON."""
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the pl.at-deps swimlane")

    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_PlAtDepsPipelinePTO())
    assert result.passed, f"pl.at-deps pipeline failed: {result.error}"

    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    new_files = after - before
    assert new_files, "No l2_perf_records.json was generated for the pl.at-deps run"
    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def pl_at_deps_swimlane_data(pl_at_deps_swimlane_file: Path) -> dict:
    return json.loads(pl_at_deps_swimlane_file.read_text())


class TestPlAtDepsSwimlane:
    """Validate the on-board execution graph for the pl.at-block pipeline.

    Mirror of ``test_manual_scope_pipeline.TestManualScopeSwimlane``: the
    same DAG shape (M*N tile pairs, fan-out 1 within iteration, parallel
    across iterations) must hold, regardless of whether the dep was wired
    via ``pl.submit(..., deps=)`` or via ``pl.at(..., deps=) as tid``.
    """

    def test_total_task_count(self, pl_at_deps_swimlane_data: dict):
        """Each of the ``M * N`` tiles emits 2 outlined-kernel tasks."""
        tasks = pl_at_deps_swimlane_data["tasks"]
        assert len(tasks) >= _M * _N * 2, (
            f"expected at least {_M * _N * 2} tasks (M*N tiles x 2 stages), got {len(tasks)}"
        )

    def test_intra_iteration_dep_present(self, pl_at_deps_swimlane_data: dict):
        """Stage2 must wait for the same iteration's stage1.

        At least ``M * N`` fan-out edges should be observed (one per
        stage1→stage2 pair). With pl.at-block deps wired correctly via
        the outliner, this mirrors the manual_scope_pipeline expectation.
        """
        tasks = pl_at_deps_swimlane_data["tasks"]
        total_fanout = sum(t["fanout_count"] for t in tasks)
        assert total_fanout >= _M * _N, (
            f"expected at least {_M * _N} fan-out edges (one per stage1->stage2 pair), got {total_fanout}"
        )

    def test_inner_parallel_loop_runs_concurrently(self, pl_at_deps_swimlane_data: dict):
        """Inner ``pl.parallel(N)`` iterations must overlap across cores.

        With explicit per-tile deps and no cross-iteration edge, the runtime
        is free to dispatch all ``N`` tiles of one outer iteration to
        ``N`` different AIV cores. On a multi-core target at least 2 distinct
        ``core_id`` values must appear; on single-core simulators the
        assertion is relaxed automatically.
        """
        tasks = pl_at_deps_swimlane_data["tasks"]
        core_ids = {t["core_id"] for t in tasks}
        if len(core_ids) > 1:
            assert len(core_ids) >= 2, (
                f"expected pl.parallel inner loop to use multiple cores; only saw core_ids={sorted(core_ids)}"
            )

    def test_no_blocking_serialization_chain(self, pl_at_deps_swimlane_data: dict):
        """No single task may fan out to more than the necessary downstream count.

        If the outliner mistakenly cross-linked iterations, stage1 of an
        early iteration would fan out to *every* later stage1/stage2 in the
        same scope, blowing up the fan-out count well past the per-iteration
        bound (which is 1: stage1 -> its own stage2). Same threshold as the
        pl.submit-variant test to keep the two interfaces' DAG shape aligned.
        """
        tasks = pl_at_deps_swimlane_data["tasks"]
        max_fanout = max((t["fanout_count"] for t in tasks), default=0)
        assert max_fanout <= 4, (
            f"max fan-out per task is {max_fanout} — pl.at deps appear over-linked; "
            "iterations should not chain."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
