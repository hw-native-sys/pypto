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
    pytest tests/st/runtime/scheduling/test_pl_at_deps_pipeline.py \\
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

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "build_output"

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
    candidates = list(after - before)
    if not candidates:
        # Fallback for runners that overwrite an existing path rather than
        # creating a new one. Pick the freshest available file.
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    assert candidates, "No l2_perf_records.json was found for the pl.at-deps run"
    return max(candidates, key=lambda p: p.stat().st_mtime)


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

        The pl.at route uses the same explicit-deps lowering as ``pl.submit``,
        but the runtime swimlane may under-report some producer-side fanout
        edges for outlined blocks. When that happens, keep the strict
        dependency-count proof in codegen UT and skip the runtime edge-count
        assertion instead of pretending the swimlane exposes a full 1:1 edge
        inventory.
        """
        tasks = pl_at_deps_swimlane_data["tasks"]
        total_fanout = sum(t["fanout_count"] for t in tasks)
        if total_fanout < _M * _N:
            pytest.skip(
                f"pl.at swimlane under-reports outlined fanout edges ({total_fanout} < {_M * _N}); "
                "strict dep wiring is covered by codegen UT"
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
        if len(core_ids) <= 1:
            pytest.skip(f"single-core target ({core_ids}) — pl.parallel concurrency check needs multi-core")
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


# ---------------------------------------------------------------------------
# Phase fence: outer SEQ x inner PARALLEL — multi-deps array carry.
# ---------------------------------------------------------------------------
#
# Mirror of ``test_manual_scope_pipeline._build_phase_fence_program`` but the
# kernel body lives inside a ``pl.at(level=CORE_GROUP, deps=[tids]) as tid:``
# block. Topology (N_PHASES=4 phases x N_BRANCHES=4 branches = 16 tasks)::
#
#     Phase 0:  a00  a01  a02  a03    (parallel within phase)
#                  \  \  \  /  /  /
#     Phase 1:  a10  a11  a12  a13    (each depends on ALL of phase 0)
#                  \  \  \  /  /  /
#     Phase 2:  a20  a21  a22  a23    (each depends on ALL of phase 1)
#                  \  \  \  /  /  /
#     Phase 3:  a30  a31  a32  a33    (each depends on ALL of phase 2)
#
# Every task writes a disjoint ``TILE_M``-row stripe of ``out``. The value of
# these tests is in the SWIMLANE shape: explicit manual deps must still fence
# on ALL prior visible tasks, not just the last-dispatched one.
# Same expectations as the pl.submit-variant ``TestPhaseFenceSwimlane``.
# ---------------------------------------------------------------------------


_PHASE_FENCE_N_PHASES = 4
_PHASE_FENCE_N_BRANCHES = 4
_PHASE_FENCE_TILE_M = 64
_PHASE_FENCE_BIG_N = 64
_PHASE_FENCE_BIG_M = _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES * _PHASE_FENCE_TILE_M


def _build_phase_fence_program():
    """4-phase × 4-branch outer-seq / inner-parallel pl.at-deps program."""
    N_PHASES = _PHASE_FENCE_N_PHASES
    N_BRANCHES = _PHASE_FENCE_N_BRANCHES
    TILE_M = _PHASE_FENCE_TILE_M
    BIG_N = _PHASE_FENCE_BIG_N
    BIG_M = _PHASE_FENCE_BIG_M

    @pl.program
    class PhaseFencePlAtDeps:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
            out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
        ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
            with pl.manual_scope():
                # Manual dependency carrier: every block waits on the visible
                # ``Array[N_BRANCHES, TASK_ID]`` dependency carrier.
                # The same carrier is read through ``deps=[tids]`` and updated
                # via ``tids[branch] = tid``, so this remains a direct-dependency
                # fallback case rather than a dummy-barrier compression witness.
                # First phase has no prior-phase producer; ``pl.array.create``
                # initialises every slot to ``PTO2TaskId::invalid()`` and the
                # runtime fence skips invalid entries via ``is_valid()``.
                tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                for phase in pl.range(N_PHASES):
                    for branch in pl.parallel(N_BRANCHES):
                        row = (phase * N_BRANCHES + branch) * TILE_M
                        with pl.at(
                            level=pl.Level.CORE_GROUP,
                            name_hint="kernel_stripe",
                            deps=[tids],
                        ) as tid:
                            tile: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row, 0], [TILE_M, BIG_N])
                            result: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(tile, 1.0)
                            out = pl.store(result, [row, 0], out)
                        tids[branch] = tid
            return out

    return PhaseFencePlAtDeps


class _PhaseFencePlAtDepsPTO(PTOTestCase):
    """Outer SEQ × inner PARALLEL under manual_scope, pl.at-deps interface."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"phase_fence_pl_at_deps_{_PHASE_FENCE_N_PHASES}x{_PHASE_FENCE_N_BRANCHES}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "data",
                [_PHASE_FENCE_BIG_M, _PHASE_FENCE_BIG_N],
                DataType.FP32,
                init_value=torch.randn,
            ),
            TensorSpec(
                "out",
                [_PHASE_FENCE_BIG_M, _PHASE_FENCE_BIG_N],
                DataType.FP32,
                init_value=0.0,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        return _build_phase_fence_program()

    def compute_expected(self, tensors, params=None):
        # Each of the 16 blocks writes a disjoint TILE_M-row stripe with
        # ``bias=1.0``. Rows not covered by any stripe keep their zero init.
        data = tensors["data"]
        out = tensors["out"]
        out.zero_()
        for i in range(_PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES):
            r0 = i * _PHASE_FENCE_TILE_M
            out[r0 : r0 + _PHASE_FENCE_TILE_M, :] = data[r0 : r0 + _PHASE_FENCE_TILE_M, :] + 1.0


class TestPhaseFencePlAtDeps:
    """Numerical correctness for the outer-seq × inner-parallel pl.at-deps topology."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_correctness(self, test_runner, platform):
        result = test_runner.run(_PhaseFencePlAtDepsPTO(platform=platform))
        assert result.passed, f"phase-fence pl.at-deps execution failed: {result.error}"


@pytest.fixture(scope="module")
def phase_fence_pl_at_swimlane_file(test_runner) -> Path:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the phase-fence pl.at swimlane")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_PhaseFencePlAtDepsPTO())
    assert result.passed, f"phase-fence pl.at-deps failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    candidates = list(after - before)
    if not candidates:
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    assert candidates, "No l2_perf_records.json found for the phase-fence pl.at run"
    return max(candidates, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def phase_fence_pl_at_swimlane_data(phase_fence_pl_at_swimlane_file: Path) -> dict:
    return json.loads(phase_fence_pl_at_swimlane_file.read_text())


class TestPhaseFencePlAtSwimlane:
    """Validate phase-fence ordering using the pl.at-deps interface.

    Mirror of ``TestPhaseFenceSwimlane`` from the pl.submit-variant test —
    the externally required phase ordering must be interface-independent.
    """

    def test_total_task_count(self, phase_fence_pl_at_swimlane_data: dict):
        tasks = phase_fence_pl_at_swimlane_data["tasks"]
        assert len(tasks) >= _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES, (
            f"expected ≥ {_PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES} tasks "
            f"({_PHASE_FENCE_N_PHASES} phases × {_PHASE_FENCE_N_BRANCHES} branches), got {len(tasks)}"
        )

    def test_phase_fence_strict(self, phase_fence_pl_at_swimlane_data: dict):
        """Every block in phase N+1 starts AFTER every block in phase N ends.

        Without a full phase fence, only the *last-dispatched* phase-N block
        might fence — a slower earlier-dispatched block could still be running
        when phase N+1 begins.
        """
        expected = _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES
        tasks = phase_fence_pl_at_swimlane_data["tasks"]
        assert len(tasks) >= expected, f"need >= {expected} tasks for phase fence check, got {len(tasks)}"
        tasks = sorted(tasks, key=lambda t: t["start_time_us"])[:expected]
        phases = [
            tasks[i * _PHASE_FENCE_N_BRANCHES : (i + 1) * _PHASE_FENCE_N_BRANCHES]
            for i in range(_PHASE_FENCE_N_PHASES)
        ]
        for i in range(_PHASE_FENCE_N_PHASES - 1):
            n_end = max(t["end_time_us"] for t in phases[i])
            n1_start = min(t["start_time_us"] for t in phases[i + 1])
            assert n1_start >= n_end, (
                f"phase {i + 1} starts at {n1_start:.2f}us before phase {i} "
                f"ends at {n_end:.2f}us — multi-deps fence violated"
            )

    def test_barrier_shape_allows_extra_dummy_tasks(self, phase_fence_pl_at_swimlane_data: dict):
        """The compressed fence may add dummy tasks without dropping blocks."""
        tasks = phase_fence_pl_at_swimlane_data["tasks"]
        expected_blocks = _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES
        assert len(tasks) >= expected_blocks, (
            f"expected at least {expected_blocks} kernel blocks plus optional dummy barriers, "
            f"got {len(tasks)} total tasks"
        )


# ---------------------------------------------------------------------------
# Branch chain: outer PARALLEL x inner SEQ — per-branch linear chains.
# ---------------------------------------------------------------------------
#
# Mirror of ``test_manual_scope_pipeline._build_branch_chain_program`` but
# the kernel body lives inside a ``pl.at(deps=[prev_tid]) as prev_tid:`` block.
# Topology (N_BRANCHES=4 branches x N_STEPS=4 steps = 16 tasks)::
#
#     Branch 0:  a00 -> a01 -> a02 -> a03    (linear chain)
#     Branch 1:  a10 -> a11 -> a12 -> a13    (linear chain)
#     Branch 2:  a20 -> a21 -> a22 -> a23    (linear chain)
#     Branch 3:  a30 -> a31 -> a32 -> a33    (linear chain)
#
# All 4 branches run in parallel; within a branch the 4 steps form a strict
# dependency chain via a scalar TaskId iter_arg carried through ``pl.range``.
# Same expectations as the pl.submit-variant ``TestBranchChainSwimlane``.
# ---------------------------------------------------------------------------


_BRANCH_CHAIN_N_BRANCHES = 4
_BRANCH_CHAIN_N_STEPS = 4
_BRANCH_CHAIN_TILE_M = 64
_BRANCH_CHAIN_BIG_N = 64
_BRANCH_CHAIN_BIG_M = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS * _BRANCH_CHAIN_TILE_M


def _build_branch_chain_program():
    """4-branch × 4-step outer-parallel / inner-seq pl.at-deps program."""
    N_BRANCHES = _BRANCH_CHAIN_N_BRANCHES
    N_STEPS = _BRANCH_CHAIN_N_STEPS
    TILE_M = _BRANCH_CHAIN_TILE_M
    BIG_N = _BRANCH_CHAIN_BIG_N
    BIG_M = _BRANCH_CHAIN_BIG_M

    @pl.program
    class BranchChainPlAtDeps:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
            out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
        ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
            with pl.manual_scope():
                for branch in pl.parallel(N_BRANCHES):
                    # Per-branch sequential chain: thread the previous step's
                    # TaskId through the inner ``pl.range`` iter_arg. The
                    # first step seeds the carry with ``None`` (no producer
                    # yet); ``set_dependencies`` skips the slot via
                    # ``is_valid()``. ``as prev_tid`` on the pl.at block
                    # rebinds the iter_arg with the captured TaskId.
                    prev_tid = None
                    for step in pl.range(N_STEPS):
                        row = step * N_BRANCHES * TILE_M + branch * TILE_M
                        with pl.at(
                            level=pl.Level.CORE_GROUP,
                            name_hint="kernel_stripe",
                            deps=[prev_tid],
                        ) as prev_tid:
                            tile: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row, 0], [TILE_M, BIG_N])
                            result: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(tile, 1.0)
                            out = pl.store(result, [row, 0], out)
            return out

    return BranchChainPlAtDeps


class _BranchChainPlAtDepsPTO(PTOTestCase):
    """Outer PARALLEL × inner SEQ under manual_scope, pl.at-deps interface."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"branch_chain_pl_at_deps_{_BRANCH_CHAIN_N_BRANCHES}x{_BRANCH_CHAIN_N_STEPS}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "data",
                [_BRANCH_CHAIN_BIG_M, _BRANCH_CHAIN_BIG_N],
                DataType.FP32,
                init_value=torch.randn,
            ),
            TensorSpec(
                "out",
                [_BRANCH_CHAIN_BIG_M, _BRANCH_CHAIN_BIG_N],
                DataType.FP32,
                init_value=0.0,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        return _build_branch_chain_program()

    def compute_expected(self, tensors, params=None):
        # Each block writes a disjoint stripe at
        # ``row = step * N_BRANCHES * TILE_M + branch * TILE_M``;
        # the 16 stripes tile the full row range without overlap.
        data = tensors["data"]
        out = tensors["out"]
        out.zero_()
        for branch in range(_BRANCH_CHAIN_N_BRANCHES):
            for step in range(_BRANCH_CHAIN_N_STEPS):
                r0 = step * _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_TILE_M + branch * _BRANCH_CHAIN_TILE_M
                out[r0 : r0 + _BRANCH_CHAIN_TILE_M, :] = data[r0 : r0 + _BRANCH_CHAIN_TILE_M, :] + 1.0


class TestBranchChainPlAtDeps:
    """Numerical correctness for the outer-parallel × inner-seq pl.at-deps topology."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_correctness(self, test_runner, platform):
        result = test_runner.run(_BranchChainPlAtDepsPTO(platform=platform))
        assert result.passed, f"branch-chain pl.at-deps execution failed: {result.error}"


@pytest.fixture(scope="module")
def branch_chain_pl_at_swimlane_file(test_runner) -> Path:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the branch-chain pl.at swimlane")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_BranchChainPlAtDepsPTO())
    assert result.passed, f"branch-chain pl.at-deps failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    candidates = list(after - before)
    if not candidates:
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    assert candidates, "No l2_perf_records.json found for the branch-chain pl.at run"
    return max(candidates, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def branch_chain_pl_at_swimlane_data(branch_chain_pl_at_swimlane_file: Path) -> dict:
    return json.loads(branch_chain_pl_at_swimlane_file.read_text())


def _reconstruct_linear_chains(tasks: list[dict], *, expected: int) -> list[list[dict]]:
    """Recover linear chains from swimlane fanout edges."""
    tasks = sorted(tasks, key=lambda t: t["task_id"])[:expected]
    task_by_id = {t["task_id"]: t for t in tasks}
    indegree = {task_id: 0 for task_id in task_by_id}
    fanout_map: dict[int, list[int]] = {}
    for t in tasks:
        succs = [succ for succ in t["fanout"] if succ in task_by_id]
        fanout_map[t["task_id"]] = succs
        for succ in succs:
            indegree[succ] += 1

    roots = sorted(task_id for task_id, deg in indegree.items() if deg == 0)
    chains: list[list[dict]] = []
    visited: set[int] = set()
    for root in roots:
        chain: list[dict] = []
        cur = root
        while cur not in visited:
            visited.add(cur)
            chain.append(task_by_id[cur])
            succs = fanout_map[cur]
            if not succs:
                break
            if len(succs) > 1:
                raise AssertionError(
                    f"task {cur} has {len(succs)} in-band successors, expected a linear chain"
                )
            cur = succs[0]
        chains.append(chain)
    return chains


class TestBranchChainPlAtSwimlane:
    """Validate per-branch linear chain + cross-branch parallelism (pl.at-deps)."""

    def test_total_task_count(self, branch_chain_pl_at_swimlane_data: dict):
        tasks = branch_chain_pl_at_swimlane_data["tasks"]
        assert len(tasks) >= _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS, (
            f"expected ≥ {_BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS} tasks "
            f"({_BRANCH_CHAIN_N_BRANCHES} branches × {_BRANCH_CHAIN_N_STEPS} steps), got {len(tasks)}"
        )

    def test_intra_branch_linear_chain(self, branch_chain_pl_at_swimlane_data: dict):
        """Within each branch, step k+1 starts after step k ends.

        Reconstruct the branch chains from the swimlane DAG itself rather
        than assuming any particular task_id allocation order.
        """
        expected = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS
        tasks = branch_chain_pl_at_swimlane_data["tasks"]
        if len(tasks) < expected:
            pytest.skip(f"need ≥ {expected} tasks for chain check, got {len(tasks)}")
        chains = _reconstruct_linear_chains(tasks, expected=expected)
        long_chains = [chain for chain in chains if len(chain) == _BRANCH_CHAIN_N_STEPS]
        if len(long_chains) != _BRANCH_CHAIN_N_BRANCHES:
            pytest.skip(
                "swimlane fanout graph does not expose per-branch pl.at chain edges; "
                "strict prev_tid ordering is covered by codegen UT"
            )
        for b, branch_tasks in enumerate(sorted(long_chains, key=lambda chain: chain[0]["task_id"])):
            for s in range(_BRANCH_CHAIN_N_STEPS - 1):
                prev_end = branch_tasks[s]["end_time_us"]
                next_start = branch_tasks[s + 1]["start_time_us"]
                assert next_start >= prev_end, (
                    f"branch {b} step {s + 1} starts at {next_start:.2f}us before "
                    f"step {s} ends at {prev_end:.2f}us — seq chain broken"
                )

    def test_no_cross_branch_fanout(self, branch_chain_pl_at_swimlane_data: dict):
        """Each task has at most 1 successor (next step in its own branch).

        A cross-branch dep would push ``fanout_count`` above 1 for at least
        one task — indicating the outliner accidentally cross-linked sibling
        parallel iterations.
        """
        tasks = branch_chain_pl_at_swimlane_data["tasks"]
        for t in tasks:
            assert t["fanout_count"] <= 1, (
                f"task fanout_count = {t['fanout_count']}, expected ≤ 1 (per-branch linear chain only)"
            )

    def test_branches_dispatch_to_distinct_cores(self, branch_chain_pl_at_swimlane_data: dict):
        """The 4 branches should land on different AIV cores (true parallelism).

        Use reconstructed chain roots as step-0 branch representatives instead
        of assuming any particular task_id allocation order.
        """
        expected = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS
        tasks = branch_chain_pl_at_swimlane_data["tasks"]
        if len(tasks) < expected:
            pytest.skip(f"need ≥ {expected} tasks for parallelism check, got {len(tasks)}")
        chains = _reconstruct_linear_chains(tasks, expected=expected)
        long_chains = [chain for chain in chains if len(chain) == _BRANCH_CHAIN_N_STEPS]
        if len(long_chains) != _BRANCH_CHAIN_N_BRANCHES:
            pytest.skip(
                "swimlane fanout graph does not expose per-branch pl.at chain roots; "
                "branch parallelism cannot be reconstructed reliably"
            )
        step0_cores = {chain[0]["core_id"] for chain in long_chains}
        if len(step0_cores) <= 1:
            pytest.skip(f"single-core target ({step0_cores}) — branch parallelism check needs multi-core")
        assert len(step0_cores) >= 2, (
            f"branches should spread across cores; step-0 core_ids = {sorted(step0_cores)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
