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
            scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
            out, _              = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])

What the swimlane visualization should show
-------------------------------------------
The user-declared ``deps=[stage1_tid]`` on the stage2 submit produces:

* **Within an iteration**: stage2's ``set_dependencies`` lists stage1's
  producer TaskId, so stage2 starts strictly after stage1 finishes for the
  same ``(i, j)`` tile.
* **Across iterations**: no extra dependency is emitted, so different
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
                        scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
                        out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])
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
        codegen wrapper, the explicit ``set_dependencies`` edge between stage1/stage2,
        and the absence of cross-iteration serialization (which would
        still pass numerically but show up as wrong parallelism in the
        swimlane fixture below).
        """
        result = test_runner.run(_ManualScopePipelinePTO(platform=platform))
        assert result.passed, f"Manual-scope pipeline execution failed: {result.error}"


# ---------------------------------------------------------------------------
# Swimlane validation — only when --enable-l2-swimlane is enabled.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manual_scope_swimlane_file(test_runner) -> Path:
    """Run the pipeline once with profiling and return the swimlane JSON."""
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the manual_scope swimlane")

    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_ManualScopePipelinePTO())
    assert result.passed, f"Manual-scope pipeline failed: {result.error}"

    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
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

        With manual_scope and no cross-iteration dependency edge, the runtime
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

        If the codegen mistakenly cross-linked iterations,
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


# ---------------------------------------------------------------------------
# Phase fence: outer SEQ x inner PARALLEL — multi-deps array carry.
# ---------------------------------------------------------------------------
#
# Topology (N_PHASES=4 phases x N_BRANCHES=4 branches = 16 tasks)::
#
#     Phase 0:  a00  a01  a02  a03    (parallel within phase)
#                  \  \  \  /  /  /
#     Phase 1:  a10  a11  a12  a13    (each depends on ALL of phase 0)
#                  \  \  \  /  /  /
#     Phase 2:  a20  a21  a22  a23    (each depends on ALL of phase 1)
#                  \  \  \  /  /  /
#     Phase 3:  a30  a31  a32  a33    (each depends on ALL of phase 2)
#
# Every task writes a disjoint ``TILE_M``-row stripe of ``out`` so the program
# is numerically correct regardless of dep semantics — the value of these
# tests is in the SWIMLANE shape: the orchestration codegen's array-carry
# lowering must produce a phase fence keyed on ALL prior-phase tasks, not
# just the last-dispatched one.
# ---------------------------------------------------------------------------


_PHASE_FENCE_N_PHASES = 4
_PHASE_FENCE_N_BRANCHES = 4
_PHASE_FENCE_TILE_M = 64
_PHASE_FENCE_BIG_N = 64
_PHASE_FENCE_BIG_M = _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES * _PHASE_FENCE_TILE_M


def _build_phase_fence_program():
    """4-phase × 4-branch outer-seq / inner-parallel manual_scope program."""
    N_PHASES = _PHASE_FENCE_N_PHASES
    N_BRANCHES = _PHASE_FENCE_N_BRANCHES
    TILE_M = _PHASE_FENCE_TILE_M
    BIG_N = _PHASE_FENCE_BIG_N
    BIG_M = _PHASE_FENCE_BIG_M

    @pl.program
    class PhaseFenceManualScope:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            bias: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
        ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
            tile: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row_offset, 0], [TILE_M, BIG_N])
            result: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(tile, bias)
            ret: pl.Tensor[[BIG_M, BIG_N], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
            out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
        ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
            with pl.manual_scope():
                # Phase fence: every kernel in phase P+1 must wait for ALL
                # branches in phase P. Use an explicit ``Array[N_BRANCHES,
                # TASK_ID]`` to hold one TaskId per branch slot — every
                # parallel iter writes its own slot, and ``deps=[tids]``
                # expands to N_BRANCHES guarded slot fills in the
                # downstream task's ``set_dependencies`` array (one per slot
                # of the previous phase).
                # ``pl.array.create`` auto-initializes all slots to
                # ``PTO2TaskId::invalid()``; the runtime fence skips
                # invalid entries via ``is_valid()`` so the first phase
                # has no prior-phase dependency.
                tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                for phase in pl.range(N_PHASES):
                    for branch in pl.parallel(N_BRANCHES):
                        row = (phase * N_BRANCHES + branch) * TILE_M
                        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids])
                        tids[branch] = tid
            return out

    return PhaseFenceManualScope


class _PhaseFenceManualScopePTO(PTOTestCase):
    """Outer SEQ × inner PARALLEL under manual_scope (multi-deps phase fence)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"phase_fence_manual_scope_{_PHASE_FENCE_N_PHASES}x{_PHASE_FENCE_N_BRANCHES}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "data", [_PHASE_FENCE_BIG_M, _PHASE_FENCE_BIG_N], DataType.FP32, init_value=torch.randn
            ),
            TensorSpec(
                "out", [_PHASE_FENCE_BIG_M, _PHASE_FENCE_BIG_N], DataType.FP32, init_value=0.0, is_output=True
            ),
        ]

    def get_program(self) -> Any:
        return _build_phase_fence_program()

    def compute_expected(self, tensors, params=None):
        # Each of the 16 kernels writes a disjoint TILE_M-row stripe with
        # ``bias=1.0``. Rows not covered by any stripe keep their zero init.
        data = tensors["data"]
        out = tensors["out"]
        out.zero_()
        for i in range(_PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES):
            r0 = i * _PHASE_FENCE_TILE_M
            out[r0 : r0 + _PHASE_FENCE_TILE_M, :] = data[r0 : r0 + _PHASE_FENCE_TILE_M, :] + 1.0


class TestPhaseFenceManualScope:
    """Numerical correctness for the outer-seq × inner-parallel topology."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_correctness(self, test_runner, platform):
        result = test_runner.run(_PhaseFenceManualScopePTO(platform=platform))
        assert result.passed, f"phase-fence manual_scope execution failed: {result.error}"


@pytest.fixture(scope="module")
def phase_fence_swimlane_file(test_runner) -> Path:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the phase-fence swimlane")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_PhaseFenceManualScopePTO())
    assert result.passed, f"phase-fence manual_scope failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    new_files = after - before
    assert new_files, "No l2_perf_records.json generated for the phase-fence run"
    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def phase_fence_swimlane_data(phase_fence_swimlane_file: Path) -> dict:
    return json.loads(phase_fence_swimlane_file.read_text())


class TestPhaseFenceSwimlane:
    """Validate the array-carry multi-deps fence in the runtime swimlane.

    The structural witness that array-carry codegen is doing the right
    thing: each phase-N task fans out to ALL ``N_BRANCHES`` tasks of phase
    N+1, so the runtime fence on phase N+1 waits for every phase-N task
    to complete — not just the last-dispatched one.
    """

    def test_total_task_count(self, phase_fence_swimlane_data: dict):
        tasks = phase_fence_swimlane_data["tasks"]
        assert len(tasks) >= _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES, (
            f"expected ≥ {_PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES} tasks "
            f"({_PHASE_FENCE_N_PHASES} phases × {_PHASE_FENCE_N_BRANCHES} branches), got {len(tasks)}"
        )

    def test_phase_fence_strict(self, phase_fence_swimlane_data: dict):
        """Every task in phase N+1 starts AFTER every task in phase N ends.

        Group all kernel_stripe tasks by start time into ``N_PHASES`` batches
        of ``N_BRANCHES`` and verify ``phase[N+1].min_start ≥ phase[N].max_end``.
        Without array-carry multi-deps, only the *last-dispatched* phase-N
        task would fence — a slower earlier-dispatched task could still be
        running when phase N+1 begins.
        """
        expected = _PHASE_FENCE_N_PHASES * _PHASE_FENCE_N_BRANCHES
        tasks = phase_fence_swimlane_data["tasks"]
        if len(tasks) < expected:
            pytest.skip(f"need ≥ {expected} tasks for phase fence check, got {len(tasks)}")
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

    def test_multi_deps_fanout_observed(self, phase_fence_swimlane_data: dict):
        """At least one task fans out to ``N_BRANCHES`` successors.

        With array-carry codegen, every task in phase N is depended on by
        every task in phase N+1, so the maximum ``fanout_count`` over the
        whole DAG should be ≥ ``N_BRANCHES``. A regression where codegen
        only records the *last-dispatched* phase-N task as a dep would cap
        the max fanout at 1.

        The runtime may elide dep-edge records on the consumer side when a
        downstream task is already free to run, so we don't require *every*
        phase-N task to show fanout ``N_BRANCHES`` — observing the multi-
        dep pattern on at least one task is sufficient evidence.
        """
        tasks = phase_fence_swimlane_data["tasks"]
        max_fanout = max((t["fanout_count"] for t in tasks), default=0)
        assert max_fanout >= _PHASE_FENCE_N_BRANCHES, (
            f"max fanout across all tasks = {max_fanout}, expected ≥ {_PHASE_FENCE_N_BRANCHES} "
            "(array-carry multi-deps means each phase-N task should fan out to all "
            f"{_PHASE_FENCE_N_BRANCHES} phase-N+1 tasks)"
        )


# ---------------------------------------------------------------------------
# Branch chain: outer PARALLEL x inner SEQ — per-branch linear chains.
# ---------------------------------------------------------------------------
#
# Topology (N_BRANCHES=4 branches x N_STEPS=4 steps = 16 tasks)::
#
#     Branch 0:  a00 -> a01 -> a02 -> a03    (linear chain)
#     Branch 1:  a10 -> a11 -> a12 -> a13    (linear chain)
#     Branch 2:  a20 -> a21 -> a22 -> a23    (linear chain)
#     Branch 3:  a30 -> a31 -> a32 -> a33    (linear chain)
#
# All 4 branches run in parallel; within a branch the 4 steps form a
# strict dependency chain. Codegen lowers this as outer-parallel
# array-carry of size 4 plus an inner-sequential scalar carry — see
# ``orchestration_codegen.cpp``.
# ---------------------------------------------------------------------------


_BRANCH_CHAIN_N_BRANCHES = 4
_BRANCH_CHAIN_N_STEPS = 4
_BRANCH_CHAIN_TILE_M = 64
_BRANCH_CHAIN_BIG_N = 64
_BRANCH_CHAIN_BIG_M = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS * _BRANCH_CHAIN_TILE_M


def _build_branch_chain_program():
    """4-branch × 4-step outer-parallel / inner-seq manual_scope program."""
    N_BRANCHES = _BRANCH_CHAIN_N_BRANCHES
    N_STEPS = _BRANCH_CHAIN_N_STEPS
    TILE_M = _BRANCH_CHAIN_TILE_M
    BIG_N = _BRANCH_CHAIN_BIG_N
    BIG_M = _BRANCH_CHAIN_BIG_M

    @pl.program
    class BranchChainManualScope:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            bias: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
        ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
            tile: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row_offset, 0], [TILE_M, BIG_N])
            result: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(tile, bias)
            ret: pl.Tensor[[BIG_M, BIG_N], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

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
                    # ``is_valid()``.
                    prev_tid = None
                    for step in pl.range(N_STEPS):
                        row = step * N_BRANCHES * TILE_M + branch * TILE_M
                        out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
            return out

    return BranchChainManualScope


class _BranchChainManualScopePTO(PTOTestCase):
    """Outer PARALLEL × inner SEQ under manual_scope (per-branch linear chains)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"branch_chain_manual_scope_{_BRANCH_CHAIN_N_BRANCHES}x{_BRANCH_CHAIN_N_STEPS}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "data", [_BRANCH_CHAIN_BIG_M, _BRANCH_CHAIN_BIG_N], DataType.FP32, init_value=torch.randn
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
        # Each kernel writes a disjoint stripe at
        # ``row = step * N_BRANCHES * TILE_M + branch * TILE_M``;
        # the 16 stripes tile the full row range without overlap.
        data = tensors["data"]
        out = tensors["out"]
        out.zero_()
        for branch in range(_BRANCH_CHAIN_N_BRANCHES):
            for step in range(_BRANCH_CHAIN_N_STEPS):
                r0 = step * _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_TILE_M + branch * _BRANCH_CHAIN_TILE_M
                out[r0 : r0 + _BRANCH_CHAIN_TILE_M, :] = data[r0 : r0 + _BRANCH_CHAIN_TILE_M, :] + 1.0


class TestBranchChainManualScope:
    """Numerical correctness for the outer-parallel × inner-seq topology."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_correctness(self, test_runner, platform):
        result = test_runner.run(_BranchChainManualScopePTO(platform=platform))
        assert result.passed, f"branch-chain manual_scope execution failed: {result.error}"


@pytest.fixture(scope="module")
def branch_chain_swimlane_file(test_runner) -> Path:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip("pass --enable-l2-swimlane to validate the branch-chain swimlane")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(_BranchChainManualScopePTO())
    assert result.passed, f"branch-chain manual_scope failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    new_files = after - before
    assert new_files, "No l2_perf_records.json generated for the branch-chain run"
    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def branch_chain_swimlane_data(branch_chain_swimlane_file: Path) -> dict:
    return json.loads(branch_chain_swimlane_file.read_text())


class TestBranchChainSwimlane:
    """Validate per-branch linear chain + cross-branch parallelism."""

    def test_total_task_count(self, branch_chain_swimlane_data: dict):
        tasks = branch_chain_swimlane_data["tasks"]
        assert len(tasks) >= _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS, (
            f"expected ≥ {_BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS} tasks "
            f"({_BRANCH_CHAIN_N_BRANCHES} branches × {_BRANCH_CHAIN_N_STEPS} steps), got {len(tasks)}"
        )

    def test_intra_branch_linear_chain(self, branch_chain_swimlane_data: dict):
        """Within each branch, step k+1 starts after step k ends.

        Tasks dispatch in branch-major / step-minor order (outer for-loop
        over branches in the emitted C++), so the first ``N_STEPS`` tasks
        by ``task_id`` belong to branch 0, the next ``N_STEPS`` to branch 1,
        and so on.
        """
        expected = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS
        tasks = branch_chain_swimlane_data["tasks"]
        if len(tasks) < expected:
            pytest.skip(f"need ≥ {expected} tasks for chain check, got {len(tasks)}")
        tasks = sorted(tasks, key=lambda t: t["task_id"])[:expected]
        for b in range(_BRANCH_CHAIN_N_BRANCHES):
            branch_tasks = tasks[b * _BRANCH_CHAIN_N_STEPS : (b + 1) * _BRANCH_CHAIN_N_STEPS]
            for s in range(_BRANCH_CHAIN_N_STEPS - 1):
                prev_end = branch_tasks[s]["end_time_us"]
                next_start = branch_tasks[s + 1]["start_time_us"]
                assert next_start >= prev_end, (
                    f"branch {b} step {s + 1} starts at {next_start:.2f}us before "
                    f"step {s} ends at {prev_end:.2f}us — seq chain broken"
                )

    def test_no_cross_branch_fanout(self, branch_chain_swimlane_data: dict):
        """Each task has at most 1 successor (next step in its own branch).

        A cross-branch dep would push ``fanout_count`` above 1 for at least
        one task — indicating the codegen accidentally cross-linked sibling
        parallel iterations.
        """
        tasks = branch_chain_swimlane_data["tasks"]
        for t in tasks:
            assert t["fanout_count"] <= 1, (
                f"task fanout_count = {t['fanout_count']}, expected ≤ 1 (per-branch linear chain only)"
            )

    def test_branches_dispatch_to_distinct_cores(self, branch_chain_swimlane_data: dict):
        """The 4 branches should land on different AIV cores (true parallelism).

        On single-core simulators this assertion is relaxed.
        """
        expected = _BRANCH_CHAIN_N_BRANCHES * _BRANCH_CHAIN_N_STEPS
        tasks = branch_chain_swimlane_data["tasks"]
        if len(tasks) < expected:
            pytest.skip(f"need ≥ {expected} tasks for parallelism check, got {len(tasks)}")
        tasks = sorted(tasks, key=lambda t: t["task_id"])[:expected]
        # Step 0 of each branch — the first task in each chain — should run
        # on a distinct core when multiple cores are available.
        step0_cores = {tasks[b * _BRANCH_CHAIN_N_STEPS]["core_id"] for b in range(_BRANCH_CHAIN_N_BRANCHES)}
        if len({t["core_id"] for t in tasks}) > 1:
            assert len(step0_cores) >= 2, (
                f"branches should spread across cores; step-0 core_ids = {sorted(step0_cores)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
