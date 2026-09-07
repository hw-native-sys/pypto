# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end guard for the ``st.cases`` declaration path.

A case declared with ``@st.cases(...)`` has to survive the whole route:
collection reads it out of ``callspec.params``, the pre-compile pool builds its
IR from the ``@pl.jit`` entry, the golden runs in this process, and the device
step is batched like any other case.

The discovery half is what this file actually guards, and it is guarded by
assertion rather than by inspection: a case the pipeline picked up has a
published artifact directory, and one that fell through to the per-case inline
path does not.  Before ``st.cases`` existed, a ``@pl.jit`` test could not be
discovered at all — it silently took the inline path — so a regression here
would otherwise show up only as CI getting slower.

Runs card-free under ``--codegen-only``; on a device it is one small kernel in
the batched pool.
"""

from concurrent.futures import Future
from dataclasses import dataclass

import pypto.language as pl
import pytest
import torch

from harness import st
from harness.core import test_runner as tr

M = 16
N = 16


@pl.jit.incore
def _abs_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@pl.jit
def _abs_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _abs_kernel(a, out)
    return out


@dataclass
class _StubConfig:
    rtol: float
    atol: float


@dataclass
class _StubCase:
    """The surface ``_schedule_exec_after_golden`` reads: a name and a tolerance."""

    rtol: float
    atol: float

    @property
    def config(self) -> _StubConfig:
        return _StubConfig(self.rtol, self.atol)

    def get_name(self) -> str:
        return "shared_case"


torch.manual_seed(0)
_A = torch.randn(M, N, dtype=torch.float32)


@st.cases(
    st.case(
        _abs_entry,
        _A,
        torch.zeros(M, N, dtype=torch.float32),
        name="harness_pipeline_abs",
        golden=lambda tensors: torch.abs(tensors["a"]),
    ),
)
def test_declared_case_runs(case_run, request):
    """The declared case compiles, runs, and matches its golden."""
    case_run.assert_passed()

    # Discovery check: with the pre-compile pool active, a discovered case owns
    # a published artifact directory. ``None`` here means collection did not see
    # the declaration and the case fell back to inline compilation.
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None, (
            "case was not picked up by the pre-compile pipeline — "
            "pytest_collection_finish did not read it from callspec.params"
        )
        assert (case_run.work_dir / "golden.py").exists()


class TestSharedRun:
    """One declared case, several test functions, one device run.

    A swimlane or dump-args group asserts a dozen different things about a
    single profiled run.  Written as ``@st.cases(SAME_CASE)`` on each test, the
    device work has to happen once: the artifact is already shared, so
    scheduling a run per assertion buys nothing and costs a card run each time.

    Driven against ``_schedule_exec_after_golden`` directly rather than through
    a session, because the property is about how many executions are submitted
    -- something a passing test run cannot show.
    """

    @staticmethod
    def _artifact(tmp_path):
        class _Artifact:
            error = None
            work_dir = tmp_path

        return _Artifact()

    def test_one_submit_per_key(self, tmp_path, monkeypatch):
        submitted: list[str] = []

        class _Pool:
            def submit(self, _fn, _tc, cache_key, _artifact):
                submitted.append(cache_key)
                fut = Future()
                fut.set_result(None)
                return fut

        monkeypatch.setattr(tr, "_execute_pool", _Pool())
        monkeypatch.setattr(tr, "_execute_futures", {})
        monkeypatch.setitem(tr._pipeline_ctx, "codegen_only", True)

        tc = _StubCase(rtol=1e-5, atol=1e-5)
        artifact = self._artifact(tmp_path)
        first = tr._schedule_exec_after_golden(tc, "shared@a2a3@default", artifact)
        second = tr._schedule_exec_after_golden(tc, "shared@a2a3@default", artifact)

        assert submitted == ["shared@a2a3@default"], "the second declaration re-ran the case"
        assert first is second, "both tests must await the same run"

    def test_a_name_collision_is_named_not_absorbed(self, tmp_path, monkeypatch):
        """Same key, different tolerance means two cases share a name.

        They already share the compile artifact and its golden.py, so silently
        reusing the run would validate one case against the other's threshold.
        """

        class _Pool:
            def submit(self, _fn, _tc, _cache_key, _artifact):
                fut = Future()
                fut.set_result(None)
                return fut

        monkeypatch.setattr(tr, "_execute_pool", _Pool())
        monkeypatch.setattr(tr, "_execute_futures", {})
        monkeypatch.setitem(tr._pipeline_ctx, "codegen_only", True)

        artifact = self._artifact(tmp_path)
        tr._schedule_exec_after_golden(_StubCase(1e-5, 1e-5), "dup@a2a3@default", artifact)
        with pytest.raises(AssertionError, match="disagree on tolerance"):
            tr._schedule_exec_after_golden(_StubCase(1e-3, 1e-3), "dup@a2a3@default", artifact)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
