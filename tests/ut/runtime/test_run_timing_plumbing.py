# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for ``RunTiming`` plumbing (issue #1679).

Verifies that the per-run ``RunTiming`` measured by the simpler ``Worker`` is
surfaced (rather than silently discarded) through the pypto dispatch layers:

1. ``ChipWorker._run_chip`` forwards the ``RunTiming`` from ``self._impl.run``.
2. ``execute_on_device`` returns it on both the one-shot ``Worker`` path and the
   active-``ChipWorker`` reuse path (and still calls ``close()`` on the one-shot
   worker before returning).
3. ``execute_compiled`` returns the ``RunTiming`` it gets back from
   ``execute_on_device`` (after the post-run DFX collection step).
"""

from unittest.mock import MagicMock, patch

import pytest

# ``device_runner`` and ``task_interface`` eagerly import the optional
# ``simpler`` runtime package, so these plumbing tests need simpler installed.
_simpler_required = pytest.importorskip(
    "simpler", reason="RunTiming plumbing tests require the simpler package"
)


# A plain stand-in for the simpler ``RunTiming`` nanobind type — identity is all
# the plumbing asserts care about, so a sentinel object suffices.
_TIMING_SENTINEL = object()


# ---------------------------------------------------------------------------
# ChipWorker._run_chip — forwards the RunTiming from the C++ impl
# ---------------------------------------------------------------------------


def test_run_chip_forwards_impl_run_result():
    """``_run_chip`` must return whatever ``self._impl.run`` returns."""
    from pypto.runtime.worker import ChipWorker  # noqa: PLC0415

    worker = ChipWorker.__new__(ChipWorker)  # bypass __init__/device setup
    worker._initialized = True
    worker._cid_cache = {}
    worker._impl = MagicMock(name="impl")
    worker._impl.register.return_value = 7
    worker._impl.run.return_value = _TIMING_SENTINEL

    result = worker._run_chip(
        MagicMock(name="chip_callable"), MagicMock(name="orch_args"), MagicMock(name="cfg")
    )

    assert result is _TIMING_SENTINEL
    worker._impl.run.assert_called_once()


# ---------------------------------------------------------------------------
# execute_on_device — returns the RunTiming on both dispatch paths
# ---------------------------------------------------------------------------


def test_execute_on_device_returns_timing_one_shot_path():
    """One-shot ``Worker`` path returns ``worker.run``'s timing after close()."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    fake_worker = MagicMock(name="worker_instance")
    fake_worker.run.return_value = _TIMING_SENTINEL
    fake_worker_cls = MagicMock(name="WorkerClass", return_value=fake_worker)

    with (
        patch("pypto.runtime.device_runner.Worker", fake_worker_cls),
        # No active ChipWorker → take the one-shot path.
        patch("pypto.runtime.worker.Worker.current", return_value=None),
    ):
        timing = execute_on_device(
            MagicMock(name="chip_callable"),
            MagicMock(name="orch_args"),
            platform="a2a3sim",
            runtime_name="host_build_graph",
            device_id=0,
        )

    assert timing is _TIMING_SENTINEL
    # close() must still run before the timing is returned.
    fake_worker.close.assert_called_once()


def test_execute_on_device_returns_timing_reuse_path():
    """Active-ChipWorker reuse path returns ``_run_chip``'s timing."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    active_worker = MagicMock(name="active_chip_worker")
    active_worker._run_chip.return_value = _TIMING_SENTINEL

    with patch("pypto.runtime.worker.Worker.current", return_value=active_worker):
        timing = execute_on_device(
            MagicMock(name="chip_callable"),
            MagicMock(name="orch_args"),
            platform="a2a3sim",
            runtime_name="host_build_graph",
            device_id=0,
        )

    assert timing is _TIMING_SENTINEL
    active_worker._run_chip.assert_called_once()


# ---------------------------------------------------------------------------
# execute_compiled — propagates the RunTiming up to its caller
# ---------------------------------------------------------------------------


def test_execute_compiled_returns_timing(tmp_path):
    """``execute_compiled`` must return the timing from ``execute_on_device``."""

    def _fake_execute_on_device(*_args, **_kwargs):
        return _TIMING_SENTINEL

    with (
        patch("pypto.runtime.runner._patch_orchestration_headers"),
        patch(
            "pypto.runtime.device_runner.compile_and_assemble",
            return_value=(MagicMock(name="chip_callable"), "host_build_graph", {}),
        ),
        patch(
            "pypto.runtime.device_runner.execute_on_device",
            side_effect=_fake_execute_on_device,
        ),
        patch("pypto.runtime.device_runner.ChipStorageTaskArgs", return_value=MagicMock(name="orch_args")),
    ):
        from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

        timing = execute_compiled(tmp_path, [], platform="a2a3sim", device_id=0)

    assert timing is _TIMING_SENTINEL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
