# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for ``block_dim`` plumbing (issue #1173).

Verifies that:

1. ``execute_on_device`` no longer hard-codes ``block_dim=24`` /
   ``aicpu_thread_num=4`` on the ``ChipCallConfig`` and instead leaves
   those fields untouched when the caller passes ``None``.
2. Explicit values still flow through to ``ChipCallConfig``.
3. ``execute_compiled`` reads ``RUNTIME_CONFIG`` defaults and forwards
   them to ``execute_on_device``, with caller-supplied arguments
   overriding the baked-in values.
4. ``_generate_config_file`` only emits ``"block_dim"`` when explicitly
   requested.
"""

from unittest.mock import MagicMock, patch

import pytest

# ``device_runner`` and ``task_interface`` eagerly import the optional
# ``simpler`` runtime package, so most plumbing tests need simpler installed.
# The ``_generate_config_file`` tests live in a separate module
# (``test_kernel_config_block_dim.py``) so they can run in environments
# without simpler.
_simpler_required = pytest.importorskip(
    "simpler", reason="block_dim plumbing tests require the simpler package"
)


# ---------------------------------------------------------------------------
# execute_on_device — config-assignment policy
# ---------------------------------------------------------------------------


class _SpyConfig:
    """Stand-in for ``CallConfig`` that records every attribute assignment.

    Behaves like a plain object — attributes not explicitly set are simply
    absent, so ``hasattr`` is a clean way to assert "the caller did not
    write this field".
    """

    def __init__(self) -> None:
        self._writes: list[str] = []

    def __setattr__(self, name: str, value: object) -> None:
        if name != "_writes":
            self._writes.append(name)
        object.__setattr__(self, name, value)


@pytest.fixture
def patched_execute_on_device():
    """Patch the simpler-side dependencies of ``execute_on_device``.

    Yields the captured ``_SpyConfig`` instance so tests can inspect which
    fields the runner assigned.
    """
    spy_cfg = _SpyConfig()

    fake_worker_cls = MagicMock(name="WorkerClass")
    fake_worker = MagicMock(name="worker_instance")
    fake_worker_cls.return_value = fake_worker

    with (
        patch("pypto.runtime.device_runner.CallConfig", return_value=spy_cfg),
        patch("pypto.runtime.device_runner.Worker", fake_worker_cls),
        # Disable active-Worker lookup so the one-shot path runs.
        patch("pypto.runtime.worker.Worker.current", return_value=None),
    ):
        yield spy_cfg


def test_execute_on_device_skips_config_assignment_when_block_dim_none(patched_execute_on_device):
    """``block_dim=None`` must leave ``ChipCallConfig.block_dim`` untouched."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    execute_on_device(
        MagicMock(name="chip_callable"),
        MagicMock(name="orch_args"),
        platform="a2a3sim",
        runtime_name="host_build_graph",
        device_id=0,
        block_dim=None,
        aicpu_thread_num=None,
    )

    cfg = patched_execute_on_device
    # Neither block_dim nor aicpu_thread_num should have been written.
    assert "block_dim" not in cfg._writes
    assert "aicpu_thread_num" not in cfg._writes
    # Profiling flag is always set — sanity-check the spy is active.
    assert "enable_l2_swimlane" in cfg._writes


def test_execute_on_device_sets_block_dim_when_provided(patched_execute_on_device):
    """An explicit ``block_dim`` must flow through to ``ChipCallConfig``."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    execute_on_device(
        MagicMock(name="chip_callable"),
        MagicMock(name="orch_args"),
        platform="a2a3sim",
        runtime_name="host_build_graph",
        device_id=0,
        block_dim=8,
        aicpu_thread_num=3,
    )

    cfg = patched_execute_on_device
    assert cfg.block_dim == 8
    assert cfg.aicpu_thread_num == 3


# ---------------------------------------------------------------------------
# execute_compiled — RUNTIME_CONFIG fallback / caller override precedence
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_execute_compiled(tmp_path):
    """Patch every external dependency of ``execute_compiled`` so it runs
    without a device and captures the ``execute_on_device`` call kwargs.
    """
    captured: dict = {"calls": []}

    def _fake_execute_on_device(*args, **kwargs) -> None:
        captured["calls"].append(kwargs)

    chip_args = MagicMock(name="ChipStorageTaskArgs_instance")

    def _compile_and_assemble(_work_dir, _platform, _commit=None):
        # The third element is the freshly-loaded RUNTIME_CONFIG dict; tests
        # parametrize it via ``patched_execute_compiled_runtime_config``.
        return (
            MagicMock(name="chip_callable"),
            "host_build_graph",
            captured.get("runtime_config", {}),
        )

    with (
        patch("pypto.runtime.runner._patch_orchestration_headers"),
        patch(
            "pypto.runtime.device_runner.compile_and_assemble",
            side_effect=_compile_and_assemble,
        ),
        patch("pypto.runtime.device_runner.execute_on_device", side_effect=_fake_execute_on_device),
        patch("pypto.runtime.device_runner.ChipStorageTaskArgs", return_value=chip_args),
    ):
        yield captured, tmp_path


def test_execute_compiled_reads_block_dim_from_runtime_config(patched_execute_compiled):
    """When the caller omits ``block_dim``, fall back to ``RUNTIME_CONFIG``."""
    captured, tmp = patched_execute_compiled
    captured["runtime_config"] = {
        "runtime": "host_build_graph",
        "block_dim": 12,
        "aicpu_thread_num": 4,
    }

    from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

    execute_compiled(tmp, [], platform="a2a3sim", device_id=0)

    assert len(captured["calls"]) == 1
    call_kwargs = captured["calls"][0]
    assert call_kwargs["block_dim"] == 12
    assert call_kwargs["aicpu_thread_num"] == 4


def test_execute_compiled_caller_override_wins(patched_execute_compiled):
    """Explicit caller arg must beat the ``RUNTIME_CONFIG`` default."""
    captured, tmp = patched_execute_compiled
    captured["runtime_config"] = {
        "runtime": "host_build_graph",
        "block_dim": 24,
        "aicpu_thread_num": 4,
    }

    from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

    execute_compiled(tmp, [], platform="a2a3sim", device_id=0, block_dim=8)

    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["block_dim"] == 8
    # aicpu_thread_num not overridden by caller → falls back to RUNTIME_CONFIG.
    assert captured["calls"][0]["aicpu_thread_num"] == 4


def test_execute_compiled_no_runtime_config_block_dim_forwards_none(patched_execute_compiled):
    """No ``block_dim`` anywhere → ``execute_on_device`` gets ``None``,
    letting simpler's own ``ChipCallConfig`` default apply.
    """
    captured, tmp = patched_execute_compiled
    captured["runtime_config"] = {"runtime": "host_build_graph", "aicpu_thread_num": 4}

    from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

    execute_compiled(tmp, [], platform="a2a3sim", device_id=0)

    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["block_dim"] is None
    assert captured["calls"][0]["aicpu_thread_num"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
