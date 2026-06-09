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
4. ``CompiledProgram.__call__`` / ``_SubChipCallable.__call__`` surface it on
   ``last_run_timing`` while still returning outputs/None.
5. ``JITFunction.__call__`` forwards the dispatched program's
   ``last_run_timing`` so a plain ``kernel(*args, config=...)`` can read timing.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

# ``device_runner`` / ``task_interface`` / ``worker`` eagerly import the optional
# ``simpler`` runtime package, so the dispatch-chain tests (1–3) need it. The
# CompiledProgram / JITFunction tests (4–5) mock ``_invoke_compiled`` and never
# touch simpler, so they run unconditionally.
requires_simpler = pytest.mark.skipif(
    importlib.util.find_spec("simpler") is None,
    reason="RunTiming dispatch-chain tests require the simpler package",
)


# A plain stand-in for the simpler ``RunTiming`` nanobind type — identity is all
# the plumbing asserts care about, so a sentinel object suffices.
_TIMING_SENTINEL = object()


# ---------------------------------------------------------------------------
# ChipWorker._run_chip — forwards the RunTiming from the C++ impl
# ---------------------------------------------------------------------------


@requires_simpler
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


@requires_simpler
def test_execute_on_device_returns_timing_one_shot_path():
    """One-shot ``Worker`` path returns ``worker.run``'s timing after close()."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    fake_worker = MagicMock(name="worker_instance")
    fake_worker.run.return_value = _TIMING_SENTINEL
    fake_worker_cls = MagicMock(name="WorkerClass", return_value=fake_worker)

    with (
        patch("pypto.runtime.device_runner.Worker", fake_worker_cls),
        # No active ChipWorker → take the one-shot path. ``execute_on_device``
        # resolves the active worker via ``ChipWorker.current`` (imported as
        # ``_PyptoWorker``), so that is the attribute to stub.
        patch("pypto.runtime.worker.ChipWorker.current", return_value=None),
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


@requires_simpler
def test_execute_on_device_returns_timing_reuse_path():
    """Active-ChipWorker reuse path returns ``_run_chip``'s timing."""
    from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

    active_worker = MagicMock(name="active_chip_worker")
    active_worker._run_chip.return_value = _TIMING_SENTINEL

    with patch("pypto.runtime.worker.ChipWorker.current", return_value=active_worker):
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


@requires_simpler
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


# ---------------------------------------------------------------------------
# _execute_on_device (harness-shared path) — returns the RunTiming so the
# golden harness can surface device/host wall on RunResult (issue #1679)
# ---------------------------------------------------------------------------


@requires_simpler
def test_internal_execute_on_device_returns_timing(tmp_path):
    """``runner._execute_on_device`` returns ``execute_on_device``'s timing.

    ``test_runner.py`` (the golden harness) builds ``RunResult`` from this
    helper, so the timing must reach the caller rather than being discarded.
    """
    import torch  # noqa: PLC0415

    golden = tmp_path / "golden.py"
    golden.write_text(
        "import torch\n"
        "__outputs__ = ['c']\n"
        "RTOL = 1e-5\n"
        "ATOL = 1e-5\n"
        "def generate_inputs(params):\n"
        "    return [('a', torch.zeros(1)), ('c', torch.zeros(1))]\n"
        "def compute_golden(tensors, params):\n"
        "    pass\n",
        encoding="utf-8",
    )

    from pypto.runtime.runner import _execute_on_device  # noqa: PLC0415

    with (
        patch(
            "pypto.runtime.device_runner.build_orch_args_from_inputs",
            return_value=(MagicMock(name="orch_args"), {}, {}, {"c": torch.zeros(1)}),
        ),
        patch(
            "pypto.runtime.device_runner.execute_on_device",
            return_value=_TIMING_SENTINEL,
        ),
        patch("pypto.runtime.device_runner.validate_golden"),
    ):
        timing = _execute_on_device(
            tmp_path,
            golden,
            MagicMock(name="chip_callable"),
            "host_build_graph",
            "a2a3sim",
            0,
        )

    assert timing is _TIMING_SENTINEL


@requires_simpler
def test_runtiming_reexported_from_package_root():
    """``RunTiming`` resolves from the ``pypto.runtime`` package root (issue #1679).

    Users read ``RunTiming`` off ``last_run_timing`` / ``execute_compiled``, so
    it must be discoverable from the package root — not only the deep
    ``pypto.runtime.task_interface`` path. The re-export stays lazy (via
    ``__getattr__``) so importing ``pypto.runtime`` does not pull in simpler.
    """
    import pypto.runtime as rt  # noqa: PLC0415
    from pypto.runtime.task_interface import (  # noqa: PLC0415
        RunTiming as _DeepRunTiming,  # pyright: ignore[reportAttributeAccessIssue]
    )

    assert rt.RunTiming is _DeepRunTiming
    assert "RunTiming" in rt.__all__


# ---------------------------------------------------------------------------
# CompiledProgram / _SubChipCallable — store timing on last_run_timing
# ---------------------------------------------------------------------------


def test_compiled_program_call_stores_last_run_timing():
    """``CompiledProgram.__call__`` returns outputs and stores the timing."""
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415

    cp = CompiledProgram.__new__(CompiledProgram)  # bypass __init__/codegen layout
    cp._sub_chip_dirs = {}
    cp._output_dir = Path("out")
    cp._platform = "a2a3sim"
    cp.last_run_timing = None

    sentinel_out = object()
    with (
        patch.object(CompiledProgram, "_get_metadata", return_value=([], [], [])),
        patch(
            "pypto.ir.compiled_program._invoke_compiled",
            return_value=(sentinel_out, _TIMING_SENTINEL),
        ),
    ):
        result = cp()

    assert result is sentinel_out
    assert cp.last_run_timing is _TIMING_SENTINEL


def test_sub_chip_callable_call_stores_last_run_timing():
    """``_SubChipCallable.__call__`` returns outputs and stores the timing."""
    from pypto.ir.compiled_program import _SubChipCallable  # noqa: PLC0415

    sub = _SubChipCallable.__new__(_SubChipCallable)  # bypass __init__
    sub._name = "orch0"
    sub._output_dir = Path("out")
    sub._platform = "a2a3sim"
    sub._param_infos = []
    sub._output_indices = []
    sub._return_types = []
    sub.last_run_timing = None

    sentinel_out = object()
    with patch(
        "pypto.ir.compiled_program._invoke_compiled",
        return_value=(sentinel_out, _TIMING_SENTINEL),
    ):
        result = sub()

    assert result is sentinel_out
    assert sub.last_run_timing is _TIMING_SENTINEL


# ---------------------------------------------------------------------------
# JITFunction — forwards the dispatched CompiledProgram's last_run_timing
# ---------------------------------------------------------------------------


def test_jit_function_call_forwards_last_run_timing():
    """``JITFunction.__call__`` mirrors the dispatched program's timing."""
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415

    jf = JITFunction.__new__(JITFunction)  # bypass __init__/specialization
    jf.last_run_timing = None

    sentinel_out = object()
    compiled = MagicMock(name="CompiledProgram", return_value=sentinel_out)
    compiled.last_run_timing = _TIMING_SENTINEL

    with patch.object(jf, "_resolve_compiled", return_value=(compiled, (), None)):
        result = jf()

    assert result is sentinel_out
    assert jf.last_run_timing is _TIMING_SENTINEL
    compiled.assert_called_once()


# ---------------------------------------------------------------------------
# L3 distributed — _dispatch / DistributedCompiledProgram / DistributedWorker
# ---------------------------------------------------------------------------


def test_dispatch_returns_worker_run_timing():
    """``_dispatch`` must return the ``RunTiming`` from ``w.run``."""
    from pypto.runtime.distributed_runner import _dispatch  # noqa: PLC0415

    w = MagicMock(name="worker")
    w.run.return_value = _TIMING_SENTINEL

    timing = _dispatch(
        w,
        MagicMock(name="entry_fn"),
        tensors={},
        chip_cids={},
        sub_ids={},
        call_config=MagicMock(name="call_config"),
        device_nums=1,
    )

    assert timing is _TIMING_SENTINEL
    w.run.assert_called_once()


def test_distributed_compiled_program_call_stores_last_run_timing():
    """``DistributedCompiledProgram.__call__`` stores execute_distributed's timing."""
    import torch  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

    dcp = DistributedCompiledProgram.__new__(DistributedCompiledProgram)  # bypass __init__
    dcp.last_run_timing = None

    info = SimpleNamespace(name="x")
    with (
        patch.object(DistributedCompiledProgram, "_get_metadata", return_value=([info], [], [])),
        patch(
            "pypto.runtime.distributed_runner.execute_distributed",
            return_value=_TIMING_SENTINEL,
        ),
    ):
        # One in-place tensor param → return_style is False, call returns None.
        result = dcp(torch.zeros(2))

    assert result is None
    assert dcp.last_run_timing is _TIMING_SENTINEL


def test_distributed_worker_call_stores_last_run_timing():
    """``DistributedWorker.__call__`` stores the ``_dispatch`` timing."""
    import torch  # noqa: PLC0415
    from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

    rt = DistributedWorker.__new__(DistributedWorker)  # bypass __init__/Worker setup
    rt._closed = False
    # Minimal stand-in: __call__ only reads ``.name`` off each param info.
    rt._param_infos = cast(Any, [SimpleNamespace(name="x")])
    rt._base_tensors = {}
    rt._w = MagicMock(name="worker")
    rt._entry_fn = MagicMock(name="entry_fn")
    rt._chip_cids = {}
    rt._sub_ids = {}
    rt._call_config = MagicMock(name="call_config")
    rt.dc = cast(Any, SimpleNamespace(device_ids=[0]))
    rt.last_run_timing = None

    shared = torch.zeros(2).share_memory_()  # DistributedWorker rejects non-shared host tensors
    with patch(
        "pypto.runtime.distributed_runner._dispatch",
        return_value=_TIMING_SENTINEL,
    ):
        rt(shared)

    assert rt.last_run_timing is _TIMING_SENTINEL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
