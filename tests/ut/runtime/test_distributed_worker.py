# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``DistributedWorker`` (the ``prepare()`` reuse handle).

Runs without a device or the ``simpler`` package by patching the module-level
setup helpers in :mod:`pypto.runtime.distributed_runner`, so construction does
no real compile/fork. The tests cover both ordinary prepared dispatch and the
persistent contract: bounded asynchronous submission, retained per-program
domains, handle-owned input lifetimes, and complete cleanup before publication.
"""

import gc
import importlib.util
import json
import sys
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.ir.compiled_program import _ParamInfo
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import DeviceTensor
from pypto.runtime.bench import (
    _L3_SWIMLANE_GRAPH_BEGIN,
    _L3_SWIMLANE_GRAPH_END,
    _L3_SWIMLANE_TIMING_BEGIN,
    _L3_SWIMLANE_TIMING_END,
)
from pypto.runtime.distributed_runner import (
    DistributedRunHandle,
    DistributedWorker,
    _assemble_chip_callables,
    _clear_dfx_dispatch_dirs,
    _collect_l3_swimlane,
    _construct_worker,
    _make_call_config,
    _reset_dfx_dispatch_state,
    _submit_chip,
)
from pypto.runtime.runner import RunConfig


def _param(name: str, shape: list[int], direction: ParamDirection = ParamDirection.In) -> _ParamInfo:
    return _ParamInfo(name=name, direction=direction, shape=shape, dtype=DataType.FP32)


def _fake_compiled(param_infos, output_indices):
    """A minimal stand-in for DistributedCompiledProgram used by DistributedWorker."""
    compiled = MagicMock(name="DistributedCompiledProgram")
    compiled._get_metadata.return_value = (param_infos, output_indices, [])
    compiled._distributed_config = DistributedConfig()
    compiled.platform = "a2a3sim"
    return compiled


class _ImmediateNativeHandle:
    """Minimal Simpler RunHandle stand-in used by prepared-worker tests."""

    done = True

    def result(self, timeout=None):
        del timeout


class _ControlledNativeHandle:
    """RunHandle stand-in with an explicit terminal-completion gate."""

    def __init__(
        self,
        error: BaseException | None = None,
        on_result: Callable[[], None] | None = None,
    ) -> None:
        self._terminal = threading.Event()
        self.result_started = threading.Event()
        self.error = error
        self._on_result = on_result

    @property
    def done(self) -> bool:
        return self._terminal.is_set()

    def complete(self, error: BaseException | None = None) -> None:
        self.error = error
        self._terminal.set()

    def result(self, timeout=None):
        self.result_started.set()
        if not self._terminal.wait(timeout):
            raise TimeoutError("native handle timed out")
        if self._on_result is not None:
            self._on_result()
        if self.error is not None:
            raise self.error


@pytest.fixture
def patched_setup():
    """Patch every setup helper so DistributedWorker() does no real work.

    Yields a dict of the mocks so individual tests can assert call counts.
    The worker mock records malloc/copy_to/free for alloc_tensor checks.
    """
    worker = MagicMock(name="Worker(level=3)")
    worker.chip_contexts = []
    worker._live_domains = {}
    # Device-memory ops route through the Orchestrator facade (worker._orch).
    worker._orch.malloc.return_value = 0xDEAD0000
    worker.submit.side_effect = lambda fn: (fn(worker._orch, None, None), _ImmediateNativeHandle())[1]

    mod = "pypto.runtime.distributed_runner"
    chip_callables = ({"chip_orch": object()}, "rt_name", False)
    with (
        patch(f"{mod}._assemble_chip_callables", return_value=chip_callables) as assemble,
        patch(f"{mod}._load_orch_entry", return_value=(MagicMock(name="entry_fn"), None)) as load_entry,
        patch(f"{mod}._load_sub_worker_fns", return_value={}) as load_subs,
        patch(f"{mod}._load_required_callbacks", return_value=set()) as load_required,
        patch(f"{mod}._construct_worker", return_value=worker) as construct,
        patch(f"{mod}._register_callables", return_value=({}, {"chip_orch": 0})) as register,
        patch(f"{mod}._make_call_config", return_value=MagicMock(name="CallConfig")) as make_call_config,
        patch(f"{mod}._dispatch") as dispatch,
        patch(f"{mod}._submit_dispatch", return_value=_ImmediateNativeHandle()) as submit_dispatch,
    ):
        yield {
            "worker": worker,
            "assemble": assemble,
            "load_entry": load_entry,
            "load_subs": load_subs,
            "load_required": load_required,
            "construct": construct,
            "register": register,
            "make_call_config": make_call_config,
            "dispatch": dispatch,
            "submit_dispatch": submit_dispatch,
        }


class TestSetupOnce:
    def test_setup_runs_once_dispatch_many(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])

        rt = DistributedWorker(compiled)
        # All expensive setup happened exactly once at construction.
        m["assemble"].assert_called_once()
        m["construct"].assert_called_once()
        m["register"].assert_called_once()
        m["worker"].init.assert_called_once()
        # Simpler's public init owns eager hierarchy startup.
        m["worker"]._start_hierarchical.assert_not_called()

        a = DeviceTensor(0x1000, (128, 128), torch.float32)
        b = DeviceTensor(0x2000, (128, 128), torch.float32)
        rt(a, b)
        rt(a, b)
        rt(a, b)

        # Setup still once; dispatch ran per call.
        assert m["submit_dispatch"].call_count == 3
        m["assemble"].assert_called_once()
        m["construct"].assert_called_once()
        assert m["worker"].init.call_count == 1
        rt.close()


class TestAsyncDispatchHandle:
    def test_submit_returns_handle_and_retires_frame_on_result(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)

        handle = rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))

        assert isinstance(handle, DistributedRunHandle)
        assert handle.done is False
        assert len(rt._active_dispatch_handles) == 1
        native.complete()
        handle.result()
        assert handle.done is True
        assert not rt._active_dispatch_handles
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_timeout_keeps_frame_owned_until_later_completion(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        handle = rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))

        with pytest.raises(TimeoutError):
            handle.result(timeout=0.0)
        assert handle.done is False
        assert any(frame.in_use for frame in rt._dispatch_frames)
        with pytest.raises(ValueError, match="non-negative finite"):
            handle.result(timeout=-1.0)

        native.complete()
        handle.result()
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_handle_keeps_input_alive_until_completion(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        arg = torch.zeros((16, 16), dtype=torch.float32).share_memory_()
        arg_ref = weakref.ref(arg)

        handle = rt.submit(compiled, arg)
        m["submit_dispatch"].reset_mock()
        del arg
        gc.collect()
        assert arg_ref() is not None

        native.complete()
        handle.result()
        gc.collect()
        assert arg_ref() is None
        rt.close()

    def test_third_submit_drains_oldest_before_allocating_metadata(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        natives = [_ControlledNativeHandle() for _ in range(3)]
        m["submit_dispatch"].side_effect = natives
        rt = DistributedWorker(compiled)
        arg = DeviceTensor(0x1000, (16, 16), torch.float32)
        first = rt.submit(compiled, arg)
        second = rt.submit(compiled, arg)
        assert m["make_call_config"].call_count == 3  # prepare + two accepted dispatches

        third_result: list[DistributedRunHandle] = []
        third_error: list[BaseException] = []

        def submit_third() -> None:
            try:
                third_result.append(rt.submit(compiled, arg))
            except BaseException as exc:  # noqa: BLE001 - asserted below
                third_error.append(exc)

        caller = threading.Thread(target=submit_third)
        caller.start()
        assert natives[0].result_started.wait(timeout=2)
        # Backpressure happens before a CallConfig or frame-local tensor map is
        # created for the third dispatch.
        assert m["make_call_config"].call_count == 3
        assert not third_result

        natives[0].complete()
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert not third_error
        assert len(third_result) == 1
        assert first.done is True
        assert m["make_call_config"].call_count == 4

        natives[1].complete()
        natives[2].complete()
        second.result()
        third_result[0].result()
        rt.close()

    def test_failed_handle_recycles_frame_and_caches_error(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        failed = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = failed
        rt = DistributedWorker(compiled)
        handle = rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))
        failed.complete(RuntimeError("dispatch failed"))

        with pytest.raises(RuntimeError, match="dispatch failed"):
            handle.result()
        with pytest.raises(RuntimeError, match="dispatch failed"):
            handle.result()
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_close_drains_outstanding_handle_before_worker_close(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))

        closed = threading.Event()
        closer = threading.Thread(target=lambda: (rt.close(), closed.set()))
        closer.start()
        assert native.result_started.wait(timeout=2)
        assert not closed.is_set()
        m["worker"].close.assert_not_called()

        native.complete()
        closer.join(timeout=2)
        assert closed.is_set()
        m["worker"].close.assert_called_once_with()
        with pytest.raises(RuntimeError, match="after close"):
            rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))


class TestPerTaskRingSizing:
    """A per-dispatch ``RunConfig`` sizes that dispatch's runtime ring buffers.

    ``_make_call_config`` runs once at construction to build the program's
    prewarm baseline. Every accepted asynchronous dispatch receives a fresh
    snapshot; a ``RunConfig`` adds that dispatch's overrides.
    """

    # ``_submit_dispatch(w, entry_fn, tensors, chip_cids, sub_ids, call_config, ...)``
    _CALL_CONFIG_ARG = 5

    def test_no_config_snapshots_prepared_baseline(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        # Construction builds the baseline exactly once.
        assert m["make_call_config"].call_count == 1
        baseline = m["make_call_config"].return_value
        fresh = MagicMock(name="FreshCallConfig")
        m["make_call_config"].return_value = fresh

        rt(DeviceTensor(0x1000, (16, 16), torch.float32))

        assert m["make_call_config"].call_count == 2
        assert fresh is not baseline
        assert m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is fresh
        rt.close()

    def test_per_dispatch_config_rebuilds_call_config(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        assert m["make_call_config"].call_count == 1  # baseline at construction

        rc = RunConfig(platform="a2a3sim", ring_task_window=64, ring_heap=4 * 1024 * 1024)
        rt(DeviceTensor(0x1000, (16, 16), torch.float32), config=rc)

        # A per-dispatch config rebuilds from (program DistributedConfig, rc).
        assert m["make_call_config"].call_count == 2
        rebuild = m["make_call_config"].call_args
        assert rebuild.args[0] is compiled._distributed_config
        assert rebuild.args[1] is rc
        # The freshly built config (not None) is what reaches _dispatch.
        assert (
            m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is m["make_call_config"].return_value
        )
        rt.close()

    def test_run_method_forwards_per_dispatch_config(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)

        rc = RunConfig(platform="a2a3sim", ring_dep_pool=256)
        rt.run(compiled, DeviceTensor(0x1000, (16, 16), torch.float32), config=rc)

        # rt.run(...) honors the same per-dispatch ring sizing as rt(...).
        assert m["make_call_config"].call_count == 2
        assert m["make_call_config"].call_args.args[1] is rc
        rt.close()


class TestPreparedSwimlaneTwoPass:
    """Prepared onboard L3 captures deps, then measures without dep_gen."""

    _CALL_CONFIG_ARG = 5

    def test_onboard_reuses_worker_for_graph_then_clean_timing(self, patched_setup, tmp_path, capsys):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        # Ignore the baseline config constructed during prepare(); this test
        # observes the two configs built for the caller-visible dispatch.
        m["make_call_config"].reset_mock()
        deps_call_config = MagicMock(name="DepsCallConfig")
        timing_call_config = MagicMock(name="TimingCallConfig")
        m["make_call_config"].side_effect = [deps_call_config, timing_call_config]
        events: list[str] = []
        m["submit_dispatch"].side_effect = lambda *args: (
            events.append("deps" if args[self._CALL_CONFIG_ARG] is deps_call_config else "timing"),
            _ImmediateNativeHandle(),
        )[1]

        run_config = RunConfig(
            platform="a2a3",
            enable_l2_swimlane=1,  # pyright: ignore[reportArgumentType]
            enable_pmu=3,
            enable_scope_stats=True,
            enable_dump_args=2,
        )
        with (
            patch(
                "pypto.runtime.distributed_runner._clear_dfx_dispatch_dirs",
                side_effect=lambda _path: events.append("clear"),
            ) as clear,
            patch(
                "pypto.runtime.distributed_runner._collect_l3_swimlane",
                side_effect=lambda _output, _platform: events.append("collect"),
            ) as collect,
        ):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32), config=run_config)

        assert events == ["clear", "deps", "timing", "collect"]
        assert m["submit_dispatch"].call_count == 2
        assert all(call.args[0] is m["worker"] for call in m["submit_dispatch"].call_args_list)
        assert m["construct"].call_count == 1
        assert m["worker"].init.call_count == 1
        clear.assert_called_once_with(tmp_path / "dfx_outputs")
        collect.assert_called_once_with(tmp_path, "a2a3")

        assert m["make_call_config"].call_count == 2
        deps_build, timing_build = m["make_call_config"].call_args_list
        deps_config = deps_build.args[1]
        assert deps_config.enable_l2_swimlane is False
        assert deps_config.enable_dep_gen is True
        assert deps_config.enable_pmu == 0
        assert deps_config.enable_scope_stats is False
        assert deps_config.enable_dump_args == 0

        timing_config = timing_build.args[1]
        assert timing_config.enable_l2_swimlane == 1
        assert timing_config.enable_dep_gen is False
        assert timing_config.enable_pmu == 3
        assert timing_config.enable_scope_stats is True
        assert timing_config.enable_dump_args == 2
        assert timing_build.kwargs["co_enable_swimlane_dep_gen"] is False
        captured = capsys.readouterr()
        assert [line for line in captured.err.splitlines() if "l3_swimlane_pass=" in line] == [
            _L3_SWIMLANE_GRAPH_BEGIN,
            _L3_SWIMLANE_GRAPH_END,
            _L3_SWIMLANE_TIMING_BEGIN,
            _L3_SWIMLANE_TIMING_END,
        ]
        rt.close()

    def test_simulator_keeps_single_pass(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        m["make_call_config"].reset_mock()
        call_config = MagicMock(name="SimCallConfig")
        m["make_call_config"].return_value = call_config
        run_config = RunConfig(
            platform="a2a3sim",
            enable_l2_swimlane=1,  # pyright: ignore[reportArgumentType]
        )
        with patch("pypto.runtime.distributed_runner._collect_l3_swimlane") as collect:
            rt(DeviceTensor(0x1000, (16, 16), torch.float32), config=run_config)

        m["submit_dispatch"].assert_called_once()
        assert m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is call_config
        m["make_call_config"].assert_called_once_with(
            compiled._distributed_config,
            run_config,
            dfx_base=tmp_path / "dfx_outputs",
        )
        collect.assert_called_once_with(tmp_path, "a2a3sim")
        rt.close()

    def test_dep_gen_without_swimlane_keeps_single_pass(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        m["make_call_config"].reset_mock()
        run_config = RunConfig(platform="a2a3", enable_dep_gen=True)
        with patch("pypto.runtime.distributed_runner._collect_l3_swimlane") as collect:
            rt(DeviceTensor(0x1000, (16, 16), torch.float32), config=run_config)

        m["submit_dispatch"].assert_called_once()
        m["make_call_config"].assert_called_once_with(
            compiled._distributed_config,
            run_config,
            dfx_base=tmp_path / "dfx_outputs",
        )
        collect.assert_not_called()
        rt.close()

    def test_persistent_route_waits_for_graph_then_timing_requests(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        # Exercise _dispatch_prepared's persistent branch without starting a
        # background thread: the two-pass fallback still fences both requests.
        rt._persistent = True
        m["make_call_config"].reset_mock()
        deps_call_config = MagicMock(name="DepsCallConfig")
        timing_call_config = MagicMock(name="TimingCallConfig")
        m["make_call_config"].side_effect = [deps_call_config, timing_call_config]
        with (
            patch.object(
                rt,
                "_submit_persistent",
                return_value=_ImmediateNativeHandle(),
            ) as submit_persistent,
            patch("pypto.runtime.distributed_runner._collect_l3_swimlane"),
        ):
            rt(
                DeviceTensor(0x1000, (16, 16), torch.float32),
                config=RunConfig(platform="a2a3", enable_l2_swimlane=True),
            )

        assert [call.args[2] for call in submit_persistent.call_args_list] == [
            deps_call_config,
            timing_call_config,
        ]
        rt.close()


class TestArenaPrewarm:
    """``init`` prewarms the prebuilt runtime-arena cache with the ring sizing the
    first dispatch will use, so the ~800ms cold build lands at prepare() time
    rather than inside the first (usually timed) dispatch.
    """

    def test_prewarms_with_prepared_baseline_when_no_config(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)

        # No worker RunConfig → the program's baseline CallConfig (the same one
        # config-less dispatches reuse) is what init prewarms with; no rebuild.
        assert m["make_call_config"].call_count == 1
        assert m["worker"].init.call_args.kwargs["prewarm_config"] is m["make_call_config"].return_value
        rt.close()

    def test_prewarms_with_worker_config_ring_sizing(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rc = RunConfig(platform="a2a3sim", ring_heap=4 * 1024 * 1024)

        rt = DistributedWorker(compiled, rc)

        # A worker RunConfig builds a second CallConfig from (program
        # DistributedConfig, rc) — the same construction a dispatch with rc uses,
        # so the prewarmed arena's sizing key matches that dispatch's.
        assert m["make_call_config"].call_count == 2
        prewarm_build = m["make_call_config"].call_args
        assert prewarm_build.args[0] is compiled._distributed_config
        assert prewarm_build.args[1] is rc
        assert m["worker"].init.call_args.kwargs["prewarm_config"] is m["make_call_config"].return_value
        rt.close()


class TestPerCallValidation:
    def test_accepts_device_tensor(self, patched_setup):
        submitted_tensors: dict[str, Any] = {}

        def submit_dispatch(*args):
            submitted_tensors.update(args[2])
            return _ImmediateNativeHandle()

        patched_setup["submit_dispatch"].side_effect = submit_dispatch
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        rt(DeviceTensor(0x1000, (128, 128), torch.float32), DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["submit_dispatch"].assert_called_once()
        assert set(submitted_tensors) == {"a", "b"}
        rt.close()

    def test_accepts_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        host_a = torch.zeros(128, 128, dtype=torch.float32).share_memory_()
        rt(host_a, DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_rejects_non_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="shared memory"):
            rt(torch.zeros(128, 128), DeviceTensor(0x2000, (128, 128), torch.float32))
        rt.close()

    def test_releasing_registered_tensors_disables_later_uploads(self, patched_setup):
        compiled = _fake_compiled([_param("weight", [4, 4])], [])
        weight = torch.zeros(4, 4, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[weight])

        rt.release_inherited_host_tensor_refs()
        rt.release_inherited_host_tensor_refs()

        assert rt._inherited_host_tensors == ()
        assert not rt._inherited_host_storage_ptrs
        with pytest.raises(ValueError, match="inherited_host_tensors"):
            rt.alloc_tensor(weight.shape, weight.dtype, init=weight)
        rt.close()

    def test_registered_tensor_still_requires_shared_memory_for_dispatch(self, patched_setup):
        compiled = _fake_compiled([_param("buffer", [128, 128])], [])
        buffer = torch.zeros(128, 128, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[buffer])

        with pytest.raises(TypeError, match="shared memory"):
            rt(buffer)

        rt.close()

    @pytest.mark.parametrize(
        ("weight", "expected_exception"),
        [
            (object(), TypeError),
            (torch.zeros(128, 128, dtype=torch.float32).t(), ValueError),
            (torch.empty(1, device="meta"), ValueError),
        ],
    )
    def test_rejects_invalid_prefork_tensor_registration(self, patched_setup, weight, expected_exception):
        compiled = _fake_compiled([_param("weight", [128, 128])], [])

        with pytest.raises(expected_exception, match=r"torch\.Tensor|contiguous CPU"):
            DistributedWorker(compiled, inherited_host_tensors=[weight])

    def test_scalar_param_forwarded_as_is(self, patched_setup):
        # Scalar params (shape=None, e.g. seq_len) bypass tensor validation and
        # are forwarded verbatim to the entry — common in serving dispatch.
        submitted_tensors: dict[str, Any] = {}

        def submit_dispatch(*args):
            submitted_tensors.update(args[2])
            return _ImmediateNativeHandle()

        patched_setup["submit_dispatch"].side_effect = submit_dispatch
        scalar = _ParamInfo(name="seq_len", direction=ParamDirection.In, shape=None, dtype=DataType.FP32)
        compiled = _fake_compiled([scalar, _param("kv", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt(7, DeviceTensor(0x1000, (16, 16), torch.float32))
        assert submitted_tensors["seq_len"] == 7
        rt.close()

    def test_rejects_wrong_arg_count(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="expects 2 arguments"):
            rt(DeviceTensor(0x1000, (128, 128), torch.float32))
        rt.close()

    def test_validates_device_tensor_shape(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="shape"):
            rt(
                DeviceTensor(0x1000, (64, 64), torch.float32),  # wrong shape
                DeviceTensor(0x2000, (128, 128), torch.float32),
            )
        rt.close()


class TestDeviceMemoryApi:
    def test_alloc_tensor_forwards_malloc_and_copy(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        # init must be a CPU, contiguous, shared-memory tensor (read by the
        # forked chip worker via the inherited mapping).
        host = torch.arange(256, dtype=torch.float32).view(16, 16).share_memory_()

        dev = rt.alloc_tensor((16, 16), torch.float32, init=host)

        assert isinstance(dev, DeviceTensor)
        assert dev.data_ptr == 0xDEAD0000
        assert dev.shape == (16, 16)
        # worker_id first for the Orchestrator facade; nbytes = 16*16*4.
        patched_setup["worker"]._orch.malloc.assert_called_once_with(0, 16 * 16 * 4)
        # copy_to(worker_id, dst=ptr, src=host.data_ptr(), nbytes) — no defensive copy.
        patched_setup["worker"]._orch.copy_to.assert_called_once_with(
            0, 0xDEAD0000, host.data_ptr(), 16 * 16 * 4
        )
        rt.close()

    def test_alloc_tensor_rejects_non_shared_init(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(ValueError, match="shared-memory"):
            rt.alloc_tensor((16, 16), torch.float32, init=torch.zeros(16, 16, dtype=torch.float32))
        # rolled back the malloc'd pointer.
        patched_setup["worker"]._orch.free.assert_called_once_with(0, 0xDEAD0000)
        rt.close()

    def test_alloc_tensor_rolls_back_on_copy_failure(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        patched_setup["worker"]._orch.copy_to.side_effect = RuntimeError("boom")
        host = torch.zeros(16, 16, dtype=torch.float32).share_memory_()

        with pytest.raises(RuntimeError, match="boom"):
            rt.alloc_tensor((16, 16), torch.float32, init=host)

        # malloc'd pointer is freed on the failure path.
        patched_setup["worker"]._orch.free.assert_called_once_with(0, 0xDEAD0000)
        rt.close()

    def test_alloc_tensor_forwards_nonzero_worker_id(self, patched_setup):
        # A non-default worker_id is supported: malloc is forwarded to that
        # worker (facade order is ``malloc(worker_id, nbytes)``) and the buffer
        # is tracked under (worker_id, ptr) for per-worker auto-free.
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        dev = rt.alloc_tensor((16, 16), torch.float32, worker_id=1)
        patched_setup["worker"]._orch.malloc.assert_called_once_with(1, 16 * 16 * 4)
        assert (1, dev.data_ptr) in rt._owned_tensors
        rt.free_tensor(dev, worker_id=1)
        patched_setup["worker"]._orch.free.assert_called_once_with(1, 0xDEAD0000)
        rt.close()


def _compiled_2cards():
    compiled = _fake_compiled([_param("b", [2, 4, 4])], [])
    compiled._distributed_config = DistributedConfig(device_ids=[0, 1])
    return compiled


class TestAllocStackedTensor:
    """``alloc_stacked_tensor`` uploads each leading-dim shard to its worker once."""

    def test_identity_uploads_shard_per_worker(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4).share_memory_()

        stacked = rt.alloc_stacked_tensor(host)  # default worker_ids = range(2)

        assert stacked.full_shape == (2, 4, 4)
        assert stacked.worker_ids == (0, 1)
        assert tuple(s.shape for s in stacked.shards) == ((4, 4), (4, 4))
        orch = patched_setup["worker"]._orch
        # shard 0 -> worker 0, shard 1 -> worker 1 (facade arg order worker_id first).
        nbytes = 4 * 4 * 4
        orch.malloc.assert_any_call(0, nbytes)
        orch.malloc.assert_any_call(1, nbytes)
        orch.copy_to.assert_any_call(0, 0xA000, host[0].contiguous().data_ptr(), nbytes)
        orch.copy_to.assert_any_call(1, 0xB000, host[1].contiguous().data_ptr(), nbytes)
        # Tracked per (worker_id, ptr) for auto-free.
        assert (0, 0xA000) in rt._owned_tensors
        assert (1, 0xB000) in rt._owned_tensors
        rt.close()

    def test_registered_inherited_storage_uploads_without_shared_memory(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        host = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
        rt = DistributedWorker(_compiled_2cards(), inherited_host_tensors=[host])

        stacked = rt.alloc_stacked_tensor(host)

        assert stacked.worker_ids == (0, 1)
        orch = patched_setup["worker"]._orch
        nbytes = 4 * 4 * 4
        orch.copy_to.assert_any_call(0, 0xA000, host[0].data_ptr(), nbytes)
        orch.copy_to.assert_any_call(1, 0xB000, host[1].data_ptr(), nbytes)
        rt.close()

    def test_permuted_worker_ids_place_shards(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        stacked = rt.alloc_stacked_tensor(host, worker_ids=[1, 0])

        assert stacked.worker_ids == (1, 0)
        orch = patched_setup["worker"]._orch
        nbytes = 4 * 4 * 4
        # shard 0 -> worker 1, shard 1 -> worker 0.
        orch.malloc.assert_any_call(1, nbytes)
        orch.malloc.assert_any_call(0, nbytes)
        assert (1, 0xA000) in rt._owned_tensors
        assert (0, 0xB000) in rt._owned_tensors
        rt.close()

    def test_free_stacked_tensor_releases_each_shard(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        stacked = rt.alloc_stacked_tensor(host, worker_ids=[1, 0])

        patched_setup["worker"]._orch.free.reset_mock()
        rt.free_stacked_tensor(stacked)

        orch = patched_setup["worker"]._orch
        orch.free.assert_any_call(1, 0xA000)
        orch.free.assert_any_call(0, 0xB000)
        assert (1, 0xA000) not in rt._owned_tensors
        assert (0, 0xB000) not in rt._owned_tensors
        rt.close()

    def test_close_auto_frees_stacked_shards(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        rt.alloc_stacked_tensor(host)  # leak — close() must release both shards

        patched_setup["worker"]._orch.free.reset_mock()
        rt.close()
        orch = patched_setup["worker"]._orch
        orch.free.assert_any_call(0, 0xA000)
        orch.free.assert_any_call(1, 0xB000)

    def test_worker_ids_out_of_range_rejected(self, patched_setup):
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="out of range"):
            rt.alloc_stacked_tensor(host, worker_ids=[0, 5])
        rt.close()

    def test_empty_leading_dim_rejected(self, patched_setup):
        # B == 0 must fail cleanly (before any malloc), not build an empty
        # StackedDeviceTensor that IndexErrors on .dtype / __repr__.
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(0, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="at least one shard"):
            rt.alloc_stacked_tensor(host)
        patched_setup["worker"]._orch.malloc.assert_not_called()
        rt.close()

    def test_worker_ids_length_mismatch_rejected(self, patched_setup):
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="entries"):
            rt.alloc_stacked_tensor(host, worker_ids=[0])
        rt.close()

    def test_non_shared_host_rejected_and_rolled_back(self, patched_setup):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32)  # NOT shared

        with pytest.raises(ValueError, match="shared-memory"):
            rt.alloc_stacked_tensor(host)
        # No shard should remain tracked after the rollback.
        assert not any(ptr in (0xA000, 0xB000) for _w, ptr in rt._owned_tensors)
        rt.close()


class TestCopyStackedFrom:
    """``copy_stacked_from`` reads each resident shard back into host[i] (D2H)."""

    def _make_stacked(self, patched_setup, worker_ids=None):
        patched_setup["worker"]._orch.malloc.side_effect = [0xA000, 0xB000]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        stacked = rt.alloc_stacked_tensor(host, worker_ids=worker_ids)
        patched_setup["worker"]._orch.copy_from.reset_mock()
        return rt, stacked

    def test_reads_each_shard_back(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)  # worker_ids == (0, 1)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        rt.copy_stacked_from(stacked, out)

        orch = patched_setup["worker"]._orch
        nbytes = 4 * 4 * 4
        # Facade arg order: copy_from(worker_id, dst_host_ptr, src_dev_ptr, nbytes).
        orch.copy_from.assert_any_call(0, out[0].data_ptr(), 0xA000, nbytes)
        orch.copy_from.assert_any_call(1, out[1].data_ptr(), 0xB000, nbytes)
        assert orch.copy_from.call_count == 2
        rt.close()

    def test_permuted_worker_ids(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup, worker_ids=[1, 0])
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        rt.copy_stacked_from(stacked, out)

        orch = patched_setup["worker"]._orch
        nbytes = 4 * 4 * 4
        # shard 0 resides on worker 1, shard 1 on worker 0.
        orch.copy_from.assert_any_call(1, out[0].data_ptr(), 0xA000, nbytes)
        orch.copy_from.assert_any_call(0, out[1].data_ptr(), 0xB000, nbytes)
        rt.close()

    def test_shape_mismatch_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(3, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="does not match stacked full_shape"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_dtype_mismatch_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float16).share_memory_()
        with pytest.raises(ValueError, match="does not match stacked dtype"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_non_shared_host_rejected(self, patched_setup):
        # A plain (non-shared) host buffer is invisible to the forked worker's
        # D2H write — reject it up front rather than silently returning zeros.
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32)  # NOT shared
        with pytest.raises(ValueError, match="shared-memory"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_non_contiguous_host_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        # Shared but transposed -> non-contiguous; still rejected.
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_().transpose(1, 2)
        assert not out.is_contiguous()
        with pytest.raises(ValueError, match="shared-memory"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_wrong_type_rejected(self, patched_setup):
        rt, _stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(TypeError, match="expects a StackedDeviceTensor"):
            rt.copy_stacked_from(object(), out)  # type: ignore[arg-type]  # runtime guard under test
        rt.close()

    def test_after_close_raises(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        rt.close()
        with pytest.raises(RuntimeError, match="called after close"):
            rt.copy_stacked_from(stacked, out)


class TestLifecycle:
    def test_close_idempotent_and_closes_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        rt.close()  # second close is a no-op
        assert patched_setup["worker"].close.call_count == 1

    def test_close_releases_inherited_refs_when_worker_close_raises(self, patched_setup):
        compiled = _fake_compiled([_param("weight", [16, 16])], [])
        weight = torch.zeros(16, 16, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[weight])
        patched_setup["worker"].close.side_effect = RuntimeError("worker close failed")

        with pytest.raises(RuntimeError, match="worker close failed"):
            rt.close()

        assert rt._inherited_host_tensors == ()
        assert not rt._inherited_host_storage_ptrs

    def test_context_manager_closes(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with DistributedWorker(compiled) as rt:
            assert rt is not None
        assert patched_setup["worker"].close.call_count == 1

    def test_call_after_close_raises(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        with pytest.raises(RuntimeError, match="after close"):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32))


class TestCallbacks:
    def test_callback_reaches_register(self, patched_setup):
        m = patched_setup
        placeholder = object()

        def real(args):
            return None

        m["load_subs"].return_value = {"sample_and_prepare": placeholder}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedWorker(compiled, callbacks={"sample_and_prepare": real})

        # _register_callables(w, sub_worker_fns, chip_callables): arg[1] is the bound set.
        passed = m["register"].call_args.args[1]
        assert passed == {"sample_and_prepare": real}
        rt.close()

    def test_no_callback_passes_loaded_unchanged(self, patched_setup):
        m = patched_setup
        loaded = {"sample_and_prepare": object()}
        m["load_subs"].return_value = loaded
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedWorker(compiled)

        assert m["register"].call_args.args[1] == loaded
        rt.close()

    def test_callback_unknown_name_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].return_value = {"sample_and_prepare": object()}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(ValueError, match="not sub-workers"):
            DistributedWorker(compiled, callbacks={"typo": lambda args: None})

    def test_missing_required_callback_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].return_value = {"sample": object()}
        m["load_required"].return_value = {"sample"}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(ValueError, match="runtime-bound callbacks"):
            DistributedWorker(compiled)  # abstract SubWorker not supplied

    def test_deprecated_alias_warns_and_binds(self, patched_setup):
        m = patched_setup

        def real(args):
            return None

        m["load_subs"].return_value = {"sample_and_prepare": object()}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.warns(DeprecationWarning, match="sub_worker_overrides is deprecated"):
            rt = DistributedWorker(compiled, sub_worker_overrides={"sample_and_prepare": real})

        assert m["register"].call_args.args[1] == {"sample_and_prepare": real}
        rt.close()


class TestBindSubWorkers:
    def test_none_callbacks_returns_equal_set(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        loaded = {"a": object()}
        assert _bind_sub_workers(loaded, None, set()) == loaded
        assert _bind_sub_workers(loaded, {}, set()) == loaded

    def test_valid_callback_replaces(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        placeholder, other = object(), object()

        def real(args):
            return None

        loaded = {"a": placeholder, "b": other}
        bound = _bind_sub_workers(loaded, {"a": real}, set())
        assert bound == {"a": real, "b": other}

    def test_unknown_name_raises_listing_available(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"not sub-workers.*Available sub-workers"):
            _bind_sub_workers({"a": object()}, {"b": lambda args: None}, set())

    def test_missing_required_raises(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(ValueError, match="runtime-bound callbacks"):
            _bind_sub_workers({"sample": object()}, None, {"sample"})

    def test_bad_arity_callback_rejected(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(TypeError, match="single positional"):
            _bind_sub_workers({"a": object()}, {"a": lambda: None}, set())


class TestOneShotRegression:
    """The one-shot execute_distributed path still works after helper extraction."""

    def test_one_shot_setup_dispatch_close(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [8, 8]), _param("b", [8, 8])], [])
        a = torch.zeros(8, 8, dtype=torch.float32)
        b = torch.zeros(8, 8, dtype=torch.float32)

        execute_distributed(compiled, [a, b])

        patched_setup["assemble"].assert_called_once()
        patched_setup["construct"].assert_called_once()
        patched_setup["worker"].init.assert_called_once()
        patched_setup["dispatch"].assert_called_once()
        patched_setup["worker"].close.assert_called_once()

    def test_one_shot_enables_sdma_when_a_chip_requires_it(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        patched_setup["assemble"].return_value = ({"chip_orch": object()}, "rt_name", True)
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        execute_distributed(compiled, [torch.zeros(8, 8, dtype=torch.float32)])

        assert patched_setup["construct"].call_args.kwargs["enable_sdma"] is True


class TestWorkerConstruction:
    def test_forwards_enable_sdma_to_simpler_worker(self, monkeypatch):
        worker_cls = MagicMock(name="simpler.Worker")
        monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=worker_cls))
        dc = DistributedConfig(device_ids=[0, 1])

        _construct_worker(dc, "a2a3", "tensormap_and_ringbuffer", 3, enable_sdma=True)

        worker_cls.assert_called_once_with(
            level=3,
            device_ids=[0, 1],
            num_sub_workers=3,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            enable_sdma=True,
        )


class TestExplicitDispatchAPI:
    """The new ``run`` / ``register`` surface that mirrors ChipWorker.

    DistributedWorker.run() and ``__call__`` are blocking submit/result
    compositions. register() returns a :class:`RegistrationHandle` whose call
    delegates to run().
    """

    def test_run_delegates_to_call(self, patched_setup):
        from pypto.runtime import RegistrationHandle  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [4]), _param("b", [4])], [])
        rt = DistributedWorker(compiled)

        a = torch.zeros(4).share_memory_()
        b = torch.zeros(4).share_memory_()
        rt.run(compiled, a, b)
        patched_setup["submit_dispatch"].assert_called_once()

        # register() returns a usable handle.
        rt2 = DistributedWorker(compiled)
        h = rt2.register(compiled)
        assert isinstance(h, RegistrationHandle)
        assert h.compiled is compiled
        rt.close()
        rt2.close()

    def test_run_rejects_unregistered_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        a = torch.zeros(4).share_memory_()
        with pytest.raises(ValueError, match="registered when this worker"):
            rt.run(compiled_b, a)
        rt.close()

    def test_register_rejects_unregistered_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        with pytest.raises(ValueError, match="registered when this worker"):
            rt.register(compiled_b)
        rt.close()

    def test_register_rejects_after_close(self, patched_setup):
        """register() after close() must raise; mirrors ChipWorker behaviour."""
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        with pytest.raises(RuntimeError, match="register"):
            rt.register(compiled)

    def test_handle_call_dispatches(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4]), _param("b", [4])], [])
        rt = DistributedWorker(compiled)
        a = torch.zeros(4).share_memory_()
        b = torch.zeros(4).share_memory_()

        h = rt.register(compiled)
        patched_setup["submit_dispatch"].reset_mock()
        h(a, b)
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_close_marks_handle_closed(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        h = rt.register(compiled)
        assert h.closed is False
        rt.close()
        assert h.closed is True

    def test_close_auto_frees_owned_device_tensors(self, patched_setup):
        """alloc_tensor on DistributedWorker is also tracked through the ABC."""
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)

        # alloc_tensor goes through Worker ABC -> records in _owned_tensors.
        host = torch.zeros(4, dtype=torch.float32).share_memory_()
        t = rt.alloc_tensor((4,), torch.float32, init=host)
        assert (0, t.data_ptr) in rt._owned_tensors

        # Spy on the orchestrator's free so we can assert close drove the
        # auto-free path (L3 routes free through the orchestrator facade).
        orch = patched_setup["worker"]._orch
        orch.free.reset_mock()
        rt.close()
        assert orch.free.called


class TestLoadOrchEntry:
    """Entry resolution in ``_load_orch_entry`` (issue #1678).

    The dispatch entry is the unique module-level function tagged with the
    ``_pypto_distributed_entry`` marker — resolution must not depend on the
    function's Python name nor fall back to scanning callables by name.
    """

    @staticmethod
    def _write_orch(tmp_path, src: str):
        orch_dir = tmp_path / "orchestration"
        orch_dir.mkdir()
        (orch_dir / "host_orch.py").write_text(src)
        return tmp_path

    def test_resolves_marked_function_not_imported_class(self, tmp_path):
        """Resolution follows the marker, never an alphabetically-earlier import
        such as ``CommBufferSpec`` (the original failure mode of issue #1678)."""
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "class CommBufferSpec:\n"
            "    def __init__(self, **kw):\n"
            "        raise AssertionError('wrong callable resolved')\n\n\n"
            "def moe_ep_l3(orch, _args, config, **kw):\n"
            "    return 'ok'\n\n\n"
            "moe_ep_l3._pypto_distributed_entry = True\n",
        )
        entry_fn, alloc_fn = _load_orch_entry(root)
        assert entry_fn.__name__ == "moe_ep_l3"
        assert alloc_fn is None

    def test_returns_alloc_intermediates_when_present(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def host_orch(orch, _args, config, **kw):\n"
            "    return 'ok'\n\n\n"
            "host_orch._pypto_distributed_entry = True\n\n\n"
            "def _alloc_intermediates(tensors):\n"
            "    return None\n",
        )
        entry_fn, alloc_fn = _load_orch_entry(root)
        assert entry_fn.__name__ == "host_orch"
        assert alloc_fn is not None and alloc_fn.__name__ == "_alloc_intermediates"

    def test_no_marker_raises(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def moe_ep_l3(orch, _args, config, **kw):\n    return 'ok'\n",
        )
        with pytest.raises(RuntimeError, match="exactly one entry function"):
            _load_orch_entry(root)

    def test_multiple_markers_raise(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def a(orch, _args, config, **kw):\n    return 'a'\n\n\n"
            "def b(orch, _args, config, **kw):\n    return 'b'\n\n\n"
            "a._pypto_distributed_entry = True\n"
            "b._pypto_distributed_entry = True\n",
        )
        with pytest.raises(RuntimeError, match="exactly one entry function"):
            _load_orch_entry(root)


class TestMultiProgram:
    """Multiple compatible programs share one L3 worker (issue #1698).

    Each program registers its own callables/entry/state; dispatch selects the
    program via ``run(compiled, ...)``. The shared worker is constructed and
    init()'d exactly once across all programs.
    """

    def test_prepares_multiple_programs_on_one_worker(self, patched_setup):
        m = patched_setup
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        # One worker, init()'d once; per-program setup ran twice.
        m["construct"].assert_called_once()
        m["worker"].init.assert_called_once()
        assert m["assemble"].call_count == 2
        assert m["load_entry"].call_count == 2
        assert m["register"].call_count == 2
        # Both programs are dispatchable; the first is primary.
        assert set(rt._states) == {prog_a, prog_b}
        assert rt._compiled is prog_a
        rt.close()

    def test_run_selects_program_state(self, patched_setup):
        m = patched_setup
        # Distinct entry_fns per program so we can prove dispatch picks the
        # selected program's state, not the primary's.
        entry_a, entry_b = MagicMock(name="entry_a"), MagicMock(name="entry_b")
        m["load_entry"].side_effect = [(entry_a, None), (entry_b, None)]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])

        a = torch.zeros(4).share_memory_()
        b = torch.zeros(8).share_memory_()

        rt.run(prog_b, b)
        assert m["submit_dispatch"].call_args.args[1] is entry_b
        rt.run(prog_a, a)
        assert m["submit_dispatch"].call_args.args[1] is entry_a
        rt.close()

    def test_num_sub_workers_is_max_across_programs(self, patched_setup):
        m = patched_setup
        m["load_subs"].side_effect = [{"s0": object()}, {"s0": object(), "s1": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        # _construct_worker(dc, platform, runtime_name, num_sub) — num_sub is the
        # max sub-worker count across all programs (2 here).
        assert m["construct"].call_args.args[3] == 2
        rt.close()

    def test_enables_sdma_when_any_program_requires_it(self, patched_setup):
        m = patched_setup
        m["assemble"].side_effect = [
            ({"chip_a": object()}, "rt_name", False),
            ({"chip_b": object()}, "rt_name", True),
        ]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        assert m["construct"].call_args.kwargs["enable_sdma"] is True
        rt.close()

    def test_single_program_preserves_default_sdma_capability(self, patched_setup):
        rt = DistributedWorker(_fake_compiled([_param("a", [4])], []))

        assert patched_setup["construct"].call_args.kwargs["enable_sdma"] is False
        rt.close()

    def test_single_program_list_keeps_call_shortcut(self, patched_setup):
        # A one-element list is what ``compiled.prepare()`` builds; the
        # ``rt(*args)`` shortcut must keep working for it.
        prog = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker([prog])
        assert rt._multi_program is False
        rt(torch.zeros(4).share_memory_())
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_call_raises_in_multi_program_mode(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])
        with pytest.raises(TypeError, match="ambiguous"):
            rt(torch.zeros(4).share_memory_())
        rt.close()

    def test_shared_device_tensor_across_programs(self, patched_setup):
        m = patched_setup
        submitted_tensors: list[dict[str, Any]] = []

        def submit_dispatch(*args):
            submitted_tensors.append(dict(args[2]))
            return _ImmediateNativeHandle()

        m["submit_dispatch"].side_effect = submit_dispatch
        # Both programs take a same-shaped KV param; one resident DeviceTensor
        # is dispatched through both (the serving KV-cache sharing contract).
        prog_a = _fake_compiled([_param("kv", [16, 16])], [])
        prog_b = _fake_compiled([_param("kv", [16, 16])], [])
        rt = DistributedWorker([prog_a, prog_b])

        kv = DeviceTensor(0x5000, (16, 16), torch.float32)
        rt.run(prog_a, kv)
        rt.run(prog_b, kv)

        assert m["submit_dispatch"].call_count == 2
        for tensors in submitted_tensors:
            assert tensors["kv"] is kv  # same pointer in both tensor maps
        rt.close()

    def test_register_each_program_returns_handle(self, patched_setup):
        from pypto.runtime import RegistrationHandle  # noqa: PLC0415

        m = patched_setup
        entry_a, entry_b = MagicMock(name="entry_a"), MagicMock(name="entry_b")
        m["load_entry"].side_effect = [(entry_a, None), (entry_b, None)]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])

        h_a = rt.register(prog_a)
        h_b = rt.register(prog_b)
        assert isinstance(h_a, RegistrationHandle) and isinstance(h_b, RegistrationHandle)
        assert h_a.compiled is prog_a
        assert h_b.compiled is prog_b

        # Each handle dispatches its own program's state.
        h_a(torch.zeros(4).share_memory_())
        assert m["submit_dispatch"].call_args.args[1] is entry_a
        h_b(torch.zeros(8).share_memory_())
        assert m["submit_dispatch"].call_args.args[1] is entry_b

        # close() marks every program's handle closed and tears down the one worker.
        rt.close()
        assert h_a.closed is True
        assert h_b.closed is True
        assert m["worker"].close.call_count == 1

    def test_callbacks_apply_per_program(self, patched_setup):
        m = patched_setup

        # prog_a declares sub-worker 'sample'; prog_b declares 'route'. A callback
        # for each binds only to the program that declares it — heterogeneous
        # sub-worker sets across programs must not raise.
        def cb_sample(args):
            return None

        def cb_route(args):
            return None

        m["load_subs"].side_effect = [{"sample": object()}, {"route": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b], callbacks={"sample": cb_sample, "route": cb_route})

        bound_sets = [call.args[1] for call in m["register"].call_args_list]
        assert {"sample": cb_sample} in bound_sets
        assert {"route": cb_route} in bound_sets
        rt.close()

    def test_callback_matching_no_program_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].side_effect = [{"sample": object()}, {"route": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        with pytest.raises(ValueError, match="not sub-workers of any prepared program"):
            DistributedWorker([prog_a, prog_b], callbacks={"typo": lambda args: None})

    def test_prepare_extra_compiled_forwards_program_list(self):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        primary = _fake_compiled([_param("a", [4])], [])
        extra = _fake_compiled([_param("b", [8])], [])
        with patch("pypto.runtime.distributed_runner.DistributedWorker") as fake_worker:
            DistributedCompiledProgram.prepare(primary, extra_compiled=[extra])
        # prepare() delegates to DistributedWorker([primary, *extra_compiled], ...).
        assert fake_worker.call_args.args[0] == [primary, extra]

    def test_prepare_forwards_persistent_flag(self):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        primary = _fake_compiled([_param("a", [4])], [])
        with patch("pypto.runtime.distributed_runner.DistributedWorker") as fake_worker:
            DistributedCompiledProgram.prepare(primary, persistent=True)
        assert fake_worker.call_args.kwargs["persistent"] is True
        assert fake_worker.call_args.kwargs["reset_persistent_windows"] is None

    def test_empty_sequence_raises(self, patched_setup):
        with pytest.raises(ValueError, match="at least one compiled program"):
            DistributedWorker([])

    def test_rejects_mismatched_platform(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        prog_b.platform = "different_platform"
        with pytest.raises(ValueError, match="same platform"):
            DistributedWorker([prog_a, prog_b])

    def test_rejects_mismatched_device_ids(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        prog_b._distributed_config = DistributedConfig(device_ids=[0, 1])
        with pytest.raises(ValueError, match="same device_ids"):
            DistributedWorker([prog_a, prog_b])

    def test_rejects_mismatched_runtime(self, patched_setup):
        m = patched_setup
        m["assemble"].side_effect = [
            ({"chip_orch": object()}, "rt_name", False),
            ({"chip_orch": object()}, "other_rt", False),
        ]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        with pytest.raises(ValueError, match="same runtime"):
            DistributedWorker([prog_a, prog_b])


class TestAssembleChipCallables:
    """``_assemble_chip_callables`` is driven by the on-disk ``next_levels/``
    layout (no live IR), so it works for both freshly-compiled programs and ones
    reconstructed via ``from_dir`` (the L3 runtime_dir replay path, #1689)."""

    @staticmethod
    def _build(tmp_path, chip_names, *, stray=False) -> Any:
        nl = tmp_path / "next_levels"
        for name in chip_names:
            (nl / name).mkdir(parents=True, exist_ok=True)
            (nl / name / "kernel_config.py").write_text("KERNELS = []\nORCHESTRATION = {}\n")
        if stray:  # a dir without kernel_config.py must be skipped, not assembled
            (nl / "_not_a_chip").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=tmp_path, platform="a2a3sim")

    @staticmethod
    def _stub_device_runner(monkeypatch, ca) -> None:
        """Inject a stub ``device_runner`` so ``_assemble_chip_callables`` can be
        exercised without importing the real module (which pulls in the simpler
        toolchain via ``kernel_compiler`` and is absent in the unit-test env)."""
        monkeypatch.setitem(
            sys.modules, "pypto.runtime.device_runner", SimpleNamespace(compile_and_assemble=ca)
        )

    def test_picks_up_chip_dirs_with_kernel_config(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"], stray=True)
        ca = MagicMock(return_value=(MagicMock(name="ChipCallable"), "tensormap_and_ringbuffer", {}))
        self._stub_device_runner(monkeypatch, ca)
        chip_callables, runtime_name, enable_sdma = _assemble_chip_callables(compiled)

        assert set(chip_callables) == {"chip_a", "chip_b"}  # stray dir skipped
        assert runtime_name == "tensormap_and_ringbuffer"
        assert enable_sdma is False
        called_dirs = {call.args[0] for call in ca.call_args_list}
        assert called_dirs == {tmp_path / "next_levels" / "chip_a", tmp_path / "next_levels" / "chip_b"}
        assert all(call.args[1] == "a2a3sim" for call in ca.call_args_list)

    def test_aggregates_enable_sdma_across_chip_configs(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"])
        ca = MagicMock(
            side_effect=[
                (MagicMock(name="ChipCallableA"), "tensormap_and_ringbuffer", {}),
                (
                    MagicMock(name="ChipCallableB"),
                    "tensormap_and_ringbuffer",
                    {"enable_sdma": True},
                ),
            ]
        )
        self._stub_device_runner(monkeypatch, ca)

        _, _, enable_sdma = _assemble_chip_callables(compiled)

        assert enable_sdma is True

    def test_raises_on_inconsistent_runtime(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"])
        ca = MagicMock(
            side_effect=[
                (MagicMock(name="ChipCallable"), "rt_one", {}),
                (MagicMock(name="ChipCallable"), "rt_two", {}),
            ]
        )
        self._stub_device_runner(monkeypatch, ca)
        with pytest.raises(RuntimeError, match="Inconsistent runtime"):
            _assemble_chip_callables(compiled)

    def test_raises_when_no_chip_dirs(self, tmp_path):
        # No next_levels/, so the helpful error must surface without importing the
        # device_runner toolchain (the import is deferred until a chip is found).
        compiled: Any = SimpleNamespace(output_dir=tmp_path, platform="a2a3sim")
        with pytest.raises(RuntimeError, match="No chip-level tasks found"):
            _assemble_chip_callables(compiled)


class _SpyDfxConfig:
    """Minimal stand-in for ``CallConfig`` exposing a mutable ``output_prefix``."""

    def __init__(self, output_prefix: str = "") -> None:
        self.output_prefix = output_prefix


class _RecordingOrch:
    """Records the ``output_prefix`` observed at each ``submit_next_level``.

    Captures the prefix *at submit time* (not after) so tests can prove
    ``_submit_chip`` applied the per-dispatch suffix before the task was queued.
    """

    def __init__(self, chip_count: int | None = None) -> None:
        self.calls: list[tuple[Any, int, str]] = []
        # ``_submit_chip`` reads/writes this per-card dispatch counter on the
        # orch; declare it so the attribute is known to the type checker.
        self._dfx_dispatch_idx: dict[str, int] = {}
        # ``_dispatch.orch_fn`` stamps the placement state on the real
        # Orchestrator; mirror it here. Leaving ``chip_count`` unset models a
        # caller that bypassed ``orch_fn``.
        if chip_count is not None:
            self._pypto_chip_count: int = chip_count
        self._pypto_commless_seq: int = 0

    def submit_next_level(self, callable_id: Any, task_args: Any, config: Any, *, worker: int) -> str:
        self.calls.append((callable_id, worker, config.output_prefix))
        return "submitted"


class TestSubmitChip:
    """``_submit_chip`` namespaces per-dispatch DFX ``output_prefix`` then restores it."""

    def test_suffixes_prefix_at_submit_and_restores(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        ret = _submit_chip(orch, "chip_a", "ta", cfg, 3)
        # Card + the card's 0th dispatch was visible to the runtime at submit
        # time...
        assert orch.calls == [("chip_a", 3, "/work/dfx_outputs/rank3/d0")]
        # ...and the shared config is restored afterward.
        assert cfg.output_prefix == "/work/dfx_outputs"
        assert ret == "submitted"

    def test_distinct_ranks_get_distinct_dirs(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        for r in (0, 1, 2):
            _submit_chip(orch, "chip", "ta", cfg, r)
        # Each card's first dispatch is ``d0``.
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank1/d0",
            "/work/dfx_outputs/rank2/d0",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_multiple_dispatches_same_card_get_distinct_dirs(self):
        # The bug this fix targets: several dispatches to ONE card must not
        # share a dir (the runtime rewrites fixed-name artifacts per run, so a
        # shared dir means all-but-the-last are clobbered). Each gets ``d{k}``.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        _submit_chip(orch, "chip_a", "ta", cfg, 0)
        _submit_chip(orch, "chip_b", "ta", cfg, 0)  # different program, same card
        _submit_chip(orch, "chip_a", "ta", cfg, 0)  # repeat dispatch, same card
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank0/d1",
            "/work/dfx_outputs/rank0/d2",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_counter_resets_when_orch_dispatch_idx_cleared(self):
        # ``orch_fn`` clears ``_dfx_dispatch_idx`` at the top of every run, so a
        # given card's dispatch numbering matches across the swimlane two-pass.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        _submit_chip(orch, "chip", "ta", cfg, 0)  # pass 1: d0
        orch._dfx_dispatch_idx = {}  # what orch_fn does between passes
        _submit_chip(orch, "chip", "ta", cfg, 0)  # pass 2: d0 again
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank0/d0",
        ]

    def test_dfx_off_forwards_unchanged(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, 5)
        assert orch.calls == [("chip", 5, "")]
        assert cfg.output_prefix == ""

    def test_commless_dispatches_round_robin_over_chips(self):
        # A comm-less dispatch (``worker=None``) names no chip, but simpler
        # #1436 requires an exact target, so consecutive ones are handed out
        # round-robin over the program's chips — a host_orch with one comm-less
        # dispatch per chip still spreads across them.
        orch = _RecordingOrch(chip_count=2)
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        for _ in range(3):
            _submit_chip(orch, "chip", "ta", cfg, None)
        assert [c[1] for c in orch.calls] == [0, 1, 0]
        # Each resolved chip gets its own dispatch counter.
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank1/d0",
            "/work/dfx_outputs/rank0/d1",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_commless_dispatch_without_chip_count_falls_back_to_chip_zero(self):
        # A caller that bypassed ``orch_fn`` leaves no chip count on ``orch``;
        # chip 0 always exists, so it is the safe fallback.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, None)
        _submit_chip(orch, "chip", "ta", cfg, None)
        assert [c[1] for c in orch.calls] == [0, 0]

    def test_pinned_dispatch_keeps_its_rank(self):
        # A ``device=``-pinned dispatch is never re-placed, even when comm-less
        # dispatches are round-robining alongside it.
        orch = _RecordingOrch(chip_count=2)
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, 1)
        _submit_chip(orch, "chip", "ta", cfg, None)
        _submit_chip(orch, "chip", "ta", cfg, 1)
        assert [c[1] for c in orch.calls] == [1, 0, 1]

    def test_records_each_dispatchs_l2_program(self, tmp_path):
        # Issue #2169: ``rank{w}/d{k}`` says where a dispatch ran, not what it
        # ran, and ``func_id`` only means something within one L2 program. The
        # marker written here is what lets the offline post-pass label a
        # dispatch's records with its own program's kernel names.
        chip_cids = {"lm_head": object(), "mtp_decode_layer": object()}
        orch = _RecordingOrch()
        _reset_dfx_dispatch_state(orch, chip_cids)
        cfg = _SpyDfxConfig(output_prefix=str(tmp_path))

        # One card, two different programs -> d0 and d1 name their own program.
        _submit_chip(orch, chip_cids["mtp_decode_layer"], "ta", cfg, 0)
        _submit_chip(orch, chip_cids["lm_head"], "ta", cfg, 0)

        assert json.loads((tmp_path / "rank0" / "d0" / "dispatch_program.json").read_text()) == {
            "program": "mtp_decode_layer"
        }
        assert json.loads((tmp_path / "rank0" / "d1" / "dispatch_program.json").read_text()) == {
            "program": "lm_head"
        }

    def test_no_marker_when_chip_names_unstamped(self, tmp_path):
        # A caller that bypassed ``_reset_dfx_dispatch_state`` leaves no name
        # table on ``orch``; the dispatch must still go through (the marker is a
        # diagnostic, never a precondition).
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix=str(tmp_path))

        _submit_chip(orch, "chip_a", "ta", cfg, 0)

        assert orch.calls == [("chip_a", 0, f"{tmp_path}/rank0/d0")]
        assert not (tmp_path / "rank0" / "d0" / "dispatch_program.json").exists()


def _write_dfx_dispatch_dirs(dfx: Path, *rels: str) -> None:
    """Lay down ``<dfx>/<rel>/l2_swimlane_records.json`` for each dispatch dir.

    Shared by the cleaner and collector tests below so the on-disk DFX layout
    they both assume is spelled out once.
    """
    for rel in rels:
        (dfx / rel).mkdir(parents=True)
        (dfx / rel / "l2_swimlane_records.json").write_text("{}", encoding="utf-8")


def _write_chip_program(output_dir: Path, program: str, *kernel_names: str) -> None:
    """Lay down ``next_levels/<program>/kernel_config.py`` naming *kernel_names*.

    Every L2 program numbers its kernels from ``func_id`` 0 — that shared
    numbering is exactly what makes a name map merged across programs wrong
    (issue #2169), so each program written here starts at 0 on purpose.
    """
    chip_dir = output_dir / "next_levels" / program
    chip_dir.mkdir(parents=True)
    kernels = [{"func_id": i, "name": name} for i, name in enumerate(kernel_names)]
    (chip_dir / "kernel_config.py").write_text(f"KERNELS = {kernels!r}\n", encoding="utf-8")


def _mark_dispatch_program(disp_dir: Path, program: str) -> None:
    """Stamp the marker ``_submit_chip`` writes for a dispatch of *program*."""
    (disp_dir / "dispatch_program.json").write_text(json.dumps({"program": program}), encoding="utf-8")


@pytest.fixture
def fake_swimlane_converter(monkeypatch):
    """Register a fake ``simpler_setup.tools.swimlane_converter``.

    The real module ships with the optional ``simpler`` runtime package, which is
    not installed in CI. The fake reproduces the one function pypto calls,
    ``load_kernel_config``, with the real contract: import the ``kernel_config.py``
    and return its ``func_id`` (as ``str``) -> ``name`` mapping. Tests therefore
    still exercise the genuine on-disk layout.
    """
    pkg = ModuleType("simpler_setup")
    tools = ModuleType("simpler_setup.tools")
    mod = ModuleType("simpler_setup.tools.swimlane_converter")

    def load_kernel_config(config_path: str) -> dict[str, str]:
        spec = importlib.util.spec_from_file_location("kernel_config", config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {str(k["func_id"]): k["name"] for k in module.KERNELS}

    mod.load_kernel_config = load_kernel_config  # pyright: ignore[reportAttributeAccessIssue]
    tools.swimlane_converter = mod  # pyright: ignore[reportAttributeAccessIssue]
    pkg.tools = tools  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "simpler_setup", pkg)
    monkeypatch.setitem(sys.modules, "simpler_setup.tools", tools)
    monkeypatch.setitem(sys.modules, "simpler_setup.tools.swimlane_converter", mod)
    return mod


class TestClearDfxDispatchDirs:
    """``_clear_dfx_dispatch_dirs`` drops stale ``rank*/d{k}`` dirs before a run."""

    def test_removes_only_dispatch_dirs(self, tmp_path):
        # A prior run left rank0/{d0,d1,d2} and rank1/d0; the current run will
        # only write d0, so the stale d1/d2 must be cleared. A sibling non-d{k}
        # dir (e.g. a future diagnostic) is preserved.
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1", "rank0/d2", "rank1/d0", "rank0/keepme")

        _clear_dfx_dispatch_dirs(dfx)

        # All d{k} dirs gone, on every card...
        assert not (dfx / "rank0" / "d0").exists()
        assert not (dfx / "rank0" / "d1").exists()
        assert not (dfx / "rank0" / "d2").exists()
        assert not (dfx / "rank1" / "d0").exists()
        # ...but the non-dispatch dir and the rank dirs themselves remain.
        assert (dfx / "rank0" / "keepme").is_dir()
        assert (dfx / "rank0").is_dir()

    def test_missing_base_is_noop(self, tmp_path):
        # No dfx_outputs yet (first dispatch) -> nothing to clear, no error.
        _clear_dfx_dispatch_dirs(tmp_path / "dfx_outputs")


class TestCollectL3Swimlane:
    """``_collect_l3_swimlane`` converts every ``rank*/d{k}`` dispatch's records."""

    @staticmethod
    def _spy_generate_swimlane(monkeypatch) -> list[SimpleNamespace]:
        """Record each converter invocation (dispatch dir, ``-k`` dir, name map)."""
        import pypto.runtime.runner as _runner  # noqa: PLC0415

        seen: list[SimpleNamespace] = []

        def _fake(work_dir, out_dir, records, func_names=None):  # noqa: ANN001
            seen.append(
                SimpleNamespace(work_dir=work_dir, out_dir=out_dir, records=records, func_names=func_names)
            )

        monkeypatch.setattr(_runner, "_generate_swimlane", _fake)
        return seen

    def test_collects_every_cards_dispatch_dirs(self, tmp_path, monkeypatch):
        # Globbing ``rank*`` (rather than iterating a rank count) is what lets a
        # run whose cards are not known up front — e.g. a comm-less program
        # whose dispatches were placed round-robin — get converted at all.
        seen = self._spy_generate_swimlane(monkeypatch)
        dfx = tmp_path / "dfx_outputs"
        # ``rank0/keepme`` carries a records file like a real dispatch dir, so
        # only the ``d[0-9]*`` filter can exclude it — that makes the assertion
        # below a genuine discriminator for the glob rather than for the
        # ``records.exists()`` guard.
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1", "rank1/d0", "rank0/keepme")
        # A dispatch dir with no records (DFX wrote nothing) is skipped.
        (dfx / "rank1" / "d1").mkdir(parents=True)

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert sorted(str(s.out_dir.relative_to(dfx)) for s in seen) == [
            "rank0/d0",
            "rank0/d1",
            "rank1/d0",
        ]

    def test_simulator_platform_skips_conversion(self, tmp_path, monkeypatch):
        # Onboard-only: the simulator emits records but not the task metadata
        # the converter joins against, so the raw records are kept as-is.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_dfx_dispatch_dirs(tmp_path / "dfx_outputs", "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3sim")

        assert seen == []

    def test_missing_dfx_base_is_noop(self, tmp_path, monkeypatch):
        # DFX was off (or nothing was written) -> nothing to convert, no error.
        seen = self._spy_generate_swimlane(monkeypatch)

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert seen == []

    def test_name_map_is_scoped_to_each_dispatchs_own_program(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # Regression for issue #2169. Two L2 programs both number their kernels
        # from func_id 0, so a name map merged across them relabels one
        # program's tasks with the other's names — silently and plausibly
        # (``lm_head_dispatch_wait``, a cross-card spin-wait, printed as
        # ``mtp_projection_norm``). Each dispatch must get its own program's map.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push", "lm_head_dispatch_wait")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms", "mtp_projection_norm")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1")
        _mark_dispatch_program(dfx / "rank0" / "d0", "mtp_decode_layer")
        _mark_dispatch_program(dfx / "rank0" / "d1", "lm_head")

        _collect_l3_swimlane(tmp_path, "a2a3")

        by_dir = {s.out_dir.name: s for s in seen}
        assert set(by_dir) == {"d0", "d1"}
        # Each dispatch's name map holds its own program's kernels...
        for disp, program, names in (
            ("d0", "mtp_decode_layer", ["mtp_projection_rms", "mtp_projection_norm"]),
            ("d1", "lm_head", ["lm_head_dispatch_push", "lm_head_dispatch_wait"]),
        ):
            name_map = json.loads((dfx / "rank0" / disp / "name_map.json").read_text())
            assert name_map["callable_id_to_name"] == {"0": names[0], "1": names[1]}
            assert by_dir[disp].func_names == dfx / "rank0" / disp / "name_map.json"
            # ...and the converter's ``-k`` fallback names the same program, so
            # the two label sources can never disagree.
            assert by_dir[disp].work_dir == tmp_path / "next_levels" / program

    def test_sole_program_names_an_unmarked_dispatch(self, tmp_path, monkeypatch, fake_swimlane_converter):
        # Only one L2 program in the build: there is no namespace to confuse, so
        # a dispatch without a marker (e.g. artifacts from an older run) is still
        # labelled rather than degraded to anonymous tasks.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "only_chip", "rms", "matmul")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        name_map = json.loads((dfx / "rank0" / "d0" / "name_map.json").read_text())
        assert name_map["callable_id_to_name"] == {"0": "rms", "1": "matmul"}
        assert seen[0].work_dir == tmp_path / "next_levels" / "only_chip"

    def test_unresolvable_dispatch_converts_without_names(
        self, tmp_path, monkeypatch, fake_swimlane_converter, capsys
    ):
        # Several programs and no marker: the program is genuinely unknown. The
        # records still convert, but with anonymous labels — a wrong name is
        # worse than no name, since it reads as a real measurement.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert len(seen) == 1
        assert seen[0].func_names is None
        assert not (dfx / "rank0" / "d0" / "name_map.json").exists()
        # ``work_dir`` holds no kernel_config.py, so no other program's table is
        # handed to the converter's ``-k`` fallback either.
        assert not (seen[0].work_dir / "kernel_config.py").exists()
        assert "No L2 program recorded for rank0/d0" in capsys.readouterr().out

    def test_unresolvable_dispatch_drops_a_stale_name_map(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # With no map passed, the converter auto-discovers a sibling
        # ``name_map*.json`` — so a map left by an earlier run would quietly
        # resurrect the mislabelling this fix removes.
        self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")
        stale = dfx / "rank0" / "d0" / "name_map.json"
        stale.write_text('{"callable_id_to_name": {"0": "mtp_projection_rms"}}', encoding="utf-8")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert not stale.exists()

    def test_resolved_program_without_a_table_drops_a_stale_name_map(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # The program resolves, but its ``kernel_config.py`` names no kernels, so
        # no map is written for this run. A previous run's map must not survive to
        # be picked up in its place — this dispatch renders anonymously.
        self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head")  # KERNELS = []
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")
        _mark_dispatch_program(dfx / "rank0" / "d0", "lm_head")
        stale = dfx / "rank0" / "d0" / "name_map.json"
        stale.write_text('{"callable_id_to_name": {"0": "mtp_projection_rms"}}', encoding="utf-8")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert not stale.exists()

    def test_stray_subdir_does_not_hide_the_sole_program(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # ``next_levels/`` may hold a subdir that is not an L2 program (no
        # kernel_config.py). Counting it would make the build look multi-program
        # and needlessly drop the unmarked dispatch to anonymous labels.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "only_chip", "rms")
        (tmp_path / "next_levels" / "scratch").mkdir()
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert seen[0].work_dir == tmp_path / "next_levels" / "only_chip"
        assert json.loads((dfx / "rank0" / "d0" / "name_map.json").read_text())["callable_id_to_name"] == {
            "0": "rms"
        }


class _BoolStrictCallConfig:
    """Fake ``CallConfig`` whose ``enable_dep_gen`` mirrors simpler's pybind setter.

    The real ``CallConfig.enable_dep_gen`` pybind overload accepts only ``bool``
    and raises ``TypeError`` on an ``int`` — exactly the crash issue #1952
    reproduces when the int ``enable_l2_swimlane`` CLI flag (0/1/2) leaks through
    the ``and``/``or`` chain unwrapped. ``bool`` is a subclass of ``int``, so
    ``isinstance(value, bool)`` matches the pybind behavior (rejects ``1``/``0``).
    """

    def __init__(self) -> None:
        self.aicpu_thread_num = 0
        self.enable_dump_args = 0
        self.enable_pmu = 0
        self.enable_scope_stats = False
        self.enable_l2_swimlane: Any = 0
        self.output_prefix = ""
        self.runtime_env = SimpleNamespace(ring_task_window=0, ring_heap=0, ring_dep_pool=0)
        self._enable_dep_gen = False

    @property
    def enable_dep_gen(self) -> bool:
        return self._enable_dep_gen

    @enable_dep_gen.setter
    def enable_dep_gen(self, value: object) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"incompatible function arguments: enable_dep_gen expects bool, got {type(value).__name__}"
            )
        self._enable_dep_gen = value


@pytest.fixture
def fake_simpler_task_interface(monkeypatch):
    """Register a fake ``simpler.task_interface`` exposing a bool-strict ``CallConfig``.

    Lets ``_make_call_config`` run without the real (optional) ``simpler`` runtime
    package while still enforcing the pybind ``bool``-only contract on
    ``enable_dep_gen``.
    """
    pkg = ModuleType("simpler")
    mod = ModuleType("simpler.task_interface")
    mod.CallConfig = _BoolStrictCallConfig  # pyright: ignore[reportAttributeAccessIssue]
    pkg.task_interface = mod  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "simpler", pkg)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", mod)
    return mod


class TestMakeCallConfigDepGenType:
    """``_make_call_config`` must assign a ``bool`` to ``enable_dep_gen``.

    Regression for issue #1952: ``enable_l2_swimlane`` is an int (0/1/2), so the
    ``dfx.enable_dep_gen or (co_enable_swimlane_dep_gen and dfx.enable_l2_swimlane)``
    chain can yield an int, which the ``bool``-only pybind setter rejects.
    """

    # The pypto-lib CLI wires ``--enable-l2-swimlane`` as ``type=int,
    # choices=(0, 1, 2)``, so ``RunConfig`` receives an ``int`` here even though
    # the field is annotated ``bool`` — that int is precisely the crash trigger
    # under test, hence the deliberate ``pyright: ignore[reportArgumentType]``.

    def test_int_swimlane_flag_yields_bool_dep_gen(self, tmp_path, fake_simpler_task_interface):
        # ``--enable-l2-swimlane 1`` reaches RunConfig as the int ``1``; the
        # co-enable path must still hand ``enable_dep_gen`` a genuine ``bool``.
        run_config = RunConfig(enable_l2_swimlane=1)  # pyright: ignore[reportArgumentType]
        cfg = _make_call_config(DistributedConfig(), run_config, dfx_base=tmp_path / "dfx")
        assert cfg.enable_dep_gen is True
        assert cfg.enable_l2_swimlane == 1

    def test_int_zero_swimlane_yields_bool_false_dep_gen(self, tmp_path, fake_simpler_task_interface):
        # Another DFX flag opens the block while swimlane is the int ``0``; the
        # ``and``/``or`` chain would otherwise assign int ``0`` and still crash.
        run_config = RunConfig(enable_dump_args=1, enable_l2_swimlane=0)  # pyright: ignore[reportArgumentType]
        cfg = _make_call_config(DistributedConfig(), run_config, dfx_base=tmp_path / "dfx")
        assert cfg.enable_dep_gen is False

    def test_clean_timing_suppresses_implicit_dep_gen(self, tmp_path, fake_simpler_task_interface):
        run_config = RunConfig(enable_l2_swimlane=1)  # pyright: ignore[reportArgumentType]
        cfg = _make_call_config(
            DistributedConfig(),
            run_config,
            dfx_base=tmp_path / "dfx",
            co_enable_swimlane_dep_gen=False,
        )
        assert cfg.enable_dep_gen is False

    def test_explicit_dep_gen_still_wins_when_co_enable_is_off(self, tmp_path, fake_simpler_task_interface):
        run_config = RunConfig(
            enable_l2_swimlane=1,  # pyright: ignore[reportArgumentType]
            enable_dep_gen=True,
        )
        cfg = _make_call_config(
            DistributedConfig(),
            run_config,
            dfx_base=tmp_path / "dfx",
            co_enable_swimlane_dep_gen=False,
        )
        assert cfg.enable_dep_gen is True


class _PersistentDomainHandle:
    def __init__(self, name: str, workers: list[int], window_size: int, allocation_index: int) -> None:
        self.name = name
        self.workers = tuple(workers)
        self.contexts = {
            worker: SimpleNamespace(
                local_window_base=0x10000000 + allocation_index * 0x100000 + worker * 0x10000,
                actual_window_size=window_size,
            )
            for worker in workers
        }
        self.release_count = 0
        self.freed = False

    def __getitem__(self, worker_id: int):
        return self.contexts[worker_id]

    def release(self) -> None:
        self.release_count += 1


class _PersistentOrch:
    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.allocate_calls: list[dict[str, Any]] = []
        self.copy_calls: list[tuple[int, int, int, int]] = []
        self.handles: list[_PersistentDomainHandle] = []
        self.worker._execute_pending_domain_releases.side_effect = self._mark_released_domains_freed

    def allocate_domain(self, **kwargs):
        self.allocate_calls.append(kwargs)
        handle = _PersistentDomainHandle(
            kwargs["name"],
            list(kwargs["workers"]),
            int(kwargs["window_size"]),
            len(self.handles),
        )
        self.handles.append(handle)
        self.worker._live_domains[handle.name] = handle
        return handle

    def copy_to(self, worker_id: int, dst: int, src: int, size: int) -> None:
        self.copy_calls.append((worker_id, dst, src, size))

    def _mark_released_domains_freed(self) -> None:
        for handle in self.handles:
            if handle.release_count:
                handle.freed = True


def _persistent_entry(window_size: int, seen_handles: list[Any]):
    def entry(
        orch,
        _args,
        config,
        *,
        tensors,
        callables,
        sub_ids,
        _keep,
        world_size,
        _domain_provider=None,
    ):
        del orch, _args, config, tensors, callables, sub_ids, _keep
        assert _domain_provider is not None
        with _domain_provider(
            name="comm_d0",
            workers=[*range(world_size)],
            window_size=window_size,
            buffers=[SimpleNamespace(name="signal", dtype="opaque", count=4, nbytes=4)],
        ) as domain:
            seen_handles.append(domain)

    return entry


class TestPersistentDistributedWorker:
    def test_window_reset_requires_persistent_mode(self):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with pytest.raises(ValueError, match="requires persistent=True"):
            DistributedWorker(compiled, reset_persistent_windows=True)

    def test_rejects_artifact_without_domain_provider_hook(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with pytest.raises(ValueError, match="requires regenerated host orchestration"):
            DistributedWorker(compiled, persistent=True)

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("_live_domains", None),
            ("_execute_pending_domain_releases", None),
        ],
    )
    def test_rejects_missing_persistent_runtime_hooks_before_init(self, patched_setup, attribute, value):
        m = patched_setup
        setattr(m["worker"], attribute, value)
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])

        with pytest.raises(RuntimeError, match=attribute):
            DistributedWorker(compiled, persistent=True)

        m["worker"].init.assert_not_called()
        m["worker"].close.assert_called_once_with()

    def test_request_run_fences_reuse_and_zero_domain_by_default(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]
        seen_handles: list[Any] = []
        window_size = (1 << 20) + 17
        m["load_entry"].return_value = (_persistent_entry(window_size, seen_handles), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled._distributed_config = DistributedConfig(device_ids=[0, 1])

        rt = DistributedWorker(compiled, persistent=True)
        arg = DeviceTensor(0x1000, (16, 16), torch.float32)
        rt(arg)
        rt(arg)
        rt.close()

        assert m["worker"].submit.call_count == 2
        assert [call["name"] for call in orch.allocate_calls] == ["p0:comm_d0"]
        assert len(seen_handles) == 2
        assert seen_handles[0] is seen_handles[1]
        # The first request receives the freshly-zeroed allocation. The second
        # restores 1 MiB + 17 bytes on each of two workers before dispatch.
        assert [(worker, size) for worker, _dst, _src, size in orch.copy_calls] == [
            (0, 1 << 20),
            (0, 17),
            (1, 1 << 20),
            (1, 17),
        ]
        # A retained domain survives both request run-fences and is released
        # once when the persistent dispatcher closes.
        assert orch.handles[0].release_count == 1
        assert m["worker"]._live_domains == {}
        m["worker"]._execute_pending_domain_releases.assert_called_once_with()

    def test_reused_domain_skips_window_reset_when_disabled(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]
        seen_handles: list[Any] = []
        m["load_entry"].return_value = (_persistent_entry(64, seen_handles), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled._distributed_config = DistributedConfig(device_ids=[0, 1])

        rt = DistributedWorker(compiled, persistent=True, reset_persistent_windows=False)
        arg = DeviceTensor(0x1000, (16, 16), torch.float32)
        rt(arg)
        rt(arg)
        rt.close()

        assert len(orch.allocate_calls) == 1
        assert seen_handles[0] is seen_handles[1]
        assert orch.copy_calls == []
        assert m["worker"].submit.call_count == 2

    def test_task_args_stay_alive_through_request_drain(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        task_args_ref = None

        class TaskArgsSentinel:
            pass

        def entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, world_size, _domain_provider
            nonlocal task_args_ref
            task_args = TaskArgsSentinel()
            task_args_ref = weakref.ref(task_args)
            _keep.append(task_args)

        def assert_task_args_alive() -> None:
            assert task_args_ref is not None
            assert task_args_ref() is not None

        native = _ControlledNativeHandle(on_result=assert_task_args_alive)

        def worker_submit(fn):
            fn(orch, None, None)
            native.complete()
            return native

        m["worker"].submit.side_effect = worker_submit
        m["load_entry"].return_value = (entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])

        rt = DistributedWorker(compiled, persistent=True)
        rt(DeviceTensor(0x1000, (16, 16), torch.float32))

        assert task_args_ref is not None
        # Once the caller waits the handle, its bounded frame releases the
        # request keepalive instead of retaining it for the worker lifetime.
        assert task_args_ref() is None
        rt.close()

    def test_multi_program_domains_are_isolated_and_reused(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]
        seen_a: list[Any] = []
        seen_b: list[Any] = []
        m["load_entry"].side_effect = [
            (_persistent_entry(64, seen_a), None),
            (_persistent_entry(128, seen_b), None),
        ]
        compiled_a = _fake_compiled([_param("a", [16, 16])], [])
        compiled_b = _fake_compiled([_param("b", [16, 16])], [])
        compiled_a._distributed_config = DistributedConfig(device_ids=[0, 1])
        compiled_b._distributed_config = DistributedConfig(device_ids=[0, 1])
        arg = DeviceTensor(0x1000, (16, 16), torch.float32)

        rt = DistributedWorker(
            [compiled_a, compiled_b],
            persistent=True,
            reset_persistent_windows=True,
        )
        rt.run(compiled_a, arg)
        rt.run(compiled_b, arg)
        rt.run(compiled_a, arg)
        rt.close()

        assert m["worker"].submit.call_count == 3
        assert [call["name"] for call in orch.allocate_calls] == ["p0:comm_d0", "p1:comm_d0"]
        assert seen_a[0] is seen_a[1]
        assert seen_a[0] is not seen_b[0]
        # Only program A is reused; program B's first use needs no reset.
        assert [(worker, size) for worker, _dst, _src, size in orch.copy_calls] == [
            (0, 64),
            (1, 64),
        ]
        # Isolation does not change final ownership: each retained domain is
        # released exactly once when the shared persistent worker closes.
        assert [handle.release_count for handle in orch.handles] == [1, 1]

    def test_domain_release_error_reaches_close(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        rt(DeviceTensor(0x1000, (16, 16), torch.float32))
        m["worker"]._execute_pending_domain_releases.side_effect = RuntimeError(
            "persistent domain release failed"
        )

        with pytest.raises(RuntimeError, match="persistent domain release failed"):
            rt.close()

        assert orch.handles[0].release_count == 1
        m["worker"].close.assert_called_once_with()

    def test_unfreed_domain_release_reaches_close(self, patched_setup):
        m = patched_setup
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        rt(DeviceTensor(0x1000, (16, 16), torch.float32))
        m["worker"]._execute_pending_domain_releases.side_effect = None

        with pytest.raises(RuntimeError, match="did not free.*p0:comm_d0"):
            rt.close()

        assert not orch.handles[0].freed
        m["worker"].close.assert_called_once_with()

    def test_dispatch_error_reaches_caller_and_releases_domain(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]

        def failing_entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, _keep
            assert _domain_provider is not None
            with _domain_provider(
                name="comm_d0",
                workers=[*range(world_size)],
                window_size=64,
                buffers=[SimpleNamespace(name="signal", dtype="opaque", count=4, nbytes=4)],
            ):
                raise RuntimeError("persistent dispatch failed")

        m["load_entry"].return_value = (failing_entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = DeviceTensor(0x1000, (16, 16), torch.float32)

        with pytest.raises(RuntimeError, match="persistent dispatch failed"):
            rt(arg)
        rt.close()

        # The failed submission reaches the caller before close releases the
        # retained domain.
        assert orch.handles[0].release_count == 1
        assert m["worker"].submit.call_count == 1

    def test_dispatch_error_reports_later_teardown_error_during_close(self, patched_setup):
        m = patched_setup
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (fn(orch, None, None), _ImmediateNativeHandle())[1]

        def failing_entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, _keep, world_size, _domain_provider
            raise RuntimeError("persistent dispatch failed")

        m["load_entry"].return_value = (failing_entry, None)
        m["worker"]._execute_pending_domain_releases.side_effect = RuntimeError("persistent teardown failed")
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)

        with pytest.raises(RuntimeError, match="persistent dispatch failed"):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32))

        # A prior accepted handle may still use the retained domain, so domain
        # teardown stays at close instead of being pulled into submission error
        # handling. Both failures remain observable at their owning boundary.
        with pytest.raises(RuntimeError, match="persistent teardown failed"):
            rt.close()

    def test_dispatch_error_waits_for_request_worker_cleanup(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        request_finalizer_started = threading.Event()
        allow_request_finalizer_to_finish = threading.Event()
        native = _ControlledNativeHandle()

        def worker_submit(fn):
            del fn
            return native

        m["worker"].submit.side_effect = worker_submit

        def failing_entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, _keep, world_size, _domain_provider
            raise RuntimeError("persistent dispatch failed before cleanup")

        m["load_entry"].return_value = (failing_entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        caller_done = threading.Event()
        errors: list[BaseException] = []

        def call_worker() -> None:
            try:
                rt(DeviceTensor(0x1000, (16, 16), torch.float32))
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)
            finally:
                caller_done.set()

        caller = threading.Thread(target=call_worker)
        caller.start()
        assert native.result_started.wait(timeout=2)
        request_finalizer_started.set()
        # A failing request may already have submitted device work. Its caller
        # must not observe completion while the native handle is still finalizing.
        assert not caller_done.is_set()

        allow_request_finalizer_to_finish.set()
        native.complete(RuntimeError("persistent dispatch failed before cleanup"))
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert caller_done.is_set()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "persistent dispatch failed before cleanup"
        rt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
