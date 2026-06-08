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
no real compile/fork. The reuse contract is observed by counting how often the
setup helpers vs. ``_dispatch`` run.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.ir.compiled_program import _ParamInfo
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import DeviceTensor
from pypto.runtime.distributed_runner import DistributedWorker


def _param(name: str, shape: list[int], direction: ParamDirection = ParamDirection.In) -> _ParamInfo:
    return _ParamInfo(name=name, direction=direction, shape=shape, dtype=DataType.FP32)


def _fake_compiled(param_infos, output_indices):
    """A minimal stand-in for DistributedCompiledProgram used by DistributedWorker."""
    compiled = MagicMock(name="DistributedCompiledProgram")
    compiled._get_metadata.return_value = (param_infos, output_indices, [])
    compiled._distributed_config = DistributedConfig()
    compiled.platform = "a2a3sim"
    return compiled


@pytest.fixture
def patched_setup():
    """Patch every setup helper so DistributedWorker() does no real work.

    Yields a dict of the mocks so individual tests can assert call counts.
    The worker mock records malloc/copy_to/free for alloc_tensor checks.
    """
    worker = MagicMock(name="Worker(level=3)")
    worker.chip_contexts = []
    # Device-memory ops route through the Orchestrator facade (worker._orch).
    worker._orch.malloc.return_value = 0xDEAD0000

    mod = "pypto.runtime.distributed_runner"
    chip_callables = ({"chip_orch": object()}, "rt_name")
    with (
        patch(f"{mod}._assemble_chip_callables", return_value=chip_callables) as assemble,
        patch(f"{mod}._load_orch_entry", return_value=(MagicMock(name="entry_fn"), None)) as load_entry,
        patch(f"{mod}._load_sub_worker_fns", return_value={}) as load_subs,
        patch(f"{mod}._load_required_callbacks", return_value=set()) as load_required,
        patch(f"{mod}._construct_worker", return_value=worker) as construct,
        patch(f"{mod}._register_callables", return_value=({}, {"chip_orch": 0})) as register,
        patch(f"{mod}._make_call_config", return_value=MagicMock(name="CallConfig")),
        patch(f"{mod}._dispatch") as dispatch,
    ):
        yield {
            "worker": worker,
            "assemble": assemble,
            "load_entry": load_entry,
            "load_subs": load_subs,
            "load_required": load_required,
            "construct": construct,
            "register": register,
            "dispatch": dispatch,
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
        # Hierarchy is forked eagerly so the device-memory API works before the
        # first dispatch (comm-less programs otherwise defer the fork to run()).
        m["worker"]._start_hierarchical.assert_called_once()

        a = DeviceTensor(0x1000, (128, 128), torch.float32)
        b = DeviceTensor(0x2000, (128, 128), torch.float32)
        rt(a, b)
        rt(a, b)
        rt(a, b)

        # Setup still once; dispatch ran per call.
        assert m["dispatch"].call_count == 3
        m["assemble"].assert_called_once()
        m["construct"].assert_called_once()
        assert m["worker"].init.call_count == 1
        rt.close()


class TestPerCallValidation:
    def test_accepts_device_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        rt(DeviceTensor(0x1000, (128, 128), torch.float32), DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["dispatch"].assert_called_once()
        # The merged tensors dict (5th positional arg of _dispatch) carries the inputs by name.
        tensors = patched_setup["dispatch"].call_args.args[2]
        assert set(tensors) == {"a", "b"}
        rt.close()

    def test_accepts_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        host_a = torch.zeros(128, 128, dtype=torch.float32).share_memory_()
        rt(host_a, DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["dispatch"].assert_called_once()
        rt.close()

    def test_rejects_non_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="shared memory"):
            rt(torch.zeros(128, 128), DeviceTensor(0x2000, (128, 128), torch.float32))
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

    def test_alloc_tensor_rejects_nonzero_worker_id(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(ValueError, match="worker_id=0"):
            rt.alloc_tensor((16, 16), torch.float32, worker_id=1)
        rt.close()


class TestLifecycle:
    def test_close_idempotent_and_closes_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        rt.close()  # second close is a no-op
        assert patched_setup["worker"].close.call_count == 1

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


class TestExplicitDispatchAPI:
    """The new ``run`` / ``register`` surface that mirrors ChipWorker.

    DistributedWorker.run() is an alias for ``__call__`` (existing dispatch
    path). register() returns a :class:`RegistrationHandle` whose call
    delegates to run().
    """

    def test_run_delegates_to_call(self, patched_setup):
        from pypto.runtime import RegistrationHandle  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [4]), _param("b", [4])], [])
        rt = DistributedWorker(compiled)

        a = torch.zeros(4).share_memory_()
        b = torch.zeros(4).share_memory_()
        rt.run(compiled, a, b)
        patched_setup["dispatch"].assert_called_once()

        # register() returns a usable handle.
        rt2 = DistributedWorker(compiled)
        h = rt2.register(compiled)
        assert isinstance(h, RegistrationHandle)
        assert h.compiled is compiled
        rt.close()
        rt2.close()

    def test_run_rejects_other_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        a = torch.zeros(4).share_memory_()
        with pytest.raises(ValueError, match="prepared from"):
            rt.run(compiled_b, a)
        rt.close()

    def test_register_rejects_other_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        with pytest.raises(ValueError, match="prepared from"):
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
        patched_setup["dispatch"].reset_mock()
        h(a, b)
        patched_setup["dispatch"].assert_called_once()
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
        assert t.data_ptr in rt._owned_tensors

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
