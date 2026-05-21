# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``DistributedRuntime`` (the ``prepare()`` reuse handle).

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
from pypto.runtime.distributed_runner import DistributedRuntime


def _param(name: str, shape: list[int], direction: ParamDirection = ParamDirection.In) -> _ParamInfo:
    return _ParamInfo(name=name, direction=direction, shape=shape, dtype=DataType.FP32)


def _fake_compiled(param_infos, output_indices):
    """A minimal stand-in for DistributedCompiledProgram used by DistributedRuntime."""
    compiled = MagicMock(name="DistributedCompiledProgram")
    compiled._get_metadata.return_value = (param_infos, output_indices, [])
    compiled._distributed_config = DistributedConfig()
    compiled.platform = "a2a3sim"
    return compiled


@pytest.fixture
def patched_setup():
    """Patch every setup helper so DistributedRuntime() does no real work.

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
        patch(f"{mod}._build_chip_bootstrap", return_value=(None, None)) as bootstrap,
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
            "bootstrap": bootstrap,
            "construct": construct,
            "register": register,
            "dispatch": dispatch,
        }


class TestSetupOnce:
    def test_setup_runs_once_dispatch_many(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])

        rt = DistributedRuntime(compiled)
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
        rt = DistributedRuntime(compiled)
        rt(DeviceTensor(0x1000, (128, 128), torch.float32), DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["dispatch"].assert_called_once()
        # The merged tensors dict (5th positional arg of _dispatch) carries the inputs by name.
        tensors = patched_setup["dispatch"].call_args.args[2]
        assert set(tensors) == {"a", "b"}
        rt.close()

    def test_accepts_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedRuntime(compiled)
        host_a = torch.zeros(128, 128, dtype=torch.float32).share_memory_()
        rt(host_a, DeviceTensor(0x2000, (128, 128), torch.float32))
        patched_setup["dispatch"].assert_called_once()
        rt.close()

    def test_rejects_non_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedRuntime(compiled)
        with pytest.raises(TypeError, match="shared memory"):
            rt(torch.zeros(128, 128), DeviceTensor(0x2000, (128, 128), torch.float32))
        rt.close()

    def test_rejects_wrong_arg_count(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedRuntime(compiled)
        with pytest.raises(TypeError, match="expects 2 arguments"):
            rt(DeviceTensor(0x1000, (128, 128), torch.float32))
        rt.close()

    def test_validates_device_tensor_shape(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedRuntime(compiled)
        with pytest.raises(TypeError, match="shape"):
            rt(
                DeviceTensor(0x1000, (64, 64), torch.float32),  # wrong shape
                DeviceTensor(0x2000, (128, 128), torch.float32),
            )
        rt.close()


class TestDeviceMemoryApi:
    def test_alloc_tensor_forwards_malloc_and_copy(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedRuntime(compiled)
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
        rt = DistributedRuntime(compiled)
        with pytest.raises(ValueError, match="shared-memory"):
            rt.alloc_tensor((16, 16), torch.float32, init=torch.zeros(16, 16, dtype=torch.float32))
        # rolled back the malloc'd pointer.
        patched_setup["worker"]._orch.free.assert_called_once_with(0, 0xDEAD0000)
        rt.close()

    def test_alloc_tensor_rolls_back_on_copy_failure(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedRuntime(compiled)
        patched_setup["worker"]._orch.copy_to.side_effect = RuntimeError("boom")
        host = torch.zeros(16, 16, dtype=torch.float32).share_memory_()

        with pytest.raises(RuntimeError, match="boom"):
            rt.alloc_tensor((16, 16), torch.float32, init=host)

        # malloc'd pointer is freed on the failure path.
        patched_setup["worker"]._orch.free.assert_called_once_with(0, 0xDEAD0000)
        rt.close()

    def test_alloc_tensor_rejects_nonzero_worker_id(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedRuntime(compiled)
        with pytest.raises(ValueError, match="worker_id=0"):
            rt.alloc_tensor((16, 16), torch.float32, worker_id=1)
        rt.close()


class TestLifecycle:
    def test_close_idempotent_and_closes_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedRuntime(compiled)
        rt.close()
        rt.close()  # second close is a no-op
        assert patched_setup["worker"].close.call_count == 1

    def test_context_manager_closes(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with DistributedRuntime(compiled) as rt:
            assert rt is not None
        assert patched_setup["worker"].close.call_count == 1

    def test_call_after_close_raises(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedRuntime(compiled)
        rt.close()
        with pytest.raises(RuntimeError, match="after close"):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32))


class TestSubWorkerOverrides:
    def test_override_reaches_register(self, patched_setup):
        m = patched_setup
        placeholder, real = object(), object()
        m["load_subs"].return_value = {"sample_and_prepare": placeholder}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedRuntime(compiled, sub_worker_overrides={"sample_and_prepare": real})

        # _register_callables(w, sub_worker_fns, chip_callables): arg[1] is the merged set.
        passed = m["register"].call_args.args[1]
        assert passed == {"sample_and_prepare": real}
        rt.close()

    def test_no_override_passes_loaded_unchanged(self, patched_setup):
        m = patched_setup
        loaded = {"sample_and_prepare": object()}
        m["load_subs"].return_value = loaded
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedRuntime(compiled)

        assert m["register"].call_args.args[1] == loaded
        rt.close()

    def test_override_unknown_name_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].return_value = {"sample_and_prepare": object()}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(ValueError, match="not sub-workers"):
            DistributedRuntime(compiled, sub_worker_overrides={"typo": object()})


class TestMergeSubWorkerOverrides:
    def test_none_overrides_returns_loaded_identity(self):
        from pypto.runtime.distributed_runner import _merge_sub_worker_overrides  # noqa: PLC0415

        loaded = {"a": object()}
        assert _merge_sub_worker_overrides(loaded, None) is loaded
        assert _merge_sub_worker_overrides(loaded, {}) is loaded

    def test_valid_override_replaces(self):
        from pypto.runtime.distributed_runner import _merge_sub_worker_overrides  # noqa: PLC0415

        placeholder, real, other = object(), object(), object()
        loaded = {"a": placeholder, "b": other}
        merged = _merge_sub_worker_overrides(loaded, {"a": real})
        assert merged == {"a": real, "b": other}

    def test_unknown_name_raises_listing_available(self):
        from pypto.runtime.distributed_runner import _merge_sub_worker_overrides  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"not sub-workers.*Available sub-workers"):
            _merge_sub_worker_overrides({"a": object()}, {"b": object()})


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
