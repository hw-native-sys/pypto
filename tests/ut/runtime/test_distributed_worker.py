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

import sys
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
from pypto.runtime.distributed_runner import (
    DistributedWorker,
    _assemble_chip_callables,
    _clear_dfx_dispatch_dirs,
    _make_call_config,
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
        patch(f"{mod}._make_call_config", return_value=MagicMock(name="CallConfig")) as make_call_config,
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
            "make_call_config": make_call_config,
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


class TestPerTaskRingSizing:
    """A per-dispatch ``RunConfig`` sizes that dispatch's runtime ring buffers.

    ``_make_call_config`` runs once at construction to build the program's
    shared baseline. With no per-dispatch ``config`` that baseline is reused
    (no rebuild); a ``RunConfig`` triggers a fresh rebuild from the program's
    ``DistributedConfig`` plus the ring overrides, for this dispatch only.
    """

    # ``_dispatch(w, entry_fn, tensors, chip_cids, sub_ids, call_config, ...)``
    _CALL_CONFIG_ARG = 5

    def test_no_config_reuses_prepared_baseline(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        # Construction builds the baseline exactly once.
        assert m["make_call_config"].call_count == 1
        baseline = m["make_call_config"].return_value

        rt(DeviceTensor(0x1000, (16, 16), torch.float32))

        # No per-dispatch config → baseline reused, no rebuild, and the prepared
        # config is what reaches _dispatch.
        assert m["make_call_config"].call_count == 1
        assert m["dispatch"].call_args.args[self._CALL_CONFIG_ARG] is baseline
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
        assert m["dispatch"].call_args.args[self._CALL_CONFIG_ARG] is m["make_call_config"].return_value
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

    def test_scalar_param_forwarded_as_is(self, patched_setup):
        # Scalar params (shape=None, e.g. seq_len) bypass tensor validation and
        # are forwarded verbatim to the entry — common in serving dispatch.
        scalar = _ParamInfo(name="seq_len", direction=ParamDirection.In, shape=None, dtype=DataType.FP32)
        compiled = _fake_compiled([scalar, _param("kv", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt(7, DeviceTensor(0x1000, (16, 16), torch.float32))
        tensors = patched_setup["dispatch"].call_args.args[2]
        assert tensors["seq_len"] == 7
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
        assert m["dispatch"].call_args.args[1] is entry_b
        rt.run(prog_a, a)
        assert m["dispatch"].call_args.args[1] is entry_a
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

    def test_single_program_list_keeps_call_shortcut(self, patched_setup):
        # A one-element list is what ``compiled.prepare()`` builds; the
        # ``rt(*args)`` shortcut must keep working for it.
        prog = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker([prog])
        assert rt._multi_program is False
        rt(torch.zeros(4).share_memory_())
        patched_setup["dispatch"].assert_called_once()
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
        # Both programs take a same-shaped KV param; one resident DeviceTensor
        # is dispatched through both (the serving KV-cache sharing contract).
        prog_a = _fake_compiled([_param("kv", [16, 16])], [])
        prog_b = _fake_compiled([_param("kv", [16, 16])], [])
        rt = DistributedWorker([prog_a, prog_b])

        kv = DeviceTensor(0x5000, (16, 16), torch.float32)
        rt.run(prog_a, kv)
        rt.run(prog_b, kv)

        assert m["dispatch"].call_count == 2
        for call in m["dispatch"].call_args_list:
            assert call.args[2]["kv"] is kv  # same pointer in both tensor maps
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
        assert m["dispatch"].call_args.args[1] is entry_a
        h_b(torch.zeros(8).share_memory_())
        assert m["dispatch"].call_args.args[1] is entry_b

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
            ({"chip_orch": object()}, "rt_name"),
            ({"chip_orch": object()}, "other_rt"),
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
        chip_callables, runtime_name = _assemble_chip_callables(compiled)

        assert set(chip_callables) == {"chip_a", "chip_b"}  # stray dir skipped
        assert runtime_name == "tensormap_and_ringbuffer"
        called_dirs = {call.args[0] for call in ca.call_args_list}
        assert called_dirs == {tmp_path / "next_levels" / "chip_a", tmp_path / "next_levels" / "chip_b"}
        assert all(call.args[1] == "a2a3sim" for call in ca.call_args_list)

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

    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, str]] = []
        # ``_submit_chip`` reads/writes this per-card dispatch counter on the
        # orch; declare it so the attribute is known to the type checker.
        self._dfx_dispatch_idx: dict[int, int] = {}

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

    def test_unconstrained_worker_not_suffixed(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        _submit_chip(orch, "chip", "ta", cfg, -1)
        assert orch.calls == [("chip", -1, "/work/dfx_outputs")]
        assert cfg.output_prefix == "/work/dfx_outputs"


class TestClearDfxDispatchDirs:
    """``_clear_dfx_dispatch_dirs`` drops stale ``rank*/d{k}`` dirs before a run."""

    def test_removes_only_dispatch_dirs(self, tmp_path):
        # A prior run left rank0/{d0,d1,d2} and rank1/d0; the current run will
        # only write d0, so the stale d1/d2 must be cleared. A sibling non-d{k}
        # dir (e.g. a future diagnostic) is preserved.
        dfx = tmp_path / "dfx_outputs"
        for d in ("rank0/d0", "rank0/d1", "rank0/d2", "rank1/d0", "rank0/keepme"):
            (dfx / d).mkdir(parents=True)
            (dfx / d / "l2_swimlane_records.json").write_text("{}", encoding="utf-8")

        _clear_dfx_dispatch_dirs(dfx)

        # All d{k} dirs gone...
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


class _BoolStrictCallConfig:
    """Fake ``CallConfig`` whose ``enable_dep_gen`` mirrors simpler's pybind setter.

    The real ``CallConfig.enable_dep_gen`` pybind overload accepts only ``bool``
    and raises ``TypeError`` on an ``int`` — exactly the crash issue #1952
    reproduces when the int ``enable_l2_swimlane`` CLI flag (0/1/2) leaks through
    the ``and``/``or`` chain unwrapped. ``bool`` is a subclass of ``int``, so
    ``isinstance(value, bool)`` matches the pybind behavior (rejects ``1``/``0``).
    """

    def __init__(self) -> None:
        self.block_dim: Any = None
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
