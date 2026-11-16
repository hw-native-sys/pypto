# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External-domain binding and dispatch plumbing, without opening an NPU."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pypto.runtime import RunConfig
from pypto.runtime.distributed_runner import DistributedWorker, _make_dispatch_orchestration
from pypto.runtime.shmem_gloo import ShmemWindow, make_shmem_domain_provider


@pytest.fixture
def provider_setup(monkeypatch):
    registrations = []

    class FakeBuffer:
        @staticmethod
        def wrap(**kwargs):
            handle = SimpleNamespace(**kwargs, closed=False)

            def close():
                handle.closed = True

            handle.close = close
            registrations.append(handle)
            return handle

    buffer_module = ModuleType("simpler.buffer")
    monkeypatch.setattr(buffer_module, "Buffer", FakeBuffer, raising=False)
    task_module = ModuleType("simpler.task_interface")
    monkeypatch.setattr(
        task_module, "ChipDomainContext", lambda **kwargs: SimpleNamespace(**kwargs), raising=False
    )
    monkeypatch.setattr(
        task_module, "CommDomainHandle", lambda **kwargs: SimpleNamespace(**kwargs), raising=False
    )
    monkeypatch.setitem(sys.modules, "simpler.buffer", buffer_module)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", task_module)
    window = ShmemWindow(
        tensor=None,
        handle=None,
        window_bytes=1216,
        local_base=0x1000,
        peer_bases=[0x1000, 0x2000],
        offsets={"_comm_ctx": 0, "data": 1088, "signal": 1152},
        device_ctx_tensor=None,
        device_ctx_ptr=0x1000,
    )
    worker = SimpleNamespace(level=2, device_location=object())
    return window, worker, registrations


def _spec(name="data__ssa_v2", nbytes=64):
    return SimpleNamespace(name=name, nbytes=nbytes, load_from_host=False, store_to_host=False)


def test_provider_binds_buffers_and_releases_registrations(provider_setup):
    window, worker, registrations = provider_setup
    provider = make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)
    handle = provider(name="external", workers=(0,), buffers=[_spec()])
    ctx = handle.contexts[0]
    assert ctx.device_ctx == window.device_ctx_ptr
    assert ctx.domain_size == 2
    assert ctx.buffers["data__ssa_v2"].address == window.local_base + 1088
    assert ctx.buffers["data__ssa_v2"].location is worker.device_location
    assert not registrations[0].closed
    handle._release_fn(handle)
    assert registrations[0].closed


def test_provider_rolls_back_partial_registration(provider_setup):
    window, worker, registrations = provider_setup
    provider = make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)
    with pytest.raises(KeyError, match="missing"):
        provider(buffers=[_spec(), _spec("missing")])
    assert registrations[0].closed


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"workspace_engines": ("urma",)}, "workspaces"),
        ({"workers": (0, 1)}, "local chip"),
        ({"window_size": 2048}, "exceeds"),
        ({"buffers": [_spec(nbytes=128)]}, "capacity"),
        ({"buffers": [_spec(nbytes=1)]}, "capacity"),
    ],
)
def test_provider_rejects_incompatible_requests(provider_setup, kwargs, match):
    window, worker, registrations = provider_setup
    provider = make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)
    with pytest.raises(ValueError, match=match):
        provider(**({"buffers": [_spec()]} | kwargs))
    assert registrations == []


def test_provider_rejects_l3_and_fork(provider_setup, monkeypatch):
    window, worker, _ = provider_setup
    worker.level = 3
    with pytest.raises(ValueError, match="L3"):
        make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)
    worker.level = 2
    provider = make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)
    pid = os.getpid()
    monkeypatch.setattr("pypto.runtime.shmem_gloo.os.getpid", lambda: pid + 1)
    with pytest.raises(RuntimeError, match="forked"):
        provider(buffers=[_spec()])
    with pytest.raises(RuntimeError, match="forked"):
        make_shmem_domain_provider(window, rank=0, world_size=2, simpler_worker=worker)


def test_domain_provider_is_runtime_only_and_reaches_generated_entry():
    def provider(**kwargs):
        return kwargs

    cfg = RunConfig(domain_provider=provider)
    assert cfg.run_options().domain_provider is provider
    assert "domain_provider" not in cfg.compile_kwargs()
    seen = {}

    def entry(*args, **kwargs):
        seen.update(kwargs)

    closure = _make_dispatch_orchestration(entry, {}, {}, {}, None, 1, [], domain_provider=provider)
    closure(SimpleNamespace(), None, None)
    assert seen["_domain_provider"] is provider


def test_prepared_dispatch_forwards_provider_and_persistent_rejects_it(monkeypatch):
    seen = {}

    def submit(*args, **kwargs):
        seen.update(kwargs)
        return "submitted"

    monkeypatch.setattr("pypto.runtime.distributed_runner._submit_dispatch", submit)
    worker = DistributedWorker.__new__(DistributedWorker)
    worker._persistent = False
    worker._w = object()
    worker._dfx_dispatch_idx = {}
    state = {"entry_fn": None, "chip_cids": {}, "sub_ids": {}, "device_nums": 1}

    def provider(**kwargs):
        return kwargs

    cfg = RunConfig(domain_provider=provider)
    assert worker._submit_prepared_native(state, {}, None, [cfg], []) == "submitted"
    assert seen["domain_provider"] is provider
    worker._persistent = True
    with pytest.raises(ValueError, match="persistent"):
        worker._submit_prepared_native(state, {}, None, [cfg], [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
