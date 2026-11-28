# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Portable data contracts independent of the compiler extension and NPU runtime."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load(name, filename):
    path = Path(__file__).resolve().parents[3] / "python/pypto/runtime" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout = _load("_pypto_contract_layout", "domain_layout.py")
tensors = _load("_pypto_contract_tensors", "device_tensor.py")
staging = _load("_pypto_contract_staging", "staging.py")


def test_scalar_layout_alignment_and_abi_validation(tmp_path):
    directory = tmp_path / "orchestration"
    directory.mkdir()
    source = directory / "host_orch.py"
    specs = (
        {
            "name": "d0",
            "workers": "[*range(world_size)]",
            "window_size": "((tokens * 4 + 63) // 64) * 64 + 64",
            "buffers": (("input", "tokens * 4", "((tokens * 4 + 63) // 64) * 64"),),
            "memory_source": "device",
            "engines": (),
        },
    )
    source.write_text(f"entry._pypto_domain_layout_specs = {specs!r}", encoding="utf-8")
    with source.open("a", encoding="utf-8") as stream:
        stream.write("\nentry._pypto_comm_abi = (1104, 0, 16, 528)\n")
    abi = layout.read_comm_abi(source)
    layout.validate_comm_abi(abi, (1104, 0, 16, 528))
    with pytest.raises(RuntimeError, match="ABI differs"):
        layout.validate_comm_abi(abi, (1088, 0, 16, 528))
    (resolved,) = layout.resolve_layouts(layout.read_layout_specs(source), 4, {"tokens": 5})
    assert resolved.workers == (0, 1, 2, 3)
    assert resolved.window_size == 64
    assert resolved.buffers == (layout.DomainBufferLayout("input", 20, 64, 0),)
    with pytest.raises(ValueError, match="bindings"):
        layout.resolve_layouts(specs, 4, {})
    with pytest.raises(ValueError, match="bindings"):
        layout._evaluate("__import__('os').getcwd()", {})
    # The codegen cold-reload test covers source tampering. Keep traversal
    # rejection here, where the manifest is supplied directly.
    with pytest.raises(ValueError, match="outside"):
        layout.validate_artifact_files(tmp_path, {"../missing.py": "0" * 64})


def test_nonleading_shard_axis_preserves_geometry_and_owner():
    shards = tuple(tensors.DeviceTensor(0x1000 + 0x100 * rank, (3, 4), torch.float32) for rank in range(2))
    stacked = tensors.StackedDeviceTensor(shards, (3, 2, 4), (0, 1), shard_axis=1)
    assert stacked[:, 1] is shards[1]
    view = stacked[2, 1]
    assert (view.shape, view.strides, view.byte_offset, view.data_ptr) == ((4,), (1,), 32, 0x1120)
    partial = stacked[1:3, :]
    assert (partial.shape, partial.shard_axis) == ((2, 2, 4), 1)
    assert partial[1, 1].data_ptr == view.data_ptr
    sparse = stacked[::2, 1, ::2]
    assert (sparse.shape, sparse.strides) == ((2, 2), (8, 2))
    with pytest.raises(ValueError, match="positive-step"):
        stacked[::-1, 1]
    with pytest.raises(IndexError):
        stacked[:, 2]


@pytest.fixture
def backend(monkeypatch):
    tags = SimpleNamespace(INPUT=1, INOUT=2, OUTPUT=3, OUTPUT_EXISTING=4)

    def spec(name, dtype, count, nbytes):
        return SimpleNamespace(name=name, dtype=dtype, count=count, nbytes=nbytes)

    monkeypatch.setitem(
        sys.modules, "simpler.task_interface", SimpleNamespace(TensorArgType=tags, CommBufferSpec=spec)
    )
    monkeypatch.setitem(
        sys.modules,
        "simpler_setup.torch_interop",
        SimpleNamespace(torch_dtype_to_datatype=lambda dtype: dtype),
    )

    class Worker:
        world_size = 2
        _dispatch_submit_mu = threading.Lock()

        def allocate_domain(self, name, *, window_size, buffers):
            members = []
            for _rank in range(self.world_size):
                member = {}
                for item in buffers:
                    buffer = SimpleNamespace(data=torch.zeros(item.nbytes, dtype=torch.uint8))
                    buffer.tensor = lambda **kwargs: SimpleNamespace(**kwargs)
                    member[item.name] = buffer
                members.append(SimpleNamespace(buffers=member))
            domain = SimpleNamespace(name=name, member=lambda rank: members[rank], released=False)
            domain.release = lambda: setattr(domain, "released", True)
            self._owned_domains = {name: domain}
            return domain

        def copy_domain_to(self, domain, rank, name, host, *, byte_offset=0):
            domain.member(rank).buffers[name].data[byte_offset : byte_offset + host.numel()].copy_(host)

        def copy_domain_from(self, domain, rank, name, host, *, byte_offset=0):
            if getattr(self, "fail_download_rank", None) == rank:
                raise RuntimeError("download failed")
            host.copy_(domain.member(rank).buffers[name].data[byte_offset : byte_offset + host.numel()])

    return Worker(), tags


def test_staging_preserves_disjoint_outputs_and_rejects_cross_rank_aliases(backend):
    worker, tags = backend
    host = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    before = host.clone()
    invocation = staging._StagedInvocation(worker, (host,))
    invocation.upload()
    for rank in range(2):
        view = invocation.args[0][rank]._pypto_chip_view(rank, tags.OUTPUT)
        assert view.byte_offset == rank * 16
        data = invocation.domain.member(rank).buffers["storage0"].data
        data[rank * 16 : (rank + 1) * 16].view(torch.int32).add_(100)
    assert torch.equal(host, before)
    with pytest.raises(ValueError, match="cross-rank"):
        invocation.args[0][0]._pypto_chip_view(1, tags.INPUT)
    invocation.publish()
    assert torch.equal(host, before + 100)
    invocation.release()
    assert invocation.domain.released and not worker._owned_domains


def test_failed_output_download_does_not_publish_partial_results(backend):
    worker, tags = backend
    host = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    before = host.clone()
    invocation = staging._StagedInvocation(worker, (host,))
    invocation.upload()
    for rank in range(2):
        invocation.args[0][rank]._pypto_chip_view(rank, tags.OUTPUT)
        invocation.domain.member(rank).buffers["storage0"].data.fill_(0)
    worker.fail_download_rank = 1
    with pytest.raises(RuntimeError, match="download failed"):
        invocation.publish()
    assert torch.equal(host, before)
    assert not invocation.domain.released
