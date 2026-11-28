# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise Gloo group selection and device synchronization with a CPU window."""

from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import shmem_gloo
from torch.distributed import _symmetric_memory as symm_mem


def test_world_group_and_target_device(monkeypatch):
    group = SimpleNamespace(group_name="world")
    dist = shmem_gloo.dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=group))
    monkeypatch.setattr(dist, "get_rank", lambda pg: 1 if pg is group else -1)
    monkeypatch.setattr(dist, "get_world_size", lambda pg: 2 if pg is group else -1)
    monkeypatch.setattr(dist, "get_backend", lambda pg: "gloo" if pg is group else "hccl")
    sync_calls = []
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=sync_calls.append), raising=False)

    enabled = []
    monkeypatch.setattr(symm_mem, "is_symm_mem_enabled_for_group", lambda name: False)
    monkeypatch.setattr(symm_mem, "enable_symm_mem_for_group", enabled.append)
    monkeypatch.setattr(symm_mem, "empty", torch.empty)

    def rendezvous(tensor, *, group):
        assert group == "world"
        return SimpleNamespace(buffer_ptrs=[1, tensor.data_ptr()])

    monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)

    def acquire(rank: int, size: int, group_name: str | None = None):
        return shmem_gloo.acquire_gloo_shmem_window(
            rank=rank,
            world_size=size,
            group_name=group_name,
            device="cpu",
            slot_names=("data",),
            slot_nbytes=(64,),
        )

    with pytest.raises(NotImplementedError, match="Subgroup allocation is unsupported"):
        acquire(1, 2, "subgroup")
    assert not enabled and not sync_calls
    with pytest.raises(ValueError, match="default WORLD Gloo process group"):
        acquire(2, 3)
    window = acquire(1, 2, "world")
    assert enabled == ["world"]
    assert sync_calls == [window.tensor.device]
    assert window.device_ctx_ptr == window.tensor.data_ptr()
    assert bytes(window.device_ctx_tensor.tolist()) == shmem_gloo.pack_comm_context(
        1, 2, window.window_bytes, window.peer_bases
    )
