# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Gloo + torch_npu SHMEM window layout, views and context ABI."""

import struct

import pytest
import torch
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.shmem_gloo import (
    COMM_CONTEXT_SIZE,
    COMM_CTX_SLOT,
    ShmemWindow,
    align_up,
    carve_window_layout,
    pack_comm_context,
)


def _unpack_comm_context(blob: bytes) -> dict[str, object]:
    """Decode a packed ``CommContext`` for tests / debug dumps."""
    if len(blob) != COMM_CONTEXT_SIZE:
        raise ValueError(f"CommContext blob must be {COMM_CONTEXT_SIZE} bytes, got {len(blob)}")
    rank = struct.unpack_from("<I", blob, 0)[0]
    world = struct.unpack_from("<I", blob, 4)[0]
    win_size = struct.unpack_from("<Q", blob, 8)[0]
    bases = [struct.unpack_from("<Q", blob, 16 + i * 8)[0] for i in range(world)]
    return {"rank": rank, "world_size": world, "win_size": win_size, "window_bases": bases}


def test_align_up() -> None:
    assert align_up(1) == 64
    assert align_up(32) == 64
    assert align_up(33) == 64
    assert align_up(64) == 64
    assert align_up(65) == 128


def test_carve_window_layout_matches_codegen_alignment() -> None:
    offsets, total = carve_window_layout([2048, 2048, 8])
    assert offsets == [0, 2048, 4096]
    assert total == 4096 + 64


def test_context_slot_reserves_the_full_aligned_runtime_abi() -> None:
    offsets, total = carve_window_layout([COMM_CONTEXT_SIZE, 64])
    assert offsets == [0, 1152]
    assert total == 1216


def test_pack_comm_context_roundtrip() -> None:
    bases = [0x1000, 0x2000]
    blob = pack_comm_context(rank=1, world_size=2, win_size=4128, window_bases=bases)
    assert len(blob) == COMM_CONTEXT_SIZE == 1104
    decoded = _unpack_comm_context(blob)
    assert decoded["rank"] == 1
    assert decoded["world_size"] == 2
    assert decoded["win_size"] == 4128
    assert decoded["window_bases"] == bases


def test_comm_ctx_slot_is_first_carved_region() -> None:
    offsets, total = carve_window_layout([COMM_CONTEXT_SIZE, 2048])
    assert offsets[0] == 0
    assert offsets[1] == align_up(COMM_CONTEXT_SIZE)
    assert COMM_CTX_SLOT == "_comm_ctx"
    assert total == align_up(COMM_CONTEXT_SIZE) + 2048


def test_as_device_tensor_uses_slot_offset() -> None:
    window = ShmemWindow(
        tensor=None,
        handle=None,
        window_bytes=4096,
        local_base=0x1000,
        peer_bases=[0x1000, 0x2000],
        offsets={"data_buf": 32},
        device_ctx_tensor=None,
        device_ctx_ptr=0x1000,
    )
    dt = window.as_device_tensor("data_buf", (1, 8), torch.float16)
    assert isinstance(dt, DeviceTensor)
    assert dt.data_ptr == 0x1000 + 32
    assert dt.shape == (1, 8)
    assert dt.dtype == torch.float16
    with pytest.raises(ValueError, match="SHMEM slot"):
        window.as_device_tensor("data_buf", (4096,), torch.float16)


def test_torch_view_shares_slot_storage_and_checks_capacity() -> None:
    tensor = torch.zeros(128, dtype=torch.uint8)
    window = ShmemWindow(
        tensor=tensor,
        handle=None,
        window_bytes=128,
        local_base=tensor.data_ptr(),
        peer_bases=[tensor.data_ptr()],
        offsets={"a": 0, "b": 64},
        device_ctx_tensor=None,
        device_ctx_ptr=0,
    )
    view = window.as_torch_tensor("b", (2, 4), torch.float32)
    assert view.data_ptr() == tensor.data_ptr() + 64
    view.fill_(3)
    assert torch.equal(tensor[64:96].view(torch.float32), torch.full((8,), 3.0))
    with pytest.raises(ValueError, match="SHMEM slot"):
        window.as_torch_tensor("a", (17,), torch.float32)
    window.owner_pid = -1
    with pytest.raises(RuntimeError, match="forked"):
        window.as_torch_tensor("a", (1,), torch.float32)


@pytest.mark.parametrize("world_size", [1, 2, 64])
def test_pack_comm_context_runtime_abi(world_size: int) -> None:
    # Check literal runtime offsets independently of _unpack_comm_context, so
    # matching producer/decoder bugs cannot hide an ABI mismatch. Rank 63 also
    # catches windowsOut writes that spill into the workspace descriptors.
    bases = [0x100000000 + rank * 0x10000 for rank in range(world_size)]
    blob = pack_comm_context(world_size - 1, world_size, 8192, bases)
    assert len(blob) == 1104
    assert struct.unpack_from("<IIQ", blob, 0) == (world_size - 1, world_size, 8192)
    expected = tuple(bases) + (0,) * (64 - world_size)
    assert struct.unpack_from("<64Q", blob, 16) == expected
    assert struct.unpack_from("<64Q", blob, 528) == expected
    assert struct.unpack_from("<8Q", blob, 1040) == (0,) * 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
