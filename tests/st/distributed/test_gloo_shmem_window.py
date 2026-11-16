# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Gloo + torch_npu SHMEM window helpers and live-input goldens."""

import struct

import pypto.language.distributed as pld
import pytest
import torch
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.shmem_gloo import (
    COMM_CONTEXT_SIZE,
    COMM_CTX_SLOT,
    ShmemWindow,
    align_up,
    allreduce_sum_expected,
    carve_window_layout,
    pack_comm_context,
    peer_copy_expected,
    unpack_comm_context,
)


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


def test_pack_comm_context_roundtrip() -> None:
    bases = [0x1000, 0x2000]
    blob = pack_comm_context(rank=1, world_size=2, win_size=4128, window_bases=bases)
    assert len(blob) == COMM_CONTEXT_SIZE == 1088
    decoded = unpack_comm_context(blob)
    assert decoded["rank"] == 1
    assert decoded["world_size"] == 2
    assert decoded["win_size"] == 4128
    assert decoded["window_bases"] == bases


def test_peer_copy_golden_from_live_inputs() -> None:
    mine = torch.arange(8, dtype=torch.float16)
    peer = torch.arange(8, 16, dtype=torch.float16)
    got = peer_copy_expected(mine, peer)
    assert torch.equal(got, peer)
    assert not torch.equal(got, mine)


def test_import_window_buffer_is_exported() -> None:
    assert hasattr(pld, "import_window_buffer")
    assert hasattr(pld.tensor, "import_window_buffer")


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


def test_allreduce_sum_golden_from_live_inputs() -> None:
    mine = torch.arange(8, dtype=torch.float16)
    peer = torch.ones(8, dtype=torch.float16)
    got = allreduce_sum_expected(mine, peer)
    assert torch.allclose(got, mine + peer)


@pytest.mark.parametrize("world_size", [1, 2, 64])
def test_pack_comm_context_runtime_abi(world_size: int) -> None:
    # Check literal runtime offsets independently of unpack_comm_context, so
    # matching producer/decoder bugs cannot hide an ABI mismatch. Rank 63 also
    # catches windowsOut writes that spill into the workspace descriptors.
    bases = [0x100000000 + rank * 0x10000 for rank in range(world_size)]
    blob = pack_comm_context(world_size - 1, world_size, 8192, bases)
    assert len(blob) == 1088
    assert struct.unpack_from("<IIQ", blob, 0) == (world_size - 1, world_size, 8192)
    expected = tuple(bases) + (0,) * (64 - world_size)
    assert struct.unpack_from("<64Q", blob, 16) == expected
    assert struct.unpack_from("<64Q", blob, 528) == expected
    assert struct.unpack_from("<6Q", blob, 1040) == (0,) * 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
