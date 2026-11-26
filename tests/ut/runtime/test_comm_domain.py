# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logical layout, transactional creation and retained resource identity."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pypto import DataType
from pypto.runtime import DomainBufferSpec, DomainTensor
from pypto.runtime.comm_domain import DomainState, allocate_external_domain, build_domain_layout


@pytest.mark.parametrize(
    ("logical", "physical"), [(1, 64), (63, 64), (64, 64), (65, 128), (127, 128), (128, 128), (129, 192)]
)
def test_each_slot_is_padded(logical, physical):
    slots, size = build_domain_layout([DomainBufferSpec("x", (logical,), DataType.UINT8)])
    assert (slots[0].logical, slots[0].physical, size) == (logical, physical, physical)


def test_layout_checks_multiplication_padding_and_total():
    spec = DomainBufferSpec("x", (17,), DataType.FP32)
    assert build_domain_layout([spec])[0][0].logical == 68
    for limit in (67, 100):
        with pytest.raises(ValueError, match="size limit"):
            build_domain_layout([spec], limit)
    with pytest.raises(ValueError, match="size limit"):
        build_domain_layout([spec, DomainBufferSpec("y", (1,), DataType.UINT8)], 130)
    for shape in ((True,), (0,), (-1,), (2**63,)):
        with pytest.raises(ValueError):
            build_domain_layout([DomainBufferSpec("x", shape, DataType.INT32)])
    with pytest.raises(TypeError, match="domain.tensor"):
        DomainTensor()


class PartialReader(io.BytesIO):
    def readinto(self, view):
        return super().readinto(view[:7])


def test_native_specs_and_rank_order_initialization():
    CommDomainRef = pytest.importorskip("simpler.comm_domain").CommDomainRef

    worker = MagicMock()
    worker.dc.device_ids = [5, 3, 6]
    worker._external_domains = {}
    worker._next_external_domain = 0
    allocations = [bytearray([255]) * 256 for _ in range(2)]
    names, sizes, offsets = ("a", "b", "c"), (1, 64, 65), (0, 64, 128)
    sources = {
        k: [PartialReader(bytes([10 * r + i]) * n) for r in range(2)]
        for i, (k, n) in enumerate(zip(names, sizes, strict=True))
    }
    specs = [DomainBufferSpec(k, (n,), DataType.UINT8) for k, n in zip(names, sizes, strict=True)]
    native = MagicMock()
    native.ref = CommDomainRef(b"instance", 2, 1)
    native.buffer_range.side_effect = dict(
        zip(names, zip(offsets, (64, 64, 128), strict=True), strict=True)
    ).__getitem__
    native.member.side_effect = lambda r: SimpleNamespace(
        buffers={k: SimpleNamespace(shm=SimpleNamespace(buf=memoryview(allocations[r]))) for k in names}
    )
    orch = MagicMock()

    def allocate(**kwargs):
        assert kwargs["members"] == (2, 0)
        assert kwargs["retain_after_run"]
        assert kwargs["window_size"] == 256
        assert [(s.name, s.count, s.nbytes) for s in kwargs["buffers"]] == [
            ("a", 1, 64),
            ("b", 64, 64),
            ("c", 65, 128),
        ]
        kwargs["initializer"](native)
        return native

    orch.allocate_domain.side_effect = allocate
    worker._w.submit.side_effect = lambda cb: (cb(orch, None, None), MagicMock())[1]
    domain = allocate_external_domain(worker, [2, 0], specs, "host", (), sources)
    assert domain.tensor("b").buffer_index == 1
    assert domain.members == (2, 0)
    assert worker._external_domains == {native.ref: domain}
    for r, data in enumerate(allocations):
        assert data == (
            bytes([r * 10])
            + bytes([255]) * 63
            + bytes([r * 10 + 1]) * 64
            + bytes([r * 10 + 2]) * 65
            + bytes([255]) * 63
        )
        assert all(sources[n][r].tell() == 0 for n in names)
    sources["c"][1] = PartialReader(b"short")
    with pytest.raises(EOFError):
        allocate_external_domain(worker, [2, 0], specs, "host", (), sources)
    assert worker._w.release_buffer.call_count == 2
    assert worker._external_domains == {native.ref: domain}


def test_cleanup_debt_keeps_tokens_until_native_free_is_confirmed():
    from pypto.runtime.comm_domain import CommDomain  # noqa: PLC0415

    worker = MagicMock()
    slots, _ = build_domain_layout([DomainBufferSpec("x", (1,), DataType.UINT8)])
    domain = CommDomain(worker, (0,), slots, "device", ())
    native = MagicMock(freed=False)
    domain._native = native
    domain._tokens[0] = token = object()
    with pytest.raises(RuntimeError, match="has not completed"):
        domain._free()
    assert domain._state is DomainState.CLEANUP_FAILED
    assert domain._tokens[0] is token
    worker._w.release_buffer.assert_not_called()
    native.freed = True
    domain._free()
    assert domain._state is DomainState.RELEASED
    worker._w.release_buffer.assert_called_once_with(token)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
