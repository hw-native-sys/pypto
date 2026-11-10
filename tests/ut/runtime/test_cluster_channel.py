# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller-side direct session staging and bounded control protocol."""

import errno
import json
import socket
import struct
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from pypto.runtime import _cluster_channel as channel


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan"), -0.0])
def test_direct_staging_preserves_float_scalar_bits(tmp_path, value):
    with channel.PackedValues([value], [], tmp_path) as packed:
        # JSON is strict: special IEEE values must have an explicit wire encoding.
        request = json.loads(json.dumps(packed.request(), allow_nan=False))
        with channel.map_values(request, tmp_path) as values:
            assert struct.pack("!d", values[0]) == struct.pack("!d", value)


def test_direct_staging_preserves_storage_aliases_offsets_and_write_ranges(tmp_path):
    backing = torch.arange(16, dtype=torch.float32)
    first, output = backing[4:12], backing[8:16]
    with channel.PackedValues([first, output, 7, True], [1], tmp_path) as packed:
        request = packed.request()
        assert len(request["storages"]) == 1
        with channel.map_values(request, tmp_path) as values:
            assert values[0].untyped_storage().data_ptr() == values[1].untyped_storage().data_ptr()
            assert values[0].storage_offset() == 4
            assert values[1].storage_offset() == 8
            assert values[2:] == [7, True]
            assert torch.equal(values[0], first)
            values[0][:4].fill_(-999)
            values[1].add_(100)
        assert torch.equal(backing, torch.arange(16, dtype=torch.float32))
        packed.publish()
        assert torch.equal(backing[:8], torch.arange(8, dtype=torch.float32))
        assert torch.equal(backing[8:], torch.arange(8, 16, dtype=torch.float32) + 100)
    assert not list(tmp_path.glob("storage-*"))


def test_direct_failed_invocation_does_not_publish_partial_outputs(tmp_path):
    output = torch.zeros(8)
    with channel.PackedValues([output], [0], tmp_path) as packed:
        with channel.map_values(packed.request(), tmp_path) as values:
            values[0].fill_(123)
        # No success acknowledgement: the caller must not invoke publish().
    assert torch.equal(output, torch.zeros(8))


def test_direct_staging_rejects_noncontiguous_views_before_launch(tmp_path):
    with pytest.raises(ValueError, match="contiguous CPU"):
        channel.PackedValues([torch.zeros(4, 4).t()], [0], tmp_path)
    assert not list(tmp_path.glob("storage-*"))


def test_direct_mapping_rejects_escaped_paths_and_out_of_bounds_views(tmp_path):
    with channel.PackedValues([torch.ones(4)], [0], tmp_path) as packed:
        request = packed.request()
        request["storages"][0]["name"] = "../outside"
        with pytest.raises(ValueError, match="storage"):
            with channel.map_values(request, tmp_path):
                pass
        request = packed.request()
        request["values"][0]["offset"] = 10000
        with pytest.raises(ValueError, match="view"):
            with channel.map_values(request, tmp_path):
                pass


def test_direct_record_protocol_is_bounded_and_preserves_plain_values():
    left, right = socket.socketpair()
    try:
        channel.send_record(left, {"op": "run", "program": 2, "values": [1, 2.5, True, None]})
        assert channel.recv_record(right) == {"op": "run", "program": 2, "values": [1, 2.5, True, None]}
        left.sendall(struct.pack("!I", channel.MAX_RECORD_BYTES + 1))
        with pytest.raises(ValueError, match="limit"):
            channel.recv_record(right)
    finally:
        left.close()
        right.close()


def test_direct_controller_channel_rejects_wrong_session_authentication():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    key = bytes(range(32))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            server = pool.submit(channel.accept_session, listener, key, 2.0)
            with pytest.raises(ValueError, match="auth"):
                channel.connect_session(listener.getsockname(), b"x" * 32, 2.0)
            with pytest.raises(ValueError, match="auth"):
                server.result(timeout=3)
    finally:
        listener.close()


def test_direct_authenticated_channel_does_not_apply_startup_timeout_to_runs():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            server = pool.submit(channel.accept_session, listener, b"key", 2.0)
            client = channel.connect_session(listener.getsockname(), b"key", 2.0)
            accepted = server.result(timeout=3)
            try:
                assert client.gettimeout() is None
                assert accepted.gettimeout() is None
            finally:
                client.close()
                accepted.close()
    finally:
        listener.close()


def test_failed_mapping_lives_until_retained_tensor_is_released(tmp_path, monkeypatch):
    owners = []
    frombuffer = torch.frombuffer

    def track_owner(buffer, **kwargs):
        owners.append(weakref.ref(buffer))
        return frombuffer(buffer, **kwargs)

    monkeypatch.setattr(torch, "frombuffer", track_owner)
    retained = []
    with channel.PackedValues([torch.ones(2)], [], tmp_path) as packed:
        with pytest.raises(RuntimeError, match="failed invocation"):
            with channel.map_values(packed.request(), tmp_path) as values:
                retained.append(values[0])
                raise RuntimeError("failed invocation")
        assert owners[0]() is not None and not owners[0]().closed
        assert retained[0].tolist() == [1, 1]
        retained.clear()
        assert owners[0]() is None


def test_staging_accepts_a_typed_view_with_unused_trailing_bytes(tmp_path):
    storage = torch.arange(9, dtype=torch.uint8)
    view = storage[:8].view(torch.float32)
    with channel.PackedValues([view], [0], tmp_path) as packed:
        with channel.map_values(packed.request(), tmp_path) as values:
            assert torch.equal(values[0], view)
            values[0].fill_(3.0)
        packed.publish()
    assert view.tolist() == [3.0, 3.0]
    assert storage[8].item() == 8


@pytest.mark.parametrize("kind", ["conjugate", "negative"])
def test_staging_preserves_lazy_views_and_their_storage_aliases(tmp_path, kind):
    backing = torch.tensor([1 + 2j])
    view = backing.conj() if kind == "conjugate" else torch._neg_view(backing)
    with channel.PackedValues([backing, view], [1], tmp_path) as packed:
        with channel.map_values(packed.request(), tmp_path) as values:
            assert torch.equal(values[1], view)
            values[1].fill_(3 + 4j)
            expected = 3 - 4j if kind == "conjugate" else -3 - 4j
            assert values[0].item() == expected
        packed.publish()
    assert view.item() == 3 + 4j


def test_staging_removes_file_if_mmap_creation_fails(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(channel.mmap, "mmap", unavailable)
    with pytest.raises(OSError, match="no space"):
        channel.PackedValues([torch.zeros(8)], [], tmp_path)
    assert not list(tmp_path.glob("storage-*"))


def test_authentication_timeout_covers_the_whole_exchange(monkeypatch):
    now = 0.0

    class Stream:
        def settimeout(self, timeout):
            self.timeout = timeout

        def sendall(self, _value):
            pass

        def recv(self, _size):
            nonlocal now
            now += 0.4
            if self.timeout < 0.4:
                raise TimeoutError("receive deadline")
            return b"x"

        def close(self):
            pass

    class Listener:
        def settimeout(self, _timeout):
            pass

        def accept(self):
            return Stream(), ("127.0.0.1", 12345)

    monkeypatch.setattr(time, "monotonic", lambda: now)
    with pytest.raises(TimeoutError):
        channel.accept_session(Listener(), b"key", 1.0)


def test_staging_writes_the_payload_before_mapping(tmp_path, monkeypatch):
    import os  # noqa: PLC0415

    real_mmap = channel.mmap.mmap

    def check_payload(descriptor, size):
        assert os.pread(descriptor, size, 0) == bytes([1, 2, 3])
        return real_mmap(descriptor, size)

    monkeypatch.setattr(channel.mmap, "mmap", check_payload)
    with channel.PackedValues([torch.tensor([1, 2, 3], dtype=torch.uint8)], [], tmp_path):
        pass
