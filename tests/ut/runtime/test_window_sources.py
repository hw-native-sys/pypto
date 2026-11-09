# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host file initialization uses logical buffer extents and releases CPU exports."""

import io
import mmap
from types import SimpleNamespace

import pytest
from pypto.runtime.window_sources import make_window_initializer, normalize_window_sources


class _Mapping:
    def __init__(self, size):
        self.data = mmap.mmap(-1, size)
        self.data[:] = b"!" * size

    @property
    def buf(self):
        return memoryview(self.data)


class _Domain:
    def __init__(self, ranks=2):
        self.mappings = [_Mapping(192) for _ in range(ranks)]

    def buffer_range(self, name):
        return {"first": (0, 64), "second": (64, 128)}[name]

    def member(self, rank):
        return SimpleNamespace(
            buffers={name: SimpleNamespace(shm=self.mappings[rank]) for name in ("first", "second")}
        )

    def close(self):
        for mapping in self.mappings:
            mapping.data.close()  # Raises BufferError if even an error traceback retains an export.


class _PartialFile(io.BytesIO):
    def readinto(self, target):
        with target[:3] as partial:
            return super().readinto(partial)


def test_rank_order_offsets_padding_partial_reads_and_cursor():
    domain = _Domain(8)
    sources = {
        "a": [_PartialFile(bytes([rank]) * 7) for rank in range(8)],
        "b": [_PartialFile(bytes([rank + 10]) * 83 + b"ignored") for rank in range(8)],
    }
    for files in sources.values():
        for source in files:
            source.seek(2)
    initializer = make_window_initializer(sources, (("first", "a", 7), ("second", "b", 83)), 8)
    try:
        initializer(domain)
        for rank, mapping in enumerate(domain.mappings):
            assert mapping.data[:] == bytes([rank]) * 7 + b"!" * 57 + bytes([rank + 10]) * 83 + b"!" * 45
        for files in sources.values():
            for source in files:
                assert source.tell() == 2
                assert not source.closed
    finally:
        domain.close()


def test_short_file_releases_exports_and_restores_cursor():
    domain = _Domain(1)
    source = io.BytesIO(b"short")
    source.seek(2)
    initializer = make_window_initializer({"table": [source]}, (("second", "table", 83),), 1)
    with pytest.raises(EOFError, match="table.*0.*5.*83") as error:
        initializer(domain)
    assert error.value.__traceback__ is not None
    assert source.tell() == 2
    domain.close()


def test_snapshot_does_not_read_or_own_files():
    source = io.BytesIO(b"data")
    source.seek(2)
    files = [source]
    normalized = normalize_window_sources({"table": files})
    files.clear()
    assert normalized["table"] == (source,)
    assert source.tell() == 2
    source.close()
    # Building another callback for a retained-domain hit must not touch closed files.
    make_window_initializer(normalized, (("first", "table", 4),), 1)


@pytest.mark.parametrize("sources,pattern", [({}, "requires DistributedWorker"), ({"table": []}, "1 files")])
def test_missing_or_wrong_rank_count(sources, pattern):
    with pytest.raises(ValueError, match=pattern):
        make_window_initializer(sources, (("first", "table", 4),), 1)


@pytest.mark.parametrize("source", ["path.bin", io.StringIO("text")])
def test_sources_require_open_binary_files(source):
    with pytest.raises(TypeError, match="open binary file"):
        normalize_window_sources({"table": [source]})


def test_closed_file_is_rejected_at_worker_binding():
    source = io.BytesIO()
    source.close()
    with pytest.raises(ValueError, match="open, readable and seekable"):
        normalize_window_sources({"table": [source]})
