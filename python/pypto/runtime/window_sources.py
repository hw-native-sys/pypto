# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bind runtime files to compiler-declared Host communication windows."""

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

_READ_CHUNK_BYTES = 8 * 1024 * 1024


def normalize_window_sources(
    sources: Mapping[str, Sequence[BinaryIO]] | None,
) -> dict[str, tuple[BinaryIO, ...]]:
    """Snapshot source bindings without reading or taking ownership of files."""
    result: dict[str, tuple[BinaryIO, ...]] = {}
    for key, files in (sources or {}).items():
        if type(key) is not str or not key:
            raise ValueError(f"window_sources keys must be non-empty strings, got {key!r}")
        rank_files = tuple(files)
        for rank, source in enumerate(rank_files):
            if not isinstance(source, (io.BufferedIOBase, io.RawIOBase)):
                raise TypeError(f"window_sources[{key!r}][{rank}] must be an open binary file")
            if source.closed or not source.readable() or not source.seekable():
                raise ValueError(f"window_sources[{key!r}][{rank}] must be open, readable and seekable")
        result[key] = rank_files
    return result


@dataclass(frozen=True)
class _WindowInitializer:
    bindings: tuple[tuple[str, str, int], ...]
    sources: Mapping[str, Sequence[BinaryIO]]
    rank_count: int

    def __call__(self, domain: Any) -> None:
        # Simpler calls this once after all rank mappings are registered, before
        # publishing the new domain. A retained-domain hit never enters here.
        for buffer_name, key, logical_nbytes in self.bindings:
            offset, allocated_nbytes = domain.buffer_range(buffer_name)
            if logical_nbytes < 0 or logical_nbytes > allocated_nbytes:
                raise ValueError(
                    f"Host window {buffer_name!r} requires {logical_nbytes} bytes, "
                    f"but its allocation contains {allocated_nbytes} bytes"
                )
            for rank in range(self.rank_count):
                source = cast(io.BufferedIOBase | io.RawIOBase, self.sources[key][rank])
                position = source.tell()
                try:
                    source.seek(0)
                    buffer = domain.member(rank).buffers[buffer_name]
                    # shm.buf covers the entire rank allocation, not just this
                    # named buffer. Release every export even on short reads so
                    # Simpler can roll back the allocation after an exception.
                    with buffer.shm.buf[offset : offset + logical_nbytes] as target:
                        copied = 0
                        while copied < logical_nbytes:
                            end = min(copied + _READ_CHUNK_BYTES, logical_nbytes)
                            with target[copied:end] as chunk:
                                count = source.readinto(chunk)
                            if count is None or count <= 0:
                                raise EOFError(
                                    f"window_sources[{key!r}][{rank}] ended after {copied} bytes; "
                                    f"expected {logical_nbytes} bytes"
                                )
                            if count > end - copied:
                                raise ValueError(
                                    f"window_sources[{key!r}][{rank}] returned invalid read size"
                                )
                            copied += count
                finally:
                    source.seek(position)


def make_window_initializer(
    sources: Mapping[str, Sequence[BinaryIO]] | None,
    bindings: Sequence[tuple[str, str, int]],
    rank_count: int,
) -> _WindowInitializer:
    """Build the internal Simpler callback without accessing file contents."""
    sources = sources or {}
    for _buffer_name, key, _logical_nbytes in bindings:
        if key not in sources:
            raise ValueError(f"Host window init={key!r} requires DistributedWorker(window_sources=...)")
        if len(sources[key]) != rank_count:
            raise ValueError(
                f"window_sources[{key!r}] requires {rank_count} files in domain rank order, "
                f"got {len(sources[key])}"
            )
    return _WindowInitializer(tuple(bindings), sources, rank_count)
