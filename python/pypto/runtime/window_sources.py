# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bind runtime data sources to compiler-declared Host communication windows."""

import ctypes
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

import torch

_READ_CHUNK_BYTES = 8 * 1024 * 1024


def normalize_window_sources(
    sources: Mapping[str, Sequence[BinaryIO | torch.Tensor]] | None,
) -> dict[str, tuple[BinaryIO | torch.Tensor, ...]]:
    """Snapshot rank bindings without copying data or taking ownership of files."""
    result: dict[str, tuple[BinaryIO | torch.Tensor, ...]] = {}
    for key, rank_sources in (sources or {}).items():
        if type(key) is not str or not key:
            raise ValueError(f"window_sources keys must be non-empty strings, got {key!r}")
        bound_sources = tuple(rank_sources)
        for rank, source in enumerate(bound_sources):
            if isinstance(source, torch.Tensor):
                _validate_tensor_source(source, key, rank)
                continue
            if not isinstance(source, (io.BufferedIOBase, io.RawIOBase)):
                raise TypeError(
                    f"window_sources[{key!r}][{rank}] must be an open binary file or a contiguous CPU tensor"
                )
            if source.closed or not source.readable() or not source.seekable():
                raise ValueError(f"window_sources[{key!r}][{rank}] must be open, readable and seekable")
        result[key] = bound_sources
    return result


def _validate_tensor_source(source: torch.Tensor, key: str, rank: int) -> None:
    if (
        source.device.type != "cpu"
        or source.layout != torch.strided
        or not source.is_contiguous()
        or source.is_quantized
        or source.is_conj()
        or source.is_neg()
    ):
        raise ValueError(
            f"window_sources[{key!r}][{rank}] must be a contiguous CPU tensor with plain strided storage; "
            f"got device={source.device}, layout={source.layout}, dtype={source.dtype}"
        )


def _copy_source(source: BinaryIO | torch.Tensor, target: memoryview, key: str, rank: int) -> None:
    """Copy raw bytes into the logical destination; never stage an entire table."""
    if isinstance(source, torch.Tensor):
        # Recheck at creation: caller-owned tensors can have changed since binding.
        # data_ptr includes storage_offset. Copying bytes supports BF16/FP8 without
        # a NumPy conversion and avoids autograd or dtype conversion semantics.
        _validate_tensor_source(source, key, rank)
        nbytes = source.numel() * source.element_size()
        if nbytes < len(target):
            raise ValueError(
                f"window_sources[{key!r}][{rank}] has {nbytes} bytes; expected at least {len(target)}"
            )
        if target:
            address = ctypes.addressof(ctypes.c_char.from_buffer(target))
            ctypes.memmove(address, source.data_ptr(), len(target))
        return

    stream = cast(io.BufferedIOBase | io.RawIOBase, source)
    position = stream.tell()
    try:
        stream.seek(0)
        copied = 0
        while copied < len(target):
            end = min(copied + _READ_CHUNK_BYTES, len(target))
            with target[copied:end] as chunk:
                count = stream.readinto(chunk)
            if count is None or count <= 0:
                raise EOFError(
                    f"window_sources[{key!r}][{rank}] ended after {copied} bytes; "
                    f"expected {len(target)} bytes"
                )
            if count > end - copied:
                raise ValueError(f"window_sources[{key!r}][{rank}] returned invalid read size")
            copied += count
    finally:
        stream.seek(position)


@dataclass(frozen=True)
class _WindowInitializer:
    bindings: tuple[tuple[str, str, int], ...]
    sources: Mapping[str, Sequence[BinaryIO | torch.Tensor]]
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
                buffer = domain.member(rank).buffers[buffer_name]
                # shm.buf covers the entire rank allocation, not just this
                # named buffer. Release every export on error so Simpler can roll
                # back even while an exception traceback retains these frames.
                with buffer.shm.buf[offset : offset + logical_nbytes] as target:
                    _copy_source(self.sources[key][rank], target, key, rank)


def make_window_initializer(
    sources: Mapping[str, Sequence[BinaryIO | torch.Tensor]] | None,
    bindings: Sequence[tuple[str, str, int]],
    rank_count: int,
) -> _WindowInitializer:
    """Build the internal Simpler callback without accessing source contents."""
    sources = sources or {}
    for _buffer_name, key, _logical_nbytes in bindings:
        if key not in sources:
            raise ValueError(f"Host window init={key!r} requires DistributedWorker(window_sources=...)")
        if len(sources[key]) != rank_count:
            raise ValueError(
                f"window_sources[{key!r}] requires {rank_count} sources in domain rank order, "
                f"got {len(sources[key])}"
            )
    return _WindowInitializer(tuple(bindings), sources, rank_count)
