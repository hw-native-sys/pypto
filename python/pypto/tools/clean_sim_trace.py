# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Convert the operator simulator's ``visualize_data.bin`` dump into a clean,
Chrome-viewable AI-core pipeline trace.

The dump is a flat sequence of length-prefixed JSON blocks. This tool rebuilds
the ``TRACE`` block into a de-cluttered Chrome Trace Event JSON and reshapes the
``API_INSTR`` block into a per-core metrics sidecar.
"""

import struct
from collections.abc import Iterator

# --- visualize_data.bin block container format ---------------------------------
# Header: contentSize:u64, type:u8, padding:u8, instrVersion:u8, reserve:u8.
_HEADER = struct.Struct("<QBBBB")
_MAGIC = 0x5A  # the reserve byte; also the binary-format magic
_TYPE_SOURCE, _TYPE_TRACE, _TYPE_API_INSTR = 1, 2, 4
_SOURCE_PATH_LEN = 4096  # SOURCE blocks prefix the payload with a fixed-size path


def iter_blocks(data: bytes) -> Iterator[tuple[int, bytes]]:
    """Decode the block container of a ``visualize_data.bin`` buffer.

    Args:
        data: Raw bytes of a ``visualize_data.bin`` file.

    Yields:
        ``(block_type, payload)`` tuples. ``SOURCE`` blocks have their fixed
        4096-byte path prefix removed; every payload has trailing 4-byte
        alignment padding stripped.

    Raises:
        ValueError: A block header is corrupt (bad magic or oversized length).
    """
    off = 0
    while off + _HEADER.size <= len(data):
        size, btype, padding, _instr_ver, reserve = _HEADER.unpack_from(data, off)
        body = off + _HEADER.size
        if reserve != _MAGIC or size > len(data) - body:
            raise ValueError(f"corrupt block at offset {off}: size={size}, reserve={reserve:#x}")
        payload = data[body : body + size]
        if btype == _TYPE_SOURCE:
            payload = payload[_SOURCE_PATH_LEN:]
        yield btype, payload[: len(payload) - padding]
        off = body + size
