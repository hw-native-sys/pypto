# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the simulator trace cleaning tool."""

import pytest
from pypto.tools import clean_sim_trace


def _make_bin(blocks: list[tuple[int, bytes]]) -> bytes:
    """Build a visualize_data.bin buffer from (block_type, payload) pairs."""
    out = b""
    for btype, payload in blocks:
        pad = (-len(payload)) % 4
        body = payload + b"\x00" * pad
        out += clean_sim_trace._HEADER.pack(len(body), btype, pad, 0, clean_sim_trace._MAGIC)
        out += body
    return out


def test_iter_blocks_roundtrip():
    trace_json = b'{"traceEvents":[]}'
    api_json = b'{"Cores":[]}'
    buf = _make_bin([(clean_sim_trace._TYPE_TRACE, trace_json), (clean_sim_trace._TYPE_API_INSTR, api_json)])
    assert list(clean_sim_trace.iter_blocks(buf)) == [
        (clean_sim_trace._TYPE_TRACE, trace_json),
        (clean_sim_trace._TYPE_API_INSTR, api_json),
    ]


def test_iter_blocks_rejects_corrupt():
    bad_magic = clean_sim_trace._HEADER.pack(4, clean_sim_trace._TYPE_TRACE, 0, 0, 0x00) + b"abcd"
    with pytest.raises(ValueError, match="corrupt"):
        list(clean_sim_trace.iter_blocks(bad_magic))
    oversize = (
        clean_sim_trace._HEADER.pack(9999, clean_sim_trace._TYPE_TRACE, 0, 0, clean_sim_trace._MAGIC)
        + b"abcd"
    )
    with pytest.raises(ValueError, match="corrupt"):
        list(clean_sim_trace.iter_blocks(oversize))


def test_source_block_path_skipped():
    src = b"x" * clean_sim_trace._SOURCE_PATH_LEN + b'{"src":1}'
    buf = _make_bin([(clean_sim_trace._TYPE_SOURCE, src)])
    assert list(clean_sim_trace.iter_blocks(buf)) == [
        (clean_sim_trace._TYPE_SOURCE, b'{"src":1}'),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
