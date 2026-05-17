# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the simulator trace cleaning tool."""

import json

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


def test_parse_detail():
    assert clean_sim_trace._parse_detail("PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,") == {
        "PIPE": "MTE2",
        "TRIGGERPIPE": "VEC",
        "FLAGID": "0",
    }
    assert clean_sim_trace._parse_detail("") == {}


def test_build_sync_arrows_reanchored():
    insts = [
        {"name": "MOV_SRC_TO_DST_ALIGN", "ph": "X", "pid": "c0", "tid": "MTE2", "ts": 2.0, "dur": 0.4},
        {"name": "VADD", "ph": "X", "pid": "c0", "tid": "VECTOR", "ts": 3.0, "dur": 0.1},
    ]
    events = insts + [
        {
            "name": "SET_FLAG",
            "ph": "B",
            "pid": "c0",
            "tid": "MTE2",
            "ts": 2.4,
            "args": {"detail": "PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"},
        },
        {"name": "SET_FLAG", "ph": "E", "pid": "c0", "tid": "MTE2", "ts": 2.41, "args": {}},
        {
            "name": "WAIT_FLAG",
            "ph": "B",
            "pid": "c0",
            "tid": "VECTOR",
            "ts": 1.5,
            "args": {"detail": "PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"},
        },
        {"name": "WAIT_FLAG", "ph": "E", "pid": "c0", "tid": "VECTOR", "ts": 2.9, "args": {}},
    ]
    arrows, skipped = clean_sim_trace._build_sync_arrows(insts, events)
    assert skipped == 0
    assert len(arrows) == 2
    start = next(a for a in arrows if a["ph"] == "s")
    end = next(a for a in arrows if a["ph"] == "f")
    assert start["id"] == end["id"]
    assert start["cat"] == "sync" and end["bp"] == "e"
    assert start["tid"] == "MTE2" and start["ts"] == 2.0
    assert end["tid"] == "VECTOR" and end["ts"] == 3.0


def test_build_sync_arrows_unmatchable():
    insts = [{"name": "VADD", "ph": "X", "pid": "c0", "tid": "VECTOR", "ts": 3.0, "dur": 0.1}]
    events = insts + [
        {
            "name": "SET_FLAG",
            "ph": "B",
            "pid": "c0",
            "tid": "MTE2",
            "ts": 2.4,
            "args": {"detail": "PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"},
        },
        {"name": "SET_FLAG", "ph": "E", "pid": "c0", "tid": "MTE2", "ts": 2.41, "args": {}},
    ]
    arrows, skipped = clean_sim_trace._build_sync_arrows(insts, events)
    assert arrows == []
    assert skipped == 1


def _raw_trace() -> dict:
    """A synthetic raw trace covering noise lanes, scalar setup, pipeline, sync."""
    return {
        "displayTimeUnit": "ns",
        "profilingType": "op",
        "schemaVersion": 1,
        "traceEvents": [
            {
                "name": "thread_state_runnable",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "CACHEMISS",
                "ts": 1.0,
                "dur": 0.001,
                "args": {},
            },
            {
                "name": "JUMP",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "FLOWCTRL",
                "ts": 1.1,
                "dur": 0.001,
                "args": {},
            },
            {
                "name": "MOV_XD_IMM",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "SCALAR",
                "ts": 1.2,
                "dur": 0.001,
                "args": {},
            },
            {
                "name": "MOV_SRC_TO_DST_ALIGN",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "MTE2",
                "ts": 2.0,
                "dur": 0.4,
                "args": {"pc_addr": "0x10"},
            },
            {
                "name": "VADD",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "VECTOR",
                "ts": 3.0,
                "dur": 0.1,
                "args": {"pc_addr": "0x20"},
            },
            {
                "name": "MOV_SRC_TO_DST_ALIGN",
                "ph": "X",
                "pid": "core0.veccore0",
                "tid": "MTE3",
                "ts": 3.5,
                "dur": 0.1,
                "args": {"pc_addr": "0x30"},
            },
            {
                "name": "SET_FLAG",
                "ph": "B",
                "pid": "core0.veccore0",
                "tid": "MTE2",
                "ts": 2.4,
                "args": {"detail": "PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"},
            },
            {"name": "SET_FLAG", "ph": "E", "pid": "core0.veccore0", "tid": "MTE2", "ts": 2.41, "args": {}},
            {
                "name": "WAIT_FLAG",
                "ph": "B",
                "pid": "core0.veccore0",
                "tid": "VECTOR",
                "ts": 1.5,
                "args": {"detail": "PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"},
            },
            {"name": "WAIT_FLAG", "ph": "E", "pid": "core0.veccore0", "tid": "VECTOR", "ts": 2.9, "args": {}},
        ],
    }


def _x_lanes(trace: dict) -> set[str]:
    return {e["tid"] for e in trace["traceEvents"] if e["ph"] == "X"}


def test_rebuild_drops_noise():
    trace, _ = clean_sim_trace.rebuild_trace(_raw_trace())
    names = {e["name"] for e in trace["traceEvents"] if e["ph"] == "X"}
    assert "CACHEMISS" not in _x_lanes(trace)
    assert "FLOWCTRL" not in _x_lanes(trace)
    assert "SET_FLAG" not in names and "WAIT_FLAG" not in names
    assert {"MOV_SRC_TO_DST_ALIGN", "VADD"} <= names


def test_rebuild_scalar_flag():
    assert "SCALAR" not in _x_lanes(clean_sim_trace.rebuild_trace(_raw_trace())[0])
    kept, _ = clean_sim_trace.rebuild_trace(_raw_trace(), keep_scalar=True)
    assert "SCALAR" in _x_lanes(kept)


def test_rebuild_lane_order_and_naming():
    trace, _ = clean_sim_trace.rebuild_trace(_raw_trace())
    sort = {
        e["tid"]: e["args"]["sort_index"] for e in trace["traceEvents"] if e["name"] == "thread_sort_index"
    }
    assert sort["MTE2"] < sort["VECTOR"] < sort["MTE3"]
    names = {e["tid"]: e["args"]["name"] for e in trace["traceEvents"] if e["name"] == "thread_name"}
    assert names["MTE2"].startswith("MTE2")
    procs = [e for e in trace["traceEvents"] if e["name"] == "process_name"]
    assert procs and procs[0]["args"]["name"] == "core0.veccore0"


def test_rebuild_recolors_and_keeps_timestamps():
    trace, _ = clean_sim_trace.rebuild_trace(_raw_trace())
    vadd = next(e for e in trace["traceEvents"] if e["ph"] == "X" and e["name"] == "VADD")
    assert vadd["ts"] == 3.0 and vadd["dur"] == 0.1
    assert vadd["cname"] == "good"
    assert vadd["args"]["pc_addr"] == "0x20"


def test_rebuild_sync_arrows():
    trace, skipped = clean_sim_trace.rebuild_trace(_raw_trace())
    assert skipped == 0
    flows = [e for e in trace["traceEvents"] if e.get("cat") == "sync"]
    assert {e["ph"] for e in flows} == {"s", "f"}
    assert len({e["id"] for e in flows}) == 1


def test_reshape_metrics():
    api = {
        "Cores": ["c0", "c1"],
        "Instructions": [
            {
                "Address": "0x10",
                "Pipe": "VECTOR",
                "Cycles": [5, 7],
                "Vector Utilization Percentage": [12.5, 0.0],
            },
        ],
        "Instructions Dtype": {"Instructions": {"Cycles": 1}},
    }
    out = clean_sim_trace.reshape_metrics(api)
    assert out["cores"] == ["c0", "c1"]
    c0 = out["instructions"]["c0"][0]
    c1 = out["instructions"]["c1"][0]
    assert c0["address"] == "0x10" and c0["pipe"] == "VECTOR"
    assert c0["cycles"] == 5 and c1["cycles"] == 7
    assert c0["vector_utilization_percentage"] == 12.5
    assert out["column_types"] == {"Instructions": {"Cycles": 1}}


def test_main_end_to_end(tmp_path):
    trace = (
        b'{"traceEvents":[{"name":"VADD","ph":"X","pid":"c0","tid":"VECTOR","ts":1.0,"dur":0.1,"args":{}}]}'
    )
    api = b'{"Cores":["c0"],"Instructions":[{"Address":"0x1","Cycles":[3]}],"Instructions Dtype":{}}'
    buf = _make_bin([(clean_sim_trace._TYPE_TRACE, trace), (clean_sim_trace._TYPE_API_INSTR, api)])
    bin_path = tmp_path / "visualize_data.bin"
    bin_path.write_bytes(buf)

    assert clean_sim_trace.main([str(bin_path)]) == 0

    clean = json.loads((tmp_path / "trace.clean.json").read_text())
    assert any(e["ph"] == "X" and e["name"] == "VADD" for e in clean["traceEvents"])
    metrics = json.loads((tmp_path / "instr_metrics.json").read_text())
    assert metrics["instructions"]["c0"][0]["cycles"] == 3


def test_main_resolves_opprof_dir(tmp_path):
    sim_dir = tmp_path / "simulator"
    sim_dir.mkdir()
    buf = _make_bin([(clean_sim_trace._TYPE_TRACE, b'{"traceEvents":[]}')])
    (sim_dir / "visualize_data.bin").write_bytes(buf)
    assert clean_sim_trace.main([str(tmp_path)]) == 0
    assert (sim_dir / "trace.clean.json").is_file()


def test_main_missing_file(tmp_path, capsys):
    assert clean_sim_trace.main([str(tmp_path / "nope")]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_missing_trace_block(tmp_path, capsys):
    buf = _make_bin([(clean_sim_trace._TYPE_API_INSTR, b'{"Cores":[]}')])
    bin_path = tmp_path / "visualize_data.bin"
    bin_path.write_bytes(buf)
    assert clean_sim_trace.main([str(bin_path)]) == 1
    assert "no TRACE block" in capsys.readouterr().err


def test_main_missing_api_instr_block(tmp_path, capsys):
    buf = _make_bin([(clean_sim_trace._TYPE_TRACE, b'{"traceEvents":[]}')])
    bin_path = tmp_path / "visualize_data.bin"
    bin_path.write_bytes(buf)
    assert clean_sim_trace.main([str(bin_path)]) == 0
    assert (tmp_path / "trace.clean.json").is_file()
    assert not (tmp_path / "instr_metrics.json").exists()
    assert "no API_INSTR block" in capsys.readouterr().err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
