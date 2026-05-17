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


# --- sync-flag arrow reconstruction --------------------------------------------
# Pipeline names used in SET_FLAG/WAIT_FLAG detail strings differ from trace
# thread (tid) names; map the aliases.
_PIPE_ALIAS = {"VEC": "VECTOR"}


def _parse_detail(detail: str) -> dict[str, str]:
    """Parse a comma-separated ``KEY:VALUE,`` detail string into a dict."""
    out: dict[str, str] = {}
    for part in detail.split(","):
        if ":" in part:
            key, _, val = part.partition(":")
            out[key.strip()] = val.strip()
    return out


def _pair_flag_events(events: list[dict], name: str) -> list[dict]:
    """Pair the B/E phases of a flag op into begin/end records.

    Returns one record per matched pair with keys ``pid``, ``tid``, ``detail``
    (taken from the B phase), ``begin_ts`` and ``end_ts``.
    """
    open_stack: dict[tuple[str, str], list[dict]] = {}
    records: list[dict] = []
    for event in events:
        if event.get("name") != name:
            continue
        key = (event["pid"], event["tid"])
        if event.get("ph") == "B":
            open_stack.setdefault(key, []).append(event)
        elif event.get("ph") == "E" and open_stack.get(key):
            begin = open_stack[key].pop()
            records.append(
                {
                    "pid": event["pid"],
                    "tid": event["tid"],
                    "detail": begin.get("args", {}).get("detail", ""),
                    "begin_ts": begin["ts"],
                    "end_ts": event["ts"],
                }
            )
    return records


def _last_at_or_before(insts_sorted: list[dict], ts: float) -> dict | None:
    """Return the last instruction with ``ts`` <= the given timestamp."""
    found = None
    for inst in insts_sorted:
        if inst["ts"] <= ts:
            found = inst
        else:
            break
    return found


def _first_at_or_after(insts_sorted: list[dict], ts: float) -> dict | None:
    """Return the first instruction with ``ts`` >= the given timestamp."""
    for inst in insts_sorted:
        if inst["ts"] >= ts:
            return inst
    return None


def _build_sync_arrows(insts: list[dict], events: list[dict]) -> tuple[list[dict], int]:
    """Rebuild SET_FLAG/WAIT_FLAG pairs as re-anchored Chrome flow arrows.

    Args:
        insts: The kept instruction slices (used as arrow anchor points).
        events: All raw trace events (the SET_FLAG/WAIT_FLAG source).

    Returns:
        ``(flow_events, skipped_count)`` — one ``s``/``f`` flow-event pair per
        re-anchored flag, plus the count of flags that could not be anchored.
    """
    by_lane: dict[tuple[str, str], list[dict]] = {}
    for inst in insts:
        by_lane.setdefault((inst["pid"], inst["tid"]), []).append(inst)
    for lane in by_lane.values():
        lane.sort(key=lambda inst: inst["ts"])

    waits_by_key: dict[tuple[str, str], list[dict]] = {}
    for wait in _pair_flag_events(events, "WAIT_FLAG"):
        waits_by_key.setdefault((wait["pid"], wait["detail"]), []).append(wait)
    for wlist in waits_by_key.values():
        wlist.sort(key=lambda rec: rec["begin_ts"])

    arrows: list[dict] = []
    skipped = 0
    flow_id = 0
    for flag in sorted(_pair_flag_events(events, "SET_FLAG"), key=lambda rec: rec["begin_ts"]):
        detail = _parse_detail(flag["detail"])
        producer = _PIPE_ALIAS.get(detail.get("PIPE", ""), detail.get("PIPE", ""))
        consumer = _PIPE_ALIAS.get(detail.get("TRIGGERPIPE", ""), detail.get("TRIGGERPIPE", ""))
        wlist = waits_by_key.get((flag["pid"], flag["detail"]))
        if not wlist:
            skipped += 1
            continue
        wait = wlist.pop(0)
        prod = _last_at_or_before(by_lane.get((flag["pid"], producer), []), flag["begin_ts"])
        cons = _first_at_or_after(by_lane.get((flag["pid"], consumer), []), wait["end_ts"])
        if prod is None or cons is None:
            skipped += 1
            continue
        flow_id += 1
        label = f"{detail.get('PIPE', '?')}->{detail.get('TRIGGERPIPE', '?')} flag{detail.get('FLAGID', '?')}"
        arrows.append(
            {
                "ph": "s",
                "id": flow_id,
                "cat": "sync",
                "name": label,
                "pid": flag["pid"],
                "tid": producer,
                "ts": prod["ts"],
            }
        )
        arrows.append(
            {
                "ph": "f",
                "id": flow_id,
                "cat": "sync",
                "bp": "e",
                "name": label,
                "pid": flag["pid"],
                "tid": consumer,
                "ts": cons["ts"],
            }
        )
    return arrows, skipped


# --- clean trace rebuild -------------------------------------------------------
# Pipeline lanes in dataflow order (load -> compute -> store -> setup).
_PIPELINE_ORDER = {
    "MTE2": 0,
    "MTE1": 1,
    "CUBE": 2,
    "VECTOR": 3,
    "FIXPIPE": 4,
    "MTE3": 5,
    "SCALAR": 6,
}
_DROP_LANES = frozenset({"CACHEMISS", "FLOWCTRL", "ALL"})
_SYNC_NAMES = frozenset({"SET_FLAG", "WAIT_FLAG", "BAR"})
_LANE_LABEL = {
    "MTE2": "MTE2 (load GM->UB)",
    "MTE1": "MTE1 (load L1->L0)",
    "CUBE": "CUBE (matmul)",
    "VECTOR": "VECTOR (compute)",
    "FIXPIPE": "FIXPIPE (quant/out)",
    "MTE3": "MTE3 (store UB->GM)",
    "SCALAR": "SCALAR (setup)",
}
_LANE_CNAME = {
    "MTE2": "thread_state_iowait",
    "MTE1": "thread_state_iowait",
    "CUBE": "rail_response",
    "VECTOR": "good",
    "FIXPIPE": "yellow",
    "MTE3": "cq_build_passed",
    "SCALAR": "grey",
}


def rebuild_trace(raw: dict, keep_scalar: bool = False) -> tuple[dict, int]:
    """Rebuild a raw simulator trace into a clean AI-core pipeline view.

    Args:
        raw: The parsed ``TRACE`` block (a Chrome Trace Event JSON dict).
        keep_scalar: Keep the ``SCALAR`` setup lane (dropped by default).

    Returns:
        ``(clean_trace, skipped_arrows)`` — the rebuilt Chrome trace dict and
        the number of sync flags that could not be re-anchored.
    """
    events = raw.get("traceEvents", [])

    def lane_kept(tid: str) -> bool:
        if tid in _DROP_LANES:
            return False
        if tid == "SCALAR" and not keep_scalar:
            return False
        return True

    # Rules 1 + 2: keep only X (complete) instruction events on pipeline lanes.
    insts = [
        e
        for e in events
        if e.get("ph") == "X" and lane_kept(e.get("tid", "")) and e.get("name") not in _SYNC_NAMES
    ]

    out: list[dict] = []

    # Rule 3: process/thread metadata for a deterministic dataflow ordering.
    for proc_index, pid in enumerate(sorted({e["pid"] for e in insts})):
        out.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": pid}})
        out.append({"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": proc_index}})
        lanes = sorted(
            {e["tid"] for e in insts if e["pid"] == pid}, key=lambda tid: _PIPELINE_ORDER.get(tid, 99)
        )
        for tid in lanes:
            out.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": _LANE_LABEL.get(tid, tid)},
                }
            )
            out.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": _PIPELINE_ORDER.get(tid, 99)},
                }
            )

    # Rules 5 + 6: copy instruction slices, recolor by lane, timestamps verbatim.
    for e in insts:
        slice_ = {
            "name": e["name"],
            "ph": "X",
            "pid": e["pid"],
            "tid": e["tid"],
            "ts": e["ts"],
            "cname": _LANE_CNAME.get(e["tid"], "grey"),
        }
        if "dur" in e:
            slice_["dur"] = e["dur"]
        if "args" in e:
            slice_["args"] = e["args"]
        out.append(slice_)

    # Rule 4: rebuild SET_FLAG/WAIT_FLAG pairs as re-anchored flow arrows.
    arrows, skipped = _build_sync_arrows(insts, events)
    out.extend(arrows)

    clean = {
        "displayTimeUnit": "ns",
        "profilingType": raw.get("profilingType", "op"),
        "schemaVersion": raw.get("schemaVersion", 1),
        "traceEvents": out,
    }
    return clean, skipped
