# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""RFC #2521 — AllToAllV benchmark harness (§8.1).

Times a prepared contiguous window plus one managed ``pld.tensor.all_to_all_v``
call. Stage / consume / L3→L2 dispatch are excluded from the official metric
(swimlane AIV gang span, fastest-rank mean). ``--impl managed-host`` and
``managed-l2`` share the same kernel source; report those two as separate
columns, never combined.

Official NPU sweep is a2a3 + persistent workers. Sim Docker smoke
(``--smoke`` / ``--rounds 2 --warmup 0``) only validates that the harness
produces schema-valid JSON.

Run::

    python tests/st/distributed/collectives/all_to_all_v_benchmark.py \\
        --ep 8 --peer-bytes 24960 --count-pattern uniform --core-num 1 \\
        --impl managed-l2 --rounds 100 --warmup 5 --profile swimlane \\
        --output-json /tmp/a2av.json --platform a2a3 --devices 0-7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

Impl = Literal["managed-host", "managed-l2"]
ProfileKind = Literal["swimlane", "timing-slot", "both"]
CountPattern = Literal[
    "uniform",
    "zero",
    "self-only",
    "single-hot",
    "mixed",
    "asymmetric",
    "random",
]

CANONICAL_C = 4160
CANONICAL_MAX_PEER_BYTES = 1024 * 1024
DTYPE_BYTES = 1  # INT8 canonical payload
RANDOM_SEED = 2521
# Cap InCore stage/consume ``pl.load`` tiles under a2a3 VEC UB (~188416 bytes).
# Matches the InCore collective chunk budget (`kAllReduceChunkBytes`); C' rows
# larger than this (256 KiB+) used to fail AllocateMemoryAddr at compile time.
STAGE_TILE_BYTES = 16 * 1024

COUNT_PATTERNS: tuple[CountPattern, ...] = (
    "uniform",
    "zero",
    "self-only",
    "single-hot",
    "mixed",
    "asymmetric",
    "random",
)

# KiB-oriented main curve plus the two DSV4 anchors. Tail / non-multiple-of-C
# points live in TAIL_PEER_BYTES and compile a C' instance.
PAYLOAD_SWEEP_BYTES: tuple[int, ...] = (
    0,
    32,
    128,
    1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    24 * 1024,
    24960,  # 24.375 KiB DSV4 dispatch
    32 * 1024,
    48 * 1024,  # DSV4 combine
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
)

# RFC §8.3 sub-32-byte tail points. The current row-loop INT8 staging kernel
# requires 32-byte-aligned rows, so these are expected to be rejected by the
# alignment guard below; they are kept to document the intended sweep and the
# recorded N/A outcome in the A1 archive.
TAIL_PEER_BYTES: tuple[int, ...] = (1, 15, 31, 33, 4159, 4161)

CORE_SWEEP: dict[int, tuple[int, ...]] = {
    8: (1, 2, 4, 7, 8, 10, 15, 16),
    16: (1, 4, 8, 15, 16),
}

JSON_REQUIRED_KEYS = (
    "platform",
    "pypto_commit",
    "cann",
    "device_ids",
    "P",
    "peer_bytes",
    "count_pattern",
    "requested_L",
    "launched_B",
    "persistent",
    "warmup",
    "rounds",
    "metric",
    "per_rank_kernel_us",
    "per_rank_send_bytes",
    "per_rank_recv_bytes",
    "fastest_rank",
    "fastest_mean_us",
    "fastest_p50_us",
    "fastest_p95_us",
    "egress_gbps",
    "duplex_gbps",
)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div denominator must be positive, got {b}")
    return (a + b - 1) // b


def cal_all_to_all_v_blocks(p: int, core_num: int) -> int:
    """HCCL-style AllToAllV launch width. Pure function of ``(P, L)``.

    Mirrors the entry's ``CalAllToAllVBlocks(P, L)`` exactly (``entry.cpp.in``):
    ``B = L`` when ``L < P``, else the **largest multiple of ``P`` not
    exceeding ``L``**. There is **no** ``2P`` cap — the earlier model here had
    one, and it broke EP4 ``L=16`` (P=4): the harness sized the signal for
    ``B=8`` while the entry admitted ``B=16``, so the entry's stride admission
    rejected the whole dispatch with ``INVALID_ARGS`` (found by the K3 scaling
    campaign, 2026-09-23; all ten L=16 cells). The matrix points below were all
    ``L <= 2P``, which is why the stale model matched them.

    Known-good sweep points: EP8 ``L=1,2,4,7,8,10,15,16 →
    B=1,2,4,7,8,8,8,16`` and EP16 ``L=1,4,8,15,16 → B=L``.
    """
    if p < 1 or core_num < 1:
        raise ValueError(f"CalAllToAllVBlocks requires P>=1 and L>=1, got P={p} L={core_num}")
    return core_num if core_num < p else (core_num // p) * p


def canonical_max_recv(max_peer_bytes: int = CANONICAL_MAX_PEER_BYTES, row_width: int = CANONICAL_C) -> int:
    return ceil_div(max_peer_bytes, row_width)


@dataclass(frozen=True)
class BenchShape:
    p: int
    row_width: int
    max_recv: int
    routes_per_peer: int

    @property
    def total(self) -> int:
        return self.p * self.max_recv

    @property
    def uses_canonical_c(self) -> bool:
        return self.row_width == CANONICAL_C


def shape_for_peer_bytes(
    p: int,
    peer_bytes: int,
    *,
    row_width: int = CANONICAL_C,
    max_peer_bytes: int = CANONICAL_MAX_PEER_BYTES,
) -> BenchShape:
    """Window geometry for one ``--peer-bytes`` point.

    Multiples of ``row_width`` (and the 0-byte point) share the canonical
    ``MAX_RECV = CeilDiv(max_peer_bytes, C)`` window. Any other size compiles a
    ``C' = peer_bytes`` instance with one route per peer.
    """
    if p < 2:
        raise ValueError(f"need at least 2 ranks, got P={p}")
    if peer_bytes < 0:
        raise ValueError(f"peer_bytes must be >= 0, got {peer_bytes}")
    if peer_bytes == 0 or peer_bytes % row_width == 0:
        routes = 0 if peer_bytes == 0 else peer_bytes // row_width
        return BenchShape(p, row_width, canonical_max_recv(max_peer_bytes, row_width), routes)
    return BenchShape(p, peer_bytes, 1, 1)


def make_send_counts(
    pattern: CountPattern,
    shape: BenchShape,
    *,
    seed: int = RANDOM_SEED,
) -> torch.Tensor:
    """INT32 ``[P, P, 1]`` send counts for *pattern*.

    Values are raw counts; the kernel clamps to ``[0, MAX_RECV]``.
    """
    p = shape.p
    full = shape.routes_per_peer
    small = 1 if full > 1 else full
    counts = torch.zeros((p, p, 1), dtype=torch.int32)
    if pattern == "uniform":
        counts.fill_(full)
    elif pattern == "zero":
        pass
    elif pattern == "self-only":
        for r in range(p):
            counts[r, r, 0] = full
    elif pattern == "single-hot":
        for r in range(p):
            counts[r, (r + 1) % p, 0] = full
    elif pattern == "mixed":
        cycle = (0, small, full)
        for r in range(p):
            for d in range(p):
                counts[r, d, 0] = cycle[(r + d) % 3]
    elif pattern == "asymmetric":
        for r in range(p):
            for d in range(p):
                counts[r, d, 0] = (r * 3 + d * 5 + full) % (full + 1) if full else 0
    elif pattern == "random":
        rng = torch.Generator()
        rng.manual_seed(seed)
        hi = max(full, 1)
        counts[:, :, 0] = torch.randint(0, hi + 1, (p, p), generator=rng, dtype=torch.int32)
        if full == 0:
            counts.zero_()
    else:
        raise ValueError(f"unknown count pattern {pattern!r}")
    return counts


def clamp_rows(count: int, max_recv: int) -> int:
    return max(0, min(int(count), max_recv))


def row_bytes_aligned(row_width: int, dtype_bytes: int = DTYPE_BYTES) -> bool:
    """INT8 ``pl.load([1, C])`` needs a 32-byte physical row."""
    return (row_width * dtype_bytes) % 32 == 0


def stage_tile_cols(row_width: int, dtype_bytes: int = DTYPE_BYTES) -> int:
    """Column count for one VEC bounce tile when copying a full logical row."""
    if row_width < 1:
        raise ValueError(f"row_width must be >= 1, got {row_width}")
    cols = STAGE_TILE_BYTES // dtype_bytes
    return min(row_width, cols)


def remote_bytes_per_rank(
    send_counts: torch.Tensor, shape: BenchShape, *, dtype_bytes: int = DTYPE_BYTES
) -> tuple[list[int], list[int]]:
    """Per-rank remote (non-self) send and recv byte counts after the clamp."""
    p = shape.p
    send_b = [0] * p
    recv_b = [0] * p
    for src in range(p):
        for dst in range(p):
            if src == dst:
                continue
            n = clamp_rows(int(send_counts[src, dst, 0].item()), shape.max_recv)
            nbytes = n * shape.row_width * dtype_bytes
            send_b[src] += nbytes
            recv_b[dst] += nbytes
    return send_b, recv_b


def fill_inputs(shape: BenchShape, send_counts: torch.Tensor) -> torch.Tensor:
    """INT8 ``[P, total, C]`` payload. Non-zero so an unwritten tail stays 0."""
    inputs = torch.zeros((shape.p, shape.total, shape.row_width), dtype=torch.int8)
    for r in range(shape.p):
        for d in range(shape.p):
            n_rows = clamp_rows(int(send_counts[r, d, 0].item()), shape.max_recv)
            base = d * shape.max_recv
            for k in range(n_rows):
                # Fits in int8; unique per (rank, dest, row, col) enough to
                # catch a swapped block. Avoid 0 so the unwritten tail (0) is
                # distinguishable from a transferred row.
                col = torch.arange(shape.row_width, dtype=torch.int32)
                values = ((r * 31 + d * 17 + k * 13 + col) % 127) + 1
                inputs[r, base + k, :] = values.to(torch.int8)
    return inputs


def expected_recv_counts(send_counts: torch.Tensor, shape: BenchShape) -> torch.Tensor:
    """``recv_outputs[rank, src] = clamp(send_counts[src, rank], 0, MAX_RECV)``."""
    out = torch.zeros((shape.p, shape.p, 1), dtype=torch.int32)
    for rank in range(shape.p):
        for src in range(shape.p):
            out[rank, src, 0] = clamp_rows(int(send_counts[src, rank, 0].item()), shape.max_recv)
    return out


def parse_device_spec(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec and "," not in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def read_pypto_commit() -> str:
    env = os.environ.get("PYPTO_COMMIT")
    if env:
        return env.strip()
    root = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
    if root is None:
        return "unknown"
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"


def read_cann() -> str:
    for key in ("ASCEND_VERSION", "CANN_VERSION", "ASCEND_HOME_PATH"):
        value = os.environ.get(key)
        if value:
            return value
    return "unknown"


def p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return float(ordered[idx])


def gbps(nbytes: int, us: float) -> float:
    if us <= 0.0:
        return 0.0
    return (nbytes * 8.0) / (us * 1e3)


def _task_name(task: dict[str, Any]) -> str:
    return str(task.get("name", task.get("func_name", "")))


def _span_from_named_tasks(tasks: list[Any], needle: str = "all_to_all_v") -> float | None:
    starts: list[float] = []
    ends: list[float] = []
    for task in tasks:
        if not isinstance(task, dict) or needle not in _task_name(task):
            continue
        if "start_time_us" in task and "end_time_us" in task:
            starts.append(float(task["start_time_us"]))
            ends.append(float(task["end_time_us"]))
    if not starts:
        return None
    return max(ends) - min(starts)


def _gang_span_from_merged(path: Path) -> float | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    starts: list[float] = []
    ends: list[float] = []
    events = data.get("traceEvents")
    if isinstance(events, list):
        for ev in events:
            name = str(ev.get("name", ""))
            if "all_to_all_v" not in name:
                continue
            ts = ev.get("ts")
            dur = ev.get("dur", 0.0)
            if ts is None:
                continue
            starts.append(float(ts))
            ends.append(float(ts) + float(dur or 0.0))
    span = _span_from_named_tasks(list(data.get("tasks") or []))
    if starts:
        merged = max(ends) - min(starts)
        return merged if span is None else max(merged, span)
    return span


def _load_name_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    raw = data.get("callable_id_to_name") or {}
    return {str(k): str(v) for k, v in raw.items()}


def _completion_us(tasks: list[tuple[int, float]], us_per_cycle: float) -> float | None:
    """Last-completing AIV block in the faithful trace view.

    On 910B2 the same dispatch can be stamped from two cycle origins ~seconds
    apart (a second die's trace), and those duplicate durations are inflated —
    observed from ~3x up to ~140x the faithful stamp (e.g. 160 us vs 22.3 ms
    on an archived P2 cell). Split ``tasks`` (start cycles, duration) into
    view groups on start-time gaps > 1 s, keep the view holding the fastest
    stamp, and report its **maximum duration**: the collective completes when
    its slowest block finishes, so a genuinely slow block is never filtered
    while inflated duplicates (which live in the other view) cannot pollute
    the result. A single-view dispatch reduces to the maximum of its stamps.
    """
    if not tasks:
        return None
    ordered = sorted(tasks)
    groups: list[list[tuple[int, float]]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if (cur[0] - prev[0]) * us_per_cycle > 1_000_000:
            groups.append([])
        groups[-1].append(cur)
    faithful = min(groups, key=lambda g: min(d for _, d in g))
    return max(d for _, d in faithful)


def _aiv_exec_us_from_raw_records(path: Path, needle: str = "all_to_all_v") -> float | None:
    """Last-completing AIV block in the faithful trace view (raw records + ``name_map.json``).

    Level-1 records store ``func_id`` in ``task_token_raw & 0xFFFFFFFF``. Converted
    event names are ``task_spmd`` / ``func_-1_`` and must not be used. Duration is
    ``(end - start) / clock_freq`` **per task** — ``max(end) - min(start)`` across
    cores is not a gang span on 910B2 (two cycle origins ~seconds apart). Stamps
    are split into trace views by start-time gaps (> 1 s); the view holding the
    fastest stamp is faithful (duplicate stamps from the second origin are
    inflated — observed up to ~140x), and the rank's span is the **maximum
    duration in that view**: the collective completes when its slowest block
    finishes. For a single-view dispatch this is simply the maximum stamp.
    """
    names = _load_name_map(path.parent / "name_map.json")
    if not names:
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    freq = float((rec.get("metadata") or {}).get("clock_freq_hz") or 50_000_000)
    if freq <= 0:
        return None
    us_per_cycle = 1e6 / freq
    tasks: list[tuple[int, float]] = []
    for row in rec.get("aicore_tasks") or []:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        func_id = int(row[1]) & 0xFFFFFFFF
        name = names.get(str(func_id), "")
        if needle not in name:
            continue
        start, end = int(row[3]), int(row[4])
        dur = (end - start) * us_per_cycle
        if dur > 0:
            tasks.append((start, dur))
    return _completion_us(tasks, us_per_cycle)


def _gang_span_from_records(path: Path) -> float | None:
    """Raw records first (name_map); converted event names only as fallback."""
    raw = _aiv_exec_us_from_raw_records(path)
    if raw is not None:
        return raw
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        span = _span_from_named_tasks(tasks)
        if span is not None:
            return span
    try:
        from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415
    except ImportError:
        return None
    try:
        converted = read_perf_data(str(path))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return _span_from_named_tasks(list(converted.get("tasks") or []))


def collect_swimlane_spans_us(output_dir: Path) -> dict[int, float]:
    """Collective completion per rank (last-completing block), in microseconds.

    Prefers raw ``chip_swimlane_records.json`` + ``name_map.json`` (onboard
    level-1). HOST writes stage/fill/consume as later ``d{{k}}`` folders and
    often omits the builtin collective from converted traces; name_map on the
    CHIP pipeline dispatch is what actually names ``all_to_all_v``.
    """
    dfx = Path(output_dir) / "dfx_outputs"
    spans: dict[int, float] = {}
    if not dfx.is_dir():
        return spans
    for rank_dir in dfx.glob("rank*"):
        suffix = rank_dir.name[4:]
        if not suffix.isdigit():
            continue
        rank = int(suffix)
        candidates: list[tuple[float, float]] = []
        for path in rank_dir.glob("d*/chip_swimlane_records.json"):
            if path.is_file():
                span = _aiv_exec_us_from_raw_records(path)
                if span is not None:
                    candidates.append((path.stat().st_mtime, span))
        if not candidates:
            for path in rank_dir.glob("d*/merged_swimlane_*.json"):
                if path.is_file():
                    span = _gang_span_from_merged(path)
                    if span is not None:
                        candidates.append((path.stat().st_mtime, span))
        if not candidates:
            for path in rank_dir.glob("d*/chip_swimlane_records.json"):
                if path.is_file():
                    span = _gang_span_from_records(path)
                    if span is not None:
                        candidates.append((path.stat().st_mtime, span))
        if candidates:
            spans[rank] = max(candidates, key=lambda item: item[0])[1]
    return spans


def result_json(  # noqa: PLR0913
    *,
    platform: str,
    device_ids: list[int],
    shape: BenchShape,
    peer_bytes: int,
    pattern: CountPattern,
    requested_l: int,
    launched_b: int,
    expected_b: int,
    impl: Impl,
    persistent: bool,
    warmup: int,
    rounds: int,
    metric: str,
    per_rank_kernel_us: dict[str, list[float]],
    send_bytes: list[int],
    recv_bytes: list[int],
    row_width: int,
) -> dict[str, Any]:
    means = {int(r): sum(v) / len(v) for r, v in per_rank_kernel_us.items() if v}
    if means:
        fastest_rank = min(means, key=lambda r: means[r])
        samples = per_rank_kernel_us[str(fastest_rank)]
    else:
        fastest_rank = 0
        samples = []
    t_fast = means.get(fastest_rank, 0.0)
    slowest_rank = max(means, key=lambda r: means[r]) if means else 0
    t_slow = means.get(slowest_rank, 0.0)
    send_fast = send_bytes[fastest_rank] if fastest_rank < len(send_bytes) else 0
    recv_fast = recv_bytes[fastest_rank] if fastest_rank < len(recv_bytes) else 0
    return {
        "platform": platform,
        "pypto_commit": read_pypto_commit(),
        "cann": read_cann(),
        "device_ids": list(device_ids),
        "P": shape.p,
        "peer_bytes": peer_bytes,
        "count_pattern": pattern,
        "requested_L": requested_l,
        "launched_B": launched_b,
        "expected_B": expected_b,
        "impl": impl,
        "outer_l3_l2_dispatches": outer_l3_l2_dispatches(impl),
        "row_width": row_width,
        "max_recv": shape.max_recv,
        "persistent": persistent,
        "warmup": warmup,
        "rounds": rounds,
        "metric": metric,
        "per_rank_kernel_us": per_rank_kernel_us,
        "per_rank_send_bytes": {str(i): b for i, b in enumerate(send_bytes)},
        "per_rank_recv_bytes": {str(i): b for i, b in enumerate(recv_bytes)},
        "fastest_rank": fastest_rank,
        "fastest_mean_us": t_fast,
        "fastest_p50_us": p50(samples),
        "fastest_p95_us": p95(samples),
        "slowest_rank": slowest_rank,
        "slowest_mean_us": t_slow,
        "rank_spread_us": t_slow - t_fast,
        "egress_gbps": gbps(send_fast, t_fast),
        "duplex_gbps": gbps(send_fast + recv_fast, t_fast),
    }


def validate_result_schema(doc: dict[str, Any]) -> None:
    missing = [k for k in JSON_REQUIRED_KEYS if k not in doc]
    if missing:
        raise ValueError(f"result JSON missing keys {missing}")


def _pl():
    import pypto.language as pl  # noqa: PLC0415
    import pypto.language.distributed as pld  # noqa: PLC0415

    return pl, pld


def build_host_program(shape: BenchShape, core_num: int):
    """HOST rail: collective written in ``host_orch`` (per-device builtin fan-out).

    ``core_num`` is the requested launch limit ``L`` (RFC #2521 K2). The entry admits
    ``B = CalAllToAllVBlocks(P, L)`` blocks and requires one signal lane per admitted
    block, so the signal window is sized ``[nr, max(1, B)]``.
    """
    pl, pld = _pl()
    nr = shape.p
    mr = shape.max_recv
    total = shape.total
    width = shape.row_width
    tile_cols = stage_tile_cols(width)
    blocks_b = cal_all_to_all_v_blocks(nr, core_num)
    sig_s = max(1, blocks_b)

    @pl.program
    class HostAllToAllVBench:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, width], pl.INT8],
            stage: pl.Out[pld.DistributedTensor[[total, width], pl.INT8]],
        ):
            for row in pl.range(total):
                # Compile-time unroll keeps each VEC tile ≤ STAGE_TILE_BYTES.
                for col in pl.unroll(0, width, tile_cols):
                    valid = pl.min(tile_cols, width - col)
                    tile = pl.load(inp, [row, col], [1, tile_cols], valid_shape=[1, valid])
                    stage = pl.store(tile, [row, col], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, width], pl.INT8],
            stage: pl.Out[pld.DistributedTensor[[total, width], pl.INT8]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def fill_counts_step(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)

        @pl.function(type=pl.FunctionType.Orchestration)
        def fill_counts_orch(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.fill_counts_step(counts_row, counts)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, width], pl.INT8],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, width], pl.INT8]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, width], pl.INT8], pl.Tensor[[nr, 1], pl.INT32]]:
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    for col in pl.unroll(0, width, tile_cols):
                        valid = pl.min(tile_cols, width - col)
                        tile = pl.load(data, [flat_row, col], [1, tile_cols], valid_shape=[1, valid])
                        out = pl.store(tile, [flat_row, col], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    for col in pl.unroll(0, width, tile_cols):
                        valid = pl.min(tile_cols, width - col)
                        tile = pl.load(data, [flat_row, col], [1, tile_cols], valid_shape=[1, valid])
                        out = pl.store(tile, [flat_row, col], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, width], pl.INT8],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, width], pl.INT8]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, width], pl.INT8], pl.Tensor[[nr, 1], pl.INT32]]:
            return self.consume_step(data, recv_counts, out, recv_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, width], pl.INT8],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, width], pl.INT8]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, width], pl.INT8], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            input_buf = pld.alloc_window_buffer(total * width * pl.INT8.get_byte())
            data_buf = pld.alloc_window_buffer(total * width * pl.INT8.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * sig_s * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, width], dtype=pl.INT8)
                self.stage_orch(inputs[r], stage, device=r)

            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts[r], counts, device=r)

            stage = pld.window(input_buf, [total, width], dtype=pl.INT8)
            data = pld.window(data_buf, [total, width], dtype=pl.INT8)
            signal = pld.window(signal_buf, [nr, sig_s], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, recv, outputs[r], recv_outputs[r], device=r)
            return outputs, recv_outputs

    return HostAllToAllVBench


def build_l2_program(shape: BenchShape, core_num: int):
    """CHIP/L2 rail: collective written in the CHIP pipeline (one outer dispatch).

    K2 keeps this rail gated at ``core_num=1`` (O2 wires dynamic ``L/B`` here).
    """
    if core_num != 1:
        raise ValueError("managed-l2 is gated at core_num=1 (RFC #2521 O2)")
    pl, pld = _pl()
    nr = shape.p
    mr = shape.max_recv
    total = shape.total
    width = shape.row_width
    tile_cols = stage_tile_cols(width)

    @pl.program
    class L2AllToAllVBench:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, width], pl.INT8],
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            stage: pl.InOut[pld.DistributedTensor[[total, width], pl.INT8]],
            counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[
            pld.DistributedTensor[[total, width], pl.INT8],
            pld.DistributedTensor[[nr, 1], pl.INT32],
        ]:
            for row in pl.range(total):
                for col in pl.unroll(0, width, tile_cols):
                    valid = pl.min(tile_cols, width - col)
                    tile = pl.load(inp, [row, col], [1, tile_cols], valid_shape=[1, valid])
                    stage = pl.store(tile, [row, col], stage)
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)
            return stage, counts

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, width], pl.INT8],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, width], pl.INT8]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, width], pl.INT8], pl.Tensor[[nr, 1], pl.INT32]]:
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    for col in pl.unroll(0, width, tile_cols):
                        valid = pl.min(tile_cols, width - col)
                        tile = pl.load(data, [flat_row, col], [1, tile_cols], valid_shape=[1, valid])
                        out = pl.store(tile, [flat_row, col], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    for col in pl.unroll(0, width, tile_cols):
                        valid = pl.min(tile_cols, width - col)
                        tile = pl.load(data, [flat_row, col], [1, tile_cols], valid_shape=[1, valid])
                        out = pl.store(tile, [flat_row, col], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pl.Tensor[[total, width], pl.INT8],
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, width], pl.INT8]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            stage: pl.InOut[pld.DistributedTensor[[total, width], pl.INT8]],
            data: pl.InOut[pld.DistributedTensor[[total, width], pl.INT8]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, width], pl.INT8], pl.Tensor[[nr, 1], pl.INT32]]:
            stage, counts = self.stage_step(inp, counts_row, stage, counts)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
            return self.consume_step(data, recv, out, recv_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, width], pl.INT8],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, width], pl.INT8]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, width], pl.INT8], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            stage_buf = pld.alloc_window_buffer(total * width * pl.INT8.get_byte())
            data_buf = pld.alloc_window_buffer(total * width * pl.INT8.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [total, width], dtype=pl.INT8)
                data = pld.window(data_buf, [total, width], dtype=pl.INT8)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
                self.chip_pipeline(
                    inputs[r],
                    send_counts[r],
                    outputs[r],
                    recv_outputs[r],
                    stage,
                    data,
                    sig,
                    counts,
                    recv,
                    device=r,
                )
            return outputs, recv_outputs

    return L2AllToAllVBench


def build_program(impl: Impl, shape: BenchShape, core_num: int):
    if impl == "managed-host":
        return build_host_program(shape, core_num)
    if impl == "managed-l2":
        return build_l2_program(shape, core_num)
    raise ValueError(f"unknown impl {impl!r}")


def outer_l3_l2_dispatches(impl: Impl) -> int:
    """RFC reporting column: HOST 3 outer tasks vs L2 1. Not an AIV speedup."""
    if impl == "managed-host":
        return 3
    if impl == "managed-l2":
        return 1
    raise ValueError(f"unknown impl {impl!r}")


def require_supported_core_num(core_num: int, impl: str) -> None:
    """Rail-aware gate: RFC #2521 K2 enables ``L>1`` on the managed HOST rail only.

    The CHIP/L2 rail is deliberately left gated at ``core_num=1`` (O2 wires dynamic
    ``L/B`` into the CHIP pipeline).
    """
    if core_num > 1 and impl != "managed-host":
        raise ValueError(
            f"core_num={core_num} is not implemented on impl={impl!r}: K2 enables L>1 on the "
            "managed HOST rail only, and the CHIP/L2 rail stays gated at core_num=1 (O2). "
            "Use --impl managed-host, or re-run with --core-num 1 for the baseline configuration."
        )


def check_correctness(
    shape: BenchShape,
    inputs: torch.Tensor,
    send_counts: torch.Tensor,
    outputs: torch.Tensor,
    recv_out: torch.Tensor,
) -> None:
    expected = expected_recv_counts(send_counts, shape)
    for rank in range(shape.p):
        for src in range(shape.p):
            got = int(recv_out[rank, src, 0].item())
            want = int(expected[rank, src, 0].item())
            if got != want:
                raise AssertionError(
                    f"rank={rank} src={src}: recv_counts={got} != clamped send_counts={want}"
                )
            n_rows = want
            base = src * shape.max_recv
            for k in range(n_rows):
                exp_row = inputs[src, rank * shape.max_recv + k, :]
                got_row = outputs[rank, base + k, :]
                if not torch.equal(got_row, exp_row):
                    raise AssertionError(
                        f"rank={rank} src={src} row={k}: payload mismatch "
                        f"(got {got_row[:8].tolist()} expected {exp_row[:8].tolist()})"
                    )


def _compile(program, *, platform: str, device_ids: list[int], output_dir: str | None, skip_ptoas: bool):
    from pypto import ir  # noqa: PLC0415
    from pypto.ir import DistributedConfig  # noqa: PLC0415

    return ir.compile(
        program,
        platform=platform,
        output_dir=output_dir,
        skip_ptoas=skip_ptoas,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(device_ids), num_sub_workers=0),
    )


def compile_only_json(args: argparse.Namespace, shape: BenchShape) -> dict[str, Any]:
    """Compile one ``--impl`` with skip_ptoas and emit a schema-valid stub."""
    require_supported_core_num(args.core_num, args.impl)
    if not row_bytes_aligned(shape.row_width):
        raise ValueError(
            f"row_width={shape.row_width} is not 32-byte aligned for INT8 staging; "
            "sub-32-byte tail points cannot use the row-loop stage kernel (recorded as N/A in the A1 archive)"
        )
    if len(args.device_ids) < shape.p:
        raise ValueError(f"need {shape.p} devices, got {args.device_ids}")
    device_ids = args.device_ids[: shape.p]
    impl: Impl = args.impl
    program = build_program(impl, shape, args.core_num)
    compiled = _compile(
        program,
        platform=args.platform,
        device_ids=device_ids,
        output_dir=args.output_dir,
        skip_ptoas=True,
    )
    next_levels = compiled.output_dir / "next_levels"
    if impl == "managed-l2":
        kernels = list(compiled.output_dir.glob("**/__builtin_all_to_all_v__int8.cpp"))
        if not kernels:
            raise FileNotFoundError(
                f"expected L2 INT8 kernel under {compiled.output_dir}, looked at "
                f"{next_levels / 'chip_pipeline' / 'kernels' / 'aiv' / '__builtin_all_to_all_v__int8.cpp'}"
            )
        if next_levels.is_dir() and any(p.name.startswith("builtin.tensor.") for p in next_levels.iterdir()):
            raise RuntimeError("managed-l2 must not emit a builtin chip dispatch")
    else:
        variants = list(compiled.output_dir.glob("**/builtin.tensor.all_to_all_v__int8"))
        if not variants:
            raise FileNotFoundError(f"expected HOST INT8 variant under {compiled.output_dir}")
    send_counts = make_send_counts(args.count_pattern, shape)
    send_b, recv_b = remote_bytes_per_rank(send_counts, shape)
    doc = result_json(
        platform=args.platform,
        device_ids=device_ids,
        shape=shape,
        peer_bytes=args.peer_bytes,
        pattern=args.count_pattern,
        requested_l=args.core_num,
        launched_b=cal_all_to_all_v_blocks(shape.p, args.core_num),
        expected_b=cal_all_to_all_v_blocks(shape.p, args.core_num),
        impl=impl,
        persistent=True,
        warmup=args.warmup,
        rounds=args.rounds,
        metric="compile-only",
        per_rank_kernel_us={str(i): [] for i in range(shape.p)},
        send_bytes=send_b,
        recv_bytes=recv_b,
        row_width=shape.row_width,
    )
    doc["compile_output_dir"] = str(compiled.output_dir)
    return doc


def _timed_session(  # noqa: PLR0913
    *,
    compiled,
    run_config,
    inputs,
    send_counts,
    outputs,
    recv_out,
    shape: BenchShape,
    warmup: int,
    rounds: int,
    collect_aiv: bool,
    check: bool,
) -> tuple[list[float], dict[str, list[float]]]:
    host_slots: list[float] = []
    per_rank: dict[str, list[float]] = {str(i): [] for i in range(shape.p)}
    with compiled.prepare(
        config=run_config,
        persistent=True,
        reset_persistent_windows=False,
    ) as runtime:
        runtime.run(compiled, inputs, send_counts, outputs, recv_out, config=run_config)
        if check:
            check_correctness(shape, inputs, send_counts, outputs, recv_out)
        for _ in range(warmup):
            runtime.run(compiled, inputs, send_counts, outputs, recv_out, config=run_config)
        for _ in range(rounds):
            t0 = time.perf_counter()
            runtime.run(compiled, inputs, send_counts, outputs, recv_out, config=run_config)
            slot_us = (time.perf_counter() - t0) * 1e6
            host_slots.append(slot_us)
            if collect_aiv:
                spans = collect_swimlane_spans_us(compiled.output_dir)
                for r, span in spans.items():
                    per_rank[str(r)].append(float(span))
            else:
                for r in range(shape.p):
                    per_rank[str(r)].append(slot_us)
    return host_slots, per_rank


def run_benchmark(args: argparse.Namespace, shape: BenchShape) -> dict[str, Any]:
    from pypto.runtime import RunConfig  # noqa: PLC0415

    require_supported_core_num(args.core_num, args.impl)
    if not row_bytes_aligned(shape.row_width):
        raise ValueError(
            f"row_width={shape.row_width} is not 32-byte aligned for INT8 staging; "
            "sub-32-byte tail points cannot use the row-loop stage kernel (recorded as N/A in the A1 archive)"
        )
    if len(args.device_ids) < shape.p:
        raise ValueError(f"need {shape.p} devices, got {args.device_ids}")

    program = build_program(args.impl, shape, args.core_num)
    compiled = _compile(
        program,
        platform=args.platform,
        device_ids=args.device_ids[: shape.p],
        output_dir=args.output_dir,
        skip_ptoas=False,
    )

    send_counts = make_send_counts(args.count_pattern, shape).share_memory_()
    inputs = fill_inputs(shape, send_counts).share_memory_()
    outputs = torch.zeros_like(inputs).share_memory_()
    recv_out = torch.zeros((shape.p, shape.p, 1), dtype=torch.int32).share_memory_()
    send_b, recv_b = remote_bytes_per_rank(send_counts, shape)
    expected_b = cal_all_to_all_v_blocks(shape.p, args.core_num)

    slot_cfg = RunConfig(platform=args.platform, enable_chip_swimlane=0)
    swim_cfg = RunConfig(platform=args.platform, enable_chip_swimlane=1)
    common = dict(
        compiled=compiled,
        inputs=inputs,
        send_counts=send_counts,
        outputs=outputs,
        recv_out=recv_out,
        shape=shape,
    )

    host_slots: list[float] = []
    aiv_rank: dict[str, list[float]] = {str(i): [] for i in range(shape.p)}
    slot_includes_tax = False
    profile = args.profile
    swim_rounds = args.swimlane_rounds if profile == "both" else args.rounds

    if profile in ("timing-slot", "both"):
        print(
            f"[a2av-bench] session=timing-slot enter warmup={args.warmup} rounds={args.rounds}",
            flush=True,
        )
        host_slots, slot_rank = _timed_session(
            **common,
            run_config=slot_cfg,
            warmup=args.warmup,
            rounds=args.rounds,
            collect_aiv=False,
            check=True,
        )
        print("[a2av-bench] session=timing-slot exit ok", flush=True)
    else:
        slot_rank = {str(i): [] for i in range(shape.p)}

    if profile in ("swimlane", "both"):
        swim_warmup = args.warmup if profile == "swimlane" else min(args.warmup, 2)
        print(
            f"[a2av-bench] session=swimlane enter warmup={swim_warmup} rounds={swim_rounds}",
            flush=True,
        )
        swim_slots, aiv_rank = _timed_session(
            **common,
            run_config=swim_cfg,
            warmup=swim_warmup,
            rounds=swim_rounds,
            collect_aiv=True,
            check=profile == "swimlane",
        )
        print("[a2av-bench] session=swimlane exit ok", flush=True)
        if profile == "swimlane":
            host_slots = swim_slots
            slot_includes_tax = True

    aiv_found = any(aiv_rank.values())
    if aiv_found:
        metric = "aicore_gang_span"
        per_rank = aiv_rank
    else:
        metric = "timing_slot"
        per_rank = (
            slot_rank if any(slot_rank.values()) else {str(i): list(host_slots) for i in range(shape.p)}
        )

    doc = result_json(
        platform=args.platform,
        device_ids=args.device_ids[: shape.p],
        shape=shape,
        peer_bytes=args.peer_bytes,
        pattern=args.count_pattern,
        requested_l=args.core_num,
        launched_b=expected_b,
        expected_b=expected_b,
        impl=args.impl,
        persistent=True,
        warmup=args.warmup,
        rounds=args.rounds,
        metric=metric,
        per_rank_kernel_us=per_rank,
        send_bytes=send_b,
        recv_bytes=recv_b,
        row_width=shape.row_width,
    )
    if host_slots:
        doc["host_timing_slot_mean_us"] = sum(host_slots) / len(host_slots)
        doc["host_timing_slot_p50_us"] = p50(host_slots)
        doc["host_timing_slot_includes_swimlane_tax"] = slot_includes_tax
    if aiv_found:
        doc["aiv_rounds_captured"] = max(len(v) for v in aiv_rank.values())
        doc["aiv_rounds_requested"] = swim_rounds
        missing = sorted(int(r) for r, v in aiv_rank.items() if not v)
        if missing:
            doc["aiv_ranks_missing"] = missing
        short = {
            r: len(v)
            for r, v in sorted(aiv_rank.items(), key=lambda kv: int(kv[0]))
            if 0 < len(v) < swim_rounds
        }
        if short:
            doc["aiv_rounds_incomplete"] = short
    return doc


def _apply_smoke(args: argparse.Namespace) -> None:
    args.ep = 2
    args.peer_bytes = 256
    args.row_width = 64
    args.max_peer_bytes = 256
    args.rounds = 2
    args.warmup = 0
    args.profile = "timing-slot"
    args.codegen_only = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFC #2521 AllToAllV benchmark harness")
    parser.add_argument("--ep", type=int, default=8, help="rank count (8 or 16 for the official sweep)")
    parser.add_argument("--peer-bytes", type=int, default=24960, help="bytes sent to each peer (uniform)")
    parser.add_argument("--count-pattern", choices=COUNT_PATTERNS, default="uniform")
    parser.add_argument("--core-num", type=int, default=1, dest="core_num", help="requested L")
    parser.add_argument("--impl", choices=("managed-host", "managed-l2"), default="managed-l2")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", choices=("swimlane", "timing-slot", "both"), default="swimlane")
    parser.add_argument(
        "--swimlane-rounds",
        type=int,
        default=8,
        dest="swimlane_rounds",
        help="AIV-capture rounds when --profile both (timing-slot still uses --rounds)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument(
        "--devices",
        default=os.environ.get("TASK_DEVICE", os.environ.get("PYPTO_DEVICES", "0,1")),
        help="device id list, e.g. 0-7 or 0,1,2,3",
    )
    parser.add_argument("--row-width", type=int, default=CANONICAL_C, dest="row_width")
    parser.add_argument("--max-peer-bytes", type=int, default=CANONICAL_MAX_PEER_BYTES, dest="max_peer_bytes")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--codegen-only",
        action="store_true",
        help="compile with skip_ptoas and emit schema-valid JSON without dispatch",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny P=2 INT8 compile-only smoke (sim Docker gate)",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        _apply_smoke(args)
    args.device_ids = parse_device_spec(str(args.devices))
    if args.ep < 2:
        parser.error("--ep must be >= 2")
    if args.rounds < 0 or args.warmup < 0 or args.swimlane_rounds < 0:
        parser.error("--rounds, --warmup, and --swimlane-rounds must be >= 0")
    if args.core_num < 1:
        parser.error("--core-num must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shape = shape_for_peer_bytes(
        args.ep,
        args.peer_bytes,
        row_width=args.row_width,
        max_peer_bytes=args.max_peer_bytes,
    )
    if args.codegen_only:
        doc = compile_only_json(args, shape)
    else:
        doc = run_benchmark(args, shape)
    validate_result_schema(doc)
    text = json.dumps(doc, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
