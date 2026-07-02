# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the register-once benchmark helper (issue #1858).

After simpler PR #1177, ``benchmark`` reads per-launch timing from the
runtime's ``[STRACE]`` stderr markers rather than a ``run_timed`` return value.
The parse + aggregate path (:func:`_parse_stats_from_strace`) delegates the
marker grammar to simpler's ``strace_timing``, so those tests feed synthetic
marker lines through it and **skip when the optional ``simpler`` runtime is not
installed** (e.g. the unit-test CI host) via the ``require_simpler`` fixture.
The ``benchmark`` driver (register-once, warmup, log-level + stderr capture) and
the pure-``BenchmarkStats`` aggregate helpers patch the parse seam out, so they
run everywhere without ``simpler``.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pypto.ir import distributed_compiled_program as dcp_mod
from pypto.runtime import RunConfig
from pypto.runtime.bench import BenchmarkStats, _parse_stats_from_strace, benchmark


@pytest.fixture
def require_simpler():
    """Skip the test unless the optional ``simpler`` runtime is importable.

    :func:`_parse_stats_from_strace` lazily imports ``simpler_setup.tools.
    strace_timing`` (the single source of truth for the ``[STRACE]`` grammar);
    it is absent on the unit-test CI host, where these parse tests skip.
    """
    pytest.importorskip("simpler_setup.tools.strace_timing")


def _strace_line(
    inv: int,
    name: str,
    dur_ns: int,
    *,
    hid: str = "abc",
    depth: int = 0,
    dev: bool = False,
    pid: int = 100,
    ts: int | None = None,
) -> str:
    """One synthetic ``[STRACE]`` marker line (matches strace_timing's grammar)."""
    attrs = " clk=dev" if dev else ""
    ts_ns = ts if ts is not None else inv * 1000
    return (
        f"[2026-01-01][T0x1][INFO_V9] run_prepared: [STRACE] v=1 pid={pid} tid=1 "
        f"inv={inv} hid={hid} depth={depth} name={name} ts={ts_ns} dur={dur_ns}{attrs}"
    )


def _launch_lines(
    inv: int, *, host_us: float, device_us: float, pid: int = 100, hid: str = "abc"
) -> list[str]:
    """The two markers one launch emits: the ``run_prepared`` host span + device wall."""
    return [
        _strace_line(inv, "run_prepared", int(host_us * 1000), depth=0, pid=pid, hid=hid),
        _strace_line(
            inv,
            "run_prepared.runner_run.device_wall",
            int(device_us * 1000),
            depth=2,
            dev=True,
            pid=pid,
            hid=hid,
        ),
    ]


def _row_present(tree: str, expected: str) -> bool:
    """True if some tree line contains *expected* ignoring column-alignment
    whitespace runs (tree output right-aligns value columns with padding)."""
    want = " ".join(expected.split())
    return any(want in " ".join(line.split()) for line in tree.splitlines())


# ---------------------------------------------------------------------------
# _parse_stats_from_strace — span extraction, warmup discard, aggregation
# ---------------------------------------------------------------------------


def test_parse_discards_warmup_and_collects_rounds(require_simpler):
    """Warmup invocations are dropped; only the trailing ``rounds`` are measured."""
    lines: list[str] = []
    # 2 warmup launches (inv 0,1) then 3 measured (inv 2,3,4).
    lines += _launch_lines(0, host_us=99, device_us=99)
    lines += _launch_lines(1, host_us=99, device_us=99)
    lines += _launch_lines(2, host_us=100, device_us=10)
    lines += _launch_lines(3, host_us=200, device_us=20)
    lines += _launch_lines(4, host_us=300, device_us=30)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=3, warmup=2)

    assert stats.device_wall_us == [10.0, 20.0, 30.0]
    assert stats.host_wall_us == [100.0, 200.0, 300.0]
    assert stats.rounds == 3
    assert stats.warmup == 2


def test_parse_no_warmup_keeps_all(require_simpler):
    lines = _launch_lines(0, host_us=50, device_us=5) + _launch_lines(1, host_us=60, device_us=15)
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)
    assert stats.device_wall_us == [5.0, 15.0]
    assert stats.host_wall_us == [50.0, 60.0]


def test_parse_no_device_span_reads_zero(require_simpler):
    """On sim / non-profiling builds only the host span is emitted -> device 0."""
    lines = [_strace_line(0, "run_prepared", 50_000, depth=0)]
    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)
    assert stats.host_wall_us == [50.0]
    assert stats.device_wall_us == [0.0]
    assert stats.all_zero_device is True


def test_parse_no_markers_returns_empty(require_simpler):
    stats = _parse_stats_from_strace("no strace markers here\n", rounds=5, warmup=1)
    assert stats.device_wall_us == []
    assert stats.host_wall_us == []
    assert stats.invocations == []


def test_parse_populates_full_span_tree_and_format(require_simpler):
    """Each measured launch keeps its full span tree; format_tree draws the
    hierarchy with ``|-`` / `` `- `` connectors and tags device spans."""
    # A branching tree (siblings tie on ts -> kept in line order):
    #   run_prepared
    #   |- bind
    #   |  |- args
    #   |  `- prebuilt
    #   `- runner_run
    #      `- device_wall [dev]
    lines = [
        _strace_line(0, "run_prepared", 10_000, depth=0),
        _strace_line(0, "run_prepared.bind", 6_000, depth=1),
        _strace_line(0, "run_prepared.bind.args", 4_000, depth=2),
        _strace_line(0, "run_prepared.bind.prebuilt", 2_000, depth=2),
        _strace_line(0, "run_prepared.runner_run", 3_000, depth=1),
        _strace_line(0, "run_prepared.runner_run.device_wall", 2_000, depth=2, dev=True),
    ]
    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)

    assert stats.device_wall_us == [2.0]
    assert stats.host_wall_us == [10.0]
    assert len(stats.invocations) == 1
    inv = stats.invocations[0]
    root = inv.root()
    assert root is not None
    assert root.name == "run_prepared"
    assert inv.by_name()["run_prepared.runner_run.device_wall"].is_device

    tree = stats.format_tree(launch=0)
    # Branch connectors mark hierarchy (not indentation alone).
    assert "|- bind" in tree
    assert "|  |- args" in tree
    assert "|  `- prebuilt" in tree
    assert "`- runner_run" in tree
    assert "   `- device_wall [dev]" in tree


def test_format_tree_no_capture_message():
    stats = BenchmarkStats(rounds=2, warmup=0)
    assert "no span tree captured" in stats.format_tree()
    assert "no span tree captured" in stats.format_mean_tree()


def test_mean_tree_averages_durations_across_launches(require_simpler):
    """The mean tree averages each span's duration across measured launches."""
    # Two launches; run_prepared -> runner_run.device_wall. Device wall is 10
    # then 20 us -> mean 15; host run_prepared 100 then 300 -> mean 200.
    lines = []
    for inv, host_us, dev_us in [(0, 100.0, 10.0), (1, 300.0, 20.0)]:
        lines.append(_strace_line(inv, "run_prepared", int(host_us * 1000), depth=0))
        lines.append(
            _strace_line(inv, "run_prepared.runner_run.device_wall", int(dev_us * 1000), depth=2, dev=True)
        )
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)

    mean = stats.mean_invocation()
    assert mean is not None
    by = mean.by_name()
    assert by["run_prepared"].dur == 200_000  # mean of 100k, 300k ns
    assert by["run_prepared.runner_run.device_wall"].dur == 15_000  # mean of 10k, 20k
    assert by["run_prepared.runner_run.device_wall"].is_device

    tree = stats.format_mean_tree()
    assert "mean of 2 launches" in tree
    assert _row_present(tree, "run_prepared 200.0us")
    assert _row_present(tree, "device_wall [dev] 15.0us")


def test_mean_tree_spread_annotations(require_simpler):
    """Mean-tree nodes carry ±stdev and [min..max] across launches."""
    lines = []
    for inv, host_us, dev_us in [(0, 100.0, 10.0), (1, 300.0, 20.0)]:
        lines.append(_strace_line(inv, "run_prepared", int(host_us * 1000), depth=0))
        lines.append(
            _strace_line(inv, "run_prepared.runner_run.device_wall", int(dev_us * 1000), depth=2, dev=True)
        )
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)

    # stdev([10,20]) = 7.07; min/max = 10/20.
    stdev_tree = stats.format_mean_tree(spread="stdev")
    assert _row_present(stdev_tree, "device_wall [dev] 15.0us ±7.1")
    assert "[10.0..20.0]" not in stdev_tree

    minmax_tree = stats.format_mean_tree(spread="minmax")
    assert _row_present(minmax_tree, "device_wall [dev] 15.0us [10.0..20.0]")
    assert "±" not in minmax_tree

    both_tree = stats.format_mean_tree(spread="both")
    assert _row_present(both_tree, "device_wall [dev] 15.0us ±7.1 [10.0..20.0]")

    none_tree = stats.format_mean_tree(spread="none")
    assert _row_present(none_tree, "device_wall [dev] 15.0us")
    # No spread markers (the "[" in "[dev]" is the device tag, not a range).
    assert "±" not in none_tree and ".." not in none_tree


# ---------------------------------------------------------------------------
# _parse_stats_from_strace — L3 (distributed=True) per-rank aggregation
# ---------------------------------------------------------------------------


def test_parse_l3_two_ranks_per_round_max(require_simpler):
    """Two ranks (pids), one dispatch/round each: headline = per-round max across ranks."""
    lines: list[str] = []
    # rank0 (pid 100): warmup inv0, measured inv1=10us, inv2=30us device.
    lines += _launch_lines(0, host_us=99, device_us=99, pid=100)
    lines += _launch_lines(1, host_us=100, device_us=10, pid=100)
    lines += _launch_lines(2, host_us=300, device_us=30, pid=100)
    # rank1 (pid 101): warmup inv0, measured inv1=20us, inv2=5us device.
    lines += _launch_lines(0, host_us=99, device_us=99, pid=101)
    lines += _launch_lines(1, host_us=200, device_us=20, pid=101)
    lines += _launch_lines(2, host_us=50, device_us=5, pid=101)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 30.0], 101: [20.0, 5.0]}
    assert stats.per_rank("host") == {100: [100.0, 300.0], 101: [200.0, 50.0]}
    # Per-round max across ranks: round0 = max(10,20)=20; round1 = max(30,5)=30.
    assert stats.device_wall_us == [20.0, 30.0]
    assert stats.host_wall_us == [200.0, 300.0]


def test_parse_l3_multi_dispatch_sums_within_round(require_simpler):
    """A rank dispatched multiple times per round (heterogeneous hids) sums within the round."""
    lines: list[str] = []
    # rank0 (pid 100): 2 dispatches/round, different hids. No warmup.
    # round0 = inv0(hidA,4us) + inv1(hidB,6us) = 10us; round1 = inv2(3us)+inv3(7us) = 10us.
    lines += _launch_lines(0, host_us=1, device_us=4, pid=100, hid="A")
    lines += _launch_lines(1, host_us=1, device_us=6, pid=100, hid="B")
    lines += _launch_lines(2, host_us=1, device_us=3, pid=100, hid="A")
    lines += _launch_lines(3, host_us=1, device_us=7, pid=100, hid="B")
    # rank1 (pid 101): 1 dispatch/round -> different per-rank d.
    lines += _launch_lines(0, host_us=1, device_us=5, pid=101, hid="C")
    lines += _launch_lines(1, host_us=1, device_us=8, pid=101, hid="C")

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 10.0], 101: [5.0, 8.0]}
    # Per-round max: round0 = max(10,5)=10; round1 = max(10,8)=10.
    assert stats.device_wall_us == [10.0, 10.0]


def test_parse_l3_non_divisible_falls_back_to_flattened(require_simpler):
    """A non-deterministic dispatch shape (count not divisible by launches) flattens."""
    lines: list[str] = []
    # 3 invocations, rounds=2 warmup=0 -> 3 % 2 != 0 -> fallback.
    lines += _launch_lines(0, host_us=1, device_us=5, pid=100)
    lines += _launch_lines(1, host_us=1, device_us=6, pid=100)
    lines += _launch_lines(2, host_us=1, device_us=7, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is True
    assert stats.per_rank("device") == {}
    # Flattened pool of all per-dispatch device walls (warmup=0 -> none dropped).
    assert sorted(stats.device_wall_us) == [5.0, 6.0, 7.0]
    # No round alignment -> no cross-rank union window.
    assert stats.per_round("union") == []


def test_parse_l3_union_window_captures_cross_rank_skew(require_simpler):
    """``per_round("union")`` = cross-rank host-timeline window (max end - min start).

    ``_strace_line`` sets each span's host ``ts`` to ``inv * 1000`` ns, so giving
    the two ranks different ``inv`` values models a cross-rank start skew.
    """
    lines: list[str] = []
    # rank0 (pid100): inv=0 -> run_prepared host window [0, 5000] ns.
    lines += _launch_lines(0, host_us=5, device_us=10, pid=100)
    # rank1 (pid101): inv=3 -> starts later, window [3000, 8000] ns.
    lines += _launch_lines(3, host_us=5, device_us=20, pid=101)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.device_wall_us == [20.0]  # per-round max across ranks
    # Union: max(end)=8000 - min(start)=0 = 8000 ns = 8.0 us.
    assert stats.per_round("union") == [8.0]


def test_parse_l3_rounds_dispatches_round_rank_dispatch_view(require_simpler):
    """``rounds_dispatches[k][pid]`` gives the per-round, per-rank dispatch list,
    each carrying its task (hid) and precise per-dispatch timing."""
    lines: list[str] = []
    # rank0 (pid100): 2 dispatches/round (hids A, B); rank1 (pid101): 1/round (hid C).
    # round0                          round1
    lines += _launch_lines(0, host_us=1, device_us=4, pid=100, hid="a")
    lines += _launch_lines(1, host_us=1, device_us=6, pid=100, hid="b")
    lines += _launch_lines(2, host_us=1, device_us=3, pid=100, hid="a")
    lines += _launch_lines(3, host_us=1, device_us=7, pid=100, hid="b")
    lines += _launch_lines(0, host_us=1, device_us=5, pid=101, hid="c")
    lines += _launch_lines(1, host_us=1, device_us=8, pid=101, hid="c")

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    # Shape: rounds -> {rank: [dispatch, ...]}.
    assert len(stats.rounds_dispatches) == 2
    assert set(stats.rounds_dispatches[0]) == {100, 101}
    # Round 0, rank 100: two dispatches (tasks a, b) with precise per-dispatch device walls.
    r0_rank0 = stats.rounds_dispatches[0][100]
    assert [d.task for d in r0_rank0] == ["a", "b"]
    assert [d.device_wall_us for d in r0_rank0] == [4.0, 6.0]
    # Round 1, rank 101: single dispatch (task c), device wall 8.
    r1_rank1 = stats.rounds_dispatches[1][101]
    assert [d.task for d in r1_rank1] == ["c"]
    assert r1_rank1[0].device_wall_us == 8.0
    # The nested view sums to the per-rank per-round busy figures.
    assert stats.per_rank("device")[100] == [10.0, 10.0]  # 4+6, 3+7


def test_parse_l3_per_rank_effective_time(require_simpler):
    """Per-card L2 Effective = orch∪sched window; exposed per dispatch and per rank."""
    DEV = "run_prepared.runner_run.device_wall"

    def dispatch(inv, pid, *, orch, sched):
        # orch / sched are (start_ns, dur_ns) device-domain spans.
        return [
            _strace_line(inv, "run_prepared", 5000, depth=0, pid=pid),
            _strace_line(inv, DEV, 9000, depth=2, dev=True, pid=pid),
            _strace_line(inv, DEV + ".orch", orch[1], depth=3, dev=True, pid=pid, ts=orch[0]),
            _strace_line(inv, DEV + ".sched", sched[1], depth=3, dev=True, pid=pid, ts=sched[0]),
        ]

    lines: list[str] = []
    # rank0: orch [1000,4000], sched [2000,6000] -> effective = 6000-1000 = 5.0us.
    lines += dispatch(0, 100, orch=(1000, 3000), sched=(2000, 4000))
    # rank1: orch [0,2000], sched [5000,3000]->[5000,8000] -> effective = 8000-0 = 8.0us.
    lines += dispatch(0, 101, orch=(0, 2000), sched=(5000, 3000))

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0, distributed=True)

    # Per-dispatch effective (via the navigable view).
    assert stats.rounds_dispatches[0][100][0].effective_us == 5.0
    assert stats.rounds_dispatches[0][101][0].effective_us == 8.0
    # Per-card per-round effective (summed within the round; 1 dispatch here).
    assert stats.per_rank("effective") == {100: [5.0], 101: [8.0]}


def test_parse_l3_degenerates_to_l2_for_single_rank(require_simpler):
    """One rank, one dispatch/round: L3 aggregation matches the L2 per-launch values."""
    lines: list[str] = []
    lines += _launch_lines(0, host_us=99, device_us=99, pid=100)  # warmup
    lines += _launch_lines(1, host_us=100, device_us=10, pid=100)
    lines += _launch_lines(2, host_us=200, device_us=20, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.device_wall_us == [10.0, 20.0]
    assert stats.per_rank("device") == {100: [10.0, 20.0]}


# ---------------------------------------------------------------------------
# BenchmarkStats — aggregate helpers
# ---------------------------------------------------------------------------


def test_stats_aggregates():
    stats = BenchmarkStats(
        device_wall_us=[10.0, 20.0, 30.0], host_wall_us=[1.0, 2.0, 3.0], rounds=3, warmup=0
    )
    assert stats.device_us_min == 10.0
    assert stats.device_us_max == 30.0
    assert stats.device_us_median == 20.0
    assert stats.device_us_mean == 20.0
    # Aliases mirror the device_us_* accessors.
    assert stats.device_wall_us_median == stats.device_us_median
    assert stats.samples is stats.device_wall_us
    assert stats.all_zero_device is False


def test_stats_all_zero_device():
    stats = BenchmarkStats(device_wall_us=[0.0, 0.0], host_wall_us=[1.0, 2.0], rounds=2)
    assert stats.all_zero_device is True
    assert "all 0" in str(stats)


# ---------------------------------------------------------------------------
# benchmark() — register-once driver, log-level + capture seams
# ---------------------------------------------------------------------------


class _FakeWorker:
    """A ``ChipWorker`` stand-in: context manager handing out one counting handle."""

    def __init__(self) -> None:
        self.register_calls = 0
        self.handle = MagicMock(name="RegistrationHandle")

    def __enter__(self) -> "_FakeWorker":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def register(self, _compiled: object) -> MagicMock:
        self.register_calls += 1
        return self.handle


def _compiled_mock() -> MagicMock:
    cp = MagicMock(name="CompiledProgram")
    cp.platform = "a2a3sim"
    cp.runtime_name = "tensormap_and_ringbuffer"
    return cp


def _run_benchmark(*, rounds: int, warmup: int, **kwargs: Any):
    """Run ``benchmark`` with the worker, log-level, and parse seams patched."""
    worker = _FakeWorker()
    sentinel = BenchmarkStats(device_wall_us=[1.0], host_wall_us=[2.0], rounds=rounds, warmup=warmup)
    with (
        patch("pypto.runtime.bench.ChipWorker", return_value=worker) as ctor,
        patch("pypto.runtime.bench.configure_log") as cfg,
        patch("pypto.runtime.bench.current_level", return_value=20),
        patch("pypto.runtime.bench._parse_stats_from_strace", return_value=sentinel) as parse,
    ):
        stats = benchmark(_compiled_mock(), [MagicMock(name="arg")], rounds=rounds, warmup=warmup, **kwargs)
    return stats, worker, ctor, cfg, parse


def test_benchmark_registers_once_and_loops_warmup_plus_rounds():
    stats, worker, _ctor, _cfg, parse = _run_benchmark(rounds=3, warmup=2)
    assert worker.register_calls == 1  # registered exactly once
    assert worker.handle.call_count == 5  # warmup + rounds launches
    # The captured log text is forwarded to the parser with rounds/warmup + the
    # L2/L3 selector (a plain CompiledProgram mock -> distributed=False).
    assert parse.call_args.kwargs == {"rounds": 3, "warmup": 2, "distributed": False}
    assert stats.rounds == 3


def test_benchmark_raises_log_level_to_v9_and_restores():
    _stats, _worker, _ctor, cfg, _parse = _run_benchmark(rounds=1, warmup=0)
    # First call raises to v9; the final call restores the saved level (20).
    assert cfg.call_args_list[0].args == ("v9",)
    assert cfg.call_args_list[-1].args == (20,)


def test_benchmark_binds_worker_to_compiled_runtime():
    _stats, _worker, ctor, _cfg, _parse = _run_benchmark(rounds=1, warmup=0)
    assert ctor.call_args.kwargs["runtime"] == "tensormap_and_ringbuffer"


def test_benchmark_platform_device_id_build_runconfig():
    _stats, _worker, ctor, _cfg, _parse = _run_benchmark(rounds=1, warmup=0, platform="a2a3", device_id=2)
    rc = ctor.call_args.args[0]  # ChipWorker(rc, runtime=...)
    assert rc.platform == "a2a3"
    assert rc.device_id == 2


def test_benchmark_rejects_bad_rounds_warmup():
    with pytest.raises(ValueError, match="rounds must be positive"):
        benchmark(_compiled_mock(), [MagicMock()], rounds=0)
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        benchmark(_compiled_mock(), [MagicMock()], warmup=-1)


def test_benchmark_rejects_config_with_platform():
    with pytest.raises(ValueError, match="not both"):
        benchmark(_compiled_mock(), [MagicMock()], config=RunConfig(platform="a2a3"), platform="a2a3")


class _FakeDistributedWorker:
    """A ``DistributedWorker`` stand-in: context manager handing out one handle."""

    def __init__(self) -> None:
        self.register_calls = 0
        self.handle = MagicMock(name="RegistrationHandle")

    def __enter__(self) -> "_FakeDistributedWorker":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def register(self, _compiled: object) -> MagicMock:
        self.register_calls += 1
        return self.handle


class _FakeDistributedCompiled:
    """A ``DistributedCompiledProgram`` stand-in whose ``prepare()`` hands out *rt*."""

    def __init__(self, rt: _FakeDistributedWorker) -> None:
        self._rt = rt
        self.platform = "a2a3sim"

    def prepare(self) -> _FakeDistributedWorker:
        return self._rt


def test_benchmark_l3_dispatches_via_distributed_worker():
    """An L3 program routes through ``prepare()`` / ``DistributedWorker``, not ``ChipWorker``."""
    rt = _FakeDistributedWorker()
    sentinel = BenchmarkStats(device_wall_us=[1.0], host_wall_us=[2.0], rounds=2, warmup=1)
    with (
        patch.object(dcp_mod, "DistributedCompiledProgram", _FakeDistributedCompiled),
        patch("pypto.runtime.bench.ChipWorker") as chip_ctor,
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
        patch("pypto.runtime.bench._parse_stats_from_strace", return_value=sentinel) as parse,
    ):
        compiled = _FakeDistributedCompiled(rt)
        stats = benchmark(compiled, [MagicMock(name="arg")], rounds=2, warmup=1)

    assert rt.register_calls == 1  # registered exactly once
    assert rt.handle.call_count == 3  # warmup + rounds launches
    assert chip_ctor.call_count == 0  # L3 must NOT touch ChipWorker
    # The parser is told this is a distributed run.
    assert parse.call_args.kwargs == {"rounds": 2, "warmup": 1, "distributed": True}
    assert stats.rounds == 2


def test_benchmark_l3_rejects_platform_device_id():
    """platform=/device_id= do not apply to L3 (device set is compile-fixed)."""
    rt = _FakeDistributedWorker()
    with (
        patch.object(dcp_mod, "DistributedCompiledProgram", _FakeDistributedCompiled),
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
    ):
        with pytest.raises(ValueError, match="do not apply to an L3"):
            benchmark(_FakeDistributedCompiled(rt), [MagicMock()], rounds=1, warmup=0, platform="a2a3")
        with pytest.raises(ValueError, match="do not apply to an L3"):
            benchmark(_FakeDistributedCompiled(rt), [MagicMock()], rounds=1, warmup=0, device_id=2)


def test_benchmark_l3_capture_wraps_prepare(require_simpler):
    """Markers emitted during ``prepare()`` are captured — the stderr redirect
    must wrap ``prepare()`` (where chip workers fork), not just the loop.

    A real L3 chip child writes its ``[STRACE]`` markers from a process forked
    inside ``prepare()``. Here a fake ``prepare()`` writes rank markers straight
    to fd 2; if the capture only wrapped the loop they would escape to the real
    stderr and the parse would find nothing. Using the real parser, we assert the
    prepare-time markers were captured and aggregated.
    """

    class _CompiledEmittingAtPrepare(_FakeDistributedCompiled):
        def prepare(self) -> _FakeDistributedWorker:
            # Two ranks each emit one dispatch's markers at fork/prepare time,
            # before the measured loop runs.
            for pid, dev_us in ((100, 10.0), (101, 20.0)):
                for line in _launch_lines(0, host_us=5.0, device_us=dev_us, pid=pid):
                    os.write(2, (line + "\n").encode())
            return self._rt

    rt = _FakeDistributedWorker()
    with (
        patch.object(dcp_mod, "DistributedCompiledProgram", _FakeDistributedCompiled),
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
    ):
        # handle() is a mock (emits nothing), so all markers come from prepare().
        stats = benchmark(_CompiledEmittingAtPrepare(rt), [MagicMock(name="arg")], rounds=1, warmup=0)

    # Prepare-time markers were captured and aggregated (per-round max across ranks).
    assert stats.device_wall_us == [20.0]
    assert set(stats.per_rank("device")) == {100, 101}


def test_benchmark_raises_when_no_markers_captured():
    """A runtime built without SIMPLER_PROFILING emits no markers; the parser
    returns empty stats and ``benchmark`` surfaces a clear error rather than a
    silently-empty result (which callers could misread as 0 device timing)."""
    worker = _FakeWorker()
    empty = BenchmarkStats(rounds=1, warmup=0)  # no markers -> empty host/device
    with (
        patch("pypto.runtime.bench.ChipWorker", return_value=worker),
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
        patch("pypto.runtime.bench._parse_stats_from_strace", return_value=empty),
        pytest.raises(RuntimeError, match="no \\[STRACE\\] markers captured"),
    ):
        benchmark(_compiled_mock(), [MagicMock(name="arg")], rounds=1, warmup=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
