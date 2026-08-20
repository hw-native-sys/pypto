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
installed** (e.g. the unit-test CI host) via the ``span_root`` fixture.
The ``benchmark`` driver (register-once, warmup, log-level + stderr capture) and
the pure-``BenchmarkStats`` aggregate helpers patch the parse seam out, so they
run everywhere without ``simpler``.
"""

import os
import struct
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pypto.ir import distributed_compiled_program as dcp_mod
from pypto.runtime import RunConfig
from pypto.runtime.bench import (
    _L3_SWIMLANE_GRAPH_BEGIN,
    _L3_SWIMLANE_GRAPH_END,
    _L3_SWIMLANE_TIMING_BEGIN,
    _L3_SWIMLANE_TIMING_END,
    _LEGACY_SPAN_NAMES,
    BenchmarkStats,
    _parse_stats_from_strace,
    _span_names,
    benchmark,
)
from pypto.runtime.elf_parser import elf_build_id_64, fnv1a_64


@pytest.fixture
def span_root() -> str:
    """Skip unless the optional ``simpler`` runtime is importable; return the
    ``[STRACE]`` span root :func:`_parse_stats_from_strace` will look for.

    That function lazily imports ``simpler_setup.tools.strace_timing`` (the
    single source of truth for the ``[STRACE]`` grammar *and* span names), so
    the synthetic markers below build their names off the same
    :func:`_span_names` resolution the parser uses rather than hardcoding a
    root. The root has been renamed twice (``run_prepared`` -> ``simpler_run``
    in simpler #1210, then ``simpler_run`` -> ``chip.run`` in #1877/#1893), and
    these tests exercise the parse *logic*, not the spelling — the spelling is
    what :func:`test_span_names_resolve_from_installed_runtime` guards.

    ``strace_timing`` is absent on the unit-test CI host, where these parse
    tests skip.
    """
    pytest.importorskip("simpler_setup.tools.strace_timing")
    return _span_names()["host"]


def test_span_names_resolve_from_installed_runtime():
    """``_span_names`` must read the installed runtime, not its legacy fallback.

    The span names live in ``strace_timing._ROUNDS_TABLE_NAMES``, whose keys and
    values both moved when simpler renamed the span ladder (#1877/#1893: the
    whole-run entry re-keyed ``host`` -> ``run``, every value re-rooted
    ``simpler_run`` -> ``chip.run``). :func:`_span_names` swallows a lookup miss
    and returns :data:`_LEGACY_SPAN_NAMES`, which no *table-bearing* runtime
    emits — so a future rename would leave every ``host_wall_us`` /
    ``device_wall_us`` sample silently 0.0 rather than raising. The ``span_root``
    fixture cannot catch that (it would drift to the same fallback and stay
    green), so assert here that the resolution actually came from the runtime.

    Gated on the table existing. A pre-#1210 simpler has no
    ``_ROUNDS_TABLE_NAMES`` at all, and there the legacy names are the documented
    answer rather than drift — asserting against them would fail a configuration
    :func:`_span_names` still supports. Once the table exists, a fallback can
    only mean pypto does not know how that table spells its keys, which is
    exactly what this guards.
    """
    strace_timing = pytest.importorskip("simpler_setup.tools.strace_timing")
    if not hasattr(strace_timing, "_ROUNDS_TABLE_NAMES"):
        pytest.skip("pre-#1210 simpler has no _ROUNDS_TABLE_NAMES; the legacy fallback is correct there")
    names = _span_names()

    assert names != _LEGACY_SPAN_NAMES, (
        "_span_names() fell back to the legacy pre-#1210 names, so the installed "
        "simpler spells _ROUNDS_TABLE_NAMES differently than pypto expects; "
        "every benchmark timing sample would parse as 0.0. Teach "
        "_RUNTIME_SPAN_KEYS the new spelling."
    )
    # The three subdivisions all nest under the run root, so the resolved set
    # must describe one tree — a half-migrated table would not.
    root = names["host"]
    for key in ("device", "orch", "sched"):
        assert names[key].startswith(f"{root}."), f"{key}={names[key]!r} is not nested under {root!r}"


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
    """One synthetic ``[STRACE]`` marker line (matches strace_timing's grammar).

    Only the ``name=`` field is parsed for the span tree; the leading log tag is
    ignored by ``strace_timing``'s regex.
    """
    attrs = " clk=dev" if dev else ""
    ts_ns = ts if ts is not None else inv * 1000
    return (
        f"[2026-01-01][T0x1][TIMING] {name}: [STRACE] v=1 pid={pid} tid=1 "
        f"inv={inv} hid={hid} depth={depth} name={name} ts={ts_ns} dur={dur_ns}{attrs}"
    )


def _launch_lines(
    inv: int, root: str, *, host_us: float, device_us: float, pid: int = 100, hid: str = "abc"
) -> list[str]:
    """The two markers one launch emits: the host span (*root*) + device wall."""
    return [
        _strace_line(inv, root, int(host_us * 1000), depth=0, pid=pid, hid=hid),
        _strace_line(
            inv,
            f"{root}.runner_run.device_wall",
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


def test_parse_discards_warmup_and_collects_rounds(span_root):
    """Warmup invocations are dropped; only the trailing ``rounds`` are measured."""
    lines: list[str] = []
    # 2 warmup launches (inv 0,1) then 3 measured (inv 2,3,4).
    lines += _launch_lines(0, span_root, host_us=99, device_us=99)
    lines += _launch_lines(1, span_root, host_us=99, device_us=99)
    lines += _launch_lines(2, span_root, host_us=100, device_us=10)
    lines += _launch_lines(3, span_root, host_us=200, device_us=20)
    lines += _launch_lines(4, span_root, host_us=300, device_us=30)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=3, warmup=2)

    assert stats.device_wall_us == [10.0, 20.0, 30.0]
    assert stats.host_wall_us == [100.0, 200.0, 300.0]
    assert stats.rounds == 3
    assert stats.warmup == 2


def test_parse_no_warmup_keeps_all(span_root):
    lines = _launch_lines(0, span_root, host_us=50, device_us=5) + _launch_lines(
        1, span_root, host_us=60, device_us=15
    )
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)
    assert stats.device_wall_us == [5.0, 15.0]
    assert stats.host_wall_us == [50.0, 60.0]


def test_parse_no_device_span_reads_zero(span_root):
    """On sim / non-profiling builds only the host span is emitted -> device 0."""
    lines = [_strace_line(0, span_root, 50_000, depth=0)]
    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)
    assert stats.host_wall_us == [50.0]
    assert stats.device_wall_us == [0.0]
    assert stats.all_zero_device is True


def test_parse_no_markers_returns_empty(span_root):
    stats = _parse_stats_from_strace("no strace markers here\n", rounds=5, warmup=1)
    assert stats.device_wall_us == []
    assert stats.host_wall_us == []
    assert stats.invocations == []


@pytest.mark.parametrize("scheduler_span", ["l3.dispatch", "node.dispatch", "network1.dispatch"])
def test_parse_ignores_host_scheduling_spans(span_root, monkeypatch, scheduler_span):
    """Per-task host-scheduler markers must not form a bogus benchmark lane.

    These carry no invocation id, so admitting one groups every such span into a
    single forged invocation. simpler #1893 re-spelled the family after the
    emitting level's topology position (``l3.`` -> ``node.``, plus ``network1..3``
    for the hops above it), so all three spellings are covered.
    """
    lines = [_strace_line(0, scheduler_span, 900_000, pid=99, hid="0")]
    lines += _launch_lines(0, span_root, host_us=50, device_us=5, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)

    assert stats.host_wall_us == [50.0]
    assert stats.device_wall_us == [5.0]
    assert len(stats.invocations) == 1

    # Older compatible runtimes predate simpler's own filter, and #1877 renamed
    # it ``legacy_spans`` -> ``invocation_spans``. With neither present PyPTO
    # still filters the namespaces locally instead of failing import or
    # selecting the bogus lane.
    strace_timing = sys.modules["simpler_setup.tools.strace_timing"]
    monkeypatch.delattr(strace_timing, "invocation_spans", raising=False)
    monkeypatch.delattr(strace_timing, "legacy_spans", raising=False)
    fallback_stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)
    assert fallback_stats.host_wall_us == [50.0]
    assert fallback_stats.device_wall_us == [5.0]


def test_parse_populates_full_span_tree_and_format(span_root):
    """Each measured launch keeps its full span tree; format_tree draws the
    hierarchy with ``|-`` / `` `- `` connectors and tags device spans."""
    # A branching tree (siblings tie on ts -> kept in line order):
    #   <root>
    #   |- bind
    #   |  |- args
    #   |  `- prebuilt
    #   `- runner_run
    #      `- device_wall [dev]
    lines = [
        _strace_line(0, span_root, 10_000, depth=0),
        _strace_line(0, f"{span_root}.bind", 6_000, depth=1),
        _strace_line(0, f"{span_root}.bind.args", 4_000, depth=2),
        _strace_line(0, f"{span_root}.bind.prebuilt", 2_000, depth=2),
        _strace_line(0, f"{span_root}.runner_run", 3_000, depth=1),
        _strace_line(0, f"{span_root}.runner_run.device_wall", 2_000, depth=2, dev=True),
    ]
    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)

    assert stats.device_wall_us == [2.0]
    assert stats.host_wall_us == [10.0]
    assert len(stats.invocations) == 1
    inv = stats.invocations[0]
    root = inv.root()
    assert root is not None
    assert root.name == span_root
    assert inv.by_name()[f"{span_root}.runner_run.device_wall"].is_device

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


def test_mean_tree_averages_durations_across_launches(span_root):
    """The mean tree averages each span's duration across measured launches."""
    # Two launches; <root> -> runner_run.device_wall. Device wall is 10
    # then 20 us -> mean 15; host <root> 100 then 300 -> mean 200.
    lines = []
    for inv, host_us, dev_us in [(0, 100.0, 10.0), (1, 300.0, 20.0)]:
        lines.append(_strace_line(inv, span_root, int(host_us * 1000), depth=0))
        lines.append(
            _strace_line(inv, f"{span_root}.runner_run.device_wall", int(dev_us * 1000), depth=2, dev=True)
        )
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)

    mean = stats.mean_invocation()
    assert mean is not None
    by = mean.by_name()
    assert by[span_root].dur == 200_000  # mean of 100k, 300k ns
    assert by[f"{span_root}.runner_run.device_wall"].dur == 15_000  # mean of 10k, 20k
    assert by[f"{span_root}.runner_run.device_wall"].is_device

    tree = stats.format_mean_tree()
    assert "mean of 2 launches" in tree
    assert _row_present(tree, f"{span_root} 200.0us")
    assert _row_present(tree, "device_wall [dev] 15.0us")


def test_mean_tree_spread_annotations(span_root):
    """Mean-tree nodes carry ±stdev and [min..max] across launches."""
    lines = []
    for inv, host_us, dev_us in [(0, 100.0, 10.0), (1, 300.0, 20.0)]:
        lines.append(_strace_line(inv, span_root, int(host_us * 1000), depth=0))
        lines.append(
            _strace_line(inv, f"{span_root}.runner_run.device_wall", int(dev_us * 1000), depth=2, dev=True)
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


def test_parse_l3_two_ranks_per_round_max(span_root):
    """Two ranks (pids), one dispatch/round each: headline = per-round max across ranks."""
    lines: list[str] = []
    # rank0 (pid 100): warmup inv0, measured inv1=10us, inv2=30us device.
    lines += _launch_lines(0, span_root, host_us=99, device_us=99, pid=100)
    lines += _launch_lines(1, span_root, host_us=100, device_us=10, pid=100)
    lines += _launch_lines(2, span_root, host_us=300, device_us=30, pid=100)
    # rank1 (pid 101): warmup inv0, measured inv1=20us, inv2=5us device.
    lines += _launch_lines(0, span_root, host_us=99, device_us=99, pid=101)
    lines += _launch_lines(1, span_root, host_us=200, device_us=20, pid=101)
    lines += _launch_lines(2, span_root, host_us=50, device_us=5, pid=101)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 30.0], 101: [20.0, 5.0]}
    assert stats.per_rank("host") == {100: [100.0, 300.0], 101: [200.0, 50.0]}
    # Per-round max across ranks: round0 = max(10,20)=20; round1 = max(30,5)=30.
    assert stats.device_wall_us == [20.0, 30.0]
    assert stats.host_wall_us == [200.0, 300.0]


def test_parse_l3_two_pass_swimlane_keeps_clean_timing_half(span_root):
    """Prepared swimlane benchmark excludes every dep-gen graph pass."""
    lines: list[str] = []
    timings = {100: [99.0, 10.0, 30.0], 101: [99.0, 20.0, 5.0]}
    invs = {100: 0, 101: 0}
    for launch in range(3):  # one warmup + two measured launches
        lines.append(_L3_SWIMLANE_GRAPH_BEGIN)
        for pid in timings:
            # Deliberately give the graph pass two dispatches and timing one.
            # Boundary sentinels, rather than an unsafe "keep the latter half"
            # count heuristic, must decide what survives.
            lines += _launch_lines(invs[pid], span_root, host_us=900, device_us=900, pid=pid)
            invs[pid] += 1
            lines += _launch_lines(invs[pid], span_root, host_us=901, device_us=901, pid=pid)
            invs[pid] += 1
        lines.append(_L3_SWIMLANE_GRAPH_END)
        timing_lines: list[str] = []
        for pid, timing_us in timings.items():
            clean_us = timing_us[launch]
            timing_lines += _launch_lines(
                invs[pid],
                span_root,
                host_us=clean_us * 10,
                device_us=clean_us,
                pid=pid,
            )
            invs[pid] += 1
        # Model shared-fd writes that concatenate a pass sentinel and STRACE
        # record onto one physical line; substring boundary extraction must
        # still retain every clean record.
        lines.append(_L3_SWIMLANE_TIMING_BEGIN + timing_lines[0])
        lines.extend(timing_lines[1:-1])
        lines.append(timing_lines[-1] + _L3_SWIMLANE_TIMING_END)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 30.0], 101: [20.0, 5.0]}
    assert stats.device_wall_us == [20.0, 30.0]
    assert all(inv.device_wall_us < 900 for inv in stats.invocations)


def test_parse_l3_incomplete_timing_region_returns_no_contaminated_samples(span_root):
    """An incomplete capture must not silently mix graph and timing passes."""
    lines = [_L3_SWIMLANE_TIMING_BEGIN]
    lines += _launch_lines(0, span_root, host_us=900, device_us=900, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0, distributed=True)

    assert stats.fallback_flattened is True
    assert stats.device_wall_us == []
    assert stats.host_wall_us == []


def test_parse_l3_ignores_prepare_time_prewarm_groups(span_root):
    """Setup-only prewarm groups do not break per-rank round segmentation."""
    lines: list[str] = []
    for pid, measured_us in ((100, [10.0, 30.0]), (101, [20.0, 5.0])):
        # ``prepare()`` runs before benchmark dispatches while stderr is already
        # captured. Simpler's arena prewarm emits this non-dispatch STRACE group
        # with no canonical run root or device-wall span.
        lines.append(_strace_line(0, "chip.prewarm.build", 800_000, pid=pid, hid="0"))
        lines += _launch_lines(1, span_root, host_us=99, device_us=99, pid=pid)  # warmup
        lines += _launch_lines(2, span_root, host_us=100, device_us=measured_us[0], pid=pid)
        lines += _launch_lines(3, span_root, host_us=300, device_us=measured_us[1], pid=pid)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 30.0], 101: [20.0, 5.0]}
    assert len(stats.invocations) == 4
    roots = [inv.root() for inv in stats.invocations]
    assert all(root is not None for root in roots)
    assert {root.name for root in roots if root is not None} == {span_root}


def test_parse_l3_keeps_dispatch_missing_one_device_marker(span_root):
    """A real dispatch with no device span remains aligned and reports zero."""
    lines = [_strace_line(0, span_root, 100_000, depth=0, pid=100)]
    lines += _launch_lines(1, span_root, host_us=300, device_us=30, pid=100)
    lines += _launch_lines(0, span_root, host_us=200, device_us=20, pid=101)
    lines += _launch_lines(1, span_root, host_us=50, device_us=5, pid=101)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [0.0, 30.0], 101: [20.0, 5.0]}
    assert len(stats.rounds_dispatches[0][100]) == 1


def test_parse_l3_recovers_interleaved_records_on_one_line(span_root, monkeypatch):
    """Two ``[STRACE]`` records mashed onto one physical line are both recovered.

    L3 forks one chip worker per rank sharing the capture fd; concurrent writes
    can interleave two complete records onto a single physical line. The parser
    must recover both records — dropping the second would zero that (round,
    rank). PyPTO normalizes the records before calling the runtime parser so
    this also works with older compatible parser implementations.
    """
    lines = _launch_lines(0, span_root, host_us=100, device_us=10, pid=100)
    lines += _launch_lines(1, span_root, host_us=300, device_us=30, pid=100)
    # Simulate the wire collision: inv0's device-wall record and inv1's host
    # record land on the same physical line (no newline between them). Every
    # other record keeps its own line.
    mashed = "\n".join([lines[0], f"{lines[1]} {lines[2]}", lines[3]])

    # Model the parser before Simpler c9ccaf65 (#1691), which yielded only the
    # first record from each physical line. The production normalization must
    # split the mashed line before this parser sees it.
    strace_timing = sys.modules["simpler_setup.tools.strace_timing"]
    parse_all = strace_timing.parse_spans

    def parse_first_per_line(raw_lines):
        for raw_line in raw_lines:
            for span in parse_all([raw_line]):
                yield span
                break

    monkeypatch.setattr(strace_timing, "parse_spans", parse_first_per_line)

    stats = _parse_stats_from_strace(mashed, rounds=2, warmup=0, distributed=True)

    # Both invocations survive after marker normalization.
    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 30.0]}
    assert stats.per_rank("host") == {100: [100.0, 300.0]}


def test_parse_l3_multi_dispatch_sums_within_round(span_root):
    """A rank dispatched multiple times per round (heterogeneous hids) sums within the round."""
    lines: list[str] = []
    # rank0 (pid 100): 2 dispatches/round, different hids. No warmup.
    # round0 = inv0(hidA,4us) + inv1(hidB,6us) = 10us; round1 = inv2(3us)+inv3(7us) = 10us.
    lines += _launch_lines(0, span_root, host_us=1, device_us=4, pid=100, hid="A")
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100, hid="B")
    lines += _launch_lines(2, span_root, host_us=1, device_us=3, pid=100, hid="A")
    lines += _launch_lines(3, span_root, host_us=1, device_us=7, pid=100, hid="B")
    # rank1 (pid 101): 1 dispatch/round -> different per-rank d.
    lines += _launch_lines(0, span_root, host_us=1, device_us=5, pid=101, hid="C")
    lines += _launch_lines(1, span_root, host_us=1, device_us=8, pid=101, hid="C")

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.per_rank("device") == {100: [10.0, 10.0], 101: [5.0, 8.0]}
    # Per-round max: round0 = max(10,5)=10; round1 = max(10,8)=10.
    assert stats.device_wall_us == [10.0, 10.0]


def _l3_multi_dispatch_lines(span_root: str) -> list[str]:
    """Markers for 2 rounds where rank 100 dispatches twice (hids a, b) per round.

    rank 100: round0 = a(4us) + b(6us), round1 = a(3us) + b(7us).
    rank 101: one dispatch/round (hid c): 5us then 8us.
    """
    lines: list[str] = []
    lines += _launch_lines(0, span_root, host_us=1, device_us=4, pid=100, hid="a")
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100, hid="b")
    lines += _launch_lines(2, span_root, host_us=1, device_us=3, pid=100, hid="a")
    lines += _launch_lines(3, span_root, host_us=1, device_us=7, pid=100, hid="b")
    lines += _launch_lines(0, span_root, host_us=1, device_us=5, pid=101, hid="c")
    lines += _launch_lines(1, span_root, host_us=1, device_us=8, pid=101, hid="c")
    return lines


def test_parse_l3_per_dispatch_keeps_a_ranks_dispatches_separate(span_root):
    """``per_dispatch`` does not fuse a rank's several dispatches per round.

    ``per_rank`` sums rank 100's two dispatches into one per-round busy figure;
    ``per_dispatch`` keys on ``(pid, slot)`` so each dispatch keeps its own series.
    """
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )

    assert stats.fallback_flattened is False
    # Un-fused: one series per (pid, slot), ordered by (pid, slot).
    assert stats.per_dispatch("device") == {
        (100, 0): [4.0, 3.0],
        (100, 1): [6.0, 7.0],
        (101, 0): [5.0, 8.0],
    }
    # Slots are identified by their task (callable hash).
    assert stats.dispatch_tasks() == {(100, 0): "a", (100, 1): "b", (101, 0): "c"}
    # The summed per-rank view is unchanged (4+6, 3+7).
    assert stats.per_rank("device")[100] == [10.0, 10.0]
    # Each (pid, slot) group holds one TraceInvocation per measured round.
    groups = stats.dispatch_groups()
    assert [d.device_wall_us for d in groups[100, 1]] == [6.0, 7.0]


def test_parse_l3_reordered_dispatches_disable_the_per_dispatch_views(span_root):
    """A rank that swaps its dispatch order between rounds must not be grouped.

    The dispatch *count* is constant (so the round boundaries hold), but rank 100
    issues ``a, b`` in round 0 and ``b, a`` in round 1. Keying on the ordinal slot
    would average two different callables together under round 0's label — the
    fusing these views exist to remove — so the per-dispatch views report empty
    while the order-independent per-rank sums stay valid.
    """
    lines: list[str] = []
    # round 0: a(4us) then b(6us);  round 1: b(7us) then a(3us).
    lines += _launch_lines(0, span_root, host_us=1, device_us=4, pid=100, hid="a")
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100, hid="b")
    lines += _launch_lines(2, span_root, host_us=1, device_us=7, pid=100, hid="b")
    lines += _launch_lines(3, span_root, host_us=1, device_us=3, pid=100, hid="a")

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is False  # round segmentation is still sound
    assert stats.unstable_dispatch_slots is True
    assert stats.per_dispatch("device") == {}
    assert stats.dispatch_groups() == {}
    assert stats.dispatch_tasks() == {}
    # Per-rank sums are order-independent, so they remain correct and populated.
    assert stats.per_rank("device") == {100: [10.0, 10.0]}
    assert stats.device_wall_us == [10.0, 10.0]
    # The mean tree says why rather than blending the two callables into one tree.
    tree = stats.format_mean_tree()
    assert "per-dispatch view unavailable" in tree
    assert "dispatch pid=" not in tree
    assert stats.mean_invocation() is None
    # __str__ drops its per-dispatch line rather than mislabelling the slots.
    assert "per-dispatch" not in str(stats)


def test_parse_l3_stable_slots_are_not_flagged(span_root):
    """The same callables in the same order every round group as before."""
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )
    assert stats.unstable_dispatch_slots is False
    assert set(stats.per_dispatch("device")) == {(100, 0), (100, 1), (101, 0)}


def test_per_dispatch_host_metric_and_bad_metric(span_root):
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )
    assert stats.per_dispatch("host") == {
        (100, 0): [1.0, 1.0],
        (100, 1): [1.0, 1.0],
        (101, 0): [1.0, 1.0],
    }
    with pytest.raises(ValueError, match="per_dispatch\\(\\): metric must be one of"):
        stats.per_dispatch("nope")


def test_per_dispatch_empty_without_dispatch_grid():
    """No L3 grid (L2 / flatten fallback) -> the per-dispatch views are empty."""
    stats = BenchmarkStats(device_wall_us=[1.0], host_wall_us=[2.0], rounds=1, warmup=0)
    assert stats.per_dispatch("device") == {}
    assert stats.dispatch_groups() == {}
    assert stats.dispatch_tasks() == {}


def test_mean_tree_renders_one_tree_per_dispatch(span_root):
    """The mean tree groups per ``(pid, slot)`` instead of averaging dispatches.

    Rank 100's two dispatches (4/3us and 6/7us device) must show as 3.5us and
    6.5us trees — a single fused tree would report their 5.0us average.
    """
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )

    tree = stats.format_mean_tree(spread="none")
    assert "dispatch pid=100 slot=0 task=a — mean of 2 launches" in tree
    assert "dispatch pid=100 slot=1 task=b — mean of 2 launches" in tree
    assert "dispatch pid=101 slot=0 task=c — mean of 2 launches" in tree
    assert _row_present(tree, "device_wall [dev] 3.5us")  # (4+3)/2, not fused with slot 1
    assert _row_present(tree, "device_wall [dev] 6.5us")  # (6+7)/2
    assert not _row_present(tree, "device_wall [dev] 5.0us")  # the old fused average

    # Selectors narrow the rendering to one dispatch.
    one = stats.format_mean_tree(spread="none", pid=100, slot=1)
    assert "slot=1" in one and "slot=0" not in one
    assert _row_present(one, "device_wall [dev] 6.5us")
    assert "no dispatch matches" in stats.format_mean_tree(pid=999)


def test_mean_invocation_requires_a_single_dispatch(span_root):
    """``mean_invocation`` refuses to average distinct dispatches together."""
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )

    with pytest.raises(ValueError, match="dispatches match pid=None slot=None"):
        stats.mean_invocation()
    # Narrowed to one dispatch -> the mean of just that dispatch's launches.
    mean = stats.mean_invocation(pid=100, slot=1)
    assert mean is not None
    assert mean.device_wall_us == 6.5
    assert stats.mean_invocation(pid=999) is None


def test_format_tree_labels_round_and_slot(span_root):
    """L3 launch headers carry the (round, slot) so repeats are distinguishable."""
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )
    header_lines = [line for line in stats.format_tree().splitlines() if line.startswith("launch[")]
    assert any("round=0 slot=0" in line for line in header_lines)
    assert any("round=0 slot=1" in line for line in header_lines)
    assert any("round=1 slot=1" in line for line in header_lines)


def test_str_breaks_down_fused_dispatches(span_root):
    """``__str__`` adds a per-dispatch line when a rank dispatches more than once."""
    stats = _parse_stats_from_strace(
        "\n".join(_l3_multi_dispatch_lines(span_root)), rounds=2, warmup=0, distributed=True
    )
    text = str(stats)
    assert "per-dispatch device mean us:" in text
    assert "(pid=100,slot=0,task=a)=3.5" in text
    assert "(pid=100,slot=1,task=b)=6.5" in text

    # One dispatch per rank: nothing is fused, so no extra line.
    single = _parse_stats_from_strace(
        "\n".join(_launch_lines(0, span_root, host_us=1, device_us=5, pid=100)),
        rounds=1,
        warmup=0,
        distributed=True,
    )
    assert "per-dispatch" not in str(single)


def test_parse_l3_non_divisible_falls_back_to_flattened(span_root):
    """A non-deterministic dispatch shape (count not divisible by launches) flattens."""
    lines: list[str] = []
    # 3 invocations, rounds=2 warmup=0 -> 3 % 2 != 0 -> fallback.
    lines += _launch_lines(0, span_root, host_us=1, device_us=5, pid=100)
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100)
    lines += _launch_lines(2, span_root, host_us=1, device_us=7, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0, distributed=True)

    assert stats.fallback_flattened is True
    assert stats.per_rank("device") == {}
    # Flattened pool of all per-dispatch device walls (warmup=0 -> none dropped).
    assert sorted(stats.device_wall_us) == [5.0, 6.0, 7.0]
    # No round alignment -> no cross-rank union window.
    assert stats.per_round("union") == []


def test_parse_l3_union_window_captures_cross_rank_skew(span_root):
    """``per_round("union")`` = cross-rank host-timeline window (max end - min start).

    ``_strace_line`` sets each span's host ``ts`` to ``inv * 1000`` ns, so giving
    the two ranks different ``inv`` values models a cross-rank start skew.
    """
    lines: list[str] = []
    # rank0 (pid100): inv=0 -> host span window [0, 5000] ns.
    lines += _launch_lines(0, span_root, host_us=5, device_us=10, pid=100)
    # rank1 (pid101): inv=3 -> starts later, window [3000, 8000] ns.
    lines += _launch_lines(3, span_root, host_us=5, device_us=20, pid=101)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.device_wall_us == [20.0]  # per-round max across ranks
    # Union: max(end)=8000 - min(start)=0 = 8000 ns = 8.0 us.
    assert stats.per_round("union") == [8.0]


def test_parse_l3_rounds_dispatches_round_rank_dispatch_view(span_root):
    """``rounds_dispatches[k][pid]`` gives the per-round, per-rank dispatch list,
    each carrying its task (hid) and precise per-dispatch timing."""
    lines: list[str] = []
    # rank0 (pid100): 2 dispatches/round (hids A, B); rank1 (pid101): 1/round (hid C).
    # round0                          round1
    lines += _launch_lines(0, span_root, host_us=1, device_us=4, pid=100, hid="a")
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100, hid="b")
    lines += _launch_lines(2, span_root, host_us=1, device_us=3, pid=100, hid="a")
    lines += _launch_lines(3, span_root, host_us=1, device_us=7, pid=100, hid="b")
    lines += _launch_lines(0, span_root, host_us=1, device_us=5, pid=101, hid="c")
    lines += _launch_lines(1, span_root, host_us=1, device_us=8, pid=101, hid="c")

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


def test_parse_l3_per_rank_effective_time(span_root):
    """Per-card L2 Effective = orch union sched window; exposed per dispatch and per rank."""
    DEV = f"{span_root}.runner_run.device_wall"

    def dispatch(inv, pid, *, orch, sched):
        # orch / sched are (start_ns, dur_ns) device-domain spans.
        return [
            _strace_line(inv, span_root, 5000, depth=0, pid=pid),
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


def test_parse_l3_degenerates_to_l2_for_single_rank(span_root):
    """One rank, one dispatch/round: L3 aggregation matches the L2 per-launch values."""
    lines: list[str] = []
    lines += _launch_lines(0, span_root, host_us=99, device_us=99, pid=100)  # warmup
    lines += _launch_lines(1, span_root, host_us=100, device_us=10, pid=100)
    lines += _launch_lines(2, span_root, host_us=200, device_us=20, pid=100)

    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=1, distributed=True)

    assert stats.fallback_flattened is False
    assert stats.device_wall_us == [10.0, 20.0]
    assert stats.per_rank("device") == {100: [10.0, 20.0]}


# ---------------------------------------------------------------------------
# Callable identity — resolving a marker ``hid`` to an orchestration name
# ---------------------------------------------------------------------------

# A real aarch64 .so's Build-ID and the 64-bit id the runtime derives from it
# (the first 8 descriptor bytes read little-endian).
_REAL_BUILD_ID = bytes.fromhex("ac6a376891802e4aa47c89a076b5e4b48a461a47")
_REAL_BUILD_ID_64 = 0x4A2E809168376AAC


def _elf64_with_build_id(build_id: bytes | None) -> bytes:
    """A minimal ELF64 carrying one ``PT_NOTE`` segment (Build-ID when given).

    Just enough header for ``elf_build_id_64`` to walk: an ELF64 header pointing
    at a single program header, which points at the note payload.
    """
    if build_id is None:
        note = b""
    else:
        note = struct.pack("<III", 4, len(build_id), 3) + b"GNU\x00" + build_id
        note += b"\x00" * (-len(note) % 4)
    phoff, note_off = 64, 120

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4:7] = bytes((2, 1, 1))  # ELFCLASS64, little endian, version 1
    struct.pack_into("<Q", ehdr, 0x20, phoff)  # e_phoff
    struct.pack_into("<HH", ehdr, 0x36, 56, 1)  # e_phentsize, e_phnum

    phdr = bytearray(56)
    struct.pack_into("<I", phdr, 0, 4)  # p_type = PT_NOTE
    struct.pack_into("<Q", phdr, 8, note_off)  # p_offset
    struct.pack_into("<Q", phdr, 32, len(note))  # p_filesz

    body = bytes(ehdr) + bytes(phdr)
    return body.ljust(note_off, b"\x00") + note


def test_elf_build_id_64_reads_the_gnu_build_id():
    """``hid`` is the Build-ID's first 8 descriptor bytes, little-endian.

    Pinned against a real ``.so``: ``readelf -n`` reports Build-ID
    ``ac6a3768 91802e4a ...``, which the runtime turns into
    ``0x4a2e809168376aac`` — the value that reaches the ``[STRACE]`` markers.
    """
    assert elf_build_id_64(_elf64_with_build_id(_REAL_BUILD_ID)) == _REAL_BUILD_ID_64


def test_elf_build_id_64_falls_back_to_fnv1a():
    """No Build-ID / not an ELF -> FNV-1a over the whole buffer, as the runtime does."""
    not_elf = b"definitely not an ELF file"
    assert elf_build_id_64(not_elf) == fnv1a_64(not_elf)
    # Truncated input is degenerate, not an error.
    assert elf_build_id_64(b"") == fnv1a_64(b"")
    # Well-formed ELF whose note segment carries no Build-ID.
    no_build_id = _elf64_with_build_id(None)
    assert elf_build_id_64(no_build_id) == fnv1a_64(no_build_id)


def test_callable_display_name_prefers_the_source_stem():
    """The manifest's ``function_name`` cannot tell two callables apart.

    Every generated ``ORCHESTRATION`` declares the same fixed AICPU entry symbol
    (verified on a real v4-flash L3 build: both ``prefill_fwd`` and
    ``lm_head_test`` carry ``function_name="aicpu_orchestration_entry"``), so the
    per-program name has to come from the generated source file's stem.
    """
    dr = pytest.importorskip("pypto.runtime.device_runner")

    entry = "aicpu_orchestration_entry"
    assert (
        dr.callable_display_name(
            {
                "source": "/b/_jit_l3/next_levels/prefill_fwd/orchestration/prefill_fwd.cpp",
                "function_name": entry,
            }
        )
        == "prefill_fwd"
    )
    assert (
        dr.callable_display_name(
            {
                "source": "/b/_jit_l3/next_levels/lm_head_test/orchestration/lm_head_test.cpp",
                "function_name": entry,
            }
        )
        == "lm_head_test"
    )
    # No source in the manifest: fall back rather than losing the label entirely.
    assert dr.callable_display_name({"function_name": entry}) == entry


def test_register_callable_identity_maps_hid_to_name():
    """``device_runner`` records hid -> orchestration name at assemble time."""
    dr = pytest.importorskip("pypto.runtime.device_runner")

    so = _elf64_with_build_id(_REAL_BUILD_ID)
    hid = dr.register_callable_identity(so, "decode_orch")

    assert hid == f"{_REAL_BUILD_ID_64:x}"  # marker wire format: lowercase hex
    assert dr.callable_name(hid) == "decode_orch"
    assert dr.callable_name(hid.upper()) == "decode_orch"  # lookup is case-insensitive
    assert dr.callable_name("dead" * 4) is None  # never assembled here


def test_dispatch_tasks_show_orchestration_names(span_root):
    """A measured dispatch is labelled with its orchestration name, not the hash."""
    dr = pytest.importorskip("pypto.runtime.device_runner")

    hid = dr.register_callable_identity(_elf64_with_build_id(_REAL_BUILD_ID), "decode_orch")
    lines = _launch_lines(0, span_root, host_us=1, device_us=4, pid=100, hid=hid)
    lines += _launch_lines(1, span_root, host_us=1, device_us=6, pid=100, hid="feedface")

    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0, distributed=True)

    # Slot 0's hash resolves; slot 1 was never assembled here, so it stays raw.
    assert stats.dispatch_tasks() == {(100, 0): "decode_orch", (100, 1): "feedface"}
    dispatches = stats.rounds_dispatches[0][100]
    assert dispatches[0].task == hid  # .task is still the wire identity
    assert dispatches[0].task_name == "decode_orch"
    assert dispatches[1].task_name == "feedface"

    # Both the mean-tree group headers and the launch headers carry the name.
    assert "dispatch pid=100 slot=0 task=decode_orch" in stats.format_mean_tree()
    assert f"hid={hid} round=0 slot=0 task=decode_orch" in stats.format_tree()
    # An unresolved hid is not annotated twice.
    assert "task=feedface" not in stats.format_tree()


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


def _compiled_mock(*, enable_sdma: bool = False) -> MagicMock:
    cp = MagicMock(name="CompiledProgram")
    cp.platform = "a2a3sim"
    cp.runtime_name = "tensormap_and_ringbuffer"
    cp.runtime_config = {"enable_sdma": True} if enable_sdma else {}
    return cp


def _run_benchmark(
    *,
    rounds: int,
    warmup: int,
    artifact_enable_sdma: bool = False,
    **kwargs: Any,
):
    """Run ``benchmark`` with the worker, log-level, and parse seams patched."""
    worker = _FakeWorker()
    sentinel = BenchmarkStats(device_wall_us=[1.0], host_wall_us=[2.0], rounds=rounds, warmup=warmup)
    with (
        patch("pypto.runtime.bench.ChipWorker", return_value=worker) as ctor,
        patch("pypto.runtime.bench.configure_log") as cfg,
        patch("pypto.runtime.bench.current_level", return_value=20),
        patch("pypto.runtime.bench._parse_stats_from_strace", return_value=sentinel) as parse,
    ):
        stats = benchmark(
            _compiled_mock(enable_sdma=artifact_enable_sdma),
            [MagicMock(name="arg")],
            rounds=rounds,
            warmup=warmup,
            **kwargs,
        )
    return stats, worker, ctor, cfg, parse


def test_benchmark_registers_once_and_loops_warmup_plus_rounds():
    stats, worker, _ctor, _cfg, parse = _run_benchmark(rounds=3, warmup=2)
    assert worker.register_calls == 1  # registered exactly once
    assert worker.handle.call_count == 5  # warmup + rounds launches
    # The captured log text is forwarded to the parser with rounds/warmup + the
    # L2/L3 selector (a plain CompiledProgram mock -> distributed=False).
    assert parse.call_args.kwargs == {"rounds": 3, "warmup": 2, "distributed": False}
    assert stats.rounds == 3


def test_benchmark_sets_log_level_to_timing_and_restores():
    _stats, _worker, _ctor, cfg, _parse = _run_benchmark(rounds=1, warmup=0)
    # First call enables TIMING markers; the final call restores the saved level (20).
    assert cfg.call_args_list[0].args == ("timing",)
    assert cfg.call_args_list[-1].args == (20,)


def test_benchmark_binds_worker_to_compiled_runtime():
    _stats, _worker, ctor, _cfg, _parse = _run_benchmark(rounds=1, warmup=0)
    assert ctor.call_args.kwargs["runtime"] == "tensormap_and_ringbuffer"


@pytest.mark.parametrize(("artifact_enable_sdma", "expected"), [(False, False), (True, True)])
def test_benchmark_binds_worker_to_compiled_sdma_capability(artifact_enable_sdma, expected):
    _stats, _worker, ctor, _cfg, _parse = _run_benchmark(
        rounds=1,
        warmup=0,
        artifact_enable_sdma=artifact_enable_sdma,
    )

    assert ctor.call_args.kwargs["enable_sdma"] is expected


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
        self.prepare_config: Any = "unset"
        self.prepare_kwargs: dict[str, Any] = {}

    def prepare(self, config: Any = None, **kwargs: Any) -> _FakeDistributedWorker:
        self.prepare_config = config
        self.prepare_kwargs = kwargs
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
        stats = benchmark(
            compiled,
            [MagicMock(name="arg")],
            rounds=2,
            warmup=1,
        )

    assert rt.register_calls == 1  # registered exactly once
    assert rt.handle.call_count == 3  # warmup + rounds launches
    assert chip_ctor.call_count == 0  # L3 must NOT touch ChipWorker
    # prepare() gets the dispatch config, so it prewarms the runtime arena with
    # the ring sizing the loop dispatches with (here: None -> baseline sizing).
    assert compiled.prepare_config is None
    assert compiled.prepare_kwargs == {
        "persistent": False,
        "reset_persistent_windows": None,
    }
    # The parser is told this is a distributed run.
    assert parse.call_args.kwargs == {"rounds": 2, "warmup": 1, "distributed": True}
    assert stats.rounds == 2


def test_benchmark_l3_forwards_persistent_options():
    """Persistent L3 benchmark controls are forwarded to ``prepare()``."""
    rt = _FakeDistributedWorker()
    sentinel = BenchmarkStats(device_wall_us=[1.0], host_wall_us=[2.0], rounds=1, warmup=0)
    with (
        patch.object(dcp_mod, "DistributedCompiledProgram", _FakeDistributedCompiled),
        patch("pypto.runtime.bench.ChipWorker") as chip_ctor,
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
        patch("pypto.runtime.bench._parse_stats_from_strace", return_value=sentinel),
    ):
        compiled = _FakeDistributedCompiled(rt)
        benchmark(
            compiled,
            [MagicMock(name="arg")],
            rounds=1,
            warmup=0,
            persistent=True,
            reset_persistent_windows=False,
        )

    assert rt.register_calls == 1
    assert rt.handle.call_count == 1
    assert chip_ctor.call_count == 0
    assert compiled.prepare_kwargs == {
        "persistent": True,
        "reset_persistent_windows": False,
    }


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persistent": True},
        {"reset_persistent_windows": True},
        {"reset_persistent_windows": False},
    ],
)
def test_benchmark_l2_rejects_persistent_options(kwargs):
    """Persistent controls, including explicit ``False``, do not apply to L2."""
    with pytest.raises(ValueError, match="apply only to an L3"):
        benchmark(_compiled_mock(), [MagicMock()], rounds=1, warmup=0, **kwargs)


def test_benchmark_l3_capture_wraps_prepare(span_root):
    """Markers emitted during ``prepare()`` are captured — the stderr redirect
    must wrap ``prepare()`` (where chip workers fork), not just the loop.

    A real L3 chip child writes its ``[STRACE]`` markers from a process forked
    inside ``prepare()``. Here a fake ``prepare()`` writes rank markers straight
    to fd 2; if the capture only wrapped the loop they would escape to the real
    stderr and the parse would find nothing. Using the real parser, we assert the
    prepare-time markers were captured and aggregated.
    """

    class _CompiledEmittingAtPrepare(_FakeDistributedCompiled):
        def prepare(self, config: Any = None, **kwargs: Any) -> _FakeDistributedWorker:
            del config, kwargs
            # Two ranks each emit one dispatch's markers at fork/prepare time,
            # before the measured loop runs.
            for pid, dev_us in ((100, 10.0), (101, 20.0)):
                for line in _launch_lines(0, span_root, host_us=5.0, device_us=dev_us, pid=pid):
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


def test_benchmark_l3_ignores_prepare_setup_groups(span_root):
    """Setup-only groups captured around ``prepare()`` are not dispatches."""

    class _CompiledEmittingPrewarmAtPrepare(_FakeDistributedCompiled):
        def prepare(self, config: Any = None, **kwargs: Any) -> _FakeDistributedWorker:
            del config, kwargs
            for pid in (100, 101):
                line = _strace_line(0, "chip.prewarm.build", 800_000, pid=pid, hid="0")
                os.write(2, (line + "\n").encode())
            return self._rt

    rt = _FakeDistributedWorker()

    def emit_dispatch(*_args: Any, **_kwargs: Any) -> None:
        for pid, dev_us in ((100, 10.0), (101, 20.0)):
            for line in _launch_lines(1, span_root, host_us=5.0, device_us=dev_us, pid=pid):
                os.write(2, (line + "\n").encode())

    rt.handle.side_effect = emit_dispatch
    with (
        patch.object(dcp_mod, "DistributedCompiledProgram", _FakeDistributedCompiled),
        patch("pypto.runtime.bench.configure_log"),
        patch("pypto.runtime.bench.current_level", return_value=20),
    ):
        stats = benchmark(
            _CompiledEmittingPrewarmAtPrepare(rt),
            [MagicMock(name="arg")],
            rounds=1,
            warmup=0,
        )

    assert stats.device_wall_us == [20.0]
    assert set(stats.per_rank("device")) == {100, 101}
    assert all(len(dispatches) == 1 for dispatches in stats.rounds_dispatches[0].values())


def test_benchmark_raises_when_no_markers_captured():
    """A runtime built without SIMPLER_HOST_STRACE emits no markers; the parser
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
