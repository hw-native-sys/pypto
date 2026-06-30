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
The parse + aggregate path (:func:`_parse_stats_from_strace`) is pure host
logic, so it is tested directly against synthetic marker lines — no device and
no ``simpler`` runtime needed. The ``benchmark`` driver (register-once, warmup,
log-level + stderr capture) is tested with the ``ChipWorker`` and the capture /
parse seams patched out.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime import RunConfig
from pypto.runtime.bench import BenchmarkStats, _parse_stats_from_strace, benchmark


def _strace_line(
    inv: int, name: str, dur_ns: int, *, hid: str = "abc", depth: int = 0, dev: bool = False
) -> str:
    """One synthetic ``[STRACE]`` marker line (matches strace_timing's grammar)."""
    attrs = " clk=dev" if dev else ""
    return (
        f"[2026-01-01][T0x1][INFO_V9] run_prepared: [STRACE] v=1 pid=100 tid=1 "
        f"inv={inv} hid={hid} depth={depth} name={name} ts={inv * 1000} dur={dur_ns}{attrs}"
    )


def _launch_lines(inv: int, *, host_us: float, device_us: float) -> list[str]:
    """The two markers one launch emits: the ``run_prepared`` host span + device wall."""
    return [
        _strace_line(inv, "run_prepared", int(host_us * 1000), depth=0),
        _strace_line(inv, "run_prepared.runner_run.device_wall", int(device_us * 1000), depth=2, dev=True),
    ]


# ---------------------------------------------------------------------------
# _parse_stats_from_strace — span extraction, warmup discard, aggregation
# ---------------------------------------------------------------------------


def test_parse_discards_warmup_and_collects_rounds():
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


def test_parse_no_warmup_keeps_all():
    lines = _launch_lines(0, host_us=50, device_us=5) + _launch_lines(1, host_us=60, device_us=15)
    stats = _parse_stats_from_strace("\n".join(lines), rounds=2, warmup=0)
    assert stats.device_wall_us == [5.0, 15.0]
    assert stats.host_wall_us == [50.0, 60.0]


def test_parse_no_device_span_reads_zero():
    """On sim / non-profiling builds only the host span is emitted -> device 0."""
    lines = [_strace_line(0, "run_prepared", 50_000, depth=0)]
    stats = _parse_stats_from_strace("\n".join(lines), rounds=1, warmup=0)
    assert stats.host_wall_us == [50.0]
    assert stats.device_wall_us == [0.0]
    assert stats.all_zero_device is True


def test_parse_no_markers_returns_empty():
    stats = _parse_stats_from_strace("no strace markers here\n", rounds=5, warmup=1)
    assert stats.device_wall_us == []
    assert stats.host_wall_us == []
    assert stats.rounds == 5


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
    # The captured log text is forwarded to the parser with rounds/warmup.
    assert parse.call_args.kwargs == {"rounds": 3, "warmup": 2}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
