# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the register-once benchmark helper (issue #1858).

``benchmark`` is exercised with the ``ChipWorker`` fully patched out — the loop,
warmup discard, and aggregation are pure host logic, so these tests need no
device and no ``simpler`` runtime. ``BenchmarkStats`` aggregation is tested
directly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime.bench import BenchmarkStats, benchmark


def _timing(device_us: float, host_us: float = 0.0):
    """Stand-in for the simpler ``RunTiming`` (only the two fields are read)."""
    return SimpleNamespace(device_wall_us=device_us, host_wall_us=host_us)


class _FakeHandle:
    """A ``RegistrationHandle`` stand-in returning canned timings per launch."""

    def __init__(self, timings):
        self._timings = iter(timings)
        self.calls = 0

    def run_timed(self, *_args, **_kwargs):
        self.calls += 1
        return None, next(self._timings)


class _FakeWorker:
    """A ``ChipWorker`` stand-in: context manager that hands out one handle."""

    def __init__(self, handle):
        self._handle = handle
        self.register_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def register(self, _compiled):
        self.register_calls += 1
        return self._handle


def _compiled_mock():
    cp = MagicMock(name="CompiledProgram")
    cp.platform = "a2a3sim"
    cp.runtime_name = "tensormap_and_ringbuffer"
    return cp


def _run_benchmark(timings, *, rounds, warmup):
    """Run ``benchmark`` with ChipWorker patched to a fake yielding *timings*."""
    handle = _FakeHandle(timings)
    worker = _FakeWorker(handle)
    compiled = _compiled_mock()
    with patch("pypto.runtime.bench.ChipWorker", return_value=worker) as ctor:
        stats = benchmark(compiled, [MagicMock(name="arg")], rounds=rounds, warmup=warmup)
    return stats, worker, handle, compiled, ctor


# ---------------------------------------------------------------------------
# benchmark() — loop, warmup discard, aggregation
# ---------------------------------------------------------------------------


def test_benchmark_discards_warmup_and_collects_rounds():
    """Warmup launches run but are excluded; only ``rounds`` samples remain."""
    # 2 warmup (discarded) + 3 measured.
    timings = [_timing(99, 99), _timing(99, 99), _timing(10, 100), _timing(20, 200), _timing(30, 300)]
    stats, worker, handle, _compiled, _ctor = _run_benchmark(timings, rounds=3, warmup=2)

    assert handle.calls == 5  # warmup + rounds launches total
    assert worker.register_calls == 1  # registered exactly once
    assert stats.device_wall_us == [10.0, 20.0, 30.0]
    assert stats.host_wall_us == [100.0, 200.0, 300.0]
    assert stats.rounds == 3
    assert stats.warmup == 2


def test_benchmark_no_warmup():
    """``warmup=0`` keeps every launch."""
    timings = [_timing(5), _timing(15)]
    stats, _worker, handle, _compiled, _ctor = _run_benchmark(timings, rounds=2, warmup=0)

    assert handle.calls == 2
    assert stats.device_wall_us == [5.0, 15.0]


def test_benchmark_binds_worker_to_compiled_runtime():
    """ChipWorker is constructed with the compiled program's runtime name."""
    timings = [_timing(1)]
    _stats, _worker, _handle, compiled, ctor = _run_benchmark(timings, rounds=1, warmup=0)

    _args, kwargs = ctor.call_args
    assert kwargs["runtime"] == compiled.runtime_name


def test_benchmark_platform_device_id_build_runconfig():
    """``platform=`` / ``device_id=`` build the dispatch ``RunConfig``."""
    worker = _FakeWorker(_FakeHandle([_timing(1)]))
    with patch("pypto.runtime.bench.ChipWorker", return_value=worker) as ctor:
        benchmark(_compiled_mock(), [MagicMock(name="arg")], rounds=1, warmup=0, platform="a2a3", device_id=2)

    rc = ctor.call_args.args[0]  # ChipWorker(rc, runtime=...)
    assert rc.platform == "a2a3"
    assert rc.device_id == 2


def test_benchmark_default_platform_from_compiled():
    """Without ``platform=``/``config=``, the RunConfig binds to compiled.platform."""
    worker = _FakeWorker(_FakeHandle([_timing(1)]))
    compiled = _compiled_mock()
    with patch("pypto.runtime.bench.ChipWorker", return_value=worker) as ctor:
        benchmark(compiled, [MagicMock(name="arg")], rounds=1, warmup=0)

    assert ctor.call_args.args[0].platform == compiled.platform


def test_benchmark_config_with_kwargs_conflict_raises():
    """Passing ``config=`` together with ``platform``/``device_id`` is rejected."""
    with pytest.raises(ValueError, match="not both"):
        benchmark(_compiled_mock(), [], config=MagicMock(name="cfg"), platform="a2a3")
    with pytest.raises(ValueError, match="not both"):
        benchmark(_compiled_mock(), [], config=MagicMock(name="cfg"), device_id=1)


def test_benchmark_invalid_rounds_raises():
    with pytest.raises(ValueError, match="rounds must be positive"):
        benchmark(_compiled_mock(), [], rounds=0)


def test_benchmark_invalid_warmup_raises():
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        benchmark(_compiled_mock(), [], rounds=1, warmup=-1)


# ---------------------------------------------------------------------------
# BenchmarkStats — aggregation helpers
# ---------------------------------------------------------------------------


def test_benchmark_stats_aggregation():
    stats = BenchmarkStats(device_wall_us=[10.0, 20.0, 30.0, 40.0], rounds=4, warmup=1)
    assert stats.device_us_min == 10.0
    assert stats.device_us_max == 40.0
    assert stats.device_us_median == 25.0
    assert stats.device_us_mean == 25.0
    assert stats.device_us_stdev == pytest.approx(12.909944, rel=1e-5)
    assert not stats.all_zero_device


def test_benchmark_stats_empty():
    stats = BenchmarkStats()
    assert stats.device_us_min == 0.0
    assert stats.device_us_median == 0.0
    assert stats.device_us_stdev == 0.0
    assert not stats.all_zero_device  # no samples → not "all zero"
    assert "no samples" in str(stats)


def test_benchmark_stats_all_zero_device():
    """All-zero device samples flag a non-PTO2_PROFILING build."""
    stats = BenchmarkStats(device_wall_us=[0.0, 0.0, 0.0], host_wall_us=[5.0, 6.0, 7.0], rounds=3)
    assert stats.all_zero_device
    assert "PTO2_PROFILING" in str(stats)


def test_benchmark_stats_str_reports_metrics():
    stats = BenchmarkStats(device_wall_us=[10.0, 20.0], rounds=2, warmup=1)
    text = str(stats)
    assert "rounds=2" in text
    assert "median=" in text


def test_benchmark_stats_issue_aliases():
    """``device_wall_us_*`` / ``samples`` mirror the issue #1858 sketch names."""
    stats = BenchmarkStats(device_wall_us=[10.0, 20.0, 30.0], rounds=3)
    assert stats.samples is stats.device_wall_us
    assert stats.device_wall_us_min == stats.device_us_min == 10.0
    assert stats.device_wall_us_median == stats.device_us_median == 20.0
    assert stats.device_wall_us_mean == stats.device_us_mean
    assert stats.device_wall_us_max == stats.device_us_max == 30.0
    assert stats.device_wall_us_stdev == stats.device_us_stdev


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
