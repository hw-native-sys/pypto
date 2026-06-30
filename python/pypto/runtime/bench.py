# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register-once, multi-round on-device benchmark (issue #1858).

Mirrors simpler's ``scene_test --rounds`` mode through pypto's public Worker:
register the compiled program once, dispatch ``rounds`` cheap launches via
:meth:`pypto.runtime.RegistrationHandle.__call__`, and aggregate per-launch
``device_wall_us``. This avoids the one-shot ``execute_compiled`` /
``CompiledProgram.__call__`` path, which re-pays ``compile_and_assemble`` +
register/load every call (hundreds of ms of host overhead that swamps the
~1 ms device time).

Timing source (simpler PR #1177)
--------------------------------
``Worker.run`` no longer returns a ``RunTiming``. The host runtime instead
emits one ``[STRACE]`` marker line per stage to **stderr** on every launch
(``fprintf(stderr, ...)`` from the C++ host logger, gated by the compile-time
``SIMPLER_PROFILING`` macro and emitted at the ``LOG_INFO_V9`` tier). This
module therefore:

1. raises the simpler runtime log level to ``v9`` so the markers print (the
   C++ host logger is seeded from the Python logger snapshot at
   ``ChipWorker.init``), then restores the prior level afterward;
2. redirects ``stderr`` at the file-descriptor level (``os.dup2`` — Python's
   ``contextlib.redirect_stderr`` cannot capture the C++ writes) into a temp
   file around the measured loop;
3. parses the captured markers, reading each launch's on-NPU ``device_wall``
   and host ``run_prepared`` span.

Because the capture is fd-level, **all** stderr produced during the measured
loop is diverted into the temp file (not shown live). Warmup/teardown logging
outside the loop is unaffected.
"""

import os
import re
import statistics
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log_config import configure_log, current_level
from .runner import RunConfig
from .worker import ChipWorker

__all__ = ["BenchmarkStats", "benchmark"]

# ``[STRACE]`` marker grammar (simpler ``v=1`` wire format, emitted by the host
# logger — see ``src/common/log/include/common/strace.h`` and parsed by
# ``simpler_setup.tools.strace_timing``). Matched anywhere on the line so the
# CANN/host log prefix is ignored. Only the fields ``benchmark`` reads are
# captured; the parser is inlined (not imported from the simpler tool) so it
# stays self-contained and unit-testable without a simpler runtime.
_STRACE_RE = re.compile(
    r"\[STRACE\]\s+v=\d+\s+pid=(?P<pid>\d+)\s+tid=\d+\s+inv=(?P<inv>\d+)\s+"
    r"hid=(?P<hid>[0-9a-fA-F]+)\s+depth=\d+\s+name=(?P<name>\S+)\s+ts=\d+\s+dur=(?P<dur>\d+)"
)

# Span names read per launch (mirror ``strace_timing._ROUNDS_TABLE_NAMES``).
# ``host`` is the whole ``run_prepared`` wall; ``device`` is the on-NPU
# orchestrator wall.
_SPAN_HOST = "run_prepared"
_SPAN_DEVICE = "run_prepared.runner_run.device_wall"

# Runtime log level that makes the ``LOG_INFO_V9`` ``[STRACE]`` markers visible.
_STRACE_LOG_LEVEL = "v9"


@dataclass
class BenchmarkStats:
    """Aggregated per-launch timing from :func:`benchmark`.

    The min / median / mean / max / stdev helpers operate on
    ``device_wall_us`` — the on-NPU metric. ``host_wall_us`` samples are kept
    for context, but they include per-launch arg coercion + H2D and so are not
    the device metric.

    Attributes:
        device_wall_us: Per-measured-launch on-NPU orchestrator wall times
            (microseconds), read from each launch's ``[STRACE]``
            ``run_prepared.runner_run.device_wall`` span. Length is ``rounds``
            (warmup launches excluded).
        host_wall_us: Per-measured-launch host wall times (microseconds), read
            from each launch's ``[STRACE]`` ``run_prepared`` span.
        rounds: Number of measured launches.
        warmup: Number of leading launches discarded before measurement.
    """

    device_wall_us: list[float] = field(default_factory=list)
    host_wall_us: list[float] = field(default_factory=list)
    rounds: int = 0
    warmup: int = 0

    @property
    def device_us_min(self) -> float:
        return min(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_median(self) -> float:
        return statistics.median(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_mean(self) -> float:
        return statistics.fmean(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_max(self) -> float:
        return max(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_stdev(self) -> float:
        return statistics.stdev(self.device_wall_us) if len(self.device_wall_us) > 1 else 0.0

    # ``device_wall_us_*`` / ``samples`` are issue #1858-sketch-aligned aliases
    # of the ``device_us_*`` / ``device_wall_us`` accessors above.
    @property
    def samples(self) -> list[float]:
        """Alias for :attr:`device_wall_us` — the measured device-wall samples."""
        return self.device_wall_us

    @property
    def device_wall_us_min(self) -> float:
        return self.device_us_min

    @property
    def device_wall_us_median(self) -> float:
        return self.device_us_median

    @property
    def device_wall_us_mean(self) -> float:
        return self.device_us_mean

    @property
    def device_wall_us_max(self) -> float:
        return self.device_us_max

    @property
    def device_wall_us_stdev(self) -> float:
        return self.device_us_stdev

    @property
    def all_zero_device(self) -> bool:
        """``True`` if no real device wall was measured.

        Happens on a runtime built without ``SIMPLER_PROFILING`` or on a
        ``*sim`` platform, where the device-domain ``[STRACE]`` spans are not
        captured (``device_wall_us`` reads ``0``, not absent) — benchmark
        callers should then fall back to ``host_wall_us`` or rebuild with
        profiling enabled.
        """
        return bool(self.device_wall_us) and not any(self.device_wall_us)

    def __str__(self) -> str:
        if not self.device_wall_us:
            return f"BenchmarkStats(rounds={self.rounds}: no samples)"
        if self.all_zero_device:
            return (
                f"BenchmarkStats(rounds={self.rounds}): device_wall_us all 0 — runtime "
                f"built without SIMPLER_PROFILING or sim platform (use host_wall_us)"
            )
        return (
            f"BenchmarkStats(rounds={self.rounds}, warmup={self.warmup}): "
            f"device_wall_us min={self.device_us_min:.1f} median={self.device_us_median:.1f} "
            f"mean={self.device_us_mean:.1f} max={self.device_us_max:.1f} "
            f"stdev={self.device_us_stdev:.1f}"
        )


@contextmanager
def _capture_fd_stderr(path: Path) -> Iterator[None]:
    """Redirect the process ``stderr`` file descriptor into *path* for the block.

    The ``[STRACE]`` markers are written by the C++ host logger via
    ``fprintf(stderr, ...)``, so they bypass Python's ``sys.stderr`` /
    ``contextlib.redirect_stderr``. Capturing them needs an fd-level
    ``os.dup2`` swap of fd 2. The original fd is duplicated and restored on
    exit (including on exception) so later stderr is unaffected.
    """
    saved_fd = os.dup(2)
    flushed = False
    try:
        with open(path, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 2)
            try:
                yield
            finally:
                # Flush the C runtime's stderr buffer into the file before we
                # swap fd 2 back, or trailing markers can be lost.
                try:
                    os.fsync(sink.fileno())
                except OSError:
                    pass
                os.dup2(saved_fd, 2)
                flushed = True
    finally:
        if not flushed:
            os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _parse_stats_from_strace(log_text: str, *, rounds: int, warmup: int) -> BenchmarkStats:
    """Build a :class:`BenchmarkStats` from captured ``[STRACE]`` log text.

    Groups markers into per-launch invocations keyed by ``(pid, inv)``, buckets
    them by callable hash, takes the busiest bucket (our register-once callable
    emits one invocation per launch — warmup + measured all share one hash),
    orders by ``inv``, drops the first *warmup* invocations, and reads each
    remaining launch's host (``run_prepared``) and device
    (``run_prepared.runner_run.device_wall``) span duration in microseconds.

    Pure host logic — no device or worker needed — so it is unit-tested
    directly against synthetic marker lines.
    """
    stats = BenchmarkStats(rounds=rounds, warmup=warmup)

    # (pid, inv) -> {span name -> dur_ns}, plus the launch's hid and a stable
    # first-seen order. One pass, O(N) over marker lines.
    launches: dict[tuple[int, int], dict[str, int]] = {}
    launch_hid: dict[tuple[int, int], str] = {}
    order: list[tuple[int, int]] = []
    for line in log_text.splitlines():
        m = _STRACE_RE.search(line)
        if m is None:
            continue
        key = (int(m["pid"]), int(m["inv"]))
        if key not in launches:
            launches[key] = {}
            launch_hid[key] = m["hid"].lower()
            order.append(key)
        launches[key].setdefault(m["name"], int(m["dur"]))

    if not launches:
        return stats

    # Busiest hid bucket = our register-once callable. Order keys within it by
    # inv (the process-wide monotonic launch id) so warmup is dropped in
    # dispatch order regardless of log interleaving.
    by_hid: dict[str, list[tuple[int, int]]] = {}
    for key in order:
        by_hid.setdefault(launch_hid[key], []).append(key)
    busiest = sorted(max(by_hid.values(), key=len), key=lambda k: k[1])

    for key in busiest[warmup:]:
        spans = launches[key]
        host_ns = spans.get(_SPAN_HOST)
        device_ns = spans.get(_SPAN_DEVICE)
        stats.host_wall_us.append(host_ns / 1000.0 if host_ns is not None else 0.0)
        stats.device_wall_us.append(device_ns / 1000.0 if device_ns is not None else 0.0)

    return stats


def benchmark(
    compiled: Any,
    args: Sequence[Any],
    *,
    rounds: int = 100,
    warmup: int = 3,
    platform: str | None = None,
    device_id: int | None = None,
    config: RunConfig | None = None,
) -> BenchmarkStats:
    """Register *compiled* once and dispatch *rounds* timed launches.

    Opens a single :class:`~pypto.runtime.ChipWorker`, registers *compiled*
    once, then loops the bound handle so each launch only re-pays argument
    coercion + dispatch (not register/load). The on-NPU ``device_wall_us`` is
    measured between the orchestrator's ``orch_start`` / ``orch_end`` and is
    unaffected by the per-launch host-side arg building.

    Timing is read from the runtime's ``[STRACE]`` stderr markers (simpler PR
    #1177): this raises the runtime log level to ``v9`` for the worker's
    lifetime (restored afterward) and captures ``stderr`` at the
    file-descriptor level around the measured loop, so any stderr emitted
    during the loop is diverted into a temp file rather than shown live.

    Args:
        compiled: A single-orchestration
            :class:`~pypto.ir.CompiledProgram` from ``ir.compile`` /
            ``compile_program``. Multi-orch programs must pass
            ``compiled[<name>]``.
        args: Positional dispatch args, same as ``compiled(*args)``.
        rounds: Number of measured launches. Must be positive.
        warmup: Number of leading launches discarded before measurement
            (page-in / cache warm). Total launches = ``warmup + rounds``.
        platform: Target platform shorthand, e.g. ``"a2a3"``. Defaults to
            ``compiled.platform``. Mutually exclusive with *config*.
        device_id: NPU device index. Defaults to ``RunConfig``'s default.
            Mutually exclusive with *config*.
        config: Optional :class:`~pypto.runtime.RunConfig` for full control
            (``block_dim`` / ``aicpu_thread_num`` / ``pto_isa_commit``). Pass
            this *or* *platform*/*device_id*, not both.

    Returns:
        A :class:`BenchmarkStats` with the per-launch ``device_wall_us`` /
        ``host_wall_us`` samples and aggregate helpers.

    Raises:
        ValueError: ``rounds <= 0``, ``warmup < 0``, or *config* passed
            together with *platform* / *device_id*.

    Note:
        Only L2 single-chip runs carry a real ``device_wall_us``. On a runtime
        built without ``SIMPLER_PROFILING`` or on a ``*sim`` platform every
        sample is ``0`` — check :attr:`BenchmarkStats.all_zero_device`.
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if config is not None and (platform is not None or device_id is not None):
        raise ValueError("benchmark(): pass either config=... or platform=/device_id=, not both")

    if config is not None:
        rc = config
    else:
        rc_kwargs: dict[str, Any] = {"platform": platform or compiled.platform}
        if device_id is not None:
            rc_kwargs["device_id"] = device_id
        rc = RunConfig(**rc_kwargs)

    # The C++ host logger that prints the ``[STRACE]`` markers is seeded from
    # the simpler Python logger snapshot at ``ChipWorker.init``, so raise the
    # level before constructing the worker. Restore it afterward.
    prior_level = current_level()
    configure_log(_STRACE_LOG_LEVEL)
    try:
        with ChipWorker(rc, runtime=compiled.runtime_name) as worker:
            handle = worker.register(compiled)  # register once; cid cached
            with tempfile.TemporaryDirectory(prefix="pypto-bench-") as tmp:
                log_path = Path(tmp) / "strace.log"
                with _capture_fd_stderr(log_path):
                    for _ in range(warmup):  # warm caches / page-in; markers discarded
                        handle(*args, config=rc)
                    for _ in range(rounds):  # measured launches
                        handle(*args, config=rc)
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        configure_log(prior_level)

    return _parse_stats_from_strace(log_text, rounds=rounds, warmup=warmup)
