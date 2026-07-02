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
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log_config import configure_log, current_level
from .runner import RunConfig
from .worker import ChipWorker

__all__ = ["BenchmarkStats", "TraceInvocation", "TraceSpan", "benchmark"]

# ``[STRACE]`` marker parsing is delegated to simpler's ``strace_timing`` (the
# single source of truth for the ``v=1`` wire grammar). Its ``Span`` /
# ``Invocation`` types are mirrored into the pypto-owned ``TraceSpan`` /
# ``TraceInvocation`` below so ``benchmark`` callers get the full per-launch
# span tree without importing simpler types (see ``_parse_stats_from_strace``).


@dataclass
class TraceSpan:
    """One ``[STRACE]`` span — a node in a measured launch's call tree.

    A pypto-owned mirror of simpler's ``strace_timing.Span`` so ``benchmark``
    callers never depend on simpler types.

    Attributes:
        depth: Nesting level; a span at depth ``d`` is a child of the nearest
            enclosing span at depth ``d-1``.
        name: Dotted span path (e.g. ``run_prepared.runner_run.device_wall``).
        ts: Start timestamp in nanoseconds (host clock, or device clock when
            :attr:`is_device`).
        dur: Span duration in nanoseconds.
        attrs: Raw trailing attribute string (carries ``clk=dev`` for device
            spans).
    """

    pid: int
    tid: int
    inv: int
    hid: str
    depth: int
    name: str
    ts: int
    dur: int
    attrs: str

    @property
    def is_device(self) -> bool:
        """``True`` for device-domain spans (emitted with ``clk=dev``)."""
        return "clk=dev" in self.attrs

    @property
    def dur_us(self) -> float:
        """Span duration in microseconds."""
        return self.dur / 1000.0


@dataclass
class TraceInvocation:
    """Every ``[STRACE]`` span emitted by one measured launch.

    One ``(pid, inv)`` group, in emission (scope-exit) order. Use
    :meth:`format_tree` to render the nested call tree.
    """

    pid: int
    inv: int
    hid: str
    spans: list[TraceSpan] = field(default_factory=list)

    def root(self) -> "TraceSpan | None":
        """The depth-0 span (``run_prepared``), or ``None`` if absent."""
        for s in self.spans:
            if s.depth == 0:
                return s
        return None

    def by_name(self) -> dict[str, "TraceSpan"]:
        """Map span name → its first-seen span."""
        m: dict[str, TraceSpan] = {}
        for s in self.spans:
            m.setdefault(s.name, s)
        return m

    def format_tree(
        self, *, us: bool = True, value_fn: "Callable[[TraceSpan], list[str]] | None" = None
    ) -> str:
        """Render this launch's span tree with ``|-`` / `` `- `` branch connectors.

        Hierarchy is drawn with ASCII connectors (``|- `` for a non-last child,
        `` `- `` for the last, ``|  `` / ``   `` for continuation) rather than by
        indentation alone. Nesting is reconstructed from the dotted span names (a
        span's parent is the span whose name is its longest proper dotted
        prefix), which is robust to the host/device clock-domain split —
        device-domain spans (``run_prepared.runner_run.device_wall.*``) correctly
        nest under their host parent even though they are emitted as a separate
        batch. Siblings are ordered by start timestamp; device-domain spans are
        tagged ``[dev]``.

        Output is column-aligned: the name column (connectors + leaf + tag) is
        left-padded to a common width and the value columns are right-aligned, so
        the numbers line up regardless of nesting depth. ``value_fn`` returns the
        value column(s) per span (default: a single duration column, microseconds
        when ``us`` else nanoseconds); :meth:`BenchmarkStats.format_mean_tree`
        uses it to add aligned ``±stdev`` / ``[min..max]`` columns.
        """
        by_name: dict[str, TraceSpan] = {}
        for s in self.spans:
            by_name.setdefault(s.name, s)

        def _parent(name: str) -> "str | None":
            parts = name.split(".")
            for cut in range(len(parts) - 1, 0, -1):
                cand = ".".join(parts[:cut])
                if cand in by_name:
                    return cand
            return None

        children: dict[str, list[str]] = defaultdict(list)
        roots: list[str] = []
        for name in by_name:
            parent = _parent(name)
            (children[parent] if parent is not None else roots).append(name)

        def _by_ts(names: list[str]) -> list[str]:
            return sorted(names, key=lambda n: by_name[n].ts)

        def _columns(name: str) -> list[str]:
            span = by_name[name]
            if value_fn is not None:
                return value_fn(span)
            return [f"{span.dur / 1000.0:.1f}us" if us else f"{span.dur}ns"]

        # First pass: collect (name column, value columns) in display order.
        rows: list[tuple[str, list[str]]] = []

        def _walk(name: str, prefix: str, child_prefix: str) -> None:
            parent = _parent(name)
            leaf = name[len(parent) + 1 :] if parent is not None else name
            tag = " [dev]" if by_name[name].is_device else ""
            rows.append((f"{prefix}{leaf}{tag}", _columns(name)))
            kids = _by_ts(children[name])
            for i, kid in enumerate(kids):
                last = i == len(kids) - 1
                _walk(
                    kid,
                    child_prefix + ("`- " if last else "|- "),
                    child_prefix + ("   " if last else "|  "),
                )

        for r in _by_ts(roots):
            _walk(r, "", "")

        # Second pass: left-align the name column, right-align each value column.
        name_w = max((len(label) for label, _ in rows), default=0)
        ncols = max((len(cols) for _, cols in rows), default=0)
        col_w = [0] * ncols
        for _, cols in rows:
            for i, c in enumerate(cols):
                col_w[i] = max(col_w[i], len(c))

        lines: list[str] = []
        for label, cols in rows:
            line = label.ljust(name_w)
            for i, c in enumerate(cols):
                line += "  " + c.rjust(col_w[i])
            lines.append(line.rstrip())
        return "\n".join(lines)


# Per-launch spans are read root-agnostically: ``host`` is the whole dispatch
# wall (the depth-0 root span) and ``device`` is the on-NPU orchestrator wall
# (``<root>.runner_run.device_wall``). The root span name is the runtime's
# dispatch entrypoint, which simpler #1210 renamed ``run_prepared`` ->
# ``simpler_run``; matching by depth (host) and by dotted suffix (device) reads
# timing from both the pre- and post-#1210 STRACE vocabularies.
_DEVICE_WALL_SUFFIX = ".runner_run.device_wall"

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
        invocations: Full per-measured-launch span tree (one
            :class:`TraceInvocation` per measured launch, warmup excluded).
            Empty when no ``[STRACE]`` markers were captured. Render with
            :meth:`format_tree` / :meth:`print_tree`.
    """

    device_wall_us: list[float] = field(default_factory=list)
    host_wall_us: list[float] = field(default_factory=list)
    rounds: int = 0
    warmup: int = 0
    invocations: list[TraceInvocation] = field(default_factory=list)

    def format_tree(self, launch: int | None = None, *, us: bool = True) -> str:
        """Render the captured ``[STRACE]`` span tree(s) as indented text.

        Args:
            launch: Measured-launch index to render; ``None`` (default) renders
                every measured launch.
            us: Show durations in microseconds (default) or nanoseconds.
        """
        if not self.invocations:
            return "BenchmarkStats: no span tree captured (non-SIMPLER_PROFILING build or *sim platform)"
        selected = (
            list(enumerate(self.invocations)) if launch is None else [(launch, self.invocations[launch])]
        )
        out: list[str] = []
        for i, inv in selected:
            out.append(f"launch[{i}] (pid={inv.pid} inv={inv.inv} hid={inv.hid}):")
            out.append(inv.format_tree(us=us))
        return "\n".join(out)

    def print_tree(self, launch: int | None = None, *, us: bool = True, file: Any = None) -> None:
        """Print :meth:`format_tree` to *file* (default stdout)."""
        print(self.format_tree(launch, us=us), file=file)

    def mean_invocation(self) -> "TraceInvocation | None":
        """A synthetic :class:`TraceInvocation` whose every span's ``dur`` (and
        ``ts``) is the mean across all measured launches (warmup excluded).

        Spans are matched by name; ``depth`` / ``attrs`` (hence
        :attr:`TraceSpan.is_device`) come from the first launch that carried the
        span. ``inv`` is ``-1`` to mark the aggregate. Returns ``None`` when no
        span tree was captured. Useful for rendering one noise-smoothed tree.
        """
        if not self.invocations:
            return None
        durs: dict[str, list[int]] = defaultdict(list)
        tss: dict[str, list[int]] = defaultdict(list)
        template: dict[str, TraceSpan] = {}
        for inv in self.invocations:
            for s in inv.spans:
                durs[s.name].append(s.dur)
                tss[s.name].append(s.ts)
                template.setdefault(s.name, s)
        spans = [
            TraceSpan(
                pid=t.pid,
                tid=t.tid,
                inv=-1,
                hid=t.hid,
                depth=t.depth,
                name=name,
                ts=round(statistics.fmean(tss[name])),
                dur=round(statistics.fmean(durs[name])),
                attrs=t.attrs,
            )
            for name, t in template.items()
        ]
        first = self.invocations[0]
        return TraceInvocation(pid=first.pid, inv=-1, hid=first.hid, spans=spans)

    def format_mean_tree(self, *, us: bool = True, spread: str = "stdev") -> str:
        """Render a span tree whose every node's duration is the mean across all
        measured launches (warmup excluded), annotated with the per-node spread.

        Args:
            us: Show values in microseconds (default) or nanoseconds.
            spread: Spread shown after each node's mean — ``"stdev"`` (``±sd``,
                default), ``"minmax"`` (``[min..max]``), ``"both"``, or
                ``"none"``. Computed across the measured launches.
        """
        mean_inv = self.mean_invocation()
        if mean_inv is None:
            return "BenchmarkStats: no span tree captured (non-SIMPLER_PROFILING build or *sim platform)"

        durs: dict[str, list[int]] = defaultdict(list)
        for inv in self.invocations:
            for s in inv.spans:
                durs[s.name].append(s.dur)
        scale = 1000.0 if us else 1.0
        unit = "us" if us else "ns"

        def _value(span: TraceSpan) -> list[str]:
            ds = durs[span.name]
            cols = [f"{statistics.fmean(ds) / scale:.1f}{unit}"]
            if spread in ("stdev", "both"):
                sd = statistics.stdev(ds) / scale if len(ds) > 1 else 0.0
                cols.append(f"±{sd:.1f}")
            if spread in ("minmax", "both"):
                cols.append(f"[{min(ds) / scale:.1f}..{max(ds) / scale:.1f}]")
            return cols

        legend = "mean"
        if spread in ("stdev", "both"):
            legend += " ±stdev"
        if spread in ("minmax", "both"):
            legend += " [min..max]"
        header = (
            f"mean of {len(self.invocations)} launches (warmup {self.warmup} excluded); each node: {legend}:"
        )
        return f"{header}\n{mean_inv.format_tree(us=us, value_fn=_value)}"

    def print_mean_tree(self, *, us: bool = True, spread: str = "stdev", file: Any = None) -> None:
        """Print :meth:`format_mean_tree` to *file* (default stdout)."""
        print(self.format_mean_tree(us=us, spread=spread), file=file)

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

    Parsing is delegated to simpler's ``strace_timing`` — the single source of
    truth for the marker grammar — then each launch's full span tree is mirrored
    into pypto-owned :class:`TraceInvocation` / :class:`TraceSpan` so callers
    never import simpler types. Groups markers by ``(pid, inv)``, buckets by
    callable hash, takes the busiest bucket (our register-once callable emits one
    invocation per launch), orders by ``inv``, drops the first *warmup*
    invocations, and reads each remaining launch's host (the depth-0 root span)
    and device (``<root>.runner_run.device_wall``) span durations (µs). The root
    span name is matched by depth/suffix rather than a literal so both the pre-
    and post-#1210 (``run_prepared`` -> ``simpler_run``) vocabularies work.
    """
    # ``simpler`` is an optional runtime-provided package: present on devices
    # where the runtime is installed, absent on the lint / unit-test host. The
    # import is resolved lazily at call time; pyright cannot see it in the lint
    # env, and unit tests skip the parse path when it is not installed.
    from simpler_setup.tools.strace_timing import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        bucket_by_hid,
        group_invocations,
        parse_spans,
    )

    stats = BenchmarkStats(rounds=rounds, warmup=warmup)
    invocations = group_invocations(parse_spans(log_text.splitlines()))
    if not invocations:
        return stats

    # Busiest hid bucket = our register-once callable (one invocation per launch);
    # bucket_by_hid orders each bucket by inv, so warmup drops in dispatch order.
    busiest = max(bucket_by_hid(invocations).values(), key=len)
    for inv in busiest[warmup:]:
        host = next((s for s in inv.spans if s.depth == 0), None)
        device = next((s for s in inv.spans if s.name.endswith(_DEVICE_WALL_SUFFIX)), None)
        stats.host_wall_us.append(host.dur / 1000.0 if host is not None else 0.0)
        stats.device_wall_us.append(device.dur / 1000.0 if device is not None else 0.0)
        stats.invocations.append(
            TraceInvocation(
                pid=inv.pid,
                inv=inv.inv,
                hid=inv.hid,
                spans=[
                    TraceSpan(
                        pid=s.pid,
                        tid=s.tid,
                        inv=s.inv,
                        hid=s.hid,
                        depth=s.depth,
                        name=s.name,
                        ts=s.ts,
                        dur=s.dur,
                        attrs=s.attrs,
                    )
                    for s in inv.spans
                ],
            )
        )

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
        RuntimeError: No ``[STRACE]`` markers were captured at all, so no timing
            could be read. The markers are gated by the runtime's compile-time
            ``SIMPLER_PROFILING`` macro; a runtime built without it emits none.

    Note:
        Only L2 single-chip runs carry a real ``device_wall_us``. On a ``*sim``
        platform the host ``run_prepared`` span is still emitted but the
        device-domain spans are not, so every ``device_wall_us`` sample is ``0``
        — check :attr:`BenchmarkStats.all_zero_device`.
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

    stats = _parse_stats_from_strace(log_text, rounds=rounds, warmup=warmup)
    # We dispatched warmup + rounds launches, so a marker-emitting runtime always
    # yields at least one host span. Zero markers means the runtime emitted none
    # (built without SIMPLER_PROFILING) — surface that rather than returning a
    # silently-empty result a caller could misread as "0 device timing".
    if not stats.host_wall_us:
        raise RuntimeError(
            f"benchmark(): no [STRACE] markers captured across {warmup + rounds} launches. "
            "The runtime emits per-launch timing markers only when built with the "
            "SIMPLER_PROFILING macro (LOG_INFO_V9 tier); this runtime emitted none. "
            "Rebuild the runtime with SIMPLER_PROFILING enabled to read benchmark timing."
        )
    return stats
