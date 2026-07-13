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
``SIMPLER_HOST_STRACE`` macro and emitted at the ``LOG_INFO_V9`` tier). This
module therefore:

1. raises the simpler runtime log level to ``v9`` so the markers print (the
   C++ host logger is seeded from the Python logger snapshot at
   ``ChipWorker.init``), then restores the prior level afterward;
2. redirects ``stderr`` at the file-descriptor level (``os.dup2`` — Python's
   ``contextlib.redirect_stderr`` cannot capture the C++ writes) into a temp
   file around the measured region (for L3, also around ``prepare()`` so the
   forked chip-worker processes inherit the redirected fd);
3. parses the captured markers, reading each launch's on-NPU ``device_wall``
   and host ``simpler_run`` span.

Because the capture is fd-level, **all** stderr produced during the measured
loop is diverted into the temp file (not shown live). Warmup/teardown logging
outside the loop is unaffected.
"""

import functools
import os
import statistics
import sys
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
        name: Dotted span path (e.g. ``simpler_run.runner_run.device_wall``).
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
        """The depth-0 span (``simpler_run``), or ``None`` if absent."""
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

    @property
    def task(self) -> str:
        """Task identity for this dispatch — the callable-hash ``hid``.

        This is the only per-callable identifier the ``[STRACE]`` markers carry;
        distinct ``hid`` values are distinct kernels/callables. It is an opaque
        hash, **not** a human-readable kernel name (markers do not carry one).
        """
        return self.hid

    @property
    def device_wall_us(self) -> float:
        """On-NPU ``<root>.runner_run.device_wall`` duration (µs), 0 if absent."""
        span = self.by_name().get(_span_names()["device"])
        return span.dur_us if span is not None else 0.0

    @property
    def host_wall_us(self) -> float:
        """Host ``<root>`` (whole-run) duration (µs), 0 if absent."""
        span = self.by_name().get(_span_names()["host"])
        return span.dur_us if span is not None else 0.0

    @property
    def effective_us(self) -> float:
        """L2 **Effective** window (µs): the orch/sched merged on-device window.

        ``max(orch_end, sched_end) - min(orch_start, sched_start)`` over this
        dispatch's device-domain ``orch`` / ``sched`` spans — the runtime's
        "Effective" metric (the old device-log "Total") for this single L2 run.
        Both spans share this invocation's device-clock origin, so the union is
        valid within the dispatch. Returns ``0.0`` when neither span is present
        (``*sim`` / non-profiling build).
        """
        by = self.by_name()
        names = _span_names()
        spans = [s for s in (by.get(names["orch"]), by.get(names["sched"])) if s is not None]
        if not spans:
            return 0.0
        return (max(s.ts + s.dur for s in spans) - min(s.ts for s in spans)) / 1000.0

    def format_tree(
        self, *, us: bool = True, value_fn: "Callable[[TraceSpan], list[str]] | None" = None
    ) -> str:
        """Render this launch's span tree with ``|-`` / `` `- `` branch connectors.

        Hierarchy is drawn with ASCII connectors (``|- `` for a non-last child,
        `` `- `` for the last, ``|  `` / ``   `` for continuation) rather than by
        indentation alone. Nesting is reconstructed from the dotted span names (a
        span's parent is the span whose name is its longest proper dotted
        prefix), which is robust to the host/device clock-domain split —
        device-domain spans (``simpler_run.runner_run.device_wall.*``) correctly
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


# Per-launch ``[STRACE]`` span names. ``host`` is the whole run wall; ``device``
# is the on-NPU orchestrator wall; ``orch`` / ``sched`` subdivide it (their union
# is the "Effective" on-device execution window). The span root was renamed
# ``run_prepared`` -> ``simpler_run`` in simpler #1210, so the names are sourced
# at call time from the installed runtime's ``strace_timing._ROUNDS_TABLE_NAMES``
# via :func:`_span_names` rather than hardcoded — this keeps ``benchmark`` working
# against both runtime generations. These legacy names are the pre-#1210 fallback.
_LEGACY_SPAN_NAMES = {
    "host": "run_prepared",
    "device": "run_prepared.runner_run.device_wall",
    "orch": "run_prepared.runner_run.device_wall.orch",
    "sched": "run_prepared.runner_run.device_wall.sched",
}


@functools.lru_cache(maxsize=1)
def _span_names() -> dict[str, str]:
    """Resolve the four ``[STRACE]`` span names from the installed runtime.

    Reads ``strace_timing._ROUNDS_TABLE_NAMES`` (added in simpler #1210 alongside
    the ``run_prepared`` -> ``simpler_run`` root rename). ``_ROUNDS_TABLE_NAMES``
    is a private symbol absent from pre-#1210 simpler, so fall back to the legacy
    hardcoded names when it (or one of its keys) is missing.
    """
    try:
        from simpler_setup.tools.strace_timing import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            _ROUNDS_TABLE_NAMES,
        )

        return {key: _ROUNDS_TABLE_NAMES[key] for key in _LEGACY_SPAN_NAMES}
    except (ImportError, AttributeError, TypeError, KeyError):
        return dict(_LEGACY_SPAN_NAMES)


# Runtime log level that makes the ``LOG_INFO_V9`` ``[STRACE]`` markers visible.
_STRACE_LOG_LEVEL = "v9"

# Metric name → the per-dispatch :class:`TraceInvocation` attribute it sums over
# (used by ``BenchmarkStats.per_rank`` / ``per_round``).
_METRIC_ATTR = {"device": "device_wall_us", "host": "host_wall_us", "effective": "effective_us"}


@dataclass
class BenchmarkStats:
    """Aggregated per-launch timing from :func:`benchmark`.

    The min / median / mean / max / stdev helpers operate on
    ``device_wall_us`` — the on-NPU metric. ``host_wall_us`` samples are kept
    for context, but they include per-launch arg coercion + H2D and so are not
    the device metric.

    The data is organized into three tiers; per-metric summaries are derived on
    demand via :meth:`per_round` / :meth:`per_rank` rather than stored as many
    parallel fields:

    - **Stored headline** (list, length ``rounds``): :attr:`device_wall_us`,
      :attr:`host_wall_us`. Same as ``per_round("device")`` / ``per_round("host")``.
    - **Stored raw detail**: :attr:`rounds_dispatches` (the L3 ``round → rank →
      [dispatch]`` grid, the single source for per-rank / effective / union
      derivations) and :attr:`invocations` (the same dispatches flattened, for
      tree rendering).
    - **Derived summaries**: :meth:`per_round` / :meth:`per_rank` for the four
      metrics ``device`` / ``host`` / ``effective`` / ``union``.

    The min / median / mean / max / stdev helpers operate on
    :attr:`device_wall_us` — the on-NPU metric.

    Attributes:
        device_wall_us: Per-round on-NPU device wall (µs). L2: one per measured
            launch (the ``<root>.runner_run.device_wall`` span). L3: per-round max
            across ranks of each rank's summed dispatch device walls. Length is
            ``rounds`` (warmup excluded); in the L3 flatten fallback it is instead
            the pooled per-dispatch samples. Equals ``per_round("device")``.
        host_wall_us: Per-round host wall (µs), analogous over the ``<root>``
            span. Equals ``per_round("host")``.
        rounds: Number of measured launches.
        warmup: Number of leading launches discarded before measurement.
        invocations: All measured dispatches' span trees, flattened. L2: one per
            launch; L3: every rank's per-dispatch invocation. Empty when no
            ``[STRACE]`` markers were captured. Render with :meth:`format_tree`.
        rounds_dispatches: L3 only — the navigable ``round → rank → [dispatch]``
            grid: ``rounds_dispatches[k][pid]`` is that rank's
            :class:`TraceInvocation` dispatches in round ``k`` (ordered by
            ``inv``). Each dispatch exposes :attr:`TraceInvocation.task` (the
            ``hid``) and :attr:`TraceInvocation.device_wall_us` / ``host_wall_us``
            / ``effective_us``. It is the single source :meth:`per_rank` /
            :meth:`per_round` derive from. Same objects as :attr:`invocations`.
            Length is ``rounds``; empty for L2 and the flatten fallback.
        fallback_flattened: L3 only — ``True`` when per-round segmentation was not
            possible (a rank's marker count was not divisible by
            ``warmup + rounds``, i.e. a non-deterministic dispatch shape). Then
            :attr:`device_wall_us` / :attr:`host_wall_us` hold the pooled
            per-dispatch samples (warmup dropped per rank), :attr:`rounds_dispatches`
            is empty, and ``per_rank`` / ``per_round("union")`` return empty.
    """

    device_wall_us: list[float] = field(default_factory=list)
    host_wall_us: list[float] = field(default_factory=list)
    rounds: int = 0
    warmup: int = 0
    invocations: list[TraceInvocation] = field(default_factory=list)
    rounds_dispatches: list[dict[int, list[TraceInvocation]]] = field(default_factory=list)
    fallback_flattened: bool = False

    def per_rank(self, metric: str = "device") -> dict[int, list[float]]:
        """Per-rank, per-round summary (µs): ``{pid: [round0, round1, ...]}``.

        *metric* is one of ``"device"`` / ``"host"`` / ``"effective"`` — each
        round entry sums that rank's dispatches'
        :attr:`TraceInvocation.device_wall_us` / ``host_wall_us`` / ``effective_us``
        (a card runs its dispatches serially). Derived from
        :attr:`rounds_dispatches`, so it is **L3 only** — returns ``{}`` for L2
        and for the flatten fallback.
        """
        attr = _METRIC_ATTR.get(metric)
        if attr is None:
            raise ValueError(f"per_rank(): metric must be one of {sorted(_METRIC_ATTR)}, got {metric!r}")
        n = len(self.rounds_dispatches)
        out: dict[int, list[float]] = {}
        for k, ranks in enumerate(self.rounds_dispatches):
            for pid, dispatches in ranks.items():
                out.setdefault(pid, [0.0] * n)[k] = sum(getattr(d, attr) for d in dispatches)
        return out

    def per_round(self, metric: str = "device") -> list[float]:
        """Per-round summary (µs), one entry per measured round.

        *metric* is one of:

        - ``"device"`` / ``"host"`` — the stored headline (L3: max across ranks).
        - ``"effective"`` — L3: per-round max across ranks of each rank's summed
          Effective (orch/sched) window; L2 / fallback: each dispatch's
          :attr:`TraceInvocation.effective_us` in order.
        - ``"union"`` — L3 only: per-round cross-rank **host-timeline** union
          window ``max(host-span end) - min(host-span start)`` across all
          ranks' dispatches (host clocks are ``CLOCK_MONOTONIC``, cross-process
          comparable, so this captures overlap / start skew — but includes host
          dispatch overhead). ``[]`` for L2 and the flatten fallback.
        """
        if metric == "device":
            return list(self.device_wall_us)
        if metric == "host":
            return list(self.host_wall_us)
        if metric == "effective":
            ranks = self.per_rank("effective")
            if ranks:  # L3: slowest rank bounds the round
                return [max(v[k] for v in ranks.values()) for k in range(len(self.rounds_dispatches))]
            return [iv.effective_us for iv in self.invocations]  # L2 / fallback
        if metric == "union":
            return self._union_per_round()
        raise ValueError(f"per_round(): unknown metric {metric!r}")

    def _union_per_round(self) -> list[float]:
        """L3 cross-rank host-timeline union window per round (µs); ``[]`` otherwise."""
        out: list[float] = []
        for ranks in self.rounds_dispatches:
            starts: list[int] = []
            ends: list[int] = []
            for dispatches in ranks.values():
                for d in dispatches:
                    span = d.by_name().get(_span_names()["host"])
                    if span is not None:
                        starts.append(span.ts)
                        ends.append(span.ts + span.dur)
            out.append((max(ends) - min(starts)) / 1000.0 if starts else 0.0)
        return out

    def format_tree(self, launch: int | None = None, *, us: bool = True) -> str:
        """Render the captured ``[STRACE]`` span tree(s) as indented text.

        Args:
            launch: Measured-launch index to render; ``None`` (default) renders
                every measured launch.
            us: Show durations in microseconds (default) or nanoseconds.
        """
        if not self.invocations:
            return "BenchmarkStats: no span tree captured (non-SIMPLER_HOST_STRACE build or *sim platform)"
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
            return "BenchmarkStats: no span tree captured (non-SIMPLER_HOST_STRACE build or *sim platform)"

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

        Happens on a runtime built without ``SIMPLER_HOST_STRACE`` or on a
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
                f"built without SIMPLER_HOST_STRACE or sim platform (use host_wall_us)"
            )
        suffix = ""
        if self.rounds_dispatches:
            n_ranks = len({pid for ranks in self.rounds_dispatches for pid in ranks})
            suffix = f" [L3: {n_ranks} ranks, per-round max across ranks"
            union = self.per_round("union")
            if union:
                suffix += f"; host-union mean={statistics.fmean(union):.1f}us"
            suffix += "]"
        elif self.fallback_flattened:
            suffix = " [L3: flattened per-dispatch pool — non-deterministic dispatch shape]"
        return (
            f"BenchmarkStats(rounds={self.rounds}, warmup={self.warmup}): "
            f"device_wall_us min={self.device_us_min:.1f} median={self.device_us_median:.1f} "
            f"mean={self.device_us_mean:.1f} max={self.device_us_max:.1f} "
            f"stdev={self.device_us_stdev:.1f}{suffix}"
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


def _mirror_invocation(inv: Any) -> TraceInvocation:
    """Mirror a simpler ``strace_timing.Invocation`` into a pypto TraceInvocation.

    Copies the full span tree into pypto-owned :class:`TraceInvocation` /
    :class:`TraceSpan` so ``benchmark`` callers never depend on simpler types.
    """
    return TraceInvocation(
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


def _inv_span_us(inv: Any, name: str) -> float:
    """Duration (µs) of the first span named *name* in *inv*, or ``0.0`` if absent."""
    span = inv.by_name().get(name)
    return span.dur / 1000.0 if span is not None else 0.0


def _parse_l3_stats(invocations: Any, stats: BenchmarkStats, *, rounds: int, warmup: int) -> BenchmarkStats:
    """Aggregate L3 (distributed) ``[STRACE]`` markers into per-round stats.

    An L3 host-orch launch dispatches to one or more forked chip processes (one
    pid per rank); the host-orch parent emits no DAG-level ``device_wall``, so
    timing is recovered from the per-rank chip-child markers. Markers carry no
    round tag, but for a deterministic replay each rank emits a **constant**
    number of dispatches per launch, so its ``inv``-ordered stream splits into
    ``warmup + rounds`` equal chunks; chunk *k* (after dropping warmup) is round
    *k*. Segmentation is by count only — independent of ``hid`` — so repeated and
    heterogeneous dispatches to one card are handled.

    Per round, a rank's busy time is the **sum** of its dispatch spans (a card
    runs its dispatches serially); the headline (:attr:`BenchmarkStats.device_wall_us`)
    is the **max across ranks** (the round ends when the slowest rank finishes).
    The per-round-per-rank grid is stored in :attr:`BenchmarkStats.rounds_dispatches`;
    per-rank / effective / union summaries are derived from it on demand via
    :meth:`BenchmarkStats.per_rank` / :meth:`per_round`.

    Falls back to a flattened per-dispatch pool (best-effort per-rank warmup drop)
    and sets :attr:`BenchmarkStats.fallback_flattened` when a rank's marker count
    is not a positive multiple of ``warmup + rounds`` — a non-deterministic
    dispatch shape where per-round alignment cannot be trusted.
    """
    launches = warmup + rounds
    by_pid: dict[int, list[Any]] = defaultdict(list)
    for inv in invocations:
        by_pid[inv.pid].append(inv)
    for invs in by_pid.values():
        invs.sort(key=lambda i: i.inv)

    # Keep only chip-child ranks. A real rank emits a ``device_wall`` span; the
    # L3 host-orch parent process emits its own ``run_prepared`` root without any
    # chip ``device_wall``, so its pid must not be grouped as a rank — otherwise
    # it adds a fake zero-device rank that corrupts ``per_rank`` / rank counts and
    # pollutes the ``host_wall`` / ``union`` windows with the parent orch span.
    device_name = _span_names()["device"]
    by_pid = {
        pid: invs
        for pid, invs in by_pid.items()
        if any(inv.by_name().get(device_name) is not None for inv in invs)
    }
    if not by_pid:
        return stats

    segmentable = launches > 0 and all(invs and len(invs) % launches == 0 for invs in by_pid.values())

    if not segmentable:
        # Non-deterministic dispatch shape: don't guess round boundaries. Pool
        # every rank's per-dispatch samples, dropping `warmup` leading dispatches
        # per rank as a best effort.
        stats.fallback_flattened = True
        for invs in by_pid.values():
            names = _span_names()
            for inv in invs[min(warmup, len(invs)) :]:
                stats.host_wall_us.append(_inv_span_us(inv, names["host"]))
                stats.device_wall_us.append(_inv_span_us(inv, names["device"]))
                stats.invocations.append(_mirror_invocation(inv))
        return stats

    # Segment each rank's inv-ordered stream into per-round chunks (drop warmup),
    # mirror each dispatch once, and index it under round → rank. This grid is the
    # single source of truth; per-rank / effective / union summaries are derived
    # from it via ``BenchmarkStats.per_rank`` / ``per_round``. The mirrored
    # dispatches are shared with the flat ``invocations`` list.
    stats.rounds_dispatches = [{} for _ in range(rounds)]
    for pid, invs in by_pid.items():
        d = len(invs) // launches
        chunks = [invs[k * d : (k + 1) * d] for k in range(launches)][warmup:]
        for k, chunk in enumerate(chunks):
            dispatches = [_mirror_invocation(inv) for inv in chunk]
            stats.invocations.extend(dispatches)
            stats.rounds_dispatches[k][pid] = dispatches

    # Headline per round: max across ranks (slowest rank bounds the round).
    pr_dev = stats.per_rank("device")
    pr_host = stats.per_rank("host")
    for k in range(rounds):
        stats.device_wall_us.append(max(v[k] for v in pr_dev.values()))
        stats.host_wall_us.append(max(v[k] for v in pr_host.values()))
    return stats


def _parse_stats_from_strace(
    log_text: str, *, rounds: int, warmup: int, distributed: bool = False
) -> BenchmarkStats:
    """Build a :class:`BenchmarkStats` from captured ``[STRACE]`` log text.

    Parsing is delegated to simpler's ``strace_timing`` — the single source of
    truth for the marker grammar — then each launch's full span tree is mirrored
    into pypto-owned :class:`TraceInvocation` / :class:`TraceSpan` so callers
    never import simpler types.

    L2 (``distributed=False``): groups markers by ``(pid, inv)``, buckets by
    callable hash, takes the busiest bucket (our register-once callable emits one
    invocation per launch), orders by ``inv``, drops the first *warmup*
    invocations, and reads each remaining launch's host (``<root>``) and device
    (``<root>.runner_run.device_wall``) span durations (µs). The ``<root>`` span
    name is resolved per :func:`_span_names` (``run_prepared`` / ``simpler_run``).

    L3 (``distributed=True``): delegates to :func:`_parse_l3_stats`, which folds
    the per-rank chip-child markers into per-round aggregates (see that function).
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

    names = _span_names()
    stats = BenchmarkStats(rounds=rounds, warmup=warmup)
    invocations = group_invocations(parse_spans(log_text.splitlines()))
    if not invocations:
        return stats

    if distributed:
        return _parse_l3_stats(invocations, stats, rounds=rounds, warmup=warmup)

    # Busiest hid bucket = our register-once callable (one invocation per launch);
    # bucket_by_hid orders each bucket by inv, so warmup drops in dispatch order.
    busiest = max(bucket_by_hid(invocations).values(), key=len)
    for inv in busiest[warmup:]:
        stats.host_wall_us.append(_inv_span_us(inv, names["host"]))
        stats.device_wall_us.append(_inv_span_us(inv, names["device"]))
        stats.invocations.append(_mirror_invocation(inv))

    return stats


def _dispatch_loop(
    handle: Any, args: Sequence[Any], *, rounds: int, warmup: int, dispatch_config: Any
) -> None:
    """Dispatch ``warmup + rounds`` launches on *handle* (no capture).

    Shared by the L2 (``ChipWorker``) and L3 (``DistributedWorker``) paths: both
    expose the same register-once :class:`RegistrationHandle`. The ``[STRACE]``
    stderr capture is set up by the caller — its scope differs per path (L2 wraps
    only this loop; L3 must wrap ``prepare()`` too, see :func:`benchmark`).
    """
    for _ in range(warmup):  # warm caches / page-in; markers discarded
        handle(*args, config=dispatch_config)
    for _ in range(rounds):  # measured launches
        handle(*args, config=dispatch_config)


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

    Dispatches by *compiled* type:

    - **L2** (:class:`~pypto.ir.CompiledProgram`): opens a single
      :class:`~pypto.runtime.ChipWorker`.
    - **L3** (:class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`):
      opens a :class:`~pypto.runtime.distributed_runner.DistributedWorker` via
      ``compiled.prepare()``.

    Either way it registers *compiled* once, then loops the bound handle so each
    launch only re-pays argument coercion + dispatch (not register/load). The
    on-NPU ``device_wall_us`` is measured between the orchestrator's
    ``orch_start`` / ``orch_end`` and is unaffected by the per-launch host-side
    arg building.

    Timing is read from the runtime's ``[STRACE]`` stderr markers (simpler PR
    #1177): this raises the runtime log level to ``v9`` for the worker's
    lifetime (restored afterward) and captures ``stderr`` at the file-descriptor
    level, so the emitted stderr is diverted into a temp file rather than shown
    live. For L2 the capture wraps only the measured loop; for L3 it must wrap
    ``compiled.prepare()`` as well, because the chip workers are forked there and
    inherit fd 2 at fork time — a redirect set up after the fork would miss the
    children's markers entirely. (On L3 failure the diverted setup stderr is
    echoed back so diagnostics are not lost.)

    Args:
        compiled: A single-orchestration
            :class:`~pypto.ir.CompiledProgram` (L2) or a
            :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
            (L3) from ``ir.compile`` / ``compile_program``. Multi-orch L2
            programs must pass ``compiled[<name>]``.
        args: Positional dispatch args, same as ``compiled(*args)``. **L3
            requires shared-memory host** ``torch.Tensor`` **args** (allocated
            with ``.share_memory_()`` and reused in place) or worker-resident
            :class:`~pypto.runtime.DeviceTensor` buffers — a buffer allocated
            after ``prepare()`` is invisible to the forked chip workers.
        rounds: Number of measured launches. Must be positive.
        warmup: Number of leading launches discarded before measurement
            (page-in / cache warm). Total launches = ``warmup + rounds``.
        platform: Target platform shorthand, e.g. ``"a2a3"``. Defaults to
            ``compiled.platform``. Mutually exclusive with *config*. **L2 only** —
            not accepted for L3 (device set fixed at compile time).
        device_id: NPU device index. Defaults to ``RunConfig``'s default.
            Mutually exclusive with *config*. **L2 only** — not accepted for L3
            (device set comes from ``distributed_config.device_ids``).
        config: Optional :class:`~pypto.runtime.RunConfig`. L2: full control
            (``block_dim`` / ``aicpu_thread_num`` / ``pto_isa_commit``); pass this
            *or* *platform*/*device_id*, not both. L3: forwarded per dispatch for
            ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
            ``ring_dep_pool``); ``None`` reuses the prepared baseline.

    Returns:
        A :class:`BenchmarkStats` with the per-round ``device_wall_us`` /
        ``host_wall_us`` samples and aggregate helpers. For L3 the samples are
        per-round maxima across ranks; per-rank / effective / union summaries are
        derived on demand via :meth:`BenchmarkStats.per_rank` (``device`` /
        ``host`` / ``effective``) and :meth:`BenchmarkStats.per_round` (those plus
        ``union``), and the ``round → rank → [dispatch]`` grid is in
        :attr:`BenchmarkStats.rounds_dispatches`.

    Raises:
        ValueError: ``rounds <= 0``, ``warmup < 0``, *config* passed together
            with *platform* / *device_id* (L2), or *platform* / *device_id*
            passed for an L3 program.
        RuntimeError: No ``[STRACE]`` markers were captured at all, so no timing
            could be read. The markers are gated by the runtime's compile-time
            ``SIMPLER_HOST_STRACE`` macro; a runtime built without it emits none.

    Note:
        On a ``*sim`` platform the host ``<root>`` span is still emitted but the
        device-domain spans are not, so every ``device_wall_us`` sample is ``0``
        — check :attr:`BenchmarkStats.all_zero_device`.

        L3 has no DAG-level device wall (only the forked chip children emit
        markers). Each round's ``device_wall_us`` is the **max across ranks** of
        that round's per-rank **summed** dispatch device walls — a proxy for round
        device time that excludes inter-dispatch idle gaps and cross-rank start
        skew (device clocks are per-invocation, so gaps/skew are unmeasurable).
        ``per_round("union")`` complements it with the cross-rank **host-timeline**
        union window (the ``<root>`` host clocks are ``CLOCK_MONOTONIC``,
        cross-process comparable), which *does* capture overlap / start skew but
        includes host-side dispatch overhead. A true pure-device end-to-end DAG
        wall is not recoverable until the runtime emits a device→host clock
        anchor. Per-round alignment assumes a deterministic dispatch shape
        (constant dispatches per round); if that does not hold,
        :attr:`BenchmarkStats.fallback_flattened` is set, the samples become a
        flattened per-dispatch pool, and ``per_rank`` / ``per_round("union")`` are
        empty.
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    # L3 distributed programs run through DistributedWorker, not ChipWorker.
    from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
        DistributedCompiledProgram,
    )

    distributed = isinstance(compiled, DistributedCompiledProgram)

    # Validate mutually-exclusive arguments up front, before any logging setup,
    # so a bad argument combination is rejected regardless of whether the
    # simpler-backed logger is importable (it is optional in offline envs).
    if distributed:
        # Device set is fixed at compile time via ``distributed_config``;
        # platform/device_id do not apply. ``config`` is still forwarded per
        # dispatch (ring overrides).
        if platform is not None or device_id is not None:
            raise ValueError(
                "benchmark(): platform=/device_id= do not apply to an L3 "
                "DistributedCompiledProgram — the device set is fixed at compile "
                "time via distributed_config. Pass config=RunConfig(...) for "
                "per-dispatch ring overrides instead."
            )
    elif config is not None and (platform is not None or device_id is not None):
        raise ValueError("benchmark(): pass either config=... or platform=/device_id=, not both")

    # The C++ host logger that prints the ``[STRACE]`` markers is seeded from the
    # simpler Python logger snapshot at worker ``init`` (and inherited by the L3
    # fork), so raise the level before constructing the worker. Restore afterward.
    prior_level = current_level()
    configure_log(_STRACE_LOG_LEVEL)
    try:
        with tempfile.TemporaryDirectory(prefix="pypto-bench-") as tmp:
            log_path = Path(tmp) / "strace.log"
            if distributed:
                # The L3 chip workers are forked inside ``prepare()`` and inherit
                # fd 2 at fork time, so the stderr redirect MUST wrap ``prepare()``
                # — a redirect established after the fork would not capture the
                # children's markers. This diverts ``prepare()``'s own setup
                # stderr too; on failure it is echoed back so diagnostics survive.
                try:
                    with _capture_fd_stderr(log_path):
                        with compiled.prepare() as rt:
                            handle = rt.register(compiled)  # register once (cid=0)
                            _dispatch_loop(handle, args, rounds=rounds, warmup=warmup, dispatch_config=config)
                except Exception:
                    captured = log_path.read_text(encoding="utf-8", errors="replace")
                    if captured:
                        print(captured, file=sys.stderr, end="")
                    raise
            else:
                if config is not None:
                    rc = config
                else:
                    rc_kwargs: dict[str, Any] = {"platform": platform or compiled.platform}
                    if device_id is not None:
                        rc_kwargs["device_id"] = device_id
                    rc = RunConfig(**rc_kwargs)
                # L2 runs the chip in-process (no fork), so the parent's fd 2
                # redirect during the loop captures its markers.
                with ChipWorker(rc, runtime=compiled.runtime_name) as worker:
                    handle = worker.register(compiled)  # register once; cid cached
                    with _capture_fd_stderr(log_path):
                        _dispatch_loop(handle, args, rounds=rounds, warmup=warmup, dispatch_config=rc)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        configure_log(prior_level)

    stats = _parse_stats_from_strace(log_text, rounds=rounds, warmup=warmup, distributed=distributed)
    # We dispatched warmup + rounds launches, so a marker-emitting runtime always
    # yields at least one host span. Zero markers means the runtime emitted none
    # (built without SIMPLER_HOST_STRACE) — surface that rather than returning a
    # silently-empty result a caller could misread as "0 device timing".
    if not stats.host_wall_us:
        raise RuntimeError(
            f"benchmark(): no [STRACE] markers captured across {warmup + rounds} launches. "
            "The runtime emits per-launch timing markers only when built with the "
            "SIMPLER_HOST_STRACE macro (LOG_INFO_V9 tier); this runtime emitted none. "
            "Rebuild the runtime with SIMPLER_HOST_STRACE enabled to read benchmark timing."
        )
    return stats
