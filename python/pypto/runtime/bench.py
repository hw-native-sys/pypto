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
``SIMPLER_HOST_STRACE`` macro and emitted at the ``LOG_TIMING`` tier). This
module therefore:

1. sets the simpler runtime log level to ``timing`` so the markers print (the
   C++ host logger is seeded from the Python logger snapshot at
   ``ChipWorker.init``), then restores the prior level afterward;
2. redirects ``stderr`` at the file-descriptor level (``os.dup2`` — Python's
   ``contextlib.redirect_stderr`` cannot capture the C++ writes) into a temp
   file around the measured region (for L3, also around ``prepare()`` so the
   forked chip-worker processes inherit the redirected fd);
3. parses the captured markers, reading each launch's on-NPU ``device_wall``
   and host ``chip.run`` span.

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
#
# Prepared onboard L3 swimlane executes each caller-visible launch twice. The
# parent brackets the clean pass with these sentinels so the parser can discard
# the dep-gen graph pass instead of reporting contaminated benchmark numbers.
# Keep the values stable: pypto-lib's resident benchmark calls the same private
# parser.
_L3_SWIMLANE_PASS_PREFIX = "[pypto] l3_swimlane_pass="
_L3_SWIMLANE_GRAPH_BEGIN = f"{_L3_SWIMLANE_PASS_PREFIX}graph event=begin"
_L3_SWIMLANE_GRAPH_END = f"{_L3_SWIMLANE_PASS_PREFIX}graph event=end"
_L3_SWIMLANE_TIMING_BEGIN = f"{_L3_SWIMLANE_PASS_PREFIX}timing event=begin"
_L3_SWIMLANE_TIMING_END = f"{_L3_SWIMLANE_PASS_PREFIX}timing event=end"


@dataclass
class TraceSpan:
    """One ``[STRACE]`` span — a node in a measured launch's call tree.

    A pypto-owned mirror of simpler's ``strace_timing.Span`` so ``benchmark``
    callers never depend on simpler types.

    Attributes:
        depth: Nesting level; a span at depth ``d`` is a child of the nearest
            enclosing span at depth ``d-1``.
        name: Dotted span path (e.g. ``chip.run.runner_run.device_wall``).
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
        """The depth-0 span (``chip.run``), or ``None`` if absent."""
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
        hash — see :attr:`task_name` for the orchestration's readable name.
        """
        return self.hid

    @property
    def task_name(self) -> str:
        """This dispatch's orchestration name, or its :attr:`task` hash.

        The markers carry no name, so it is recovered by matching ``hid`` against
        the ``hid → name`` pairings ``device_runner`` records when it
        assembles each callable (the hid is the ELF Build-ID of the very
        orchestration ``.so`` the runtime hashes). Degrades to the raw hash when
        the callable was not assembled in this process, on ``*sim`` platforms, or
        when the optional runtime is not installed — see :func:`_task_label`.
        """
        return _task_label(self.hid)

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
        device-domain spans (``chip.run.runner_run.device_wall.*``) correctly
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
# is the "Effective" on-device execution window). The span root has been renamed
# twice — ``run_prepared`` -> ``simpler_run`` (simpler #1210), then
# ``simpler_run`` -> ``chip.run`` when #1877/#1893 made every span lead with the
# word for the level that emitted it — so the names are sourced at call time from
# the installed runtime's ``strace_timing._ROUNDS_TABLE_NAMES`` via
# :func:`_span_names` rather than hardcoded. These legacy names are the pre-#1210
# fallback, used only when that table is absent entirely.
_LEGACY_SPAN_NAMES = {
    "host": "run_prepared",
    "device": "run_prepared.runner_run.device_wall",
    "orch": "run_prepared.runner_run.device_wall.orch",
    "sched": "run_prepared.runner_run.device_wall.sched",
}

# PyPTO's span key -> the ``_ROUNDS_TABLE_NAMES`` keys that may carry it, newest
# generation first. The whole-run entry was keyed ``host`` until simpler #1893
# renamed it ``run``: the word ``host`` names a *processor*, and the span it
# labels is the chip's run, so the table stopped spelling it that way. PyPTO
# keeps ``host`` as its own key (it is the host-side wall, and it is the name
# :class:`BenchmarkStats` exposes); only the lookup has to know both spellings.
# ``device`` / ``orch`` / ``sched`` are stable across both generations.
_RUNTIME_SPAN_KEYS = {
    "host": ("run", "host"),
    "device": ("device",),
    "orch": ("orch",),
    "sched": ("sched",),
}


# Span families the invocation-keyed views must not consume. They share the
# ``[STRACE]`` grammar but carry no invocation id, so admitting one groups all of
# its spans into a single forged invocation. ``l3.`` is the pre-#1877 spelling of
# the per-task scheduler family; #1893 re-spelled it after the level's topology
# position (``node``, plus ``network1..3`` for each hop above it), and #1886
# reserved ``ext.`` for producers outside simpler. Applied on top of simpler's own
# filter, which keeps spellings it does not recognize — see
# :func:`_parse_stats_from_strace`.
_NON_INVOCATION_PREFIXES = ("l3.", "node.", "network1.", "network2.", "network3.", "ext.")


@functools.lru_cache(maxsize=1)
def _span_names() -> dict[str, str]:
    """Resolve the four ``[STRACE]`` span names from the installed runtime.

    Reads ``strace_timing._ROUNDS_TABLE_NAMES`` (added in simpler #1210), trying
    each spelling in :data:`_RUNTIME_SPAN_KEYS` so one lookup covers both the
    ``host``-keyed and the ``run``-keyed generation of the table.

    Resolution is all-or-nothing: a table that answers only some of the four
    keys is a generation this function does not know, and mixing its names with
    the legacy ones would silently yield spans that match nothing. Falling back
    wholesale keeps the failure to "every sample is zero" in one place rather
    than spreading it across metrics.
    """
    try:
        from simpler_setup.tools.strace_timing import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            _ROUNDS_TABLE_NAMES,
        )
    except (ImportError, AttributeError):
        return dict(_LEGACY_SPAN_NAMES)

    resolved: dict[str, str] = {}
    for key, candidates in _RUNTIME_SPAN_KEYS.items():
        for candidate in candidates:
            try:
                resolved[key] = _ROUNDS_TABLE_NAMES[candidate]
            except (TypeError, KeyError):
                continue
            break
        else:
            return dict(_LEGACY_SPAN_NAMES)
    return resolved


# Runtime log level that makes the ``LOG_TIMING`` ``[STRACE]`` markers visible.
_STRACE_LOG_LEVEL = "timing"

# Metric name → the per-dispatch :class:`TraceInvocation` attribute it reads
# (used by ``BenchmarkStats.per_dispatch`` / ``per_rank`` / ``per_round``).
_METRIC_ATTR = {"device": "device_wall_us", "host": "host_wall_us", "effective": "effective_us"}

# Shown when no ``[STRACE]`` span tree was captured at all.
_NO_TREE_MSG = "BenchmarkStats: no span tree captured (non-SIMPLER_HOST_STRACE build or *sim platform)"

# Shown when the dispatch slots are not stable enough to group by (see
# ``BenchmarkStats.unstable_dispatch_slots``).
_UNSTABLE_SLOTS_MSG = (
    "BenchmarkStats: per-dispatch view unavailable — a rank's dispatch order "
    "varies between rounds, so a slot does not identify one callable. The "
    "per-rank / per-round sums are unaffected."
)

# Max ``(pid, slot)`` groups listed inline by ``BenchmarkStats.__str__`` before
# the tail is elided (the full set is always available via ``per_dispatch``).
_STR_MAX_DISPATCHES = 8


def _task_label(hid: str) -> str:
    """*hid* resolved to its orchestration's name, or *hid* unchanged.

    The ``[STRACE]`` markers identify a callable only by ``hid`` — the ELF
    Build-ID of its orchestration ``.so``. ``device_runner`` records
    ``hid → name`` for every callable it assembles, which covers any
    program compiled in this process, so the lookup normally succeeds.

    Falls back to the raw hash when it cannot: the callable was assembled in a
    different process, the platform is ``*sim`` (its host seeds the marker hid
    with the runtime ``callable_id`` instead of a Build-ID), or the optional
    runtime package is not installed at all (``device_runner`` pulls in the
    native ``_task_interface``, hence the guarded import).
    """
    try:
        from .device_runner import callable_name  # noqa: PLC0415
    except ImportError:
        return hid
    return callable_name(hid) or hid


def _metric_attr(metric: str, *, caller: str) -> str:
    """The :class:`TraceInvocation` attribute *metric* names, validated."""
    attr = _METRIC_ATTR.get(metric)
    if attr is None:
        raise ValueError(f"{caller}: metric must be one of {sorted(_METRIC_ATTR)}, got {metric!r}")
    return attr


def _mean_of(invocations: Sequence["TraceInvocation"]) -> "TraceInvocation | None":
    """A synthetic :class:`TraceInvocation` averaging *invocations* span-by-span.

    Spans are matched by name; ``depth`` / ``attrs`` (hence
    :attr:`TraceSpan.is_device`) come from the first invocation carrying the span.
    ``inv`` is ``-1`` to mark the aggregate. ``None`` when *invocations* is empty.

    Callers must only pass invocations of the **same** dispatch (same ``(pid,
    slot)``): averaging distinct dispatches by span name fuses unrelated kernels
    into one meaningless tree.
    """
    if not invocations:
        return None
    durs: dict[str, list[int]] = defaultdict(list)
    tss: dict[str, list[int]] = defaultdict(list)
    template: dict[str, TraceSpan] = {}
    for inv in invocations:
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
    first = invocations[0]
    return TraceInvocation(pid=first.pid, inv=-1, hid=first.hid, spans=spans)


@dataclass
class BenchmarkStats:
    """Aggregated per-launch timing from :func:`benchmark`.

    **Quick-reference:**

    | Accessor | Description |
    | --- | --- |
    | `stats.device_us_median` | Median device wall (µs). |
    | `stats.device_us_min` | Minimum device wall (µs). |
    | `stats.device_us_max` | Maximum device wall (µs). |
    | `stats.device_us_mean` | Arithmetic mean device wall (µs). |
    | `stats.device_us_stdev` | Std-dev of device wall (µs). |
    | `stats.all_zero_device` | True if samples exist and every sample is 0 (e.g. sim builds). |
    | `stats.samples` | Alias for `device_wall_us` (the raw list). |
    | `stats.per_round("device")` | List[float]: device wall per round. |
    | `stats.per_rank("device")` | Dict[int, List[float]]: per-rank breakdown (L3). |
    | `stats.per_dispatch("device")` | Dict[(pid, slot), List[float]]: per-dispatch (unsummed) view (L3). |
    | `stats.dispatch_tasks()` | Dict[(pid, slot), str]: task name (or hash) labelling each dispatch slot. |
    | `stats.unstable_dispatch_slots` | True if a rank's dispatch order varies between rounds (L3). |
    | `stats.print_tree()` | Render per-dispatch span tree to stdout. |

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
      [dispatch]`` grid, the single source for per-dispatch / per-rank /
      effective / union derivations) and :attr:`invocations` (the same dispatches
      flattened, for tree rendering).
    - **Derived summaries**: :meth:`per_dispatch` (no fusing — one series per
      ``(pid, slot)`` dispatch), :meth:`per_rank` (that rank's dispatches summed
      per round) and :meth:`per_round`, for the metrics ``device`` / ``host`` /
      ``effective`` (plus ``union`` on ``per_round``).

    A rank that dispatches more than once per round is **not** fused by
    :meth:`per_dispatch` or by the mean-tree views (:meth:`format_mean_tree`
    groups per dispatch by default); :meth:`per_rank` / :meth:`per_round`
    deliberately do sum, as a rank's round busy-time.

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
            / ``effective_us``. It is the single source :meth:`per_dispatch` /
            :meth:`per_rank` / :meth:`per_round` derive from, and its dispatch
            position is the ``slot`` those views key on. Same objects as
            :attr:`invocations`.
            Length is ``rounds``; empty for L2 and the flatten fallback.
        fallback_flattened: L3 only — ``True`` when per-round segmentation was not
            possible (a rank's marker count was not divisible by
            ``warmup + rounds``, i.e. a non-deterministic dispatch shape). Then
            :attr:`device_wall_us` / :attr:`host_wall_us` hold the pooled
            per-dispatch samples (warmup dropped per rank), :attr:`rounds_dispatches`
            is empty, and ``per_rank`` / ``per_round("union")`` return empty.
        unstable_dispatch_slots: L3 only — ``True`` when some ``(pid, slot)``
            carried more than one :attr:`TraceInvocation.task` across the measured
            rounds, i.e. a rank issued a *constant number* of dispatches but not
            always in the same order (round 0 ``A, B`` then round 1 ``B, A``).
            The ordinal slot then does not identify a callable, so grouping by it
            would average distinct kernels under the first round's label — the
            very fusing the per-dispatch views exist to remove. They therefore
            report empty (:meth:`dispatch_groups`, :meth:`per_dispatch`,
            :meth:`dispatch_tasks`) and :meth:`format_mean_tree` explains why.
            Round boundaries are unaffected, so :attr:`rounds_dispatches`,
            :meth:`per_rank` and :meth:`per_round` stay valid and populated.
    """

    device_wall_us: list[float] = field(default_factory=list)
    host_wall_us: list[float] = field(default_factory=list)
    rounds: int = 0
    warmup: int = 0
    invocations: list[TraceInvocation] = field(default_factory=list)
    rounds_dispatches: list[dict[int, list[TraceInvocation]]] = field(default_factory=list)
    fallback_flattened: bool = False
    unstable_dispatch_slots: bool = False

    def dispatch_groups(self) -> dict[tuple[int, int], list[TraceInvocation]]:
        """Per-dispatch invocation series: ``{(pid, slot): [round0, round1, ...]}``.

        *slot* is the dispatch's position within its rank's round (dispatches are
        ordered by ``inv``). Keys are sorted by ``(pid, slot)``.

        This is the un-fused view: a rank issuing several dispatches per round
        gets one entry per dispatch instead of a single summed number. Derived
        from :attr:`rounds_dispatches`, so it is **L3 only** — returns ``{}`` for
        L2 and for the flatten fallback, and also when
        :attr:`unstable_dispatch_slots` says a slot does not identify one
        callable across rounds.
        """
        if self.unstable_dispatch_slots:
            return {}
        out: dict[tuple[int, int], list[TraceInvocation]] = {}
        for ranks in self.rounds_dispatches:
            for pid, dispatches in ranks.items():
                for slot, dispatch in enumerate(dispatches):
                    out.setdefault((pid, slot), []).append(dispatch)
        return {key: out[key] for key in sorted(out)}

    def dispatch_tasks(self) -> dict[tuple[int, int], str]:
        """``{(pid, slot): name}`` — what each dispatch runs, for labelling slots.

        The dispatch's :attr:`TraceInvocation.task_name` (its orchestration
        name, or the raw ``hid`` hash when that cannot be resolved),
        taken from its first measured round. ``{}`` for L2 / fallback. For the
        wire identity itself use ``dispatch_groups()[key][0].task``.
        """
        return {key: group[0].task_name for key, group in self.dispatch_groups().items() if group}

    def per_dispatch(self, metric: str = "device") -> dict[tuple[int, int], list[float]]:
        """Per-dispatch, per-round summary (µs): ``{(pid, slot): [round0, ...]}``.

        *metric* is one of ``"device"`` / ``"host"`` / ``"effective"``. Unlike
        :meth:`per_rank`, nothing is summed — each entry is one dispatch's own
        :attr:`TraceInvocation.device_wall_us` / ``host_wall_us`` /
        ``effective_us``, so a rank's repeated or heterogeneous dispatches stay
        separate. Pair with :meth:`dispatch_tasks` to label the slots.

        Derived from :attr:`rounds_dispatches`, so it is **L3 only** — returns
        ``{}`` for L2 and for the flatten fallback.
        """
        attr = _metric_attr(metric, caller="per_dispatch()")
        return {key: [getattr(d, attr) for d in group] for key, group in self.dispatch_groups().items()}

    def per_rank(self, metric: str = "device") -> dict[int, list[float]]:
        """Per-rank, per-round summary (µs): ``{pid: [round0, round1, ...]}``.

        *metric* is one of ``"device"`` / ``"host"`` / ``"effective"`` — each
        round entry sums that rank's dispatches'
        :attr:`TraceInvocation.device_wall_us` / ``host_wall_us`` / ``effective_us``
        (a card runs its dispatches serially), i.e. that rank's busy time for the
        round. Use :meth:`per_dispatch` when you need the individual dispatches
        rather than their sum. Derived from :attr:`rounds_dispatches`, so it is
        **L3 only** — returns ``{}`` for L2 and for the flatten fallback.
        """
        attr = _metric_attr(metric, caller="per_rank()")
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
            return _NO_TREE_MSG
        selected = (
            list(enumerate(self.invocations)) if launch is None else [(launch, self.invocations[launch])]
        )
        # L3: label each launch with the (round, slot) it belongs to, so a rank's
        # repeated dispatches are told apart in the flat ``invocations`` ordering.
        where = self._dispatch_coords()
        out: list[str] = []
        for i, inv in selected:
            coord = where.get(id(inv))
            tag = f" round={coord[0]} slot={coord[1]}" if coord is not None else ""
            name = inv.task_name
            if name != inv.hid:  # resolved to a readable orchestration name
                tag += f" task={name}"
            out.append(f"launch[{i}] (pid={inv.pid} inv={inv.inv} hid={inv.hid}{tag}):")
            out.append(inv.format_tree(us=us))
        return "\n".join(out)

    def print_tree(self, launch: int | None = None, *, us: bool = True, file: Any = None) -> None:
        """Print :meth:`format_tree` to *file* (default stdout)."""
        print(self.format_tree(launch, us=us), file=file)

    def _dispatch_coords(self) -> dict[int, tuple[int, int]]:
        """``id(dispatch) -> (round, slot)`` for the L3 grid; ``{}`` otherwise.

        :attr:`rounds_dispatches` holds the very same :class:`TraceInvocation`
        objects as :attr:`invocations`, so identity is a valid key.
        """
        out: dict[int, tuple[int, int]] = {}
        for k, ranks in enumerate(self.rounds_dispatches):
            for dispatches in ranks.values():
                for slot, dispatch in enumerate(dispatches):
                    out[id(dispatch)] = (k, slot)
        return out

    def _mean_tree_groups(
        self, *, pid: int | None = None, slot: int | None = None
    ) -> list[tuple[str | None, list[TraceInvocation]]]:
        """The ``(label, invocations)`` groups the mean-tree views average over.

        L3 (:attr:`rounds_dispatches` populated): one group per ``(pid, slot)``
        dispatch — so a rank's distinct dispatches are never averaged into one
        tree — narrowed by the optional *pid* / *slot* selectors. L2 and the
        flatten fallback have no dispatch grid, so they yield a single unlabeled
        group holding every measured launch.

        Yields nothing when :attr:`unstable_dispatch_slots` is set: falling back
        to the single unlabeled group there would blend a rank's distinct
        callables into one tree, which is exactly what the flag exists to
        prevent.
        """
        if self.unstable_dispatch_slots:
            return []
        groups = self.dispatch_groups()
        if not groups:
            if pid is not None or slot is not None:
                return []
            return [(None, self.invocations)] if self.invocations else []
        tasks = self.dispatch_tasks()
        return [
            (f"pid={key[0]} slot={key[1]} task={tasks[key]}", invs)
            for key, invs in groups.items()
            if (pid is None or key[0] == pid) and (slot is None or key[1] == slot)
        ]

    def mean_invocation(self, *, pid: int | None = None, slot: int | None = None) -> "TraceInvocation | None":
        """A synthetic :class:`TraceInvocation` whose every span's ``dur`` (and
        ``ts``) is the mean across the measured launches of **one** dispatch
        (warmup excluded).

        Spans are matched by name; ``depth`` / ``attrs`` (hence
        :attr:`TraceSpan.is_device`) come from the first launch that carried the
        span. ``inv`` is ``-1`` to mark the aggregate. Returns ``None`` when no
        span tree was captured. Useful for rendering one noise-smoothed tree.

        Args:
            pid: Restrict to this rank (L3). ``None`` means "any".
            slot: Restrict to this dispatch slot within the round (L3).

        Raises:
            ValueError: The selectors do not narrow an L3 run down to a single
                ``(pid, slot)`` dispatch. Averaging distinct dispatches by span
                name would fuse unrelated kernels into one meaningless tree, so
                pass ``pid=`` / ``slot=`` (see :meth:`dispatch_tasks` for the
                available keys) or use :meth:`format_mean_tree`, which renders
                every dispatch's tree.
        """
        groups = self._mean_tree_groups(pid=pid, slot=slot)
        if not groups:
            return None
        if len(groups) > 1:
            raise ValueError(
                f"mean_invocation(): {len(groups)} dispatches match pid={pid} slot={slot} "
                f"({sorted(self.dispatch_groups())}); averaging them by span name would fuse "
                "distinct dispatches. Pass pid=/slot= to select one, or use format_mean_tree()."
            )
        return _mean_of(groups[0][1])

    def format_mean_tree(
        self,
        *,
        us: bool = True,
        spread: str = "stdev",
        pid: int | None = None,
        slot: int | None = None,
    ) -> str:
        """Render a span tree whose every node's duration is the mean across the
        measured launches (warmup excluded), annotated with the per-node spread.

        On an L3 run one tree is rendered **per dispatch** (``(pid, slot)``, see
        :meth:`dispatch_groups`): a rank that dispatches several kernels per round
        gets one tree each rather than a single tree averaging them together.

        Args:
            us: Show values in microseconds (default) or nanoseconds.
            spread: Spread shown after each node's mean — ``"stdev"`` (``±sd``,
                default), ``"minmax"`` (``[min..max]``), ``"both"``, or
                ``"none"``. Computed across the measured launches.
            pid: Render only this rank's dispatches (L3).
            slot: Render only this dispatch slot within the round (L3).
        """
        groups = self._mean_tree_groups(pid=pid, slot=slot)
        if not groups:
            if self.unstable_dispatch_slots:
                return _UNSTABLE_SLOTS_MSG
            if self.invocations:
                return f"BenchmarkStats: no dispatch matches pid={pid} slot={slot}"
            return _NO_TREE_MSG

        scale = 1000.0 if us else 1.0
        unit = "us" if us else "ns"
        legend = "mean"
        if spread in ("stdev", "both"):
            legend += " ±stdev"
        if spread in ("minmax", "both"):
            legend += " [min..max]"

        out: list[str] = []
        for label, invs in groups:
            mean_inv = _mean_of(invs)
            if mean_inv is None:
                continue
            durs: dict[str, list[int]] = defaultdict(list)
            for inv in invs:
                for s in inv.spans:
                    durs[s.name].append(s.dur)

            def _value(span: TraceSpan, durs: dict[str, list[int]] = durs) -> list[str]:
                ds = durs[span.name]
                cols = [f"{statistics.fmean(ds) / scale:.1f}{unit}"]
                if spread in ("stdev", "both"):
                    sd = statistics.stdev(ds) / scale if len(ds) > 1 else 0.0
                    cols.append(f"±{sd:.1f}")
                if spread in ("minmax", "both"):
                    cols.append(f"[{min(ds) / scale:.1f}..{max(ds) / scale:.1f}]")
                return cols

            prefix = f"dispatch {label} — " if label is not None else ""
            if out:
                out.append("")
            out.append(
                f"{prefix}mean of {len(invs)} launches (warmup {self.warmup} excluded); each node: {legend}:"
            )
            out.append(mean_inv.format_tree(us=us, value_fn=_value))
        return "\n".join(out)

    def print_mean_tree(
        self,
        *,
        us: bool = True,
        spread: str = "stdev",
        pid: int | None = None,
        slot: int | None = None,
        file: Any = None,
    ) -> None:
        """Print :meth:`format_mean_tree` to *file* (default stdout)."""
        print(self.format_mean_tree(us=us, spread=spread, pid=pid, slot=slot), file=file)

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
            f"{self._per_dispatch_line()}"
        )

    def _per_dispatch_line(self) -> str:
        """A trailing ``__str__`` line breaking the headline down per dispatch.

        Only emitted when some rank dispatches more than once per round — the case
        where the summed per-rank / per-round headline hides the individual
        dispatches. Long dispatch sets are elided; :meth:`per_dispatch` has all of
        them.
        """
        per_dispatch = self.per_dispatch("device")
        if len(per_dispatch) <= len(self.per_rank("device")):
            return ""  # at most one dispatch per rank: nothing is being fused
        tasks = self.dispatch_tasks()
        shown = list(per_dispatch.items())[:_STR_MAX_DISPATCHES]
        cells = [
            f"(pid={pid},slot={slot},task={tasks[pid, slot]})={statistics.fmean(vals):.1f}"
            for (pid, slot), vals in shown
        ]
        elided = len(per_dispatch) - len(shown)
        if elided:
            cells.append(f"... +{elided} more")
        return "\n  per-dispatch device mean us: " + "  ".join(cells)


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


def _is_dispatch_invocation(inv: Any, host_name: str) -> bool:
    """Whether *inv* contains the canonical dispatch host span at depth 0."""
    return any(span.depth == 0 and span.name == host_name for span in inv.spans)


def _has_unstable_slots(rounds_dispatches: list[dict[int, list[TraceInvocation]]]) -> bool:
    """Whether any ``(pid, slot)`` carried more than one task across the rounds.

    See :attr:`BenchmarkStats.unstable_dispatch_slots`. First mismatch wins, so
    this is O(total dispatches).
    """
    seen: dict[tuple[int, int], str] = {}
    for ranks in rounds_dispatches:
        for pid, dispatches in ranks.items():
            for slot, dispatch in enumerate(dispatches):
                if seen.setdefault((pid, slot), dispatch.task) != dispatch.task:
                    return True
    return False


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

    # The capture must begin before ``prepare()`` so forked chip workers inherit
    # its fd, but prepare-time setup can emit unrelated invocation groups such as
    # ``chip.prewarm.build``. Only a depth-0 canonical run root identifies an
    # actual dispatch. Do not filter on ``device_wall`` per invocation: retaining
    # a real run with a missing device marker preserves its round alignment and
    # exposes a zero metric.
    names = _span_names()
    host_name = names["host"]
    device_name = names["device"]
    by_pid: dict[int, list[Any]] = defaultdict(list)
    for inv in invocations:
        if _is_dispatch_invocation(inv, host_name):
            by_pid[inv.pid].append(inv)
    for invs in by_pid.values():
        invs.sort(key=lambda i: i.inv)

    # A real chip-child rank emits at least one ``device_wall`` span; the L3
    # host-orch parent process emits its own run root without any chip
    # ``device_wall``, so its pid must not be grouped as a rank — otherwise it
    # adds a fake zero-device rank that corrupts ``per_rank`` / rank counts and
    # pollutes the ``host_wall`` / ``union`` windows with the parent orch span.
    by_pid = {
        pid: invs
        for pid, invs in by_pid.items()
        if invs and any(inv.by_name().get(device_name) is not None for inv in invs)
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

    # A constant dispatch count per round fixes the round boundaries, but it does
    # not make the ordinal slot a callable identity: a rank could issue A, B one
    # round and B, A the next. Grouping those by slot would average distinct
    # kernels under the first round's label, so flag it and let the per-dispatch
    # views report empty instead. The per-rank sums are order-independent and
    # stay valid.
    stats.unstable_dispatch_slots = _has_unstable_slots(stats.rounds_dispatches)

    # Headline per round: max across ranks (slowest rank bounds the round).
    pr_dev = stats.per_rank("device")
    pr_host = stats.per_rank("host")
    for k in range(rounds):
        stats.device_wall_us.append(max(v[k] for v in pr_dev.values()))
        stats.host_wall_us.append(max(v[k] for v in pr_host.values()))
    return stats


def _extract_l3_swimlane_timing(log_text: str) -> tuple[str, int | None]:
    """Return only complete clean-pass regions from a two-pass L3 capture.

    The sentinels are emitted by the parent around the blocking timing
    ``Worker.run()`` call, so all child-process STRACE records between them
    belong to the dep-gen-disabled pass. Substring splitting is deliberate:
    concurrent processes can concatenate otherwise complete records onto one
    physical line.

    Returns ``(log_text, None)`` when no two-pass sentinel is present, otherwise
    returns the filtered text and number of complete timing regions. Malformed
    or incomplete sentinels return ``("", -1)`` so callers report no sample
    instead of silently including graph-pass timing.
    """
    parts = log_text.split(_L3_SWIMLANE_TIMING_BEGIN)
    if len(parts) == 1:
        if _L3_SWIMLANE_PASS_PREFIX in log_text:
            return "", -1
        return log_text, None
    if _L3_SWIMLANE_TIMING_END in parts[0]:
        return "", -1

    timing_regions: list[str] = []
    for part in parts[1:]:
        timing, end, outside = part.partition(_L3_SWIMLANE_TIMING_END)
        if not end or _L3_SWIMLANE_TIMING_END in outside:
            return "", -1
        timing_regions.append(timing)
    return "\n".join(timing_regions), len(timing_regions)


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
    name is resolved per :func:`_span_names` (``chip.run`` on current simpler).

    L3 (``distributed=True``): prepared swimlane pass sentinels first restrict
    the input to complete dep-gen-disabled timing regions, when present; then
    :func:`_parse_l3_stats` folds the per-rank chip-child markers into per-round
    aggregates (see that function).
    """
    # ``simpler`` is an optional runtime-provided package: present on devices
    # where the runtime is installed, absent on the lint / unit-test host. The
    # import is resolved lazily at call time; pyright cannot see it in the lint
    # env, and unit tests skip the parse path when it is not installed.
    from simpler_setup.tools import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        strace_timing as _strace_timing,
    )

    timing_blocks: int | None = None
    if distributed:
        log_text, timing_blocks = _extract_l3_swimlane_timing(log_text)

    names = _span_names()
    stats = BenchmarkStats(rounds=rounds, warmup=warmup)
    if timing_blocks is not None and timing_blocks != warmup + rounds:
        stats.fallback_flattened = True
        return stats
    # L3 forks one chip worker per rank, all sharing the capture fd, so two
    # complete records can land on one physical line. Normalize those records
    # before parsing: this is harmless with the current ``finditer`` parser and
    # preserves compatibility with older parsers that consumed one per line.
    lines = log_text.replace("[STRACE]", "\n[STRACE]").splitlines()
    spans = _strace_timing.parse_spans(lines)
    # Drop the families that carry no invocation id, so none of them forges a
    # lane in ``group_invocations`` below. The local prefix filter runs
    # unconditionally rather than only as a fallback: simpler's own filter
    # deliberately *keeps* any family it does not recognize (dropping unfamiliar
    # names silently is how ``chip.prewarm.build`` once vanished from its
    # tables), so a current simpler passes the pre-#1893 ``l3.`` spelling
    # through. Then apply simpler's filter too, since it is the authority on the
    # families its own generation emits — #1877 renamed it ``legacy_spans`` ->
    # ``invocation_spans``, so try both names before giving up on it.
    spans = [span for span in spans if not span.name.startswith(_NON_INVOCATION_PREFIXES)]
    span_filter = getattr(_strace_timing, "invocation_spans", None) or getattr(
        _strace_timing, "legacy_spans", None
    )
    if span_filter is not None:
        spans = span_filter(spans)
    invocations = _strace_timing.group_invocations(spans)
    if not invocations:
        return stats

    if distributed:
        return _parse_l3_stats(invocations, stats, rounds=rounds, warmup=warmup)

    # Busiest hid bucket = our register-once callable (one invocation per launch);
    # bucket_by_hid orders each bucket by inv, so warmup drops in dispatch order.
    busiest = max(_strace_timing.bucket_by_hid(invocations).values(), key=len)
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
    persistent: bool = False,
    reset_persistent_windows: bool | None = None,
) -> BenchmarkStats:
    """Register *compiled* once and dispatch *rounds* timed launches.

    Dispatches by *compiled* type:

    - **L2** (:class:`~pypto.ir.CompiledProgram`): opens a single
      :class:`~pypto.runtime.ChipWorker` with the capabilities recorded in the
      compiled artifact.
    - **L3** (:class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`):
      opens a :class:`~pypto.runtime.distributed_runner.DistributedWorker` via
      ``compiled.prepare()``.

    Either way it registers *compiled* once, then loops the bound handle so each
    launch only re-pays argument coercion + dispatch (not register/load). The
    on-NPU ``device_wall_us`` is measured between the orchestrator's
    ``orch_start`` / ``orch_end`` and is unaffected by the per-launch host-side
    arg building.

    Timing is read from the runtime's ``[STRACE]`` stderr markers (simpler PR
    #1177): this sets the runtime log level to ``timing`` for the worker's
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
            with ``.share_memory_()`` and reused in place). This helper creates
            its own prepared Worker, so it cannot accept a DeviceTensor owned by
            another Worker; benchmark resident tensors with an explicit
            ``compiled.prepare()`` dispatch loop.
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
            (``aicpu_thread_num``); pass this
            *or* *platform*/*device_id*, not both. L3: forwarded per dispatch for
            ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
            ``ring_dep_pool``); ``None`` reuses the prepared baseline.
        persistent: L3 only. Reuse retained CommDomains across all warmup and
            measured dispatches while fencing each launch with ``Worker.run``.
        reset_persistent_windows: L3 persistent mode only. Restore retained
            windows to zero before reuse. ``None`` (the default) enables reset
            in persistent mode. Set to ``False`` only when the benchmarked
            program manually clears or otherwise manages all reused
            communication-buffer state.

    Returns:
        A :class:`BenchmarkStats` with the per-round ``device_wall_us`` /
        ``host_wall_us`` samples and aggregate helpers. For L3 the samples are
        per-round maxima across ranks; summaries are derived on demand via
        :meth:`BenchmarkStats.per_dispatch` (per ``(pid, slot)`` dispatch — nothing
        summed), :meth:`BenchmarkStats.per_rank` (``device`` / ``host`` /
        ``effective``, that rank's dispatches summed per round) and
        :meth:`BenchmarkStats.per_round` (those plus ``union``), and the
        ``round → rank → [dispatch]`` grid is in
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
        A rank that dispatches more than once per round is therefore summed in the
        headline; use :meth:`BenchmarkStats.per_dispatch` (and the per-dispatch
        mean trees) to keep those dispatches apart.
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
    elif persistent or reset_persistent_windows is not None:
        raise ValueError(
            "benchmark(): persistent/reset_persistent_windows apply only to an L3 DistributedCompiledProgram"
        )
    elif config is not None and (platform is not None or device_id is not None):
        raise ValueError("benchmark(): pass either config=... or platform=/device_id=, not both")

    # The C++ host logger that prints the ``[STRACE]`` markers is seeded from the
    # simpler Python logger snapshot at worker ``init`` (and inherited by the L3
    # fork), so set the level to ``timing`` before constructing the worker. Restore afterward.
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
                        # Pass the dispatch config so prepare() prewarms the ring
                        # sizing the loop below actually dispatches with.
                        with compiled.prepare(
                            config,
                            persistent=persistent,
                            reset_persistent_windows=reset_persistent_windows,
                        ) as rt:
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
                enable_sdma = bool(compiled.runtime_config.get("enable_sdma", False))
                # L2 runs the chip in-process (no fork), so the parent's fd 2
                # redirect during the loop captures its markers.
                with ChipWorker(
                    rc,
                    runtime=compiled.runtime_name,
                    enable_sdma=enable_sdma,
                ) as worker:
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
            "SIMPLER_HOST_STRACE macro (LOG_TIMING tier); this runtime emitted none. "
            "Rebuild the runtime with SIMPLER_HOST_STRACE enabled to read benchmark timing."
        )
    return stats
