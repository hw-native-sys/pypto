# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end pipeline profiling for PyPTO compilation stages.

Records wall-clock timing at stable stage boundaries (parse, passes, codegen,
device execution) and outputs structured JSON or a human-readable hierarchical
summary suitable for CI logs and local debugging.

Three opt-in mechanisms are supported:

1. **Environment variable**: ``PYPTO_PIPELINE_PROFILING=1``
2. **Function parameter**: ``ir.compile(..., profiling=True)``
3. **Context manager**::

       from pypto.pipeline_profiling import PipelineProfiler

       with PipelineProfiler() as prof:
           prog = MyProgram          # @pl.program parse is timed
           ir.compile(prog, ...)     # passes + codegen are timed
       print(prof.summary())
"""

import json
import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageRecord:
    """A single timed stage in the compilation pipeline."""

    name: str
    start: float
    end: float = 0.0
    children: list["StageRecord"] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.duration, 6),
            "children": [c.to_dict() for c in self.children],
        }


class PipelineProfiler:
    """Hierarchical wall-clock profiler for the PyPTO compilation pipeline.

    Implements a thread-local stack so that ``PipelineProfiler.current()``
    returns the innermost active profiler (similar to ``PassContext``).
    """

    _local = threading.local()

    def __init__(self) -> None:
        self._root_stages: list[StageRecord] = []
        self._stack: list[StageRecord] = []
        self._previous: PipelineProfiler | None = None
        self._total_start: float = 0.0
        self._total_end: float = 0.0

    # ------------------------------------------------------------------
    # Context manager (thread-local stack)
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineProfiler":
        self._previous = getattr(PipelineProfiler._local, "current", None)
        PipelineProfiler._local.current = self
        self._total_start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._total_end = time.perf_counter()
        PipelineProfiler._local.current = self._previous
        self._previous = None

    @staticmethod
    def current() -> "PipelineProfiler | None":
        """Return the innermost active profiler, or ``None``."""
        return getattr(PipelineProfiler._local, "current", None)

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Generator[StageRecord, None, None]:
        """Record a timed stage. Stages may be nested."""
        record = self._begin_stage(name)
        try:
            yield record
        finally:
            self._end_stage()

    def _begin_stage(self, name: str) -> StageRecord:
        """Open a stage (for split before/after callbacks)."""
        record = StageRecord(name=name, start=time.perf_counter())
        if self._stack:
            self._stack[-1].children.append(record)
        else:
            self._root_stages.append(record)
        self._stack.append(record)
        return record

    def _end_stage(self) -> None:
        """Close the innermost open stage."""
        if self._stack:
            self._stack[-1].end = time.perf_counter()
            self._stack.pop()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def total_seconds(self) -> float:
        if self._total_end > 0:
            return self._total_end - self._total_start
        if self._total_start > 0:
            return time.perf_counter() - self._total_start
        return sum(s.duration for s in self._root_stages)

    def to_dict(self) -> dict[str, Any]:
        """Return profiling data as a nested dict."""
        return {
            "total_seconds": round(self.total_seconds, 6),
            "stages": [s.to_dict() for s in self._root_stages],
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """Serialise profiling data to JSON.

        If *path* is given the JSON is also written to that file.
        """
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
                f.write("\n")
        return text

    def summary(self) -> str:
        """Return a human-readable hierarchical summary."""
        total = self.total_seconds
        lines: list[str] = [
            "PyPTO Pipeline Profile",
            "=" * 22,
            f"Total: {_fmt_time(total)}",
            "",
        ]
        for stage in self._root_stages:
            _format_stage(lines, stage, total, depth=1)
        return "\n".join(lines) + "\n"

    def write_report(self, report_dir: str) -> None:
        """Write summary and JSON to *report_dir*."""
        os.makedirs(report_dir, exist_ok=True)
        txt_path = os.path.join(report_dir, "pipeline_profile.txt")
        json_path = os.path.join(report_dir, "pipeline_profile.json")
        with open(txt_path, "w") as f:
            f.write(self.summary())
        self.to_json(json_path)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_ENV_VAR = "PYPTO_PIPELINE_PROFILING"

# Lazily-created profiler driven by the environment variable.
_env_profiler_lock = threading.Lock()
_env_profiler: PipelineProfiler | None = None


def get_active_profiler() -> PipelineProfiler | None:
    """Return the active profiler if one exists.

    Checks (in order):
    1. An explicit ``PipelineProfiler`` on the thread-local stack.
    2. The ``PYPTO_PIPELINE_PROFILING`` environment variable.

    Returns ``None`` when profiling is not enabled.
    """
    prof = PipelineProfiler.current()
    if prof is not None:
        return prof

    if os.environ.get(_ENV_VAR, "").strip() in ("1", "true", "yes"):
        global _env_profiler  # noqa: PLW0603
        if _env_profiler is None:
            with _env_profiler_lock:
                if _env_profiler is None:
                    _env_profiler = PipelineProfiler()
                    _env_profiler._total_start = time.perf_counter()
                    PipelineProfiler._local.current = _env_profiler
        return _env_profiler

    return None


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.3f}s"


def _format_stage(
    lines: list[str],
    stage: StageRecord,
    total: float,
    depth: int,
) -> None:
    indent = "  " * depth
    dur = _fmt_time(stage.duration)
    pct = f"({stage.duration / total * 100:5.1f}%)" if total > 0 else ""
    lines.append(f"{indent}{stage.name:<28s} {dur:>10s}  {pct}")
    for child in stage.children:
        _format_stage(lines, child, total, depth + 1)
