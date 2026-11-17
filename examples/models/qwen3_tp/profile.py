# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Summarize measured-only native spans after a profiling allocation finishes."""

import argparse
import json
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SPAN = re.compile(r"\[mono_ns=(\d+)\].*?\[STRACE\].*?pid=(\d+) .*?name=(\S+) ts=\d+ dur=(\d+)")


def read_spans(log_path: Path) -> Iterator[tuple[int, int, str, int]]:
    paths = sorted(log_path.glob("native-rank*.log")) if log_path.is_dir() else [log_path]
    if not paths:
        raise ValueError("No native trace files found")
    for path in paths:
        with path.open() as stream:
            for line in stream:
                match = SPAN.search(line)
                if match:
                    stamp, pid, name, duration = match.groups()
                    yield int(stamp), int(pid), name, int(duration)


def analyze(timings: dict[str, Any], log_path: Path) -> dict[str, Any]:
    windows: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    phases: dict[str, Any] = {}
    for row in timings["rows"]:
        phase = row["phase"]
        result = phases.setdefault(
            phase, {"rank_samples": 0, "forward_ms": 0.0, "collective_ms": 0.0, "dispatch": {}, "native": {}}
        )
        for i, profile in enumerate(row["rank_profiles"]):
            if profile is None:
                raise ValueError("Requires --profile-dispatch timings")
            windows[profile["pid"]].append((profile["start_ns"], profile["end_ns"], phase))
            result["rank_samples"] += 1
            result["forward_ms"] += row["rank_ms"][i]
            result["collective_ms"] += row["rank_collective_ms"][i]
            for name, value in profile["dispatch"].items():
                entry = result["dispatch"].setdefault(name, {"calls": 0, "ms": 0.0})
                entry["calls"] += value["calls"]
                entry["ms"] += value["ms"]
    for entries in windows.values():
        entries.sort()
    starts = {pid: [entry[0] for entry in entries] for pid, entries in windows.items()}
    for stamp, pid, name, duration in read_spans(log_path):
        if pid not in windows:
            continue
        index = bisect_right(starts[pid], stamp) - 1
        if index < 0 or stamp > windows[pid][index][1]:
            continue
        result = phases[windows[pid][index][2]]
        entry = result["native"].setdefault(name, {"calls": 0, "ms": 0.0})
        entry["calls"] += 1
        entry["ms"] += duration / 1e6
    for result in phases.values():
        count = result["rank_samples"]
        result["forward_ms"] /= count
        result["collective_ms"] /= count
        for group in ("dispatch", "native"):
            for entry in result[group].values():
                entry["calls"] /= count
                entry["ms"] /= count
        dispatch_ms = sum(entry["ms"] for entry in result["dispatch"].values())
        result["dispatch_total_ms"] = dispatch_ms
        result["outside_dispatch_ms"] = result["forward_ms"] - dispatch_ms
        result["collective_barrier_and_python_ms"] = result["collective_ms"] - sum(
            result["dispatch"].get(name, {"ms": 0})["ms"] for name in ("publish_chip", "allreduce_chip")
        )
        native = result["native"]
        if "chip.run" not in native or "chip.run.runner_run.device_wall" not in native:
            raise ValueError("Missing measured native chip/device spans")
        expected_calls = sum(entry["calls"] for entry in result["dispatch"].values())
        if native["chip.run"]["calls"] != expected_calls:
            raise ValueError(
                f"Native dispatch count {native['chip.run']['calls']} "
                f"does not match Python count {expected_calls}"
            )
        if native["chip.run.runner_run.device_wall"]["calls"] != expected_calls:
            raise ValueError("Device-window count does not match Python dispatch count")
        result["python_dispatch_overhead_ms"] = dispatch_ms - native["chip.run"]["ms"]
        result["native_outside_device_window_ms"] = (
            native["chip.run"]["ms"] - native["chip.run.runner_run.device_wall"]["ms"]
        )
    return phases


def analyze_setup(timings: dict[str, Any]) -> dict[str, Any]:
    """Average per-rank costs without treating nested totals as additive stages."""
    result: dict[str, Any] = {}
    if not timings.get("profile_setup"):
        raise ValueError("Requires --profile-setup timings")
    for row in timings["rows"]:
        phase = result.setdefault(
            row["phase"],
            {
                "rank_samples": 0,
                "forward_ms": 0.0,
                "dispatch_ms": 0.0,
                "dispatch_calls": 0,
                "collective_ms": 0.0,
                "stages_ms": {},
            },
        )
        for rank, profile in enumerate(row["rank_profiles"]):
            phase["rank_samples"] += 1
            phase["forward_ms"] += row["rank_ms"][rank]
            phase["collective_ms"] += row["rank_collective_ms"][rank]
            for stats in profile["dispatch"].values():
                phase["dispatch_ms"] += stats["ms"]
                phase["dispatch_calls"] += stats["calls"]
            for stage, value in profile["setup_ms"].items():
                if value < 0:
                    raise ValueError("Negative setup duration")
                phase["stages_ms"][stage] = phase["stages_ms"].get(stage, 0.0) + value
    for phase in result.values():
        count = phase["rank_samples"]
        for key in ("forward_ms", "dispatch_ms", "dispatch_calls", "collective_ms"):
            phase[key] /= count
        phase["stages_ms"] = {key: value / count for key, value in phase["stages_ms"].items()}
        phase["unattributed_dispatch_ms"] = phase["dispatch_ms"] - sum(phase["stages_ms"].values())
        phase["outside_dispatch_ms"] = phase["forward_ms"] - phase["dispatch_ms"]
        if phase["unattributed_dispatch_ms"] < -1e-6 or phase["outside_dispatch_ms"] < -1e-6:
            raise ValueError("Stage durations exceed enclosing interval")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timings = json.loads(args.timings.read_text())
    if args.setup_only:
        report = analyze_setup(timings)
    else:
        if args.log is None:
            parser.error("Native analysis requires --log")
        report = analyze(timings, args.log)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
