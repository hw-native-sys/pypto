#!/usr/bin/env python3
# Copyright (c) PyPTO contributors.
"""Non-failing performance-regression guard for CI model runs.

Reads a captured run log (stdout of a `--enable-l2-swimlane` model run), extracts the
device makespan that the swimlane converter prints, and compares it against a committed
baseline. A regression beyond the threshold (or missing perf data) is reported as a
GitHub warning annotation + a Step Summary row and signalled with a non-zero exit code.

The non-zero exit is intentional: paired with ``continue-on-error: true`` on the CI step
it renders the step yellow ⚠ while keeping the job green. The guard never asserts
correctness — it only reports performance.

Metric source (printed by runtime/simpler_setup/tools/swimlane_converter.py):

    Total Test Time: 907.88 us (from earliest dispatch to latest finish)   # makespan_us
    TOTAL                   579       22672.70         33239.52             # exec / latency
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Earliest-dispatch -> latest-finish device wall time (the primary metric).
_MAKESPAN_RE = re.compile(r"Total Test Time:\s*([0-9.]+)\s*us")
# `TOTAL <count> <exec_us> <latency_us>` summary row (secondary, reported only).
_TOTAL_RE = re.compile(r"^TOTAL\s+\d+\s+([0-9.]+)\s+([0-9.]+)", re.MULTILINE)


def _parse_log(path: Path) -> dict[str, float]:
    """Extract perf numbers from a captured run log.

    Returns a dict with ``makespan_us`` plus optional ``exec_us`` / ``latency_us``.
    Raises ValueError if the makespan line is absent (no usable perf signal).
    """
    text = path.read_text(errors="replace") if path.exists() else ""
    makespan_match = _MAKESPAN_RE.search(text)
    if makespan_match is None:
        raise ValueError(
            f"no 'Total Test Time: <X> us' line found in {path} "
            "(perf run produced no usable number)"
        )
    metrics = {"makespan_us": float(makespan_match.group(1))}
    total_match = _TOTAL_RE.search(text)
    if total_match is not None:
        metrics["exec_us"] = float(total_match.group(1))
        metrics["latency_us"] = float(total_match.group(2))
    return metrics


def _load_baseline(path: Path) -> dict | None:
    """Load the baseline JSON. Returns None when absent or unseeded (value is null)."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("value") is None:
        return None
    return data


def _summary(line: str) -> None:
    """Append a line to the GitHub Step Summary (no-op when not on GitHub)."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _warning(title: str, message: str) -> None:
    """Emit a GitHub warning annotation (renders yellow, never fails the job)."""
    # Annotation messages cannot span lines; collapse newlines.
    flat = message.replace("\n", " ")
    print(f"::warning title={title}::{flat}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="captured run log to parse for the perf number")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="baseline JSON with metric/value/threshold_pct")
    parser.add_argument("--name", type=str, default="model",
                        help="display name for messages and the summary")
    parser.add_argument("--metric", type=str, default="makespan_us",
                        help="which parsed metric to compare against the baseline")
    args = parser.parse_args()

    # 1. Parse the captured log.
    try:
        metrics = _parse_log(args.log)
    except ValueError as exc:
        _warning(f"{args.name} perf: no data", str(exc))
        _summary(f"### ⚠️ {args.name} performance guard\n\nNo perf data: {exc}")
        return 1

    measured_line = ", ".join(f"{k}={v:.2f}" for k, v in metrics.items())
    print(f"[perf-guard] {args.name} measured: {measured_line}")

    if args.metric not in metrics:
        _warning(f"{args.name} perf: missing metric",
                 f"metric '{args.metric}' not present in parsed log; got {measured_line}")
        _summary(f"### ⚠️ {args.name} performance guard\n\n"
                 f"Metric `{args.metric}` not found. Measured: {measured_line}")
        return 1
    measured = metrics[args.metric]

    # 2. Load the baseline (None => unseeded bootstrap, stays green).
    baseline = _load_baseline(args.baseline)
    if baseline is None:
        print(f"[perf-guard] no seeded baseline at {args.baseline} — reporting only")
        _summary(
            f"### {args.name} performance guard (baseline not seeded)\n\n"
            f"- Measured `{args.metric}`: **{measured:.2f}**\n"
            f"- Other metrics: {measured_line}\n\n"
            f"Seed `{args.baseline}` `value` with this number to enable regression "
            f"comparison."
        )
        return 0

    base_value = float(baseline["value"])
    threshold_pct = float(baseline.get("threshold_pct", 10))
    delta_pct = (measured - base_value) / base_value * 100.0

    row = (
        f"### {args.name} performance guard\n\n"
        f"| Metric | Measured | Baseline | Δ | Threshold |\n"
        f"| --- | --- | --- | --- | --- |\n"
        f"| `{args.metric}` | {measured:.2f} | {base_value:.2f} | "
        f"{delta_pct:+.1f}% | {threshold_pct:.1f}% |\n"
    )

    # 3. Compare. Regression (slower => larger) beyond threshold => yellow ⚠.
    if delta_pct > threshold_pct:
        _warning(
            f"{args.name} perf regression",
            f"{args.metric} {measured:.2f}us vs baseline {base_value:.2f}us "
            f"({delta_pct:+.1f}%, threshold {threshold_pct:.1f}%)",
        )
        _summary(row + "\n⚠️ **Regression detected** — exceeds threshold.")
        return 1

    _summary(row + "\n✅ Within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
