# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Aggregate frozen-panel multi-arm reports without averaging devices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_ARMS = ("geometry_ff", "cypress", "dsa_rp_cg")


def _report_argument(value: str) -> tuple[str, str, Path]:
    identity, separator, raw_path = value.partition("=")
    tag, device_separator, device = identity.rpartition("@")
    if not separator or not device_separator or not tag or not device or not raw_path:
        raise argparse.ArgumentTypeError("expected TAG@DEVICE=REPORT.json")
    return tag, device, Path(raw_path)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def load_reports(
    values: list[tuple[str, str, Path]],
) -> dict[tuple[str, str], dict[str, Any]]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for tag, device, path in values:
        identity = (tag, device)
        if identity in reports:
            raise ValueError(f"duplicate report identity: {tag}@{device}")
        report = json.loads(path.read_text(encoding="utf-8"))
        variants = report.get("summary", {}).get("variants", {})
        if set(variants) != set(EXPECTED_ARMS):
            raise ValueError(f"{tag}@{device} has arms {sorted(variants)}, expected {list(EXPECTED_ARMS)}")
        reports[identity] = report
    if not reports:
        raise ValueError("at least one timing report is required")
    return reports


def summarize_panel(
    reports: dict[tuple[str, str], dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    kernel_rows: list[dict[str, Any]] = []
    ratios: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (tag, device), report in sorted(reports.items()):
        variants = report["summary"]["variants"]
        medians = {arm: float(variants[arm]["median_us"]) for arm in EXPECTED_ARMS}
        for arm in EXPECTED_ARMS:
            kernel_rows.append(
                {
                    "tag": tag,
                    "device": device,
                    "arm": arm,
                    "median_us": medians[arm],
                }
            )
        for reference_index, reference in enumerate(EXPECTED_ARMS):
            for candidate in EXPECTED_ARMS[reference_index + 1 :]:
                ratios[(device, reference, candidate)].append(medians[candidate] / medians[reference])

    panel_rows: list[dict[str, Any]] = []
    for (device, reference, candidate), values in sorted(ratios.items()):
        rng = random.Random(f"{seed}:{device}:{reference}:{candidate}")
        bootstrapped = [
            _geometric_mean([rng.choice(values) for _ in values]) for _ in range(bootstrap_samples)
        ]
        ratio = _geometric_mean(values)
        panel_rows.append(
            {
                "device": device,
                "reference": reference,
                "candidate": candidate,
                "kernels": len(values),
                "geomean_ratio": ratio,
                "geomean_delta_percent": 100.0 * (ratio - 1.0),
                "bootstrap_95_low_percent": 100.0 * (_percentile(bootstrapped, 0.025) - 1.0),
                "bootstrap_95_high_percent": 100.0 * (_percentile(bootstrapped, 0.975) - 1.0),
            }
        )
    return kernel_rows, panel_rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=_report_argument, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)
    reports = load_reports(args.report)
    kernel_rows, panel_rows = summarize_panel(reports, bootstrap_samples=args.bootstrap_samples, seed=0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.output_root / "kernel-latency.tsv", kernel_rows)
    _write_tsv(args.output_root / "panel-latency.tsv", panel_rows)
    print(json.dumps(panel_rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
