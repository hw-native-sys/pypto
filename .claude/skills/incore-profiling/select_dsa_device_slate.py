# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select an inclusive, non-frozen device-screen slate from a DSA model screen."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ARMS = ("geometry_ff", "cypress", "dsa_rp_cg")
_MIXED_HALF = re.compile(r"_(?:aic|aiv)(?:_\d+)?$")
_NUMBERED_CLONE = re.compile(r"_\d+$")


@dataclass(frozen=True)
class CandidateInputs:
    """Inputs required to construct a broad device-screen slate."""

    separation_rows: list[dict[str, str]]
    screen_rows: list[dict[str, str]]
    invocations: list[dict[str, str]]
    problems_dir: Path
    screen_root: Path
    min_geometry_advantage: int
    forced: dict[str, str]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forced_reasons(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = _read_tsv(path)
    reasons: dict[str, str] = {}
    for row in rows:
        tag = row.get("tag", "")
        reason = row.get("reason", "")
        if not tag or not reason:
            raise ValueError(f"forced candidate row requires tag and reason: {row!r}")
        if tag in reasons:
            raise ValueError(f"forced candidate tag is repeated: {tag}")
        reasons[tag] = reason
    return reasons


def _source_scripts(invocations: list[dict[str, str]]) -> dict[str, set[str]]:
    by_fingerprint: dict[str, set[str]] = defaultdict(set)
    for row in invocations:
        by_fingerprint[row["problem_fingerprint"]].add(row["script"])
    return by_fingerprint


def _source_problem_counts(invocations: list[dict[str, str]]) -> dict[str, int]:
    """Count distinct exported problems per source entry point."""
    by_script: dict[str, set[str]] = defaultdict(set)
    for row in invocations:
        by_script[row["script"]].add(row["problem_fingerprint"])
    return {script: len(fingerprints) for script, fingerprints in by_script.items()}


def _cell_rows(screen_rows: list[dict[str, str]]) -> dict[str, list[dict[str, dict[str, str]]]]:
    cells: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in screen_rows:
        if not row.get("arm"):
            continue
        key = (row["tag"], row["pool_id"], row["capacity_label"])
        if row["arm"] in cells[key]:
            raise ValueError(f"screen table repeats cell arm {(*key, row['arm'])!r}")
        cells[key][row["arm"]] = row
    by_tag: dict[str, list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (tag, unused_pool, unused_capacity), arms in cells.items():
        del unused_pool, unused_capacity
        by_tag[tag].append(arms)
    return by_tag


def _geometry_metrics(cells: list[dict[str, dict[str, str]]]) -> dict[str, Any]:
    best_gap = -1
    best_fraction = 0.0
    best_location = ""
    native_gap = 0
    for arms in cells:
        geometry = arms.get("geometry_ff")
        rp = arms.get("dsa_rp_cg")
        if geometry is None or rp is None or geometry["status"] != "feasible" or rp["status"] != "feasible":
            continue
        geometry_cost = int(geometry["reuse_cost"])
        gap = geometry_cost - int(rp["reuse_cost"])
        fraction = gap / geometry_cost if geometry_cost else 0.0
        if gap > best_gap:
            best_gap = gap
            best_fraction = fraction
            best_location = f"pool={geometry['pool_id']},capacity={geometry['capacity_label']}"
        if geometry["capacity_label"] == "native":
            native_gap = max(native_gap, gap)
    return {
        "max_geometry_cost_advantage": best_gap,
        "max_geometry_fractional_advantage": f"{best_fraction:.6f}",
        "max_geometry_advantage_location": best_location,
        "native_geometry_cost_advantage": native_gap,
    }


def _native_solution_paths(
    screen_root: Path,
    tag: str,
    cells: list[dict[str, dict[str, str]]],
) -> dict[str, str]:
    native = [arms for arms in cells if next(iter(arms.values()))["capacity_label"] == "native"]
    if not native:
        raise ValueError(f"candidate has no native-capacity cell: {tag}")
    for arms in native:
        missing = set(_ARMS) - arms.keys()
        if missing:
            raise ValueError(f"candidate has an incomplete native cell for {tag}: {sorted(missing)}")
    paths: dict[str, str] = {}
    for arm in _ARMS:
        rows = [arms[arm] for arms in native if arm in arms]
        if not rows or any(row["status"] != "feasible" for row in rows):
            raise ValueError(f"candidate lacks a feasible native {arm} placement: {tag}")
        hashes = {row["placement_sha256"] for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"native {arm} placement changes with screened pool for {tag}: {hashes}")
        row = min(rows, key=lambda item: int(item["pool_id"]))
        path = (
            screen_root
            / "raw"
            / tag
            / f"pool-{row['pool_id']}-{row['pool']}"
            / "native"
            / arm
            / "solution.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[f"native_{arm}_solution"] = str(path.relative_to(screen_root))
        paths[f"native_{arm}_solution_sha256"] = _sha256(path)
        paths[f"native_{arm}_placement_sha256"] = next(iter(hashes))
    return paths


def _problem_metadata(
    problem: Path,
    separation: dict[str, str],
) -> dict[str, Any]:
    document = json.loads(problem.read_text(encoding="utf-8"))
    body = document["problem"]
    if document["instance"] != separation["instance"]:
        raise ValueError(f"problem instance disagrees with model screen: {problem}")
    if len(body["buffers"]) != int(separation["buffers"]):
        raise ValueError(f"problem buffer count disagrees with model screen: {problem}")
    penalties = body["cost_model"]["reuse_penalties"]
    if len(penalties) != int(separation["reuse_penalties"]):
        raise ValueError(f"problem penalty count disagrees with model screen: {problem}")
    pools = sorted(body["pools"], key=lambda pool: int(pool["id"]))
    structural_body = {
        "buffers": [
            {key: value for key, value in buffer.items() if key != "name"} for buffer in body["buffers"]
        ],
        "constraints": body.get("constraints", {}),
        "cost_model": body.get("cost_model"),
        "objective": body.get("objective"),
        "pools": body["pools"],
    }
    return {
        "problem_sha256": _sha256(problem),
        "structural_class_sha256": hashlib.sha256(
            json.dumps(structural_body, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "pool_count": len(pools),
        "pool_names": ";".join(str(pool["name"]) for pool in pools),
        "native_capacities": ";".join(str(pool["capacity"]) for pool in pools),
    }


def _annotate_structural_classes(candidates: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate["structural_class_sha256"])].append(candidate)
    for members in groups.values():
        representative = min(
            members,
            key=lambda row: (
                bool(row["mixed_group_hint"]),
                int(row["preferred_source_problem_count"]),
                bool(_NUMBERED_CLONE.search(str(row["instance"]))),
                str(row["tag"]),
            ),
        )
        for member in members:
            member["structural_class_size"] = len(members)
            member["structural_class_representative"] = member is representative


def _selection_reasons(
    separation: dict[str, str],
    geometry: dict[str, Any],
    forced_reason: str | None,
    min_geometry_advantage: int,
) -> list[str]:
    reasons: list[str] = []
    if int(separation["rp_feasible_cypress_no_fit_cells"]) > 0:
        reasons.append("cypress_no_fit")
    if int(separation["rp_better_cypress_cells"]) > 0:
        reasons.append("rp_beats_cypress")
    if int(geometry["max_geometry_cost_advantage"]) >= min_geometry_advantage:
        reasons.append(f"geometry_gap_ge_{min_geometry_advantage}")
    if forced_reason is not None:
        reasons.append(f"prior_evidence:{forced_reason}")
    return reasons


def build_candidate_rows(inputs: CandidateInputs) -> list[dict[str, Any]]:
    """Return every model-positive or explicitly forced candidate."""
    separation_by_tag = {row["tag"]: row for row in inputs.separation_rows}
    if len(separation_by_tag) != len(inputs.separation_rows):
        raise ValueError("model-separation table repeats a tag")
    unknown_forced = sorted(inputs.forced.keys() - separation_by_tag.keys())
    if unknown_forced:
        raise ValueError(f"forced tags are absent from the model screen: {unknown_forced}")
    sources = _source_scripts(inputs.invocations)
    source_problem_counts = _source_problem_counts(inputs.invocations)
    cells_by_tag = _cell_rows(inputs.screen_rows)
    candidates: list[dict[str, Any]] = []
    for tag, separation in separation_by_tag.items():
        cells = cells_by_tag.get(tag, [])
        geometry = _geometry_metrics(cells)
        reasons = _selection_reasons(
            separation,
            geometry,
            inputs.forced.get(tag),
            inputs.min_geometry_advantage,
        )
        if not reasons:
            continue
        fingerprint = tag.rsplit("-", 1)[-1]
        scripts = sorted(sources.get(fingerprint, set()))
        if not scripts:
            raise ValueError(f"candidate fingerprint has no source invocation: {tag}")
        preferred_source = min(scripts, key=lambda script: (source_problem_counts[script], script))
        problem = inputs.problems_dir / f"{tag}.dsa.json"
        if not problem.is_file():
            raise FileNotFoundError(problem)
        instance = separation["instance"]
        candidates.append(
            {
                "tag": tag,
                "instance": instance,
                "buffers": int(separation["buffers"]),
                "reuse_penalties": int(separation["reuse_penalties"]),
                "screened_cells": int(separation["screened_cells"]),
                "rp_better_cypress_cells": int(separation["rp_better_cypress_cells"]),
                "rp_equal_cypress_cells": int(separation["rp_equal_cypress_cells"]),
                "rp_worse_cypress_cells": int(separation["rp_worse_cypress_cells"]),
                "cypress_no_fit_cells": int(separation["rp_feasible_cypress_no_fit_cells"]),
                **geometry,
                "selection_reasons": ";".join(reasons),
                "source_scripts": ";".join(scripts),
                "source_count": len(scripts),
                "preferred_source_script": preferred_source,
                "preferred_source_problem_count": source_problem_counts[preferred_source],
                "mixed_group_hint": bool(_MIXED_HALF.search(instance)),
                "measurement_state": "NEEDS_CURRENT_LAUNCH_PREFLIGHT",
                "problem": problem.name,
                **_problem_metadata(problem, separation),
                **_native_solution_paths(inputs.screen_root, tag, cells),
            }
        )
    _annotate_structural_classes(candidates)
    candidates.sort(
        key=lambda row: (
            -int(row["cypress_no_fit_cells"] > 0),
            -int(row["rp_better_cypress_cells"]),
            -int(row["max_geometry_cost_advantage"]),
            -int(row["buffers"]),
            str(row["tag"]),
        )
    )
    return candidates


def _parent_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        for script in str(candidate["source_scripts"]).split(";"):
            grouped[script].append(str(candidate["tag"]))
    return [
        {
            "source_script": script,
            "candidate_count": len(tags),
            "candidate_tags": ";".join(sorted(tags)),
            "measurement_state": "NEEDS_PARENT_AND_DISPATCH_PREFLIGHT",
        }
        for script, tags in sorted(grouped.items())
    ]


def _preferred_parent_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["preferred_source_script"])].append(str(candidate["tag"]))
    return [
        {
            "source_script": script,
            "candidate_count": len(tags),
            "candidate_tags": ";".join(sorted(tags)),
            "measurement_state": "NEEDS_PARENT_AND_DISPATCH_PREFLIGHT",
        }
        for script, tags in sorted(grouped.items())
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--problems-dir", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--forced-tsv", type=Path)
    parser.add_argument("--min-geometry-advantage", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_geometry_advantage < 0:
        raise ValueError("minimum geometry advantage must be non-negative")
    if args.output_root.exists():
        raise ValueError(f"output root already exists: {args.output_root}")
    separation_path = args.screen_root / "model-separation.tsv"
    screen_results_path = args.screen_root / "screen-results.tsv"
    candidates = build_candidate_rows(
        CandidateInputs(
            separation_rows=_read_tsv(separation_path),
            screen_rows=_read_tsv(screen_results_path),
            invocations=_read_tsv(args.invocations),
            problems_dir=args.problems_dir,
            screen_root=args.screen_root,
            min_geometry_advantage=args.min_geometry_advantage,
            forced=_forced_reasons(args.forced_tsv),
        )
    )
    representatives = [row for row in candidates if row["structural_class_representative"]]
    parents = _parent_rows(candidates)
    representative_parents = _preferred_parent_rows(representatives)
    args.output_root.mkdir(parents=True)
    _write_tsv(args.output_root / "candidate-slate.tsv", candidates)
    _write_tsv(args.output_root / "parent-slate.tsv", parents)
    _write_tsv(args.output_root / "representative-slate.tsv", representatives)
    _write_tsv(args.output_root / "representative-parent-slate.tsv", representative_parents)
    (args.output_root / "candidate-slate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_policy": "all_model_positive_or_forced_v1",
                "min_geometry_advantage": args.min_geometry_advantage,
                "frozen": False,
                "input_sha256": {
                    "model_separation": _sha256(separation_path),
                    "screen_results": _sha256(screen_results_path),
                    "invocations": _sha256(args.invocations),
                    "forced": _sha256(args.forced_tsv) if args.forced_tsv is not None else None,
                },
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"SELECTED candidates={len(candidates)} structural_classes={len(representatives)} "
        f"source_programs={len(parents)} frozen=false"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
