# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze a broad DSA device panel and solve the three device-study arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SELECTION_CLASSES = {"historical_winner", "broad_inventory"}
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("geometry_ff", ("--solver", "geometry-first-fit")),
    ("cypress", ("--solver", "cypress-relaxation")),
    ("dsa_rp_cg", ("--solver", "canonical-greedy", "--seed", "0", "--restarts", "8")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_file_stem(instance: str) -> str:
    """Mirror PyPTO's replay filename normalization."""
    encoded = instance.encode("utf-8")
    suffix = "".join(
        chr(value)
        if (
            ord("0") <= value <= ord("9")
            or ord("A") <= value <= ord("Z")
            or ord("a") <= value <= ord("z")
            or value in b"-_."
        )
        else f"_{value:02x}"
        for value in encoded
    )
    return f"pypto_{suffix or 'unnamed'}"


def replay_solution_filename(instance: str) -> str:
    """Return the exact filename consumed by ``dsa_solution_dir`` replay."""
    return f"{_corpus_file_stem(instance)}.dsa.solution.json"


def load_panel(path: Path, *, enforce_protocol: bool = True) -> dict[str, Any]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    if panel.get("schema_version") != 1:
        raise ValueError("device panel must use schema_version 1")
    recognizer = panel.get("recognizer")
    if (
        not isinstance(recognizer, dict)
        or not recognizer.get("policy")
        or not recognizer.get("source_sha256")
    ):
        raise ValueError("device panel must freeze recognizer policy and source_sha256")
    kernels = panel.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("device panel must contain a non-empty kernels list")

    tags: set[str] = set()
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            raise ValueError(f"kernel {index} must be an object")
        missing = sorted({"tag", "program", "kernel", "selection_class", "problem"} - set(kernel))
        if missing:
            raise ValueError(f"kernel {index} is missing fields: {missing}")
        tag = str(kernel["tag"])
        if tag in tags:
            raise ValueError(f"duplicate kernel tag: {tag}")
        tags.add(tag)
        selection_class = str(kernel["selection_class"])
        if selection_class not in SELECTION_CLASSES:
            raise ValueError(f"kernel {tag} has unknown selection_class: {selection_class}")

    if enforce_protocol:
        if panel.get("selection_policy") != "all_current_eligible_plus_historical_winners_v1":
            raise ValueError("device panel must freeze the broad-screen selection_policy")
        if not any(kernel["selection_class"] == "historical_winner" for kernel in kernels):
            raise ValueError("device panel must retain at least one historical-winner attempt")
    return panel


def freeze_panel(panel_path: Path, panel: dict[str, Any]) -> dict[str, Any]:
    frozen = json.loads(json.dumps(panel))
    frozen["panel_source_sha256"] = _sha256(panel_path)
    for kernel in frozen["kernels"]:
        problem = (panel_path.parent / kernel["problem"]).resolve()
        if not problem.is_file():
            raise FileNotFoundError(f"DSA problem not found: {problem}")
        document = json.loads(problem.read_text(encoding="utf-8"))
        kernel["problem"] = str(problem)
        kernel["problem_sha256"] = _sha256(problem)
        kernel["problem_instance"] = document.get("instance")
        kernel["problem_profile"] = document.get("profile")
        kernel["exported_edge_policy"] = document.get("metadata", {}).get("reuse_penalty_recognizer")
        kernel["buffers"] = len(document.get("problem", {}).get("buffers", []))
        kernel["reuse_penalties"] = len(
            document.get("problem", {}).get("cost_model", {}).get("reuse_penalties", [])
        )
        instance = document.get("instance")
        if not isinstance(instance, str):
            raise ValueError(f"kernel {kernel['tag']} problem has no string instance")
        kernel["replay_solution_filename"] = replay_solution_filename(instance)
    return frozen


def solver_command(solver: Path, problem: Path, solution: Path, result: Path, arm: str) -> list[str]:
    arguments = dict(ARMS).get(arm)
    if arguments is None:
        raise ValueError(f"unknown device-panel arm: {arm}")
    return [
        str(solver),
        "--input",
        str(problem),
        *arguments,
        "--solution-output",
        str(solution),
        "--json-output",
        str(result),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dsa-bench", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--allow-nonstandard-panel", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if not args.dsa_bench.is_file():
        parser.error(f"dsa-bench not found: {args.dsa_bench}")
    if args.output_root.exists():
        parser.error(f"output root already exists: {args.output_root}")

    panel = load_panel(args.panel, enforce_protocol=not args.allow_nonstandard_panel)
    frozen = freeze_panel(args.panel, panel)
    args.output_root.mkdir(parents=True)
    frozen_path = args.output_root / "panel-frozen.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, set[str]] = {kernel["tag"]: set() for kernel in frozen["kernels"]}
    for kernel in frozen["kernels"]:
        tag = kernel["tag"]
        problem = Path(kernel["problem"])
        arm_root = args.output_root / "solutions" / tag
        arm_root.mkdir(parents=True)
        for arm, _ in ARMS:
            replay_dir = arm_root / arm
            replay_dir.mkdir()
            solution = replay_dir / kernel["replay_solution_filename"]
            result_path = replay_dir / "solver-result.json"
            completed = subprocess.run(
                solver_command(args.dsa_bench, problem, solution, result_path, arm),
                text=True,
                capture_output=True,
                check=False,
                timeout=args.timeout,
            )
            (replay_dir / "solver-stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{tag}/{arm} failed with exit code {completed.returncode}:\n{completed.stderr}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "feasible" or int(result.get("capacity_overflow", -1)) != 0:
                raise RuntimeError(
                    f"{tag}/{arm} did not produce a capacity-feasible placement: "
                    f"status={result.get('status')!r}, "
                    f"capacity_overflow={result.get('capacity_overflow')!r}"
                )
            if not solution.is_file():
                raise RuntimeError(f"{tag}/{arm} did not write replay solution {solution}")
            solution_document = json.loads(solution.read_text(encoding="utf-8"))
            fingerprint = str(solution_document["problem_fingerprint"])
            fingerprints[tag].add(fingerprint)
            solution_sha = _sha256(solution)
            placements = sorted(
                solution_document["placements"], key=lambda placement: int(placement["buffer"])
            )
            placement_sha = hashlib.sha256(
                json.dumps(placements, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            rows.append(
                {
                    "tag": tag,
                    "arm": arm,
                    "status": result["status"],
                    "capacity_overflow": result.get("capacity_overflow", ""),
                    "reuse_cost": result.get("reuse_cost", ""),
                    "total_peak": result.get("total_peak", ""),
                    "max_peak": result.get("peak", ""),
                    "runtime_us": result.get("runtime_us", ""),
                    "problem_fingerprint": fingerprint,
                    "solution_sha256": solution_sha,
                    "placement_sha256": placement_sha,
                    "solution": str(solution),
                    "replay_dir": str(replay_dir),
                }
            )

    mismatched = [tag for tag, values in fingerprints.items() if len(values) > 1]
    if mismatched:
        raise RuntimeError(f"solver arms produced mismatched problem fingerprints: {mismatched}")
    missing = [tag for tag, values in fingerprints.items() if len(values) != 1]
    if missing:
        raise RuntimeError(f"solver arms did not produce one common problem fingerprint: {missing}")
    columns = list(rows[0])
    with (args.output_root / "solver-results.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared {len(frozen['kernels'])} kernels x {len(ARMS)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
