# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Screen a DSA-RP corpus across algorithms and per-memory-space capacities."""

import argparse
import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

ARM_GEOMETRY = "geometry_ff"
ARM_CYPRESS = "cypress"
ARM_DSA_RP = "dsa_rp_cg"
CAPACITY_LABELS = {
    Fraction(0): "tight",
    Fraction(1, 4): "q1",
    Fraction(1, 2): "half",
    Fraction(1): "native",
}


@dataclass(frozen=True)
class CypressVariant:
    order: str
    seed: int = 0

    @property
    def label(self) -> str:
        return f"{self.order}_seed{self.seed}" if self.order == "random" else self.order


@dataclass(frozen=True)
class ScreenConfig:
    binary: Path
    variants: tuple[CypressVariant, ...]
    restarts: int
    timeout: int


@dataclass(frozen=True)
class CapacityCell:
    tag: str
    pool: dict[str, Any]
    fraction: Fraction
    capacity: int
    native: int
    first_fit_peak: int
    penalty_count: int


def parse_fractions(text: str) -> tuple[Fraction, ...]:
    values = tuple(Fraction(item.strip()) for item in text.split(",") if item.strip())
    if not values or values[0] != 0 or values[-1] != 1:
        raise ValueError("capacity fractions must include 0 and 1")
    if tuple(sorted(set(values))) != values or any(value < 0 or value > 1 for value in values):
        raise ValueError("capacity fractions must be unique, increasing, and lie in [0, 1]")
    return values


def capacity_grid(first_fit_peak: int, native: int, fractions: tuple[Fraction, ...]) -> tuple[int, ...]:
    if first_fit_peak > native:
        raise ValueError(f"geometry-first-fit peak {first_fit_peak} exceeds capacity {native}")
    span = native - first_fit_peak
    capacities = []
    for fraction in fractions:
        numerator = span * fraction.numerator
        capacities.append(first_fit_peak + (numerator + fraction.denominator // 2) // fraction.denominator)
    capacities[0] = first_fit_peak
    capacities[-1] = native
    return tuple(capacities)


def _placement_map(solution: dict[str, Any]) -> dict[int, dict[str, int]]:
    placements: dict[int, dict[str, int]] = {}
    for placement in solution.get("placements", []):
        buffer = int(placement["buffer"])
        if buffer in placements:
            raise ValueError(f"solution repeats buffer {buffer}")
        placements[buffer] = {
            "pool": int(placement["pool"]),
            "offset": int(placement["offset"]),
        }
    return placements


def pool_peak(document: dict[str, Any], solution: dict[str, Any], pool_id: int) -> int:
    body = document["problem"]
    placements = _placement_map(solution)
    peak = 0
    pool = next(pool for pool in body["pools"] if int(pool["id"]) == pool_id)
    for reserved in pool.get("reserved_ranges", []):
        peak = max(peak, int(reserved["end"]))
    for buffer in body["buffers"]:
        placement = placements.get(int(buffer["id"]))
        if placement is not None and placement["pool"] == pool_id:
            peak = max(peak, placement["offset"] + int(buffer["size"]))
    return peak


def penalty_counts_by_pool(document: dict[str, Any], solution: dict[str, Any]) -> dict[int, int]:
    placements = _placement_map(solution)
    counts: dict[int, int] = {}
    penalties = (document["problem"].get("cost_model") or {}).get("reuse_penalties", [])
    for penalty in penalties:
        first = placements.get(int(penalty["first"]))
        second = placements.get(int(penalty["second"]))
        if first is not None and second is not None and first["pool"] == second["pool"]:
            counts[first["pool"]] = counts.get(first["pool"], 0) + 1
    return counts


def with_pool_capacity(document: dict[str, Any], pool_id: int, capacity: int) -> dict[str, Any]:
    derived = copy.deepcopy(document)
    changed = 0
    for pool in derived["problem"]["pools"]:
        if int(pool["id"]) == pool_id:
            pool["capacity"] = capacity
            changed += 1
    if changed != 1:
        raise ValueError(f"expected exactly one pool {pool_id}, changed {changed}")
    derived.setdefault("metadata", {})["capacity_screen_override"] = f"pool={pool_id},capacity={capacity}"
    return derived


def cypress_portfolio_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Choose Cypress without consulting penalty weights or measured time."""
    metrics = record.get("solver_metrics", {})
    return (
        0 if record.get("status") == "feasible" else 1,
        int(metrics.get("actual_alias_pairs", 2**63 - 1)),
        int(metrics.get("relaxed_edges", 2**63 - 1)),
        int(record.get("total_peak", 2**63 - 1)),
        str(record.get("cypress_order", "")),
        int(record.get("cypress_seed", 0)),
    )


def _placement_sha(solution: dict[str, Any], pool_id: int | None = None) -> str:
    placements = solution.get("placements", [])
    if pool_id is not None:
        placements = [placement for placement in placements if int(placement["pool"]) == pool_id]
    placements = [
        {
            "buffer": int(placement["buffer"]),
            "pool": int(placement["pool"]),
            "offset": int(placement["offset"]),
        }
        for placement in placements
    ]
    canonical = json.dumps(
        sorted(placements, key=lambda placement: int(placement["buffer"])),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _run_solver(
    binary: Path,
    problem: Path,
    output: Path,
    solver: str,
    *,
    seed: int = 0,
    restarts: int = 0,
    cypress_order: str | None = None,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "solver-result.json"
    solution_path = output / "solution.json"
    command = [
        str(binary),
        "--input",
        str(problem),
        "--solver",
        solver,
        "--seed",
        str(seed),
        "--json-output",
        str(result_path),
        "--solution-output",
        str(solution_path),
    ]
    if restarts:
        command.extend(["--restarts", str(restarts)])
    if cypress_order is not None:
        command.extend(["--cypress-order", cypress_order])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        (output / "stdout.log").write_text(stdout, encoding="utf-8")
        (output / "stderr.log").write_text(stderr, encoding="utf-8")
        return ({"status": "timeout", "returncode": "", "stderr": stderr.strip()}, None)
    if not result_path.is_file() and _is_best_effort_serialization_failure(completed.stderr):
        (output / "solution-serialization.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "solution-serialization.stderr.log").write_text(completed.stderr, encoding="utf-8")
        solution_path.unlink(missing_ok=True)
        retry_command = command[:]
        flag = retry_command.index("--solution-output")
        del retry_command[flag : flag + 2]
        try:
            completed = subprocess.run(
                retry_command, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            (output / "stdout.log").write_text(stdout, encoding="utf-8")
            (output / "stderr.log").write_text(stderr, encoding="utf-8")
            return ({"status": "timeout", "returncode": "", "stderr": stderr.strip()}, None)
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if not result_path.is_file():
        return (
            {
                "status": "tool_error",
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            },
            None,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["returncode"] = completed.returncode
    solution = json.loads(solution_path.read_text(encoding="utf-8")) if solution_path.is_file() else None
    return result, solution


def _is_best_effort_serialization_failure(stderr: str) -> bool:
    return "cannot serialize invalid structured solution:" in stderr and "exceeds pool capacity" in stderr


def _arm_row(
    cell: CapacityCell,
    *,
    arm: str,
    result: dict[str, Any],
    solution: dict[str, Any] | None,
    cypress_variant: str = "",
) -> dict[str, Any]:
    metrics = result.get("solver_metrics", {})
    return {
        "tag": cell.tag,
        "pool_id": int(cell.pool["id"]),
        "pool": cell.pool["name"],
        "capacity_label": CAPACITY_LABELS.get(cell.fraction, str(cell.fraction)),
        "capacity_fraction": str(cell.fraction),
        "capacity": cell.capacity,
        "native_capacity": cell.native,
        "geometry_ff_peak": cell.first_fit_peak,
        "pool_penalties": cell.penalty_count,
        "arm": arm,
        "status": result.get("status", "tool_error"),
        "capacity_overflow": result.get("capacity_overflow", ""),
        "reuse_cost": result.get("reuse_cost", ""),
        "total_peak": result.get("total_peak", ""),
        "max_peak": result.get("peak", ""),
        "runtime_us": result.get("runtime_us", ""),
        "selected_pool_peak": "" if solution is None else pool_peak_from_sizes(solution, cell.pool),
        "placement_sha256": "" if solution is None else _placement_sha(solution),
        "pool_placement_sha256": "" if solution is None else _placement_sha(solution, int(cell.pool["id"])),
        "cypress_variant": cypress_variant,
        "cypress_auxiliary_edges": metrics.get("auxiliary_edges", ""),
        "cypress_relaxed_edges": metrics.get("relaxed_edges", ""),
        "cypress_actual_alias_pairs": metrics.get("actual_alias_pairs", ""),
        "cypress_packing_attempts": metrics.get("packing_attempts", ""),
    }


def pool_peak_from_sizes(solution: dict[str, Any], pool: dict[str, Any]) -> int:
    """Read a peak stamped by ``_annotate_solution_sizes``."""
    pool_id = int(pool["id"])
    peak = max((int(item["end"]) for item in pool.get("reserved_ranges", [])), default=0)
    for placement in solution.get("placements", []):
        if int(placement["pool"]) == pool_id:
            peak = max(peak, int(placement["offset"]) + int(placement["_buffer_size"]))
    return peak


def _annotate_solution_sizes(document: dict[str, Any], solution: dict[str, Any] | None) -> None:
    if solution is None:
        return
    sizes = {int(buffer["id"]): int(buffer["size"]) for buffer in document["problem"]["buffers"]}
    for placement in solution.get("placements", []):
        placement["_buffer_size"] = sizes[int(placement["buffer"])]


def _screen_cell(
    document: dict[str, Any],
    cell_root: Path,
    cell: CapacityCell,
    config: ScreenConfig,
) -> list[dict[str, Any]]:
    derived = with_pool_capacity(document, int(cell.pool["id"]), cell.capacity)
    derived_path = cell_root / "derived-problem.json"
    derived_path.write_text(json.dumps(derived, sort_keys=True), encoding="utf-8")

    geometry_result, geometry_solution = _run_solver(
        config.binary,
        derived_path,
        cell_root / ARM_GEOMETRY,
        "geometry-first-fit",
        timeout=config.timeout,
    )
    _annotate_solution_sizes(derived, geometry_solution)
    rows = [
        _arm_row(
            cell,
            arm=ARM_GEOMETRY,
            result=geometry_result,
            solution=geometry_solution,
        )
    ]

    canonical_result, canonical_solution = _run_solver(
        config.binary,
        derived_path,
        cell_root / ARM_DSA_RP,
        "canonical-greedy",
        seed=0,
        restarts=config.restarts,
        timeout=config.timeout,
    )
    _annotate_solution_sizes(derived, canonical_solution)
    rows.append(
        _arm_row(
            cell,
            arm=ARM_DSA_RP,
            result=canonical_result,
            solution=canonical_solution,
        )
    )

    cypress_runs: list[tuple[CypressVariant, dict[str, Any], dict[str, Any] | None]] = []
    for variant in config.variants:
        result, solution = _run_solver(
            config.binary,
            derived_path,
            cell_root / "cypress-variants" / variant.label,
            "cypress-relaxation",
            seed=variant.seed,
            cypress_order=variant.order,
            timeout=config.timeout,
        )
        result["cypress_order"] = variant.order
        result["cypress_seed"] = variant.seed
        _annotate_solution_sizes(derived, solution)
        cypress_runs.append((variant, result, solution))
    variant, cypress_result, cypress_solution = min(
        cypress_runs, key=lambda item: cypress_portfolio_key(item[1])
    )
    selected_root = cell_root / ARM_CYPRESS
    selected_root.mkdir()
    source_root = cell_root / "cypress-variants" / variant.label
    for name in ("solver-result.json", "solution.json", "stdout.log", "stderr.log"):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, selected_root / name)
    rows.append(
        _arm_row(
            cell,
            arm=ARM_CYPRESS,
            result=cypress_result,
            solution=cypress_solution,
            cypress_variant=variant.label,
        )
    )
    derived_path.unlink()
    return rows


def _screen_problem(
    problem_path: Path,
    output_root: Path,
    config: ScreenConfig,
    fractions: tuple[Fraction, ...],
) -> list[dict[str, Any]]:
    tag = problem_path.name.removesuffix(".dsa.json")
    problem_root = output_root / "raw" / tag
    completed_path = problem_root / "problem-summary.json"
    if completed_path.is_file():
        return json.loads(completed_path.read_text(encoding="utf-8"))["rows"]
    if problem_root.exists():
        incomplete_root = output_root / "incomplete"
        incomplete_root.mkdir(exist_ok=True)
        attempt = 1
        archived = incomplete_root / f"{tag}-attempt-{attempt:03d}"
        while archived.exists():
            attempt += 1
            archived = incomplete_root / f"{tag}-attempt-{attempt:03d}"
        problem_root.rename(archived)
    problem_root.mkdir(parents=True)
    document = json.loads(problem_path.read_text(encoding="utf-8"))
    scratch = problem_root / "native-problem.json"
    scratch.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    baseline_result, baseline_solution = _run_solver(
        config.binary,
        scratch,
        problem_root / "baseline",
        "geometry-first-fit",
        timeout=config.timeout,
    )
    if baseline_result.get("status") != "feasible" or baseline_solution is None:
        rows = [{"tag": tag, "status": f"baseline_{baseline_result.get('status', 'tool_error')}"}]
        completed_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        scratch.unlink(missing_ok=True)
        return rows
    penalty_counts = penalty_counts_by_pool(document, baseline_solution)
    pools = {int(pool["id"]): pool for pool in document["problem"]["pools"]}
    rows: list[dict[str, Any]] = []
    for pool_id, penalty_count in sorted(penalty_counts.items()):
        pool = pools[pool_id]
        native = pool.get("capacity")
        if not isinstance(native, int):
            rows.append({"tag": tag, "pool_id": pool_id, "pool": pool["name"], "status": "no_capacity"})
            continue
        first_fit_peak = pool_peak(document, baseline_solution, pool_id)
        capacities = capacity_grid(first_fit_peak, native, fractions)
        for fraction, capacity in zip(fractions, capacities, strict=True):
            label = CAPACITY_LABELS.get(fraction, str(fraction).replace("/", "_"))
            cell_root = problem_root / f"pool-{pool_id}-{pool['name']}" / label
            cell_root.mkdir(parents=True, exist_ok=False)
            rows.extend(
                _screen_cell(
                    document,
                    cell_root,
                    CapacityCell(
                        tag=tag,
                        pool=pool,
                        fraction=fraction,
                        capacity=capacity,
                        native=native,
                        first_fit_peak=first_fit_peak,
                        penalty_count=penalty_count,
                    ),
                    config,
                )
            )
    scratch.unlink(missing_ok=True)
    completed_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return rows


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_root / "screen-results.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_model_separation_rows(
    documents: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate solver-objective separation without interpreting it as latency."""
    by_cell: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if "arm" not in row or "pool_id" not in row or "capacity_fraction" not in row:
            continue
        key = (str(row["tag"]), int(row["pool_id"]), str(row["capacity_fraction"]))
        by_cell.setdefault(key, {})[str(row["arm"])] = row

    summaries: list[dict[str, Any]] = []
    for tag, document in sorted(documents.items()):
        body = document["problem"]
        cells = [(key, arms) for key, arms in by_cell.items() if key[0] == tag]
        counts = {
            "all_arms_feasible_cells": 0,
            "rp_better_geometry_cells": 0,
            "rp_better_cypress_cells": 0,
            "rp_equal_cypress_cells": 0,
            "rp_worse_cypress_cells": 0,
            "rp_feasible_cypress_no_fit_cells": 0,
        }
        best_gap: int | None = None
        best_location = ""
        best_peak_delta = ""
        for (unused_tag, pool_id, fraction), arms in cells:
            del unused_tag
            geometry = arms.get(ARM_GEOMETRY)
            cypress = arms.get(ARM_CYPRESS)
            rp = arms.get(ARM_DSA_RP)
            if rp is None or cypress is None or geometry is None:
                continue
            rp_feasible = rp.get("status") == "feasible"
            cypress_feasible = cypress.get("status") == "feasible"
            geometry_feasible = geometry.get("status") == "feasible"
            if rp_feasible and not cypress_feasible:
                counts["rp_feasible_cypress_no_fit_cells"] += 1
            if not (rp_feasible and cypress_feasible and geometry_feasible):
                continue
            counts["all_arms_feasible_cells"] += 1
            rp_cost = int(rp["reuse_cost"])
            cypress_cost = int(cypress["reuse_cost"])
            geometry_cost = int(geometry["reuse_cost"])
            counts["rp_better_geometry_cells"] += rp_cost < geometry_cost
            counts["rp_better_cypress_cells"] += rp_cost < cypress_cost
            counts["rp_equal_cypress_cells"] += rp_cost == cypress_cost
            counts["rp_worse_cypress_cells"] += rp_cost > cypress_cost
            gap = cypress_cost - rp_cost
            if best_gap is None or gap > best_gap:
                best_gap = gap
                best_location = f"pool={pool_id},fraction={fraction}"
                best_peak_delta = int(rp["total_peak"]) - int(cypress["total_peak"])
        summaries.append(
            {
                "tag": tag,
                "instance": document.get("instance", ""),
                "buffers": len(body["buffers"]),
                "reuse_penalties": len((body.get("cost_model") or {}).get("reuse_penalties", [])),
                "pools": len(body["pools"]),
                "screened_cells": len(cells),
                **counts,
                "max_rp_cost_advantage_vs_cypress": "" if best_gap is None else best_gap,
                "max_advantage_location": best_location,
                "rp_minus_cypress_total_peak_at_max_advantage": best_peak_delta,
            }
        )
    summaries.sort(
        key=lambda row: (
            -int(row["rp_feasible_cypress_no_fit_cells"]),
            -int(row["max_rp_cost_advantage_vs_cypress"])
            if row["max_rp_cost_advantage_vs_cypress"] != ""
            else 2**63,
            -int(row["buffers"]),
            str(row["tag"]),
        )
    )
    return summaries


def _write_model_separation(problems_dir: Path, output_root: Path, rows: list[dict[str, Any]]) -> None:
    documents = {
        path.name.removesuffix(".dsa.json"): json.loads(path.read_text(encoding="utf-8"))
        for path in problems_dir.glob("*.dsa.json")
    }
    summaries = build_model_separation_rows(documents, rows)
    with (output_root / "model-separation.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)


def _write_cypress_variants(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("raw/*/pool-*/*/cypress-variants/*/solver-result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result.get("solver_metrics", {})
        pool_fields = path.parents[3].name.split("-", 2)
        rows.append(
            {
                "tag": path.parents[4].name,
                "pool_id": pool_fields[1],
                "pool": pool_fields[2],
                "capacity_label": path.parents[2].name,
                "variant": path.parents[0].name,
                "status": result.get("status", ""),
                "reuse_cost": result.get("reuse_cost", ""),
                "total_peak": result.get("total_peak", ""),
                "runtime_us": result.get("runtime_us", ""),
                "auxiliary_edges": metrics.get("auxiliary_edges", ""),
                "relaxed_edges": metrics.get("relaxed_edges", ""),
                "actual_alias_pairs": metrics.get("actual_alias_pairs", ""),
                "packing_attempts": metrics.get("packing_attempts", ""),
            }
        )
    if not rows:
        return
    with (output_root / "cypress-variants.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems-dir", type=Path, required=True)
    parser.add_argument("--dsa-bench", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", default="0,1/4,1/2,1")
    parser.add_argument("--canonical-restarts", type=int, default=8)
    parser.add_argument("--cypress-orders", default="stable,reverse,largest-overlap,random")
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if not args.problems_dir.is_dir() or not args.dsa_bench.is_file():
        raise FileNotFoundError("problems directory and dsa-bench must exist")
    if args.output_root.exists() and not args.resume:
        raise ValueError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "raw").mkdir(exist_ok=args.resume)
    fractions = parse_fractions(args.fractions)
    orders = tuple(item for item in args.cypress_orders.split(",") if item)
    seeds = tuple(int(item) for item in args.random_seeds.split(",") if item)
    variants = tuple(
        CypressVariant(order, seed) for order in orders for seed in (seeds if order == "random" else (0,))
    )
    config = ScreenConfig(args.dsa_bench, variants, args.canonical_restarts, args.timeout)
    paths = sorted(args.problems_dir.glob("*.dsa.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    all_rows: list[dict[str, Any]] = []
    had_errors = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _screen_problem,
                path,
                args.output_root,
                config,
                fractions,
            ): path
            for path in paths
        }
        completed = 0
        for future in as_completed(futures):
            path = futures[future]
            try:
                rows = future.result()
            except Exception as error:  # Keep every corpus member terminal and fail the command.
                had_errors = True
                rows = [
                    {
                        "tag": path.name.removesuffix(".dsa.json"),
                        "status": "screen_error",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                ]
            all_rows.extend(rows)
            completed += 1
            print(f"[{completed}/{len(paths)}] {path.name}: {len(rows)} rows", flush=True)
    all_rows.sort(
        key=lambda row: (
            str(row.get("tag", "")),
            int(row.get("pool_id", -1)),
            str(row.get("capacity_fraction", "")),
            str(row.get("arm", "")),
        )
    )
    _write_summary(args.output_root, all_rows)
    _write_model_separation(args.problems_dir, args.output_root, all_rows)
    _write_cypress_variants(args.output_root)
    print(f"SCREENED problems={len(paths)} result_rows={len(all_rows)}", flush=True)
    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
