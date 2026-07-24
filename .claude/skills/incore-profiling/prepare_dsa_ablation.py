# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare fingerprinted, independently checked DSA placement ablations."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _fingerprint(problem: dict[str, Any]) -> str:
    canonical = (json.dumps(problem, indent=2, sort_keys=True) + "\n").encode()
    value = _FNV_OFFSET
    for byte in canonical:
        value ^= byte
        value = value * _FNV_PRIME & _UINT64_MASK
    return f"{value:016x}"


def _parse_named_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"{option} repeats name {name!r}")
        result[name] = Path(path)
    return result


def _hard_geometry(problem: dict[str, Any]) -> dict[str, Any]:
    body = problem["problem"]
    # PyPTO alias member strings include generated SSA suffixes that can vary
    # between otherwise identical exports. Alias classes and pipeline groups
    # are provenance in schema v1; all placement-affecting requirements have
    # already been materialized into buffers and constraints.
    return {
        "pools": body["pools"],
        "buffers": body["buffers"],
        "constraints": body["constraints"],
    }


def _ranges_overlap(first_offset: int, first_size: int, second_offset: int, second_size: int) -> bool:
    return first_offset < second_offset + second_size and second_offset < first_offset + first_size


def _lifetimes_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return any(
        left["lower"] < right["upper"] and right["lower"] < left["upper"]
        for left in first["live_intervals"]
        for right in second["live_intervals"]
    )


def _pair(first: int, second: int) -> tuple[int, int]:
    return min(first, second), max(first, second)


def _placement_map(solution: dict[str, Any]) -> dict[int, dict[str, int]]:
    placements: dict[int, dict[str, int]] = {}
    for placement in solution["placements"]:
        buffer = placement["buffer"]
        if buffer in placements:
            raise ValueError(f"solution repeats buffer {buffer}")
        placements[buffer] = placement
    return placements


def _validate_solution(problem: dict[str, Any], solution: dict[str, Any]) -> None:
    body = problem["problem"]
    constraints = body["constraints"]
    unsupported = {
        key for key in ("colocations", "pinned_allocations", "temporal_exclusions") if constraints.get(key)
    }
    if unsupported:
        raise ValueError(f"ablation tool does not support non-empty constraints: {sorted(unsupported)}")

    pools = {pool["id"]: pool for pool in body["pools"]}
    buffers = {buffer["id"]: buffer for buffer in body["buffers"]}
    placements = _placement_map(solution)
    if set(placements) != set(buffers):
        missing = sorted(set(buffers) - set(placements))
        extra = sorted(set(placements) - set(buffers))
        raise ValueError(f"solution buffer set mismatch: missing={missing}, extra={extra}")

    for buffer_id, placement in placements.items():
        buffer = buffers[buffer_id]
        pool_id = placement["pool"]
        offset = placement["offset"]
        if pool_id not in pools or pool_id not in buffer["allowed_pools"]:
            raise ValueError(f"buffer {buffer_id} uses disallowed pool {pool_id}")
        if offset < 0 or offset % buffer["alignment"] != 0:
            raise ValueError(f"buffer {buffer_id} has invalid aligned offset {offset}")
        end = offset + buffer["size"]
        pool = pools[pool_id]
        if end > _UINT64_MASK:
            raise ValueError(f"buffer {buffer_id} address range overflows uint64")
        if "capacity" in pool and end > pool["capacity"]:
            raise ValueError(f"buffer {buffer_id} exceeds pool {pool_id} capacity")
        for reserved in pool.get("reserved_ranges", []):
            if _ranges_overlap(
                offset,
                buffer["size"],
                reserved["begin"],
                reserved["end"] - reserved["begin"],
            ):
                raise ValueError(f"buffer {buffer_id} overlaps a reserved range in pool {pool_id}")

    separations = {_pair(value["first"], value["second"]) for value in constraints.get("separations", [])}
    ordered_buffers = sorted(buffers)
    for index, first_id in enumerate(ordered_buffers):
        first = buffers[first_id]
        first_placement = placements[first_id]
        for second_id in ordered_buffers[index + 1 :]:
            second = buffers[second_id]
            second_placement = placements[second_id]
            if first_placement["pool"] != second_placement["pool"]:
                continue
            if not _ranges_overlap(
                first_placement["offset"],
                first["size"],
                second_placement["offset"],
                second["size"],
            ):
                continue
            if _lifetimes_overlap(first, second) or _pair(first_id, second_id) in separations:
                raise ValueError(f"buffers {first_id} and {second_id} overlap illegally")


def _validate_envelope(problem: dict[str, Any], solution: dict[str, Any]) -> None:
    for key in ("schema_version", "profile", "instance"):
        if solution.get(key) != problem.get(key):
            raise ValueError(
                f"solution {key} {solution.get(key)!r} does not match problem {problem.get(key)!r}"
            )
    expected = _fingerprint(problem)
    if solution.get("problem_fingerprint") != expected:
        raise ValueError(
            f"solution fingerprint {solution.get('problem_fingerprint')!r} "
            f"does not match problem fingerprint {expected!r}"
        )
    _validate_solution(problem, solution)


def _overlap_geometry(problem: dict[str, Any], solution: dict[str, Any]) -> dict[tuple[int, int], int]:
    body = problem["problem"]
    buffers = {buffer["id"]: buffer for buffer in body["buffers"]}
    placements = _placement_map(solution)
    separations = {
        _pair(value["first"], value["second"]) for value in body["constraints"].get("separations", [])
    }
    result: dict[tuple[int, int], int] = {}
    ordered = sorted(buffers)
    for index, first_id in enumerate(ordered):
        first = buffers[first_id]
        first_placement = placements[first_id]
        for second_id in ordered[index + 1 :]:
            second = buffers[second_id]
            second_placement = placements[second_id]
            if (
                first_placement["pool"] != second_placement["pool"]
                or _lifetimes_overlap(first, second)
                or _pair(first_id, second_id) in separations
            ):
                continue
            begin = max(first_placement["offset"], second_placement["offset"])
            end = min(
                first_placement["offset"] + first["size"],
                second_placement["offset"] + second["size"],
            )
            if begin < end:
                result[(first_id, second_id)] = end - begin
    return result


def _statistics(problem: dict[str, Any], solution: dict[str, Any]) -> dict[str, int]:
    body = problem["problem"]
    buffers = {buffer["id"]: buffer for buffer in body["buffers"]}
    placements = _placement_map(solution)
    peak_by_pool: dict[int, int] = {}
    for buffer_id, placement in placements.items():
        end = placement["offset"] + buffers[buffer_id]["size"]
        peak_by_pool[placement["pool"]] = max(peak_by_pool.get(placement["pool"], 0), end)
    overlaps = _overlap_geometry(problem, solution)
    penalties = body.get("cost_model", {}).get("reuse_penalties", [])
    active_cost = sum(edge["cost"] for edge in penalties if _pair(edge["first"], edge["second"]) in overlaps)
    return {
        "max_peak": max(peak_by_pool.values(), default=0),
        "total_peak": sum(peak_by_pool.values()),
        "reuse_pairs": len(overlaps),
        "reuse_bytes": sum(overlaps.values()),
        "reuse_cost": active_cost,
    }


def _apply_variant(
    problem: dict[str, Any],
    base_solution: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(base_solution)
    placements = _placement_map(result)
    buffers = {buffer["id"]: buffer for buffer in problem["problem"]["buffers"]}
    for move in variant["moves"]:
        buffer_id = move["buffer"]
        if buffer_id not in placements or buffer_id not in buffers:
            raise ValueError(f"variant {variant['name']!r} references unknown buffer {buffer_id}")
        if buffers[buffer_id]["name"] != move["name"]:
            raise ValueError(
                f"variant {variant['name']!r} expected buffer {buffer_id} name {move['name']!r}, "
                f"got {buffers[buffer_id]['name']!r}"
            )
        if placements[buffer_id]["offset"] != move["from_offset"]:
            raise ValueError(
                f"variant {variant['name']!r} expected buffer {buffer_id} at "
                f"{move['from_offset']}, got {placements[buffer_id]['offset']}"
            )
        placements[buffer_id]["offset"] = move["to_offset"]

    result["metadata"] = {
        "base": variant["base"],
        "experiment": "expert_dsa_rp_ablation_v1",
        "variant": variant["name"],
    }
    _validate_envelope(problem, result)
    before = _overlap_geometry(problem, base_solution)
    after = _overlap_geometry(problem, result)
    report = {
        "name": variant["name"],
        "base": variant["base"],
        "hypothesis": variant["hypothesis"],
        "moves": variant["moves"],
        "statistics": _statistics(problem, result),
        "overlap_delta": [
            {
                "first": pair[0],
                "second": pair[1],
                "before_bytes": before.get(pair, 0),
                "after_bytes": after.get(pair, 0),
            }
            for pair in sorted(before.keys() | after.keys())
            if before.get(pair, 0) != after.get(pair, 0)
        ],
        "removed_overlaps": [
            {"first": pair[0], "second": pair[1], "bytes": before[pair]}
            for pair in sorted(before.keys() - after.keys())
        ],
        "added_overlaps": [
            {"first": pair[0], "second": pair[1], "bytes": after[pair]}
            for pair in sorted(after.keys() - before.keys())
        ],
    }
    expected = variant.get("expected", {})
    actual = {
        **report["statistics"],
        "removed_pairs": len(report["removed_overlaps"]),
        "added_pairs": len(report["added_overlaps"]),
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"variant {variant['name']!r} expectation mismatch: {mismatches}")
    return result, report


def prepare(
    problem_path: Path,
    specification_path: Path,
    base_solution_paths: dict[str, Path],
    base_problem_paths: dict[str, Path],
    output_root: Path,
    *,
    case_name: str,
) -> dict[str, Any]:
    """Prepare one case's placement variants and return its validation report."""
    problem = _read_json(problem_path)
    specification = _read_json(specification_path)
    cases = {case["name"]: case for case in specification["cases"]}
    if case_name not in cases:
        raise ValueError(f"unknown experiment case {case_name!r}")
    case = cases[case_name]
    if problem["instance"] != case["instance"]:
        raise ValueError(
            f"case {case_name!r} expects instance {case['instance']!r}, got {problem['instance']!r}"
        )

    bases: dict[str, dict[str, Any]] = {}
    for name, solution_path in base_solution_paths.items():
        if name not in base_problem_paths:
            raise ValueError(f"base {name!r} has no matching --base-problem")
        source_problem = _read_json(base_problem_paths[name])
        source_solution = _read_json(solution_path)
        _validate_envelope(source_problem, source_solution)
        if _hard_geometry(source_problem) != _hard_geometry(problem):
            raise ValueError(f"base {name!r} hard geometry differs from the target problem")
        rebound = copy.deepcopy(source_solution)
        rebound["schema_version"] = problem["schema_version"]
        rebound["profile"] = problem["profile"]
        rebound["instance"] = problem["instance"]
        rebound["problem_fingerprint"] = _fingerprint(problem)
        rebound["metadata"] = {
            "base": name,
            "experiment": "expert_dsa_rp_ablation_v1",
            "variant": name,
        }
        _validate_envelope(problem, rebound)
        bases[name] = rebound

    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for name, solution in sorted(bases.items()):
        output = output_root / name / f"pypto_{problem['instance']}.dsa.solution.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(solution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for variant in case["variants"]:
        if variant["base"] not in bases:
            raise ValueError(f"variant {variant['name']!r} requires missing base {variant['base']!r}")
        solution, report = _apply_variant(problem, bases[variant["base"]], variant)
        output = output_root / variant["name"] / f"pypto_{problem['instance']}.dsa.solution.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(solution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["solution"] = str(output.relative_to(output_root))
        reports.append(report)

    summary = {
        "schema_version": 1,
        "experiment": specification["experiment"],
        "case": case,
        "problem": str(problem_path),
        "problem_fingerprint": _fingerprint(problem),
        "bases": {name: _statistics(problem, solution) for name, solution in sorted(bases.items())},
        "variants": reports,
    }
    (output_root / "ablation-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--base-solution", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--base-problem", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = prepare(
            args.problem,
            args.specification,
            _parse_named_paths(args.base_solution, "--base-solution"),
            _parse_named_paths(args.base_problem, "--base-problem"),
            args.output_root,
            case_name=args.case,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
