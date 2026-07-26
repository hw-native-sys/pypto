# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Construct four replay-ready solutions that isolate one physical-overlap pair."""

import argparse
import bisect
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import prepare_dsa_ablation as ablation

OverlapSignature = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Witness:
    first_offset: int
    disjoint_second_offset: int
    overlapping_second_offset: int
    overlap_bytes: int
    overlap_range: tuple[int, int]
    first_signature: OverlapSignature
    second_signature: OverlapSignature


def _overlap(
    first_offset: int,
    first_size: int,
    second_offset: int,
    second_size: int,
) -> tuple[int, tuple[int, int] | None]:
    begin = max(first_offset, second_offset)
    end = min(first_offset + first_size, second_offset + second_size)
    return (end - begin, (begin, end)) if begin < end else (0, None)


def _problem_parts(
    problem: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], set[tuple[int, int]]]:
    body = problem["problem"]
    buffers = {buffer["id"]: buffer for buffer in body["buffers"]}
    pools = {pool["id"]: pool for pool in body["pools"]}
    separations = {
        ablation._pair(value["first"], value["second"])
        for value in body["constraints"].get("separations", [])
    }
    return buffers, pools, separations


def _validate_candidate(
    problem: dict[str, Any],
    base_solution: dict[str, Any],
    first_id: int,
    second_id: int,
) -> tuple[int, dict[int, dict[str, Any]], dict[int, dict[str, int]]]:
    ablation._validate_envelope(problem, base_solution)
    buffers, _, separations = _problem_parts(problem)
    placements = ablation._placement_map(base_solution)
    if first_id == second_id or first_id not in buffers or second_id not in buffers:
        raise ValueError(f"invalid candidate pair ({first_id}, {second_id})")
    first = buffers[first_id]
    second = buffers[second_id]
    first_pool = placements[first_id]["pool"]
    second_pool = placements[second_id]["pool"]
    if first_pool != second_pool:
        raise ValueError(f"candidate buffers use different pools: {first_pool} and {second_pool}")
    if ablation._lifetimes_overlap(first, second):
        raise ValueError(f"candidate buffers {first_id} and {second_id} have overlapping lifetimes")
    if ablation._pair(first_id, second_id) in separations:
        raise ValueError(f"candidate buffers {first_id} and {second_id} have a hard separation")
    return first_pool, buffers, placements


def _legal_positions(
    problem: dict[str, Any],
    base_solution: dict[str, Any],
    buffer_id: int,
    other_target: int,
) -> dict[OverlapSignature, list[int]]:
    buffers, pools, separations = _problem_parts(problem)
    placements = ablation._placement_map(base_solution)
    buffer = buffers[buffer_id]
    pool_id = placements[buffer_id]["pool"]
    pool = pools[pool_id]
    capacity = pool.get("capacity")
    if not isinstance(capacity, int):
        raise ValueError(f"pool {pool_id} has no finite capacity")
    alignment = buffer["alignment"]
    grouped: dict[OverlapSignature, list[int]] = {}
    for offset in range(0, capacity - buffer["size"] + 1, alignment):
        if any(
            ablation._ranges_overlap(
                offset,
                buffer["size"],
                reserved["begin"],
                reserved["end"] - reserved["begin"],
            )
            for reserved in pool.get("reserved_ranges", [])
        ):
            continue
        signature: list[tuple[int, int]] = []
        legal = True
        for neighbor_id in sorted(buffers):
            neighbor = buffers[neighbor_id]
            if neighbor_id in {buffer_id, other_target}:
                continue
            neighbor_placement = placements[neighbor_id]
            if neighbor_placement["pool"] != pool_id:
                continue
            overlap_bytes, _ = _overlap(
                offset,
                buffer["size"],
                neighbor_placement["offset"],
                neighbor["size"],
            )
            if overlap_bytes == 0:
                continue
            if (
                ablation._lifetimes_overlap(buffer, neighbor)
                or ablation._pair(buffer_id, neighbor_id) in separations
            ):
                legal = False
                break
            signature.append((neighbor_id, overlap_bytes))
        if legal:
            grouped.setdefault(tuple(signature), []).append(offset)
    return grouped


def _ordered_groups(groups: dict[OverlapSignature, list[int]]) -> list[tuple[OverlapSignature, list[int]]]:
    return sorted(groups.items(), key=lambda item: (len(item[0]), -len(item[1]), item[0]))


def _find_witness_pair(
    first_groups: dict[OverlapSignature, list[int]],
    second_groups: dict[OverlapSignature, list[int]],
    first_size: int,
    second_size: int,
) -> tuple[_Witness, _Witness]:
    first_witness: dict[tuple[OverlapSignature, OverlapSignature, int], _Witness] = {}
    best: tuple[int, _Witness, _Witness] | None = None
    for first_signature, first_offsets in _ordered_groups(first_groups):
        for second_signature, second_offsets in _ordered_groups(second_groups):
            for first_offset in first_offsets:
                overlap_lower = bisect.bisect_right(second_offsets, first_offset - second_size)
                overlap_upper = bisect.bisect_left(second_offsets, first_offset + first_size)
                if overlap_lower == 0 and overlap_upper == len(second_offsets):
                    continue
                if overlap_lower:
                    disjoint_offset = second_offsets[0]
                else:
                    disjoint_offset = second_offsets[overlap_upper]
                for overlap_offset in second_offsets[overlap_lower:overlap_upper]:
                    overlap_bytes, overlap_range = _overlap(
                        first_offset,
                        first_size,
                        overlap_offset,
                        second_size,
                    )
                    if overlap_range is None:
                        continue
                    witness = _Witness(
                        first_offset=first_offset,
                        disjoint_second_offset=disjoint_offset,
                        overlapping_second_offset=overlap_offset,
                        overlap_bytes=overlap_bytes,
                        overlap_range=overlap_range,
                        first_signature=first_signature,
                        second_signature=second_signature,
                    )
                    key = (first_signature, second_signature, overlap_bytes)
                    previous = first_witness.get(key)
                    if previous is None:
                        first_witness[key] = witness
                    elif (
                        previous.first_offset != witness.first_offset
                        and previous.overlap_range != witness.overlap_range
                    ):
                        distance = abs(previous.overlap_range[0] - witness.overlap_range[0])
                        if best is None or distance > best[0]:
                            best = distance, previous, witness
    if best is not None:
        return best[1], best[2]
    raise ValueError("no capacity-fitting pair isolation with a translated address control")


def _solution_with_offsets(
    problem: dict[str, Any],
    base_solution: dict[str, Any],
    *,
    name: str,
    first_id: int,
    first_offset: int,
    second_id: int,
    second_offset: int,
) -> dict[str, Any]:
    solution = copy.deepcopy(base_solution)
    placements = ablation._placement_map(solution)
    placements[first_id]["offset"] = first_offset
    placements[second_id]["offset"] = second_offset
    solution["metadata"] = {
        "experiment": "exact_pair_isolation_v1",
        "solver": "pair_isolation",
        "variant": name,
    }
    ablation._validate_envelope(problem, solution)
    return solution


def _validate_endpoint_geometry(
    problem: dict[str, Any],
    solutions: dict[str, dict[str, Any]],
    pair: tuple[int, int],
) -> dict[str, Any]:
    geometry = {name: ablation._overlap_geometry(problem, solution) for name, solution in solutions.items()}
    if geometry["D0"] != geometry["D1"]:
        raise ValueError("constructed controls differ: overlap_geometry(D0) != overlap_geometry(D1)")
    if geometry["O0"] != geometry["O1"]:
        raise ValueError("constructed controls differ: overlap_geometry(O0) != overlap_geometry(O1)")
    if pair in geometry["D0"]:
        raise ValueError(f"candidate pair {pair} overlaps in the disjoint endpoints")
    if pair not in geometry["O0"]:
        raise ValueError(f"candidate pair {pair} does not overlap in the overlapping endpoints")
    expected = dict(geometry["D0"])
    expected[pair] = geometry["O0"][pair]
    if geometry["O0"] != expected:
        changed = sorted(set(geometry["D0"]) ^ set(geometry["O0"]))
        raise ValueError(f"constructed endpoint changes more than candidate pair {pair}: {changed}")
    return {
        name: [
            {"first": first, "second": second, "bytes": overlap_bytes}
            for (first, second), overlap_bytes in sorted(values.items())
        ]
        for name, values in geometry.items()
    }


def construct(
    problem_path: Path,
    base_solution_path: Path,
    output_root: Path,
    *,
    first_id: int,
    second_id: int,
    sibling_problem_dir: Path | None = None,
    sibling_solution_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct and write an exact four-endpoint pair isolation."""
    problem = ablation._read_json(problem_path)
    base_solution = ablation._read_json(base_solution_path)
    pool_id, buffers, _ = _validate_candidate(problem, base_solution, first_id, second_id)
    first_groups = _legal_positions(problem, base_solution, first_id, second_id)
    second_groups = _legal_positions(problem, base_solution, second_id, first_id)
    if not first_groups or not second_groups:
        raise ValueError("candidate has no legal aligned positions")
    first_witness, second_witness = _find_witness_pair(
        first_groups,
        second_groups,
        buffers[first_id]["size"],
        buffers[second_id]["size"],
    )

    endpoint_offsets = {
        "D0": (first_witness.first_offset, first_witness.disjoint_second_offset),
        "O0": (first_witness.first_offset, first_witness.overlapping_second_offset),
        "D1": (second_witness.first_offset, second_witness.disjoint_second_offset),
        "O1": (second_witness.first_offset, second_witness.overlapping_second_offset),
    }
    solutions = {
        name: _solution_with_offsets(
            problem,
            base_solution,
            name=name,
            first_id=first_id,
            first_offset=offsets[0],
            second_id=second_id,
            second_offset=offsets[1],
        )
        for name, offsets in endpoint_offsets.items()
    }
    pair = ablation._pair(first_id, second_id)
    geometry = _validate_endpoint_geometry(problem, solutions, pair)
    siblings = ablation._load_sibling_solutions(
        sibling_problem_dir,
        sibling_solution_dir,
        target_instance=problem["instance"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for name, solution in solutions.items():
        endpoint = output_root / name
        endpoint.mkdir(parents=True, exist_ok=True)
        solution_name = f"pypto_{problem['instance']}.dsa.solution.json"
        (endpoint / solution_name).write_text(
            json.dumps(solution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for sibling_name, sibling in siblings.items():
            (endpoint / sibling_name).write_text(
                json.dumps(sibling, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    report = {
        "schema_version": 1,
        "experiment": "exact_pair_isolation_v1",
        "problem": str(problem_path),
        "problem_fingerprint": ablation._fingerprint(problem),
        "instance": problem["instance"],
        "pool": pool_id,
        "pair": {"first": pair[0], "second": pair[1]},
        "position_counts": {
            "first": sum(len(offsets) for offsets in first_groups.values()),
            "second": sum(len(offsets) for offsets in second_groups.values()),
        },
        "endpoint_offsets": {
            name: {"first": offsets[0], "second": offsets[1]} for name, offsets in endpoint_offsets.items()
        },
        "overlap_bytes": first_witness.overlap_bytes,
        "overlap_ranges": {
            "O0": list(first_witness.overlap_range),
            "O1": list(second_witness.overlap_range),
        },
        "control_address_delta": abs(first_witness.overlap_range[0] - second_witness.overlap_range[0]),
        "first_signature": [
            {"buffer": buffer, "bytes": overlap_bytes}
            for buffer, overlap_bytes in first_witness.first_signature
        ],
        "second_signature": [
            {"buffer": buffer, "bytes": overlap_bytes}
            for buffer, overlap_bytes in first_witness.second_signature
        ],
        "overlap_geometry": geometry,
        "sibling_solutions": sorted(siblings),
    }
    (output_root / "pair-isolation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--base-solution", type=Path, required=True)
    parser.add_argument("--first-buffer", type=int, required=True)
    parser.add_argument("--second-buffer", type=int, required=True)
    parser.add_argument("--sibling-problem-dir", type=Path)
    parser.add_argument("--sibling-solution-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = construct(
            args.problem,
            args.base_solution,
            args.output_root,
            first_id=args.first_buffer,
            second_id=args.second_buffer,
            sibling_problem_dir=args.sibling_problem_dir,
            sibling_solution_dir=args.sibling_solution_dir,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
