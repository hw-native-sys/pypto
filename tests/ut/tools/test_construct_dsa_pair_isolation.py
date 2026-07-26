# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for exact DSA pair-isolation construction."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SKILL_DIR = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling"
_SCRIPT = _SKILL_DIR / "construct_dsa_pair_isolation.py"


@pytest.fixture(scope="module")
def constructor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_construct_dsa_pair_isolation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(_SKILL_DIR))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SKILL_DIR))
    return module


def _problem(*, capacity: int = 512, conflicting_pair: bool = False) -> dict:
    second_lifetime = {"lower": 1, "upper": 3} if conflicting_pair else {"lower": 2, "upper": 4}
    return {
        "schema_version": 1,
        "profile": "pypto_research_v1",
        "instance": "pair_sample",
        "metadata": {
            "lifetime_ordering": "pypto_read_before_write",
            "solver_input": "pre_memory_reuse",
        },
        "problem": {
            "buffers": [
                {
                    "alignment": 32,
                    "allowed_pools": [1],
                    "id": 0,
                    "live_intervals": [{"lower": 0, "upper": 2}],
                    "name": "first",
                    "size": 64,
                },
                {
                    "alignment": 32,
                    "allowed_pools": [1],
                    "id": 1,
                    "live_intervals": [second_lifetime],
                    "name": "second",
                    "size": 64,
                },
                {
                    "alignment": 32,
                    "allowed_pools": [1],
                    "id": 2,
                    "live_intervals": [{"lower": 1, "upper": 3}],
                    "name": "fixed",
                    "size": 32,
                },
            ],
            "constraints": {
                "colocations": [],
                "pinned_allocations": [],
                "separations": [],
                "temporal_exclusions": [],
            },
            "cost_model": {"reuse_penalties": [{"cost": 7, "first": 0, "reason": "cross_pipe", "second": 1}]},
            "objective": {
                "aggregation": "lexicographic",
                "terms": ["capacity_overflow", "reuse_cost"],
            },
            "pools": [
                {
                    "capacity": capacity,
                    "id": 1,
                    "name": "Vec",
                    "reserved_ranges": [{"begin": 256, "end": 288}],
                }
            ],
            "pypto_structure": {
                "alias_classes": [
                    {"buffer": 0, "members": ["first"]},
                    {"buffer": 1, "members": ["second"]},
                    {"buffer": 2, "members": ["fixed"]},
                ],
                "pipeline_groups": [],
            },
        },
    }


def _solution(constructor: ModuleType, problem: dict) -> dict:
    return {
        "instance": problem["instance"],
        "metadata": {"solver": "test"},
        "placements": [
            {"buffer": 0, "offset": 0, "pool": 1},
            {"buffer": 1, "offset": 0, "pool": 1},
            {"buffer": 2, "offset": 320, "pool": 1},
        ],
        "problem_fingerprint": constructor.ablation._fingerprint(problem),
        "profile": problem["profile"],
        "schema_version": 1,
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_construct_emits_exact_pair_toggle_and_translated_control(
    constructor: ModuleType,
    tmp_path: Path,
):
    problem = _problem()
    problem_path = tmp_path / "problem.dsa.json"
    solution_path = tmp_path / "base.dsa.solution.json"
    _write(problem_path, problem)
    _write(solution_path, _solution(constructor, problem))

    report = constructor.construct(
        problem_path,
        solution_path,
        tmp_path / "out",
        first_id=0,
        second_id=1,
    )

    assert report["pair"] == {"first": 0, "second": 1}
    assert report["overlap_ranges"]["O0"] != report["overlap_ranges"]["O1"]
    assert report["control_address_delta"] > 0
    assert report["overlap_geometry"]["D0"] == report["overlap_geometry"]["D1"]
    assert report["overlap_geometry"]["O0"] == report["overlap_geometry"]["O1"]
    assert report["overlap_geometry"]["O0"] == [{"first": 0, "second": 1, "bytes": 64}]
    for endpoint in ("D0", "O0", "D1", "O1"):
        output = tmp_path / "out" / endpoint / "pypto_pair_sample.dsa.solution.json"
        assert output.is_file()
        constructor.ablation._validate_envelope(problem, json.loads(output.read_text()))


def test_construct_preserves_unrelated_overlap_signature(constructor: ModuleType, tmp_path: Path):
    problem = _problem()
    problem["problem"]["buffers"][2]["live_intervals"] = [{"lower": 4, "upper": 6}]
    problem["problem"]["pools"][0]["reserved_ranges"] = []
    base = _solution(constructor, problem)
    base["placements"][2]["offset"] = 0
    problem_path = tmp_path / "problem.dsa.json"
    solution_path = tmp_path / "base.dsa.solution.json"
    _write(problem_path, problem)
    _write(solution_path, base)

    report = constructor.construct(
        problem_path,
        solution_path,
        tmp_path / "out",
        first_id=0,
        second_id=1,
    )

    assert report["overlap_geometry"]["D0"] == report["overlap_geometry"]["D1"]
    assert report["overlap_geometry"]["O0"] == report["overlap_geometry"]["O1"]
    disjoint_pairs = {(entry["first"], entry["second"]) for entry in report["overlap_geometry"]["D0"]}
    overlap_pairs = {(entry["first"], entry["second"]) for entry in report["overlap_geometry"]["O0"]}
    assert overlap_pairs - disjoint_pairs == {(0, 1)}


def test_construct_rejects_temporally_conflicting_candidate(constructor: ModuleType, tmp_path: Path):
    problem = _problem(conflicting_pair=True)
    problem_path = tmp_path / "problem.dsa.json"
    solution_path = tmp_path / "base.dsa.solution.json"
    _write(problem_path, problem)
    base = _solution(constructor, problem)
    base["placements"][1]["offset"] = 64
    _write(solution_path, base)

    with pytest.raises(ValueError, match="overlapping lifetimes"):
        constructor.construct(
            problem_path,
            solution_path,
            tmp_path / "out",
            first_id=0,
            second_id=1,
        )


def test_construct_reports_insufficient_capacity(constructor: ModuleType, tmp_path: Path):
    problem = _problem(capacity=64)
    problem["problem"]["buffers"] = problem["problem"]["buffers"][:2]
    problem["problem"]["pypto_structure"]["alias_classes"] = problem["problem"]["pypto_structure"][
        "alias_classes"
    ][:2]
    problem["problem"]["pools"][0]["reserved_ranges"] = []
    problem_path = tmp_path / "problem.dsa.json"
    solution_path = tmp_path / "base.dsa.solution.json"
    _write(problem_path, problem)
    base = _solution(constructor, problem)
    base["placements"] = base["placements"][:2]
    base["problem_fingerprint"] = constructor.ablation._fingerprint(problem)
    _write(solution_path, base)

    with pytest.raises(ValueError, match="no capacity-fitting pair isolation"):
        constructor.construct(
            problem_path,
            solution_path,
            tmp_path / "out",
            first_id=0,
            second_id=1,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
