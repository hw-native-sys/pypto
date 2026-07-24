# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for controlled DSA placement ablations."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "prepare_dsa_ablation.py"


@pytest.fixture(scope="module")
def ablation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_prepare_dsa_ablation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem(*, with_cost: bool) -> dict:
    problem = {
        "schema_version": 1,
        "profile": "pypto_research_v1" if with_cost else "pypto_hard_v1",
        "instance": "sample",
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
                    "name": "a",
                    "size": 64,
                },
                {
                    "alignment": 32,
                    "allowed_pools": [1],
                    "id": 1,
                    "live_intervals": [{"lower": 2, "upper": 4}],
                    "name": "b",
                    "size": 64,
                },
                {
                    "alignment": 32,
                    "allowed_pools": [1],
                    "id": 2,
                    "live_intervals": [{"lower": 1, "upper": 3}],
                    "name": "c",
                    "size": 32,
                },
            ],
            "constraints": {
                "colocations": [],
                "pinned_allocations": [],
                "separations": [],
                "temporal_exclusions": [],
            },
            "objective": {
                "aggregation": "lexicographic",
                "terms": ["capacity_overflow", "reuse_cost"] if with_cost else ["max_peak"],
            },
            "pools": [{"capacity": 160, "id": 1, "name": "Vec", "reserved_ranges": []}],
            "pypto_structure": {
                "alias_classes": [
                    {"buffer": 0, "members": ["a"]},
                    {"buffer": 1, "members": ["b"]},
                    {"buffer": 2, "members": ["c"]},
                ],
                "pipeline_groups": [],
            },
        },
    }
    if with_cost:
        problem["problem"]["cost_model"] = {
            "reuse_penalties": [{"cost": 7, "first": 0, "reason": "cross_pipe", "second": 1}]
        }
    return problem


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(ablation: ModuleType, problem: dict) -> dict:
    return {
        "instance": "sample",
        "metadata": {"solver": "test"},
        "placements": [
            {"buffer": 0, "offset": 0, "pool": 1},
            {"buffer": 1, "offset": 0, "pool": 1},
            {"buffer": 2, "offset": 128, "pool": 1},
        ],
        "problem_fingerprint": ablation._fingerprint(problem),
        "profile": problem["profile"],
        "schema_version": 1,
    }


def test_prepare_rebinds_hard_base_and_emits_checked_variant(ablation: ModuleType, tmp_path: Path):
    hard = _problem(with_cost=False)
    target = _problem(with_cost=True)
    hard_path = tmp_path / "hard.json"
    target_path = tmp_path / "target.json"
    base_path = tmp_path / "base.solution.json"
    spec_path = tmp_path / "spec.json"
    sibling_problem_dir = tmp_path / "sibling-problems"
    sibling_solution_dir = tmp_path / "sibling-solutions"
    sibling_problem_dir.mkdir()
    sibling_solution_dir.mkdir()
    _write(hard_path, hard)
    _write(target_path, target)
    _write(base_path, _solution(ablation, hard))
    sibling_problem = _problem(with_cost=True)
    sibling_problem["instance"] = "other"
    sibling_solution = _solution(ablation, sibling_problem)
    sibling_solution["instance"] = "other"
    sibling_solution["problem_fingerprint"] = ablation._fingerprint(sibling_problem)
    _write(sibling_problem_dir / "pypto_other.dsa.json", sibling_problem)
    _write(sibling_solution_dir / "pypto_other.dsa.solution.json", sibling_solution)
    _write(
        spec_path,
        {
            "cases": [
                {
                    "instance": "sample",
                    "name": "case",
                    "variants": [
                        {
                            "base": "compact",
                            "expected": {
                                "added_pairs": 0,
                                "max_peak": 160,
                                "removed_pairs": 1,
                                "reuse_cost": 0,
                            },
                            "hypothesis": "separate the penalized pair",
                            "moves": [{"buffer": 1, "from_offset": 0, "name": "b", "to_offset": 64}],
                            "name": "separate_b",
                        }
                    ],
                }
            ],
            "experiment": "test",
            "schema_version": 1,
        },
    )

    summary = ablation.prepare(
        target_path,
        spec_path,
        {"compact": base_path},
        {"compact": hard_path},
        tmp_path / "out",
        case_name="case",
        sibling_problem_dir=sibling_problem_dir,
        sibling_solution_dir=sibling_solution_dir,
    )

    variant = json.loads(
        (tmp_path / "out" / "separate_b" / "pypto_sample.dsa.solution.json").read_text(encoding="utf-8")
    )
    assert variant["problem_fingerprint"] == ablation._fingerprint(target)
    assert variant["profile"] == "pypto_research_v1"
    assert summary["variants"][0]["statistics"]["reuse_cost"] == 0
    assert summary["variants"][0]["removed_overlaps"] == [{"first": 0, "second": 1, "bytes": 64}]
    assert summary["variants"][0]["solution"] == "separate_b/pypto_sample.dsa.solution.json"
    assert summary["sibling_solutions"] == ["pypto_other.dsa.solution.json"]
    assert (tmp_path / "out" / "compact" / "pypto_other.dsa.solution.json").is_file()
    assert (tmp_path / "out" / "separate_b" / "pypto_other.dsa.solution.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda spec: spec["cases"][0]["variants"][0]["moves"][0].update(name="wrong"), "expected buffer"),
        (
            lambda spec: spec["cases"][0]["variants"][0]["moves"][0].update(from_offset=32),
            "expected buffer 1 at 32",
        ),
        (
            lambda spec: spec["cases"][0]["variants"][0]["moves"][0].update(to_offset=48),
            "invalid aligned offset",
        ),
    ],
)
def test_prepare_rejects_stale_or_invalid_moves(ablation: ModuleType, tmp_path: Path, mutation, message: str):
    problem = _problem(with_cost=True)
    problem_path = tmp_path / "problem.json"
    solution_path = tmp_path / "solution.json"
    spec = {
        "cases": [
            {
                "instance": "sample",
                "name": "case",
                "variants": [
                    {
                        "base": "base",
                        "hypothesis": "test",
                        "moves": [{"buffer": 1, "from_offset": 0, "name": "b", "to_offset": 64}],
                        "name": "variant",
                    }
                ],
            }
        ],
        "experiment": "test",
        "schema_version": 1,
    }
    mutation(spec)
    spec_path = tmp_path / "spec.json"
    _write(problem_path, problem)
    _write(solution_path, _solution(ablation, problem))
    _write(spec_path, spec)

    with pytest.raises(ValueError, match=message):
        ablation.prepare(
            problem_path,
            spec_path,
            {"base": solution_path},
            {"base": problem_path},
            tmp_path / "out",
            case_name="case",
        )


def test_prepare_rejects_incompatible_hard_geometry(ablation: ModuleType, tmp_path: Path):
    target = _problem(with_cost=True)
    source = _problem(with_cost=False)
    source["problem"]["buffers"][1]["size"] = 96
    target_path = tmp_path / "target.json"
    source_path = tmp_path / "source.json"
    solution_path = tmp_path / "solution.json"
    spec_path = tmp_path / "spec.json"
    _write(target_path, target)
    _write(source_path, source)
    _write(solution_path, _solution(ablation, source))
    _write(
        spec_path,
        {
            "cases": [{"instance": "sample", "name": "case", "variants": []}],
            "experiment": "test",
            "schema_version": 1,
        },
    )

    with pytest.raises(ValueError, match="hard geometry differs"):
        ablation.prepare(
            target_path,
            spec_path,
            {"base": solution_path},
            {"base": source_path},
            tmp_path / "out",
            case_name="case",
        )


def test_prepare_rejects_incomplete_sibling_solution_set(ablation: ModuleType, tmp_path: Path):
    problem = _problem(with_cost=True)
    problem_path = tmp_path / "problem.json"
    solution_path = tmp_path / "solution.json"
    spec_path = tmp_path / "spec.json"
    sibling_problem_dir = tmp_path / "sibling-problems"
    sibling_solution_dir = tmp_path / "sibling-solutions"
    sibling_problem_dir.mkdir()
    sibling_solution_dir.mkdir()
    _write(problem_path, problem)
    _write(solution_path, _solution(ablation, problem))
    _write(
        spec_path,
        {
            "cases": [{"instance": "sample", "name": "case", "variants": []}],
            "experiment": "test",
            "schema_version": 1,
        },
    )
    sibling = _problem(with_cost=True)
    sibling["instance"] = "other"
    _write(sibling_problem_dir / "pypto_other.dsa.json", sibling)

    with pytest.raises(ValueError, match="has no matching solution"):
        ablation.prepare(
            problem_path,
            spec_path,
            {"base": solution_path},
            {"base": problem_path},
            tmp_path / "out",
            case_name="case",
            sibling_problem_dir=sibling_problem_dir,
            sibling_solution_dir=sibling_solution_dir,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
