# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "screen_dsa_capacity_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_screen_dsa_capacity_corpus", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
screen = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = screen
_SPEC.loader.exec_module(screen)


def _document() -> dict:
    return {
        "instance": "mixed",
        "problem": {
            "pools": [
                {"id": 1, "name": "Vec", "capacity": 200, "reserved_ranges": []},
                {"id": 2, "name": "Mat", "capacity": 300, "reserved_ranges": []},
            ],
            "buffers": [
                {"id": 0, "size": 40, "allowed_pools": [1]},
                {"id": 1, "size": 60, "allowed_pools": [1]},
                {"id": 2, "size": 100, "allowed_pools": [2]},
            ],
            "cost_model": {
                "reuse_penalties": [
                    {"first": 0, "second": 1, "cost": 1},
                    {"first": 1, "second": 2, "cost": 1},
                ]
            },
        },
    }


def _solution() -> dict:
    return {
        "placements": [
            {"buffer": 0, "pool": 1, "offset": 0},
            {"buffer": 1, "pool": 1, "offset": 32},
            {"buffer": 2, "pool": 2, "offset": 64},
        ]
    }


def test_capacity_grid_uses_geometry_peak_and_native_endpoints() -> None:
    fractions = screen.parse_fractions("0,1/4,1/2,1")
    assert screen.capacity_grid(100, 200, fractions) == (100, 125, 150, 200)


def test_fraction_validation_fails_closed() -> None:
    for value in ("", "1/4,1", "0,1/2", "0,1/2,1/4,1", "0,-1/2,1"):
        with pytest.raises(ValueError):
            screen.parse_fractions(value)


def test_selected_pool_capacity_changes_without_touching_other_pool() -> None:
    original = _document()
    derived = screen.with_pool_capacity(original, 1, 96)
    assert [pool["capacity"] for pool in derived["problem"]["pools"]] == [96, 300]
    assert [pool["capacity"] for pool in original["problem"]["pools"]] == [200, 300]


def test_pool_peak_and_penalty_partition_follow_placements() -> None:
    document = _document()
    solution = _solution()
    assert screen.pool_peak(document, solution, 1) == 92
    assert screen.pool_peak(document, solution, 2) == 164
    assert screen.penalty_counts_by_pool(document, solution) == {1: 1}


def test_cypress_selection_is_penalty_weight_blind() -> None:
    fewer_aliases = {
        "status": "feasible",
        "reuse_cost": 1000,
        "total_peak": 100,
        "solver_metrics": {"actual_alias_pairs": 2, "relaxed_edges": 5},
    }
    lower_reuse_cost = {
        "status": "feasible",
        "reuse_cost": 0,
        "total_peak": 90,
        "solver_metrics": {"actual_alias_pairs": 3, "relaxed_edges": 4},
    }
    assert min([fewer_aliases, lower_reuse_cost], key=screen.cypress_portfolio_key) is fewer_aliases


def test_placement_hash_ignores_analysis_annotations() -> None:
    solution = _solution()
    annotated = _solution()
    for placement in annotated["placements"]:
        placement["_buffer_size"] = 4096
    assert screen._placement_sha(solution) == screen._placement_sha(annotated)


def test_only_capacity_no_fit_serialization_error_is_retriable() -> None:
    assert screen._is_best_effort_serialization_failure(
        "dsa-bench: cannot serialize invalid structured solution: buffer 44 exceeds pool capacity"
    )
    assert not screen._is_best_effort_serialization_failure(
        "dsa-bench: cannot serialize invalid structured solution: duplicate buffer"
    )


def test_capacity_labels_match_device_protocol() -> None:
    assert screen.CAPACITY_LABELS == {
        Fraction(0): "tight",
        Fraction(1, 4): "q1",
        Fraction(1, 2): "half",
        Fraction(1): "native",
    }


def test_model_separation_counts_only_jointly_feasible_cells() -> None:
    document = _document()
    rows = []
    for fraction, costs in (("0", (4, 2, 3)), ("1", (4, 0, 0))):
        for arm, cost in zip(
            (screen.ARM_GEOMETRY, screen.ARM_DSA_RP, screen.ARM_CYPRESS), costs, strict=True
        ):
            rows.append(
                {
                    "tag": "case",
                    "pool_id": 1,
                    "capacity_fraction": fraction,
                    "arm": arm,
                    "status": "feasible",
                    "reuse_cost": cost,
                    "total_peak": 100 + cost,
                }
            )
    summary = screen.build_model_separation_rows({"case": document}, rows)[0]
    assert summary["screened_cells"] == 2
    assert summary["rp_better_geometry_cells"] == 2
    assert summary["rp_better_cypress_cells"] == 1
    assert summary["rp_equal_cypress_cells"] == 1
    assert summary["max_rp_cost_advantage_vs_cypress"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
