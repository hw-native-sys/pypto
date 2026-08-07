# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "select_dsa_device_slate.py"
_SPEC = importlib.util.spec_from_file_location("_test_select_dsa_device_slate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
selector = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = selector
_SPEC.loader.exec_module(selector)
_SHA256_HEX_LENGTH = 64


def _separation(tag: str, *, better: int = 0, no_fit: int = 0) -> dict[str, str]:
    return {
        "tag": tag,
        "instance": "kernel_aiv" if "mixed" in tag else "kernel",
        "buffers": "24",
        "reuse_penalties": "40",
        "screened_cells": "4",
        "rp_better_cypress_cells": str(better),
        "rp_equal_cypress_cells": str(4 - better - no_fit),
        "rp_worse_cypress_cells": "0",
        "rp_feasible_cypress_no_fit_cells": str(no_fit),
    }


def _screen_rows(tag: str, geometry_cost: int, rp_cost: int) -> list[dict[str, str]]:
    rows = []
    for capacity in ("tight", "native"):
        for arm, cost in (("geometry_ff", geometry_cost), ("cypress", rp_cost), ("dsa_rp_cg", rp_cost)):
            rows.append(
                {
                    "tag": tag,
                    "pool_id": "1",
                    "pool": "Vec",
                    "capacity_label": capacity,
                    "arm": arm,
                    "status": "feasible",
                    "reuse_cost": str(cost),
                    "placement_sha256": f"{arm}-placement",
                }
            )
    return rows


def _write_native_solutions(screen_root: Path, tag: str) -> None:
    for arm in selector._ARMS:
        path = screen_root / "raw" / tag / "pool-1-Vec" / "native" / arm / "solution.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")


def _write_problem(path: Path, instance: str = "kernel") -> None:
    path.write_text(
        json.dumps(
            {
                "instance": instance,
                "problem": {
                    "buffers": [{} for _ in range(24)],
                    "cost_model": {"reuse_penalties": [{} for _ in range(40)]},
                    "pools": [{"id": 1, "name": "Vec", "capacity": 188416}],
                },
            }
        ),
        encoding="utf-8",
    )


def test_selects_model_positive_large_gap_and_forced_candidates(tmp_path: Path) -> None:
    screen_root = tmp_path / "screen"
    problems = tmp_path / "problems"
    problems.mkdir()
    tags = (
        "positive-aaaaaaaaaaaaaaaa",
        "large-bbbbbbbbbbbbbbbb",
        "forced-cccccccccccccccc",
        "no-fit-dddddddddddddddd",
    )
    for tag in tags:
        _write_problem(problems / f"{tag}.dsa.json")
        _write_native_solutions(screen_root, tag)
    candidates = selector.build_candidate_rows(
        selector.CandidateInputs(
            separation_rows=[
                _separation(tags[0], better=1),
                _separation(tags[1]),
                _separation(tags[2]),
                _separation(tags[3], no_fit=1),
            ],
            screen_rows=_screen_rows(tags[0], 10, 9)
            + _screen_rows(tags[1], 30, 5)
            + _screen_rows(tags[2], 4, 4)
            + _screen_rows(tags[3], 4, 4),
            invocations=[
                {"problem_fingerprint": tag.rsplit("-", 1)[-1], "script": f"models/{index}.py"}
                for index, tag in enumerate(tags)
            ],
            problems_dir=problems,
            screen_root=screen_root,
            min_geometry_advantage=20,
            forced={tags[2]: "historical_winner"},
        )
    )
    by_tag = {row["tag"]: row for row in candidates}
    assert set(by_tag) == set(tags)
    assert by_tag[tags[0]]["selection_reasons"] == "rp_beats_cypress"
    assert by_tag[tags[1]]["selection_reasons"] == "geometry_gap_ge_20"
    assert by_tag[tags[2]]["selection_reasons"] == "prior_evidence:historical_winner"
    assert by_tag[tags[3]]["selection_reasons"] == "cypress_no_fit"
    assert by_tag[tags[0]]["pool_names"] == "Vec"
    assert len(by_tag[tags[0]]["problem_sha256"]) == _SHA256_HEX_LENGTH
    assert len(by_tag[tags[0]]["native_geometry_ff_solution_sha256"]) == _SHA256_HEX_LENGTH
    assert len(by_tag[tags[0]]["structural_class_sha256"]) == _SHA256_HEX_LENGTH
    assert sum(row["structural_class_representative"] for row in candidates) == 1


def test_prefers_smallest_source_program_for_device_preflight(tmp_path: Path) -> None:
    tag = "positive-aaaaaaaaaaaaaaaa"
    screen_root = tmp_path / "screen"
    problems = tmp_path / "problems"
    problems.mkdir()
    _write_problem(problems / f"{tag}.dsa.json")
    _write_native_solutions(screen_root, tag)
    invocations = [
        {"problem_fingerprint": tag.rsplit("-", 1)[-1], "script": "models/large.py"},
        {"problem_fingerprint": tag.rsplit("-", 1)[-1], "script": "models/small.py"},
        {"problem_fingerprint": "another", "script": "models/large.py"},
    ]
    candidates = selector.build_candidate_rows(
        selector.CandidateInputs(
            separation_rows=[_separation(tag, better=1)],
            screen_rows=_screen_rows(tag, 10, 9),
            invocations=invocations,
            problems_dir=problems,
            screen_root=screen_root,
            min_geometry_advantage=20,
            forced={},
        )
    )
    assert candidates[0]["preferred_source_script"] == "models/small.py"
    assert candidates[0]["preferred_source_problem_count"] == 1
    assert selector._preferred_parent_rows(candidates) == [
        {
            "source_script": "models/small.py",
            "candidate_count": 1,
            "candidate_tags": tag,
            "measurement_state": "NEEDS_PARENT_AND_DISPATCH_PREFLIGHT",
        }
    ]


def test_rejects_native_placements_that_depend_on_screened_pool(tmp_path: Path) -> None:
    tag = "case-aaaaaaaaaaaaaaaa"
    screen_root = tmp_path / "screen"
    cells = []
    for pool_id, digest in ((1, "first"), (2, "second")):
        arms = {}
        for arm in selector._ARMS:
            arms[arm] = {
                "tag": tag,
                "pool_id": str(pool_id),
                "pool": "Vec",
                "capacity_label": "native",
                "arm": arm,
                "status": "feasible",
                "placement_sha256": f"{arm}-{digest}",
            }
        cells.append(arms)
    with pytest.raises(ValueError, match="changes with screened pool"):
        selector._native_solution_paths(screen_root, tag, cells)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
