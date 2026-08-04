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
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "prepare_dsa_device_panel.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_prepare_dsa_device_panel", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
panel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(panel)


def _panel() -> dict:
    kernels = [
        {
            "tag": "winner",
            "program": "program_a",
            "kernel": "function_a",
            "selection_class": "historical_winner",
            "problem": "winner.json",
        },
        {
            "tag": "inventory",
            "program": "program_b",
            "kernel": "function_b",
            "selection_class": "broad_inventory",
            "problem": "inventory.json",
        },
    ]
    return {
        "schema_version": 1,
        "selection_policy": "all_current_eligible_plus_historical_winners_v1",
        "recognizer": {"policy": "quadratic_v0", "source_sha256": "abc123"},
        "kernels": kernels,
    }


def test_load_panel_enforces_broad_selection_contract(tmp_path: Path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_panel()), encoding="utf-8")
    loaded = panel.load_panel(path)
    assert len(loaded["kernels"]) == 2

    invalid = _panel()
    invalid["kernels"][0]["selection_class"] = "broad_inventory"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="historical-winner"):
        panel.load_panel(path)


def test_solver_commands_freeze_three_distinct_arms(tmp_path: Path):
    commands = {
        arm: panel.solver_command(
            tmp_path / "dsa-bench",
            tmp_path / "problem.json",
            tmp_path / f"{arm}.solution.json",
            tmp_path / f"{arm}.result.json",
            arm,
        )
        for arm, _ in panel.ARMS
    }
    assert set(commands) == {"geometry_ff", "cypress", "dsa_rp_cg"}
    assert "geometry-first-fit" in commands["geometry_ff"]
    assert "cypress-relaxation" in commands["cypress"]
    assert "canonical-greedy" in commands["dsa_rp_cg"]
    assert len({tuple(command) for command in commands.values()}) == 3


def test_freeze_panel_records_problem_and_edge_provenance(tmp_path: Path):
    problem = {
        "instance": "winner_instance",
        "profile": "pypto_research_v1",
        "metadata": {"reuse_penalty_recognizer": "quadratic_route_frontier_v3"},
        "problem": {
            "buffers": [{"id": 0}, {"id": 1}],
            "cost_model": {
                "reuse_penalties": [{"first": 0, "second": 1, "weight": 1, "reason": "cross_pipe"}]
            },
        },
    }
    (tmp_path / "winner.json").write_text(json.dumps(problem), encoding="utf-8")
    (tmp_path / "inventory.json").write_text(json.dumps(problem), encoding="utf-8")
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(_panel()), encoding="utf-8")

    frozen = panel.freeze_panel(panel_path, panel.load_panel(panel_path))

    winner = frozen["kernels"][0]
    assert frozen["panel_source_sha256"] == panel._sha256(panel_path)
    assert winner["problem_sha256"] == panel._sha256(tmp_path / "winner.json")
    assert winner["problem_instance"] == "winner_instance"
    assert winner["problem_profile"] == "pypto_research_v1"
    assert winner["exported_edge_policy"] == "quadratic_route_frontier_v3"
    assert winner["buffers"] == 2
    assert winner["reuse_penalties"] == 1
