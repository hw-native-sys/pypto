# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for diagnostic reporting and process isolation; no PyPTO import needed."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).with_name("run_urma_mfe.py")
SPEC = importlib.util.spec_from_file_location("urma_mfe_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ALL_SPEC = importlib.util.spec_from_file_location("urma_all_cards", SCRIPT.with_name("run_all_cards.py"))
all_cards = importlib.util.module_from_spec(ALL_SPEC)
ALL_SPEC.loader.exec_module(all_cards)


def test_pairs_cover_every_card_without_crossing_the_four_card_groups():
    text = "Total NPU count: 8\n" + "\n".join(f"NPU ID : {card}" for card in range(8))
    assert all_cards.card_pairs(text) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_odd_final_card_is_not_silently_omitted():
    assert all_cards.card_pairs("NPU ID : 0\nNPU ID : 1\nNPU ID : 2") == [[0, 1], [1, 2]]


def test_missing_inventory_fails_instead_of_testing_no_cards():
    with pytest.raises(RuntimeError, match="two discoverable"):
        all_cards.card_pairs("query failed")


def test_all_pairs_are_submitted_and_failed_pair_keeps_job_red(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_all_cards.py", "--expected-soc", "PR", "--output", str(tmp_path)])
    submitted = []

    def execute(command, **kwargs):
        if command[0] == "npu-smi":
            return SimpleNamespace(
                returncode=0, stdout="\n".join(f"NPU ID : {i}" for i in range(4)), stderr=""
            )
        if command[0] == "task-submit":
            pair = command[command.index("--device") + 1]
            submitted.append(pair)
            shell = command[-1]
            assert str(all_cards.ROOT) in shell  # cancellation cleanup matches this absolute path
            assert "source activate.sh" not in shell  # preserve warmup's loader fingerprint
            return SimpleNamespace(returncode=1 if pair == "0,1" else 0)
        assert command[1] == "tests/diagnostics/prepare_urma_mfe.py"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(all_cards.subprocess, "run", execute)
    assert all_cards.main() == 1
    assert sorted(submitted) == ["0,1", "2,3"]
    assert len(json.loads((tmp_path / "pair-results.json").read_text())) == 2


@pytest.mark.parametrize(
    "body,passed",
    [
        ('<testcase name="transfer"/>', True),
        ('<testcase name="transfer"><skipped/></testcase>', False),
        ('<testcase name="transfer"><failure/></testcase>', False),
        ('<testcase name="transfer"><error/></testcase>', False),
        ("", False),
        ("<testcase/><testcase/>", False),
    ],
)
def test_junit_requires_exactly_one_executed_success(tmp_path, body, passed):
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuite>{body}</testsuite>")
    assert runner.validate_junit(path) is passed


def test_success_exit_without_execution_is_failure(tmp_path):
    result = runner.run_case([sys.executable, "-c", "print('no tests')"], tmp_path, tmp_path / "case", {})
    assert result["returncode"] == 0
    assert not result["passed"]


def test_generic_board_name_resolves_class_from_installed_sku(tmp_path):
    config = tmp_path / "x86_64-linux/data/platform_config/Ascend950PR_9579.ini"
    config.parent.mkdir(parents=True)
    config.touch()
    board = {"returncode": 0, "stdout": "Chip Name : Ascend950\nNPU Name : 9579\n"}
    assert runner.board_matches(board, "PR", tmp_path)
    assert not runner.board_matches(board, "DT", tmp_path)


def test_outer_timeout_is_recorded_and_process_is_reaped(tmp_path):
    result = runner.run_case(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        tmp_path / "case",
        {},
        timeout=0.05,
    )
    assert result["timed_out"]
    assert not result["passed"]
    assert result["returncode"] != 0
    assert json.loads((tmp_path / "case/result.json").read_text())["timed_out"]


def test_all_attempts_run_and_an_earlier_failure_stays_red(tmp_path, monkeypatch):
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--devices", "0,1", "--expected-soc", "PR", "--output", str(output)]
    )
    monkeypatch.setenv("TASK_DEVICE", "0,1")

    def capture(command, cwd=runner.ROOT):
        text = "revision\n"
        if command[0] == "npu-smi":
            text = "NPU Name: Ascend950PR\n"
        if command[0] == sys.executable:
            text = json.dumps({"simpler_build": "revision"})
        return {"command": command, "returncode": 0, "stdout": text, "stderr": ""}

    attempted = []

    def execute(command, cwd, directory, env):
        directory.mkdir(parents=True)
        attempted.append(directory.name)
        passed = len(attempted) != 1
        return {"passed": passed, "returncode": 0 if passed else 1, "timed_out": False}

    monkeypatch.setattr(runner, "capture", capture)
    monkeypatch.setattr(runner, "run_case", execute)
    assert runner.main() == 1
    assert attempted == [name for name, _, _ in runner.CASES] * 3
    rows = json.loads((output / "results.json").read_text())
    assert len(rows) == 15
    assert not rows[0]["passed"] and rows[5]["passed"] and rows[10]["passed"]
    assert "pypto-put: 2/3 passed" in (output / "summary.txt").read_text()


def test_only_diagnostic_branch_suppresses_normal_ci():
    root = SCRIPT.parents[2]
    branch = "diag/a5-urma-mfe-3085"
    for file, job in [("ci.yml", "changes"), ("ci-full.yml", "changes"), ("docs.yml", "build")]:
        workflow = yaml.safe_load((root / ".github/workflows" / file).read_text())
        assert workflow["jobs"][job]["if"] == f"github.head_ref != '{branch}'"
    workflow = yaml.safe_load((root / ".github/workflows/urma-mfe.yml").read_text())
    assert workflow["jobs"]["toolchain"]["if"] == f"github.head_ref == '{branch}'"
    matrix = workflow["jobs"]["onboard-a5-distributed"]["strategy"]["matrix"]["include"]
    assert matrix == [{"soc": "PR", "arch": "x64"}, {"soc": "DT", "arch": "arm64"}]
