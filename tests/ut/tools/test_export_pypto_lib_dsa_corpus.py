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
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "export_pypto_lib_dsa_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_export_pypto_lib_dsa_corpus", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def _problem(*, instance: str, members: list[str], penalties: int = 1) -> dict:
    return {
        "instance": instance,
        "metadata": {"reuse_penalty_recognizer": "quadratic_route_frontier_v3"},
        "problem": {
            "pools": [{"id": 1, "name": "Vec", "capacity": 1024}],
            "buffers": [{"id": 0, "size": 32, "alignment": 32, "allowed_pools": [1]}],
            "constraints": {},
            "cost_model": {
                "reuse_penalties": [{"first": 0, "second": 1, "weight": 1} for _ in range(penalties)]
            },
            "pypto_structure": {"alias_classes": [{"members": members}]},
        },
    }


def test_semantic_fingerprint_ignores_only_alias_member_spelling() -> None:
    first = _problem(instance="kernel", members=["%inline1", "%inline2"])
    renamed = _problem(instance="kernel", members=["%inline8", "%inline9"])
    resized = _problem(instance="kernel", members=["%inline1", "%inline2"])
    resized["problem"]["buffers"][0]["size"] = 64

    assert exporter.semantic_problem_fingerprint(first) == exporter.semantic_problem_fingerprint(renamed)
    assert exporter.semantic_problem_fingerprint(first) != exporter.semantic_problem_fingerprint(resized)


def test_manifest_selects_only_successful_exports(tmp_path: Path) -> None:
    manifest = tmp_path / "status.tsv"
    manifest.write_text(
        "models/a.py\tEXPORTED\tsim\t0\t3\n"
        "models/b.py\tRESOURCE_BLOCKED\tsim\t200\t0\n"
        "models/c.py\tEXPORTED\tsim\t0\t2\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [
        ("models/a.py", "a2a3sim"),
        ("models/c.py", "a2a3sim"),
    ]


def test_export_plan_preserves_real_platform_launch_mode(tmp_path: Path) -> None:
    manifest = tmp_path / "status.tsv"
    manifest.write_text(
        "models/sim.py\tEXPORTED\tfinal-sim\t0\t3\n"
        "models/a2a3.py\tEXPORTED\tfinal-a2a3\t0\t2\n"
        "models/a5.py\tEXPORTED\tfinal-a5\t0\t1\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [
        ("models/sim.py", "a2a3sim"),
        ("models/a2a3.py", "a2a3"),
        ("models/a5.py", "a5"),
    ]


def test_export_status_launch_mode_is_reusable(tmp_path: Path) -> None:
    manifest = tmp_path / "export-status.tsv"
    manifest.write_text(
        "script\tstatus\tlaunch_mode\treturncode\tproblems\nmodels/a2a3.py\tEXPORTED\tfinal-a2a3\t0\t2\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [("models/a2a3.py", "a2a3")]


def test_golden_patch_injects_compile_only_dsa_configuration(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_run(*_args, **kwargs):
        calls.append(kwargs)
        return "ok"

    golden = SimpleNamespace(run=fake_run, run_jit=fake_run)
    passes = SimpleNamespace(
        MemoryPlanner=SimpleNamespace(DSA="dsa"),
        DsaReusePenaltyRecognizer=SimpleNamespace(QUADRATIC="quadratic"),
    )
    exporter._patch_golden(golden, passes, tmp_path)

    assert golden.run(compile_cfg={"dump_passes": True}) == "ok"
    assert golden.run_jit() == "ok"
    direct, jit = calls
    assert direct["compile_only"] is True
    assert direct["save_data"] is False
    assert direct["compile_cfg"]["memory_planner"] == "dsa"
    assert direct["compile_cfg"]["dsa_reuse_penalty_recognizer"] == "quadratic"
    assert direct["compile_cfg"]["skip_ptoas"] is True
    assert direct["compile_cfg"]["dump_passes"] is False
    assert jit["compile_cfg"]["codegen_only"] is True
    assert "skip_ptoas" not in jit["compile_cfg"]
    assert (tmp_path / "call-001").is_dir()
    assert (tmp_path / "call-002").is_dir()


def test_inventory_deduplicates_semantic_exports(tmp_path: Path) -> None:
    first_root = tmp_path / "captures" / "first"
    second_root = tmp_path / "captures" / "second"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    first = _problem(instance="kernel", members=["%inline1"], penalties=1)
    second = _problem(instance="kernel", members=["%inline7"], penalties=1)
    (first_root / "first.dsa.json").write_text(json.dumps(first), encoding="utf-8")
    (second_root / "second.dsa.json").write_text(json.dumps(second), encoding="utf-8")

    invocations, unique, penalties = exporter.build_inventory(
        tmp_path,
        {first_root: "models/a.py", second_root: "models/b.py"},
    )

    assert (invocations, unique, penalties) == (2, 1, 1)
    assert len(list((tmp_path / "corpus" / "penalty-bearing").glob("*.dsa.json"))) == 1
    rows = (tmp_path / "invocations.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
