# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the compile-only library CI gate."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[2] / "ci/compile_pypto_lib.py"
_spec = importlib.util.spec_from_file_location("compile_pypto_lib", _RUNNER)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "lib"
    (lib / "models").mkdir(parents=True)
    (lib / "golden").mkdir()
    (lib / "golden/__init__.py").write_text("from .runner import run\n")
    (lib / "golden/runner.py").write_text(
        "from types import SimpleNamespace\n"
        "def run(fn, specs, compile_only=False, runtime_dir=None):\n"
        "    assert compile_only and runtime_dir is None\n"
        "    if fn == 'raise':\n"
        "        raise ValueError('Compiler failed')\n"
        "    return SimpleNamespace(passed=fn != 'bad', error='bad compile' if fn == 'bad' else None)\n"
        "def _prepare_inputs():\n"
        "    raise AssertionError('Device inputs must never be prepared')\n"
    )
    (lib / "pypto/ir").mkdir(parents=True)
    (lib / "pypto/__init__.py").write_text("")
    (lib / "pypto/ir/__init__.py").write_text("from .compile import compile\n")
    (lib / "pypto/ir/compile.py").write_text(
        "def compile(fn, *, skip_ptoas=True):\n"
        "    assert not skip_ptoas, 'PTOAS must be mandatory'\n"
        "    if fn == 'bad':\n"
        "        raise ValueError('PTOAS failed')\n"
        "    return object()\n"
    )
    return lib


@pytest.mark.parametrize(
    ("body", "passed", "count"),
    [
        ("from golden import run\nrun('ok', [])", True, 1),
        ("from golden.runner import run\nrun('ok', [], False, 'cached')", True, 1),
        ("from pypto.ir import compile\ncompile('ok', skip_ptoas=True)", True, 1),
        ("from pypto.ir.compile import compile\ncompile('ok')", True, 1),
        ("from pypto.ir import compile\ncompile('bad')", False, 1),
        ("from golden import run\nrun('bad', [])", False, 1),
        ("from golden import run\nrun('bad', [])\nrun('ok', [])", False, 2),
        (
            "from golden import run\ntry:\n    run('raise', [])\nexcept ValueError:\n    pass\nrun('ok', [])",
            False,
            2,
        ),
        ("raise SystemExit(0)", False, 0),
        ("from golden import run\nrun('ok', [])\nraise SystemExit(3)", False, 1),
        ("raise RuntimeError('broken import')", False, 0),
        ("from golden.runner import _prepare_inputs\n_prepare_inputs()", False, 0),
    ],
)
def test_compile_receipts_gate_success(library, tmp_path, body, passed, count):
    (library / "models/case.py").write_text(body)
    result = audit.run_case(library, "models/case.py", "a2a3", tmp_path / "out", 10)
    assert result["passed"] is passed
    assert len(result["compilations"]) == count


def test_timeout_cannot_reuse_previous_receipt(library, tmp_path):
    script = library / "models/case.py"
    output = tmp_path / "out"
    script.write_text("from golden import run\nrun('ok', [])")
    assert audit.run_case(library, "models/case.py", "a5", output, 10)["passed"]
    script.write_text("import time\ntime.sleep(60)")
    result = audit.run_case(library, "models/case.py", "a5", output, 1)
    assert not result["passed"]
    assert result["timed_out"]
    assert result["compilations"] == []


@pytest.mark.parametrize("receipt", ["[", "null", "{}", "[{}]", '[{"passed": "true"}]'])
def test_invalid_receipt_fails_without_aborting(library, tmp_path, receipt):
    (library / "models/case.py").write_text(
        "from golden import run\nrun('ok', [])\n"
        f"import json\njson.dumps = lambda *args, **kwargs: {receipt!r}\n"
    )
    result = audit.run_case(library, "models/case.py", "a2a3", tmp_path / "out", 10)
    assert result["exit_code"] == 0
    assert not result["passed"]
    assert result["compilations"] == []


def test_unreadable_receipt_fails_without_aborting(library, tmp_path, monkeypatch):
    (library / "models/case.py").write_text("from golden import run\nrun('ok', [])")
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.name == "compilations.json":
            raise OSError("Receipt is unreadable")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert result["exit_code"] == 0
    assert not result["passed"]
    assert result["compilations"] == []


def test_entry_arguments_preserve_declared_ranks(tmp_path):
    script = tmp_path / "distributed.py"
    script.write_text('# ci: devices=4\nparser.add_argument("--compile-only")\n')
    assert audit.entry_args(script, "a5") == [
        str(script),
        "-p",
        "a5",
        "-d",
        "0,1,2,3",
        "--compile-only",
    ]


def test_required_device_argument_without_marker(tmp_path):
    script = tmp_path / "distributed.py"
    script.write_text('parser.add_argument("-d", "--device", required=True)\n')
    assert audit.entry_args(script, "a5") == [str(script), "-p", "a5", "-d", "0"]


@pytest.mark.parametrize("entries", [[], ["models/case_draft.py"], ["../outside.py"]])
def test_empty_or_invalid_selection_fails(library, entries):
    selector = library / ".github/scripts/detect_changes.py"
    selector.parent.mkdir(parents=True)
    selector.write_text(f"print({' '.join(entries)!r})\n")
    with pytest.raises(ValueError, match="No formal|Invalid formal"):
        audit.select_entries(library, "a2a3")


def test_selector_failure_is_not_empty_success(library):
    selector = library / ".github/scripts/detect_changes.py"
    selector.parent.mkdir(parents=True)
    selector.write_text("raise SystemExit(2)\n")
    with pytest.raises(audit.subprocess.CalledProcessError):
        audit.select_entries(library, "a5")


@pytest.mark.parametrize(("kernel", "status"), [("ok", 0), ("bad", 1), ("truncated", 1)])
def test_manifest_uses_library_selector(library, tmp_path, monkeypatch, kernel, status):
    selector = library / ".github/scripts/detect_changes.py"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        "import sys\nassert sys.argv[1:] == ['--all', '--platform', 'a5']\n"
        "print('models/case.py models/glm5_3_flash/weights.py models/glm5_3_flash/prefill_ref.py')\n"
    )
    body = f"from golden import run\nrun({kernel!r}, [])"
    if kernel == "truncated":
        body += "\nimport json\njson.dumps = lambda *args, **kwargs: '['\n"
    (library / "models/case.py").write_text(body)
    audit.subprocess.run(["git", "init", str(library)], check=True, capture_output=True)
    audit.subprocess.run(
        [
            "git",
            "-C",
            str(library),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--allow-empty",
            "-m",
            "Test",
        ],
        check=True,
        capture_output=True,
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_RUNNER),
            "--lib",
            str(library),
            "--platform",
            "a5",
            "--output",
            str(output),
            "--jobs",
            "1",
        ],
    )
    assert audit.main() == status
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["entries"] == ["models/case.py"]
    assert len(manifest["lib_revision"]) == 40
    assert json.loads((output / "results.json").read_text())[0]["passed"] is (status == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
