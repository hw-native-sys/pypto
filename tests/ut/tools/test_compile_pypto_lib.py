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


@pytest.mark.parametrize(
    ("cases", "passed", "count"),
    [
        (["ok", "ok"], True, 2),
        (["ok", "bad"], False, 2),
        (["empty"], False, 1),
        (["caught"], False, 1),
        (["invalid"], False, 1),
        (["exit"], False, 1),
        (["partial_exit"], False, 2),
        (["nested"], True, 1),
    ],
)
@pytest.mark.parametrize(
    "launch",
    [
        "subprocess.run(command, check=False)",
        "subprocess.check_call(command)",
        "subprocess.Popen(command).wait()",
    ],
)
def test_python_subprocess_compilations_are_audited(library, tmp_path, cases, passed, count, launch):
    # Formal entries can isolate cases in fresh interpreters to own runtime lifetimes.
    (library / "models/case.py").write_text(
        "import argparse, json, subprocess, sys\n"
        "from golden import run\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-p')\n"
        "parser.add_argument('-d', default='0,1')\n"
        "parser.add_argument('--case', default='all')\n"
        "parser.add_argument('--compile-only', action='store_true')\n"
        "args = parser.parse_args()\n"
        "if args.case in ('all', 'nested'):\n"
        f"    cases = {cases!r} if args.case == 'all' else ['ok']\n"
        "    for case in cases:\n"
        "        command = [sys.executable, __file__, '-p', args.p, '-d', '0',\n"
        "                   '--case', case, '--compile-only']\n"
        f"        {launch}\n"
        "else:\n"
        "    assert args.d == '0' and args.compile_only\n"
        "    if args.case == 'exit':\n"
        "        raise SystemExit(3)\n"
        "    if args.case == 'caught':\n"
        "        try:\n"
        "            run('raise', [], compile_only=args.compile_only)\n"
        "        except ValueError:\n"
        "            pass\n"
        "    elif args.case != 'empty':\n"
        "        run(args.case, [], compile_only=args.compile_only)\n"
        "        if args.case == 'invalid':\n"
        "            json.dumps = lambda *args, **kwargs: '['\n"
        "        if args.case == 'partial_exit':\n"
        "            raise SystemExit(3)\n"
    )
    result = audit.run_case(library, "models/case.py", "a2a3", tmp_path / "out", 10)
    assert result["passed"] is passed
    assert len(result["compilations"]) == count


def test_python_subprocess_preserves_other_commands(library, tmp_path):
    (library / "models/helper.py").write_text("raise SystemExit(0)\n")
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "subprocess.run([sys.executable, '-c', 'raise SystemExit(0)'], check=True)\n"
        "subprocess.run([sys.executable, str(Path(__file__).with_name('helper.py'))], check=True)\n"
        "subprocess.run([b'true'], check=True)\n"
        "run('ok', [])\n"
    )
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


def test_python_subprocess_preserves_popen_options(library, tmp_path):
    (library / "models/case.py").write_text(
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "if '--case' not in sys.argv:\n"
        "    command = [sys.executable, 'case.py', '--case', 'child', '-d', '0']\n"
        "    with subprocess.Popen(args=command, cwd=Path(__file__).parent,\n"
        "                          env=os.environ | {'AUDIT_CHILD': 'preserved'},\n"
        "                          stdout=subprocess.PIPE, text=True) as child:\n"
        "        output, _ = child.communicate()\n"
        "        assert child.returncode == 0 and output.strip() == 'child-output'\n"
        "        assert child.args == command\n"
        "else:\n"
        "    assert sys.argv[1:] == ['--case', 'child', '-d', '0']\n"
        "    assert Path.cwd() == Path(__file__).parent\n"
        "    assert os.environ['AUDIT_CHILD'] == 'preserved'\n"
        "    run('ok', [])\n"
        "    print('child-output')\n"
    )
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


def test_empty_child_receipt_cannot_reuse_previous_success(library, tmp_path, monkeypatch):
    (library / "models/case.py").write_text(
        "import os, subprocess, sys\n"
        "from golden import run\n"
        "if '--child' not in sys.argv:\n"
        "    subprocess.run([sys.executable, __file__, '--child'], check=True)\n"
        "elif os.environ.get('AUDIT_EMPTY_CHILD') != '1':\n"
        "    run('ok', [])\n"
    )
    output = tmp_path / "out"
    assert audit.run_case(library, "models/case.py", "a5", output, 10)["passed"]
    monkeypatch.setenv("AUDIT_EMPTY_CHILD", "1")
    result = audit.run_case(library, "models/case.py", "a5", output, 10)
    assert not result["passed"]
    assert len(result["compilations"]) == 1


def test_caught_child_start_failure_cannot_pass(library, tmp_path):
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "try:\n"
        "    subprocess.run([sys.executable, __file__], cwd=Path(__file__).parent / 'missing')\n"
        "except OSError:\n"
        "    pass\n"
        "run('ok', [])\n"
    )
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert not result["passed"]
    assert len(result["compilations"]) == 2


@pytest.mark.parametrize("keyword_args", [False, True])
def test_self_dispatch_collects_each_case(library, tmp_path, keyword_args):
    invocation = "args=command" if keyword_args else "command"
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "cases = ['golden', 'runner', 'compile', 'module', 'guard', 'last']\n"
        "if '--case' not in sys.argv:\n"
        "    for case in cases:\n"
        "        command = [sys.executable, str(Path(__file__).resolve()),\n"
        "                   '--case', case, '-d', '3,5', '--label', 'two words']\n"
        f"        subprocess.run({invocation}, check=True)\n"
        "else:\n"
        "    case = sys.argv[2]\n"
        "    assert sys.argv[1:] == ['--case', case, '-d', '3,5', '--label', 'two words']\n"
        "    if case in ('golden', 'last'):\n"
        "        from golden import run\n"
        "        run('ok', [], False, 'cached')\n"
        "    elif case == 'runner':\n"
        "        from golden.runner import run\n"
        "        run('ok', [])\n"
        "    elif case == 'compile':\n"
        "        from pypto.ir import compile\n"
        "        compile('ok', skip_ptoas=True)\n"
        "    elif case == 'module':\n"
        "        from pypto.ir.compile import compile\n"
        "        compile('ok')\n"
        "    else:\n"
        "        from golden.runner import _prepare_inputs, run\n"
        "        try:\n"
        "            _prepare_inputs()\n"
        "        except RuntimeError as error:\n"
        "            assert 'prepare runtime inputs' in str(error)\n"
        "        else:\n"
        "            raise AssertionError('Runtime guard was not installed')\n"
        "        run('ok', [])\n"
    )
    result = audit.run_case(library, "models/case.py", "a2a3", tmp_path / "out", 20)
    assert result["passed"]
    assert len(result["compilations"]) == 6


@pytest.mark.parametrize(
    "child_body",
    [
        "from golden import run; run('bad', [])",
        "from golden import run; run('ok', []); raise SystemExit(3)",
        "raise SystemExit(0)",
        "import os; os._exit(0)",
        "from golden import run; run('ok', []); import json; json.dumps = lambda *a, **kw: '['",
    ],
)
def test_self_dispatch_cannot_hide_failed_or_unobserved_child(library, tmp_path, child_body):
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "if '--child' in sys.argv:\n"
        f"    {child_body}\n"
        "else:\n"
        "    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'], check=False)\n"
        "    from golden import run\n"
        "    run('ok', [])\n"
    )
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert result["exit_code"] == 0
    assert not result["passed"]
    assert any(not record["passed"] for record in result["compilations"])
    assert any(record["passed"] for record in result["compilations"])


def test_unrelated_subprocess_preserves_arguments_and_output(library, tmp_path):
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "result = subprocess.run([sys.executable, '-c', 'print(42)'],\n"
        "                        check=True, capture_output=True, text=True)\n"
        "assert result.stdout == '42\\n'\n"
        "from golden import run\n"
        "run('ok', [])\n"
    )
    result = audit.run_case(library, "models/case.py", "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


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


@pytest.mark.parametrize(
    ("child", "passed"), [("ok", True), ("bad", False), ("empty", False), ("exit", False)]
)
def test_self_spawned_entry_preserves_audit_and_child_arguments(library, tmp_path, child, passed):
    # Mirrors entries that dispatch --case=all into independent processes.
    # The wrapper must preserve each child's arguments and retain failures
    # even when the parent ignores its exit status or reports success.
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "if '--child' not in sys.argv:\n"
        "    run('ok', [])\n"
        "    subprocess.run([sys.executable, __file__, '--child', 'ok'], check=True)\n"
        f"    subprocess.run(args=[sys.executable, Path(__file__), '--child', {child!r}], check=False)\n"
        "else:\n"
        "    assert sys.argv[1] == '--child'\n"
        "    if sys.argv[2] != 'empty':\n"
        "        run(sys.argv[2], [])\n"
        "    if sys.argv[2] == 'exit':\n"
        "        raise SystemExit(3)\n"
    )
    result = audit.run_case(library, "models/case.py", "a2a3", tmp_path / "out", 10)
    assert result["exit_code"] == 0
    assert result["passed"] is passed
    statuses = [record["passed"] for record in result["compilations"]]
    assert statuses == ([True, True, True, False] if child == "exit" else [True, True, passed])


def test_cpu_prefill_entry_without_compilations_still_fails(library, tmp_path):
    # Bypassing selection must not turn a successful CPU check into kernel coverage.
    entry = "models/glm5_3_flash/prefill_fwd.py"
    path = library / entry
    path.parent.mkdir(parents=True)
    path.write_text("raise SystemExit(0)\n")
    output = tmp_path / "out"
    result = audit.run_case(library, entry, "a2a3", output, 10)
    assert result["exit_code"] == 0
    assert not result["passed"]
    assert result["compilations"] == []
    assert not list((output / Path(entry).with_suffix("")).glob("runtime-*"))


@pytest.mark.parametrize("entry", ["hc", "prefill_fwd", "sparse_attention"])
def test_compile_does_not_enable_replay_or_checkpoint_arguments(library, tmp_path, entry):
    # Mirror the A5 contracts: replay arguments change the default all-cases
    # mode, while prefill declares runtime-dir only in an inactive CLI branch.
    directory = library / "models/deepseek_v4_1_flash"
    directory.mkdir()
    (directory / f"{entry}.py").write_text(
        "import argparse\n"
        "from pypto.ir import compile\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-p')\n"
        "parser.add_argument('--compile-only', action='store_true')\n"
        + (
            "if False:\n    parser.add_argument('--runtime-dir')\n"
            if entry == "prefill_fwd"
            else "parser.add_argument('--runtime-dir')\n"
        )
        + "args = parser.parse_args()\n"
        "assert getattr(args, 'runtime_dir', None) is None, 'replay was enabled'\n"
        "assert args.compile_only\n"
        "compile('ok')\n"
    )
    result = audit.run_case(library, f"models/deepseek_v4_1_flash/{entry}.py", "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


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


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("include_kernel", [False, True])
def test_cpu_reference_entries_do_not_count_as_compilation_coverage(library, include_kernel, platform):
    # These CPU-only self-tests can pass without compiling a PyPTO kernel.
    entries = [
        "models/glm5_3_flash/weights.py",
        "models/glm5_3_flash/prefill_ref.py",
        "models/glm5_3_flash/prefill_fwd.py",
    ]
    for entry in entries:
        path = library / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise SystemExit(0)\n")
    if include_kernel:
        # Real GLM kernels need coverage even when they lack --compile-only.
        kernels = ["attention", "kda", "prefill_layer"]
        for kernel in kernels:
            entry = f"models/glm5_3_flash/{kernel}.py"
            entries.append(entry)
            (library / entry).write_text("from golden import run\nrun('ok', [])")
    selector = library / ".github/scripts/detect_changes.py"
    selector.parent.mkdir(parents=True)
    selector.write_text(f"print({' '.join(entries)!r})\n")

    if include_kernel:
        assert audit.select_entries(library, platform) == [
            "models/glm5_3_flash/attention.py",
            "models/glm5_3_flash/kda.py",
            "models/glm5_3_flash/prefill_layer.py",
        ]
    else:
        with pytest.raises(ValueError, match="No formal pypto-lib entries"):
            audit.select_entries(library, platform)


@pytest.mark.parametrize("second,passed", [("ok", True), ("bad", False), (None, False)])
def test_glm_reference_cli_audits_kernel_signatures(library, tmp_path, second, passed):
    entry = library / "models/glm5_3_flash/prefill_fwd.py"
    entry.parent.mkdir(parents=True)
    (library / "pypto/backend.py").write_text(
        "class BackendType:\n    Ascend910 = 'a2a3'\n    Ascend950 = 'a5'\n"
    )
    (library / "pypto/runtime.py").write_text(
        "class RunConfig:\n"
        "    def __init__(self, *, backend_type):\n"
        "        self.backend_type = backend_type\n"
    )
    entry.write_text(
        "from pypto.ir import compile\n"
        "class Kernel:\n"
        "    def __init__(self, value):\n        self.value = value\n"
        "    def compile(self, *, config):\n"
        "        assert config.backend_type == 'a2a3'\n"
        "        compile(self.value)\n"
        "l3_copy_request_snapshot = Kernel('ok')\n"
        f"l3_reset_request_caches = Kernel({second!r}) if {second!r} is not None else None\n"
        "if __name__ == '__main__':\n"
        "    raise AssertionError('CPU reference CLI must not stand in for kernel compilation')\n"
    )
    result = audit.run_case(library, "models/glm5_3_flash/prefill_fwd.py", "a2a3", tmp_path / "out", 30)
    assert result["passed"] is passed
    assert len(result["compilations"]) == (1 if second is None else 2)


def test_glm_legacy_entry_keeps_cli_compilation(library, tmp_path):
    entry = library / "models/glm5_3_flash/prefill_fwd.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("from golden import run\nrun('ok', [])\n")
    result = audit.run_case(library, "models/glm5_3_flash/prefill_fwd.py", "a2a3", tmp_path / "out", 30)
    assert result["passed"]
    assert len(result["compilations"]) == 1


@pytest.mark.parametrize(("kernel", "status"), [("ok", 0), ("bad", 1), ("truncated", 1)])
def test_manifest_uses_library_selector(library, tmp_path, monkeypatch, kernel, status):
    selector = library / ".github/scripts/detect_changes.py"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        "import sys\nassert sys.argv[1:] == ['--all', '--platform', 'a5']\n"
        "print('models/case.py models/glm5_3_flash/weights.py '\n"
        "      'models/glm5_3_flash/prefill_ref.py models/glm5_3_flash/prefill_fwd.py')\n"
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
