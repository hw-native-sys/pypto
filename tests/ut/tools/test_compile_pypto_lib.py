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
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, "a2a3", tmp_path / "out", 10
    )
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
@pytest.mark.parametrize("interpreter_options", [[], ["-u"]])
def test_python_subprocess_compilations_are_audited(
    library, tmp_path, cases, passed, count, launch, interpreter_options
):
    # Formal entries can isolate cases in fresh interpreters to own runtime lifetimes.
    # DSpark uses -u before the script; those children still need every audit hook.
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
        f"        command = [sys.executable, *{interpreter_options!r}, __file__, '-p', args.p, '-d', '0',\n"
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
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, "a2a3", tmp_path / "out", 10
    )
    assert result["passed"] is passed
    assert len(result["compilations"]) == count


def test_python_subprocess_preserves_other_commands(library, tmp_path):
    (library / "models/helper.py").write_text("raise SystemExit(0)\n")
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "subprocess.run([sys.executable, '-c', 'raise SystemExit(0)'], check=True)\n"
        "subprocess.run([sys.executable, '-u', '-c', 'raise SystemExit(0)'], check=True)\n"
        "subprocess.run([sys.executable, '-u', '-m', 'this'], stdout=subprocess.DEVNULL, check=True)\n"
        "subprocess.run([sys.executable, str(Path(__file__).with_name('helper.py'))], check=True)\n"
        "subprocess.run([b'true'], check=True)\n"
        "run('ok', [])\n"
    )
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


@pytest.mark.parametrize(
    "interpreter_options",
    [
        [],
        ["-u", "-B", "-W", "ignore", "-X", "dev"],
        ["-uB", "-Wignore", "-Xdev"],
        ["-uBW", "ignore", "-X", "dev"],
        ["-uBX", "dev", "-W", "ignore"],
        ["-uB", "-Wignore", "-Xdev", "--check-hash-based-pycs", "always", "--"],
    ],
)
def test_python_subprocess_preserves_popen_options(library, tmp_path, interpreter_options):
    (library / "models/case.py").write_text(
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "from golden import run\n"
        "if '--case' not in sys.argv:\n"
        f"    command = [sys.executable, *{interpreter_options!r}, 'case.py', '--case', 'child', '-d', '0']\n"
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
        f"    if {bool(interpreter_options)!r}:\n"
        "        assert sys.stdout.write_through and sys.flags.dont_write_bytecode\n"
        "        assert sys._xoptions['dev'] and 'ignore' in sys.warnoptions\n"
        "    run('ok', [])\n"
        "    print('child-output')\n"
    )
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


@pytest.mark.parametrize("interpreter_options", [["-x"], ["-ux"], ["-xu"], ["-B", "-x"]])
def test_source_skipping_subprocess_is_not_wrapped(library, tmp_path, interpreter_options):
    # The first line is valid in the parent, but the child must skip it. An
    # audit wrapper would skip its own header and then execute this line.
    (library / "models/case.py").write_text(
        "if '--child' in __import__('sys').argv: raise AssertionError('first line was not skipped')\n"
        "import subprocess, sys\n"
        "if '--child' not in sys.argv:\n"
        f"    command = [sys.executable, *{interpreter_options!r}, __file__, '--child']\n"
        "    child = subprocess.run(command, check=True, capture_output=True, text=True)\n"
        "    assert child.stdout.strip() == 'first-line-skipped'\n"
        "    from golden import run\n"
        "    run('ok', [])\n"
        "else:\n"
        "    print('first-line-skipped')\n"
    )
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


@pytest.mark.parametrize("child_passed", [True, False])
def test_relative_child_receipt_survives_changed_cwd(library, child_passed):
    (library / "models/case.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "if '--child' not in sys.argv:\n"
        "    subprocess.run([sys.executable, '-u', 'case.py', '--child'],\n"
        "                   cwd=Path(__file__).parent, check=True)\n"
        "else:\n"
        "    from golden import run\n"
        f"    run({'ok' if child_passed else 'bad'!r}, [])\n"
    )
    # Unlike run_case's absolute output path, this hidden CLI mode may receive
    # a relative receipt. Both child success and failure must reach the parent.
    child = audit.subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            "--lib",
            str(library),
            "--platform",
            "a5",
            "--entry",
            "models/case.py",
            "--receipt",
            "compilations.json",
        ],
        cwd=library,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr
    results = audit.read_receipt(library / "compilations.json")
    assert len(results) == 1 and results[0]["passed"] is child_passed
    assert (library / "compilations.0.json").is_file()
    assert not (library / "models/compilations.0.json").exists()


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
    assert audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", output, 10)["passed"]
    monkeypatch.setenv("AUDIT_EMPTY_CHILD", "1")
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", output, 10)
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
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
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
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, "a2a3", tmp_path / "out", 20
    )
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
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
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
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
    assert result["passed"]
    assert len(result["compilations"]) == 1


def test_timeout_cannot_reuse_previous_receipt(library, tmp_path):
    script = library / "models/case.py"
    output = tmp_path / "out"
    script.write_text("from golden import run\nrun('ok', [])")
    assert audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", output, 10)["passed"]
    script.write_text("import time\ntime.sleep(60)")
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", output, 1)
    assert not result["passed"]
    assert result["timed_out"]
    assert result["compilations"] == []


@pytest.mark.parametrize("receipt", ["[", "null", "{}", "[{}]", '[{"passed": "true"}]'])
def test_invalid_receipt_fails_without_aborting(library, tmp_path, receipt):
    (library / "models/case.py").write_text(
        "from golden import run\nrun('ok', [])\n"
        f"import json\njson.dumps = lambda *args, **kwargs: {receipt!r}\n"
    )
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, "a2a3", tmp_path / "out", 10
    )
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
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
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
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, "a2a3", tmp_path / "out", 10
    )
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
    result = audit.run_case(library, {"name": "case", "entry": entry}, "a2a3", output, 10)
    assert result["exit_code"] == 0
    assert not result["passed"]
    assert result["compilations"] == []
    assert not list((output / "case").glob("runtime-*"))


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
    result = audit.run_case(
        library,
        {"name": "case", "entry": f"models/deepseek_v4_1_flash/{entry}.py"},
        "a5",
        tmp_path / "out",
        10,
    )
    assert result["passed"]
    assert len(result["compilations"]) == 1


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("constructor", ["argparse.ArgumentParser", "ArgumentParser"])
@pytest.mark.parametrize("verify_kernel", ["ok", "bad"])
def test_component_specific_compile_flag_is_not_injected(
    library, tmp_path, platform, constructor, verify_kernel
):
    # KDA's default entry visits projection before verify. Only verify accepts
    # --compile-only; each golden.run must still compile without device inputs.
    (library / "models/case.py").write_text(
        "import argparse\n"
        "from argparse import ArgumentParser\n"
        "from golden import run\n"
        "def projection():\n"
        f"    parser = {constructor}()\n"
        "    parser.add_argument('-p')\n"
        "    parser.parse_args()\n"
        "    run('ok', [])\n"
        "def verify():\n"
        f"    parser = {constructor}()\n"
        "    parser.add_argument('-p')\n"
        "    parser.add_argument('--compile-only', action='store_true')\n"
        "    args = parser.parse_args()\n"
        "    assert args.compile_only\n"
        f"    run({verify_kernel!r}, [], compile_only=args.compile_only)\n"
        "projection()\n"
        "verify()\n"
    )
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, platform, tmp_path / "out", 10
    )
    assert result["exit_code"] == 0
    assert result["passed"] is (verify_kernel == "ok")
    assert [record["passed"] for record in result["compilations"]] == [True, verify_kernel == "ok"]


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("helper", ["known", "inactive", "strict"])
@pytest.mark.parametrize("kernel", ["ok", "bad"])
def test_multi_parser_compile_flag_prevents_post_compile_execution(
    library, tmp_path, platform, helper, kernel
):
    # A5 decode/prefill first parse shape options, then compile through golden
    # before an independent runtime branch guarded by the main CLI flag.
    (library / "models/case.py").write_text(
        "import argparse\n"
        "from golden import run\n"
        "helper = argparse.ArgumentParser(add_help=False)\n"
        "helper.add_argument('-p')\n"
        + (
            "helper.add_argument('--compile-only', action='store_true')\n"
            "assert helper.parse_args().compile_only\n"
            if helper == "strict"
            else "helper.parse_known_args()\n"
            if helper == "known"
            else ""
        )
        + "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-p')\n"
        "parser.add_argument('--compile-only', dest='compile_mode', action='store_true')\n"
        "args = parser.parse_args()\n"
        f"result = run({kernel!r}, [], compile_only=True)\n"
        "if result.passed and not args.compile_mode:\n"
        "    raise AssertionError('Device execution was attempted after compilation')\n"
    )
    result = audit.run_case(
        library, {"name": "case", "entry": "models/case.py"}, platform, tmp_path / "out", 10
    )
    assert result["exit_code"] == 0
    assert result["passed"] is (kernel == "ok")
    assert [record["passed"] for record in result["compilations"]] == [kernel == "ok"]


def test_inactive_compile_flag_is_not_injected(library, tmp_path):
    (library / "models/case.py").write_text(
        "import argparse\n"
        "from golden import run\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-p')\n"
        "if False:\n"
        "    parser.add_argument('--compile-only', action='store_true')\n"
        "parser.parse_args()\n"
        "run('ok', [])\n"
    )
    result = audit.run_case(library, {"name": "case", "entry": "models/case.py"}, "a5", tmp_path / "out", 10)
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
    ]


@pytest.mark.parametrize(("component", "count"), [("all", 2), ("projection", 1), ("verify", 1)])
def test_components_with_different_compile_only_options(library, tmp_path, component, count):
    # GLM kda.py forwards the same remaining argv to every selected component.
    (library / "models/case.py").write_text(
        "import argparse\n"
        "from golden import run\n"
        "def projection(argv):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-p')\n"
        "    args = parser.parse_args(argv)\n"
        "    assert not hasattr(args, 'compile_only')\n"
        "    run('ok', [])\n"
        "def verify(argv):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-p')\n"
        "    parser.add_argument('--compile-only', dest='only_compile', action='store_true')\n"
        "    args = parser.parse_args(argv)\n"
        "    assert args.only_compile, 'Must not proceed to device execution'\n"
        "    run('ok', [], compile_only=args.only_compile)\n"
        "parser = argparse.ArgumentParser(add_help=False)\n"
        "parser.add_argument('--component', default='all')\n"
        "options, remaining = parser.parse_known_args()\n"
        "if options.component in ('all', 'projection'):\n"
        "    projection(remaining)\n"
        "if options.component in ('all', 'verify'):\n"
        "    verify(remaining)\n"
    )
    case = {"name": "case", "entry": "models/case.py", "args": ["--component", component]}
    assert "--compile-only" not in audit.entry_args(library / case["entry"], "a2a3", case["args"])
    result = audit.run_case(library, case, "a2a3", tmp_path / "out", 10)
    assert result["passed"], (tmp_path / "out" / result["log"]).read_text()
    assert len(result["compilations"]) == count


def test_required_device_argument_without_marker(tmp_path):
    script = tmp_path / "distributed.py"
    script.write_text('parser.add_argument("-d", "--device", required=True)\n')
    assert audit.entry_args(script, "a5") == [str(script), "-p", "a5", "-d", "0"]


@pytest.fixture
def case_config(library, tmp_path):
    (library / "models/case.py").write_text("from golden import run\nrun('ok', [])\n")
    config = tmp_path / "cases.json"
    config.write_text(json.dumps([{"name": "case", "entry": "models/case.py", "platforms": ["a5"]}]))
    return config


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"name": "manifest.json"},
        {"name": "results.json"},
        {"name": ""},
        {"name": None},
        {"entry": ""},
        {"entry": "."},
        {"entry": "../outside.py"},
        {"entry": "/tmp/outside.py"},
        {"entry": "models/./case.py"},
        {"entry": "models/missing.py"},
        {"entry": "models/*"},
        {"entry": "models/case_draft.py"},
        {"platforms": []},
        {"platforms": "a5"},
        {"platforms": ["a6"]},
        {"platforms": ["a5", "a5"]},
        {"platforms": [{}]},
        {"args": "--batch-size 8"},
        {"args": [8]},
        {"args": ["--platform", "a5"]},
        {"args": ["--platform=a5"]},
        {"args": ["--plat=a5"]},
        {"args": ["-p", "a5"]},
        {"args": ["-pa5"]},
        {"args": ["bad\0argument"]},
        {"arg": ["--typo"]},
    ],
)
def test_invalid_case_configuration_fails(library, case_config, change):
    cases = json.loads(case_config.read_text())
    cases[0].update(change)
    case_config.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="case|platform"):
        audit.select_cases(library, "a5", case_config)


@pytest.mark.parametrize("contents", ["{}", "[]", "[null]", "[42]", "["])
def test_invalid_or_empty_case_list_fails(library, case_config, contents):
    case_config.write_text(contents)
    with pytest.raises(ValueError):
        audit.select_cases(library, "a5", case_config)


def test_case_configuration_is_utf8_under_ascii_locale(library, case_config):
    cases = json.loads(case_config.read_text())
    cases[0]["args"] = ["--label", "caf\u00e9"]
    case_config.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    result = audit.subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, locale, runpy, sys; from pathlib import Path; "
            "assert locale.getpreferredencoding(False).lower() in ('ascii', 'ansi_x3.4-1968'); "
            "audit = runpy.run_path(sys.argv[1]); "
            "print(json.dumps(audit['select_cases'](Path(sys.argv[2]), 'a5', Path(sys.argv[3]))))",
            str(_RUNNER),
            str(library),
            str(case_config),
        ],
        env=audit.os.environ | {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == cases


def test_default_cases_keep_glm_kernels_but_exclude_cpu_checks():
    entries = {case["entry"] for case in json.loads(audit.DEFAULT_CASES.read_text())}
    assert {
        "models/glm5_3_flash/attention.py",
        "models/glm5_3_flash/kda.py",
        "models/glm5_3_flash/layer.py",
    } <= entries
    assert not entries & {
        "models/glm5_3_flash/weights.py",
        "models/glm5_3_flash/prefill_ref.py",
        "models/glm5_3_flash/prefill_fwd.py",
    }


def test_duplicate_case_name_fails(library, case_config):
    cases = json.loads(case_config.read_text())
    case_config.write_text(json.dumps(cases * 2))
    with pytest.raises(ValueError, match="Duplicate case name"):
        audit.select_cases(library, "a5", case_config)


def test_platform_without_cases_fails(library, case_config):
    with pytest.raises(ValueError, match="No pypto-lib cases selected for a2a3"):
        audit.select_cases(library, "a2a3", case_config)


def test_explicit_selection_filters_platform_without_discovery(library, case_config):
    cases = json.loads(case_config.read_text())
    cases += [{"name": "other-platform", "entry": "models/case.py", "platforms": ["a2a3"]}]
    case_config.write_text(json.dumps(cases))
    # No library selector is installed, and new entries must not join implicitly.
    (library / "models/new.py").write_text("raise RuntimeError('Must not run')\n")
    assert audit.select_cases(library, "a5", case_config) == [{**cases[0], "args": []}]


def test_entry_symlink_outside_library_fails(library, case_config, tmp_path):
    target = tmp_path / "outside.py"
    target.write_text("")
    path = library / "models/case.py"
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Invalid or missing entry"):
        audit.select_cases(library, "a5", case_config)


@pytest.mark.parametrize("device_args", [["-d", "2,3"], ["--device=2,3"], ["-d2,3"]])
@pytest.mark.parametrize("marker", ["", "# ci: devices=4\n"])
def test_custom_device_arguments_replace_inferred_ranks(tmp_path, device_args, marker):
    path = tmp_path / "case.py"
    path.write_text(marker + 'parser.add_argument("--device", required=True)\n')
    extra = device_args + ["--label", "two words"]
    assert audit.entry_args(path, "a5", extra) == [str(path), "-p", "a5", *extra]


@pytest.mark.parametrize("dispatch", [False, True])
def test_argument_variants_have_separate_receipts_and_logs(library, tmp_path, dispatch):
    (library / "models/case.py").write_text(
        "# ci: devices=4\n"
        "import argparse, subprocess, sys\n"
        "from golden import run\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-p')\n"
        "parser.add_argument('-d', '--device')\n"
        "parser.add_argument('--compile-only', action='store_true')\n"
        "parser.add_argument('--kernel')\n"
        "parser.add_argument('--label')\n"
        "parser.add_argument('--child', action='store_true')\n"
        "args = parser.parse_args()\n"
        "assert args.p == 'a5' and args.device == '2,3' and args.compile_only\n"
        "assert args.label == 'two words; $(not-a-command)'\n"
        "print(args.kernel)\n"
        f"if {dispatch!r} and not args.child:\n"
        "    subprocess.run([sys.executable, __file__, *sys.argv[1:], '--child'], check=True)\n"
        "else:\n"
        "    run(args.kernel, [])\n"
    )
    cases = [
        {
            "name": f"case-{kernel}",
            "entry": "models/case.py",
            "args": ["--kernel", kernel, "--label", "two words; $(not-a-command)", "-d", "2,3"],
        }
        for kernel in ("ok", "bad")
    ]
    with audit.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda case: audit.run_case(library, case, "a5", tmp_path / "out", 10), cases)
        )
    assert [result["passed"] for result in results] == [True, False]
    for result, case in zip(results, cases):
        assert result["name"] == case["name"]
        assert result["args"] == ["-p", "a5", *case["args"]]
        assert result["log"] == f"{case['name']}/compile.log"
        assert (tmp_path / "out" / result["log"]).read_text().strip() == "\n".join(
            [case["args"][1]] * (2 if dispatch else 1)
        )
        assert len(result["compilations"]) == 1


@pytest.mark.parametrize(("kernel", "status"), [("ok", 0), ("bad", 1), ("truncated", 1)])
def test_manifest_uses_configured_cases(library, case_config, tmp_path, monkeypatch, kernel, status):
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
            "--cases",
            str(case_config),
            "--output",
            str(output),
            "--jobs",
            "1",
        ],
    )
    assert audit.main() == status
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["entries"] == ["models/case.py"]
    assert manifest["cases"] == [
        {"name": "case", "entry": "models/case.py", "platforms": ["a5"], "args": [], "argv": ["-p", "a5"]}
    ]
    assert len(manifest["lib_revision"]) == 40
    assert json.loads((output / "results.json").read_text())[0]["passed"] is (status == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
