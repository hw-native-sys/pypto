# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the environment-input inventory, including aliases."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def lint(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "lint"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "check_environment_inputs", directory / "check_environment_inputs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source",
    [
        'import os\nos.environ.get("PYPTO_NEW")',
        'import os as operating\noperating.getenv("PYPTO_NEW")',
        'from os import getenv as read\nread("PYPTO_NEW")',
        'from os import environ as env\nenv["PYPTO_NEW"]',
        'import os\n_ENV: str = "PYPTO_NEW"\nos.getenv(_ENV)',
        'import os\nread = os.environ.get\nread("PYPTO_NEW")',
        'import os\ndef work():\n    env = os.environ\n    return env.get("PYPTO_NEW")',
        'import os\ndef work():\n    env: object = os.environ\n    return env.get("PYPTO_NEW")',
        'import os\nos.environ.setdefault("PYPTO_NEW", "default")',
        'import os\nos.getenvb(b"PYPTO_NEW")',
        'import os\nos.environb[b"PYPTO_NEW"]',
    ],
)
def test_python_recognizes_literal_constants_and_aliases(lint, source):
    assert [read.variable for read in lint.python_reads(source)] == ["PYPTO_NEW"]


@pytest.mark.parametrize(
    "expression",
    ["os.getenv(name)", "os.environ[name]", "dict(os.environ)", "os.environ.copy()", "list(os.environ)"],
)
def test_dynamic_and_bulk_python_reads_need_explicit_audit(lint, expression):
    reads = lint.python_reads(f"import os\ndef work(name):\n    return {expression}\n")
    assert len(reads) == 1
    assert reads[0].variable is None
    assert reads[0].function == "work"


def test_python_writes_and_prose_are_not_environment_reads(lint):
    source = """import os
# os.getenv("PYPTO_COMMENT")
message = 'os.getenv("PYPTO_STRING")'
os.environ["PYPTO_DERIVED"] = "value"
"""
    assert lint.python_reads(source) == []


@pytest.mark.parametrize(
    "source",
    [
        'import os\n_ENV="PYPTO_KNOWN"\ndef work(_ENV):\n    return os.getenv(_ENV)',
        'import os\n_ENV="PYPTO_KNOWN"\ndef work(name):\n    _ENV=name\n    return os.getenv(_ENV)',
        'import os\n_ENV="PYPTO_KNOWN"\n_ENV=compute_name()\nos.getenv(_ENV)',
        'import os\n_ENV="PYPTO_KNOWN"\ndef change(name):\n    global _ENV\n    _ENV=name\nos.getenv(_ENV)',
        'import os\n_ENV="PYPTO_KNOWN"\ndef outer(_ENV):\n    def inner():\n        return os.getenv(_ENV)',
    ],
)
def test_shadowing_and_reassignment_cannot_hide_dynamic_inputs(lint, source):
    assert [read.variable for read in lint.python_reads(source)] == [None]


@pytest.mark.parametrize(
    "assignment",
    [
        'if enabled:\n    _ENV = "PYPTO_NEW"',
        "for _ENV in names:\n    pass",
        'try:\n    _ENV = "PYPTO_NEW"\nexcept RuntimeError:\n    pass',
        'with context():\n    _ENV = "PYPTO_NEW"',
        '_ENV += "_NEW"',
    ],
)
def test_module_control_flow_cannot_hide_unclassified_inputs(lint, tmp_path, assignment):
    directory = tmp_path / "python/pypto"
    directory.mkdir(parents=True)
    (directory / "compiler.py").write_text(
        f'import os\n_ENV = "PYPTO_KNOWN"\n{assignment}\nos.getenv(_ENV)\n'
    )
    errors = lint.check_tree(tmp_path, _registry())
    assert len(errors) == 1 and "dynamic read" in errors[0]


def test_later_assignment_cannot_relabel_an_earlier_read(lint):
    source = 'import os\n_ENV = "PYPTO_NEW"\nos.getenv(_ENV)\n_ENV = "PYPTO_KNOWN"'
    assert [read.variable for read in lint.python_reads(source)] == [None]


def test_import_aliases_do_not_leak_between_functions(lint):
    source = """def compiler():
    import os as platform
    return platform.getenv("PYPTO_NEW")
def unrelated():
    import sys as platform
    return platform.version
"""
    assert [(read.variable, read.function) for read in lint.python_reads(source)] == [
        ("PYPTO_NEW", "compiler")
    ]


def test_aliases_follow_closures_but_respect_parameter_shadowing(lint):
    source = """import os as platform
def outer():
    read = platform.getenv
    def inner():
        return read("PYPTO_NEW")
def unrelated(platform):
    return platform.getenv("NOT_AN_ENVIRONMENT_READ")
"""
    assert [(read.variable, read.function) for read in lint.python_reads(source)] == [("PYPTO_NEW", "inner")]


def test_conditional_import_aliases_preserve_all_environment_reads(lint):
    source = """if enabled:
    import os as platform
else:
    import sys as platform
platform.getenv("PYPTO_NEW")
"""
    assert [read.variable for read in lint.python_reads(source)] == ["PYPTO_NEW"]


def test_cpp_comments_strings_and_concatenated_literals(lint):
    source = """// getenv("COMMENT")
/* getenv("BLOCK_COMMENT") */
const char *message = "getenv(\\\"STRING\\\")";
auto value = std::getenv("PYPTO_" "NEW");
auto other = secure_getenv(variable);
"""
    reads = lint.cpp_reads(source)
    assert [read.variable for read in reads] == ["PYPTO_NEW", None]
    assert [read.line for read in reads] == [4, 5]


def _registry():
    return {
        "schema": 1,
        "variables": {"PYPTO_KNOWN": {"category": "semantic", "reason": "Changes generated operations."}},
        "dynamic_reads": [],
    }


def test_tree_checks_non_pypto_names_and_bindings(lint, tmp_path):
    python_dir = tmp_path / "python/pypto"
    binding_dir = tmp_path / "python/bindings"
    python_dir.mkdir(parents=True)
    binding_dir.mkdir()
    (python_dir / "compiler.py").write_text('import os\nos.getenv("COMPILER_PATH")\n')
    (binding_dir / "entry.cpp").write_text('auto x = getenv("BINDING_OPTION");\n')
    errors = lint.check_tree(tmp_path, _registry())
    assert len(errors) == 2
    assert any("COMPILER_PATH" in error for error in errors)
    assert any("BINDING_OPTION" in error for error in errors)


def test_dynamic_exception_is_scoped_and_cannot_hide_new_literal_reads(lint, tmp_path):
    directory = tmp_path / "python/pypto"
    directory.mkdir(parents=True)
    (directory / "compiler.py").write_text(
        'import os\ndef restore(name):\n    os.getenv(name)\n    os.getenv("PYPTO_NEW")\n'
        "def compile(name):\n    os.getenv(name)\n"
    )
    registry = _registry()
    registry["dynamic_reads"] = [
        {"path": "python/pypto/compiler.py", "function": "restore", "reason": "Restores runtime overrides."}
    ]
    errors = lint.check_tree(tmp_path, registry)
    assert len(errors) == 2
    assert any("PYPTO_NEW" in error for error in errors)
    assert any("dynamic read in compile" in error for error in errors)


def test_stale_dynamic_exceptions_fail(lint, tmp_path):
    registry = _registry()
    registry["dynamic_reads"] = [
        {"path": "python/pypto/removed.py", "function": "restore", "reason": "Old consumer."}
    ]
    assert lint.check_tree(tmp_path, registry) == [
        "Unused dynamic-read exception: python/pypto/removed.py:restore"
    ]


def _write(tmp_path, files):
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


# A source that uses the exception, one that does not, and one under a root the audit
# does not cover. Each registry below is checked both ways; see the test's docstring.
_SOURCES = {
    "python/pypto/compiler.py": "import os\ndef restore(name):\n    os.getenv(name)\n",
    "python/pypto/other.py": 'import os\nos.getenv("PYPTO_KNOWN")\n',
    "tests/lint/helper.py": "import os\ndef restore(name):\n    os.getenv(name)\n",
}


@pytest.mark.parametrize(
    "path, function",
    [
        # Used: the named file really does hold that dynamic read.
        ("python/pypto/compiler.py", "restore"),
        # Unused: right file, wrong function.
        ("python/pypto/compiler.py", "absent"),
        # Unused: the file does not exist.
        ("python/pypto/removed.py", "restore"),
        # Unused: the read exists but sits outside the audited roots, so a whole-tree
        # sweep never sees it either.
        ("tests/lint/helper.py", "restore"),
    ],
)
def test_unused_exceptions_agrees_with_the_whole_tree_sweep(lint, tmp_path, path, function):
    """The cheap path must reach the verdict the sweep it replaces would reach.

    ``unused_exceptions`` reads only the files the exceptions name, on the grounds that
    ``check_registry`` forbids a wildcard or absolute path so no other file can mark one
    used. That reasoning is what this pins: for each shape of exception, its answer must
    equal the unused-exception lines of a full ``check_tree`` sweep.
    """
    root = _write(tmp_path, _SOURCES)
    registry = _registry()
    registry["dynamic_reads"] = [{"path": path, "function": function, "reason": "Test."}]

    swept = [e for e in lint.check_tree(root, registry) if e.startswith("Unused dynamic-read")]
    assert lint.unused_exceptions(root, registry) == swept


def test_unused_exceptions_does_not_read_files_no_exception_names(lint, tmp_path):
    """The performance property, asserted rather than assumed.

    An unparseable source makes the whole-tree sweep raise, because it reads every file.
    ``unused_exceptions`` must not, which is only true if it never opens that file.
    """
    root = _write(tmp_path, {**_SOURCES, "python/pypto/broken.py": "def (((\n"})
    registry = _registry()
    registry["dynamic_reads"] = [
        {"path": "python/pypto/compiler.py", "function": "restore", "reason": "Test."}
    ]

    with pytest.raises(SyntaxError):
        lint.check_tree(root, registry)
    assert lint.unused_exceptions(root, registry) == []


def test_unused_exceptions_reports_nothing_when_the_registry_has_none(lint, tmp_path):
    root = _write(tmp_path, _SOURCES)
    assert lint.unused_exceptions(root, _registry()) == []


@pytest.mark.parametrize("category, reason", [("unknown", "Some input"), ("semantic", "")])
def test_registry_requires_a_supported_category_and_reason(lint, category, reason):
    registry = _registry()
    registry["variables"]["PYPTO_KNOWN"] = {"category": category, "reason": reason}
    with pytest.raises(ValueError, match="category and reason"):
        lint.check_registry(registry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
