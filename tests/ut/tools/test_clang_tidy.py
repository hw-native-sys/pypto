# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for clang-tidy file selection and compiler configuration."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_clang_tidy() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "clang_tidy.py"
    spec = importlib.util.spec_from_file_location("pypto_clang_tidy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clang_tidy = _load_clang_tidy()


@pytest.mark.parametrize(
    "path",
    [
        "tests/lint/clang_tidy.py",
        ".clang-tidy",
        ".github/workflows/ci.yml",
        ".github/workflows/daily_ci.yml",
        "CMakeLists.txt",
        "python/bindings/CMakeLists.txt",
        "cmake/libbacktrace.cmake",
        "3rdparty/libbacktrace",
        "3rdparty/msgpack-c",
        "runtime",
    ],
)
def test_lint_infrastructure_change_requires_full_check(path: str):
    assert clang_tidy._get_full_check_triggers({path}) == [path]


def test_source_change_does_not_require_full_check():
    assert clang_tidy._get_full_check_triggers({"src/ir/arith/canonical_simplify.cpp"}) == []


def test_source_lint_receives_implicit_system_includes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preconfigured build omits the compiler's implicit standard-library paths."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    database = build_dir / "compile_commands.json"
    database.write_text("[]")
    cache = build_dir / "CMakeCache.txt"
    cache.write_text("existing compiler configuration\n")
    system_include = tmp_path / "compiler" / "include"
    system_include.mkdir(parents=True)
    commands: list[list[str]] = []

    monkeypatch.setattr(clang_tidy.shutil, "which", lambda _name: "clang-tidy")
    monkeypatch.setattr(clang_tidy, "get_clang_tidy_version", lambda: clang_tidy.REQUIRED_VERSION)
    monkeypatch.setattr(clang_tidy, "collect_source_files", lambda: ["src/ir/arith/canonical_simplify.cpp"])
    monkeypatch.setattr(clang_tidy, "collect_header_files", list)
    monkeypatch.setattr(clang_tidy, "_get_system_include_paths", lambda: [str(system_include)])

    def capture_source_lint(cmd: list[str], _files: list[str], _jobs: int, **_kwargs: object) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(clang_tidy, "run_clang_tidy", capture_source_lint)
    assert clang_tidy.main(["--strict-version", "--jobs", "1", "-B", str(build_dir)]) == 0
    assert len(commands) == 1
    extra_args = [arg.removeprefix("--extra-arg=") for arg in commands[0] if arg.startswith("--extra-arg=")]
    assert extra_args == ["-isystem", str(system_include)]
    # Discovering includes must not reconfigure the developer's working build.
    assert database.read_text() == "[]"
    assert cache.read_text() == "existing compiler configuration\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
