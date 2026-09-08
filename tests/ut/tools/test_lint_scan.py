# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the shared file selection behind the per-file lint checkers.

``tests/lint/_scan.py`` is what lets those checkers run in two modes -- given the commit's files
by pre-commit, or given nothing and sweeping the tree. The property that makes the split safe is
that the two modes agree: a file the whole-tree run would visit must also be visited when it
arrives as an argument, and a file the whole-tree run would skip must be skipped either way.
These tests pin that agreement, plus the argument normalisation pre-commit relies on.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _load_scan() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "_scan.py"
    spec = importlib.util.spec_from_file_location("pypto_lint_scan", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = _load_scan()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny git repo with tracked files in and out of a `tests/` scan root."""
    for rel in ("tests/a.py", "tests/sub/b.py", "tests/c.txt", "python/d.py", "src/e.cpp"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _rel(root: Path, paths: list[Path]) -> list[str]:
    return [p.relative_to(root).as_posix() for p in paths]


def test_whole_tree_applies_root_and_suffix_filters(repo: Path) -> None:
    assert _rel(repo, scan.resolve(repo, None, ("tests",), {".py"})) == ["tests/a.py", "tests/sub/b.py"]


def test_whole_tree_without_filters_returns_every_tracked_file(repo: Path) -> None:
    assert _rel(repo, scan.resolve(repo, None)) == [
        "python/d.py",
        "src/e.cpp",
        "tests/a.py",
        "tests/c.txt",
        "tests/sub/b.py",
    ]


def test_untracked_file_is_invisible_to_the_whole_tree_mode(repo: Path) -> None:
    (repo / "tests" / "untracked.py").write_text("x\n")
    assert "tests/untracked.py" not in _rel(repo, scan.resolve(repo, None, ("tests",), {".py"}))


@pytest.mark.parametrize(
    "selected, expected",
    [
        # The plain case: a file inside the scan root with a matching suffix.
        (["tests/a.py"], ["tests/a.py"]),
        (["tests/a.py", "tests/sub/b.py"], ["tests/a.py", "tests/sub/b.py"]),
        # Wrong suffix, right root.
        (["tests/c.txt"], []),
        # Right suffix, outside the scan root -- the case that keeps a file-scoped run from
        # reporting on files the whole-tree run would never have opened.
        (["python/d.py"], []),
        # A path that does not exist (e.g. a file deleted in the same commit).
        (["tests/gone.py"], []),
        # Duplicates collapse and the result is sorted.
        (["tests/sub/b.py", "tests/a.py", "tests/a.py"], ["tests/a.py", "tests/sub/b.py"]),
    ],
)
def test_file_scoped_selection_filters(repo: Path, selected: list[str], expected: list[str]) -> None:
    resolved = scan.resolve(repo, [Path(s) for s in selected], ("tests",), {".py"})
    assert _rel(repo, resolved) == expected


def test_relative_arguments_are_repo_relative_not_cwd_relative(repo: Path) -> None:
    """pre-commit passes repo-relative paths regardless of the process working directory."""
    resolved = scan.resolve(repo, [Path("tests/a.py")], ("tests",), {".py"})
    assert _rel(repo, resolved) == ["tests/a.py"]


def test_absolute_arguments_are_accepted(repo: Path) -> None:
    resolved = scan.resolve(repo, [repo / "tests" / "a.py"], ("tests",), {".py"})
    assert _rel(repo, resolved) == ["tests/a.py"]


def test_path_outside_the_repository_is_dropped(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x\n")
    assert scan.resolve(repo, [outside], ("tests",), {".py"}) == []


def test_empty_selection_falls_back_to_the_whole_tree(repo: Path) -> None:
    """An empty list means "no files given", not "no files to check"."""
    assert scan.resolve(repo, [], ("tests",), {".py"}) == scan.resolve(repo, None, ("tests",), {".py"})


def test_the_two_modes_agree_file_by_file(repo: Path) -> None:
    """The invariant the file-scoped hooks rest on.

    Every tracked file either appears in both the whole-tree sweep and its own single-file
    selection, or in neither. A file scoped out by root or suffix must be scoped out both ways.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.split()
    whole_tree = set(_rel(repo, scan.resolve(repo, None, ("tests",), {".py"})))
    for rel in tracked:
        scoped = _rel(repo, scan.resolve(repo, [Path(rel)], ("tests",), {".py"}))
        assert scoped == ([rel] if rel in whole_tree else []), rel


def test_tracked_files_ignores_directories(repo: Path) -> None:
    """`git ls-files` never lists a directory, but the is_file() guard must survive a stale index."""
    assert all(p.is_file() for p in scan.tracked_files(repo))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
