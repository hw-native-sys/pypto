# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that tests assert concrete exception types.

Catching a bare ``Exception`` discards the distinction PyPTO deliberately maintains across the
C++/Python boundary: ``CHECK`` surfaces a user error (``ValueError``), ``INTERNAL_CHECK`` surfaces a
compiler bug (``pypto.InternalError``), and the verifier raises ``pypto.Error``. A test catching the
base class still passes when a ``CHECK`` is silently downgraded to an ``INTERNAL_CHECK`` -- exactly
the regression the rule in ``.claude/rules/error-checking.md`` exists to prevent.

ruff's B017 is not a substitute: it exempts the ``match=`` form, which is the overwhelming majority
of real occurrences, and neither ``B`` nor ``PT`` is in this repo's ruff ``select`` list.

The check parses each file rather than scanning text, so prose in comments and docstrings is never
flagged, and a broad type is caught anywhere in a tuple -- not just as its first element.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

# Tests that legitimately assert the exception *hierarchy* itself (e.g. "ValueError is catchable as
# Exception"). Scoped to the exact class or function under test -- not the whole file -- so an
# unrelated broad assertion added to one of these files later is still reported. Each entry is
# "<repo-relative path>::<dotted qualname prefix>".
ALLOWLIST = frozenset(
    {
        "tests/ut/core/test_error.py::TestErrorInheritance",
        "tests/ut/core/test_logging.py::TestCheckFunctions.test_check_preserves_exception_hierarchy",
        "tests/ut/core/test_logging.py::TestCheckFunctions.test_internal_check_preserves_exception_hierarchy",
    }
)

BROAD_NAMES = frozenset({"Exception", "BaseException"})


def get_git_tracked_test_files(root_dir: Path) -> list[Path]:
    """Get list of git-tracked Python files under tests/."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", "tests"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)

    files = []
    for line in result.stdout.splitlines():
        path = root_dir / line
        if line.endswith(".py") and path.is_file():
            files.append(path)
    return files


def _is_raises_call(node: ast.Call, aliases: set[str]) -> bool:
    """Whether *node* calls ``pytest.raises`` (or a name imported from pytest as ``raises``)."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "raises":
        return isinstance(func.value, ast.Name) and func.value.id == "pytest"
    return isinstance(func, ast.Name) and func.id in aliases


def _broad_names(expected: ast.expr) -> list[str]:
    """Return the broad exception names used in a ``pytest.raises`` first argument."""
    elements = expected.elts if isinstance(expected, ast.Tuple) else [expected]
    return [e.id for e in elements if isinstance(e, ast.Name) and e.id in BROAD_NAMES]


def _raises_aliases(tree: ast.Module) -> set[str]:
    """Names bound by ``from pytest import raises [as X]`` at module level."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "pytest"
        for alias in node.names
        if alias.name == "raises"
    }


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, exception_name, enclosing qualname) for each broad raises in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        print(f"Error: Failed to parse {path}: {e}", file=sys.stderr)
        sys.exit(1)

    aliases = _raises_aliases(tree)
    violations: list[tuple[int, str, str]] = []

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, (*scope, child.name))
                continue
            if isinstance(child, ast.Call) and _is_raises_call(child, aliases) and child.args:
                for name in _broad_names(child.args[0]):
                    violations.append((child.lineno, name, ".".join(scope)))
            walk(child, scope)

    walk(tree, ())
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that tests assert concrete exception types.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the repo containing this script)",
    )
    args = parser.parse_args()
    root_dir = args.root.resolve()

    total = 0
    for path in get_git_tracked_test_files(root_dir):
        rel = path.relative_to(root_dir).as_posix()
        for lineno, exc_name, qualname in find_violations(path):
            site = f"{rel}::{qualname}"
            if any(site == a or site.startswith(f"{a}.") for a in ALLOWLIST):
                continue
            print(f"{rel}:{lineno}: pytest.raises({exc_name}) is too broad, in {qualname or '<module>'}")
            total += 1

    if total:
        print(
            f"\nFound {total} over-broad exception assertion(s).\n"
            "Assert the concrete type instead, keeping any existing `match=`:\n"
            "    ValueError              -- a C++ CHECK (user error)\n"
            "    pypto.InternalError     -- a C++ INTERNAL_CHECK (compiler bug)\n"
            "    pypto.Error             -- VerificationError and other pypto::Error subclasses\n"
            "    ParserSyntaxError, ParserTypeError, InvalidOperationError\n"
            "                            -- from pypto.language.parser.diagnostics\n"
            "A tuple of concrete types, e.g. (ValueError, pypto.Error), is fine when a call can\n"
            "legitimately raise either. If a test genuinely asserts the exception *hierarchy*, add\n"
            f"its `<path>::<qualname>` to ALLOWLIST in {Path(__file__).name}.",
            file=sys.stderr,
        )
        return 1

    print("All test files assert concrete exception types.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
