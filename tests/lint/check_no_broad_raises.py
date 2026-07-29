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

`pytest.raises(Exception)` discards the distinction PyPTO deliberately maintains across the
C++/Python boundary: `CHECK` surfaces a user error (`ValueError`), `INTERNAL_CHECK` surfaces a
compiler bug (`pypto.InternalError`), and the verifier raises `pypto.Error`. A test catching
`Exception` still passes when a `CHECK` is silently downgraded to an `INTERNAL_CHECK` -- exactly
the regression the rule in `.claude/rules/error-checking.md` exists to prevent.

Note that ruff's B017 is not sufficient here: it exempts `pytest.raises(Exception, match=...)`,
which was the overwhelming majority of historical occurrences.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Files that legitimately assert the exception *hierarchy* itself (e.g. "ValueError is catchable
# as Exception"). For these the broad `Exception` is the property under test, not an omission.
ALLOWLIST = frozenset(
    {
        "tests/ut/core/test_error.py",
        "tests/ut/core/test_logging.py",
    }
)

# Subtrees under tests/ that hold checker scripts rather than tests. They are not test code, and
# their source legitimately spells out the forbidden pattern -- this file's own docstring does.
EXCLUDED_PREFIXES = ("tests/lint/",)

# Matches the context-manager and callable forms, a wrapped `pytest.raises(\n Exception`, and a
# leading tuple entry (`pytest.raises((Exception, ...))`) -- a tuple led by Exception is exactly as
# broad as Exception alone. A tuple of concrete types, e.g. `(ValueError, pypto.Error)`, is fine.
BROAD_RAISES = re.compile(r"pytest\.raises\(\s*\(?\s*(Exception|BaseException)\b")


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
        if line.endswith(".py") and not line.startswith(EXCLUDED_PREFIXES) and path.is_file():
            files.append(path)
    return files


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, exception_name, source_line) for each broad raises in `path`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    violations = []
    for match in BROAD_RAISES.finditer(text):
        lineno = text.count("\n", 0, match.start()) + 1
        violations.append((lineno, match.group(1), lines[lineno - 1].strip()))
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
        if rel in ALLOWLIST:
            continue
        for lineno, exc_name, source in find_violations(path):
            print(f"{rel}:{lineno}: pytest.raises({exc_name}) is too broad")
            print(f"    {source}")
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
            "If a test genuinely asserts the exception *hierarchy*, add its path to ALLOWLIST in\n"
            f"{Path(__file__).name}.",
            file=sys.stderr,
        )
        return 1

    print("All test files assert concrete exception types.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
