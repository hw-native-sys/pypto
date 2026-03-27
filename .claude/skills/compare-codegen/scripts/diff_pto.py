# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare .pto files and pass dump IR between two codegen output directories.

Usage:
    python diff_pto.py <dir_a> <dir_b> [--labels A B] [--include-passes] [--context N]

Examples:
    python diff_pto.py build_output/compare/main build_output/compare/feature-x
    python diff_pto.py dir_a dir_b --labels main my-branch --include-passes
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

DIFF_EXTENSIONS = {".pto"}
PASS_EXTENSIONS = {".mlir", ".pto", ".txt"}

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"


def _collect_files(root: Path, extensions: set[str]) -> dict[str, Path]:
    """Return {relative_path: absolute_path} for files matching extensions."""
    result: dict[str, Path] = {}
    if not root.exists():
        return result
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in extensions:
            result[str(p.relative_to(root))] = p
    return result


def _unified_diff(
    path_a: Path | None,
    path_b: Path | None,
    label_a: str,
    label_b: str,
    rel_path: str,
    context: int = 3,
) -> list[str]:
    """Produce a unified diff between two files."""
    lines_a = path_a.read_text(errors="replace").splitlines(keepends=True) if path_a else []
    lines_b = path_b.read_text(errors="replace").splitlines(keepends=True) if path_b else []
    return list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"{label_a}/{rel_path}",
            tofile=f"{label_b}/{rel_path}",
            n=context,
        )
    )


def _print_colored_diff(diff_lines: list[str]) -> None:
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            sys.stdout.write(f"{BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            sys.stdout.write(f"{CYAN}{line}{RESET}")
        elif line.startswith("+"):
            sys.stdout.write(f"{GREEN}{line}{RESET}")
        elif line.startswith("-"):
            sys.stdout.write(f"{RED}{line}{RESET}")
        else:
            sys.stdout.write(line)
    if diff_lines and not diff_lines[-1].endswith("\n"):
        sys.stdout.write("\n")


def _compare_file_sets(
    files_a: dict[str, Path],
    files_b: dict[str, Path],
    label_a: str,
    label_b: str,
    context: int,
    category: str,
) -> tuple[int, int, int, int]:
    """Compare two sets of files. Returns (only_a, only_b, changed, unchanged)."""
    all_keys = sorted(set(files_a) | set(files_b))
    only_a: list[str] = []
    only_b: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []

    for key in all_keys:
        in_a = key in files_a
        in_b = key in files_b
        if in_a and not in_b:
            only_a.append(key)
        elif in_b and not in_a:
            only_b.append(key)
        else:
            content_a = files_a[key].read_text(errors="replace")
            content_b = files_b[key].read_text(errors="replace")
            if content_a == content_b:
                unchanged.append(key)
            else:
                changed.append(key)

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}{category} Summary{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"  Only in {label_a}: {RED}{len(only_a)}{RESET}")
    print(f"  Only in {label_b}: {GREEN}{len(only_b)}{RESET}")
    print(f"  Changed:         {YELLOW}{len(changed)}{RESET}")
    print(f"  Unchanged:       {len(unchanged)}")

    if only_a:
        print(f"\n{RED}Files only in {label_a}:{RESET}")
        for f in only_a:
            print(f"  - {f}")

    if only_b:
        print(f"\n{GREEN}Files only in {label_b}:{RESET}")
        for f in only_b:
            print(f"  + {f}")

    if changed:
        print(f"\n{YELLOW}Changed files:{RESET}")
        for f in changed:
            print(f"  ~ {f}")
        print()
        for f in changed:
            diff = _unified_diff(files_a[f], files_b[f], label_a, label_b, f, context)
            if diff:
                print(f"{BOLD}--- Diff: {f} ---{RESET}")
                _print_colored_diff(diff)
                print()

    return len(only_a), len(only_b), len(changed), len(unchanged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare .pto files and pass dumps between two codegen output directories."
    )
    parser.add_argument("dir_a", type=Path, help="First output directory (e.g. main)")
    parser.add_argument("dir_b", type=Path, help="Second output directory (e.g. branch)")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="Labels for the two directories (default: directory names)",
    )
    parser.add_argument(
        "--include-passes",
        action="store_true",
        help="Also diff passes_dump/ IR files",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of context lines in unified diff (default: 3)",
    )
    args = parser.parse_args()

    dir_a: Path = args.dir_a.resolve()
    dir_b: Path = args.dir_b.resolve()
    label_a = args.labels[0] if args.labels else dir_a.name
    label_b = args.labels[1] if args.labels else dir_b.name

    if not dir_a.exists():
        print(f"Error: {dir_a} does not exist", file=sys.stderr)
        return 1
    if not dir_b.exists():
        print(f"Error: {dir_b} does not exist", file=sys.stderr)
        return 1

    print(f"{BOLD}Comparing codegen output{RESET}")
    print(f"  {label_a}: {dir_a}")
    print(f"  {label_b}: {dir_b}")

    pto_a = _collect_files(dir_a, DIFF_EXTENSIONS)
    pto_b = _collect_files(dir_b, DIFF_EXTENSIONS)
    oa, ob, ch, unch = _compare_file_sets(pto_a, pto_b, label_a, label_b, args.context, ".pto Files")

    has_pass_diff = False
    if args.include_passes:
        pass_dirs_a = [d for d in dir_a.rglob("passes_dump") if d.is_dir()]
        pass_dirs_b = [d for d in dir_b.rglob("passes_dump") if d.is_dir()]

        if pass_dirs_a or pass_dirs_b:
            pass_a: dict[str, Path] = {}
            pass_b: dict[str, Path] = {}
            for pd in pass_dirs_a:
                for k, v in _collect_files(pd, PASS_EXTENSIONS).items():
                    rel = str(pd.relative_to(dir_a) / k)
                    pass_a[rel] = v
            for pd in pass_dirs_b:
                for k, v in _collect_files(pd, PASS_EXTENSIONS).items():
                    rel = str(pd.relative_to(dir_b) / k)
                    pass_b[rel] = v

            pa, pb, pch, _ = _compare_file_sets(
                pass_a, pass_b, label_a, label_b, args.context, "Pass Dump Files"
            )
            has_pass_diff = pa > 0 or pb > 0 or pch > 0

    total_diff = oa + ob + ch
    if total_diff == 0 and not has_pass_diff:
        print(f"\n{GREEN}{BOLD}No differences found.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}{BOLD}Differences detected.{RESET}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
