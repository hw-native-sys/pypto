# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared file selection for the per-file lint checkers.

Each of these checkers answers a question about one file in isolation -- does it carry the
copyright header, does it hold a non-English run, does it assert a broad exception -- so it can
run in either of two modes:

* **file-scoped**, the pre-commit path: the hook is given the files in the commit. pre-commit
  partitions that list across ``cpu_count()`` worker subprocesses, so the scan parallelises for
  free, and a commit never pays to re-parse the ~1600 files it did not touch.
* **whole-tree**, the no-argument path: a manual run, or a CI ``--all-files`` sweep, checks
  every tracked file under the checker's scan roots.

The two modes must agree: a file the whole-tree run would flag must also be flagged when it
arrives as an argument, and a file the whole-tree run would never look at (wrong suffix, outside
the scan roots) must be dropped in either mode rather than checked on different terms. Routing
both through :func:`resolve` is what keeps that true -- the root and suffix filters are written
once and applied to both.

``pre-commit`` passes repo-relative paths; a hand-written invocation may pass absolute ones or a
path outside the repo. All three are normalised here.
"""

import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["resolve", "select", "tracked_files"]


def tracked_files(root: Path, scan_roots: Sequence[str] = ()) -> list[Path]:
    """Every git-tracked file under *scan_roots* (or the whole repo when empty).

    Args:
        root: Repository root; ``git ls-files`` runs here.
        scan_roots: Repo-relative pathspecs to limit the listing to.

    Returns:
        Absolute paths, sorted, of the tracked entries that exist as files.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *scan_roots],
            cwd=root,
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

    paths = (root / line for line in result.stdout.splitlines() if line)
    return sorted(path for path in paths if path.is_file())


def _within(path: Path, root: Path, scan_roots: Sequence[str]) -> bool:
    """Whether *path* lies inside the repo and under one of *scan_roots*."""
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return False  # outside the repository entirely
    if not scan_roots:
        return True
    return any(relative == r or relative.startswith(f"{r.rstrip('/')}/") for r in scan_roots)


def select(
    root: Path,
    selected: Iterable[Path],
    scan_roots: Sequence[str] = (),
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """The subset of *selected* that a whole-tree run over the same scope would have visited.

    Anything outside the repo or the scan roots, carrying an unwanted suffix, or no longer on
    disk (a file deleted in the same commit) is dropped rather than checked on different terms.

    Args:
        root: Repository root. Relative arguments are resolved against it, matching how
            pre-commit passes them, rather than against the process working directory.
        selected: Paths passed on the command line.
        scan_roots: Repo-relative directories the checker is scoped to; empty means the whole repo.
        suffixes: File suffixes to keep, e.g. ``{".py"}``; ``None`` keeps every suffix.

    Returns:
        Absolute paths, sorted and de-duplicated.
    """
    keep = set(suffixes) if suffixes is not None else None
    return sorted(
        {
            path
            for raw in selected
            for path in [raw if raw.is_absolute() else root / raw]
            if path.is_file() and (keep is None or path.suffix in keep) and _within(path, root, scan_roots)
        }
    )


def resolve(
    root: Path,
    selected: Iterable[Path] | None,
    scan_roots: Sequence[str] = (),
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """The files to check: the caller's explicit selection, else the whole tree.

    The same root and suffix filters apply to both modes, so a file-scoped run visits exactly
    the subset of files a whole-tree run would have visited.

    Args:
        root: Repository root.
        selected: Paths passed on the command line by pre-commit, or empty/None for whole-tree.
        scan_roots: Repo-relative directories the checker is scoped to; empty means the whole repo.
        suffixes: File suffixes to keep, e.g. ``{".py"}``; ``None`` keeps every suffix.

    Returns:
        Absolute paths, sorted and de-duplicated.
    """
    selected = list(selected or ())
    if selected:
        return select(root, selected, scan_roots, suffixes)
    keep = set(suffixes) if suffixes is not None else None
    return [path for path in tracked_files(root, scan_roots) if keep is None or path.suffix in keep]
