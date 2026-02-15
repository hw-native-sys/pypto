# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run clang-tidy on PyPTO C/C++ source files.

Handles the full workflow: version checking, compile_commands.json generation
via CMake, and parallel clang-tidy execution.

Usage:
    python tests/lint/clang_tidy.py                          # lint all files
    python tests/lint/clang_tidy.py -B my-build              # persistent build dir
    python tests/lint/clang_tidy.py --fix                    # apply fixes in-place
    python tests/lint/clang_tidy.py --diff-base origin/main  # only changed files
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_VERSION = "21.1.0"
SOURCE_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx")
HEADER_EXTENSIONS = (".h", ".hpp", ".hxx")
DEFAULT_SOURCE_DIRS = ("src", "include")


# ---------------------------------------------------------------------------
# Version checking
# ---------------------------------------------------------------------------


def get_clang_tidy_version() -> str | None:
    """Return the installed clang-tidy version string (e.g. ``"21.1.0"``), or ``None``."""
    try:
        result = subprocess.run(
            ["clang-tidy", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"version\s+(\d+\.\d+\.\d+)", result.stdout, re.IGNORECASE)
        if match:
            return match.group(1)
    except FileNotFoundError:
        pass
    return None


def check_version() -> str | None:
    """Return a warning string if the clang-tidy version mismatches, else ``None``."""
    version = get_clang_tidy_version()
    if version is None:
        return None  # Handled by the "not found" check in main()
    if version != REQUIRED_VERSION:
        return (
            f"[clang-tidy] WARNING: Version mismatch — "
            f"found {version}, expected {REQUIRED_VERSION}. "
            f"Install with: pip install clang-tidy=={REQUIRED_VERSION}"
        )
    return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def collect_source_files() -> list[str]:
    """Recursively collect C/C++ source files from ``DEFAULT_SOURCE_DIRS``."""
    files: list[str] = []
    for directory in DEFAULT_SOURCE_DIRS:
        root = Path(directory)
        if not root.is_dir():
            continue
        for child in root.rglob("*"):
            if child.is_file() and child.suffix.lower() in SOURCE_EXTENSIONS:
                files.append(str(child))
    return sorted(files)


def filter_by_diff(all_files: list[str], diff_base: str) -> list[str] | None:
    """Filter *all_files* to only those changed relative to *diff_base*.

    Returns ``None`` (meaning "lint everything") when header files changed,
    since header changes can affect any source file.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=d", diff_base, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            cmd = " ".join(str(part) for part in exc.cmd)
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = [f"command='{cmd}'", f"exit_code={exc.returncode}"]
            if stderr:
                details.append(f"stderr={stderr!r}")
            if stdout:
                details.append(f"stdout={stdout!r}")
            details_msg = ", ".join(details)
            print(
                f"[clang-tidy] Warning: git diff failed ({details_msg}), linting all files.",
                file=sys.stderr,
            )
        else:
            print(
                f"[clang-tidy] Warning: git diff failed ({exc}), linting all files.",
                file=sys.stderr,
            )
        return None

    changed = set(result.stdout.strip().splitlines())
    if not changed:
        return []

    # If any header changed, fall back to linting all source files since we
    # can't cheaply determine which sources include the changed headers.
    if any(Path(f).suffix.lower() in HEADER_EXTENSIONS for f in changed):
        print("[clang-tidy] Header files changed — linting all source files.")
        return None

    filtered = [f for f in all_files if f in changed]
    return filtered


# ---------------------------------------------------------------------------
# CMake / compile_commands.json
# ---------------------------------------------------------------------------


def _detect_nanobind_cmake_dir() -> str | None:
    """Detect the nanobind CMake directory, or ``None`` on failure."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import nanobind; print(nanobind.cmake_dir())"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        print(f"[clang-tidy] Warning: Could not detect nanobind: {exc}", file=sys.stderr)
        return None


def ensure_compile_commands(build_dir: Path) -> Path:
    """Generate ``compile_commands.json`` via CMake if it doesn't already exist.

    Also builds the ``project_libbacktrace`` target to ensure generated
    headers are available for clang-tidy analysis.
    """
    cc_path = build_dir / "compile_commands.json"
    if cc_path.exists():
        return cc_path

    build_dir.mkdir(parents=True, exist_ok=True)

    # CMake configure
    print("[clang-tidy] Configuring CMake...")
    nanobind_dir = _detect_nanobind_cmake_dir()
    cmd = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    if nanobind_dir:
        cmd.append(f"-Dnanobind_DIR={nanobind_dir}")

    ret = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if ret.returncode != 0:
        print("[clang-tidy] CMake configuration failed.", file=sys.stderr)
        if ret.stdout:
            print(ret.stdout, file=sys.stderr)
        if ret.stderr:
            print(ret.stderr, file=sys.stderr)
        sys.exit(ret.returncode)
    if not cc_path.exists():
        print(f"[clang-tidy] {cc_path} not found after CMake.", file=sys.stderr)
        sys.exit(2)

    # Build libbacktrace to generate backtrace.h header
    print("[clang-tidy] Building libbacktrace to generate headers...")
    ret = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "project_libbacktrace", "-j"],
        check=False,
        capture_output=True,
        text=True,
    )
    if ret.returncode != 0:
        print(
            "[clang-tidy] Warning: libbacktrace build failed; some headers may be missing.",
            file=sys.stderr,
        )
        if ret.stdout:
            print(ret.stdout, file=sys.stderr)
        if ret.stderr:
            print(ret.stderr, file=sys.stderr)

    return cc_path


# ---------------------------------------------------------------------------
# clang-tidy execution
# ---------------------------------------------------------------------------


def _build_clang_tidy_cmd(build_dir: Path, fix: bool) -> list[str]:
    """Build the base clang-tidy command list."""
    cmd: list[str] = ["clang-tidy", f"-p={build_dir!s}", "-quiet"]
    if fix:
        cmd.append("-fix")
    if sys.platform == "darwin":
        cmd = ["xcrun", *cmd]
    return cmd


def run_clang_tidy(cmd: list[str], files: list[str], jobs: int) -> int:
    """Run clang-tidy in parallel batches. Return 0 on success, 1 on any failure.

    Files are split into *jobs* batches, each handled by a single clang-tidy
    process.  This drastically reduces per-process startup overhead (LLVM init,
    config parsing, compile_commands.json loading) compared to spawning one
    process per file.
    """
    n_workers = min(max(1, jobs), len(files))
    batches = [files[i::n_workers] for i in range(n_workers)]

    def _run_batch(batch: list[str]) -> tuple[int, str, list[str]]:
        full_cmd = [*cmd, *batch]
        proc = subprocess.run(full_cmd, capture_output=True, text=True, check=False)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output.strip(), batch

    rc = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_batch, b) for b in batches]
        for fut in as_completed(futures):
            code, output, batch = fut.result()
            if code != 0:
                if output:
                    print(output)
                else:
                    print(
                        f"[clang-tidy] Batch of {len(batch)} file(s) failed "
                        f"with exit code {code} and no output.",
                        file=sys.stderr,
                    )
                rc = 1
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        prog="clang-tidy",
        description="Run clang-tidy on PyPTO C/C++ source files.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build-dir",
        "-B",
        default=None,
        help="CMake build directory for compile_commands.json. If omitted, a temporary directory is used.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum parallel clang-tidy processes.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply clang-tidy fixes in-place.",
    )
    parser.add_argument(
        "--diff-base",
        default=None,
        help="Only lint files changed relative to this git ref (e.g. origin/main).",
    )
    return parser.parse_args(list(argv))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the clang-tidy linting workflow.

    Steps:
        1. Verify clang-tidy is installed and check its version.
        2. Collect C/C++ source files from default directories.
        3. Generate ``compile_commands.json`` if needed.
        4. Run clang-tidy in parallel.
        5. Re-print version warning at the end (if any).
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # 1. Check clang-tidy is installed
    if not shutil.which("clang-tidy"):
        print(
            "[clang-tidy] clang-tidy not found on PATH.\n"
            f"Install with: pip install clang-tidy=={REQUIRED_VERSION}",
            file=sys.stderr,
        )
        return 1

    # 2. Version check — print warning at the beginning
    version_warning = check_version()
    if version_warning:
        print(version_warning, file=sys.stderr)

    # 3. Collect source files (optionally filtered by diff)
    files = collect_source_files()
    if args.diff_base:
        filtered = filter_by_diff(files, args.diff_base)
        if filtered is not None:
            files = filtered
            if not files:
                print("[clang-tidy] No changed source files to lint.")
                return 0
            print(f"[clang-tidy] Linting {len(files)} changed file(s).")
    if not files:
        print("[clang-tidy] No source files found to lint.")
        return 0

    # 4. Resolve build directory (temp dir if not provided)
    tmp_dir = None
    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
    else:
        tmp_dir = tempfile.mkdtemp(prefix="pypto-clang-tidy-")
        build_dir = Path(tmp_dir)

    try:
        cc_path = ensure_compile_commands(build_dir)
        base_cmd = _build_clang_tidy_cmd(cc_path.parent, fix=args.fix)
        rc = run_clang_tidy(base_cmd, files, args.jobs)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if rc != 0 and args.fix:
        print(
            "[clang-tidy] Issues found and fixes applied. Re-stage modified files.",
            file=sys.stderr,
        )

    print("[clang-tidy] All checks completed.")

    # 5. Version check — re-print warning at the end
    if version_warning:
        print(version_warning, file=sys.stderr)

    return rc


if __name__ == "__main__":
    sys.exit(main())
