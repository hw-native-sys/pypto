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
    python tests/lint/clang_tidy.py -v                       # verbose debug output
"""

import json
import os
import re
import shlex
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

_verbose = False


def _vprint(*args: object) -> None:
    """Print only when verbose mode is enabled."""
    if _verbose:
        print(*args)


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


def check_version(version: str | None = None) -> str | None:
    """Return a warning string if the clang-tidy version mismatches, else ``None``."""
    if version is None:
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


def _collect_files(extensions: tuple[str, ...]) -> list[str]:
    """Recursively collect files matching *extensions* from ``DEFAULT_SOURCE_DIRS``."""
    files: list[str] = []
    for directory in DEFAULT_SOURCE_DIRS:
        root = Path(directory)
        if not root.is_dir():
            continue
        for child in root.rglob("*"):
            if child.is_file() and child.suffix.lower() in extensions:
                files.append(str(child))
    return sorted(files)


def collect_source_files() -> list[str]:
    """Recursively collect C/C++ source files from ``DEFAULT_SOURCE_DIRS``."""
    return _collect_files(SOURCE_EXTENSIONS)


def collect_header_files() -> list[str]:
    """Recursively collect C/C++ header files from ``DEFAULT_SOURCE_DIRS``."""
    return _collect_files(HEADER_EXTENSIONS)


def _get_changed_files(diff_base: str) -> set[str] | None:
    """Return the set of files changed relative to *diff_base*, or ``None`` on error."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=d", diff_base, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
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
        return None
    except FileNotFoundError as exc:
        print(
            f"[clang-tidy] Warning: git diff failed ({exc}), linting all files.",
            file=sys.stderr,
        )
        return None

    changed = set(result.stdout.strip().splitlines())
    return changed if changed else set()


def filter_by_diff(
    all_files: list[str],
    diff_base: str,
    changed: set[str] | None = None,
) -> list[str] | None:
    """Filter *all_files* to only those changed relative to *diff_base*.

    Returns ``None`` (meaning "lint everything") when header files changed,
    since header changes can affect any source file.

    If *changed* is provided it is used directly; otherwise ``_get_changed_files``
    is called.
    """
    if changed is None:
        changed = _get_changed_files(diff_base)
    if changed is None:
        return None  # git diff failed — lint everything

    if not changed:
        return []

    # If any header changed, fall back to linting all source files since we
    # can't cheaply determine which sources include the changed headers.
    if any(Path(f).suffix.lower() in HEADER_EXTENSIONS for f in changed):
        print("[clang-tidy] Header files changed — linting all source files.")
        return None

    filtered = [f for f in all_files if f in changed]
    return filtered


def _apply_diff_filter(
    files: list[str],
    headers: list[str],
    diff_base: str,
) -> tuple[list[str], list[str]]:
    """Filter *files* and *headers* to only those changed relative to *diff_base*."""
    _vprint(f"[clang-tidy] Diff base: {diff_base}")
    changed = _get_changed_files(diff_base)
    _vprint(f"[clang-tidy] Changed files: {sorted(changed) if changed else changed}")

    # Filter source files
    filtered = filter_by_diff(files, diff_base, changed=changed)
    if filtered is not None:
        files = filtered
        if files:
            print(f"[clang-tidy] Linting {len(files)} changed source file(s).")

    # Filter header files — only lint changed headers
    if changed is not None:
        headers = [h for h in headers if h in changed]
    if headers:
        print(f"[clang-tidy] Linting {len(headers)} changed header file(s).")

    return files, headers


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


def _extract_header_compile_flags(build_dir: Path) -> list[str]:
    """Extract compile flags from ``compile_commands.json`` for header linting.

    Merges flags from all entries so that every include path, define,
    language standard, and architecture flag is captured (different TUs
    may contribute different flags).  Duplicates are deduplicated.

    For ``-isystem`` paths that don't exist on disk (e.g. generated headers
    in a temp build directory), falls back to the equivalent path under
    ``./build`` if available.
    """
    cc_path = build_dir / "compile_commands.json"
    if not cc_path.exists():
        return ["-xc++", "-std=c++17"]

    with open(cc_path) as f:
        commands = json.load(f)
    _vprint(f"[clang-tidy] compile_commands.json: {len(commands)} entries")
    if not commands:
        return ["-xc++", "-std=c++17"]

    # Merge flags from all compilation units so that every include path,
    # define, and standard flag is captured (different TUs may use different
    # flags).
    seen: set[str | tuple[str, str]] = set()
    flags: list[str] = ["-xc++"]

    for entry in commands:
        if "arguments" in entry:
            args = list(entry["arguments"])
        else:
            args = shlex.split(entry.get("command", ""))

        i = 1  # skip compiler executable
        while i < len(args):
            arg = args[i]
            if arg in ("-I", "-isystem") and i + 1 < len(args):
                path = _resolve_include_path(args[i + 1], build_dir)
                key = (arg, path)
                if key not in seen:
                    seen.add(key)
                    flags.extend([arg, path])
                i += 1
            elif arg.startswith(("-I", "-isystem", "-D", "-std=")):
                if arg not in seen:
                    seen.add(arg)
                    flags.append(arg)
            elif arg == "-arch" and i + 1 < len(args):
                key = (arg, args[i + 1])
                if key not in seen:
                    seen.add(key)
                    flags.extend([arg, args[i + 1]])
                i += 1
            i += 1

    return flags


def _resolve_include_path(path: str, build_dir: Path) -> str:
    """Return *path* if it exists, otherwise try the equivalent under ``./build``.

    Generated headers (e.g. ``backtrace.h``) live under the build directory.
    When a temp build directory is used they may not have been generated
    successfully.  This helper falls back to ``./build/<relative>`` so that
    headers can still be parsed.
    """
    if Path(path).is_dir():
        return path
    try:
        rel = Path(path).relative_to(build_dir)
    except ValueError:
        return path
    alt = Path("build") / rel
    if alt.is_dir():
        return str(alt.resolve())
    return path


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


def run_clang_tidy(
    cmd: list[str],
    files: list[str],
    jobs: int,
    *,
    suffix_args: list[str] | None = None,
    label: str = "Running clang-tidy",
) -> int:
    """Run clang-tidy in parallel, one process per file. Return 0 on success, 1 on failure.

    *suffix_args* (if given) are appended **after** the file — useful for
    passing ``-- <compile-flags>`` when linting header files without a
    compilation database entry.
    """
    print(f"[clang-tidy] {label}...")
    n_workers = min(max(1, jobs), len(files))
    tail = suffix_args or []
    _vprint(f"[clang-tidy] {len(files)} file(s), {n_workers} parallel worker(s)")

    def _run_one(filepath: str) -> tuple[int, str, str]:
        full_cmd = [*cmd, filepath, *tail]
        _vprint(f"[clang-tidy] Running: {' '.join(full_cmd)}")
        proc = subprocess.run(full_cmd, capture_output=True, text=True, check=False)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output.strip(), filepath

    rc = 0
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_one, f) for f in files]
        for fut in as_completed(futures):
            code, output, filepath = fut.result()
            done += 1
            _vprint(f"[clang-tidy] [{done}/{len(files)}] Processing file {filepath}.")
            if code != 0:
                if output:
                    print(output)
                else:
                    print(
                        f"[clang-tidy] {filepath} failed with exit code {code} and no output.",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extra debugging information (compile flags, file lists, commands).",
    )
    return parser.parse_args(list(argv))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the clang-tidy linting workflow.

    Steps:
        1. Verify clang-tidy is installed and check its version.
        2. Collect C/C++ source and header files.
        3. Resolve build directory (explicit ``-B`` > existing ``./build`` > temp dir).
        4. Lint header files first (fixes may affect source files).
        5. Lint source files (all enabled checks via compile database).
        6. Re-print version warning at the end (if any).
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)

    global _verbose  # noqa: PLW0603
    _verbose = args.verbose

    # 1. Check clang-tidy is installed
    if not shutil.which("clang-tidy"):
        print(
            "[clang-tidy] clang-tidy not found on PATH.\n"
            f"Install with: pip install clang-tidy=={REQUIRED_VERSION}",
            file=sys.stderr,
        )
        return 1

    # 2. Version check — print warning at the beginning
    version = get_clang_tidy_version()
    _vprint(f"[clang-tidy] clang-tidy version: {version}")
    version_warning = check_version(version)
    if version_warning:
        print(version_warning, file=sys.stderr)

    # 3. Collect source and header files (optionally filtered by diff)
    files = collect_source_files()
    headers = collect_header_files()
    _vprint(f"[clang-tidy] Collected {len(files)} source file(s), {len(headers)} header file(s).")

    if args.diff_base:
        files, headers = _apply_diff_filter(files, headers, args.diff_base)

    if not files and not headers:
        print("[clang-tidy] No files to lint.")
        return 0

    # 4. Resolve build directory (temp dir if not provided)
    tmp_dir = None
    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
        _vprint(f"[clang-tidy] Using explicit build dir: {build_dir}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="pypto-clang-tidy-")
        build_dir = Path(tmp_dir)
        _vprint(f"[clang-tidy] Using temp build dir: {build_dir}")

    have_errors = 0
    try:
        cc_path = ensure_compile_commands(build_dir)

        # 5a. Lint header files first (fixes here may affect source files)
        if headers:
            compile_flags = _extract_header_compile_flags(cc_path.parent)
            header_cmd = _build_clang_tidy_cmd(cc_path.parent, fix=args.fix)
            # Headers aren't in compile_commands.json, so drop -p and pass
            # compile flags via -- instead.
            header_cmd = [c for c in header_cmd if not c.startswith("-p=")]
            _vprint(f"[clang-tidy] Header compile flags: {compile_flags}")
            _vprint(f"[clang-tidy] Header files: {headers}")
            have_errors |= run_clang_tidy(
                header_cmd,
                headers,
                args.jobs,
                suffix_args=["--", *compile_flags],
                label="Linting header files",
            )

        # 5b. Lint source files (all enabled checks via compile database)
        if files:
            base_cmd = _build_clang_tidy_cmd(cc_path.parent, fix=args.fix)
            _vprint(f"[clang-tidy] Base command: {base_cmd}")
            _vprint(f"[clang-tidy] Source files ({len(files)}): {files}")
            have_errors |= run_clang_tidy(
                base_cmd,
                files,
                args.jobs,
                label="Linting source files",
            )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if have_errors and args.fix:
        print(
            "[clang-tidy] Issues found and fixes applied. Re-stage modified files.",
            file=sys.stderr,
        )

    print("[clang-tidy] All checks completed.")

    # 6. Version check — re-print warning at the end
    if version_warning:
        print(version_warning, file=sys.stderr)

    return int(have_errors)


if __name__ == "__main__":
    sys.exit(main())
