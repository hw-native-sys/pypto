# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared discovery and version validation of the ``ptoas`` executable."""

import os
import re
import shutil
import subprocess
import threading

# Oldest PTOAS release that accepts the `.pto` this PyPTO generates. It is the
# `PTOAS_VERSION` pin in toolchain/versions.env (what CI installs), restated here
# because an installed wheel does not ship that file; bump both in one change.
# tests/ut/backend/test_ptoas_locate.py fails when they differ.
PTOAS_MIN_VERSION = "v0.61"
PTOAS_RELEASES_URL = "https://github.com/hw-native-sys/PTOAS/releases"

# `ptoas --version` prints e.g. "ptoas 0.61"; a dev build may append a suffix.
_VERSION_RE = re.compile(r"\bptoas(?:\s+version)?\s+v?(\d+(?:\.\d+)+)")
_version_lock = threading.Lock()
_verified_binaries: set[str] = set()

# Probed in order under $PTOAS_ROOT — launcher first, the three entries are NOT
# interchangeable:
#
# - up to v0.50, `<root>/ptoas` is a shell launcher exporting
#   `LD_LIBRARY_PATH=<root>/lib`; the bare `<root>/bin/ptoas` has no RUNPATH and
#   dies with "libMLIR*.so: cannot open shared object file".
# - from v0.51, `<root>/ptoas` is a Python package *directory* (not executable)
#   and `<root>/bin/ptoas` links self-sufficiently.
# - from v0.55 the release bundles its own CPython and only `<root>/ptoas.sh`
#   selects it; `<root>/bin/ptoas` runs under the caller's `env python3` and
#   fails with "this ptoas compiler archive requires CPython <x.y>".
PTOAS_RELATIVE_PATHS = ("ptoas", "ptoas.sh", "bin/ptoas")


def find_ptoas_binary() -> str | None:
    """Locate the ``ptoas`` executable.

    When ``PTOAS_ROOT`` is set only that directory is searched — falling back to
    ``PATH`` would silently compile with a different PTOAS than the pinned one.

    Returns:
        Path to the executable, or ``None`` when no executable ``ptoas`` exists.
    """
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if not ptoas_root:
        return shutil.which("ptoas")

    for relative in PTOAS_RELATIVE_PATHS:
        candidate = os.path.join(ptoas_root, relative)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.removeprefix("v").split("."))


def check_ptoas_version(ptoas_bin: str) -> None:
    """Reject a ``ptoas`` older than :data:`PTOAS_MIN_VERSION`.

    An older assembler rejects instruction forms the current codegen emits, and
    reports it against a line of a generated ``.pto`` without ever naming the
    version as the cause. The ``ptoas --version`` probe runs once per executable
    per process; concurrent callers wait for that result instead of re-probing.
    A failed check is not remembered, so fixing ``PTOAS_ROOT`` takes effect.

    Args:
        ptoas_bin: Path to the ``ptoas`` executable, e.g. from
            :func:`find_ptoas_binary`.

    Raises:
        RuntimeError: If the version cannot be determined, or is older than
            :data:`PTOAS_MIN_VERSION`.
    """
    with _version_lock:
        if ptoas_bin in _verified_binaries:
            return
        install_hint = (
            f"Install PTOAS {PTOAS_MIN_VERSION} or newer from {PTOAS_RELEASES_URL} "
            "and point PTOAS_ROOT at it."
        )
        try:
            result = subprocess.run(
                [ptoas_bin, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not run '{ptoas_bin} --version': {exc}. {install_hint}") from exc

        output = f"{result.stdout}\n{result.stderr}".strip()
        match = _VERSION_RE.search(output) if result.returncode == 0 else None
        if match is None:
            raise RuntimeError(
                f"Could not determine the version of ptoas at '{ptoas_bin}': '--version' exited with "
                f"{result.returncode} and printed {output[:300]!r}. PyPTO requires PTOAS >= "
                f"{PTOAS_MIN_VERSION}. {install_hint}"
            )
        found = match.group(1)
        if _parse_version(found) < _parse_version(PTOAS_MIN_VERSION):
            raise RuntimeError(
                f"ptoas at '{ptoas_bin}' is version {found}, but PyPTO requires PTOAS >= "
                f"{PTOAS_MIN_VERSION}. {install_hint}"
            )
        _verified_binaries.add(ptoas_bin)
