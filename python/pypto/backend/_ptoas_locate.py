# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared discovery, version validation and feature probes for ``ptoas``."""

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from pypto._environment import toolchain_probe_env

# Baseline PTOAS release for ordinary kernels; manual Buffer-ID additionally
# requires the capability probe below. It is the
# `PTOAS_VERSION` pin in toolchain/versions.env (what CI installs), restated here
# because an installed wheel does not ship that file; bump both in one change.
# tests/ut/backend/test_ptoas_locate.py fails when they differ.
PTOAS_MIN_VERSION = "v0.77"
PTOAS_RELEASES_URL = "https://github.com/hw-native-sys/PTOAS/releases"

# `ptoas --version` prints e.g. "ptoas 0.61"; a dev build may append a suffix.
_VERSION_RE = re.compile(r"\bptoas(?:\s+version)?\s+v?(\d+(?:\.\d+)+)")
_version_lock = threading.Lock()
_verified_binaries: dict[str, str] = {}
_buffer_sync_lock = threading.Lock()
_verified_buffer_sync_binaries: set[tuple[str, int, int]] = set()

# The dynamic function deliberately has no static synchronization: older
# BufidSync passes otherwise mistake the static function's coverage for support.
_BUFFER_SYNC_PROBE = """\
module {
  func.func @__pypto_bufid_static() {
    pto.get_buf "PIPE_MTE2", 0, 0
    pto.rls_buf "PIPE_MTE2", 0, 0
    return
  }
  func.func @__pypto_bufid_dynamic(%src: !pto.partition_tensor_view<1x64xf32>,
                                 %dst: !pto.partition_tensor_view<1x64xf32>, %id: index) {
    %tile = pto.alloc_tile : !pto.tile_buf<vec, 1x64xf32>
    pto.get_buf_dyn "PIPE_MTE2", %id, 0
    pto.tload ins(%src : !pto.partition_tensor_view<1x64xf32>) outs(%tile : !pto.tile_buf<vec, 1x64xf32>)
    pto.rls_buf_dyn "PIPE_MTE2", %id, 0
    pto.get_buf_dyn "PIPE_MTE3", %id, 0
    pto.tstore ins(%tile : !pto.tile_buf<vec, 1x64xf32>) outs(%dst : !pto.partition_tensor_view<1x64xf32>)
    pto.rls_buf_dyn "PIPE_MTE3", %id, 0
    return
  }
}
"""
_BUFFER_SYNC_HINT = (
    "Manual Buffer-ID synchronization requires PTOAS with concrete PipeAttr lowering "
    "and dynamic manual-sync BufidSync skipping (cann/pto-as !462). "
    "Point PTOAS_ROOT at a build containing those fixes; the baseline version alone is insufficient."
)

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


def check_ptoas_version(ptoas_bin: str) -> str:
    """Reject a ``ptoas`` older than :data:`PTOAS_MIN_VERSION`.

    An older assembler rejects instruction forms the current codegen emits, and
    reports it against a line of a generated ``.pto`` without ever naming the
    version as the cause. The ``ptoas --version`` probe runs once per executable
    per process; concurrent callers wait for that result instead of re-probing.
    A failed check is not remembered, so fixing ``PTOAS_ROOT`` takes effect.

    Args:
        ptoas_bin: Path to the ``ptoas`` executable, e.g. from
            :func:`find_ptoas_binary`.

    Returns:
        The probe's complete ``--version`` text. Callers identifying the
        assembler must use this rather than the parsed number: the pattern
        below captures only the numeric part, so a dev build's suffix -- the
        very thing that distinguishes it from the release it was built from --
        is absent from the comparison value.

    Raises:
        RuntimeError: If the version cannot be determined, or is older than
            :data:`PTOAS_MIN_VERSION`.
    """
    with _version_lock:
        # Key by the file the path names now: a relative PTOAS_ROOT or a
        # repointed symlink can make one path string name a different ptoas.
        # The probe itself still runs the unresolved path, as callers do.
        cache_key = os.path.realpath(ptoas_bin)
        if cache_key in _verified_binaries:
            return _verified_binaries[cache_key]
        install_hint = (
            f"Install PTOAS {PTOAS_MIN_VERSION} or newer from {PTOAS_RELEASES_URL} "
            "and point PTOAS_ROOT at it."
        )
        try:
            result = subprocess.run(
                [ptoas_bin, "--version"],
                env=toolchain_probe_env(),
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
        _verified_binaries[cache_key] = output
        return output


def _probe_buffer_sync_output(ptoas_bin: str, source: Path, *, emit_ir: bool) -> str:
    cmd = [ptoas_bin, "--pto-arch=a5", "--enable-bufid_sync", str(source), "-o", "-"]
    if emit_ir:
        cmd.append("--emit-pto-ir")
    result = subprocess.run(
        cmd, env=toolchain_probe_env(c_locale=True), capture_output=True, text=True, check=False, timeout=60
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"{_BUFFER_SYNC_HINT} Capability probe at '{ptoas_bin}' failed "
            f"(exit {result.returncode}): {detail}"
        )
    return result.stdout


def check_ptoas_buffer_sync(ptoas_bin: str) -> None:
    """Require the manual Buffer-ID backend fixes, independently of version.

    Probe all four concrete-pipe forms through EmitC and verify that automatic
    BufidSync leaves a dynamic-only manually synchronized function untouched.
    Successful probes are cached per resolved executable, size and mtime;
    failures are not cached. Ordinary kernels do not require this capability.

    Args:
        ptoas_bin: The executable selected by :func:`find_ptoas_binary`.

    Raises:
        RuntimeError: If either backend capability is unavailable or untestable.
    """
    try:
        stat = os.stat(ptoas_bin)
        cache_key = (os.path.realpath(ptoas_bin), stat.st_size, stat.st_mtime_ns)
        with _buffer_sync_lock:
            if cache_key in _verified_buffer_sync_binaries:
                return
            with TemporaryDirectory(prefix="pypto-bufid-probe-") as directory:
                source = Path(directory) / "probe.pto"
                source.write_text(_BUFFER_SYNC_PROBE)
                cpp = _probe_buffer_sync_output(ptoas_bin, source, emit_ir=False)
                pto_ir = _probe_buffer_sync_output(ptoas_bin, source, emit_ir=True)
            _, marker, dynamic_body = pto_ir.partition("func.func @__pypto_bufid_dynamic(")
            if not all(token in cpp for token in ("get_buf(", "rls_buf(")) or not marker:
                raise RuntimeError(f"{_BUFFER_SYNC_HINT} Capability probe produced incomplete output.")
            if re.search(r"\bpto\.(?:get|rls)_buf(?=[\s\[])", dynamic_body):
                raise RuntimeError(f"{_BUFFER_SYNC_HINT} BufidSync inserted static synchronization.")
            _verified_buffer_sync_binaries.add(cache_key)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{_BUFFER_SYNC_HINT} Capability probe could not run: {exc}") from exc
