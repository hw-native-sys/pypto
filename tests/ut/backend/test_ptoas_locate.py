# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Discovery of the ``ptoas`` executable under ``$PTOAS_ROOT``.

The release layout changed twice, and discovery must handle all three shapes:

- pre-v0.51 tarball: ``<root>/ptoas`` is a shell launcher exporting
  ``LD_LIBRARY_PATH=<root>/lib`` before exec'ing the bare ``<root>/bin/ptoas``.
- wheel-into-a-venv (what CI installs): only ``<root>/bin/ptoas`` exists, and
  its shebang names the venv's own interpreter, so it is self-sufficient.
- v0.55+ standalone archive: ``<root>/ptoas`` is a Python package *directory*,
  ``<root>/bin/ptoas`` is a bare ``#!/usr/bin/env python3`` wrapper that aborts
  under anything but the CPython pinned in ``<root>/.ptoas-python-version``, and
  ``<root>/ptoas.sh`` is the launcher that execs the bundled interpreter.

So discovery must never mistake a *directory* named ``ptoas`` for the
executable, and must keep launchers ahead of ``bin/ptoas`` — on pre-v0.51 that
bare binary has no RUNPATH and dies on its bundled MLIR shared objects, and on
v0.55+ it dies on the interpreter check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import (
    PTOAS_PYTHON_REQUIREMENT_FILE,
    describe_python_requirement_mismatch,
    find_ptoas_binary,
)


def _make_executable(path: Path) -> Path:
    """Create *path* (and parents) as an executable stub file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_finds_binary_in_bin_subdir(tmp_path, monkeypatch):
    """v0.51 layout: only ``<root>/bin/ptoas`` exists."""
    root = tmp_path / "ptoas-bin"
    expected = _make_executable(root / "bin" / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def test_finds_launcher_at_root(tmp_path, monkeypatch):
    """Pre-v0.51 layout: ``<root>/ptoas`` is the launcher script."""
    root = tmp_path / "ptoas-bin"
    expected = _make_executable(root / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def test_launcher_wins_over_bare_binary(tmp_path, monkeypatch):
    """Pre-v0.51 ships both; the launcher must win.

    The bare ``bin/ptoas`` has no RUNPATH there and only resolves its bundled
    MLIR shared objects through the ``LD_LIBRARY_PATH`` the launcher exports.
    """
    root = tmp_path / "ptoas-bin"
    expected = _make_executable(root / "ptoas")
    _make_executable(root / "bin" / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def test_package_dir_named_ptoas_is_skipped(tmp_path, monkeypatch):
    """v0.51 ships ``<root>/ptoas`` as a package dir — ``bin/ptoas`` must win."""
    root = tmp_path / "ptoas-bin"
    (root / "ptoas").mkdir(parents=True)
    (root / "ptoas" / "__init__.py").write_text("")
    expected = _make_executable(root / "bin" / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def _make_archive_root(root: Path, required_python: str | None = "3.11") -> Path:
    """Build a v0.55+ standalone archive tree, minus its ``ptoas.sh`` launcher.

    ``<root>/ptoas`` is the Python package directory and ``<root>/bin/ptoas``
    the bare wrapper — the two entries that already existed before the launcher
    was introduced, which is exactly why the launcher must be probed between
    them.
    """
    (root / "ptoas").mkdir(parents=True)
    (root / "ptoas" / "_cli.py").write_text("")
    _make_executable(root / "bin" / "ptoas")
    if required_python is not None:
        (root / PTOAS_PYTHON_REQUIREMENT_FILE).write_text(f"{required_python}\n")
    return root


def test_finds_launcher_script_in_archive_layout(tmp_path, monkeypatch):
    """v0.55+ archive: ``ptoas.sh`` must win over the bare ``bin/ptoas``.

    That wrapper's ``#!/usr/bin/env python3`` resolves to whatever interpreter
    is active, and the archive aborts unless it is the pinned one.
    """
    root = _make_archive_root(tmp_path / "ptoas-0.57")
    expected = _make_executable(root / "ptoas.sh")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def test_root_launcher_wins_over_launcher_script(tmp_path, monkeypatch):
    """Ordering guard: a pre-v0.51 ``<root>/ptoas`` still outranks ``ptoas.sh``."""
    root = tmp_path / "ptoas-bin"
    expected = _make_executable(root / "ptoas")
    _make_executable(root / "ptoas.sh")
    _make_executable(root / "bin" / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(expected)


def test_launcher_script_symlink_is_followed(tmp_path, monkeypatch):
    """The installer ships ``ptoas.sh`` as a symlink to a versioned launcher."""
    launcher = _make_executable(tmp_path / "bin" / "ptoas-0.57")
    root = _make_archive_root(tmp_path / "ptoas-0.57-tree")
    (root / "ptoas.sh").symlink_to(launcher)
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(root / "ptoas.sh")


def test_dangling_launcher_symlink_falls_through(tmp_path, monkeypatch):
    """A relocated tree leaves ``ptoas.sh`` dangling — do not return it."""
    root = _make_archive_root(tmp_path / "ptoas-0.57")
    (root / "ptoas.sh").symlink_to(tmp_path / "gone" / "ptoas-0.57")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(root / "bin" / "ptoas")


def test_non_executable_launcher_script_is_skipped(tmp_path, monkeypatch):
    """A ``ptoas.sh`` without the executable bit must not shadow ``bin/ptoas``."""
    root = _make_archive_root(tmp_path / "ptoas-0.57")
    (root / "ptoas.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "ptoas.sh").chmod(0o644)
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() == str(root / "bin" / "ptoas")


def test_python_mismatch_is_described_for_bare_wrapper(tmp_path):
    """A hand-extracted archive pinning another CPython gets an actionable hint."""
    other = "3.11" if sys.version_info[:2] != (3, 11) else "3.12"
    root = _make_archive_root(tmp_path / "ptoas-0.57", required_python=other)

    message = describe_python_requirement_mismatch(str(root / "bin" / "ptoas"))

    assert message is not None
    assert other in message
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in message
    assert str(root) in message


def test_matching_python_requirement_is_not_reported(tmp_path):
    """The pinned interpreter is the running one — nothing to explain."""
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    root = _make_archive_root(tmp_path / "ptoas-0.57", required_python=running)

    assert describe_python_requirement_mismatch(str(root / "bin" / "ptoas")) is None


def test_venv_layout_pins_no_interpreter(tmp_path):
    """The wheel-into-a-venv layout has no requirement file to trip over."""
    root = tmp_path / "ptoas-venv"
    binary = _make_executable(root / "bin" / "ptoas")

    assert describe_python_requirement_mismatch(str(binary)) is None


def test_launcher_never_reports_a_mismatch(tmp_path):
    """``ptoas.sh`` execs the bundled CPython, so PyPTO's own version is moot."""
    other = "3.11" if sys.version_info[:2] != (3, 11) else "3.12"
    root = _make_archive_root(tmp_path / "ptoas-0.57", required_python=other)
    launcher = _make_executable(root / "ptoas.sh")

    assert describe_python_requirement_mismatch(str(launcher)) is None


def test_returns_none_when_root_has_no_executable(tmp_path, monkeypatch):
    """A ``ptoas`` directory with no ``bin/ptoas`` resolves to nothing."""
    root = tmp_path / "ptoas-bin"
    (root / "ptoas").mkdir(parents=True)
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() is None


def test_non_executable_file_is_rejected(tmp_path, monkeypatch):
    """A present but non-executable ``ptoas`` must not be returned."""
    root = tmp_path / "ptoas-bin"
    root.mkdir(parents=True)
    (root / "ptoas").write_text("not executable\n")
    (root / "ptoas").chmod(0o644)
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    assert find_ptoas_binary() is None


def test_ptoas_root_is_not_supplemented_by_path(tmp_path, monkeypatch):
    """An explicit PTOAS_ROOT pins the toolchain — PATH must not fill in."""
    path_dir = tmp_path / "on-path"
    _make_executable(path_dir / "ptoas")
    root = tmp_path / "ptoas-bin"
    root.mkdir(parents=True)

    monkeypatch.setenv("PTOAS_ROOT", str(root))
    monkeypatch.setenv("PATH", str(path_dir) + os.pathsep + os.environ.get("PATH", ""))

    assert find_ptoas_binary() is None


def test_falls_back_to_path_when_root_unset(tmp_path, monkeypatch):
    """Without PTOAS_ROOT, discovery goes through PATH."""
    path_dir = tmp_path / "on-path"
    expected = _make_executable(path_dir / "ptoas")

    monkeypatch.delenv("PTOAS_ROOT", raising=False)
    monkeypatch.setenv("PATH", str(path_dir))

    assert find_ptoas_binary() == str(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
