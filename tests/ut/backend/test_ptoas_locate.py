# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Discovery of the ``ptoas`` executable under ``$PTOAS_ROOT``.

The release tarball layout changed at v0.51: ``<root>/ptoas`` went from being a
shell launcher (which exports ``LD_LIBRARY_PATH=<root>/lib`` before exec'ing the
bare ``<root>/bin/ptoas``) to being a Python package *directory*, leaving the now
self-sufficient ``<root>/bin/ptoas`` as the only binary.

Discovery must therefore handle both layouts, must never mistake a *directory*
named ``ptoas`` for the executable, and must keep launcher-first ordering — on
pre-v0.51 the bare ``bin/ptoas`` has no RUNPATH and dies on its bundled MLIR
shared objects.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary


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
