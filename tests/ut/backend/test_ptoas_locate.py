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
self-sufficient ``<root>/bin/ptoas`` as the only binary. It changed again at
v0.55, which bundles its own CPython behind a new ``<root>/ptoas.sh`` launcher.

Discovery must therefore handle all three layouts, must never mistake a
*directory* named ``ptoas`` for the executable, and must keep launcher-first
ordering — on pre-v0.51 the bare ``bin/ptoas`` has no RUNPATH and dies on its
bundled MLIR shared objects, and on v0.55+ it runs under the caller's interpreter
instead of the bundled one.

Once found, the executable must be at least the pinned ``PTOAS_VERSION``;
``check_ptoas_version`` enforces that before any ``.pto`` reaches the assembler.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import PTOAS_MIN_VERSION, check_ptoas_version, find_ptoas_binary

_REPO_ROOT = Path(__file__).resolve().parents[3]
_major, _minor = (int(part) for part in PTOAS_MIN_VERSION.removeprefix("v").split(".")[:2])
_PINNED = PTOAS_MIN_VERSION.removeprefix("v")
_NEWER = f"{_major}.{_minor + 1}"
_OLDER = f"{_major}.{_minor - 1}" if _minor else f"{_major - 1}.99"


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


def test_shell_launcher_wins_on_standalone_layout(tmp_path, monkeypatch):
    """v0.55+ ships all three entries; ``ptoas.sh`` must win.

    Only ``ptoas.sh`` selects the bundled CPython. ``bin/ptoas`` is a
    ``#!/usr/bin/env python3`` script and fails with "this ptoas compiler archive
    requires CPython <x.y>" whenever the caller's interpreter differs.
    """
    root = tmp_path / "ptoas-bin"
    (root / "ptoas").mkdir(parents=True)
    (root / "ptoas" / "__init__.py").write_text("")
    expected = _make_executable(root / "ptoas.sh")
    _make_executable(root / "bin" / "ptoas")
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


def _make_versioned_ptoas(path: Path, output: str, exit_code: int = 0) -> Path:
    """Create a fake ``ptoas`` that logs its argv and answers ``--version`` with *output*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = shlex.quote(str(path.parent / "calls.log"))
    path.write_text(
        f'#!/bin/sh\necho "$*" >> {log}\nprintf "%s\\n" {shlex.quote(output)}\nexit {exit_code}\n'
    )
    path.chmod(0o755)
    return path


def _calls(ptoas: Path) -> list[str]:
    log = ptoas.parent / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def test_min_version_matches_toolchain_pin():
    """The runtime minimum must track the version CI installs."""
    pins = dict(
        line.split("=", 1)
        for line in (_REPO_ROOT / "toolchain" / "versions.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert pins["PTOAS_VERSION"] == PTOAS_MIN_VERSION, (
        "toolchain/versions.env PTOAS_VERSION and _ptoas_locate.PTOAS_MIN_VERSION must be bumped together"
    )


@pytest.mark.parametrize(
    "output",
    [
        f"ptoas {_PINNED}",
        f"ptoas v{_PINNED}",
        f"ptoas {_PINNED}.1",
        f"ptoas {_NEWER}",
        f"ptoas {_NEWER}.0git",
    ],
)
def test_accepts_pinned_or_newer_version(tmp_path, output):
    ptoas = _make_versioned_ptoas(tmp_path / "bin" / "ptoas", output)

    check_ptoas_version(str(ptoas))


def test_rejects_older_version(tmp_path):
    ptoas = _make_versioned_ptoas(tmp_path / "bin" / "ptoas", f"ptoas {_OLDER}")

    with pytest.raises(
        RuntimeError, match=rf"is version {_OLDER}, but PyPTO requires PTOAS >= {PTOAS_MIN_VERSION}"
    ):
        check_ptoas_version(str(ptoas))


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [("usage: ptoas [options] <input file>", 0), (f"ptoas {_PINNED}", 1), ("", 0)],
)
def test_rejects_undeterminable_version(tmp_path, output, exit_code):
    """A pre-``--version`` build or a failing probe must not pass as current."""
    ptoas = _make_versioned_ptoas(tmp_path / "bin" / "ptoas", output, exit_code)

    with pytest.raises(RuntimeError, match="Could not determine the version of ptoas"):
        check_ptoas_version(str(ptoas))


def test_probe_runs_once_per_binary(tmp_path):
    ptoas = _make_versioned_ptoas(tmp_path / "bin" / "ptoas", f"ptoas {_PINNED}")

    check_ptoas_version(str(ptoas))
    check_ptoas_version(str(ptoas))

    assert _calls(ptoas) == ["--version"]


def test_failed_check_is_not_cached(tmp_path):
    """Replacing a stale install in place must take effect without a restart."""
    ptoas = tmp_path / "bin" / "ptoas"
    _make_versioned_ptoas(ptoas, f"ptoas {_OLDER}")
    with pytest.raises(RuntimeError):
        check_ptoas_version(str(ptoas))

    _make_versioned_ptoas(ptoas, f"ptoas {_PINNED}")
    check_ptoas_version(str(ptoas))


def test_relative_path_cannot_reuse_another_directorys_result(tmp_path, monkeypatch):
    """A relative PTOAS_ROOT names a different ptoas once the cwd changes."""
    current, stale = tmp_path / "current", tmp_path / "stale"
    _make_versioned_ptoas(current / "ptoas-bin" / "bin" / "ptoas", f"ptoas {_PINNED}")
    _make_versioned_ptoas(stale / "ptoas-bin" / "bin" / "ptoas", f"ptoas {_OLDER}")
    monkeypatch.setenv("PTOAS_ROOT", "ptoas-bin")

    monkeypatch.chdir(current)
    ptoas = find_ptoas_binary()
    assert ptoas is not None
    assert ptoas == os.path.join("ptoas-bin", "bin", "ptoas")
    check_ptoas_version(ptoas)

    monkeypatch.chdir(stale)
    with pytest.raises(RuntimeError, match=rf"is version {_OLDER}"):
        check_ptoas_version(ptoas)


def test_run_ptoas_rejects_older_version_before_assembling(tmp_path, monkeypatch):
    root = tmp_path / "ptoas-bin"
    ptoas = _make_versioned_ptoas(root / "bin" / "ptoas", f"ptoas {_OLDER}")
    monkeypatch.setenv("PTOAS_ROOT", str(root))

    with pytest.raises(RuntimeError, match=rf"requires PTOAS >= {PTOAS_MIN_VERSION}"):
        pto_backend._run_ptoas(str(tmp_path / "k.pto"), str(tmp_path / "k.cpp"))

    assert _calls(ptoas) == ["--version"]


def test_older_version_still_leaves_the_generated_pto(tmp_path, monkeypatch):
    """The check fails the assembly step, not codegen: the ``.pto`` is written first.

    Codegen-only runs and debugging read that artifact, and a failing assembler
    always left it behind.
    """
    root = tmp_path / "ptoas-bin"
    ptoas = _make_versioned_ptoas(root / "bin" / "ptoas", f"ptoas {_OLDER}")
    monkeypatch.setenv("PTOAS_ROOT", str(root))
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: [])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match=rf"requires PTOAS >= {PTOAS_MIN_VERSION}"):
        pto_backend._compile_pto_module("module {}", "kernel_a", str(out))

    assert (out / "ptoas" / "kernel_a.pto").read_text() == "module {}"
    assert _calls(ptoas) == ["--version"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
