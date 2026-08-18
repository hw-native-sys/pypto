# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for PTO-ISA resolution and device-runner diagnostics."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from pypto.runtime import pto_isa


@pytest.fixture
def fake_simpler_pto_isa(monkeypatch, tmp_path):
    """Stand in for ``simpler_setup.pto_isa`` and count resolver calls."""
    pto_isa._resolve_pinned_pto_isa_root.cache_clear()

    pinned_root = tmp_path / "managed" / "pto-isa"
    (pinned_root / "include").mkdir(parents=True)

    resolve = Mock(return_value=str(pinned_root))
    module = ModuleType("simpler_setup.pto_isa")
    setattr(module, "ensure_pto_isa_root", resolve)
    package = ModuleType("simpler_setup")
    setattr(package, "__path__", [])
    monkeypatch.setitem(sys.modules, "simpler_setup", package)
    monkeypatch.setitem(sys.modules, "simpler_setup.pto_isa", module)

    yield SimpleNamespace(root=pinned_root, resolve=resolve)

    pto_isa._resolve_pinned_pto_isa_root.cache_clear()


def test_resolution_delegates_to_simpler_and_exports_env(monkeypatch, fake_simpler_pto_isa):
    """The pinned checkout comes from Simpler; PyPTO only exports the result."""
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)

    resolved = pto_isa.ensure_pto_isa_root()

    assert resolved == str(fake_simpler_pto_isa.root.resolve())
    assert pto_isa.os.environ["PTO_ISA_ROOT"] == resolved
    fake_simpler_pto_isa.resolve.assert_called_once()


def test_ambient_environment_root_is_ignored(monkeypatch, fake_simpler_pto_isa, tmp_path):
    """An exported PTO_ISA_ROOT is not the pin — it must never win."""
    stale_root = tmp_path / "stale-off-pin-pto-isa"
    stale_root.mkdir()
    monkeypatch.setenv("PTO_ISA_ROOT", str(stale_root))

    resolved = pto_isa.ensure_pto_isa_root()

    assert resolved == str(fake_simpler_pto_isa.root.resolve())
    assert pto_isa.os.environ["PTO_ISA_ROOT"] == resolved
    fake_simpler_pto_isa.resolve.assert_called_once()


def test_include_dir_is_derived_from_the_pinned_root(fake_simpler_pto_isa):
    include_dir = pto_isa.pto_isa_include_dir()

    assert include_dir == fake_simpler_pto_isa.root.resolve() / "include"
    assert include_dir.is_dir()


def test_resolution_is_cached_per_process(fake_simpler_pto_isa):
    """The delegate takes a file lock, so repeated compiles must not re-enter it."""
    first = pto_isa.ensure_pto_isa_root()
    second = pto_isa.ensure_pto_isa_root()

    assert first == second
    fake_simpler_pto_isa.resolve.assert_called_once()


def test_unresolvable_pin_raises_instead_of_returning_a_root(monkeypatch):
    """A failure must surface Simpler's diagnostic, not a silently unpinned path."""
    pto_isa._resolve_pinned_pto_isa_root.cache_clear()
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)

    module = ModuleType("simpler_setup.pto_isa")
    setattr(module, "ensure_pto_isa_root", Mock(side_effect=OSError("PTO-ISA not available.")))
    package = ModuleType("simpler_setup")
    setattr(package, "__path__", [])
    monkeypatch.setitem(sys.modules, "simpler_setup", package)
    monkeypatch.setitem(sys.modules, "simpler_setup.pto_isa", module)

    with pytest.raises(OSError, match="PTO-ISA not available"):
        pto_isa.ensure_pto_isa_root()

    assert "PTO_ISA_ROOT" not in pto_isa.os.environ
    pto_isa._resolve_pinned_pto_isa_root.cache_clear()


def _write_raw_pto(work_dir):
    pto_path = work_dir / "kernels" / "aiv" / "tile_add.pto"
    pto_path.parent.mkdir(parents=True)
    pto_path.write_text("module {}", encoding="utf-8")


def _missing_config_message(device_runner, work_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        device_runner.compile_and_assemble(work_dir, "a2a3")
    return str(exc_info.value)


def test_missing_kernel_config_explains_unavailable_ptoas(device_runner, monkeypatch, tmp_path):
    _write_raw_pto(tmp_path)
    monkeypatch.delenv("PTOAS_ROOT", raising=False)
    monkeypatch.setattr(device_runner.shutil, "which", lambda _name: None)

    message = _missing_config_message(device_runner, tmp_path)
    assert str(tmp_path / "kernel_config.py") in message
    assert "compile-only artifact produced with skip_ptoas=True" in message
    assert "PTOAS_ROOT is not set" in message
    assert "'ptoas' was not found on PATH" in message
    assert "export PTOAS_ROOT=/path/to/ptoas-bin" in message
    assert "Restart the Python process and rerun" in message


def test_missing_kernel_config_reports_invalid_ptoas_root(device_runner, monkeypatch, tmp_path):
    _write_raw_pto(tmp_path)
    ptoas_root = tmp_path / "invalid-ptoas"
    monkeypatch.setenv("PTOAS_ROOT", str(ptoas_root))

    message = _missing_config_message(device_runner, tmp_path)
    assert f"PTOAS_ROOT is set to '{ptoas_root}'" in message
    assert str(ptoas_root / "ptoas") in message
    assert "does not exist or is not executable" in message
    assert "Correct or remove the invalid PTOAS_ROOT setting" in message
    assert "unset PTOAS_ROOT" in message
    assert 'unset PTOAS_ROOT\n       eval "$(pypto-setup --export)"' in message


def test_missing_kernel_config_requires_recompile_when_ptoas_is_now_available(
    device_runner, monkeypatch, tmp_path
):
    _write_raw_pto(tmp_path)
    ptoas_root = tmp_path / "ptoas-bin"
    ptoas_root.mkdir()
    ptoas_path = ptoas_root / "ptoas"
    ptoas_path.write_text("#!/bin/sh\n", encoding="utf-8")
    ptoas_path.chmod(0o755)
    monkeypatch.setenv("PTOAS_ROOT", str(ptoas_root))

    message = _missing_config_message(device_runner, tmp_path)
    assert f"ptoas is now available at '{ptoas_path}'" in message
    assert "artifact was generated without it and must be recompiled" in message
    assert "export PTOAS_ROOT" not in message


def test_missing_kernel_config_requires_recompile_when_ptoas_is_now_on_path(
    device_runner, monkeypatch, tmp_path
):
    _write_raw_pto(tmp_path)
    monkeypatch.delenv("PTOAS_ROOT", raising=False)
    ptoas_path = tmp_path / "bin" / "ptoas"
    monkeypatch.setattr(device_runner.shutil, "which", lambda _name: str(ptoas_path))

    message = _missing_config_message(device_runner, tmp_path)
    assert f"ptoas is now available on PATH at '{ptoas_path}'" in message
    assert "artifact was generated without it and must be recompiled" in message
    assert "export PTOAS_ROOT" not in message


def test_missing_kernel_config_reports_incomplete_artifact(device_runner, tmp_path):
    message = _missing_config_message(device_runner, tmp_path)
    assert str(tmp_path / "kernel_config.py") in message
    assert "may be incomplete or use a different build layout" in message
    assert "skip_ptoas=True" not in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
