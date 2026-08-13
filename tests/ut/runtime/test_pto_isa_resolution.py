# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for device-runner diagnostics and PTO-ISA resolution."""

import logging
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

_RUNTIME_PIN = "83d01313d9bfc247c4b7c8bcf969d1019f0d106f"


def _configure_existing_clone(device_runner, monkeypatch, tmp_path):
    clone_path = tmp_path / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(device_runner, "_get_pto_isa_clone_path", lambda: clone_path)
    return clone_path


def test_installed_layout_reads_pin_from_runtime_package(device_runner, monkeypatch, tmp_path):
    installed_pin = tmp_path / "simpler_setup" / "_assets" / "pto_isa.pin"
    installed_pin.parent.mkdir(parents=True)
    installed_pin.write_text(f"{_RUNTIME_PIN}\n", encoding="utf-8")
    source_pin = tmp_path / "site-packages" / "runtime" / "pto_isa.pin"
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", source_pin)

    runtime_package = ModuleType("simpler_setup")
    setattr(runtime_package, "__path__", [])
    runtime_environment = ModuleType("simpler_setup.environment")
    setattr(runtime_environment, "PROJECT_ROOT", installed_pin.parent)
    monkeypatch.setitem(sys.modules, "simpler_setup", runtime_package)
    monkeypatch.setitem(sys.modules, "simpler_setup.environment", runtime_environment)

    assert device_runner._read_runtime_pto_isa_pin() == _RUNTIME_PIN


def test_default_revision_uses_runtime_pin(device_runner, monkeypatch, tmp_path):
    clone_path = _configure_existing_clone(device_runner, monkeypatch, tmp_path)
    pin_path = tmp_path / "pto_isa.pin"
    pin_path.write_text(f"{_RUNTIME_PIN}\n")
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    checkout = Mock()
    update_latest = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)
    monkeypatch.setattr(device_runner, "_update_pto_isa_to_latest", update_latest)

    assert device_runner.ensure_pto_isa_root() == str(clone_path.resolve())

    checkout.assert_called_once_with(clone_path, _RUNTIME_PIN)
    update_latest.assert_not_called()


def test_environment_root_is_used_without_checkout(device_runner, monkeypatch, tmp_path):
    pto_isa_root = tmp_path / "external-pto-isa"
    monkeypatch.setenv("PTO_ISA_ROOT", str(pto_isa_root))
    checkout = Mock()
    read_pin = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)
    monkeypatch.setattr(device_runner, "_read_runtime_pto_isa_pin", read_pin)

    assert device_runner.ensure_pto_isa_root() == str(pto_isa_root)

    checkout.assert_not_called()
    read_pin.assert_not_called()


def test_fresh_clone_checks_out_runtime_pin(device_runner, monkeypatch, tmp_path):
    clone_path = tmp_path / "pto-isa"
    pin_path = tmp_path / "pto_isa.pin"
    pin_path.write_text(f"{_RUNTIME_PIN}\n")
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    monkeypatch.setattr(device_runner, "_get_pto_isa_clone_path", lambda: clone_path)

    def clone(*_args):
        (clone_path / "include").mkdir(parents=True)
        return True

    monkeypatch.setattr(device_runner, "_clone_pto_isa", clone)
    checkout = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)

    assert device_runner.ensure_pto_isa_root() == str(clone_path.resolve())

    checkout.assert_called_once_with(clone_path, _RUNTIME_PIN)


def test_checkout_failure_does_not_return_unpinned_managed_clone(device_runner, monkeypatch, tmp_path):
    _configure_existing_clone(device_runner, monkeypatch, tmp_path)
    pin_path = tmp_path / "pto_isa.pin"
    pin_path.write_text(f"{_RUNTIME_PIN}\n", encoding="utf-8")
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", Mock(return_value=False))

    assert device_runner.ensure_pto_isa_root() is None
    assert "PTO_ISA_ROOT" not in device_runner.os.environ


@pytest.mark.parametrize("pin_contents", [None, ""])
def test_unavailable_runtime_pin_falls_back_to_latest(
    device_runner, monkeypatch, tmp_path, caplog, pin_contents
):
    clone_path = _configure_existing_clone(device_runner, monkeypatch, tmp_path)
    pin_path = tmp_path / "pto_isa.pin"
    if pin_contents is not None:
        pin_path.write_text(pin_contents)
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    checkout = Mock()
    update_latest = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)
    monkeypatch.setattr(device_runner, "_update_pto_isa_to_latest", update_latest)

    with caplog.at_level(logging.WARNING):
        device_runner.ensure_pto_isa_root()

    checkout.assert_not_called()
    update_latest.assert_called_once_with(clone_path)
    assert "falling back to the latest remote HEAD" in caplog.text


def _write_raw_pto(work_dir):
    pto_path = work_dir / "kernels" / "aiv" / "tile_add.pto"
    pto_path.parent.mkdir(parents=True)
    pto_path.write_text("module {}", encoding="utf-8")


def _missing_config_message(device_runner, work_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        device_runner.compile_and_assemble(work_dir, "a2a3")
    return str(exc_info.value)


def _make_ptoas_executable(path):
    """Create *path* (and parents) as an executable ptoas stub."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_missing_kernel_config_explains_unavailable_ptoas(device_runner, monkeypatch, tmp_path):
    _write_raw_pto(tmp_path)
    monkeypatch.delenv("PTOAS_ROOT", raising=False)
    empty_path_dir = tmp_path / "empty-path"
    empty_path_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_path_dir))

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
    assert "no executable ptoas was found there" in message
    # Every probed layout is named, so the user can see which tree shape is expected.
    for relative in device_runner._PTOAS_RELATIVE_PATHS:
        assert str(ptoas_root / relative) in message
    assert "Correct or remove the invalid PTOAS_ROOT setting" in message
    assert "unset PTOAS_ROOT" in message
    assert 'unset PTOAS_ROOT\n       eval "$(pypto-setup --export)"' in message


def test_missing_kernel_config_requires_recompile_when_ptoas_is_now_available(
    device_runner, monkeypatch, tmp_path
):
    _write_raw_pto(tmp_path)
    ptoas_root = tmp_path / "ptoas-bin"
    ptoas_path = _make_ptoas_executable(ptoas_root / "ptoas")
    monkeypatch.setenv("PTOAS_ROOT", str(ptoas_root))

    message = _missing_config_message(device_runner, tmp_path)
    assert f"ptoas is now available at '{ptoas_path}'" in message
    assert "artifact was generated without it and must be recompiled" in message
    assert "export PTOAS_ROOT" not in message


def test_missing_kernel_config_accepts_v0_55_archive_root(device_runner, monkeypatch, tmp_path):
    """A v0.55+ archive root is valid: its executable is the ``ptoas.sh`` launcher."""
    _write_raw_pto(tmp_path)
    ptoas_root = tmp_path / "ptoas-0.57"
    (ptoas_root / "ptoas").mkdir(parents=True)
    _make_ptoas_executable(ptoas_root / "bin" / "ptoas")
    launcher = _make_ptoas_executable(ptoas_root / "ptoas.sh")
    monkeypatch.setenv("PTOAS_ROOT", str(ptoas_root))

    message = _missing_config_message(device_runner, tmp_path)
    assert f"ptoas is now available at '{launcher}'" in message
    assert "Correct or remove the invalid PTOAS_ROOT setting" not in message


def test_missing_kernel_config_requires_recompile_when_ptoas_is_now_on_path(
    device_runner, monkeypatch, tmp_path
):
    _write_raw_pto(tmp_path)
    monkeypatch.delenv("PTOAS_ROOT", raising=False)
    ptoas_path = _make_ptoas_executable(tmp_path / "bin" / "ptoas")
    monkeypatch.setenv("PATH", str(ptoas_path.parent))

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
