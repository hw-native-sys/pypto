# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for managed PTO-ISA revision resolution."""

import importlib
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_RUNTIME_PIN = "83d01313d9bfc247c4b7c8bcf969d1019f0d106f"


@pytest.fixture
def device_runner(monkeypatch):
    """Import ``device_runner`` without requiring the optional simpler package."""
    fake_kernel_compiler = SimpleNamespace(KernelCompiler=object)
    fake_task_interface = SimpleNamespace(
        CallConfig=object,
        ChipCallable=object,
        ChipStorageTaskArgs=object,
        CoreCallable=object,
        Worker=object,
        make_tensor_arg=object,
        scalar_to_uint64=object,
    )
    monkeypatch.setitem(sys.modules, "pypto.runtime.kernel_compiler", fake_kernel_compiler)
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", fake_task_interface)

    module_name = "pypto.runtime.device_runner"
    previous = sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


def _configure_existing_clone(device_runner, monkeypatch, tmp_path):
    clone_path = tmp_path / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(device_runner, "_get_pto_isa_clone_path", lambda: clone_path)
    return clone_path


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


def test_explicit_revision_overrides_runtime_pin(device_runner, monkeypatch, tmp_path):
    clone_path = _configure_existing_clone(device_runner, monkeypatch, tmp_path)
    pin_path = tmp_path / "pto_isa.pin"
    pin_path.write_text(f"{_RUNTIME_PIN}\n")
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    checkout = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)

    device_runner.ensure_pto_isa_root(commit="explicit-revision")

    checkout.assert_called_once_with(clone_path, "explicit-revision")


def test_environment_root_is_checked_out_at_runtime_pin(device_runner, monkeypatch, tmp_path):
    pto_isa_root = tmp_path / "external-pto-isa"
    pin_path = tmp_path / "pto_isa.pin"
    pin_path.write_text(f"{_RUNTIME_PIN}\n")
    monkeypatch.setenv("PTO_ISA_ROOT", str(pto_isa_root))
    monkeypatch.setattr(device_runner, "_PTO_ISA_PIN_PATH", pin_path)
    checkout = Mock()
    monkeypatch.setattr(device_runner, "_checkout_pto_isa_commit", checkout)

    assert device_runner.ensure_pto_isa_root() == str(pto_isa_root)

    checkout.assert_called_once_with(pto_isa_root, _RUNTIME_PIN)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
