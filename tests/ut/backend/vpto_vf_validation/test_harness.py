# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression checks for worker startup and retained subprocess diagnostics."""

import ctypes
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN = runpy.run_path(str(Path(__file__).with_name("test_validation.py")))["run"]


def test_nonzero_exit_keeps_diagnostics(tmp_path):
    with pytest.raises(AssertionError, match="exited 7"):
        RUN(tmp_path, "failure", [sys.executable, "-c", "print('launch failed'); raise SystemExit(7)"])
    assert "launch failed" in (tmp_path / "failure.log").read_text()
    assert sys.executable in (tmp_path / "failure.command.txt").read_text()


def test_timeout_keeps_partial_diagnostics(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        RUN(
            tmp_path,
            "timeout",
            [
                sys.executable,
                "-c",
                "import time; print('before launch', flush=True); time.sleep(30)",
            ],
            timeout=1,
        )
    assert (tmp_path / "timeout.log").read_text() == "TIMEOUT after 1s\nbefore launch\n"


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/st/runtime/vpto/test_alias_helper_sync.py",
        "tests/st/runtime/vpto/test_tile_helper_sync.py",
        "tests/st/runtime/vpto/test_reg_scope_onboard.py",
        "tests/st/runtime/vpto/test_vpto_vf_onboard.py",
        "tests/ut/backend/vpto_vf_validation/test_alias_sync.py",
        "examples/runtime/vf_emitc_vpto.py",
        "examples/runtime/vf_adapter_ub_overlap.py",
    ],
)
def test_worker_imports_outside_checkout(tmp_path, relative_path, run_without_optional_runtime):
    root = Path(__file__).resolve().parents[4]
    source = root / relative_path
    # The shared guard disables installed PyTorch plugin autoload and rejects
    # every attempted optional-runtime import, even if ImportError is caught.
    result = run_without_optional_runtime(f"""
import os, runpy, sys
from pathlib import Path
sys.path = [p for p in sys.path if p and Path(p).resolve() != Path({str(root)!r})]
os.chdir({str(tmp_path)!r})
assert {str(root)!r} not in sys.path
sys.path.insert(0, {str(source.parent)!r})
runpy.run_path({str(source)!r}, run_name="vpto_import_probe")
""")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "failure", [None, "aclInit", "aclrtSetDevice", "aclrtGetDevice", "empty_name", "wrong_device"]
)
def test_acl_device_probe_selects_and_cleans_up(monkeypatch, failure):
    root = Path(__file__).resolve().parents[4]
    probe = runpy.run_path(str(root / "tests/st/runtime/vpto/device_probe.py"))["probe_device"]
    calls = []

    def operation(name):
        def call(*args):
            calls.append((name, args))
            if name == failure:
                return 17
            if name == "aclrtGetDevice":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int32))[0] = (
                    3 if failure == "wrong_device" else 4
                )
            if name == "aclrtGetSocName":
                return None if failure == "empty_name" else b"Ascend950PR_9599"
            return 0

        return call

    acl = SimpleNamespace(
        **{
            name: operation(name)
            for name in (
                "aclInit",
                "aclFinalize",
                "aclrtSetDevice",
                "aclrtResetDevice",
                "aclrtGetDevice",
                "aclrtGetSocName",
            )
        }
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: acl)
    if failure:
        with pytest.raises(RuntimeError):
            probe(4)
    else:
        assert probe(4) == "Ascend950PR_9599"
    names = [name for name, _ in calls]
    if failure == "aclInit":
        assert names == ["aclInit"]
    else:
        assert names[-1] == "aclFinalize"
        assert ("aclrtSetDevice", (4,)) in calls
        if failure != "aclrtSetDevice":
            assert names[-2] == "aclrtResetDevice"
            assert ("aclrtResetDevice", (4,)) in calls


@pytest.mark.parametrize("name", ["Ascend950PR_9599", "Ascend910B"])
def test_vf_preflight_retains_device_type_check(monkeypatch, tmp_path, name):
    root = Path(__file__).resolve().parents[4]
    fixture = runpy.run_path(str(root / "tests/st/runtime/vpto/conftest.py"))["vpto_onboard_session"]
    monkeypatch.setenv("PYPTO_RUN_VPTO_ONBOARD", "1")
    monkeypatch.setenv("PYPTO_VPTO_DEVICE_ID", "4")
    monkeypatch.setenv("PYPTO_VPTO_RUNTIME_OUTPUT", str(tmp_path))

    def run(command, **kwargs):
        assert Path(command[1]).name == "device_probe.py"
        assert command[2] == "4"
        assert kwargs["timeout"] == 60
        return SimpleNamespace(returncode=0, stdout=f"VPTO_DEVICE_NAME={name}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    if name.startswith("Ascend950"):
        assert fixture.__wrapped__() == tmp_path
    else:
        with pytest.raises(AssertionError, match="requires an A5"):
            fixture.__wrapped__()
    assert name in (tmp_path / "device.txt").read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
