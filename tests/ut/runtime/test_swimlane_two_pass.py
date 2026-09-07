# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the two-pass L2 swimlane execution protocol.

Enabling ``enable_chip_swimlane`` on an onboard platform captures the dep_gen task
graph (``deps.json``) in a subprocess, then runs a clean-timing swimlane pass
in-process — the two cannot share one process because the runtime leaks SVM
host-register mappings between DFX runs. These tests drive
:func:`pypto.runtime.runner._execute_dfx_passes` directly with recording stubs
(no device, no subprocess) and check :func:`pypto.runtime.runner._build_args_spec`.
"""

import ctypes
import json
from pathlib import Path

import pypto.runtime.runner as _runner
import pytest
import torch
from pypto.runtime import _dep_gen_capture
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.runner import DfxOptions, _build_args_spec, _execute_dfx_passes, _generate_swimlane


def _drive(dfx: DfxOptions, platform: str) -> tuple[list[DfxOptions], int]:
    """Run the helper with stubs that record each in-process pass and capture.

    Returns ``(in_process_passes, capture_calls)``. ``_execute_dfx_passes``
    returns ``None`` (per-run timing is no longer a return value — simpler PR
    #1177); the protocol it drives is asserted via the recorded passes/captures.
    """
    seen: list[DfxOptions] = []
    captures = {"n": 0}

    def run_pass(pass_dfx: DfxOptions) -> None:
        seen.append(pass_dfx)

    def capture_deps() -> None:
        captures["n"] += 1

    assert _execute_dfx_passes(run_pass, capture_deps, dfx, platform) is None
    return seen, captures["n"]


def test_onboard_swimlane_captures_deps_then_times_in_process():
    seen, captures = _drive(DfxOptions(enable_chip_swimlane=True), "a2a3")
    # deps captured once (subprocess), one in-process timing pass.
    assert captures == 1
    assert len(seen) == 1
    timing = seen[0]
    assert timing.enable_chip_swimlane == 4  # True normalizes to the full level
    assert timing.enable_dep_gen is False  # dep_gen forced off on the timing pass


def test_onboard_swimlane_preserves_requested_level():
    # Regression (issue #2385): the two-pass split must carry the requested
    # collection level through to the timing pass, not re-derive an on/off flag.
    seen, captures = _drive(DfxOptions(enable_chip_swimlane=2), "a2a3")
    assert captures == 1
    assert len(seen) == 1
    assert seen[0].enable_chip_swimlane == 2
    assert seen[0].enable_dep_gen is False


def test_onboard_swimlane_with_explicit_dep_gen_still_one_capture():
    # An explicit --enable-dep-gen alongside swimlane must NOT add an in-process
    # dep_gen run: the subprocess capture already produced deps.json.
    seen, captures = _drive(DfxOptions(enable_chip_swimlane=True, enable_dep_gen=True), "a2a3")
    assert captures == 1
    assert len(seen) == 1
    assert seen[0].enable_chip_swimlane == 4 and seen[0].enable_dep_gen is False


def test_onboard_swimlane_timing_dfx_ride_the_in_process_pass():
    # PMU / args-dump / scope-stats are timing-sensitive: they ride the clean
    # in-process timing pass (the subprocess capture is dep_gen-only).
    dfx = DfxOptions(
        enable_chip_swimlane=True,
        enable_pmu=2,
        enable_dump_args=1,
        enable_scope_stats=True,
    )
    seen, captures = _drive(dfx, "a2a3")
    assert captures == 1
    assert len(seen) == 1
    timing = seen[0]
    assert timing.enable_pmu == 2
    assert timing.enable_dump_args == 1
    assert timing.enable_scope_stats is True


def test_only_dep_gen_is_single_pass_no_capture():
    seen, captures = _drive(DfxOptions(enable_dep_gen=True), "a2a3")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0].enable_dep_gen is True
    assert seen[0].enable_chip_swimlane == 0


def test_no_dfx_is_single_pass_no_capture():
    seen, captures = _drive(DfxOptions(), "a2a3")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0] == DfxOptions()


def test_sim_swimlane_stays_single_pass_no_capture():
    # Simulator skips swimlane conversion anyway, so no capture / second run.
    seen, captures = _drive(DfxOptions(enable_chip_swimlane=True), "a2a3sim")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0].enable_chip_swimlane == 4


def test_build_args_spec_host_tensor_saves_real_data(tmp_path):
    # Host tensors are persisted verbatim so data-as-control inputs route the
    # same graph in the child.
    t = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    spec = _build_args_spec([t], tmp_path)
    assert spec[0]["kind"] == "tensor_file"
    reloaded = torch.load(spec[0]["path"])
    assert torch.equal(reloaded, t)


def test_build_args_spec_device_tensor_is_zeros_shape(tmp_path):
    # Device-resident tensors cannot cross the process boundary -> shape+dtype.
    dt = DeviceTensor(data_ptr=0x1000, shape=[16, 32], dtype=torch.bfloat16)
    spec = _build_args_spec([dt], tmp_path)
    assert spec[0] == {"kind": "tensor_zeros", "shape": [16, 32], "dtype": "bfloat16"}


def test_build_args_spec_scalar(tmp_path):
    spec = _build_args_spec([ctypes.c_int32(7)], tmp_path)
    assert spec[0] == {"kind": "scalar", "ctype": "c_int", "value": 7}


def test_build_args_spec_rejects_unknown_type(tmp_path):
    with pytest.raises(TypeError):
        _build_args_spec(["not an arg"], tmp_path)  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]


def test_dep_capture_reconstructs_ring_config(tmp_path, monkeypatch, stub_device_runner):
    stub_device_runner._compile_and_assemble.return_value = (object(), "fake_runtime", {})

    spec_path = tmp_path / "capture.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "argspec",
                "args": [],
                "work_dir": str(tmp_path),
                "platform": "a2a3",
                "device_id": 0,
                "dfx_dir": str(tmp_path / "dfx"),
                "ring_overrides": {
                    "ring_task_window": [16, 32, 64, 128],
                    "ring_heap": 512 * 1024 * 1024,
                    "ring_dep_pool": [64, 0, 0, 256],
                },
            }
        ),
        encoding="utf-8",
    )

    assert _dep_gen_capture.main([str(spec_path)]) == 0
    config = stub_device_runner._execute_on_device.call_args.kwargs["config"]
    assert config.ring_task_window == [16, 32, 64, 128]
    assert config.ring_heap == 512 * 1024 * 1024
    assert config.ring_dep_pool == [64, 0, 0, 256]


def _converter_argv(monkeypatch, work_dir: Path, swimlane_dir: Path) -> list[str]:
    """Run ``_generate_swimlane`` with the converter stubbed; return its argv."""
    monkeypatch.setattr(_runner.importlib.util, "find_spec", lambda name: object())
    argv: list[list[str]] = []
    monkeypatch.setattr(_runner.subprocess, "run", lambda cmd, check: argv.append(cmd))
    records = swimlane_dir / "chip_swimlane_records.json"
    records.write_text("{}", encoding="utf-8")
    _generate_swimlane(work_dir, swimlane_dir, records)
    assert len(argv) == 1
    return argv[0]


def test_generate_swimlane_passes_kernel_config_when_present(tmp_path, monkeypatch):
    # The usual case: ``work_dir`` owns the kernels being converted, so its
    # config is the converter's ``-k`` label fallback.
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "kernel_config.py").write_text("KERNELS = []\n", encoding="utf-8")

    cmd = _converter_argv(monkeypatch, work_dir, tmp_path)

    assert "-k" in cmd
    assert str(work_dir / "kernel_config.py") in cmd


def test_generate_swimlane_omits_missing_kernel_config(tmp_path, monkeypatch):
    # No config to point at — e.g. an L3 dispatch whose owning program could not
    # be resolved. The converter *errors out* on a ``-k`` path that does not
    # exist, so passing it would lose the swimlane entirely instead of just its
    # kernel names.
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    cmd = _converter_argv(monkeypatch, work_dir, tmp_path)

    assert "-k" not in cmd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
