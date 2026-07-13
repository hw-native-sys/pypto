# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the two-pass L2 swimlane execution protocol.

Enabling ``enable_l2_swimlane`` on an onboard platform captures the dep_gen task
graph (``deps.json``) in a subprocess, then runs a clean-timing swimlane pass
in-process — the two cannot share one process because the runtime leaks SVM
host-register mappings between DFX runs. These tests drive
:func:`pypto.runtime.runner._execute_dfx_passes` directly with recording stubs
(no device, no subprocess) and check :func:`pypto.runtime.runner._build_args_spec`.
"""

import ctypes

import pytest
import torch
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.runner import _build_args_spec, _DfxOpts, _execute_dfx_passes


def _drive(dfx: _DfxOpts, platform: str) -> tuple[list[_DfxOpts], int]:
    """Run the helper with stubs that record each in-process pass and capture.

    Returns ``(in_process_passes, capture_calls)``. ``_execute_dfx_passes``
    returns ``None`` (per-run timing is no longer a return value — simpler PR
    #1177); the protocol it drives is asserted via the recorded passes/captures.
    """
    seen: list[_DfxOpts] = []
    captures = {"n": 0}

    def run_pass(pass_dfx: _DfxOpts) -> None:
        seen.append(pass_dfx)

    def capture_deps() -> None:
        captures["n"] += 1

    assert _execute_dfx_passes(run_pass, capture_deps, dfx, platform) is None
    return seen, captures["n"]


def test_onboard_swimlane_captures_deps_then_times_in_process():
    seen, captures = _drive(_DfxOpts(enable_l2_swimlane=True), "a2a3")
    # deps captured once (subprocess), one in-process timing pass.
    assert captures == 1
    assert len(seen) == 1
    timing = seen[0]
    assert timing.enable_l2_swimlane is True
    assert timing.enable_dep_gen is False  # dep_gen forced off on the timing pass


def test_onboard_swimlane_with_explicit_dep_gen_still_one_capture():
    # An explicit --enable-dep-gen alongside swimlane must NOT add an in-process
    # dep_gen run: the subprocess capture already produced deps.json.
    seen, captures = _drive(_DfxOpts(enable_l2_swimlane=True, enable_dep_gen=True), "a2a3")
    assert captures == 1
    assert len(seen) == 1
    assert seen[0].enable_l2_swimlane is True and seen[0].enable_dep_gen is False


def test_onboard_swimlane_timing_dfx_ride_the_in_process_pass():
    # PMU / args-dump / scope-stats are timing-sensitive: they ride the clean
    # in-process timing pass (the subprocess capture is dep_gen-only).
    dfx = _DfxOpts(
        enable_l2_swimlane=True,
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
    seen, captures = _drive(_DfxOpts(enable_dep_gen=True), "a2a3")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0].enable_dep_gen is True
    assert seen[0].enable_l2_swimlane is False


def test_no_dfx_is_single_pass_no_capture():
    seen, captures = _drive(_DfxOpts(), "a2a3")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0] == _DfxOpts()


def test_sim_swimlane_stays_single_pass_no_capture():
    # Simulator skips swimlane conversion anyway, so no capture / second run.
    seen, captures = _drive(_DfxOpts(enable_l2_swimlane=True), "a2a3sim")
    assert captures == 0
    assert len(seen) == 1
    assert seen[0].enable_l2_swimlane is True


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
