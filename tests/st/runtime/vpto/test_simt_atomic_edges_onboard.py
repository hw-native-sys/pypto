# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Boundary-value atomic validation, including native floating CAS comparison."""

import json
import os
import runpy
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig, runtime
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.platforms("a5", reason="SIMT atomics require A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def integer_kernel(
    state: pl.InOut[pl.Tensor[[4, 32], pl.INT32]],
    values: pl.Tensor[[2, 32], pl.INT32],
    old: pl.InOut[pl.Tensor[[8, 32], pl.INT32]],
    copied: pl.Out[pl.Tensor[[4, 32], pl.INT32]],
):
    tile = pl.load(state, (0, 0), (4, 32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        update = pl.simt.load(values, (0, tid))
        compare = pl.simt.load(values, (1, tid))
        previous = pl.simt.atomic_add(state, (0, tid), update)
        pl.simt.store(old, (0, tid), previous)
        previous = pl.simt.atomic_min(state, (1, tid), update)
        pl.simt.store(old, (1, tid), previous)
        previous = pl.simt.atomic_max(state, (2, tid), update)
        pl.simt.store(old, (2, tid), previous)
        previous = pl.simt.atomic_cas(state, (3, tid), compare, update)
        pl.simt.store(old, (3, tid), previous)
        previous = pl.simt.atomic_add(tile, (0, tid), update)
        pl.simt.store(old, (4, tid), previous)
        previous = pl.simt.atomic_min(tile, (1, tid), update)
        pl.simt.store(old, (5, tid), previous)
        previous = pl.simt.atomic_max(tile, (2, tid), update)
        pl.simt.store(old, (6, tid), previous)
        previous = pl.simt.atomic_cas(tile, (3, tid), compare, update)
        pl.simt.store(old, (7, tid), previous)
    pl.store(tile, (0, 0), copied)
    return state, old, copied


@pl.jit
def integer_entry(
    state: pl.InOut[pl.Tensor[[4, 32], pl.INT32]],
    values: pl.Tensor[[2, 32], pl.INT32],
    old: pl.InOut[pl.Tensor[[8, 32], pl.INT32]],
    copied: pl.Out[pl.Tensor[[4, 32], pl.INT32]],
):
    return integer_kernel(state, values, old, copied)


@pl.jit.aiv
def floating_kernel(
    state: pl.InOut[pl.Tensor[[4, 32], pl.FP32]],
    values: pl.Tensor[[2, 32], pl.FP32],
    old: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    copied: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
):
    tile = pl.load(state, (0, 0), (4, 32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        update = pl.simt.load(values, (0, tid))
        compare = pl.simt.load(values, (1, tid))
        previous = pl.simt.atomic_add(state, (0, tid), update)
        pl.simt.store(old, (0, tid), previous)
        previous = pl.simt.atomic_min(state, (1, tid), update)
        pl.simt.store(old, (1, tid), previous)
        previous = pl.simt.atomic_max(state, (2, tid), update)
        pl.simt.store(old, (2, tid), previous)
        previous = pl.simt.atomic_cas(state, (3, tid), compare, update)
        pl.simt.store(old, (3, tid), previous)
        previous = pl.simt.atomic_add(tile, (0, tid), update)
        pl.simt.store(old, (4, tid), previous)
        previous = pl.simt.atomic_min(tile, (1, tid), update)
        pl.simt.store(old, (5, tid), previous)
        previous = pl.simt.atomic_max(tile, (2, tid), update)
        pl.simt.store(old, (6, tid), previous)
        previous = pl.simt.atomic_cas(tile, (3, tid), compare, update)
        pl.simt.store(old, (7, tid), previous)
    pl.store(tile, (0, 0), copied)
    return state, old, copied


@pl.jit
def floating_entry(
    state: pl.InOut[pl.Tensor[[4, 32], pl.FP32]],
    values: pl.Tensor[[2, 32], pl.FP32],
    old: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    copied: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
):
    return floating_kernel(state, values, old, copied)


def compile_case(directory: Path, *, entry=integer_entry):
    return entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def acceptance(output: Path, device: int | None):
    integer = compile_case(output / "integer")
    floating = compile_case(output / "floating", entry=floating_entry)
    integer.load()
    floating.load()
    if device is None:
        (output / "result.json").write_text(json.dumps({"checks": 0}))
        return
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    with runtime.ChipWorker(config=config, runtime=integer.runtime_name) as worker:
        int_handle, float_handle = worker.register(integer), worker.register(floating)
        initial = torch.tensor([-(2**31), 2**31 - 1, -1, 0, 1, 100, -100, 17], dtype=torch.int32).repeat(4)
        updates = torch.tensor([-1, 1, 1, -1, -(2**31), 2**31 - 1, -200, 19], dtype=torch.int32).repeat(4)
        compares = initial.clone()
        compares[1::2] = 1234
        values = torch.stack([updates, compares])
        state = initial.repeat(4, 1)
        old = torch.empty((8, 32), dtype=torch.int32)
        copied = state.clone()
        int_handle(state, values, old, copied)
        expected = torch.stack(
            [
                initial + updates,
                torch.minimum(initial, updates),
                torch.maximum(initial, updates),
                torch.where(initial == compares, updates, initial),
            ]
        )
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        torch.testing.assert_close(old, initial.repeat(8, 1), rtol=0, atol=0)
        # Use exact payloads to distinguish bitwise CAS from numeric equality,
        # and to record the target's NaN / signed-zero min/max behavior.
        initial_bits = torch.tensor(
            [0, -2147483648, 0x7FC00001, 0x7FC00002, 0x7F800000, -8388608, 0x3F800000, -1082130432],
            dtype=torch.int32,
        ).repeat(4)
        initial_f = initial_bits.view(torch.float32)
        updates_f = torch.tensor(
            [-0.0, 0.0, 2.0, float("nan"), -float("inf"), float("inf"), float("nan"), 2.0]
        ).repeat(4)
        compares_bits = initial_bits.clone()
        compares_bits[0::8] = -2147483648
        compares_bits[1::8] = 0
        compares_bits[3::8] = 0x7FC00001
        state_f = initial_f.repeat(4, 1)
        old_f = torch.empty((8, 32), dtype=torch.float32)
        copied_f = state_f.clone()
        float_handle(state_f, torch.stack([updates_f, compares_bits.view(torch.float32)]), old_f, copied_f)
        torch.testing.assert_close(old_f.view(torch.int32), initial_bits.repeat(8, 1), rtol=0, atol=0)
        torch.testing.assert_close(state_f.view(torch.int32), copied_f.view(torch.int32), rtol=0, atol=0)
        # A5 CAS compares representations: opposite signed zeros do not match,
        # equal NaN payloads match, and distinct NaN payloads do not match.
        expected_cas = torch.where(initial_bits == compares_bits, updates_f.view(torch.int32), initial_bits)
        torch.testing.assert_close(state_f[3].view(torch.int32), expected_cas, rtol=0, atol=0)
        # Add may canonicalize NaN payloads; only the numerical result matters.
        torch.testing.assert_close(state_f[0], initial_f + updates_f, rtol=0, atol=0, equal_nan=True)
        expected_min = torch.tensor(
            [-0.0, -0.0, 2.0, float("nan"), -float("inf"), -float("inf"), 1.0, -1.0]
        ).repeat(4)
        expected_max = torch.tensor(
            [0.0, 0.0, 2.0, float("nan"), float("inf"), float("inf"), 1.0, 2.0]
        ).repeat(4)
        for actual, expected_f in [(state_f[1], expected_min), (state_f[2], expected_max)]:
            torch.testing.assert_close(actual, expected_f, rtol=0, atol=0, equal_nan=True)
            zero = expected_f == 0
            torch.testing.assert_close(actual[zero].view(torch.int32), expected_f[zero].view(torch.int32))
        report = {"checks": 9, "floating_bits": state_f.view(torch.int32)[:, :8].tolist()}
    (output / "result.json").write_text(json.dumps(report))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_atomic_edges(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](
        tmp_path, "atomic_edges", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    )
    assert json.loads((tmp_path / "result.json").read_text())["checks"] == (9 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
