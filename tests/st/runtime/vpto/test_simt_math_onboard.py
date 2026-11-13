# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar SIMT math, bit operations and conversion through public JIT and A5."""

import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig, runtime
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.platforms("a5", reason="SIMT math requires A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def math_kernel(
    src: pl.Tensor[[64], pl.FP32],
    bits: pl.Tensor[[64], pl.INT32],
    floats: pl.InOut[pl.Tensor[[25, 64], pl.FP32]],
    integers: pl.InOut[pl.Tensor[[16, 64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < valid:
            x = pl.simt.load(src, (tid,))
            b = pl.simt.load(bits, (tid,))
            pl.simt.store(floats, (0, tid), pl.simt.abs(x))
            pl.simt.store(floats, (1, tid), pl.simt.min(x, pl.const(0.0, pl.FP32)))
            pl.simt.store(floats, (2, tid), pl.simt.max(x, pl.const(0.0, pl.FP32)))
            pl.simt.store(floats, (3, tid), pl.simt.exp(x))
            pl.simt.store(floats, (4, tid), pl.simt.log(x))
            pl.simt.store(floats, (5, tid), pl.simt.sqrt(x))
            # PTOAS SoftLib reduces angles in FP32. Keep trig precision separate
            # from the large inputs used to test conversion saturation.
            angle = pl.cast(tid, pl.FP32) * pl.const(0.125, pl.FP32) - pl.const(4.0, pl.FP32)
            pl.simt.store(floats, (6, tid), pl.simt.sin(angle))
            pl.simt.store(floats, (7, tid), pl.simt.cos(angle))
            pl.simt.store(floats, (8, tid), pl.simt.ceil(x))
            pl.simt.store(floats, (9, tid), pl.simt.floor(x))
            pl.simt.store(floats, (10, tid), pl.simt.rint(x))
            pl.simt.store(floats, (11, tid), pl.simt.round(x))
            pl.simt.store(floats, (12, tid), pl.simt.pow(x, pl.const(2.0, pl.FP32)))
            pl.simt.store(floats, (13, tid), pl.simt.fma(x, pl.const(1.0000001192092896, pl.FP32), -x))
            pl.simt.store(integers, (0, tid), pl.simt.ftoi(x, rounding="r", unsigned=False))
            pl.simt.store(floats, (14, tid), pl.simt.itof(b, rounding="r", unsigned=False))
            pl.simt.store(integers, (1, tid), pl.simt.ftoi(x, rounding="a", unsigned=False))
            pl.simt.store(floats, (15, tid), pl.simt.itof(b, rounding="a", unsigned=False))
            pl.simt.store(integers, (2, tid), pl.simt.ftoi(x, rounding="f", unsigned=False))
            pl.simt.store(floats, (16, tid), pl.simt.itof(b, rounding="f", unsigned=False))
            pl.simt.store(integers, (3, tid), pl.simt.ftoi(x, rounding="c", unsigned=False))
            pl.simt.store(floats, (17, tid), pl.simt.itof(b, rounding="c", unsigned=False))
            pl.simt.store(integers, (4, tid), pl.simt.ftoi(x, rounding="z", unsigned=False))
            pl.simt.store(floats, (18, tid), pl.simt.itof(b, rounding="z", unsigned=False))
            pl.simt.store(integers, (5, tid), pl.simt.ftoi(x, rounding="r", unsigned=True))
            pl.simt.store(floats, (19, tid), pl.simt.itof(b, rounding="r", unsigned=True))
            pl.simt.store(integers, (6, tid), pl.simt.ftoi(x, rounding="a", unsigned=True))
            pl.simt.store(floats, (20, tid), pl.simt.itof(b, rounding="a", unsigned=True))
            pl.simt.store(integers, (7, tid), pl.simt.ftoi(x, rounding="f", unsigned=True))
            pl.simt.store(floats, (21, tid), pl.simt.itof(b, rounding="f", unsigned=True))
            pl.simt.store(integers, (8, tid), pl.simt.ftoi(x, rounding="c", unsigned=True))
            pl.simt.store(floats, (22, tid), pl.simt.itof(b, rounding="c", unsigned=True))
            pl.simt.store(integers, (9, tid), pl.simt.ftoi(x, rounding="z", unsigned=True))
            pl.simt.store(floats, (23, tid), pl.simt.itof(b, rounding="z", unsigned=True))
            # A custom activation composes new math with existing scalar arithmetic.
            sigmoid = pl.const(1.0, pl.FP32) / (pl.const(1.0, pl.FP32) + pl.simt.exp(-x))
            pl.simt.store(floats, (24, tid), sigmoid)
            pl.simt.store(
                integers,
                (10, tid),
                pl.simt.prmt(b, pl.const(0x76543210, pl.INT32), pl.const(0x5410, pl.INT32)),
            )
            pl.simt.store(integers, (11, tid), pl.simt.mulhi(b, pl.const(-3, pl.INT32)))
            pl.simt.store(integers, (12, tid), pl.simt.mulhi(b, pl.const(-3, pl.INT32), unsigned=True))
            # High selector bits are ignored, including each nibble's bit 3.
            pl.simt.store(
                integers,
                (15, tid),
                pl.simt.prmt(b, pl.const(-19088744, pl.INT32), pl.const(0x7654BA98, pl.INT32)),
            )
            # Existing signed floor division/modulo must remain unchanged.
            pl.simt.store(integers, (13, tid), pl.cast(b // 3, pl.INT32))
            pl.simt.store(integers, (14, tid), pl.cast(b % 3, pl.INT32))
    return floats, integers


@pl.jit
def math_entry(
    src: pl.Tensor[[64], pl.FP32],
    bits: pl.Tensor[[64], pl.INT32],
    floats: pl.InOut[pl.Tensor[[25, 64], pl.FP32]],
    integers: pl.InOut[pl.Tensor[[16, 64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
):
    return math_kernel(src, bits, floats, integers, valid)


def compile_math(directory: Path):
    return math_entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def signed32(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def rounded(value: float, mode: str) -> int:
    if mode == "r":
        return round(value)
    if mode == "a":
        return int(math.copysign(math.floor(abs(value) + 0.5), value))
    return {"f": math.floor, "c": math.ceil, "z": math.trunc}[mode](value)


def integer_to_float(value: int, mode: str) -> float:
    # Compute directed and tie-away rounding independently of device conversion.
    near = np.float32(value)
    if float(near) == value:
        return float(near)
    low = near if float(near) < value else np.nextafter(near, np.float32(-math.inf))
    high = near if float(near) > value else np.nextafter(near, np.float32(math.inf))
    if mode == "f" or (mode == "z" and value >= 0):
        return float(low)
    if mode == "c" or (mode == "z" and value < 0):
        return float(high)
    if mode == "a" and value - float(low) == float(high) - value:
        return float(high if value >= 0 else low)
    return float(near)


def check_results(src, bits, floats, integers, valid):
    x = src[:valid]
    expected_f = torch.full_like(floats, -777)
    expected_i = torch.full_like(integers, -777)
    zero = torch.zeros_like(x)
    angles = torch.arange(valid, dtype=torch.float32) * 0.125 - 4.0
    # Double precision makes the selected fma reference exact before FP32 rounding.
    values = [
        x.abs(),
        torch.fmin(x, zero),
        torch.fmax(x, zero),
        x.exp(),
        x.log(),
        x.sqrt(),
        angles.sin(),
        angles.cos(),
        x.ceil(),
        x.floor(),
        x.round(),
        torch.copysign((x.abs() + 0.5).floor(), x),
        x.square(),
        (x.double() * (1.0 + 2**-23) - x.double()).float(),
    ]
    for row, value in enumerate(values):
        expected_f[row, :valid] = value
    expected_f[24, :valid] = torch.sigmoid(x)
    for tid in range(valid):
        value, carrier = float(src[tid]), int(bits[tid])
        for unsigned in (False, True):
            low, high = (0, (1 << 32) - 1) if unsigned else (-(1 << 31), (1 << 31) - 1)
            for j, mode in enumerate("rafcz"):
                row = 5 * int(unsigned) + j
                # The device's saturated NaN conversion is checked separately from
                # finite rounding; do not let a host float->int cast define it.
                if math.isnan(value):
                    result = 0
                elif value <= low:
                    result = low
                elif value >= high:
                    result = high
                else:
                    result = min(high, max(low, rounded(value, mode)))
                expected_i[row, tid] = signed32(result)
                argument = carrier & 0xFFFFFFFF if unsigned else carrier
                expected_f[14 + row, tid] = integer_to_float(argument, mode)
        expected_i[10, tid] = signed32(((carrier & 0xFFFF) << 16) | 0x3210)
        expected_i[11, tid] = signed32((carrier * -3) >> 32)
        expected_i[12, tid] = signed32(((carrier & 0xFFFFFFFF) * 0xFFFFFFFD) >> 32)
        expected_i[13, tid] = carrier // 3
        expected_i[14, tid] = carrier % 3
        expected_i[15, tid] = -19088744
    torch.testing.assert_close(
        integers,
        expected_i,
        rtol=0,
        atol=0,
        msg=lambda message: message
        + f"\nvalid={valid}, integer mismatches (row, lane, actual, expected): "
        + str(
            [
                (r, t, int(integers[r, t]), int(expected_i[r, t]))
                for r, t in (integers != expected_i).nonzero().tolist()[:20]
            ]
        ),
    )
    torch.testing.assert_close(floats[:14], expected_f[:14], rtol=3e-5, atol=2e-6, equal_nan=True)
    # A loose transcendental tolerance must not hide a non-fused multiply/add
    # or an incorrect tie rule in the exact arithmetic rows.
    for row in (0, 8, 9, 10, 11, 13):
        torch.testing.assert_close(floats[row], expected_f[row], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(floats[14:24], expected_f[14:24], rtol=0, atol=0)
    torch.testing.assert_close(floats[24], expected_f[24], rtol=3e-5, atol=2e-6, equal_nan=True)


def acceptance(output: Path, device: int | None):
    compiled = compile_math(output)
    compiled.load()
    checks = 0
    if device is not None:
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            handle = worker.register(compiled)
            specials = [
                -math.inf,
                -2147483904.0,
                -2147483648.0,
                -16777216.0,
                -2.5,
                -1.5,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.5,
                2.5,
                1.0000001192092896,
                2147483520.0,
                2147483648.0,
                4294967296.0,
                math.inf,
                math.nan,
            ]
            src = torch.tensor(specials + [float(i - 20) / 8 for i in range(64 - len(specials))])
            bit_values = [-2147483648, 2147483647, -1, 0, 1, 16777217, -16777217, 0x12345678, -7, 7]
            bits = torch.tensor((bit_values * 7)[:64], dtype=torch.int32)
            for valid in (0, 1, 17, 31, 32, 33, 63, 64):
                floats = torch.full((25, 64), -777, dtype=torch.float32)
                integers = torch.full((16, 64), -777, dtype=torch.int32)
                handle(src, bits, floats, integers, valid)
                check_results(src, bits, floats, integers, valid)
                checks += 1
    (output / "result.json").write_text(json.dumps({"checks": checks}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_math(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](tmp_path, "math", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600")))
    assert json.loads((tmp_path / "result.json").read_text())["checks"] == (8 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
