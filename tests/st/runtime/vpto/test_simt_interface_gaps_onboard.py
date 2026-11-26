# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT interface acceptance on A5, including scalar types and bit-level boundaries."""

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

pytestmark = pytest.mark.platforms("a5", reason="Native SIMT requires A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def interface_kernel(
    state: pl.InOut[pl.Tensor[[6, 32], pl.INT32]],
    values: pl.Tensor[[1, 32], pl.INT32],
    old: pl.InOut[pl.Tensor[[12, 32], pl.INT32]],
    copied: pl.Out[pl.Tensor[[6, 32], pl.INT32]],
    output: pl.InOut[pl.Tensor[[12, 32], pl.INT32]],
):
    tile = pl.load(state, (0, 0), (6, 32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        value = pl.simt.load(values, (0, tid))
        previous = pl.simt.atomic_sub(state, (0, tid), value)
        pl.simt.store(old, (0, tid), previous)
        previous = pl.simt.atomic_and(state, (1, tid), value)
        pl.simt.store(old, (1, tid), previous)
        previous = pl.simt.atomic_or(state, (2, tid), value)
        pl.simt.store(old, (2, tid), previous)
        previous = pl.simt.atomic_xor(state, (3, tid), value)
        pl.simt.store(old, (3, tid), previous)
        previous = pl.simt.atomic_min(state, (4, tid), value, unsigned=True)
        pl.simt.store(old, (4, tid), previous)
        previous = pl.simt.atomic_max(state, (5, tid), value, unsigned=True)
        pl.simt.store(old, (5, tid), previous)
        previous = pl.simt.atomic_sub(tile, (0, tid), value)
        pl.simt.store(old, (6, tid), previous)
        previous = pl.simt.atomic_and(tile, (1, tid), value)
        pl.simt.store(old, (7, tid), previous)
        previous = pl.simt.atomic_or(tile, (2, tid), value)
        pl.simt.store(old, (8, tid), previous)
        previous = pl.simt.atomic_xor(tile, (3, tid), value)
        pl.simt.store(old, (9, tid), previous)
        previous = pl.simt.atomic_min(tile, (4, tid), value, unsigned=True)
        pl.simt.store(old, (10, tid), previous)
        previous = pl.simt.atomic_max(tile, (5, tid), value, unsigned=True)
        pl.simt.store(old, (11, tid), previous)
        pl.simt.store(output, (0, tid), pl.simt.lanemask_eq())
        pl.simt.store(output, (1, tid), pl.simt.lanemask_le())
        pl.simt.store(output, (2, tid), pl.simt.lanemask_lt())
        pl.simt.store(output, (3, tid), pl.simt.lanemask_ge())
        pl.simt.store(output, (4, tid), pl.simt.lanemask_gt())
        pl.simt.store(output, (5, tid), pl.simt.shuffle_idx(value, 5, width=16, bound=3))
        pl.simt.store(output, (6, tid), pl.simt.shuffle_up(value, 1, width=16, bound=3))
        pl.simt.store(output, (7, tid), pl.simt.shuffle_down(value, 1, width=16, bound=11))
        pl.simt.store(output, (8, tid), pl.simt.shuffle_bfly(value, 1, width=16, bound=11))
        pl.simt.store(output, (9, tid), pl.simt.redux_min(value, unsigned=True))
        pl.simt.store(output, (10, tid), pl.simt.redux_max(value, unsigned=True))
        floating = pl.simt.bitcast(value, pl.FP32)
        bits = pl.simt.bitcast(floating, pl.INT32)
        pl.simt.store(output, (11, tid), bits)
    pl.store(tile, (0, 0), copied)
    return state, old, copied, output


@pl.jit
def interface_entry(
    state: pl.InOut[pl.Tensor[[6, 32], pl.INT32]],
    values: pl.Tensor[[1, 32], pl.INT32],
    old: pl.InOut[pl.Tensor[[12, 32], pl.INT32]],
    copied: pl.Out[pl.Tensor[[6, 32], pl.INT32]],
    output: pl.InOut[pl.Tensor[[12, 32], pl.INT32]],
):
    return interface_kernel(state, values, old, copied, output)


@pl.jit.aiv
def subtraction_kernel(
    state: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    values: pl.Tensor[[1, 32], pl.FP32],
    old: pl.InOut[pl.Tensor[[2, 32], pl.FP32]],
    copied: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
):
    tile = pl.load(state, (0, 0), (1, 32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        value = pl.simt.load(values, (0, tid))
        previous = pl.simt.atomic_sub(state, (0, tid), value, l2cache="nmfv")
        pl.simt.store(old, (0, tid), previous)
        previous = pl.simt.atomic_sub(tile, (0, tid), value)
        pl.simt.store(old, (1, tid), previous)
        # Unused old values must not remove the second update.
        pl.simt.atomic_sub(state, (0, tid), value)
        pl.simt.atomic_sub(tile, (0, tid), value)
    pl.store(tile, (0, 0), copied)
    return state, old, copied


@pl.jit
def subtraction_entry(
    state: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    values: pl.Tensor[[1, 32], pl.FP32],
    old: pl.InOut[pl.Tensor[[2, 32], pl.FP32]],
    copied: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
):
    return subtraction_kernel(state, values, old, copied)


@pl.jit.aiv
def extended_kernel(
    values: pl.Tensor[[1, 32], pl.FP32],
    output: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    half: pl.InOut[pl.Tensor[[1, 32], pl.FP16]],
    bfloat: pl.InOut[pl.Tensor[[1, 32], pl.BF16]],
    wide: pl.InOut[pl.Tensor[[1, 32], pl.INT64]],
    unsigned: pl.InOut[pl.Tensor[[1, 32], pl.UINT32]],
):
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        x = pl.simt.load(values, (0, tid))
        divisor = pl.const(3.0, pl.FP32)
        quotient = pl.simt.div(x, divisor, precision="precision_0ulp_ftz_true")
        remainder = pl.simt.rem(x, divisor)
        h = pl.simt.ftof(x, pl.FP16)
        b = pl.simt.ftof(x, pl.BF16)
        integer = pl.simt.ftoi(x, dtype=pl.INT64)
        pl.simt.store(half, (0, tid), h)
        pl.simt.store(bfloat, (0, tid), b)
        pl.simt.store(output, (0, tid), quotient)
        pl.simt.store(output, (1, tid), remainder)
        pl.simt.store(output, (2, tid), pl.simt.ftof(h, pl.FP32))
        pl.simt.store(output, (3, tid), pl.simt.ftof(b, pl.FP32))
        pl.simt.store(output, (4, tid), pl.simt.itof(integer))
        product = pl.simt.mul_i32toi64(tid - pl.const(16, pl.INT32), pl.const(2147483647, pl.INT32))
        pl.simt.store(wide, (0, tid), product)
        u = pl.simt.load(unsigned, (0, tid))
        shifted = u >> pl.const(1, pl.UINT32)
        pl.simt.atomic_max(unsigned, (0, tid), shifted)
        maximum = pl.simt.atomic_exch(unsigned, (0, tid), shifted)
        pl.simt.store(unsigned, (0, tid), maximum >> pl.const(1, pl.UINT32))
        pl.simt.assert_(pl.simt.veccore_id() >= 0)
    return output, half, bfloat, wide, unsigned


@pl.jit
def extended_entry(
    values: pl.Tensor[[1, 32], pl.FP32],
    output: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    half: pl.InOut[pl.Tensor[[1, 32], pl.FP16]],
    bfloat: pl.InOut[pl.Tensor[[1, 32], pl.BF16]],
    wide: pl.InOut[pl.Tensor[[1, 32], pl.INT64]],
    unsigned: pl.InOut[pl.Tensor[[1, 32], pl.UINT32]],
):
    return extended_kernel(values, output, half, bfloat, wide, unsigned)


@pl.jit.aiv
def remainder_kernel(
    lhs: pl.Tensor[[1, 32], pl.FP32],
    rhs: pl.Tensor[[1, 32], pl.FP32],
    output: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
):
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        x = pl.simt.load(lhs, (0, tid))
        y = pl.simt.load(rhs, (0, tid))
        pl.simt.store(output, (0, tid), pl.simt.rem(x, y))
    return output


@pl.jit
def remainder_entry(
    lhs: pl.Tensor[[1, 32], pl.FP32],
    rhs: pl.Tensor[[1, 32], pl.FP32],
    output: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
):
    return remainder_kernel(lhs, rhs, output)


def check_remainders(handle):
    """Divergent lanes cover exponent extremes, signed zero and invalid inputs."""
    # Use IEEE-754 bits for the oracle: CPU torch.fmod can produce NaN for
    # extreme finite ratios. Host flush-to-zero can also turn a subnormal
    # divisor into zero. Neither behavior is the FP32 fmod contract tested here.
    # Each row is (positive dividend bits, positive divisor bits, result bits).
    # None denotes NaN, whose payload/sign are intentionally unspecified.
    pairs = [
        (0, 0x40400000, 0),  # +0 / 3
        (0x40F00000, 0x40400000, 0x3FC00000),  # 7.5 / 3 = 1.5
        (0x3F800000, 0x40400000, 0x3F800000),  # |x| < |y|
        (0x40400000, 0x40400000, 0),  # equal magnitudes
        (0x7F7FFFFF, 0x40400000, 0),  # (2^24 - 1) * 2^104 is divisible by 3
        (0x7F7FFFFF, 3, 0),  # maximum normal / (3 * 2^-149)
        (0x800000, 3, 2),  # 2^23 % 3 = 2, in units of 2^-149
        (0x7FFFFF, 3, 1),  # (2^23 - 1) % 3 = 1
        (3, 2, 1),  # nonzero subnormal remainder
        (1, 0x800000, 1),  # minimum subnormal / normal
        (0x800001, 0x800000, 1),  # normal operands, subnormal remainder
        (0x3F800000, 0x7F800000, 0x3F800000),  # finite / infinity
        (0x7F800000, 0x3F800000, None),  # infinity / finite
        (0x3F800000, 0, None),  # zero divisor
        (0x7FC00001, 0x3F800000, None),  # NaN dividend
        (0x3F800000, 0x7FC00001, None),  # NaN divisor
    ]
    for divisor_sign in (0, 0x80000000):
        signed_pairs = [(x | sign, y | divisor_sign) for sign in (0, 0x80000000) for x, y, _ in pairs]
        lhs, rhs = (
            torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32).reshape(1, 32)
            for bits in zip(*signed_pairs, strict=True)
        )
        output = torch.zeros_like(lhs)
        handle(lhs, rhs, output)
        actual_bits = output.view(torch.int32).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        expected_bits = [
            None if result is None else result | sign for sign in (0, 0x80000000) for _, _, result in pairs
        ]
        for lane, (actual, expected) in enumerate(zip(actual_bits.tolist(), expected_bits, strict=True)):
            # Integer classification also avoids depending on the host's FP mode.
            matches = (actual & 0x7FFFFFFF) > 0x7F800000 if expected is None else actual == expected
            assert matches, (
                f"lane {lane}: rem({signed_pairs[lane][0]:#010x}, {signed_pairs[lane][1]:#010x}) "
                f"returned {actual:#010x}; expected {'NaN' if expected is None else hex(expected)}"
            )


def compile_case(directory: Path, *, entry=interface_entry):
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
    floating = compile_case(output / "floating", entry=subtraction_entry)
    extended = compile_case(output / "extended", entry=extended_entry)
    remainder = compile_case(output / "remainder", entry=remainder_entry)
    integer.load()
    floating.load()
    extended.load()
    remainder.load()
    checks = 0
    if device is not None:
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=integer.runtime_name) as worker:
            handle = worker.register(integer)
            float_handle = worker.register(floating)
            extended_handle = worker.register(extended)
            remainder_handle = worker.register(remainder)
            # Include bit 31, signed extrema, signed zero, infinities and NaN payloads.
            initial = torch.tensor(
                [-(1 << 31), (1 << 31) - 1, -1, 0, 1, 17, -17, 99], dtype=torch.int32
            ).repeat(4)
            values = torch.tensor(
                [0, -(1 << 31), 0x7FC00001, 0x7FC00002, 0x7F800000, -8388608, -1, 1], dtype=torch.int32
            ).repeat(4)
            state = initial.repeat(6, 1)
            old = torch.zeros((12, 32), dtype=torch.int32)
            copied = state.clone()
            observed = torch.zeros((12, 32), dtype=torch.int32)
            handle(state, values.reshape(1, 32), old, copied, observed)
            unsigned_initial = initial.to(torch.int64) & 0xFFFFFFFF
            unsigned_values = values.to(torch.int64) & 0xFFFFFFFF
            expected = torch.stack(
                [
                    initial - values,
                    initial & values,
                    initial | values,
                    initial ^ values,
                    torch.minimum(unsigned_initial, unsigned_values).to(torch.int32),
                    torch.maximum(unsigned_initial, unsigned_values).to(torch.int32),
                ]
            )
            for actual in (state, copied):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(old, initial.repeat(12, 1), rtol=0, atol=0)
            expected_output = []
            for relation in ("eq", "le", "lt", "ge", "gt"):
                masks = []
                for lane in range(32):
                    selected = {
                        "eq": range(lane, lane + 1),
                        "le": range(lane + 1),
                        "lt": range(lane),
                        "ge": range(lane, 32),
                        "gt": range(lane + 1, 32),
                    }[relation]
                    masks.append(sum(1 << bit for bit in selected))
                expected_output.append(torch.tensor(masks, dtype=torch.int64).to(torch.int32))
            # Windows are subgroup-relative; out-of-window reads copy the caller.
            for mode, bound in (("idx", 3), ("up", 3), ("down", 11), ("bfly", 11)):
                row = []
                for lane in range(32):
                    local = lane % 16
                    source = {"idx": 5, "up": local - 1, "down": local + 1, "bfly": local ^ 1}[mode]
                    valid = 0 <= source < 16 and (source >= bound if mode == "up" else source <= bound)
                    row.append(values[lane - local + source if valid else lane])
                expected_output.append(torch.stack(row))
            for reduced in (unsigned_values.min(), unsigned_values.max()):
                expected_output.append(reduced.to(torch.int32).repeat(32))
            expected_output.append(values)
            torch.testing.assert_close(observed, torch.stack(expected_output), rtol=0, atol=0)
            # The old-value result is captured before both subtractions.
            initial_f = torch.arange(32, dtype=torch.float32).reshape(1, 32) - 16
            updates_f = torch.full((1, 32), 0.5, dtype=torch.float32)
            state_f, copied_f = initial_f.clone(), initial_f.clone()
            old_f = torch.zeros((2, 32), dtype=torch.float32)
            float_handle(state_f, updates_f, old_f, copied_f)
            torch.testing.assert_close(old_f, initial_f.repeat(2, 1), rtol=0, atol=0)
            for actual in (state_f, copied_f):
                torch.testing.assert_close(actual, initial_f - 2 * updates_f, rtol=0, atol=0)
            inputs = torch.tensor([-7.5, -1.5, -0.0, 0.0, 1.5, 7.5, 0.1, 32752.0]).repeat(4).reshape(1, 32)
            extended_output = torch.zeros((5, 32), dtype=torch.float32)
            half = torch.zeros((1, 32), dtype=torch.float16)
            bfloat = torch.zeros((1, 32), dtype=torch.bfloat16)
            wide = torch.zeros((1, 32), dtype=torch.int64)
            unsigned_seed = (
                torch.tensor([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 17, 99, 256], dtype=torch.int64)
                .repeat(4)
                .reshape(1, 32)
            )
            unsigned = unsigned_seed.to(torch.uint32)
            extended_handle(inputs, extended_output, half, bfloat, wide, unsigned)
            expected_extended = torch.cat(
                [
                    inputs / 3,
                    torch.fmod(inputs, 3),
                    inputs.half().float(),
                    inputs.bfloat16().float(),
                    torch.trunc(inputs),
                ]
            )
            torch.testing.assert_close(extended_output, expected_extended, rtol=0, atol=0)
            torch.testing.assert_close(half, inputs.half(), rtol=0, atol=0)
            torch.testing.assert_close(bfloat, inputs.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(
                wide, (torch.arange(32, dtype=torch.int64) - 16).reshape(1, 32) * 2147483647, rtol=0, atol=0
            )
            torch.testing.assert_close(unsigned.to(torch.int64), unsigned_seed >> 1, rtol=0, atol=0)
            check_remainders(remainder_handle)
            checks = 14
    (output / "result.json").write_text(json.dumps({"checks": checks}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_interface_gaps(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](
        tmp_path, "interface_gaps", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    )
    assert json.loads((tmp_path / "result.json").read_text())["checks"] == (14 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
