# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare SIMT numeric conversion with bit-preserving FP32 reinterpretation.

Convert finite FP32 inputs to FP16 (after explicit range clipping), BF16, and INT32
with truncation or nearest-even rounding. Also expose the original FP32 bits
and clear their sign bit to produce a magnitude, preserving the other bits.
Inputs include rounding ties, signed zero, and values beyond the FP16 range.

Run bare to compile only. On an allocated A5 device:
    python examples/advanced/09_simt_scalar_conversion.py --device 0 --valid 47
Only the first valid elements change; every output tail keeps its seed value.
"""

import argparse

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def conversion_kernel(
    values: pl.Tensor[[64], pl.FP32],
    half: pl.InOut[pl.Tensor[[64], pl.FP16]],
    bfloat: pl.InOut[pl.Tensor[[64], pl.BF16]],
    integers: pl.InOut[pl.Tensor[[2, 64], pl.INT32]],
    bits: pl.InOut[pl.Tensor[[64], pl.INT32]],
    magnitudes: pl.InOut[pl.Tensor[[64], pl.FP32]],
    valid: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < valid:
            value = pl.simt.load(values, (tid,))
            # Clip explicitly before conversion so finite overflow stays finite.
            bounded = pl.simt.min(pl.simt.max(value, pl.const(-65504.0, pl.FP32)), pl.const(65504.0, pl.FP32))
            pl.simt.store(half, (tid,), pl.simt.ftof(bounded, pl.FP16, rounding="r"))
            pl.simt.store(bfloat, (tid,), pl.simt.ftof(value, pl.BF16, rounding="r"))
            pl.simt.store(integers, (0, tid), pl.simt.ftoi(value, rounding="z"))
            pl.simt.store(integers, (1, tid), pl.simt.ftoi(value, rounding="r"))
            raw = pl.simt.bitcast(value, pl.INT32)
            pl.simt.store(bits, (tid,), raw)
            # The DSL parser lowers &, which has no Python Scalar overload.
            magnitude = pl.simt.bitcast(raw & pl.const(0x7FFFFFFF, pl.INT32), pl.FP32)  # pyright: ignore[reportOperatorIssue, reportUnknownArgumentType]
            pl.simt.store(magnitudes, (tid,), magnitude)
    return half, bfloat, integers, bits, magnitudes


@pl.jit
def convert_scalars(
    values: pl.Tensor[[64], pl.FP32],
    half: pl.InOut[pl.Tensor[[64], pl.FP16]],
    bfloat: pl.InOut[pl.Tensor[[64], pl.BF16]],
    integers: pl.InOut[pl.Tensor[[2, 64], pl.INT32]],
    bits: pl.InOut[pl.Tensor[[64], pl.INT32]],
    magnitudes: pl.InOut[pl.Tensor[[64], pl.FP32]],
    valid: pl.Scalar[pl.INT32],
):
    return conversion_kernel(values, half, bfloat, integers, bits, magnitudes, valid)


def make_inputs() -> tuple[torch.Tensor, ...]:
    """Create finite boundary inputs and initialized output tails on the host."""
    cases = torch.tensor(
        [-70000, -65504, -2.5, -1.5, -0.0, 0.0, 0.5, 1.5, 2.5, 1.00390625, 1.00048828125, 65504, 70000],
        dtype=torch.float32,
    )
    values = cases.repeat(5)[:64].clone()
    half = torch.full((64,), -13.0, dtype=torch.float16)
    bfloat = torch.full((64,), -13.0, dtype=torch.bfloat16)
    integers = torch.full((2, 64), -123456, dtype=torch.int32)
    bits = torch.full((64,), -123456, dtype=torch.int32)
    magnitudes = torch.full((64,), -13.0, dtype=torch.float32)
    return values, half, bfloat, integers, bits, magnitudes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Allocated A5 device; omit to compile only")
    parser.add_argument("--valid", type=int, default=47)
    args = parser.parse_args()
    if not 0 <= args.valid <= 64:
        parser.error("--valid must be in [0, 64]")
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device if args.device is not None else 0,
    )
    compiled = convert_scalars.compile(config=config)
    if args.device is None:
        print("compiled; pass --device to execute on an allocated A5 device")
        return
    values, half, bfloat, integers, bits, magnitudes = make_inputs()
    compiled(values, half, bfloat, integers, bits, magnitudes, args.valid, config=config)
    print("FP32:", values[: args.valid])
    print("FP16:", half[: args.valid])
    print("BF16:", bfloat[: args.valid])
    print("Truncated:", integers[0, : args.valid])
    print("Nearest even:", integers[1, : args.valid])
    print("FP32 bits:", bits[: args.valid])
    print("Magnitudes:", magnitudes[: args.valid])


if __name__ == "__main__":
    main()
