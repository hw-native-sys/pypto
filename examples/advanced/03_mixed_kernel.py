# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Mixed kernel: cube (AIC) and vector (AIV) work inside one scope.

An Ascend core group pairs a cube unit with vector units. A matmul runs on the
cube; the bias-add that follows runs on the vector units. A *mixed* kernel puts
both in one scope and adds ``pl.split(...)``, which shards the vector work across
two lanes and inserts cube/vector transfers. This example uses a one-slot ring
to fit safely; it demonstrates mixed dataflow, not a double-buffering speedup.

Kernels:
  staged_matmul_bias    — TWO scopes: cube finishes, then vector starts
  mixed_matmul_bias     — ONE scope + pl.split(UP_DOWN)
  mixed_matmul_bias_lr  — ONE scope + pl.split(LEFT_RIGHT)

Concepts introduced:
  - pl.split(pl.SplitMode.UP_DOWN) as a pl.at optimization
  - One scope instead of two dispatches for a cube-then-vector chain
  - UP_DOWN (row axis) vs LEFT_RIGHT (column axis)
  - pl.cross_core_slot(slot_num=) — retuning the cube/vector ring depth

Run:  python examples/advanced/03_mixed_kernel.py
      python examples/advanced/03_mixed_kernel.py --mode staged
      python examples/advanced/03_mixed_kernel.py --mode left_right
Next: examples/models/qwen3_jit/kernels/projection.py — a mixed kernel in a real model

Legacy Python API example: platform= calls below are retained for compatibility
and emit DeprecationWarning. New Python code should use backend_type and
execution_mode; see docs/en/dev/08-entry-points.md#legacy-platform-examples.
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

M = 128
K = 256
N = 128


@pl.jit
def staged_matmul_bias(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b + bias`` as TWO scopes — the form the mixed kernel replaces.

    The cube work and the vector work sit in separate scopes, so the vector units
    have nothing to do until the matmul scope has finished.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cube_only"):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vector_only"):
        out[:] = pl.add(acc, bias)
    return out


@pl.jit
def mixed_matmul_bias(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b + bias`` as ONE scope, split along the row axis.

    ``pl.split(pl.SplitMode.UP_DOWN)`` marks the scope as mixed: the compiler
    shards the vector work by rows and inserts the cube-to-vector transfer.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        # The crossing [128,128] FP32 tile needs 64 KiB per ring slot. Each
        # vector lane also needs separate 32 KiB bias and output buffers: the
        # Ascend910 load/TPOP safety guard forbids their in-place reuse. Two
        # slots would need 192 KiB total, above the safe 184 KiB Vec budget.
        # One slot keeps this example at 128 KiB; it is not double buffered.
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)],
        name_hint="mixed_up_down",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)  # cube (AIC)
        out[:] = pl.add(acc, bias)  # vector (AIV)
    return out


@pl.jit
def mixed_matmul_bias_lr(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Same as :func:`mixed_matmul_bias`, halved along the column axis."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        # The column split has the same per-lane footprint as the row split.
        optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT), pl.cross_core_slot(slot_num=1)],
        name_hint="mixed_left_right",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)
        out[:] = pl.add(acc, bias)
    return out


_MODES = {
    "mixed": mixed_matmul_bias,
    "staged": staged_matmul_bias,
    "left_right": mixed_matmul_bias_lr,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed cube/vector kernel example.")
    parser.add_argument("--mode", choices=sorted(_MODES), default="mixed")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    bias = torch.randn(M, N, dtype=torch.float32)
    out = torch.zeros(M, N, dtype=torch.float32)

    _MODES[args.mode](a, b, bias, out, config=RunConfig(platform=args.platform))

    expected = a.float() @ b.float() + bias
    assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
        f"{args.mode}: max diff = {(out - expected).abs().max().item()}"
    )
    print("OK")
