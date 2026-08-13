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
cube; the bias-add that follows runs on the vector units. Written as two separate
scopes, the vector units idle while the cube works and vice versa. A *mixed*
kernel puts both in one scope and adds ``pl.split(...)``, which lets the compiler
halve the work so the two unit types overlap.

Kernels:
  staged_matmul_bias    — TWO scopes: cube finishes, then vector starts
  mixed_matmul_bias     — ONE scope + pl.split(UP_DOWN)
  mixed_matmul_bias_lr  — ONE scope + pl.split(LEFT_RIGHT)

Concepts introduced:
  - pl.split(pl.SplitMode.UP_DOWN) as a pl.at optimization
  - Why one scope beats two for a cube-then-vector chain
  - UP_DOWN (row axis) vs LEFT_RIGHT (column axis)
  - pl.cross_core_slot(slot_num=) — sizing the cube/vector ring

Run:  python examples/advanced/03_mixed_kernel.py
      python examples/advanced/03_mixed_kernel.py --mode staged
      python examples/advanced/03_mixed_kernel.py --mode left_right
Next: examples/models/qwen3_jit/kernels/projection.py — a mixed kernel in a real model
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
        out = pl.assemble(out, pl.add(acc, bias), [0, 0])
    return out


@pl.jit
def mixed_matmul_bias(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b + bias`` as ONE scope, split along the row axis.

    ``pl.split(pl.SplitMode.UP_DOWN)`` marks the scope as mixed: the compiler
    halves it by rows and drives the cube and vector units concurrently.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        # The [128,128] FP32 tile that crosses the cube/vector boundary is 64KB.
        # The C2V ring defaults to 8 slots of it (512KB) — far past the vector
        # buffer limit, so the depth has to come down with it.
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=2)],
        name_hint="mixed_up_down",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)  # cube (AIC)
        out = pl.assemble(out, pl.add(acc, bias), [0, 0])  # vector (AIV)
    return out


@pl.jit
def mixed_matmul_bias_lr(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Same as :func:`mixed_matmul_bias`, halved along the column axis."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT), pl.cross_core_slot(slot_num=2)],
        name_hint="mixed_left_right",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)
        out = pl.assemble(out, pl.add(acc, bias), [0, 0])
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
