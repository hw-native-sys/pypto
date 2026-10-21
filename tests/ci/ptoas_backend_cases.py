# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small JIT corpus shared by the PTOAS backend audit and device CI."""

import pypto.language as pl
import torch


@pl.jit
def elementwise(
    a: pl.Tensor[[32, 64], pl.FP32],
    b: pl.Tensor[[32, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        av = pl.load(a, [0, 0], [32, 64])
        bv = pl.load(b, [0, 0], [32, 64])
        pl.store(pl.mul(pl.add(av, bv), 0.5), [0, 0], out)
    return out


@pl.jit
def reduction(a: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 1], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        av = pl.load(a, [0, 0], [32, 64])
        tmp = pl.create_tile([32, 64], dtype=pl.FP32)
        pl.store(pl.row_sum(av, tmp_tile=tmp), [0, 0], out)
    return out


@pl.jit
def loop_carry(a: pl.Tensor[[128, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [32, 64])
        for i in pl.range(1, 4):
            av = pl.load(a, [i * 32, 0], [32, 64])
            acc = pl.add(acc, av)
        pl.store(acc, [0, 0], out)
    return out


@pl.jit
def valid_shape(a: pl.Tensor[[17, 48], pl.FP32], out: pl.Out[pl.Tensor[[17, 48], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [32, 64], valid_shape=[17, 48])
        pl.store(pl.add(tile, 2.0), [0, 0], out)
    return out


@pl.jit
def matmul(
    a: pl.Tensor[[32, 64], pl.FP16],
    b: pl.Tensor[[64, 32], pl.FP16],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.slice(a, [32, 64], [0, 0])
        rhs = pl.slice(b, [64, 32], [0, 0])
        out = pl.assemble(out, pl.matmul(lhs, rhs, out_dtype=pl.FP32), [0, 0])
    return out


def cases():
    """Return fresh, deterministic inputs and independent CPU goldens."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(32, 64, generator=generator)
    b = torch.randn(32, 64, generator=generator)
    blocks = torch.randn(128, 64, generator=generator)
    edge = torch.randn(17, 48, generator=generator)
    lhs = torch.randn(32, 64, generator=generator, dtype=torch.float16)
    rhs = torch.randn(64, 32, generator=generator, dtype=torch.float16)
    return [
        (elementwise, [a, b], (a + b) * 0.5),
        (reduction, [a], a.sum(dim=1, keepdim=True)),
        (loop_carry, [blocks], blocks.reshape(4, 32, 64).sum(dim=0)),
        (valid_shape, [edge], edge + 2.0),
        (matmul, [lhs, rhs], lhs.float() @ rhs.float()),
    ]
