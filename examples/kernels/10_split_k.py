# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Split-K matrix multiplication.

Split-K is a matmul optimisation for a large K (reduction) dimension with
small M and N. With small M/N there are few independent output tiles, so the
standard "one output tile per core" mapping leaves most cores idle. Split-K
also parallelises the K dimension: each core computes a partial product over
a slice of K, and the partials are summed into the shared output.

Kernel:
  matmul_split_k -- [M, K] @ [K, N] with K split across SPLIT parallel cores

Concepts introduced:
  - pl.parallel over the K dimension (one core per K-slice)
  - pl.assemble(..., atomic=pl.AtomicType.Add) -- atomic-add accumulation of
    each core's partial product into one global-memory output
  - in-kernel zero-initialisation of the output before the atomic-add loop

NOTE: atomic-add accumulation order across cores is not fixed, so the
floating-point result is non-deterministic at the ulp level.

Run:  python examples/kernels/10_split_k.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

M = 64
N = 64
K = 512
SPLIT = 4  # number of cores the K reduction is spread across
KS = K // SPLIT  # per-core K-slice width


@pl.jit
def matmul_split_k(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Split-K matmul: c = a @ b, with K split across SPLIT parallel cores.

    Each core multiplies an [M, KS] x [KS, N] slice and atomically adds its
    partial product into ``c``. ``c`` is zero-initialised in-kernel first so
    the accumulation starts from zero; the zero-init is sequenced before the
    parallel loop because its result feeds the loop.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
        c = pl.assemble(c, pl.full([M, N], dtype=pl.FP32, value=0.0), [0, 0])
    for ks in pl.parallel(0, SPLIT):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
            k0 = ks * KS
            a_k = a[:, k0 : k0 + KS]
            b_k = b[k0 : k0 + KS, :]
            partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
            c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
    return c


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    c = torch.zeros((M, N), dtype=torch.float32)
    matmul_split_k(a, b, c, config=cfg)
    assert torch.allclose(c, torch.matmul(a, b), rtol=1e-3, atol=1e-3)

    print("OK")
