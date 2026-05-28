# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO DSL program for L3 GEMM + 4-phase peer-to-peer allreduce.

Parallel model: shard ``A[r]``, replicate ``B``.

Per rank (two stages in ``chip_orch``):
  1. Local GEMM:    ``P_r = A[r] @ B``.
  2. 4-phase allreduce: reuses ``make_reduce_step_2rank`` from ``test_l3_allreduce``.

Golden: ``outputs[r] == sum_s (A[s] @ B)`` for every rank ``r``.

Two ranks only (single peer read in the reduce kernel).
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from tests.st.distributed.l3_gemm import M0, K, N, make_cube_gemm
from tests.st.distributed.test_l3_allreduce import make_reduce_step_2rank

__all__ = ["M0", "K", "N", "build_l3_allreduce_gemm_program"]


def build_l3_allreduce_gemm_program(*, nranks: int, m0: int = M0, k: int = K, n: int = N):
    """GEMM partial then 4-phase allreduce on window (2-rank peer pattern)."""

    if nranks != 2:
        raise ValueError(
            f"build_l3_allreduce_gemm_program currently requires nranks=2 (single peer read); got {nranks}"
        )

    # Window alloc is raw nbytes. FP32 == 4 bytes/element (pl.FP32); update if dtype changes.
    data_window_nbytes = m0 * n * 4
    # Single INT32 handshake cell (pl.INT32 == 4 bytes). No extra alignment beyond runtime default.
    signal_window_nbytes = 4

    a_shape = [nranks, m0, k]
    partial_shape = [nranks, m0, n]
    out_shape = [nranks, m0, n]

    gemm_fn = make_cube_gemm(m0=m0, k=k, n=n)
    reduce_step_fn = make_reduce_step_2rank(m0=m0, n=n)

    @pl.program
    class L3AllReduceGemmProgram:
        gemm = gemm_fn
        reduce_step = reduce_step_fn

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a_shard: pl.Tensor[[m0, k], pl.FP32],
            b: pl.Tensor[[k, n], pl.FP32],
            partial: pl.InOut[pl.Tensor[[m0, n], pl.FP32]],
            out: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[m0, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[m0, n], pl.FP32]:
            partial_out: pl.Tensor[[m0, n], pl.FP32] = self.gemm(a_shard, b, partial)
            return self.reduce_step(partial_out, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[a_shape, pl.FP32],  # type: ignore[valid-type]
            b: pl.Tensor[[k, n], pl.FP32],
            partials: pl.InOut[pl.Tensor[partial_shape, pl.FP32]],  # type: ignore[valid-type]
            outputs: pl.Out[pl.Tensor[out_shape, pl.FP32]],  # type: ignore[valid-type]
        ) -> pl.Tensor[out_shape, pl.FP32]:  # type: ignore[valid-type]
            data_buf = pld.alloc_window_buffer(data_window_nbytes)
            signal_buf = pld.alloc_window_buffer(signal_window_nbytes)

            data = pld.window(data_buf, [m0, n], dtype=pl.FP32)
            signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
            self.chip_orch(a[0], b, partials[0], outputs[0], data, signal, 1, device=0)
            self.chip_orch(a[1], b, partials[1], outputs[1], data, signal, 0, device=1)

            return outputs

    return L3AllReduceGemmProgram
