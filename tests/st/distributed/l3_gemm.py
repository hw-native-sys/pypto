# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO DSL program for L3 data-parallel GEMM (no collectives).

Parallel model:
  - Shard ``A`` along rows: rank ``r`` holds ``A[r]`` with shape ``[M0, K]``.
  - Replicate ``B`` on every rank: shape ``[K, N]``.

Golden (per rank): ``C[r] = A[r] @ B``.

Multi-rank dispatch contract:
  - ``nranks >= 2``: the host orchestrator allocates a comm window via
    ``pld.alloc_window_buffer`` and passes a ``pld.window`` view to each
    ``chip_orch`` dispatch under a constant or loop-induction ``device=``.
    The GEMM kernel has no collective, so it carries an unused INT32
    ``_scratch`` window.
  - ``nranks == 1``: a single ``device=0`` call, no comm window.
  - The two branches use distinct class names (``L3GemmProgramP1`` and
    ``L3GemmProgramPN``).
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

M0 = 64
K = 64
N = 64

__all__ = ["M0", "K", "N", "make_cube_gemm", "build_l3_gemm_program"]


def make_cube_gemm(*, m0: int, k: int, n: int):
    """Return an InCore cube GEMM method: ``partial = a_shard @ b``."""

    @pl.function(type=pl.FunctionType.InCore)
    def gemm(
        self,
        a_shard: pl.Tensor[[m0, k], pl.FP32],
        b: pl.Tensor[[k, n], pl.FP32],
        c_shard: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
    ) -> pl.Tensor[[m0, n], pl.FP32]:
        tile_a_l1 = pl.load(a_shard, offsets=[0, 0], shapes=[m0, k], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[k, n], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        return pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c_shard)

    return gemm


def build_l3_gemm_program(*, nranks: int, m0: int = M0, k: int = K, n: int = N):
    """HOST orchestrator dispatches per-rank GEMM only (no collectives)."""

    if nranks < 1:
        raise ValueError(f"nranks must be >= 1, got {nranks}")

    a_shape = [nranks, m0, k]
    c_shape = [nranks, m0, n]
    gemm_fn = make_cube_gemm(m0=m0, k=k, n=n)

    if nranks == 1:

        @pl.program
        class L3GemmProgramP1:
            gemm = gemm_fn

            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(
                self,
                a_shard: pl.Tensor[[m0, k], pl.FP32],
                b: pl.Tensor[[k, n], pl.FP32],
                c_shard: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
            ) -> pl.Tensor[[m0, n], pl.FP32]:
                return self.gemm(a_shard, b, c_shard)

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[a_shape, pl.FP32],  # type: ignore[valid-type]
                b: pl.Tensor[[k, n], pl.FP32],
                c: pl.Out[pl.Tensor[c_shape, pl.FP32]],  # type: ignore[valid-type]
            ) -> pl.Tensor[c_shape, pl.FP32]:  # type: ignore[valid-type]
                self.chip_orch(a[0], b, c[0], device=0)
                return c

        return L3GemmProgramP1

    @pl.program
    class L3GemmProgramPN:
        gemm = gemm_fn

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a_shard: pl.Tensor[[m0, k], pl.FP32],
            b: pl.Tensor[[k, n], pl.FP32],
            c_shard: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
            _scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[m0, n], pl.FP32]:
            return self.gemm(a_shard, b, c_shard)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[a_shape, pl.FP32],  # type: ignore[valid-type]
            b: pl.Tensor[[k, n], pl.FP32],
            c: pl.Out[pl.Tensor[c_shape, pl.FP32]],  # type: ignore[valid-type]
        ) -> pl.Tensor[c_shape, pl.FP32]:  # type: ignore[valid-type]
            # Dummy 1x1 INT32 window (4 bytes) for CollectCommGroups; GEMM does not use it.
            scratch_buf = pld.alloc_window_buffer(4)
            scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
            for r in pl.range(nranks):
                self.chip_orch(a[r], b, c[r], scratch, device=r)
            return c

    return L3GemmProgramPN
