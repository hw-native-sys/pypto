# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank data-parallel GEMM.

Shard ``A`` along rows on 2 chips; replicate ``B`` on every rank. Each rank runs a
local cube ``pl.matmul`` under HOST orchestration with ``device=r`` dispatch —
the same numerics as two independent single-device GEMMs, wired through the L3
distributed runtime (HOST orchestration, per-device dispatch, InCore codegen).

Host tensors at the test boundary:

* ``a`` — ``[2, M0, K]``; leading dim is rank; chip ``r`` receives ``a[r]``.
* ``b`` — ``[K, N]``; replicated on every rank.
* ``c`` — ``[2, M0, N]``; chip ``r`` writes ``c[r]``.

Program layers:

* **HOST** — ``host_orch`` loops ``r in pld.world_size()``, calls
  ``chip_orch(..., device=r)`` so each rank runs on its NPU.
  ``pld.alloc_window_buffer`` → ``pld.window`` → dispatch with a
  ``DistributedTensor`` satisfies ``MaterializeCommDomainScopes`` comm-group metadata.
* **Orchestration** — ``chip_orch`` sequences InCore kernels on each chip.
* **InCore** — ``gemm``: ``pl.load`` / ``pl.move`` / ``pl.matmul`` / ``pl.store``.
* **Comm window anchor** — ``anchor_comm_window`` stores into ``scratch`` on the
  local rank so ``InOutUseDiscipline`` sees an InOut write. Tile shape is
  ``[1, COMM_ANCHOR_COLS]`` INT32 (``cols * 4 >= 32`` bytes for Vec row alignment).

Golden: ``c[r] == torch.matmul(a[r], b)`` for ``r in {0, 1}``.

Driven by 2 devices via ``DistributedConfig(device_ids=device_ids[:2], ...)``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

M0 = 64
K = 64
N = 64
# INT32 Vec tiles: cols * 4 must be >= 32 bytes for row alignment (8 cols -> 32 B).
COMM_ANCHOR_COLS = 8


def _expected_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: ``c[r] = a[r] @ b`` for 2-rank sharded ``A``."""
    return torch.stack([torch.matmul(a[r], b) for r in range(a.shape[0])])


def _build_gemm_program():
    """Build the 2-rank GEMM program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class L3GemmTwoRank:
        @pl.function(type=pl.FunctionType.InCore)
        def anchor_comm_window(
            self,
            scratch: pl.InOut[pld.DistributedTensor[[1, COMM_ANCHOR_COLS], pl.INT32]],
        ) -> pl.Tensor[[1, COMM_ANCHOR_COLS], pl.INT32]:
            """Local store into this rank's comm window (MaterializeCommDomainScopes / InOut)."""
            tile = pl.tile.full([1, COMM_ANCHOR_COLS], dtype=pl.INT32, value=0)
            return pl.store(tile, [0, 0], scratch)

        @pl.function(type=pl.FunctionType.InCore)
        def gemm(
            self,
            a_shard: pl.Tensor[[M0, K], pl.FP32],
            b: pl.Tensor[[K, N], pl.FP32],
            c_shard: pl.Out[pl.Tensor[[M0, N], pl.FP32]],
        ) -> pl.Tensor[[M0, N], pl.FP32]:
            tile_a_l1 = pl.load(a_shard, offsets=[0, 0], shapes=[M0, K], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            return pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c_shard)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a_shard: pl.Tensor[[M0, K], pl.FP32],
            b: pl.Tensor[[K, N], pl.FP32],
            c_shard: pl.Out[pl.Tensor[[M0, N], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, COMM_ANCHOR_COLS], pl.INT32]],
        ) -> pl.Tensor[[M0, N], pl.FP32]:
            self.anchor_comm_window(scratch)
            return self.gemm(a_shard, b, c_shard)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[[2, M0, K], pl.FP32],
            b: pl.Tensor[[K, N], pl.FP32],
            c: pl.Out[pl.Tensor[[2, M0, N], pl.FP32]],
        ) -> pl.Tensor[[2, M0, N], pl.FP32]:
            scratch_buf = pld.alloc_window_buffer(COMM_ANCHOR_COLS * pl.INT32.get_byte())
            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, COMM_ANCHOR_COLS], dtype=pl.INT32)
                self.chip_orch(a[r], b, c[r], scratch, device=r)
            return c

    return L3GemmTwoRank


class TestL3Gemm:
    """L3 distributed runtime: 2-rank sharded-A GEMM."""

    def test_gemm_two_rank_sharded_a(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"two-rank GEMM needs 2 devices, got {device_ids}")

        program = _build_gemm_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(42)
        a = torch.randn(2, M0, K, dtype=torch.float32)  # shard dim 0 == rank
        b = torch.randn(K, N, dtype=torch.float32)  # shared across ranks
        c = torch.zeros(2, M0, N, dtype=torch.float32)

        compiled(a, b, c)

        expected = _expected_gemm(a, b)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"GEMM mismatch: max diff = {(c - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
