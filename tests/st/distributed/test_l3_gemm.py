# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST — Phase 2: data-parallel GEMM without collectives.

Data parallel:
  - Shard ``A`` along rows: rank ``r`` owns ``A[r]`` with shape ``[M0, K]``.
  - Replicate ``B`` on every rank: shape ``[K, N]``.
  - Local GEMM per rank: ``C[r] = A[r] @ B``.

Golden:
  - Single chip: ``c == a @ b``.
  - Two ranks:   ``c[r] == a[r] @ b`` for each rank ``r``.
"""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_gemm import M0, K, N, build_l3_gemm_program


class TestL3Gemm:
    """Phase 2: HOST orchestrator dispatches per-rank cube matmul (no windows)."""

    def test_gemm_single_chip(self, test_config, device_ids):
        """One device, full local GEMM golden."""
        if len(device_ids) < 1:
            pytest.skip(f"single-chip GEMM needs >=1 device, got {device_ids}")
        program = build_l3_gemm_program(nranks=1)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:1],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(0)
        a = torch.randn(1, M0, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        c = torch.zeros(1, M0, N, dtype=torch.float32)

        compiled(a, b, c)

        expected = torch.matmul(a[0], b).unsqueeze(0)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"single-chip GEMM mismatch: max diff = {(c - expected).abs().max().item()}"
        )

    def test_gemm_two_rank_sharded_a(self, test_config, device_ids):
        """Shard A across 2 ranks, replicate B, per-rank torch golden."""
        if len(device_ids) < 2:
            pytest.skip(f"two-rank GEMM needs 2 devices, got {device_ids}")

        nranks = 2
        program = build_l3_gemm_program(nranks=nranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nranks],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(42)
        a = torch.randn(nranks, M0, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        c = torch.zeros(nranks, M0, N, dtype=torch.float32)

        compiled(a, b, c)

        for r in range(nranks):
            expected_r = torch.matmul(a[r], b)
            assert torch.allclose(c[r], expected_r, rtol=1e-5, atol=1e-5), (
                f"rank {r} GEMM mismatch: max diff = {(c[r] - expected_r).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
