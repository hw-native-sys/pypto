# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST — Phase 3: GEMM partial + SUM allreduce (replicated output).

Data parallel: shard ``A[r]``, replicate ``B``.

Per rank (two stages inside ``chip_orch``):
  1. Local GEMM:    ``P_r = A[r] @ B``.
  2. 4-phase allreduce: stage-in, notify/wait, remote_load + add, stage-out.

Golden: ``outputs[r] == sum_s matmul(a[s], b)`` for every rank ``r``.

Two ranks only (single peer read in the reduce kernel).
"""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_allreduce_gemm import M0, K, N, build_l3_allreduce_gemm_program


def _golden_allreduce_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Replicated output: sum of per-rank partial GEMMs."""
    partials = [torch.matmul(a[r], b) for r in range(a.shape[0])]
    reduced = sum(partials)
    return torch.stack([reduced] * a.shape[0])


class TestL3AllReduceGemm:
    """Phase 3: compose cube GEMM with 4-phase window allreduce on 2 devices."""

    def test_allreduce_gemm(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"allreduce-gemm needs 2 devices, got {device_ids}")

        nranks = 2
        program = build_l3_allreduce_gemm_program(nranks=nranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nranks],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(7)
        a = torch.randn(nranks, M0, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        partials = torch.zeros(nranks, M0, N, dtype=torch.float32)
        outputs = torch.zeros(nranks, M0, N, dtype=torch.float32)

        compiled(a, b, partials, outputs)

        expected = _golden_allreduce_gemm(a, b)
        assert torch.allclose(outputs, expected, rtol=1e-5, atol=1e-5), (
            f"allreduce-gemm mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
