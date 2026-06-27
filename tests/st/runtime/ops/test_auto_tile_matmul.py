# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st for AutoTileMatmulL0's compiler-driven L0 tiling -- the 2x2 matrix.

Validates on device the four cases from examples/kernels/11_auto_tile_matmul.py: an oversized
``[256, 256]`` FP32 matmul output (> L0c) tiled by the compiler and placed either to **DDR**
(direct-store) or an **L1/Mat scratch** (consumed on-chip by a second matmul), each with
**full-K** (K=32, k == K) or **split-K** (K=128) reduction. Golden: torch.

This is the on-device validation the unit / codegen / pto-verify checks cannot give (actual
execution). Ascend910B (``a2a3``) only: the Mat-scratch Acc->Mat lowering is the 910B
converting-``pto.tmov`` path; the a5 ``TINSERT`` assemble is a separate lowering.
"""

import pytest
import torch
from examples.kernels.auto_tile_matmul import ddr_full_k, ddr_split_k, mat_full_k, mat_split_k


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestAutoTileMatmulL0:
    """End-to-end device checks for the placement x K-strategy matrix."""

    @pytest.mark.parametrize("kernel, K", [(ddr_split_k, 128), (ddr_full_k, 32)])
    def test_ddr_direct_store(self, test_config, kernel, K):
        """``a @ b`` -> ``[256, 256]`` stored to DDR (direct-store); split-K (K=128) and
        full-K (K=32)."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(256, K, dtype=torch.float32)
        b = torch.randn(K, 256, dtype=torch.float32)
        out = torch.zeros((256, 256), dtype=torch.float32)

        kernel(a, b, out, config=test_config)

        expected = a @ b
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
            f"{kernel.__name__} (DDR direct-store) max abs diff = {(out - expected).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("kernel, K", [(mat_split_k, 128), (mat_full_k, 32)])
    def test_mat_scratch(self, test_config, kernel, K):
        """``(a @ b) @ e`` with a bf16 ``[256, 256]`` intermediate kept on-chip in an
        L1/Mat scratch (Acc->Mat ``pto.tinsert``); split-K (K=128) and full-K (K=32).

        Operands are bf16 and the on-chip intermediate is bf16 — the cube's FIXPIPE
        writeback to L1 downcasts the f32 accumulator, which is also the cube's native
        operand precision. The golden models that downcast, so the tolerance is bf16."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(256, K, dtype=torch.bfloat16)
        b = torch.randn(K, 256, dtype=torch.bfloat16)
        e = torch.randn(256, 64, dtype=torch.bfloat16)
        out = torch.zeros((256, 64), dtype=torch.float32)

        kernel(a, b, e, out, config=test_config)

        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        expected = c_bf16 @ e.float()
        assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2), (
            f"{kernel.__name__} (Mat-scratch) max abs diff = {(out - expected).abs().max().item():.3e}"
        )
