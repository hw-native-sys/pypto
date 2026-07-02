# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st for AutoTileMatmulL0's compiler-driven L0 tiling.

Validates on device the cases from examples/kernels/11_auto_tile_matmul.py:

  - **Oversized 2x2 matrix** -- an oversized ``[256, 256]`` FP32 output (> L0c) tiled and
    placed either to **DDR** (direct-store) or an **L1/Mat scratch** (consumed on-chip by a
    second matmul), each with **full-K** (K=32, k == K) or **split-K** (K=128) reduction.
  - **Fits-L0c cast-fold** -- a chained ``(a @ b) @ e`` whose ``[128, 128]`` intermediate
    *fits* L0c (no M/N tiling); the ``pl.cast`` is folded into a single full-window Acc->Mat
    ``pto.tinsert``, so the bf16 downcast stays on the cube. full-K (K=64) and split-K (K=512).

Golden: torch. This is the on-device validation the unit / codegen / pto-verify checks cannot
give (actual execution). Ascend910B (``a2a3``): the Mat-scratch / fits-L0c Acc->Mat lowering is
the 910B bf16 ``pto.tinsert`` FIXPIPE path (the f32 accumulator is downcast into the bf16
scratch); the a5 f32 converting-``pto.tmov`` assemble is a separate lowering.
"""

import pytest
import torch
from examples.kernels.auto_tile_matmul import (
    ddr_full_k,
    ddr_split_k,
    fits_l0c_full_k,
    fits_l0c_split_k,
    mat_full_k,
    mat_split_k,
)


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

    @pytest.mark.parametrize("kernel, K", [(mat_split_k, 64), (mat_full_k, 32)])
    def test_mat_scratch(self, test_config, kernel, K):
        """``(a @ b) @ e`` with a bf16 ``[256, 256]`` intermediate kept on-chip in an
        L1/Mat scratch (Acc->Mat ``pto.tinsert``); K=64 and K=32.

        K is chosen so the chained producer and consumer pick the **same
        (output-stationary) algorithm**, so their L0 buffers have matching shapes and
        pack under `AllocateMemoryAddr`. A K where the roofline chooser makes the
        producer A/B-stationary (e.g. K=128 -> ``(256,128,128)A``) pins a monolithic
        full-L0A buffer the consumer's double-buffers cannot pack against -> `Left
        buffer usage exceeds` — the offset-packing gap tracked in #1908. Both matmuls
        here are also full-K (no K-loop peel), so this does not depend on #1924.

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

    @pytest.mark.parametrize("kernel, K", [(fits_l0c_full_k, 64), (fits_l0c_split_k, 512)])
    def test_fits_l0c_cast_fold(self, test_config, kernel, K):
        """``(a @ b) @ e`` with a ``[128, 128]`` intermediate that *fits* L0c (no M/N
        tiling): the autotiler folds ``pl.cast`` into a single full-window Acc->Mat
        ``pto.tinsert`` (cube downcast) rather than a Vector ``pto.tcvt``. full-K (K=64,
        no K-loop) and split-K (K=512, K-loop). Same bf16 FIXPIPE golden as Mat-scratch.

        On-device proof that the fold is numerically correct (the FIXPIPE bf16 rounding
        matches the reference) AND that it compiles — the un-folded Vector cast overflows
        the Vec buffer at this ``[128, 128]`` shape."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(128, K, dtype=torch.bfloat16)
        b = torch.randn(K, 128, dtype=torch.bfloat16)
        e = torch.randn(128, 64, dtype=torch.bfloat16)
        out = torch.zeros((128, 64), dtype=torch.float32)

        kernel(a, b, e, out, config=test_config)

        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        expected = c_bf16 @ e.float()
        # Frobenius relative error, not allclose: a bf16 ``(a @ b) @ e`` chain has
        # near-zero cancellation elements where the absolute bf16 rounding error (~0.7 on
        # operand magnitudes of ~500) dwarfs the small true value, so a per-element atol
        # fails on a numerically-correct result. The global relative norm is the robust
        # metric (the unit tests use the same). K=512 makes the intermediate magnitudes
        # large enough to bite; K=64 happens to pass allclose, but both use one metric.
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 5e-2, (
            f"{kernel.__name__} (fits-L0c cast-fold) Frobenius rel_err = {rel_err:.3e} exceeds 5e-2"
        )
