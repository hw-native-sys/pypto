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
    second matmul), each with **full-K** (K=32, k == K) or **split-K** reduction (K=128
    for direct-store, K=192 for the common cross-planner Mat-scratch split).
  - **Fits-L0c cast-fold** -- a chained ``(a @ b) @ e`` whose ``[128, 128]`` intermediate
    *fits* L0c (no M/N tiling); the ``pl.cast`` is folded into a single full-window Acc->Mat
    ``pto.tinsert``, so the bf16 downcast stays on the cube. full-K (K=64) and split-K (K=512).

Golden: torch. This is the on-device validation the unit / codegen / pto-verify checks cannot
give (actual execution). Ascend910B (``a2a3``): the Mat-scratch / fits-L0c Acc->Mat lowering is
the 910B bf16 ``pto.tinsert`` FIXPIPE path (the f32 accumulator is downcast into the bf16
scratch); the a5 f32 converting-``pto.tmov`` assemble is a separate lowering.
"""

import dataclasses

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
from pypto.pypto_core.passes import MemoryPlanner

# AutoTileMatmulL0 predates memory_planner=PTOAS and was initially validated under
# the PyPTO planner. Run every basic case below under both planners to catch
# planner-specific regressions in oversized tiles, GM/L1 drains, and split-K.
_PLANNERS = [pytest.param(None, id="pypto"), pytest.param(MemoryPlanner.PTOAS, id="ptoas")]


def _cfg(test_config, planner):
    """Base session config, overridden to a specific memory planner (None = PyPTO default)."""
    return test_config if planner is None else dataclasses.replace(test_config, memory_planner=planner)


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestAutoTileMatmulL0:
    """End-to-end device checks for the placement x K-strategy x planner matrix."""

    @pytest.mark.parametrize("planner", _PLANNERS)
    @pytest.mark.parametrize("kernel, K", [(ddr_split_k, 128), (ddr_full_k, 32)])
    def test_ddr_direct_store(self, test_config, kernel, K, planner):
        """``a @ b`` -> ``[256, 256]`` stored to DDR (direct-store); split-K (K=128) and
        full-K (K=32).  Run under both planners: the oversized grid reuses the L0C
        accumulator across output tiles, but the Acc->GM ``tile.store`` drain WAR is synced
        correctly by ptoas, so oversized direct-store works under PTOAS too."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(256, K, dtype=torch.float32)
        b = torch.randn(K, 256, dtype=torch.float32)
        out = torch.zeros((256, 256), dtype=torch.float32)

        kernel(a, b, out, config=_cfg(test_config, planner))

        expected = a @ b
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
            f"{kernel.__name__} (DDR direct-store) max abs diff = {(out - expected).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("planner", _PLANNERS)
    @pytest.mark.parametrize("kernel, K", [(mat_split_k, 192), (mat_full_k, 32)])
    def test_mat_scratch(self, test_config, kernel, K, planner):
        """``(a @ b) @ e`` with a bf16 ``[256, 256]`` intermediate kept on-chip in an
        L1/Mat scratch (Acc->Mat ``pto.tinsert``); split-K K=192 and full-K K=32.

        Run under both planners.  The PTOAS variants provide regression coverage
        for #1995: the chained consumer's K-reduction accumulator if-phi must reuse
        the dominating accumulator handle so all partial sums land in one L0C buffer.

        K=192 is the common cross-planner split point: both planners choose an
        output-stationary producer with k=64, so its L0 buffers pack against the
        consumer's. K=128 is planner-dependent (PyPTO splits while PTOAS can keep full K)
        and can select a monolithic A/B-stationary buffer that the consumer's two
        half-size buffers cannot pack against. The pass deliberately avoids that
        issue-1908 regime by forcing chained Mat-scratch producers output-stationary.

        Operands are bf16 and the on-chip intermediate is bf16 — the cube's FIXPIPE
        writeback to L1 downcasts the f32 accumulator, which is also the cube's native
        operand precision. The golden models that downcast; compare by global relative
        norm because cancellation-near-zero elements make per-element ``allclose``
        unstable for this chained reduction."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(256, K, dtype=torch.bfloat16)
        b = torch.randn(K, 256, dtype=torch.bfloat16)
        e = torch.randn(256, 64, dtype=torch.bfloat16)
        out = torch.zeros((256, 64), dtype=torch.float32)

        kernel(a, b, e, out, config=_cfg(test_config, planner))

        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        expected = c_bf16 @ e.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 2e-2, (
            f"{kernel.__name__} (Mat-scratch) Frobenius rel_err = {rel_err:.3e} exceeds 2e-2"
        )

    @pytest.mark.parametrize("planner", _PLANNERS)
    @pytest.mark.parametrize("kernel, K", [(fits_l0c_full_k, 64), (fits_l0c_split_k, 512)])
    def test_fits_l0c_cast_fold(self, test_config, kernel, K, planner):
        """``(a @ b) @ e`` with a ``[128, 128]`` intermediate that *fits* L0c (no M/N
        tiling): the autotiler folds ``pl.cast`` into a single full-window Acc->Mat
        ``pto.tinsert`` (cube downcast) rather than a Vector ``pto.tcvt``. full-K (K=64,
        no K-loop) and split-K (K=512, K-loop). Same bf16 FIXPIPE golden as Mat-scratch.

        Run under both planners: because the intermediate fits L0c there is exactly ONE
        Acc->Mat assemble (no cross-tile L0C reuse and no drain/MAD WAR fence).

        On-device proof that the fold is numerically correct (the FIXPIPE bf16 rounding
        matches the reference) AND that it compiles — the un-folded Vector cast overflows
        the Vec buffer at this ``[128, 128]`` shape."""
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(128, K, dtype=torch.bfloat16)
        b = torch.randn(K, 128, dtype=torch.bfloat16)
        e = torch.randn(128, 64, dtype=torch.bfloat16)
        out = torch.zeros((128, 64), dtype=torch.float32)

        kernel(a, b, e, out, config=_cfg(test_config, planner))

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
