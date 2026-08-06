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
  - **Loop-carried matmul_acc M/N tiling** -- issue #2232's INT8→INT32 ``[16, 1152]``
    split-K reduction. Its physical 32-row accumulator is 144 KiB on Ascend910B, so AutoTile
    must place an output grid outside the source K loop rather than materialize the full Acc;
    a second non-issue shape exercises simultaneous M and N boundary tiles, and a larger
    source panel composes those boundaries with AutoTile's ordinary inner-K rewrite.
  - **Biased matmul** -- ``tile.matmul_bias`` applies its bias exactly once per output tile
    while combining M/N and K tiling, for both direct-GM and chained Mat-scratch placement.

Golden: torch. This is the on-device validation the unit / codegen / pto-verify checks cannot
give (actual execution). Ascend910B (``a2a3``): the Mat-scratch / fits-L0c Acc->Mat lowering is
the 910B bf16 ``pto.tinsert`` FIXPIPE path (the f32 accumulator is downcast into the bf16
scratch); the a5 f32 converting-``pto.tmov`` assemble is a separate lowering.
"""

import dataclasses

import pypto.language as pl
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
# the PyPTO planner. Run every basic case below under all planners to catch
# planner-specific regressions in oversized tiles, GM/L1 drains, and split-K.
_PLANNERS = [
    pytest.param(MemoryPlanner.PYPTO, id="pypto"),
    pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
    pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
]

_ACC_M = 16
_ACC_N = 1152
_ACC_K = 1024
_ACC_K_TILE = 128
_ACC_N_TOTAL = _ACC_N * 8

_BOUNDARY_M = 272
_BOUNDARY_N = 144
_BOUNDARY_K = 256
_BOUNDARY_K_TILE = 128

_COMPOSE_K = 768
_COMPOSE_K_TILE = 384

_BIAS_M = 256
_BIAS_N = 256
_BIAS_K = 256
_BIAS_SCRATCH_K = 192
_BIAS_OUT_N = 64
_BIAS_BOUNDARY_M = 272
_BIAS_BOUNDARY_K = 64
_BIAS_BOUNDARY_N = 144


@pl.jit
def matmul_acc_mn_issue_2232(
    a: pl.Tensor[[_ACC_M, _ACC_K], pl.INT8],
    b: pl.Tensor[[_ACC_K, _ACC_N_TOTAL], pl.INT8],
    c: pl.Out[pl.Tensor[[_ACC_M, _ACC_N_TOTAL], pl.INT32]],
):
    """Canonical frontend split-K form whose physical Acc exceeds L0C."""
    for i in pl.spmd(_ACC_N_TOTAL // _ACC_N, name_hint="mm"):
        n0 = i * _ACC_N
        acc = pl.create_tensor([_ACC_M, _ACC_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, _ACC_K // _ACC_K_TILE, stage=2):
            k0 = kb * _ACC_K_TILE
            at = a[0:_ACC_M, k0 : k0 + _ACC_K_TILE]
            bt = b[k0 : k0 + _ACC_K_TILE, n0 : n0 + _ACC_N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:_ACC_M, n0 : n0 + _ACC_N] = acc
    return c


@pl.jit
def matmul_acc_mn_boundaries(
    a: pl.Tensor[[_BOUNDARY_M, _BOUNDARY_K], pl.INT8],
    b: pl.Tensor[[_BOUNDARY_K, _BOUNDARY_N], pl.INT8],
    c: pl.Out[pl.Tensor[[_BOUNDARY_M, _BOUNDARY_N], pl.INT32]],
):
    """General split-K case requiring both M and N boundary output tiles."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([_BOUNDARY_M, _BOUNDARY_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, _BOUNDARY_K // _BOUNDARY_K_TILE, stage=2):
            k0 = kb * _BOUNDARY_K_TILE
            at = a[0:_BOUNDARY_M, k0 : k0 + _BOUNDARY_K_TILE]
            bt = b[k0 : k0 + _BOUNDARY_K_TILE, 0:_BOUNDARY_N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:_BOUNDARY_M, 0:_BOUNDARY_N] = acc
    return c


@pl.jit
def matmul_acc_n_boundary_retiles_k(
    a: pl.Tensor[[_BOUNDARY_M, _COMPOSE_K], pl.INT8],
    b: pl.Tensor[[_COMPOSE_K, _BOUNDARY_N], pl.INT8],
    c: pl.Out[pl.Tensor[[_BOUNDARY_M, _BOUNDARY_N], pl.INT32]],
):
    """N-tail padding composed with AutoTile's ordinary inner-K rewrite."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([_BOUNDARY_M, _BOUNDARY_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, _COMPOSE_K // _COMPOSE_K_TILE, stage=2):
            k0 = kb * _COMPOSE_K_TILE
            at = a[0:_BOUNDARY_M, k0 : k0 + _COMPOSE_K_TILE]
            bt = b[k0 : k0 + _COMPOSE_K_TILE, 0:_BOUNDARY_N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:_BOUNDARY_M, 0:_BOUNDARY_N] = acc
    return c


@pl.jit
def matmul_bias_mn_k_direct(
    a: pl.Tensor[[_BIAS_M, _BIAS_K], pl.BF16],
    b: pl.Tensor[[_BIAS_K, _BIAS_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_N], pl.FP32],
    out: pl.Out[pl.Tensor[[_BIAS_M, _BIAS_N], pl.FP32]],
):
    """Biased GEMM whose output and K reduction both require AutoTile."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_mn_k_direct"):
        a_mat = pl.load(a, [0, 0], [_BIAS_M, _BIAS_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_K, _BIAS_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_N], target_memory=pl.Mem.Mat)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
        out = pl.store(c, [0, 0], out)
    return out


@pl.jit
def matmul_bias_mn_k_scratch(
    a: pl.Tensor[[_BIAS_M, _BIAS_SCRATCH_K], pl.BF16],
    b: pl.Tensor[[_BIAS_SCRATCH_K, _BIAS_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_N], pl.FP32],
    e: pl.Tensor[[_BIAS_N, _BIAS_OUT_N], pl.BF16],
    out: pl.Out[pl.Tensor[[_BIAS_M, _BIAS_OUT_N], pl.FP32]],
):
    """Biased GEMM whose tiled result stays in a bf16 Mat scratch."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_mn_k_scratch"):
        a_mat = pl.load(a, [0, 0], [_BIAS_M, _BIAS_SCRATCH_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_SCRATCH_K, _BIAS_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_N], target_memory=pl.Mem.Mat)
        e_mat = pl.load(e, [0, 0], [_BIAS_N, _BIAS_OUT_N], target_memory=pl.Mem.Mat)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
        cb = pl.cast(c, pl.BF16, mode="rint")
        d = pl.tile.matmul(cb, e_mat)
        out = pl.store(d, [0, 0], out)
    return out


@pl.jit
def matmul_bias_mn_boundary_direct(
    a: pl.Tensor[[_BIAS_BOUNDARY_M, _BIAS_BOUNDARY_K], pl.BF16],
    b: pl.Tensor[[_BIAS_BOUNDARY_K, _BIAS_BOUNDARY_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_BOUNDARY_N], pl.FP32],
    out: pl.Out[pl.Tensor[[_BIAS_BOUNDARY_M, _BIAS_BOUNDARY_N], pl.FP32]],
):
    """Biased GEMM with partial M and N output tiles."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_mn_boundary_direct"):
        a_mat = pl.load(a, [0, 0], [_BIAS_BOUNDARY_M, _BIAS_BOUNDARY_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_BOUNDARY_K, _BIAS_BOUNDARY_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_BOUNDARY_N], target_memory=pl.Mem.Mat)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
        out = pl.store(c, [0, 0], out)
    return out


def _cfg(test_config, planner):
    """Return the session config with the requested planner selected explicitly."""
    return dataclasses.replace(test_config, memory_planner=planner)


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestAutoTileMatmulL0:
    """End-to-end device checks for the placement x K-strategy x planner matrix."""

    @pytest.mark.parametrize("planner", _PLANNERS)
    @pytest.mark.parametrize("kernel, K", [(ddr_split_k, 128), (ddr_full_k, 32)])
    def test_ddr_direct_store(self, test_config, kernel, K, planner):
        """``a @ b`` -> ``[256, 256]`` stored to DDR (direct-store); split-K (K=128) and
        full-K (K=32).  Run under all three planners: the oversized grid reuses the L0C
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
    def test_matmul_bias_mn_k_direct_store(self, test_config, planner):
        """Bias is applied once while M/N sub-tiles complete a split-K reduction."""
        matmul_bias_mn_k_direct._cache.clear()
        torch.manual_seed(11)
        a = torch.randn(_BIAS_M, _BIAS_K, dtype=torch.bfloat16)
        b = torch.randn(_BIAS_K, _BIAS_N, dtype=torch.bfloat16)
        bias = torch.randn((1, _BIAS_N), dtype=torch.float32)
        out = torch.zeros((_BIAS_M, _BIAS_N), dtype=torch.float32)

        matmul_bias_mn_k_direct(a, b, bias, out, config=_cfg(test_config, planner))

        expected = a.float() @ b.float() + bias
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 2e-2, f"direct matmul_bias rel_err {rel_err:.3e} exceeds 2e-2"

    @pytest.mark.parametrize("planner", _PLANNERS)
    def test_matmul_bias_mn_k_mat_scratch(self, test_config, planner):
        """A biased producer is tiled into Mat scratch for its sole matmul consumer."""
        matmul_bias_mn_k_scratch._cache.clear()
        torch.manual_seed(12)
        a = torch.randn(_BIAS_M, _BIAS_SCRATCH_K, dtype=torch.bfloat16)
        b = torch.randn(_BIAS_SCRATCH_K, _BIAS_N, dtype=torch.bfloat16)
        bias = torch.randn((1, _BIAS_N), dtype=torch.float32)
        e = torch.randn(_BIAS_N, _BIAS_OUT_N, dtype=torch.bfloat16)
        out = torch.zeros((_BIAS_M, _BIAS_OUT_N), dtype=torch.float32)

        matmul_bias_mn_k_scratch(a, b, bias, e, out, config=_cfg(test_config, planner))

        intermediate = (a.float() @ b.float() + bias).to(torch.bfloat16).float()
        expected = intermediate @ e.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 2e-2, f"Mat-scratch matmul_bias rel_err {rel_err:.3e} exceeds 2e-2"

    @pytest.mark.parametrize("planner", _PLANNERS)
    def test_matmul_bias_mn_boundaries(self, test_config, planner):
        """Partial M/N tiles preserve the logical Bias and output regions."""
        matmul_bias_mn_boundary_direct._cache.clear()
        torch.manual_seed(13)
        a = torch.randn(_BIAS_BOUNDARY_M, _BIAS_BOUNDARY_K, dtype=torch.bfloat16)
        b = torch.randn(_BIAS_BOUNDARY_K, _BIAS_BOUNDARY_N, dtype=torch.bfloat16)
        bias = torch.randn((1, _BIAS_BOUNDARY_N), dtype=torch.float32)
        out = torch.zeros((_BIAS_BOUNDARY_M, _BIAS_BOUNDARY_N), dtype=torch.float32)

        matmul_bias_mn_boundary_direct(a, b, bias, out, config=_cfg(test_config, planner))

        expected = a.float() @ b.float() + bias
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 2e-2, f"boundary matmul_bias rel_err {rel_err:.3e} exceeds 2e-2"

    @pytest.mark.parametrize("planner", _PLANNERS)
    def test_loop_carried_matmul_acc_mn_tiling(self, test_config, planner):
        """Issue #2232: each output tile must finish all eight source K blocks.

        The logical ``[16, 1152]`` INT32 result is only 72 KiB, but its physical
        32-row L0C footprint is 144 KiB. Run both planners and compare exactly:
        integer accumulation has no numerical tolerance.
        """
        matmul_acc_mn_issue_2232._cache.clear()
        torch.manual_seed(0)
        a = torch.randint(-3, 4, (_ACC_M, _ACC_K), dtype=torch.int8)
        b = torch.randint(-3, 4, (_ACC_K, _ACC_N_TOTAL), dtype=torch.int8)
        out = torch.zeros((_ACC_M, _ACC_N_TOTAL), dtype=torch.int32)

        matmul_acc_mn_issue_2232(a, b, out, config=_cfg(test_config, planner))

        expected = a.int() @ b.int()
        assert torch.equal(out, expected), (
            f"matmul_acc M/N tiling mismatch: max abs diff = {(out - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("planner", _PLANNERS)
    def test_loop_carried_matmul_acc_both_mn_boundaries(self, test_config, planner):
        """General #2232 rewrite: exact INT8→INT32 split-K with partial tiles
        on both output axes, under both memory planners."""
        matmul_acc_mn_boundaries._cache.clear()
        torch.manual_seed(1)
        a = torch.randint(-3, 4, (_BOUNDARY_M, _BOUNDARY_K), dtype=torch.int8)
        b = torch.randint(-3, 4, (_BOUNDARY_K, _BOUNDARY_N), dtype=torch.int8)
        out = torch.zeros((_BOUNDARY_M, _BOUNDARY_N), dtype=torch.int32)

        matmul_acc_mn_boundaries(a, b, out, config=_cfg(test_config, planner))

        expected = a.int() @ b.int()
        assert torch.equal(out, expected), (
            f"matmul_acc both-boundary tiling mismatch: max abs diff = {(out - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("planner", _PLANNERS)
    def test_loop_carried_matmul_acc_n_boundary_retiles_k(self, test_config, planner):
        """A padded N tail remains valid through secondary inner-K tiling."""
        matmul_acc_n_boundary_retiles_k._cache.clear()
        torch.manual_seed(2)
        a = torch.randint(-3, 4, (_BOUNDARY_M, _COMPOSE_K), dtype=torch.int8)
        b = torch.randint(-3, 4, (_COMPOSE_K, _BOUNDARY_N), dtype=torch.int8)
        out = torch.zeros((_BOUNDARY_M, _BOUNDARY_N), dtype=torch.int32)

        matmul_acc_n_boundary_retiles_k(a, b, out, config=_cfg(test_config, planner))

        expected = a.int() @ b.int()
        assert torch.equal(out, expected), (
            f"matmul_acc padded-N + K-tiling mismatch: max abs diff = {(out - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("planner", _PLANNERS)
    @pytest.mark.parametrize("kernel, K", [(mat_split_k, 192), (mat_full_k, 32)])
    def test_mat_scratch(self, test_config, kernel, K, planner):
        """``(a @ b) @ e`` with a bf16 ``[256, 256]`` intermediate kept on-chip in an
        L1/Mat scratch (Acc->Mat ``pto.tinsert``); split-K K=192 and full-K K=32.

        Run under all three planners.  The PTOAS variants provide regression coverage
        for #1995: the chained consumer's K-reduction accumulator if-phi must reuse
        the dominating accumulator handle so all partial sums land in one L0C buffer.

        K=192 is the common cross-planner split point: all planners choose an
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

        Run under all three planners: because the intermediate fits L0c there is exactly ONE
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
