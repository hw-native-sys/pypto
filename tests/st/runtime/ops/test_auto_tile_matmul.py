# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st for AutoTileMatmulL0's compiler-driven L0 tiling.

Validates on device the cases from examples/advanced/02_auto_tile_matmul.py:

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
    A2/A3 exercises both its INT32→INT32 and FP32→FP32 Mat→Bias transfers;
    the floating FP32-bias cases are also covered on A5.

Golden: torch. This is the on-device validation the unit / codegen / pto-verify checks cannot
give (actual execution). Ascend910B (``a2a3``): the Mat-scratch / fits-L0c Acc->Mat lowering is
the 910B bf16 ``pto.tinsert`` FIXPIPE path (the f32 accumulator is downcast into the bf16
scratch); the a5 f32 converting-``pto.tmov`` assemble is a separate lowering.
"""

import pypto.language as pl
import pytest
import torch
from examples.advanced.auto_tile_matmul import (
    ddr_full_k,
    ddr_split_k,
    fits_l0c_full_k,
    fits_l0c_split_k,
    mat_full_k,
    mat_split_k,
)
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

# AutoTileMatmulL0 predates memory_planner=PTOAS and was initially validated under
# the PyPTO planner. Run every basic case below under all planners to catch
# planner-specific regressions in oversized tiles, GM/L1 drains, and split-K.
_PLANNERS = [
    pytest.param(MemoryPlanner.PYPTO, id="pypto"),
    pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
    pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
]

_N_BOUNDARY_RETILES_K_PLANNERS = [
    *_PLANNERS[:2],
    pytest.param(
        MemoryPlanner.PTOAS,
        id="ptoas",
        marks=pytest.mark.skip(
            reason=(
                "PTOAS v0.57 legacy PlanMemory assigns overlapping addresses to the two "
                "alloc_multi_tile slots; restore after the upstream planner is fixed"
            )
        ),
    ),
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
_BIAS_N = 512
_BIAS_K = 256
_BIAS_SCRATCH_M = 272
_BIAS_SCRATCH_K = 192
_BIAS_SCRATCH_N = 352
_BIAS_SCRATCH_OUT_N = 32
_BIAS_BOUNDARY_M = 528
_BIAS_BOUNDARY_K = 32
_BIAS_BOUNDARY_N = 528
_BIAS_PEEL_M = 64
_BIAS_PEEL_K = 272
_BIAS_PEEL_N = 64
_BIAS_M_ONLY_M = 1040
_BIAS_M_ONLY_K = 64
_BIAS_M_ONLY_N = 64

_BIAS_INT_M = 128
_BIAS_INT_K = 512
_BIAS_INT_N = 512


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
    a: pl.Tensor[[_BIAS_SCRATCH_M, _BIAS_SCRATCH_K], pl.BF16],
    b: pl.Tensor[[_BIAS_SCRATCH_K, _BIAS_SCRATCH_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_SCRATCH_N], pl.FP32],
    e: pl.Tensor[[_BIAS_SCRATCH_N, _BIAS_SCRATCH_OUT_N], pl.BF16],
    out: pl.Out[pl.Tensor[[_BIAS_SCRATCH_M, _BIAS_SCRATCH_OUT_N], pl.FP32]],
):
    """Biased GEMM whose tiled result stays in a bf16 Mat scratch."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_mn_k_scratch"):
        a_mat = pl.load(a, [0, 0], [_BIAS_SCRATCH_M, _BIAS_SCRATCH_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_SCRATCH_K, _BIAS_SCRATCH_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_SCRATCH_N], target_memory=pl.Mem.Mat)
        e_mat = pl.load(e, [0, 0], [_BIAS_SCRATCH_N, _BIAS_SCRATCH_OUT_N], target_memory=pl.Mem.Mat)
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


@pl.jit
def matmul_bias_nondivisor_k_tail(
    a: pl.Tensor[[_BIAS_PEEL_M, _BIAS_PEEL_K], pl.BF16],
    b: pl.Tensor[[_BIAS_PEEL_K, _BIAS_PEEL_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_PEEL_N], pl.FP32],
    out: pl.Out[pl.Tensor[[_BIAS_PEEL_M, _BIAS_PEEL_N], pl.FP32]],
):
    """Biased GEMM whose selected K tile leaves an aligned peeled tail."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_nondivisor_k_tail"):
        a_mat = pl.load(a, [0, 0], [_BIAS_PEEL_M, _BIAS_PEEL_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_PEEL_K, _BIAS_PEEL_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_PEEL_N], target_memory=pl.Mem.Mat)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
        out = pl.store(c, [0, 0], out)
    return out


@pl.jit
def matmul_bias_m_only_bias_resident(
    a: pl.Tensor[[_BIAS_M_ONLY_M, _BIAS_M_ONLY_K], pl.BF16],
    b: pl.Tensor[[_BIAS_M_ONLY_K, _BIAS_M_ONLY_N], pl.BF16],
    bias: pl.Tensor[[1, _BIAS_M_ONLY_N], pl.FP32],
    out: pl.Out[pl.Tensor[[_BIAS_M_ONLY_M, _BIAS_M_ONLY_N], pl.FP32]],
):
    """M-only output tiling reuses a full architectural Bias tile."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_m_only_bias_resident"):
        a_mat = pl.load(a, [0, 0], [_BIAS_M_ONLY_M, _BIAS_M_ONLY_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_M_ONLY_K, _BIAS_M_ONLY_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_M_ONLY_N], target_memory=pl.Mem.Mat)
        bias_l0 = pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_l0)
        out = pl.store(c, [0, 0], out)
    return out


@pl.jit
def matmul_bias_a2a3_int_direct(
    a: pl.Tensor[[_BIAS_INT_M, _BIAS_INT_K], pl.INT8],
    b: pl.Tensor[[_BIAS_INT_K, _BIAS_INT_N], pl.INT8],
    bias: pl.Tensor[[1, _BIAS_INT_N], pl.INT32],
    out: pl.Out[pl.Tensor[[_BIAS_INT_M, _BIAS_INT_N], pl.INT32]],
):
    """A2/A3 biased GEMM using its supported INT32 Mat-to-Bias path."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_bias_a2a3_int_direct"):
        a_mat = pl.load(a, [0, 0], [_BIAS_INT_M, _BIAS_INT_K], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [_BIAS_INT_K, _BIAS_INT_N], target_memory=pl.Mem.Mat)
        bias_mat = pl.load(bias, [0, 0], [1, _BIAS_INT_N], target_memory=pl.Mem.Mat)
        c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
        out = pl.store(c, [0, 0], out)
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
#
# Each original test crossed its kernel with the planner matrix and re-seeded
# per item, so every planner saw identical inputs; the builders below keep that
# by seeding inside each builder. A planner that was skipped by a run-time
# ``pytest.skip`` in the body is now a collection-time mark, so the case is not
# pre-compiled either.
#
# Three kinds of comparison, all preserved:
#   * ``rtol``/``atol``          — the elementwise default (DDR direct-store)
#   * ``rtol=atol=0``            — the exact integer checks (``torch.equal``)
#   * ``st.rel_err_under(x)``    — the Frobenius bound the bf16 matmul chains use,
#                                  where near-zero cancellation elements make a
#                                  per-element tolerance meaningless


def _expand(build, planners=_PLANNERS):
    """Expand *build(planner, planner_id)* over a planner matrix.

    Each entry of the matrix is a ``pytest.param``; its id becomes part of the
    case name and its marks (e.g. the PTOAS skip in
    ``_N_BOUNDARY_RETILES_K_PLANNERS``) carry over to the case.
    """
    expanded = []
    for entry in planners:
        planner = entry.values[0]
        case_obj = build(planner, entry.id)
        expanded.append(pytest.param(case_obj, marks=entry.marks) if entry.marks else case_obj)
    return expanded


_PTOAS_ONLY_SKIPS = {
    "mat_scratch_bias": "PTOAS planner path currently fails this Mat-scratch kernel on device",
    "mn_boundaries": "PTOAS currently fails this partial M/N boundary kernel on device",
}


def _without_ptoas(planners, reason_key):
    """The planner matrix with PTOAS marked skip rather than skipped in the body."""
    kept = []
    for entry in planners:
        if entry.values[0] is MemoryPlanner.PTOAS:
            kept.append(
                pytest.param(
                    entry.values[0],
                    id=entry.id,
                    marks=pytest.mark.skip(reason=_PTOAS_ONLY_SKIPS[reason_key]),
                )
            )
        else:
            kept.append(entry)
    return kept


def _bias_case(kernel, name, planner, pid, m, k, n, seed):
    """``a @ b + bias`` with bf16 operands and an fp32 row bias."""
    torch.manual_seed(seed)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    bias = torch.randn((1, n), dtype=torch.float32)
    out = torch.zeros((m, n), dtype=torch.float32)
    return st.case(
        kernel,
        a,
        b,
        bias,
        out,
        name=f"{name}_{pid}",
        memory_planner=planner,
        golden=lambda _: a.float() @ b.float() + bias,
        compare=st.rel_err_under(2e-2),
    )


def _int_acc_case(kernel, name, planner, pid, m, k, n, seed):
    """INT8 x INT8 -> INT32 split-K. Integer accumulation is exact, so rtol=atol=0."""
    torch.manual_seed(seed)
    a = torch.randint(-3, 4, (m, k), dtype=torch.int8)
    b = torch.randint(-3, 4, (k, n), dtype=torch.int8)
    out = torch.zeros((m, n), dtype=torch.int32)
    return st.case(
        kernel,
        a,
        b,
        out,
        name=f"{name}_{pid}",
        memory_planner=planner,
        golden=lambda _: a.int() @ b.int(),
        rtol=0.0,
        atol=0.0,
    )


def _chained_case(kernel, name, planner, pid, m, k, n, out_n, limit):
    """``(a @ b) @ e`` with a bf16 on-chip intermediate (FIXPIPE downcast)."""
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    e = torch.randn(n, out_n, dtype=torch.bfloat16)
    out = torch.zeros((m, out_n), dtype=torch.float32)

    def golden(_):
        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        return c_bf16 @ e.float()

    return st.case(
        kernel,
        a,
        b,
        e,
        out,
        name=f"{name}_{pid}",
        memory_planner=planner,
        golden=golden,
        compare=st.rel_err_under(limit),
    )


def _ddr_case(kernel, planner, pid, k):
    """``a @ b`` -> ``[256, 256]`` fp32 stored straight to DDR."""
    torch.manual_seed(0)
    a = torch.randn(256, k, dtype=torch.float32)
    b = torch.randn(k, 256, dtype=torch.float32)
    out = torch.zeros((256, 256), dtype=torch.float32)
    return st.case(
        kernel,
        a,
        b,
        out,
        name=f"{kernel.__name__}_ddr_{pid}",
        memory_planner=planner,
        golden=lambda _: a @ b,
        rtol=1e-3,
        atol=1e-3,
    )


def _int_bias_case(planner, pid):
    """A2/A3 applies INT32 bias once across an INT8 M/N+K tiled GEMM. Exact."""
    torch.manual_seed(10)
    a = torch.randint(-3, 4, (_BIAS_INT_M, _BIAS_INT_K), dtype=torch.int8)
    b = torch.randint(-3, 4, (_BIAS_INT_K, _BIAS_INT_N), dtype=torch.int8)
    bias = torch.randint(-20, 21, (1, _BIAS_INT_N), dtype=torch.int32)
    out = torch.zeros((_BIAS_INT_M, _BIAS_INT_N), dtype=torch.int32)
    return st.case(
        matmul_bias_a2a3_int_direct,
        a,
        b,
        bias,
        out,
        name=f"matmul_bias_a2a3_int_direct_{pid}",
        memory_planner=planner,
        golden=lambda _: a.int() @ b.int() + bias,
        rtol=0.0,
        atol=0.0,
    )


def _bias_scratch_case(planner, pid):
    """A biased producer tiled into Mat scratch for its sole matmul consumer."""
    torch.manual_seed(12)
    a = torch.randn(_BIAS_SCRATCH_M, _BIAS_SCRATCH_K, dtype=torch.bfloat16)
    b = torch.randn(_BIAS_SCRATCH_K, _BIAS_SCRATCH_N, dtype=torch.bfloat16)
    bias = torch.randn((1, _BIAS_SCRATCH_N), dtype=torch.float32)
    e = torch.randn(_BIAS_SCRATCH_N, _BIAS_SCRATCH_OUT_N, dtype=torch.bfloat16)
    out = torch.zeros((_BIAS_SCRATCH_M, _BIAS_SCRATCH_OUT_N), dtype=torch.float32)

    def golden(_):
        intermediate = (a.float() @ b.float() + bias).to(torch.bfloat16).float()
        return intermediate @ e.float()

    return st.case(
        matmul_bias_mn_k_scratch,
        a,
        b,
        bias,
        e,
        out,
        name=f"matmul_bias_mn_k_scratch_{pid}",
        memory_planner=planner,
        golden=golden,
        compare=st.rel_err_under(2e-2),
    )


# ---------------------------------------------------------------------------
# Tests — one per original test function, so the platform markers stay attached
# to exactly the cases they governed.
# ---------------------------------------------------------------------------


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(lambda planner, pid: _ddr_case(ddr_split_k, planner, pid, 128)),
    *_expand(lambda planner, pid: _ddr_case(ddr_full_k, planner, pid, 32)),
)
def test_ddr_direct_store(case_run):
    """``a @ b`` -> ``[256, 256]`` stored to DDR (direct-store); split-K (K=128) and
    full-K (K=32).  Run under all three planners: the oversized grid reuses the L0C
    accumulator across output tiles, but the Acc->GM ``tile.store`` drain WAR is synced
    correctly by ptoas, so oversized direct-store works under PTOAS too."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(*_expand(_int_bias_case))
def test_matmul_bias_a2a3_int_direct_store(case_run):
    """A2/A3 applies INT32 bias once across an INT8 M/N+K tiled GEMM."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(
    *_expand(
        lambda planner, pid: _bias_case(
            matmul_bias_mn_k_direct, "matmul_bias_mn_k_direct", planner, pid, _BIAS_M, _BIAS_K, _BIAS_N, 11
        )
    )
)
def test_matmul_bias_mn_k_direct_store(case_run):
    """Bias is applied once while M/N sub-tiles complete a split-K reduction."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(*_expand(_bias_scratch_case, _without_ptoas(_PLANNERS, "mat_scratch_bias")))
def test_matmul_bias_mn_k_mat_scratch(case_run):
    """A biased producer is tiled into Mat scratch for its sole matmul consumer."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(
    *_expand(
        lambda planner, pid: _bias_case(
            matmul_bias_mn_boundary_direct,
            "matmul_bias_mn_boundary_direct",
            planner,
            pid,
            _BIAS_BOUNDARY_M,
            _BIAS_BOUNDARY_K,
            _BIAS_BOUNDARY_N,
            13,
        ),
        _without_ptoas(_PLANNERS, "mn_boundaries"),
    )
)
def test_matmul_bias_mn_boundaries(case_run):
    """Partial M/N tiles preserve the logical Bias and output regions."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(
    *_expand(
        lambda planner, pid: _bias_case(
            matmul_bias_nondivisor_k_tail,
            "matmul_bias_nondivisor_k_tail",
            planner,
            pid,
            _BIAS_PEEL_M,
            _BIAS_PEEL_K,
            _BIAS_PEEL_N,
            14,
        )
    )
)
def test_matmul_bias_nondivisor_k_tail(case_run):
    """The peeled final K block accumulates without applying bias twice."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(
    *_expand(
        lambda planner, pid: _bias_case(
            matmul_bias_m_only_bias_resident,
            "matmul_bias_m_only_bias_resident",
            planner,
            pid,
            _BIAS_M_ONLY_M,
            _BIAS_M_ONLY_K,
            _BIAS_M_ONLY_N,
            15,
        )
    )
)
def test_matmul_bias_m_only_bias_resident(case_run):
    """M-only tiling reuses a full Bias-resident source without subwindowing."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(
        lambda planner, pid: _int_acc_case(
            matmul_acc_mn_issue_2232,
            "matmul_acc_mn_issue_2232",
            planner,
            pid,
            _ACC_M,
            _ACC_K,
            _ACC_N_TOTAL,
            0,
        )
    )
)
def test_loop_carried_matmul_acc_mn_tiling(case_run):
    """Issue #2232: each output tile must finish all eight source K blocks.

    The logical ``[16, 1152]`` INT32 result is only 72 KiB, but its physical
    32-row L0C footprint is 144 KiB. Run all planners and compare exactly:
    integer accumulation has no numerical tolerance."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(
        lambda planner, pid: _int_acc_case(
            matmul_acc_mn_boundaries,
            "matmul_acc_mn_boundaries",
            planner,
            pid,
            _BOUNDARY_M,
            _BOUNDARY_K,
            _BOUNDARY_N,
            1,
        )
    )
)
def test_loop_carried_matmul_acc_both_mn_boundaries(case_run):
    """General #2232 rewrite: exact INT8->INT32 split-K with partial tiles on both
    output axes, under every memory planner."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(
        lambda planner, pid: _int_acc_case(
            matmul_acc_n_boundary_retiles_k,
            "matmul_acc_n_boundary_retiles_k",
            planner,
            pid,
            _BOUNDARY_M,
            _COMPOSE_K,
            _BOUNDARY_N,
            2,
        ),
        _N_BOUNDARY_RETILES_K_PLANNERS,
    )
)
def test_loop_carried_matmul_acc_n_boundary_retiles_k(case_run):
    """A padded N tail remains valid through secondary inner-K tiling."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(
        lambda planner, pid: _chained_case(
            mat_split_k, "mat_split_k_scratch", planner, pid, 256, 192, 256, 64, 2e-2
        )
    ),
    *_expand(
        lambda planner, pid: _chained_case(
            mat_full_k, "mat_full_k_scratch", planner, pid, 256, 32, 256, 64, 2e-2
        )
    ),
)
def test_mat_scratch(case_run):
    """``(a @ b) @ e`` with a bf16 ``[256, 256]`` intermediate kept on-chip in an
    L1/Mat scratch (Acc->Mat ``pto.tinsert``); split-K K=192 and full-K K=32.

    Run under all three planners.  The PTOAS variants provide regression coverage
    for #1995: the chained consumer's K-reduction accumulator if-phi must reuse
    the dominating accumulator handle so all partial sums land in one L0C buffer.

    K=192 is the common cross-planner split point: all planners choose an
    output-stationary producer with k=64, so its L0 buffers pack against the
    consumer's. K=128 is planner-dependent (PyPTO splits while PTOAS can keep full K)
    and can select a monolithic A/B-stationary buffer that the legacy PYPTO
    allocator cannot pack against the consumer's pipelined buffers. This case
    remains output-stationary under every planner: PYPTO enforces the issue-1908
    guard, while DSA_RP/PTOAS reach the same chooser result without the guard.

    Operands are bf16 and the on-chip intermediate is bf16 — the cube's FIXPIPE
    writeback to L1 downcasts the f32 accumulator, which is also the cube's native
    operand precision. The golden models that downcast; the comparison is a global
    relative norm because cancellation-near-zero elements make per-element
    ``allclose`` unstable for this chained reduction."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *_expand(
        lambda planner, pid: _chained_case(
            fits_l0c_full_k, "fits_l0c_full_k", planner, pid, 128, 64, 128, 64, 5e-2
        )
    ),
    *_expand(
        lambda planner, pid: _chained_case(
            fits_l0c_split_k, "fits_l0c_split_k", planner, pid, 128, 512, 128, 64, 5e-2
        )
    ),
)
def test_fits_l0c_cast_fold(case_run):
    """``(a @ b) @ e`` with a ``[128, 128]`` intermediate that *fits* L0c (no M/N
    tiling): the autotiler folds ``pl.cast`` into a single full-window Acc->Mat
    ``pto.tinsert`` (cube downcast) rather than a Vector ``pto.tcvt``. full-K (K=64,
    no K-loop) and split-K (K=512, K-loop). Same bf16 FIXPIPE golden as Mat-scratch.

    Run under all three planners: because the intermediate fits L0c there is exactly ONE
    Acc->Mat assemble (no cross-tile L0C reuse and no drain/MAD WAR fence).

    On-device proof that the fold is numerically correct (the FIXPIPE bf16 rounding
    matches the reference) AND that it compiles — the un-folded Vector cast overflows
    the Vec buffer at this ``[128, 128]`` shape.

    The bound is looser (5e-2) than Mat-scratch: a bf16 ``(a @ b) @ e`` chain has
    near-zero cancellation elements where the absolute bf16 rounding error (~0.7 on
    operand magnitudes of ~500) dwarfs the small true value, so a per-element atol
    fails on a numerically-correct result. K=512 makes the intermediate magnitudes
    large enough to bite; K=64 happens to pass allclose, but both use one metric."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
