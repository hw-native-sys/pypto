# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 0: chunk packed scale concat vs full MX_A_ZZ (A5).

A — full ``quant_mx`` once → GM → full ``MX_A_ZZ`` view → ``matmul_mx``
B — two ``quant_mx(K=64)`` scales concatenated → full ``MX_A_ZZ`` view → ``matmul_mx``
C — same concat buffer → per-chunk ``view([M,2])`` + ``matmul_mx`` / ``matmul_mx_acc``
D — host-only: ``_pack_a_scale`` full vs ``torch.cat`` of per-chunk packs (UT)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

from tests.st.runtime.ops.test_matmul_mx import (
    MX_GROUP_SIZE,
    _JitMxTestCase,
    _exact_quantizable_matrix,
    _pack_a_scale,
)

P0_M, P0_K, P0_N = 32, 128, 32
P0_K_CHUNK = 64
P0_K_CHUNKS = P0_K // P0_K_CHUNK
P0_G = P0_M * P0_K // MX_GROUP_SIZE
P0_G_CHUNK = P0_M * P0_K_CHUNK // MX_GROUP_SIZE


@pl.jit.incore
def _p0_quant_full(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[P0_M, P0_K], pl.FP8E4M3FN]],
    a_s_gm: pl.Out[pl.Tensor[[1, P0_G], pl.FP8E8M0]],
):
    a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [P0_M, P0_K]), layout=pl.MX_A_ZZ)
    a_q_gm = pl.store(a_q, [0, 0], a_q_gm)
    a_s_gm = pl.store(a_s, [0, 0], a_s_gm)
    return a_q_gm, a_s_gm


@pl.jit.incore
def _p0_quant_concat(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[P0_M, P0_K], pl.FP8E4M3FN]],
    a_s_gm: pl.Out[pl.Tensor[[1, P0_G], pl.FP8E8M0]],
):
    """Per K=64 quant; concatenate packed scales; stitch FP8 data."""
    for chunk, (a_q_iter, a_s_iter) in pl.range(
        0, P0_K_CHUNKS, init_values=(a_q_gm, a_s_gm)
    ):
        k0 = chunk * P0_K_CHUNK
        a_q, a_s = pl.quant_mx(
            pl.load(a, [0, k0], [P0_M, P0_K_CHUNK]),
            layout=pl.MX_A_ZZ,
        )
        a_q_stored = pl.store(a_q, [0, k0], a_q_iter)
        a_s_stored = pl.store(a_s, [0, chunk * P0_G_CHUNK], a_s_iter)
        a_q_gm, a_s_gm = pl.yield_(a_q_stored, a_s_stored)
    return a_q_gm, a_s_gm


@pl.jit.incore
def _p0_mm_full_view(
    a_q_gm: pl.Tensor[[P0_M, P0_K], pl.FP8E4M3FN],
    a_s_gm: pl.Tensor[[1, P0_G], pl.FP8E8M0],
    b: pl.Tensor[[P0_K, P0_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[P0_K // MX_GROUP_SIZE, P0_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[P0_M, P0_N], pl.FP32]],
):
    a_s_mx = pl.tensor.view(a_s_gm, [P0_M, P0_K // MX_GROUP_SIZE], layout=pl.MX_A_ZZ)
    lhs = pl.move(
        pl.load(a_q_gm, [0, 0], [P0_M, P0_K], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s_mx, [0, 0], [P0_M, P0_K // MX_GROUP_SIZE], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b, [0, 0], [P0_K, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_scale, [0, 0], [P0_K // MX_GROUP_SIZE, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    out = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)
    return out


@pl.jit.incore
def _p0_quant_chunk_scales(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[P0_M, P0_K], pl.FP8E4M3FN]],
    a_s0_gm: pl.Out[pl.Tensor[[1, P0_G_CHUNK], pl.FP8E8M0]],
    a_s1_gm: pl.Out[pl.Tensor[[1, P0_G_CHUNK], pl.FP8E8M0]],
):
    """Per K=64 quant into separate scale GM slots (no cross-chunk ZZ)."""
    a_q0, a_s0 = pl.quant_mx(pl.load(a, [0, 0], [P0_M, P0_K_CHUNK]), layout=pl.MX_A_ZZ)
    a_q1, a_s1 = pl.quant_mx(
        pl.load(a, [0, P0_K_CHUNK], [P0_M, P0_K_CHUNK]),
        layout=pl.MX_A_ZZ,
    )
    a_q_gm = pl.store(a_q0, [0, 0], a_q_gm)
    a_q_gm = pl.store(a_q1, [0, P0_K_CHUNK], a_q_gm)
    a_s0_gm = pl.store(a_s0, [0, 0], a_s0_gm)
    a_s1_gm = pl.store(a_s1, [0, 0], a_s1_gm)
    return a_q_gm, a_s0_gm, a_s1_gm


@pl.jit.incore
def _p0_mm_chunk_scales(
    a_q_gm: pl.Tensor[[P0_M, P0_K], pl.FP8E4M3FN],
    a_s0_gm: pl.Tensor[[1, P0_G_CHUNK], pl.FP8E8M0],
    a_s1_gm: pl.Tensor[[1, P0_G_CHUNK], pl.FP8E8M0],
    b: pl.Tensor[[P0_K, P0_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[P0_K // MX_GROUP_SIZE, P0_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[P0_M, P0_N], pl.FP32]],
):
    """Independent per-chunk MX_A_ZZ views + matmul_mx / matmul_mx_acc."""
    g_chunk = P0_K_CHUNK // MX_GROUP_SIZE
    a_s0 = pl.tensor.view(a_s0_gm, [P0_M, g_chunk], layout=pl.MX_A_ZZ)
    a_s1 = pl.tensor.view(a_s1_gm, [P0_M, g_chunk], layout=pl.MX_A_ZZ)
    lhs = pl.move(
        pl.load(a_q_gm, [0, 0], [P0_M, P0_K_CHUNK], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s0, [0, 0], [P0_M, g_chunk], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b, [0, 0], [P0_K_CHUNK, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_scale, [0, 0], [g_chunk, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
    lhs = pl.move(
        pl.load(a_q_gm, [0, P0_K_CHUNK], [P0_M, P0_K_CHUNK], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s1, [0, 0], [P0_M, g_chunk], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b, [P0_K_CHUNK, 0], [P0_K_CHUNK, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_scale, [g_chunk, 0], [g_chunk, P0_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    out = pl.store(pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)
    return out


@pl.jit
def phase0_a_full_zz(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    b: pl.Tensor[[P0_K, P0_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[P0_K // MX_GROUP_SIZE, P0_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[P0_M, P0_N], pl.FP32]],
):
    a_q_gm = pl.create_tensor([P0_M, P0_K], dtype=pl.FP8E4M3FN)
    a_s_gm = pl.create_tensor([1, P0_G], dtype=pl.FP8E8M0)
    a_q_gm, a_s_gm = _p0_quant_full(a, a_q_gm, a_s_gm)
    out = _p0_mm_full_view(a_q_gm, a_s_gm, b, b_scale, out)
    return out


@pl.jit
def phase0_b_concat_full_view(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    b: pl.Tensor[[P0_K, P0_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[P0_K // MX_GROUP_SIZE, P0_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[P0_M, P0_N], pl.FP32]],
):
    a_q_gm = pl.create_tensor([P0_M, P0_K], dtype=pl.FP8E4M3FN)
    a_s_gm = pl.create_tensor([1, P0_G], dtype=pl.FP8E8M0)
    a_q_gm, a_s_gm = _p0_quant_concat(a, a_q_gm, a_s_gm)
    out = _p0_mm_full_view(a_q_gm, a_s_gm, b, b_scale, out)
    return out


@pl.jit
def phase0_c_concat_chunk_view(
    a: pl.Tensor[[P0_M, P0_K], pl.FP32],
    b: pl.Tensor[[P0_K, P0_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[P0_K // MX_GROUP_SIZE, P0_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[P0_M, P0_N], pl.FP32]],
):
    a_q_gm = pl.create_tensor([P0_M, P0_K], dtype=pl.FP8E4M3FN)
    a_s0_gm = pl.create_tensor([1, P0_G_CHUNK], dtype=pl.FP8E8M0)
    a_s1_gm = pl.create_tensor([1, P0_G_CHUNK], dtype=pl.FP8E8M0)
    a_q_gm, a_s0_gm, a_s1_gm = _p0_quant_chunk_scales(a, a_q_gm, a_s0_gm, a_s1_gm)
    out = _p0_mm_chunk_scales(a_q_gm, a_s0_gm, a_s1_gm, b, b_scale, out)
    return out


class _Phase0Case(_JitMxTestCase):
    __test__ = False
    entry: Any = None

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=1e-5, atol=1e-3), platform=platform)

    def define_tensors(self) -> list[TensorSpec]:
        a = _exact_quantizable_matrix(P0_M, P0_K)
        generator = torch.Generator().manual_seed(41)
        b = torch.randint(-2, 3, (P0_K, P0_N), generator=generator).to(torch.float8_e4m3fn)
        unit = torch.full(
            (P0_K // MX_GROUP_SIZE, P0_N),
            127,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)
        return [
            TensorSpec("a", [P0_M, P0_K], DataType.FP32, init_value=a),
            TensorSpec("b", [P0_K, P0_N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [P0_K // MX_GROUP_SIZE, P0_N],
                DataType.FP8E8M0,
                init_value=unit,
            ),
            TensorSpec("out", [P0_M, P0_N], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"].to(torch.float32))


class TestPhase0A(_Phase0Case):
    __test__ = False
    entry = phase0_a_full_zz

    def get_name(self) -> str:
        return "phase0_a_full_zz_32x128x32"


class TestPhase0B(_Phase0Case):
    __test__ = False
    entry = phase0_b_concat_full_view

    def get_name(self) -> str:
        return "phase0_b_concat_full_view_32x128x32"


class TestPhase0C(_Phase0Case):
    __test__ = False
    entry = phase0_c_concat_chunk_view

    def get_name(self) -> str:
        return "phase0_c_concat_chunk_view_32x128x32"


@pytest.mark.platforms("a5")
class TestPhase0ConcatVsFullZz:
    @pytest.mark.parametrize(
        "case_cls",
        [
            pytest.param(TestPhase0A, id="A-full-zz"),
            pytest.param(
                TestPhase0B,
                id="B-concat-full-view",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="chunk packed concat != full MX_A_ZZ; Phase0 host D + device B",
                ),
            ),
            pytest.param(TestPhase0C, id="C-concat-chunk-view"),
        ],
    )
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_phase0(self, test_runner, platform, case_cls):
        result = test_runner.run(case_cls(platform=platform))
        assert result.passed, f"{case_cls.__name__} failed: {result.error}"


def test_phase0_d_host_pack_concat_differs():
    """Host D: full MX_A_ZZ pack bytes != cat of per-K=64 packs."""
    m, k = P0_M, P0_K
    groups = k // MX_GROUP_SIZE
    exponents = (torch.arange(m * groups) % 4 - 1).reshape(m, groups)
    logical = (127 + exponents).to(torch.uint8)
    full = _pack_a_scale(logical).reshape(1, -1)
    chunks = [
        _pack_a_scale(logical[:, ki * 2 : ki * 2 + 2]).reshape(1, -1)
        for ki in range(k // P0_K_CHUNK)
    ]
    concat = torch.cat(chunks, dim=1)
    assert full.shape == concat.shape
    assert not torch.equal(full, concat), "concat unexpectedly matched full ZZ pack"
    assert int((full != concat).sum()) > 0
