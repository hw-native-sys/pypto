# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 numerical coverage for phased and biased MX GEMV and GEMM."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec

_REQUIRED_TORCH_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP8/E8M0 dtypes required", allow_module_level=True)

M, K, N = 16, 128, 128
MX_GROUP_SIZE = 32
SCALE_BLOCK_ROWS = 2
SCALE_BLOCK_COLS = 16


@pl.jit.incore
def _gemv_mx_kernel(
    a: pl.Tensor[[1, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, K // MX_GROUP_SIZE], pl.FP8E8M0, pl.MX_A_ND],
    b: pl.Tensor[[K, N], pl.FP8E5M2],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    bias: pl.Tensor[[1, N], pl.FP32],
    out_acc: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    out_bias: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [1, 512], valid_shape=[1, K])
    lhs_scale = pl.load(a_scale, [0, 0], [1, K // MX_GROUP_SIZE])
    rhs = pl.load(b, [0, 0], [K, N])
    rhs_scale = pl.load(b_scale, [0, 0], [K // MX_GROUP_SIZE, N])
    bias_tile = pl.load(bias, [0, 0], [1, N])
    acc = pl.gemv_mx(lhs, lhs_scale, rhs, rhs_scale, acc_phase=pl.AccPhase.Partial)
    acc = pl.gemv_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, acc_phase=pl.AccPhase.Final)
    out_acc = pl.store(acc, [0, 0], out_acc, st_phase=pl.STPhase.Final)
    biased = pl.gemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile)
    out_bias = pl.store(biased, [0, 0], out_bias)
    return out_acc, out_bias


@pl.jit
def _gemv_mx_entry(
    a: pl.Tensor[[1, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, K // MX_GROUP_SIZE], pl.FP8E8M0, pl.MX_A_ND],
    b: pl.Tensor[[K, N], pl.FP8E5M2],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    bias: pl.Tensor[[1, N], pl.FP32],
    out_acc: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    out_bias: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    return _gemv_mx_kernel(a, a_scale, b, b_scale, bias, out_acc, out_bias)


@pl.jit.incore
def _matmul_mx_kernel(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP_SIZE], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E5M2],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    bias: pl.Tensor[[1, N], pl.FP32],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [M, K])
    lhs_scale = pl.load(a_scale, [0, 0], [M, K // MX_GROUP_SIZE])
    rhs = pl.load(b, [0, 0], [K, N])
    rhs_scale = pl.load(b_scale, [0, 0], [K // MX_GROUP_SIZE, N])
    bias_tile = pl.load(bias, [0, 0], [1, N])
    acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, acc_phase=pl.AccPhase.Partial)
    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, acc_phase=pl.AccPhase.Final)
    out_acc = pl.store(acc, [0, 0], out_acc, st_phase=pl.STPhase.Final)
    biased = pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile, acc_phase=pl.AccPhase.Final)
    out_bias = pl.store(biased, [0, 0], out_bias, st_phase=pl.STPhase.Final)
    return out_acc, out_bias


@pl.jit
def _matmul_mx_entry(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP_SIZE], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E5M2],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    bias: pl.Tensor[[1, N], pl.FP32],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    return _matmul_mx_kernel(a, a_scale, b, b_scale, bias, out_acc, out_bias)


def _mx_specs(m):
    groups = K // MX_GROUP_SIZE
    a = (torch.arange(m * K).reshape(m, K) % 13 - 6).to(torch.float8_e4m3fn)
    b = (torch.arange(K * N).reshape(K, N) % 11 - 5).to(torch.float8_e5m2)
    a_scale = (torch.arange(m * groups).reshape(m, groups) % groups + 125).to(torch.uint8)
    if m != 1:
        a_scale = (
            a_scale.reshape(
                m // SCALE_BLOCK_COLS, SCALE_BLOCK_COLS, groups // SCALE_BLOCK_ROWS, SCALE_BLOCK_ROWS
            )
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(m, groups)
        )
    b_codes = (torch.arange(groups * N).reshape(groups, N) % 5 + 125).to(torch.uint8)
    b_scale = (
        b_codes.reshape(groups // SCALE_BLOCK_ROWS, SCALE_BLOCK_ROWS, N // SCALE_BLOCK_COLS, SCALE_BLOCK_COLS)
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(groups, N)
    )
    bias = (torch.arange(N, dtype=torch.float32) % 7 - 3).reshape(1, N)
    return [
        TensorSpec("a", [m, K], DataType.FP8E4M3FN, init_value=a),
        TensorSpec("a_scale", [m, groups], DataType.FP8E8M0, init_value=a_scale.view(torch.float8_e8m0fnu)),
        TensorSpec("b", [K, N], DataType.FP8E5M2, init_value=b),
        TensorSpec("b_scale", [groups, N], DataType.FP8E8M0, init_value=b_scale.view(torch.float8_e8m0fnu)),
        TensorSpec("bias", [1, N], DataType.FP32, init_value=bias),
        TensorSpec("out_acc", [m, N], DataType.FP32, is_output=True),
        TensorSpec("out_bias", [m, N], DataType.FP32, is_output=True),
    ]


def _mx_golden(tensors):
    groups = K // MX_GROUP_SIZE
    b_codes = (
        tensors["b_scale"]
        .view(torch.uint8)
        .reshape(N // SCALE_BLOCK_COLS, groups // SCALE_BLOCK_ROWS, SCALE_BLOCK_COLS, SCALE_BLOCK_ROWS)
        .permute(1, 3, 0, 2)
        .reshape(groups, N)
    )
    a_codes = tensors["a_scale"].view(torch.uint8)
    m = a_codes.shape[0]
    if m != 1:
        a_codes = (
            a_codes.reshape(
                m // SCALE_BLOCK_COLS, groups // SCALE_BLOCK_ROWS, SCALE_BLOCK_COLS, SCALE_BLOCK_ROWS
            )
            .permute(0, 2, 1, 3)
            .reshape(m, groups)
        )
    k_groups = torch.arange(K) // MX_GROUP_SIZE
    a_decoded = tensors["a"].float() * torch.exp2(a_codes.float() - 127)[:, k_groups]
    b_decoded = tensors["b"].float() * torch.exp2(b_codes.float() - 127)[k_groups, :]
    base = a_decoded @ b_decoded
    return {"out_acc": 2 * base, "out_bias": base + tensors["bias"]}


@pytest.mark.platforms("a5")
@st.cases(st.case(_gemv_mx_entry, tensors=_mx_specs(1), golden=_mx_golden, rtol=0.0, atol=0.0))
def test_gemv_mx_phased_and_biased(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a5")
@st.cases(st.case(_matmul_mx_entry, tensors=_mx_specs(M), golden=_mx_golden, rtol=0.0, atol=0.0))
def test_matmul_mx_phased_and_biased(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
