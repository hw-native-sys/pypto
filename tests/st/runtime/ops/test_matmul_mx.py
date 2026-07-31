# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime test for MX matmul.

The codegen unit tests verify the emitted ``tmatmul_mx`` and
``tget_scale_addr`` instructions. This test additionally executes the base and
accumulating forms on real Ascend950 hardware and compares their FP32 outputs
with torch.
"""

import pypto.language as pl
import pytest
import torch

M, K, N = 16, 64, 32
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _matmul_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
) -> torch.Tensor:
    """Compute MXFP8 matmul with logical per-32-element E8M0 scales."""
    k_group = torch.arange(K) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - 127)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - 127)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


@pl.jit
def matmul_mx_onboard(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Run base and accumulating MX matmul with shared GM operands."""
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.move(
            pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.load(a_scale, [0, 0], [M, K // 32], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.LeftScale,
        )
        rhs = pl.move(
            pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Right,
        )
        rhs_scale = pl.move(
            pl.load(b_scale, [0, 0], [K // 32, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
        out_acc = pl.store(accumulated, [0, 0], out_acc)
    return out, out_acc


@pytest.mark.platforms("a5")
class TestMatmulMxOnBoard:
    """Numerical execution coverage for the Ascend950-only MX matmul path."""

    def test_matmul_mx_onboard(self, test_config):
        matmul_mx_onboard._cache.clear()

        generator = torch.Generator().manual_seed(19)
        a = torch.randint(-2, 3, (M, K), generator=generator).to(torch.float8_e4m3fn)
        b = torch.randint(-2, 3, (K, N), generator=generator).to(torch.float8_e4m3fn)

        # E8M0 codes [126, 130) represent reproducible random scales in
        # {0.5, 1, 2, 4}. Keep logical scales for the golden and pass packed
        # physical buffers to the kernel.
        a_scale_codes = torch.randint(126, 130, (M, K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_scale_codes = torch.randint(126, 130, (K // MX_GROUP_SIZE, N), generator=generator).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)

        base = _matmul_mx_golden(a, a_scale_codes, b, b_scale_codes)
        out = torch.zeros_like(base)
        out_acc = torch.zeros_like(base)

        if test_config.codegen_only:
            matmul_mx_onboard.compile(a, a_scale, b, b_scale, out, out_acc, config=test_config)
            return

        matmul_mx_onboard(a, a_scale, b, b_scale, out, out_acc, config=test_config)

        torch.testing.assert_close(out, base, rtol=0, atol=0)
        torch.testing.assert_close(out_acc, 2 * base, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
