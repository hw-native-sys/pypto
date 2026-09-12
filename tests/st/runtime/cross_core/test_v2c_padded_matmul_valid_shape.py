# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""V2C valid-shape framing regressions for padded matmul operands.

A vector-produced tile crosses to Cube with its logical rows and physical FIFO
column frame, while its ``valid_shape`` limits the matmul contraction. The
consumer must therefore mirror that transport frame and restore the logical
extent as metadata before matmul. Popping only the logical extent into the
wider Mat frame silently made only output row zero correct.

The four cases include an exact control and the three independently measured
failure discriminators. They use ordinary automatic mixed-kernel lowering; no
explicit pipe is authored by the test.
"""

from collections.abc import Callable

import pypto.language as pl
import pytest
import torch
from harness import st

M = 16
N = 64


def _make_padded_matmul(physical_k: int, valid_k: int):
    """Create one static mixed kernel whose vector-produced LHS has a padded K frame."""

    @pl.jit
    def padded_matmul(
        lhs: pl.Tensor[[M, physical_k], pl.FP16],
        bias: pl.Tensor[[M, physical_k], pl.FP16],
        rhs: pl.Tensor[[physical_k, N], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        for _ in pl.spmd(1, name_hint="padded_v2c", optimizations=[pl.split(pl.SplitMode.NONE)]):
            lhs_valid = pl.slice(lhs, [M, physical_k], [0, 0], valid_shape=[M, valid_k])
            bias_valid = pl.slice(bias, [M, physical_k], [0, 0], valid_shape=[M, valid_k])
            vector_lhs = pl.add(lhs_valid, bias_valid)
            rhs_valid = pl.slice(rhs, [physical_k, N], [0, 0], valid_shape=[valid_k, N])
            product = pl.matmul(vector_lhs, rhs_valid, out_dtype=pl.FP32)
            out[0:M, 0:N] = product
        return out

    return padded_matmul


_CASES: tuple[tuple[str, int, int], ...] = (
    ("exact_16x160", 160, 160),
    ("softmax_tail_160_96", 160, 96),
    ("smallest_disc_32_16", 32, 16),
    ("col16x1_in_16frame", 16, 1),
)


def _case(name: str, physical_k: int, valid_k: int):
    """Build deterministic inputs and the valid-K torch reference for one discriminator."""
    torch.manual_seed(0)
    lhs = torch.randn(M, physical_k, dtype=torch.float16)
    bias = torch.randn(M, physical_k, dtype=torch.float16)
    rhs = torch.randn(physical_k, N, dtype=torch.float16)
    out = torch.full((M, N), float("nan"), dtype=torch.float32)
    kernel = _make_padded_matmul(physical_k, valid_k)

    def golden(_: dict[str, torch.Tensor]) -> torch.Tensor:
        # pl.add receives two FP16 tiles, so the kernel rounds the sum to FP16
        # and only the contraction accumulates in FP32. Widening before the add
        # models a kernel PyPTO does not emit: it drifts by ~1.4e-2 at K=160,
        # well outside this tolerance, and worst on the unpadded control.
        return (lhs[:, :valid_k] + bias[:, :valid_k]).float() @ rhs[:valid_k].float()

    return st.case(
        kernel,
        lhs,
        bias,
        rhs,
        out,
        name=f"v2c_padded_matmul_{name}",
        golden=golden,
        rtol=2e-3,
        atol=2e-3,
    )


@st.cases(*(_case(*case) for case in _CASES))
def test_v2c_padded_matmul_valid_shape(case_run: Callable[..., object]):
    """The exact control and all padded V2C contractions match their valid-K references."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
