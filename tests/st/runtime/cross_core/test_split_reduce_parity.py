# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UP_DOWN split + row-reduction runtime parity probe (gh#1864).

``pl.split(SplitMode.UP_DOWN)`` fans a vector scope across two AIV subblocks
along the row axis. A scope whose body row-normalizes per token used to compile
cleanly and run without error but produce WRONG output under ``UP_DOWN`` (the
same scope under ``SplitMode.NONE`` is correct).

Root cause is NOT the reduction being split: ``row_sum`` collapses the *last*
axis (columns) while UP_DOWN splits dim 0 (rows), so each lane's row sums are
independent and lane-local -- no cross-lane combine is needed. The bug is the
column reshape that follows it: the ``[T_TILE, 1]`` reduction result is reshaped
to a ``[1, T_TILE]`` row vector for the reciprocal (the rms_norm column-layout
trick), which moves the split data (the rows) into the column dim.
``SplitVectorKernel`` halved the reduction to ``[8, 1]`` but did not migrate the
split axis through the reshape, leaving each lane reading a stale full-width
``[1, T_TILE]`` (only its half valid, the rest garbage -> 0 -> ``recip`` = inf).

Repro: per-token row-normalize ``y[i, :] = x[i, :] / sum_j x[i, j]``. A tiny
dummy matmul is present only to *engage* UP_DOWN (a pure-vector scope does not
trigger the split otherwise). ``T_TILE = 16`` so the ``[T_TILE, 1]`` FP32
reduction tile splits to ``[8, 1]`` = 32B and does NOT hit the separate
col-major 32B-align wall.

Two reduce forms x NONE/UP_DOWN:

  * ``direct``  : ``row_sum -> [T_TILE, 1]`` used directly. (A row-major
                  normalization pass still inserts the same ``[T_TILE, 1]`` ->
                  ``[1, T_TILE]`` -> ``[T_TILE, 1]`` column reshape that the
                  reciprocal needs, so this form hits the bug too.)
  * ``reshape`` : the same ``[T_TILE, 1] -> [1, T_TILE] -> [T_TILE, 1]`` reshape
                  written explicitly (the DeepSeek-V4 rms_norm pattern).

Both forms produce a column reshape, so both miscompiled under UP_DOWN before the
fix. ``SplitVectorKernel`` now migrates the split axis through the reshape (each
lane keeps its own half), so all four cases match the golden.
"""

import sys

import pypto.language as pl
import pytest
import torch

T_TILE = 16
C = 512
NTASK = 8
T = NTASK * T_TILE

# Dummy-matmul dims -- present only to engage the UP_DOWN split.
MM_M = 32
MM_K = 128
MM_N = 32


@pl.jit
def split_none_reshape(
    x: pl.Tensor[[T, C], pl.BF16],
    y: pl.Out[pl.Tensor[[T, C], pl.BF16]],
    dummy_out: pl.Out[pl.Tensor[[NTASK * MM_M, MM_N], pl.BF16]],
):
    for t in pl.spmd(NTASK, name_hint="reduce_repro", optimizations=[pl.split(pl.SplitMode.NONE)]):
        t0 = t * T_TILE
        acc = pl.matmul(x[0:MM_M, 0:MM_K], x[0:MM_N, 0:MM_K], b_trans=True, out_dtype=pl.FP32)
        dummy_out[t * MM_M : t * MM_M + MM_M, 0:MM_N] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        xc = pl.cast(x[t0 : t0 + T_TILE, 0:C], target_type=pl.FP32)
        rs = pl.row_sum(xc)  # [T_TILE, 1]
        rs_row = pl.reshape(rs, [1, T_TILE])  # -> [1, T_TILE]  (token in column dim)
        inv_row = pl.recip(rs_row)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])  # -> [T_TILE, 1]
        yc = pl.row_expand_mul(xc, inv_col)
        y[t0 : t0 + T_TILE, 0:C] = pl.cast(yc, target_type=pl.BF16, mode="rint")
    return y


@pl.jit
def split_updown_reshape(
    x: pl.Tensor[[T, C], pl.BF16],
    y: pl.Out[pl.Tensor[[T, C], pl.BF16]],
    dummy_out: pl.Out[pl.Tensor[[NTASK * MM_M, MM_N], pl.BF16]],
):
    for t in pl.spmd(NTASK, name_hint="reduce_repro", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        t0 = t * T_TILE
        acc = pl.matmul(x[0:MM_M, 0:MM_K], x[0:MM_N, 0:MM_K], b_trans=True, out_dtype=pl.FP32)
        dummy_out[t * MM_M : t * MM_M + MM_M, 0:MM_N] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        xc = pl.cast(x[t0 : t0 + T_TILE, 0:C], target_type=pl.FP32)
        rs = pl.row_sum(xc)
        rs_row = pl.reshape(rs, [1, T_TILE])
        inv_row = pl.recip(rs_row)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])
        yc = pl.row_expand_mul(xc, inv_col)
        y[t0 : t0 + T_TILE, 0:C] = pl.cast(yc, target_type=pl.BF16, mode="rint")
    return y


@pl.jit
def split_none_direct(
    x: pl.Tensor[[T, C], pl.BF16],
    y: pl.Out[pl.Tensor[[T, C], pl.BF16]],
    dummy_out: pl.Out[pl.Tensor[[NTASK * MM_M, MM_N], pl.BF16]],
):
    for t in pl.spmd(NTASK, name_hint="reduce_repro", optimizations=[pl.split(pl.SplitMode.NONE)]):
        t0 = t * T_TILE
        acc = pl.matmul(x[0:MM_M, 0:MM_K], x[0:MM_N, 0:MM_K], b_trans=True, out_dtype=pl.FP32)
        dummy_out[t * MM_M : t * MM_M + MM_M, 0:MM_N] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        xc = pl.cast(x[t0 : t0 + T_TILE, 0:C], target_type=pl.FP32)
        inv_col = pl.recip(pl.row_sum(xc))  # [T_TILE, 1] used directly
        yc = pl.row_expand_mul(xc, inv_col)
        y[t0 : t0 + T_TILE, 0:C] = pl.cast(yc, target_type=pl.BF16, mode="rint")
    return y


@pl.jit
def split_updown_direct(
    x: pl.Tensor[[T, C], pl.BF16],
    y: pl.Out[pl.Tensor[[T, C], pl.BF16]],
    dummy_out: pl.Out[pl.Tensor[[NTASK * MM_M, MM_N], pl.BF16]],
):
    for t in pl.spmd(NTASK, name_hint="reduce_repro", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        t0 = t * T_TILE
        acc = pl.matmul(x[0:MM_M, 0:MM_K], x[0:MM_N, 0:MM_K], b_trans=True, out_dtype=pl.FP32)
        dummy_out[t * MM_M : t * MM_M + MM_M, 0:MM_N] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        xc = pl.cast(x[t0 : t0 + T_TILE, 0:C], target_type=pl.FP32)
        inv_col = pl.recip(pl.row_sum(xc))
        yc = pl.row_expand_mul(xc, inv_col)
        y[t0 : t0 + T_TILE, 0:C] = pl.cast(yc, target_type=pl.BF16, mode="rint")
    return y


_KERNELS = {
    "none_reshape": split_none_reshape,
    "updown_reshape": split_updown_reshape,
    "none_direct": split_none_direct,
    "updown_direct": split_updown_direct,
}


def _golden_y(x: torch.Tensor) -> torch.Tensor:
    """Per-token row-normalize: y[i, :] = x[i, :] / sum_j x[i, j]."""
    xf = x.float()
    inv = xf.sum(-1, keepdim=True).reciprocal()
    return (xf * inv).to(torch.bfloat16)


def _run(kernel, test_config) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a repro kernel on device; return (actual_y, golden_y)."""
    kernel._cache.clear()
    torch.manual_seed(0)
    x = (torch.randn(T, C) * 0.5 + 2.0).to(torch.bfloat16)  # strictly positive -> stable recip
    y = torch.zeros(T, C, dtype=torch.bfloat16)
    dummy_out = torch.zeros(NTASK * MM_M, MM_N, dtype=torch.bfloat16)
    kernel(x, y, dummy_out, config=test_config)
    return y, _golden_y(x)


class TestSplitReduceParity:
    """gh#1864: a row reduction inside an UP_DOWN split scope must not miscompile."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("form", ["reshape", "direct"])
    def test_none_split_row_sum_is_correct(self, test_config, form):
        """Baseline: under SplitMode.NONE both reduce forms match the golden."""
        actual, golden = _run(_KERNELS[f"none_{form}"], test_config)
        torch.testing.assert_close(actual.float(), golden.float(), rtol=1.0 / 128, atol=1e-2)

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("form", ["reshape", "direct"])
    def test_updown_split_row_sum_matches_none(self, test_config, form):
        """UP_DOWN must equal the golden (gh#1864 regression).

        Pre-fix, the column reshape that feeds the reciprocal kept its stale
        full width after the split, so each lane read garbage columns and emitted
        inf. SplitVectorKernel now migrates the split axis through the reshape.
        """
        actual, golden = _run(_KERNELS[f"updown_{form}"], test_config)
        torch.testing.assert_close(actual.float(), golden.float(), rtol=1.0 / 128, atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
