# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test row_sum/row_max/row_min/row_prod with out_layout=row_major (issue #3319).

Row reductions default to a col_major [rows, 1] result. pl.store treats that as
one linear burst, so storing it into a window of a wider ND tensor (e.g. the
window-column [M, 1] slice of an [M, N] output with N > 1) lands the reduced
values packed into one row instead of down the window's column. Passing
out_layout=pl.TileLayout.row_major instead produces a [rows, pad] result
(pad = one 32-byte Vec burst for the element dtype) with the true value in
column 0; pl.store then bursts one row at a time and honors the destination
window's row stride, landing the values in the intended column.

Columns 1..pad-1 of that burst are NOT currently guaranteed zero: TSTORE
bursts the whole physical row regardless of what the tile's type claims is
valid, and nothing zeroes those bytes yet (tracked as a follow-up to #3319).
This test uses a custom Case.compare (see _compare_column0_only) that checks
only column 0 -- the value this PR actually guarantees -- rather than the
default elementwise comparator, which would also compare the unpredictable
padding columns.

Mirrors the probe kernel from issue #3319: an unaligned [4, 42] window loaded
from a [32, 64] tensor, row_sum'd into a [4, 8] row_major result, and stored at
window (0, 0) of a [4, 8] output so the four sums land at out[0:4, 0].

Restricted to onboard platforms only (no simulator legs): out_layout=row_major
is new to PyPTO's A5 TileLib templates and has not been exercised on A2/A3
hardware before, so the a2a3 leg here is this feature's first on-board
A2/A3 validation.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec


def _row_init(shape):
    """Return a no-arg init callable (harness calls init_value with no args)."""
    return lambda: torch.randn(shape)


def _compare_column0_only(rtol: float, atol: float):
    """Build a Case.compare that checks only column 0 of each output.

    columns 1..pad-1 of a row_major result are not currently guaranteed zero
    (see module docstring); restricting the comparison to column 0 keeps this
    test meaningful -- it still proves the real reduction value lands in the
    right destination cell -- without asserting something this PR does not
    guarantee about the padding bytes.
    """

    def compare(actual, expected):
        for name, got in actual.items():
            ref = expected[name]
            assert torch.allclose(got[:, 0:1], ref[:, 0:1], rtol=rtol, atol=atol), (
                f"{name}: column 0 mismatch\ngot:\n{got[:, 0:1]}\nexpected:\n{ref[:, 0:1]}"
            )

    return compare


# =============================================================================
# Programs — mirrors issue #3319's probe kernel exactly (row_sum; row_max/
# row_min/row_prod below reuse the same shapes to cover the whole row-reduction
# family since they share DeduceTileRowReductionType).
# =============================================================================


@pl.program
class RowSum_WindowColumn_RowMajor_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        # Unaligned [4, 42] window at offset [0, 1]; physical tile [4, 48] (no
        # forced row padding -- that padding was only ever needed to satisfy
        # the col_major result's 32B row alignment, which row_major drops).
        tile: pl.Tile[[4, 48], pl.FP32] = pl.load(input_tensor, [0, 1], [4, 48], valid_shape=[4, 42])
        tmp: pl.Tile[[4, 48], pl.FP32] = pl.tile.create(
            [4, 48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[4, 8], pl.FP32] = pl.tile.row_sum(tile, tmp, out_layout=pl.TileLayout.row_major)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowMax_WindowColumn_RowMajor_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        tile: pl.Tile[[4, 48], pl.FP32] = pl.load(input_tensor, [0, 1], [4, 48], valid_shape=[4, 42])
        tmp: pl.Tile[[4, 48], pl.FP32] = pl.tile.create(
            [4, 48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[4, 8], pl.FP32] = pl.tile.row_max(tile, tmp, out_layout=pl.TileLayout.row_major)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowMin_WindowColumn_RowMajor_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        tile: pl.Tile[[4, 48], pl.FP32] = pl.load(input_tensor, [0, 1], [4, 48], valid_shape=[4, 42])
        tmp: pl.Tile[[4, 48], pl.FP32] = pl.tile.create(
            [4, 48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[4, 8], pl.FP32] = pl.tile.row_min(tile, tmp, out_layout=pl.TileLayout.row_major)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowProd_WindowColumn_RowMajor_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        tile: pl.Tile[[4, 48], pl.FP32] = pl.load(input_tensor, [0, 1], [4, 48], valid_shape=[4, 42])
        tmp: pl.Tile[[4, 48], pl.FP32] = pl.tile.create(
            [4, 48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[4, 8], pl.FP32] = pl.tile.row_prod(tile, tmp, out_layout=pl.TileLayout.row_major)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


# =============================================================================
# Goldens and cases
# =============================================================================


def _windowed_golden(reduce_fn):
    """Build a golden that computes the probe's [4, 42] window reduction.

    Only column 0 of the result is ever compared (see _compare_column0_only),
    so the pad columns 1..7 written here are never checked; they are filled
    with zero purely so the tensor has a well-defined shape to persist.
    """

    def golden(tensors):
        window = tensors["input_tensor"][0:4, 1:43]  # offset [0, 1], valid_shape [4, 42]
        expected = torch.zeros(4, 8, dtype=tensors["output"].dtype)
        expected[:, 0:1] = reduce_fn(window)
        tensors["output"][:] = expected

    return golden


_ROW_SUM_TENSORS = [
    TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_row_init([32, 64])),
    TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
]

_ROW_PROD_TENSORS = [
    # Keep the running product well-conditioned, unlike the shared
    # standard-normal init used by the other three ops.
    TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=lambda: torch.rand(32, 64) * 0.4 + 0.8),
    TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
]

_ROW_SUM_CASE = st.case(
    RowSum_WindowColumn_RowMajor_FP32,
    name="row_sum_window_column_row_major_fp32",
    tensors=_ROW_SUM_TENSORS,
    golden=_windowed_golden(lambda w: torch.sum(w, dim=1, keepdim=True)),
    compare=_compare_column0_only(rtol=1e-5, atol=1e-5),
)

_ROW_MAX_CASE = st.case(
    RowMax_WindowColumn_RowMajor_FP32,
    name="row_max_window_column_row_major_fp32",
    tensors=_ROW_SUM_TENSORS,
    golden=_windowed_golden(lambda w: torch.amax(w, dim=1, keepdim=True)),
    compare=_compare_column0_only(rtol=1e-5, atol=1e-5),
)

_ROW_MIN_CASE = st.case(
    RowMin_WindowColumn_RowMajor_FP32,
    name="row_min_window_column_row_major_fp32",
    tensors=_ROW_SUM_TENSORS,
    golden=_windowed_golden(lambda w: torch.amin(w, dim=1, keepdim=True)),
    compare=_compare_column0_only(rtol=1e-5, atol=1e-5),
)

_ROW_PROD_CASE = st.case(
    RowProd_WindowColumn_RowMajor_FP32,
    name="row_prod_window_column_row_major_fp32",
    tensors=_ROW_PROD_TENSORS,
    golden=_windowed_golden(lambda w: torch.prod(w, dim=1, keepdim=True)),
    # Product over 42 standard-normal-scale elements needs a looser tolerance
    # than the default (same rationale as test_prod_reduction.py).
    compare=_compare_column0_only(rtol=2e-2, atol=2e-2),
)


# =============================================================================
# Tests
#
# Onboard only (a2a3, a5): row_major row-reduction results are new to PyPTO's
# A5 TileLib templates (issue #3319) and have no prior A2/A3 on-board coverage,
# so the a2a3 leg here is load-bearing -- it is how this feature's A2/A3
# hardware support actually gets checked, rather than assumed from the shared,
# architecture-agnostic IR verifier accepting the layout.
# =============================================================================


@pytest.mark.platforms("a2a3", "a5")
@st.cases(_ROW_SUM_CASE)
def test_row_sum_window_column_fp32(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a5")
@st.cases(_ROW_MAX_CASE)
def test_row_max_window_column_fp32(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a5")
@st.cases(_ROW_MIN_CASE)
def test_row_min_window_column_fp32(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a5")
@st.cases(_ROW_PROD_CASE)
def test_row_prod_window_column_fp32(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
