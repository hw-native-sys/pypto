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

The columns padding out that burst (columns 1..pad-1) are zeroed in place
(tile.fillpad_inplace) before they can reach pl.store, so the extra bytes that
ride along with the real value never carry stale Vec garbage into the
destination tensor's neighboring cells — this is exercised directly below by
checking the whole output tensor, not just its first column.

Mirrors the probe kernel from issue #3319: an unaligned [4, 42] window loaded
from a [32, 64] tensor, row_sum'd into a [4, 8] row_major result, and stored at
window (0, 0) of a [4, 8] output so the four sums land at out[0:4, 0].

Restricted to onboard platforms only (no simulator legs): out_layout=row_major
is new to PyPTO's A5 TileLib templates and has not been exercised on A2/A3
hardware before, so the a2a3 leg here is this feature's first on-board
A2/A3 validation.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy


def _row_init(shape):
    """Return a no-arg init callable (harness calls init_value with no args)."""
    return lambda: torch.randn(shape)


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
# Test cases
#
# compute_expected fills the *whole* [4, 8] output, not just column 0: columns
# 1..7 must come back zero (the fillpad_inplace padding), not whatever the Vec
# buffer held before. A naive "only check out[:, 0]" assertion would miss a
# regression that reintroduced stale pad bytes.
# =============================================================================


def _windowed_expected(tensors, reduce_fn):
    window = tensors["input_tensor"][0:4, 1:43]  # offset [0, 1], valid_shape [4, 42]
    expected = torch.zeros(4, 8, dtype=tensors["output"].dtype)
    expected[:, 0:1] = reduce_fn(window)
    return expected


class RowSumWindowColumnRowMajorFP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_sum_window_column_row_major_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_row_init([32, 64])),
            TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowSum_WindowColumn_RowMajor_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = _windowed_expected(tensors, lambda w: torch.sum(w, dim=1, keepdim=True))


class RowMaxWindowColumnRowMajorFP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_max_window_column_row_major_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_row_init([32, 64])),
            TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowMax_WindowColumn_RowMajor_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = _windowed_expected(tensors, lambda w: torch.amax(w, dim=1, keepdim=True))


class RowMinWindowColumnRowMajorFP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_min_window_column_row_major_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_row_init([32, 64])),
            TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowMin_WindowColumn_RowMajor_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = _windowed_expected(tensors, lambda w: torch.amin(w, dim=1, keepdim=True))


class RowProdWindowColumnRowMajorFP32(PTOTestCase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Product over 42 standard-normal elements needs a looser tolerance
        # than the default (same rationale as test_prod_reduction.py).
        self.config.rtol = 2e-2
        self.config.atol = 2e-2

    def get_name(self) -> str:
        return "row_prod_window_column_row_major_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        # Keep the running product well-conditioned, unlike the shared
        # standard-normal init used by the other three ops in this file.
        return [
            TensorSpec(
                "input_tensor",
                [32, 64],
                DataType.FP32,
                init_value=lambda: torch.rand(32, 64) * 0.4 + 0.8,
            ),
            TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_WindowColumn_RowMajor_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = _windowed_expected(tensors, lambda w: torch.prod(w, dim=1, keepdim=True))


# =============================================================================
# Tests
#
# Onboard only (a2a3, a5): row_major row-reduction results are new to PyPTO's
# A5 TileLib templates (issue #3319) and have no prior A2/A3 on-board coverage,
# so the a2a3 leg here is load-bearing -- it is how this feature's A2/A3
# hardware support actually gets checked, rather than assumed from the shared,
# architecture-agnostic IR verifier accepting the layout.
# =============================================================================


class TestRowReductionOutLayoutRowMajor:
    """row_sum/row_max/row_min/row_prod with out_layout=row_major (issue #3319)."""

    @pytest.mark.platforms("a2a3", "a5")
    def test_row_sum_window_column_fp32(self, test_runner):
        result = test_runner.run(RowSumWindowColumnRowMajorFP32())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a5")
    def test_row_max_window_column_fp32(self, test_runner):
        result = test_runner.run(RowMaxWindowColumnRowMajorFP32())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a5")
    def test_row_min_window_column_fp32(self, test_runner):
        result = test_runner.run(RowMinWindowColumnRowMajorFP32())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a5")
    def test_row_prod_window_column_fp32(self, test_runner):
        result = test_runner.run(RowProdWindowColumnRowMajorFP32())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
