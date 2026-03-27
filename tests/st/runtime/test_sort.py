# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test sort32 operation for fixed 32-element block sorting.

TSORT32 sorts 32-element blocks descending. The result is written to dst as
interleaved (value, index) pairs:
  - float: dst cols = src cols × 2, layout [val_f32, idx_u32, val_f32, idx_u32, ...]
  - idx is INPUT only — it is NOT modified by the sort.
  - Sorted indices are stored inside dst at odd positions (u32 bits in f32 memory).

To read back indices as integers on the host side, use:
    values, indices = extract_sort32_results(output_f32)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


def extract_sort32_results(output_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract values and indices from interleaved sort32 output.

    The hardware stores (value_f32, index_u32) pairs interleaved in f32 memory.
    This function splits them and reinterprets index bits as int32.

    Args:
        output_f32: [rows, cols*2] f32 tensor with interleaved (value, index) pairs

    Returns:
        values:  [rows, cols] f32  — sorted values (descending)
        indices: [rows, cols] int32 — original positions of sorted elements
    """
    values = output_f32[:, 0::2]
    indices = output_f32.view(torch.int32)[:, 1::2]
    return values, indices


# --- Programs ---


@pl.program
class Sort32FP32Program:
    """Sort 32-element blocks of FP32 data."""

    @pl.function(type=pl.FunctionType.InCore)
    def sort32_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        output: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    ) -> pl.Tensor[[8, 64], pl.FP32]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(
            src_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        idx_tile: pl.Tile[[8, 32], pl.UINT32] = pl.load(
            idx_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        sorted_tile: pl.Tile[[8, 64], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)
        out: pl.Tensor[[8, 64], pl.FP32] = pl.store(
            sorted_tile, offsets=[0, 0], output_tensor=output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        output: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    ) -> pl.Tensor[[8, 64], pl.FP32]:
        output = self.sort32_kernel(src_tensor, idx_tensor, output)
        return output


# --- Test Cases ---


# Pre-compute idx tensor: [0, 1, 2, ..., 31] per row (logical indices)
_IDX_TENSOR_FP32 = torch.arange(0, 32, dtype=torch.int32).unsqueeze(0).expand(8, -1).contiguous()


class Sort32FP32TestCase(PTOTestCase):
    """Test sort32 with FP32 data and PTO backend.

    dst layout for float: [val_f32, idx_u32_as_f32, val_f32, idx_u32_as_f32, ...]
    Sorted values at even positions, permuted indices (u32 bits) at odd positions.
    """

    def get_name(self) -> str:
        return "sort32_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.UINT32, init_value=_IDX_TENSOR_FP32),
            TensorSpec("output", [8, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Sort32FP32Program

    def compute_expected(self, tensors, params=None):
        """Expected: descending sort, interleaved [val, idx_as_f32, ...] layout."""
        src = tensors["src_tensor"]  # [8, 32]
        expected = torch.zeros(8, 64, dtype=torch.float32)
        for row in range(8):
            sorted_vals, sorted_indices = torch.sort(src[row], descending=True)
            idx_as_f32 = sorted_indices.int().view(torch.float32)
            expected[row, 0::2] = sorted_vals
            expected[row, 1::2] = idx_as_f32
        tensors["output"][:] = expected


# --- Tests ---


class TestSort:
    """Test suite for sort32 operations."""

    def test_sort32_fp32(self, test_runner):
        """Test sort32 with FP32 data: verify descending sort with index tracking.

        To manually inspect sorted indices from the interleaved output:
            values, indices = extract_sort32_results(output_f32)
            # values:  [8, 32] f32  — sorted descending
            # indices: [8, 32] int32 — original positions
        """
        test_case = Sort32FP32TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
