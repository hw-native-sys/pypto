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

Also tests tile.gather_mask (mask-pattern form of pto.tgather).
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


@pl.program
class Sort32GatherFP32Program:
    """Sort 32-element blocks, then gather to separate values and indices."""

    @pl.function(type=pl.FunctionType.InCore)
    def sort32_gather_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        val_gather_idx: pl.Tensor[[8, 32], pl.INT32],
        idx_gather_idx: pl.Tensor[[8, 32], pl.INT32],
        gather_tmp: pl.Tensor[[8, 32], pl.INT32],
        val_output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
        idx_output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> tuple[pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[8, 32], pl.FP32]]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(
            src_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        idx_tile: pl.Tile[[8, 32], pl.UINT32] = pl.load(
            idx_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        sorted_tile: pl.Tile[[8, 64], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)

        val_gidx: pl.Tile[[8, 32], pl.INT32] = pl.load(
            val_gather_idx, offsets=[0, 0], shapes=[8, 32]
        )
        idx_gidx: pl.Tile[[8, 32], pl.INT32] = pl.load(
            idx_gather_idx, offsets=[0, 0], shapes=[8, 32]
        )
        tmp_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(
            gather_tmp, offsets=[0, 0], shapes=[8, 32]
        )

        val_tile: pl.Tile[[8, 32], pl.FP32] = pl.tile.gather(
            sorted_tile, val_gidx, tmp_tile
        )
        # Index bits are stored as raw uint32 in f32 memory by sort32.
        # Keep as FP32 — host will .view(torch.int32) to reinterpret bits.
        idx_tile_fp32: pl.Tile[[8, 32], pl.FP32] = pl.tile.gather(
            sorted_tile, idx_gidx, tmp_tile
        )

        val_out: pl.Tensor[[8, 32], pl.FP32] = pl.store(
            val_tile, offsets=[0, 0], output_tensor=val_output
        )
        idx_out: pl.Tensor[[8, 32], pl.FP32] = pl.store(
            idx_tile_fp32, offsets=[0, 0], output_tensor=idx_output
        )
        return val_out, idx_out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        val_gather_idx: pl.Tensor[[8, 32], pl.INT32],
        idx_gather_idx: pl.Tensor[[8, 32], pl.INT32],
        gather_tmp: pl.Tensor[[8, 32], pl.INT32],
        val_output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
        idx_output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> tuple[pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[8, 32], pl.FP32]]:
        val_output, idx_output = self.sort32_gather_kernel(
            src_tensor, idx_tensor, val_gather_idx, idx_gather_idx,
            gather_tmp, val_output, idx_output
        )
        return val_output, idx_output


@pl.program
class Sort32GatherMaskFP32Program:
    """Sort 32-element blocks, then extract values via P0101 gather_mask."""

    @pl.function(type=pl.FunctionType.InCore)
    def sort32_gather_mask_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(
            src_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        idx_tile: pl.Tile[[8, 32], pl.UINT32] = pl.load(
            idx_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        sorted_tile: pl.Tile[[8, 64], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)
        # P0101 selects every other element (stride=2): columns 0,2,4,...
        # sort32 layout is [val0, idx0, val1, idx1, ...], so P0101 extracts values.
        # Output shape: [8, 64/2] = [8, 32]
        gathered: pl.Tile[[8, 32], pl.FP32] = pl.tile.gather(
            sorted_tile, mask_pattern=pl.tile.MaskPattern.P0101
        )
        out: pl.Tensor[[8, 32], pl.FP32] = pl.store(
            gathered, offsets=[0, 0], output_tensor=output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.UINT32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        output = self.sort32_gather_mask_kernel(src_tensor, idx_tensor, output)
        return output



@pl.program
class GatherMaskFP32Program:
    """Gather elements using a fixed mask pattern (P1111 = all elements)."""

    @pl.function(type=pl.FunctionType.InCore)
    def gather_mask_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(
            src_tensor, offsets=[0, 0], shapes=[8, 32]
        )
        gathered: pl.Tile[[8, 32], pl.FP32] = pl.tile.gather(
            src_tile, mask_pattern=pl.tile.MaskPattern.P1111
        )
        out: pl.Tensor[[8, 32], pl.FP32] = pl.store(
            gathered, offsets=[0, 0], output_tensor=output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        output = self.gather_mask_kernel(src_tensor, output)
        return output


# --- Test Cases ---


# Pre-compute idx tensor: [0, 1, 2, ..., 31] per row (logical indices)
_IDX_TENSOR_FP32 = torch.arange(0, 32, dtype=torch.int32).unsqueeze(0).expand(8, -1).contiguous()

# Pre-compute gather index tensors for separating interleaved (val, idx) pairs.
# sort32 output layout: [val0, idx0, val1, idx1, ...] → 64 elements per row.
# pto.tgather uses FLAT element indices into the entire source tile, so each row
# needs a row-specific offset: row i uses base offset i*64.
_ROW_OFFSETS = (torch.arange(0, 8, dtype=torch.int32) * 64).unsqueeze(1)  # [8, 1]
_VAL_GATHER_IDX = (
    (torch.arange(0, 32, dtype=torch.int32) * 2).unsqueeze(0) + _ROW_OFFSETS
).contiguous()  # [8, 32]
_IDX_GATHER_IDX = (
    (torch.arange(0, 32, dtype=torch.int32) * 2 + 1).unsqueeze(0) + _ROW_OFFSETS
).contiguous()  # [8, 32]


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
        """Expected: descending sort, interleaved [val, idx_as_f32, ...] layout.

        Hardware TSORT32 stores uint32 index bits directly in f32 memory
        (bit reinterpretation, not value conversion).  The sim TSORT32
        does value conversion instead — this is a known sim discrepancy.
        """
        src = tensors["src_tensor"]  # [8, 32]
        expected = torch.zeros(8, 64, dtype=torch.float32)
        for row in range(8):
            sorted_vals, sorted_indices = torch.sort(src[row], descending=True)
            idx_as_f32 = sorted_indices.int().view(torch.float32)
            expected[row, 0::2] = sorted_vals
            expected[row, 1::2] = idx_as_f32
        tensors["output"][:] = expected


class Sort32GatherFP32TestCase(PTOTestCase):
    """Test sort32 + gather: separate values and indices from interleaved output.

    Pipeline: sort32 → gather(even positions) → val_output [FP32]
              sort32 → gather(odd positions) → idx_output [FP32, host reinterprets as int32]
    """

    def get_name(self) -> str:
        return "sort32_gather_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.UINT32, init_value=_IDX_TENSOR_FP32),
            TensorSpec("val_gather_idx", [8, 32], DataType.INT32, init_value=_VAL_GATHER_IDX),
            TensorSpec("idx_gather_idx", [8, 32], DataType.INT32, init_value=_IDX_GATHER_IDX),
            TensorSpec("gather_tmp", [8, 32], DataType.INT32, init_value=0),
            TensorSpec("val_output", [8, 32], DataType.FP32, is_output=True),
            TensorSpec("idx_output", [8, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Sort32GatherFP32Program

    def compute_expected(self, tensors, params=None):
        """Expected: sorted values (FP32) and index bits as FP32 (host reinterprets)."""
        src = tensors["src_tensor"]  # [8, 32]
        val_expected = torch.zeros(8, 32, dtype=torch.float32)
        idx_expected = torch.zeros(8, 32, dtype=torch.float32)
        for row in range(8):
            sorted_vals, sorted_indices = torch.sort(src[row], descending=True)
            # uint32 index bits reinterpreted as f32 (matches hardware sort32 output)
            idx_as_f32 = sorted_indices.int().view(torch.float32)
            val_expected[row] = sorted_vals
            idx_expected[row] = idx_as_f32
        tensors["val_output"][:] = val_expected
        tensors["idx_output"][:] = idx_expected


class Sort32GatherMaskFP32TestCase(PTOTestCase):
    """Test sort32 + gather_mask: extract values with P0101 from interleaved output.

    sort32 output layout: [val0, idx0, val1, idx1, ...] (64 elements per row).
    P0101 selects columns 0,2,4,... (stride=2) → [8, 32] of sorted values.
    """

    def get_name(self) -> str:
        return "sort32_gather_mask_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.UINT32, init_value=_IDX_TENSOR_FP32),
            TensorSpec("output", [8, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Sort32GatherMaskFP32Program

    def compute_expected(self, tensors, params=None):
        """P0101 selects even-column positions from interleaved output = sorted values."""
        src = tensors["src_tensor"]  # [8, 32]
        expected = torch.zeros(8, 32, dtype=torch.float32)
        for row in range(8):
            sorted_vals, _ = torch.sort(src[row], descending=True)
            expected[row] = sorted_vals
        tensors["output"][:] = expected


class GatherMaskFP32TestCase(PTOTestCase):
    """Test gather_mask with P1111 pattern on FP32 data.

    P1111 selects all elements, so output should match input exactly.
    """

    def get_name(self) -> str:
        return "gather_mask_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [8, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherMaskFP32Program

    def compute_expected(self, tensors, params=None):
        """P1111 selects all elements — output equals input."""
        tensors["output"][:] = tensors["src_tensor"]


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

    def test_sort32_gather_fp32(self, test_runner):
        """Test sort32 + gather: separate values and indices into distinct tensors.

        Pipeline: sort32 → gather(even) → val_output [8,32] FP32
                  sort32 → gather(odd) → idx_output [8,32] FP32 (host reinterprets)
        """
        test_case = Sort32GatherFP32TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_sort32_gather_mask_fp32(self, test_runner):
        """Test sort32 + gather_mask: extract sorted values with P0101 mask.

        Pipeline: sort32 → gather(mask_pattern=P0101) → output [8,32] FP32
        P0101 selects columns 0,2,4,... (stride=2) from [8,64] interleaved output.
        """
        test_case = Sort32GatherMaskFP32TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_gather_mask_fp32(self, test_runner):
        """Test gather_mask with P1111 pattern: output should match input."""
        test_case = GatherMaskFP32TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
