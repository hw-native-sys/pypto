# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.mgather (pto.mgather) — indexed gather-load from global memory.

Two modes (selected by the ``coalesce`` argument):
  - row  (default): dst[r, j] = mem[idx[r], j]   — idx is a [1, R] index vector,
                    output is [R, mem_cols]. This is the primary path (gathering
                    selected GM rows into an on-chip tile, e.g. paged-KV gather).
  - elem:           dst[i, j] = mem[idx[i, j]]   — mem flat-indexed; output has
                    the same shape as idx.

Test matrix (row mode):
  - Dtypes: FP32, FP16, INT32
  - Shapes: mem [64, 32] gather 16 rows; mem [128, 64] gather 32 rows
  - Index patterns: leading rows, reversed rows, random distinct rows
Plus one elem-mode FP32 case (flat-indexed gather).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

# =============================================================================
# Programs
# =============================================================================


@pl.program
class MgatherRowFP32Program:
    """Row-mode gather: dst[r, :] = mem[idx[r], :], mem [64, 32] -> dst [16, 32]."""

    @pl.function(type=pl.FunctionType.InCore)
    def mgather_kernel(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.FP32],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[1, 16])
        dst: pl.Tile[[16, 32], pl.FP32] = pl.tile.mgather(mem_tensor, idx_tile)
        out: pl.Tensor[[16, 32], pl.FP32] = pl.store(dst, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.FP32],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        output = self.mgather_kernel(mem_tensor, idx_tensor, output)
        return output


@pl.program
class MgatherRowFP16Program:
    """Row-mode gather, FP16, mem [64, 32] -> dst [16, 32]."""

    @pl.function(type=pl.FunctionType.InCore)
    def mgather_kernel(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.FP16],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ) -> pl.Tensor[[16, 32], pl.FP16]:
        idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[1, 16])
        dst: pl.Tile[[16, 32], pl.FP16] = pl.tile.mgather(mem_tensor, idx_tile)
        out: pl.Tensor[[16, 32], pl.FP16] = pl.store(dst, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.FP16],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ) -> pl.Tensor[[16, 32], pl.FP16]:
        output = self.mgather_kernel(mem_tensor, idx_tensor, output)
        return output


@pl.program
class MgatherRowINT32Program:
    """Row-mode gather, INT32, mem [64, 32] -> dst [16, 32]."""

    @pl.function(type=pl.FunctionType.InCore)
    def mgather_kernel(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.INT32],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.INT32]],
    ) -> pl.Tensor[[16, 32], pl.INT32]:
        idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[1, 16])
        dst: pl.Tile[[16, 32], pl.INT32] = pl.tile.mgather(mem_tensor, idx_tile)
        out: pl.Tensor[[16, 32], pl.INT32] = pl.store(dst, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        mem_tensor: pl.Tensor[[64, 32], pl.INT32],
        idx_tensor: pl.Tensor[[1, 16], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.INT32]],
    ) -> pl.Tensor[[16, 32], pl.INT32]:
        output = self.mgather_kernel(mem_tensor, idx_tensor, output)
        return output


@pl.program
class MgatherRowLargeFP32Program:
    """Row-mode gather, FP32, larger mem [128, 64] -> dst [32, 64]."""

    @pl.function(type=pl.FunctionType.InCore)
    def mgather_kernel(
        self,
        mem_tensor: pl.Tensor[[128, 64], pl.FP32],
        idx_tensor: pl.Tensor[[1, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        idx_tile: pl.Tile[[1, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[1, 32])
        dst: pl.Tile[[32, 64], pl.FP32] = pl.tile.mgather(mem_tensor, idx_tile)
        out: pl.Tensor[[32, 64], pl.FP32] = pl.store(dst, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        mem_tensor: pl.Tensor[[128, 64], pl.FP32],
        idx_tensor: pl.Tensor[[1, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        output = self.mgather_kernel(mem_tensor, idx_tensor, output)
        return output


@pl.program
class MgatherElemFP32Program:
    """Elem-mode gather: dst[i, j] = mem[idx[i, j]], mem [256] -> dst [8, 32]."""

    @pl.function(type=pl.FunctionType.InCore)
    def mgather_kernel(
        self,
        mem_tensor: pl.Tensor[[256], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[8, 32])
        dst: pl.Tile[[8, 32], pl.FP32] = pl.tile.mgather(mem_tensor, idx_tile, coalesce="elem")
        out: pl.Tensor[[8, 32], pl.FP32] = pl.store(dst, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        mem_tensor: pl.Tensor[[256], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        output = self.mgather_kernel(mem_tensor, idx_tensor, output)
        return output


# =============================================================================
# Named init_value functions for golden_writer.
# Each function must be self-contained (golden_writer extracts source code into
# an isolated environment with no access to module-level references).
# =============================================================================


def _idx_leading_16_of_64() -> torch.Tensor:
    return torch.arange(16, dtype=torch.int32).reshape(1, 16)


def _idx_reversed_16_of_64() -> torch.Tensor:
    return torch.arange(63, 47, -1, dtype=torch.int32).reshape(1, 16)


def _idx_rand_rows_16_of_64() -> torch.Tensor:
    return torch.randperm(64, dtype=torch.int32)[:16].reshape(1, 16)


def _idx_rand_rows_32_of_128() -> torch.Tensor:
    return torch.randperm(128, dtype=torch.int32)[:32].reshape(1, 32)


def _idx_rand_perm_256_8x32() -> torch.Tensor:
    return torch.randperm(256, dtype=torch.int32).reshape(8, 32)


def _mem_randint_64x32() -> torch.Tensor:
    return torch.randint(-10000, 10000, (64, 32), dtype=torch.int32)


# =============================================================================
# Test Cases — row mode
# =============================================================================


class MgatherRowFP32LeadingTestCase(PTOTestCase):
    """FP32 row gather, leading 16 rows [0..15]."""

    def get_name(self) -> str:
        return "mgather_row_fp32_leading"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [64, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [1, 16], DataType.INT32, init_value=_idx_leading_16_of_64),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowFP32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


class MgatherRowFP32ReversedTestCase(PTOTestCase):
    """FP32 row gather, reversed rows [63..48]."""

    def get_name(self) -> str:
        return "mgather_row_fp32_reversed"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [64, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [1, 16], DataType.INT32, init_value=_idx_reversed_16_of_64),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowFP32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


class MgatherRowFP32RandTestCase(PTOTestCase):
    """FP32 row gather, random distinct rows."""

    def get_name(self) -> str:
        return "mgather_row_fp32_rand"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [64, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [1, 16], DataType.INT32, init_value=_idx_rand_rows_16_of_64),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowFP32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


class MgatherRowFP16RandTestCase(PTOTestCase):
    """FP16 row gather, random distinct rows."""

    def get_name(self) -> str:
        return "mgather_row_fp16_rand"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [64, 32], DataType.FP16, init_value=torch.randn),
            TensorSpec("idx_tensor", [1, 16], DataType.INT32, init_value=_idx_rand_rows_16_of_64),
            TensorSpec("output", [16, 32], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowFP16Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


class MgatherRowINT32RandTestCase(PTOTestCase):
    """INT32 row gather, random distinct rows."""

    def get_name(self) -> str:
        return "mgather_row_int32_rand"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [64, 32], DataType.INT32, init_value=_mem_randint_64x32),
            TensorSpec("idx_tensor", [1, 16], DataType.INT32, init_value=_idx_rand_rows_16_of_64),
            TensorSpec("output", [16, 32], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowINT32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


class MgatherRowLargeFP32TestCase(PTOTestCase):
    """FP32 row gather, larger mem [128, 64] -> 32 rows."""

    def get_name(self) -> str:
        return "mgather_row_fp32_large"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [128, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [1, 32], DataType.INT32, init_value=_idx_rand_rows_32_of_128),
            TensorSpec("output", [32, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherRowLargeFP32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].flatten().long()
        tensors["output"][:] = mem[idx]


# =============================================================================
# Test Cases — elem mode
# =============================================================================


class MgatherElemFP32TestCase(PTOTestCase):
    """FP32 elem gather: dst[i, j] = mem[idx[i, j]], mem [256] -> [8, 32]."""

    def get_name(self) -> str:
        return "mgather_elem_fp32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mem_tensor", [256], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_idx_rand_perm_256_8x32),
            TensorSpec("output", [8, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MgatherElemFP32Program

    def compute_expected(self, tensors, params=None):
        mem = tensors["mem_tensor"]
        idx = tensors["idx_tensor"].long()
        tensors["output"][:] = mem[idx]


# =============================================================================
# pytest test functions
# =============================================================================


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestMgatherRow:
    """Row-mode tile.mgather (dst[r, :] = mem[idx[r], :])."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32_leading(self, test_runner, platform):
        """FP32, leading rows [0..15] — contiguous gather."""
        result = test_runner.run(MgatherRowFP32LeadingTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32_reversed(self, test_runner, platform):
        """FP32, reversed rows [63..48]."""
        result = test_runner.run(MgatherRowFP32ReversedTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32_random(self, test_runner, platform):
        """FP32, random distinct rows."""
        result = test_runner.run(MgatherRowFP32RandTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp16_random(self, test_runner, platform):
        """FP16, random distinct rows."""
        result = test_runner.run(MgatherRowFP16RandTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_int32_random(self, test_runner, platform):
        """INT32, random distinct rows."""
        result = test_runner.run(MgatherRowINT32RandTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32_large(self, test_runner, platform):
        """FP32, larger mem [128, 64] gathering 32 rows."""
        result = test_runner.run(MgatherRowLargeFP32TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestMgatherElem:
    """Elem-mode tile.mgather (dst[i, j] = mem[idx[i, j]])."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32(self, test_runner, platform):
        """FP32 flat-indexed gather, random permutation of [0..255]."""
        result = test_runner.run(MgatherElemFP32TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
