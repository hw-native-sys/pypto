# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.scatter_update.

tile.scatter_update(input, index, src) updates rows in input at positions
specified by a 2D index tile with corresponding rows from src.

Hardware semantics (PTO backend):
  tile.scatter_update generates nested scf.for loops with pto.tgetval +
  pto.tsetval to copy src elements into dst at scattered row positions.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


def make_scatter_update_src_fp32() -> torch.Tensor:
    """Return unique row-major FP32 values for scatter_update source."""
    return torch.arange(0, 512, dtype=torch.float32).reshape(16, 32)


def make_scatter_update_src_fp16() -> torch.Tensor:
    """Return unique row-major FP16 values for scatter_update source."""
    return torch.arange(0, 512, dtype=torch.float16).reshape(16, 32)


def make_scatter_update_src_fp32_8x32() -> torch.Tensor:
    """Return unique row-major FP32 values for single-batch scatter_update source."""
    return torch.arange(0, 256, dtype=torch.float32).reshape(8, 32)


def make_scatter_update_index_single_batch() -> torch.Tensor:
    """Return index tensor for single-batch scatter_update (b=1, s=8)."""
    return torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)


# ---------------------------------------------------------------------------
# Kernel programs
# ---------------------------------------------------------------------------


@pl.program
class TileScatterUpdateFP16Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP16],
        index_t: pl.Tensor[[2, 8], pl.INT32],
        src_t: pl.Tensor[[16, 32], pl.FP16],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
    ) -> pl.Tensor[[32, 32], pl.FP16]:
        input_tile: pl.Tile[[32, 32], pl.FP16] = pl.load(input_t, [0, 0], [32, 32])
        index_tile: pl.Tile[[2, 8], pl.INT32] = pl.load(index_t, [0, 0], [2, 8])
        src_tile: pl.Tile[[16, 32], pl.FP16] = pl.load(src_t, [0, 0], [16, 32])
        result: pl.Tile[[32, 32], pl.FP16] = pl.tile.scatter_update(
            input_tile, dim=-2, index=index_tile, src=src_tile
        )
        return pl.store(result, [0, 0], dst_t)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP16],
        index_t: pl.Tensor[[2, 8], pl.INT32],
        src_t: pl.Tensor[[16, 32], pl.FP16],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
    ) -> pl.Tensor[[32, 32], pl.FP16]:
        dst_t = self.kernel(input_t, index_t, src_t, dst_t)
        return dst_t


@pl.program
class TileScatterUpdateSingleBatchProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP32],
        index_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[8, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        input_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(input_t, [0, 0], [32, 32])
        index_tile: pl.Tile[[1, 8], pl.INT32] = pl.load(index_t, [0, 0], [1, 8])
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
        result: pl.Tile[[32, 32], pl.FP32] = pl.tile.scatter_update(
            input_tile, dim=-2, index=index_tile, src=src_tile
        )
        return pl.store(result, [0, 0], dst_t)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP32],
        index_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[8, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        dst_t = self.kernel(input_t, index_t, src_t, dst_t)
        return dst_t


@pl.program
class TileScatterUpdateProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP32],
        index_t: pl.Tensor[[2, 8], pl.INT32],
        src_t: pl.Tensor[[16, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        input_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(input_t, [0, 0], [32, 32])
        index_tile: pl.Tile[[2, 8], pl.INT32] = pl.load(index_t, [0, 0], [2, 8])
        src_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(src_t, [0, 0], [16, 32])
        result: pl.Tile[[32, 32], pl.FP32] = pl.tile.scatter_update(
            input_tile, dim=-2, index=index_tile, src=src_tile
        )
        return pl.store(result, [0, 0], dst_t)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_t: pl.Tensor[[32, 32], pl.FP32],
        index_t: pl.Tensor[[2, 8], pl.INT32],
        src_t: pl.Tensor[[16, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        dst_t = self.kernel(input_t, index_t, src_t, dst_t)
        return dst_t


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TileScatterUpdateTestCase(PTOTestCase):
    """Basic scatter_update: update rows of input[32,32] at indices from index[2,8] with src[16,32].

    index contains 16 row indices (b=2, s=8, total b*s=16).
    For each flat position k in [0, 16): input[index[k], :] = src[k, :].
    """

    def get_name(self) -> str:
        return "tile_scatter_update"

    def define_tensors(self) -> list[TensorSpec]:
        # index values must be valid row indices into input (0..31), dtype INT32
        # INT32 tile needs cols >= 8 for 32-byte alignment (4 bytes * 8 = 32)
        index_data = torch.tensor(
            [[0, 2, 4, 6, 8, 10, 12, 14], [16, 18, 20, 22, 24, 26, 28, 30]], dtype=torch.int32
        )
        return [
            TensorSpec("input_t", [32, 32], DataType.FP32, init_value=torch.ones),
            TensorSpec("index_t", [2, 8], DataType.INT32, init_value=index_data),
            TensorSpec("src_t", [16, 32], DataType.FP32, init_value=make_scatter_update_src_fp32),
            TensorSpec("dst_t", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileScatterUpdateProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        input_t = tensors["input_t"].clone()
        index_t = tensors["index_t"]
        src_t = tensors["src_t"]
        flat_index = index_t.reshape(-1)
        for k in range(flat_index.shape[0]):
            row = int(flat_index[k].item())
            input_t[row] = src_t[k]
        tensors["dst_t"][:] = input_t


class TileScatterUpdateFP16TestCase(PTOTestCase):
    """scatter_update with FP16 data type.

    FP16 tiles have different alignment requirements (cols >= 16 for 32-byte alignment).
    Validates that tgetval/tsetval correctly handle 2-byte element width.
    """

    def get_name(self) -> str:
        return "tile_scatter_update_fp16"

    def define_tensors(self) -> list[TensorSpec]:
        index_data = torch.tensor(
            [[0, 2, 4, 6, 8, 10, 12, 14], [16, 18, 20, 22, 24, 26, 28, 30]], dtype=torch.int32
        )
        return [
            TensorSpec("input_t", [32, 32], DataType.FP16, init_value=1.0),
            TensorSpec("index_t", [2, 8], DataType.INT32, init_value=index_data),
            TensorSpec("src_t", [16, 32], DataType.FP16, init_value=make_scatter_update_src_fp16),
            TensorSpec("dst_t", [32, 32], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileScatterUpdateFP16Program

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        input_t = tensors["input_t"].clone()
        index_t = tensors["index_t"]
        src_t = tensors["src_t"]
        flat_index = index_t.reshape(-1)
        for k in range(flat_index.shape[0]):
            row = int(flat_index[k].item())
            input_t[row] = src_t[k]
        tensors["dst_t"][:] = input_t


class TileScatterUpdateDuplicateIndicesTestCase(PTOTestCase):
    """scatter_update with duplicate indices (multiple writes to the same row).

    When multiple index entries point to the same row, last-write-wins semantics apply.
    This is common in embedding lookup / attention scenarios.
    """

    def get_name(self) -> str:
        return "tile_scatter_update_duplicate_indices"

    def define_tensors(self) -> list[TensorSpec]:
        # Rows 0 and 1 are written multiple times; last writer (higher flat index) wins.
        index_data = torch.tensor([[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 6, 7, 8, 9, 10, 11]], dtype=torch.int32)
        return [
            TensorSpec("input_t", [32, 32], DataType.FP32, init_value=torch.ones),
            TensorSpec("index_t", [2, 8], DataType.INT32, init_value=index_data),
            TensorSpec("src_t", [16, 32], DataType.FP32, init_value=make_scatter_update_src_fp32),
            TensorSpec("dst_t", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileScatterUpdateProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        input_t = tensors["input_t"].clone()
        index_t = tensors["index_t"]
        src_t = tensors["src_t"]
        flat_index = index_t.reshape(-1)
        for k in range(flat_index.shape[0]):
            row = int(flat_index[k].item())
            input_t[row] = src_t[k]
        tensors["dst_t"][:] = input_t


class TileScatterUpdateSingleBatchTestCase(PTOTestCase):
    """scatter_update with b=1 (single batch, degenerate outer loop).

    index shape [1, 8]: only one row in the batch dimension.
    Validates that the outer loop (i in [0, b)) handles b=1 correctly.
    """

    def get_name(self) -> str:
        return "tile_scatter_update_single_batch"

    def define_tensors(self) -> list[TensorSpec]:
        # INT32 cols=8 satisfies 32-byte alignment (4 * 8 = 32)
        return [
            TensorSpec("input_t", [32, 32], DataType.FP32, init_value=torch.ones),
            TensorSpec("index_t", [1, 8], DataType.INT32, init_value=make_scatter_update_index_single_batch),
            TensorSpec("src_t", [8, 32], DataType.FP32, init_value=make_scatter_update_src_fp32_8x32),
            TensorSpec("dst_t", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileScatterUpdateSingleBatchProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        input_t = tensors["input_t"].clone()
        index_t = tensors["index_t"]
        src_t = tensors["src_t"]
        flat_index = index_t.reshape(-1)
        for k in range(flat_index.shape[0]):
            row = int(flat_index[k].item())
            input_t[row] = src_t[k]
        tensors["dst_t"][:] = input_t


class TileScatterUpdateShuffledIndicesTestCase(PTOTestCase):
    """scatter_update with non-monotonic, shuffled indices.

    Validates that flat offset calculation is correct when indices are not
    in ascending order, which is the common case in real workloads.
    """

    def get_name(self) -> str:
        return "tile_scatter_update_shuffled_indices"

    def define_tensors(self) -> list[TensorSpec]:
        index_data = torch.tensor(
            [[31, 5, 20, 3, 11, 27, 8, 15], [1, 22, 9, 30, 17, 6, 25, 13]], dtype=torch.int32
        )
        return [
            TensorSpec("input_t", [32, 32], DataType.FP32, init_value=torch.ones),
            TensorSpec("index_t", [2, 8], DataType.INT32, init_value=index_data),
            TensorSpec("src_t", [16, 32], DataType.FP32, init_value=make_scatter_update_src_fp32),
            TensorSpec("dst_t", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileScatterUpdateProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        input_t = tensors["input_t"].clone()
        index_t = tensors["index_t"]
        src_t = tensors["src_t"]
        flat_index = index_t.reshape(-1)
        for k in range(flat_index.shape[0]):
            row = int(flat_index[k].item())
            input_t[row] = src_t[k]
        tensors["dst_t"][:] = input_t


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestScatterUpdateOperations:
    """Test suite for tile.scatter_update."""

    def test_tile_scatter_update(self, test_runner):
        """Basic scatter update: 16 src rows written into input at even-numbered indices."""
        result = test_runner.run(TileScatterUpdateTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_scatter_update_fp16(self, test_runner):
        """Scatter update with FP16 data type (2-byte elements, different alignment)."""
        result = test_runner.run(TileScatterUpdateFP16TestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_scatter_update_duplicate_indices(self, test_runner):
        """Scatter update with duplicate indices (last-write-wins semantics)."""
        result = test_runner.run(TileScatterUpdateDuplicateIndicesTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_scatter_update_single_batch(self, test_runner):
        """Scatter update with b=1 (degenerate outer loop)."""
        result = test_runner.run(TileScatterUpdateSingleBatchTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_scatter_update_shuffled_indices(self, test_runner):
        """Scatter update with non-monotonic, shuffled index order."""
        result = test_runner.run(TileScatterUpdateShuffledIndicesTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
