# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for SPMD (Single Program Multiple Data) execution.

This module tests multi-block data-parallel dispatch using pl.cluster(core_num=N)
and pl.tile.get_block_idx(). Each block processes a different slice of the input
tensors and writes its result to the corresponding output region.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Programs ---

CORE_NUM = 4
TILE_ROWS = 128
TILE_COLS = 128
TOTAL_ROWS = CORE_NUM * TILE_ROWS  # 512


@pl.program
class SPMDAddTensorProgram:
    """SPMD elementwise add (tensor-level): 4 blocks each process a [128, 128] slice."""

    @pl.function(type=pl.FunctionType.Opaque)
    def spmd_add(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
            for block_idx in pl.parallel(0, CORE_NUM, 1, chunk=1, chunk_policy="leading_full"):
                offset = block_idx * TILE_ROWS
                tile_a = pl.slice(a, [TILE_ROWS, TILE_COLS], [offset, 0])
                tile_b = pl.slice(b, [TILE_ROWS, TILE_COLS], [offset, 0])
                tile_c = pl.add(tile_a, tile_b)
                out = pl.assemble(out, tile_c, [offset, 0])
        return out


@pl.program
class SPMDAddProgram:
    """SPMD elementwise add: 4 blocks each process a [128, 128] slice of a [512, 128] tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_add(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.cluster(core_num=4):
            out = self.spmd_add(a, b, out)
        return out


# --- Test Cases ---


class SPMDAddTestCase(PTOTestCase):
    """SPMD add: 4 blocks, each processes [128, 128] of a [512, 128] tensor."""

    def get_name(self) -> str:
        return "spmd_add_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] + tensors["b"]


class SPMDAddTensorTestCase(PTOTestCase):
    """SPMD add (tensor-level): 4 blocks, each processes [128, 128] of a [512, 128] tensor."""

    def get_name(self) -> str:
        return "spmd_add_tensor_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDAddTensorProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] + tensors["b"]


class SPMDAddA5TestCase(SPMDAddTestCase):
    """SPMD add with A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_add_a5_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


# --- Tests ---


class TestSPMDOperations:
    """Test suite for SPMD multi-block dispatch."""

    def test_spmd_add(self, test_runner):
        """SPMD add: 4 blocks each process a [128, 128] slice via get_block_idx."""
        test_case = SPMDAddTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_spmd_add_tensor(self, test_runner):
        """SPMD add (tensor-level): 4 blocks via pl.parallel + pl.slice/assemble."""
        test_case = SPMDAddTensorTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.a5
    def test_spmd_add_a5(self, test_runner):
        """SPMD add with A5 (Ascend 950) backend."""
        test_case = SPMDAddA5TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A5): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
