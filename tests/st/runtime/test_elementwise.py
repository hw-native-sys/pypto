# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based elementwise operations using the PyPTO frontend.

This module defines integration tests for elementwise add and multiply
kernels implemented with the internal PTOTestCase harness, including
variants for different shapes and optimization strategies.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy


class TestTileAdd(PTOTestCase):
    """Test case for tile element-wise addition.

    This test case demonstrates the simplified pattern:
    - Just implement incore function in get_program() and compute_expected()
    - Orchestration function will be auto-generated

    Note: PyPTO requires shape dimensions to be compile-time constants in type
    annotations. For different shapes, create separate test classes (see TestTileAdd64x64).
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.block.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.block.add(tile_a, tile_b)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileAdd64x64(PTOTestCase):
    """Test tile addition with 64x64 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_add_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_b = pl.block.load(b, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.block.add(tile_a, tile_b)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileMul(PTOTestCase):
    """Test case for tile element-wise multiplication (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_mul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            # Method 1: Use Callable to generate random data (different on each run)
            TensorSpec(
                "a",
                [128, 128],
                DataType.FP32,
                init_value=torch.randn,
            ),
            # Method 2: Use scalar value (recommended - simple and serializable)
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            # For other methods, see TestCustomArrayInit class examples:
            # - Small arrays can use torch.tensor([[...]])
            # - Identity matrix: torch.eye(n)
            # - Diagonal matrix: torch.diag(torch.tensor([...]))
            # Output tensor: automatically zero-initialized
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.block.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.block.mul(tile_a, tile_b)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestTileMul64x64(PTOTestCase):
    """Test tile multiplication with 64x64 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_mul_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [64, 64],
                DataType.FP32,
                init_value=torch.randn,
            ),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_b = pl.block.load(b, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.block.mul(tile_a, tile_b)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestTileAddWithPTOAS(TestTileAdd):
    """Test tile add with PTOAS optimization strategy.

    This demonstrates how to use a custom optimization strategy.
    """

    __test__ = False  # Not a pytest test class

    def get_strategy(self):
        return OptimizationStrategy.PTOAS

    def get_name(self) -> str:
        return "tile_add_ptoas_128x128"


class TestCustomArrayInit(PTOTestCase):
    """Test case demonstrating custom array initialization patterns."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "custom_array_init"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            # Small array: custom values (will be serialized)
            TensorSpec(
                "small",
                [3, 3],
                DataType.FP32,
                init_value=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
            ),
            # Identity matrix
            TensorSpec("identity", [4, 4], DataType.FP32, init_value=torch.eye(4, dtype=torch.float32)),
            # Constant array (optimized to torch.full)
            TensorSpec("constant", [5, 5], DataType.FP32, init_value=torch.ones((5, 5)) * 3.14),
            # Diagonal matrix (small arrays will be serialized)
            TensorSpec(
                "diagonal",
                [3, 3],
                DataType.FP32,
                init_value=torch.diag(torch.tensor([1, 2, 3], dtype=torch.float32)),
            ),
            # Output
            TensorSpec("out", [3, 3], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Placeholder - this test is just for demonstrating array initialization
        return None

    def compute_expected(self, tensors, params=None):
        # Simple example: copy small array to output
        tensors["out"][:] = tensors["small"][:3, :3]


# =============================================================================
# pytest test functions
# =============================================================================


class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    def test_tile_add_64x64(self, test_runner):
        """Test tile addition with 64x64 shape."""
        test_case = TestTileAdd64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_add_128x128(self, test_runner):
        """Test tile addition with 128x128 shape."""
        test_case = TestTileAdd()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_mul_64x64(self, test_runner):
        """Test tile multiplication with 64x64 shape."""
        test_case = TestTileMul64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_mul_128x128(self, test_runner):
        """Test tile multiplication with 128x128 shape."""
        test_case = TestTileMul()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    @pytest.mark.skip(reason="PTOAS optimization strategy has calculation issues - needs investigation")
    def test_tile_add_ptoas_strategy(self, test_runner):
        """Test tile addition with PTOAS optimization strategy."""
        test_case = TestTileAddWithPTOAS()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
