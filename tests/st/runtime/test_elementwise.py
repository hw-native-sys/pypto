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

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.language.beginner.elementwise import (
    TileAdd64Program,
    TileAdd128Program,
    TileMul64Program,
    TileMul128Program,
)


class TileAddTestCase(PTOTestCase):
    """Test case for tile element-wise addition (128x128)."""

    def get_name(self) -> str:
        return "tile_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAdd128Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TileAdd64x64TestCase(PTOTestCase):
    """Test case for tile element-wise addition (64x64)."""

    def get_name(self) -> str:
        return "tile_add_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAdd64Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TileMulTestCase(PTOTestCase):
    """Test case for tile element-wise multiplication (128x128)."""

    def get_name(self) -> str:
        return "tile_mul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [128, 128],
                DataType.FP32,
                init_value=torch.randn,
            ),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileMul128Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TileMul64x64TestCase(PTOTestCase):
    """Test case for tile element-wise multiplication (64x64)."""

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
        return TileMul64Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TileAddPTOASTestCase(TileAddTestCase):
    """Test case for tile add with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return "tile_add_ptoas_128x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.PTOAS

    def get_backend_type(self) -> BackendType:
        return BackendType.PTO


class TileMulPTOASTestCase(TileMulTestCase):
    """Test case for tile mul with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return "tile_mul_ptoas_128x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.PTOAS

    def get_backend_type(self) -> BackendType:
        return BackendType.PTO


# =============================================================================
# pytest test functions
# =============================================================================


class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    def test_tile_add_64x64(self, test_runner):
        """Test tile addition with 64x64 shape."""
        test_case = TileAdd64x64TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_add_128x128(self, test_runner):
        """Test tile addition with 128x128 shape."""
        test_case = TileAddTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_mul_64x64(self, test_runner):
        """Test tile multiplication with 64x64 shape."""
        test_case = TileMul64x64TestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_mul_128x128(self, test_runner):
        """Test tile multiplication with 128x128 shape."""
        test_case = TileMulTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_add_ptoas_strategy(self, test_runner):
        """Test tile addition with PTO backend and PTOAS optimization."""
        test_case = TileAddPTOASTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_mul_ptoas_strategy(self, test_runner):
        """Test tile multiplication with PTO backend and PTOAS optimization."""
        test_case = TileMulPTOASTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
