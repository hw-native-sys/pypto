# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test fillpad operation with different pad values (zero, max, min).

Each test verifies that fillpad correctly fills the padding region:
1. Load 48x64 data into 64x64 tile (rows 48-63 are padding region)
2. fillpad with specified pad_value fills rows 48-63 AND expands valid_shape to 64x64
3. Store the full 64x64 tile to output
4. Verify: rows 0-47 = input data, rows 48-63 = expected fill value
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Programs ---


@pl.program
class FillpadZeroProgram:
    """Verify fillpad by storing the full tile including padding region."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_zero_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shapes=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile, pad_value=pl.PadValue.zero)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_zero_kernel(input_tensor, output)
        return output


@pl.program
class FillpadMaxProgram:
    """Verify fillpad with max by storing the full tile including padding region."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_max_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shapes=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile, pad_value=pl.PadValue.max)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_max_kernel(input_tensor, output)
        return output


@pl.program
class FillpadMinProgram:
    """Verify fillpad with min by storing the full tile including padding region."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_min_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shapes=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile, pad_value=pl.PadValue.min)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_min_kernel(input_tensor, output)
        return output


# --- Test Cases ---


class FillpadZeroTestCase(PTOTestCase):
    """Test fillpad - padding region should be filled with 0.0."""

    def get_name(self) -> str:
        return "fillpad_zero"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadZeroProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = 0.0"""
        expected = torch.zeros(64, 64, dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


class FillpadMaxTestCase(PTOTestCase):
    """Test fillpad - padding region should be filled with FP32 max."""

    def get_name(self) -> str:
        return "fillpad_max"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadMaxProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = FP32 max"""
        expected = torch.full((64, 64), float("inf"), dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


class FillpadMinTestCase(PTOTestCase):
    """Test fillpad - padding region should be filled with FP32 min (-inf)."""

    def get_name(self) -> str:
        return "fillpad_min"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadMinProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = -inf"""
        expected = torch.full((64, 64), float("-inf"), dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


# --- Tests ---


class TestFillpad:
    """Test suite to verify fillpad fills padding region with different pad values."""

    def test_fillpad_zero(self, test_runner):
        """Verify fillpad fills the padding region with 0.0."""
        test_case = FillpadZeroTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_fillpad_max(self, test_runner):
        """Verify fillpad fills the padding region with FP32 max value."""
        test_case = FillpadMaxTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_fillpad_min(self, test_runner):
        """Verify fillpad fills the padding region with FP32 min value (-inf)."""
        test_case = FillpadMinTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
