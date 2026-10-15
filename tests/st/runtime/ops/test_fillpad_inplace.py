# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test fillpad_inplace operation with different pad values (zero, max, min).

Each test verifies that fillpad_inplace correctly fills the padding region in place:
1. Load 48x64 data into 64x64 tile (rows 48-63 are padding region)
2. fillpad_inplace with specified pad_value fills rows 48-63 in place (shared UB buffer)
3. Store the full 64x64 tile to output
4. Verify: rows 0-47 = input data, rows 48-63 = expected fill value
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.tensor_spec import ScalarSpec

# --- Programs ---


@pl.program
class FillpadInplaceZeroProgram:
    """Load with partial valid_shape, fillpad_inplace with zero, store."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_inplace_zero_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shape=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_inplace(tile, pad_value=pl.PadValue.zero)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_inplace_zero_kernel(input_tensor, output)
        return output


@pl.program
class FillpadInplaceMaxProgram:
    """Load with partial valid_shape, fillpad_inplace with max, store."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_inplace_max_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shape=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_inplace(tile, pad_value=pl.PadValue.max)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_inplace_max_kernel(input_tensor, output)
        return output


@pl.program
class FillpadInplaceMinProgram:
    """Load with partial valid_shape, fillpad_inplace with min, store."""

    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_inplace_min_kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[64, 64], valid_shape=[48, 64]
        )
        padded_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_inplace(tile, pad_value=pl.PadValue.min)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_tile, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output = self.fillpad_inplace_min_kernel(input_tensor, output)
        return output


# --- Test Cases ---


class FillpadInplaceZeroTestCase(PTOTestCase):
    """Test fillpad_inplace - padding region should be filled with 0.0."""

    def get_name(self) -> str:
        return "fillpad_inplace_zero"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadInplaceZeroProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = 0.0"""
        expected = torch.zeros(64, 64, dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


class FillpadInplaceMaxTestCase(PTOTestCase):
    """Test fillpad_inplace - padding region should be filled with FP32 max."""

    def get_name(self) -> str:
        return "fillpad_inplace_max"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadInplaceMaxProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = FP32 max (+inf)"""
        expected = torch.full((64, 64), float("inf"), dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


class FillpadInplaceMinTestCase(PTOTestCase):
    """Test fillpad_inplace - padding region should be filled with FP32 min (-inf)."""

    def get_name(self) -> str:
        return "fillpad_inplace_min"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [48, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FillpadInplaceMinProgram

    def compute_expected(self, tensors, params=None):
        """Expected: rows 0-47 = input, rows 48-63 = -inf"""
        expected = torch.full((64, 64), float("-inf"), dtype=torch.float32)
        expected[:48, :] = tensors["input_tensor"]
        tensors["output"][:] = expected


# --- Tests ---


class TestFillpadInplace:
    """Test suite to verify fillpad_inplace fills padding region in place with different pad values."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fillpad_inplace_zero(self, test_runner, platform):
        """Verify fillpad_inplace fills the padding region with 0.0."""
        result = test_runner.run(FillpadInplaceZeroTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fillpad_inplace_max(self, test_runner, platform):
        """Verify fillpad_inplace fills the padding region with FP32 max value."""
        result = test_runner.run(FillpadInplaceMaxTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fillpad_inplace_min(self, test_runner, platform):
        """Verify fillpad_inplace fills the padding region with FP32 min value (-inf)."""
        result = test_runner.run(FillpadInplaceMinTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


@pl.jit
def fillpad_then_narrow_static(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """A post-padding validity update must mutate the original padded handle."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, 32, 48)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
        pl.store(repadded, [0, 0], out)
    return out


@pl.jit
def fillpad_then_narrow_dynamic(
    x: pl.Tensor, out: pl.Out[pl.Tensor], rows: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX]
):
    """Runtime bounds also preserve alias mutation, including previously padded rows."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, rows, cols)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
        pl.store(repadded, [0, 0], out)
    return out


def _fillpad_narrow_cases():
    for kernel, bounds in (
        (fillpad_then_narrow_static, (32, 48)),
        (fillpad_then_narrow_dynamic, (32, 48)),
        (fillpad_then_narrow_dynamic, (56, 48)),
    ):
        rows, cols = bounds
        # Nonzero deterministic input distinguishes preserved data from padding.
        x = torch.arange(1, 48 * 64 + 1, dtype=torch.float32).reshape(48, 64)
        padded = torch.zeros(64, 64, dtype=torch.float32)
        padded[:48, :] = x
        expected = torch.full((64, 64), float("inf"), dtype=torch.float32)
        expected[:rows, :cols] = padded[:rows, :cols]
        dynamic = kernel is fillpad_then_narrow_dynamic
        scalar_args = bounds if dynamic else ()
        yield st.case(
            kernel,
            x,
            torch.empty(64, 64, dtype=torch.float32),
            *scalar_args,
            name=f"fillpad_after_{'dynamic' if dynamic else 'static'}_narrow_{rows}x{cols}",
            scalars=[ScalarSpec("rows", rows), ScalarSpec("cols", cols)] if dynamic else [],
            memory_planner=MemoryPlanner.PTOAS,
            golden=lambda _tensors, expected=expected: expected,
        )


@st.cases(*_fillpad_narrow_cases())
def test_fillpad_inplace_after_set_validshape(case_run):
    """The second fill must observe the narrowed metadata through the old alias."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
