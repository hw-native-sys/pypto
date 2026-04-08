# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tensor.scatter_ (element-level scatter).

Validates that scatter_ correctly places values from a source into a destination
tensor at positions specified by an index tensor along a given dimension.
Follows PyTorch torch.Tensor.scatter_ semantics:

    input[i0]...[i_{d-1}][ index[i0...in] ][i_{d+1}]...[in] = src[i0...in]

Test matrix:
  - 2D dim=1: index selects destination columns
  - 2D dim=0: index selects destination rows
  - 2D scalar src: fill constant at scattered positions
  - 3D dim=2: scatter along last dimension of a 3D tensor

All tests use PTOAS strategy (BackendType.Ascend910B + OptimizationStrategy.Default).
scatter_ decomposes to scalar tgetval/tsetval loops so shapes are kept small.

Design notes:
  scatter_ operates on a local buffer (from pl.full) inside an InCore function.
  The ConvertTensorToTileOps pass decomposes the tensor-level scatter_ into nested
  for-loops with tile.read / tile.write.  Because scatter_ returns a TensorType at
  DSL level (not a TileType), the kernel cannot call pl.store() directly.  Instead,
  the kernel declares a pl.Out parameter and returns the scatter result; the pass
  detects the unused Out param and auto-inserts tile.store to write the result back.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# ── Deterministic index tensors ──────────────────────────────────────────────
# Pre-built so that golden_writer can inline them (numel <= 100).
#
# Tile alignment constraint: for RowMajor+NoneBox tiles, Cols*sizeof(DType) must
# be 32-byte aligned.  For int32/float32 (4 bytes), Cols must be a multiple of 8.
# Shapes are chosen accordingly.

# 2D dim=1: index [8, 8] with values in [0, 16)
_IDX_2D_DIM1 = torch.tensor(
    [
        [0, 3, 7, 15, 1, 5, 9, 12],
        [2, 6, 10, 14, 0, 4, 8, 11],
        [3, 7, 13, 15, 1, 2, 6, 10],
        [5, 9, 11, 14, 0, 4, 8, 12],
        [1, 3, 7, 15, 2, 6, 10, 14],
        [0, 4, 8, 11, 3, 7, 13, 15],
        [1, 2, 6, 10, 5, 9, 11, 14],
        [0, 4, 8, 12, 1, 5, 9, 13],
    ],
    dtype=torch.int32,
)

# 2D dim=0: index [4, 8] with values in [0, 16)
_IDX_2D_DIM0 = torch.tensor(
    [
        [0, 3, 7, 15, 1, 5, 9, 12],
        [2, 6, 10, 14, 0, 4, 8, 11],
        [3, 7, 13, 15, 1, 2, 6, 10],
        [5, 9, 11, 14, 0, 4, 8, 12],
    ],
    dtype=torch.int32,
)

# 3D dim=2: index [2, 4, 8] with values in [0, 8)
_IDX_3D_DIM2 = torch.tensor(
    [
        [[0, 3, 7, 1, 5, 6, 2, 4], [7, 0, 3, 5, 1, 2, 6, 4], [2, 7, 3, 5, 0, 4, 6, 1], [3, 0, 4, 7, 5, 2, 6, 1]],
        [[1, 2, 6, 3, 4, 7, 0, 5], [6, 1, 0, 7, 3, 4, 5, 2], [0, 5, 6, 3, 1, 4, 2, 7], [1, 2, 7, 3, 0, 5, 4, 6]],
    ],
    dtype=torch.int32,
)


# ---------------------------------------------------------------------------
# 2D dim=1: scatter along columns
# ---------------------------------------------------------------------------


@pl.program
class Scatter2dDim1Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        index: pl.Tensor[[8, 8], pl.INT32],
        src: pl.Tensor[[8, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        buf: pl.Tensor[[8, 16], pl.FP32] = pl.full([8, 16], dtype=pl.FP32, value=1.0)
        result: pl.Tensor[[8, 16], pl.FP32] = pl.scatter_(buf, dim=1, index=index, src=src)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        index: pl.Tensor[[8, 8], pl.INT32],
        src: pl.Tensor[[8, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        out = self.kernel(index, src, out)
        return out


class Scatter2dDim1TestCase(PTOTestCase):
    """2D scatter along dim=1: index selects destination columns."""

    __test__ = False

    def get_name(self) -> str:
        return "scatter_2d_dim1"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("index", [8, 8], DataType.INT32, init_value=_IDX_2D_DIM1),
            TensorSpec("src", [8, 8], DataType.FP32, init_value=2.0),
            TensorSpec("out", [8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Scatter2dDim1Program

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full([8, 16], 1.0)
        expected.scatter_(1, tensors["index"].long(), tensors["src"].float())
        tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# 2D dim=0: scatter along rows
# ---------------------------------------------------------------------------


@pl.program
class Scatter2dDim0Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        index: pl.Tensor[[4, 8], pl.INT32],
        src: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
    ) -> pl.Tensor[[16, 8], pl.FP32]:
        buf: pl.Tensor[[16, 8], pl.FP32] = pl.full([16, 8], dtype=pl.FP32, value=1.0)
        result: pl.Tensor[[16, 8], pl.FP32] = pl.scatter_(buf, dim=0, index=index, src=src)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        index: pl.Tensor[[4, 8], pl.INT32],
        src: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
    ) -> pl.Tensor[[16, 8], pl.FP32]:
        out = self.kernel(index, src, out)
        return out


class Scatter2dDim0TestCase(PTOTestCase):
    """2D scatter along dim=0: index selects destination rows."""

    __test__ = False

    def get_name(self) -> str:
        return "scatter_2d_dim0"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("index", [4, 8], DataType.INT32, init_value=_IDX_2D_DIM0),
            TensorSpec("src", [4, 8], DataType.FP32, init_value=2.0),
            TensorSpec("out", [16, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Scatter2dDim0Program

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full([16, 8], 1.0)
        expected.scatter_(0, tensors["index"].long(), tensors["src"].float())
        tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# 2D dim=1 with scalar src: fill a constant at scattered positions
# ---------------------------------------------------------------------------


@pl.program
class Scatter2dScalarProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        index: pl.Tensor[[8, 8], pl.INT32],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        buf: pl.Tensor[[8, 16], pl.FP32] = pl.full([8, 16], dtype=pl.FP32, value=1.0)
        result: pl.Tensor[[8, 16], pl.FP32] = pl.scatter_(buf, dim=1, index=index, src=99.0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        index: pl.Tensor[[8, 8], pl.INT32],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        out = self.kernel(index, out)
        return out


class Scatter2dScalarTestCase(PTOTestCase):
    """2D scatter with scalar src (99.0) along dim=1."""

    __test__ = False

    def get_name(self) -> str:
        return "scatter_2d_scalar"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("index", [8, 8], DataType.INT32, init_value=_IDX_2D_DIM1),
            TensorSpec("out", [8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Scatter2dScalarProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full([8, 16], 1.0)
        expected.scatter_(1, tensors["index"].long(), 99.0)
        tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# 3D dim=2: scatter along last dimension of a 3D tensor
# ---------------------------------------------------------------------------


@pl.program
class Scatter3dDim2Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        index: pl.Tensor[[2, 4, 8], pl.INT32],
        src: pl.Tensor[[2, 4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
        buf: pl.Tensor[[2, 4, 8], pl.FP32] = pl.full([2, 4, 8], dtype=pl.FP32, value=1.0)
        result: pl.Tensor[[2, 4, 8], pl.FP32] = pl.scatter_(buf, dim=2, index=index, src=src)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        index: pl.Tensor[[2, 4, 8], pl.INT32],
        src: pl.Tensor[[2, 4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
        out = self.kernel(index, src, out)
        return out


class Scatter3dDim2TestCase(PTOTestCase):
    """3D scatter along dim=2: index selects positions in last dimension."""

    __test__ = False

    def get_name(self) -> str:
        return "scatter_3d_dim2"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("index", [2, 4, 8], DataType.INT32, init_value=_IDX_3D_DIM2),
            TensorSpec("src", [2, 4, 8], DataType.FP32, init_value=2.0),
            TensorSpec("out", [2, 4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Scatter3dDim2Program

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full([2, 4, 8], 1.0)
        expected.scatter_(2, tensors["index"].long(), tensors["src"].float())
        tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


class TestScatterOperations:
    """Test suite for tensor.scatter_ element-level scatter."""

    def test_scatter_2d_dim1(self, test_runner):
        """2D scatter along dim=1: index selects destination columns."""
        result = test_runner.run(Scatter2dDim1TestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_scatter_2d_dim0(self, test_runner):
        """2D scatter along dim=0: index selects destination rows."""
        result = test_runner.run(Scatter2dDim0TestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_scatter_2d_scalar(self, test_runner):
        """2D scatter with scalar src (99.0) along dim=1."""
        result = test_runner.run(Scatter2dScalarTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_scatter_3d_dim2(self, test_runner):
        """3D scatter along dim=2: index selects positions in last dimension."""
        result = test_runner.run(Scatter3dDim2TestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
