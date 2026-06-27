# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test argmax/argmin reductions: row_argmax (TROWARGMAX), row_argmin (TROWARGMIN),
col_argmax (TCOLARGMAX), col_argmin (TCOLARGMIN).

row_argmax/argmin reduce along axis=1 ([M, N] -> [M, 1]) and yield, for each row,
the column index of the max/min. col_argmax/argmin reduce along axis=0
([M, N] -> [1, N]) and yield, for each column, the row index of the max/min. The
index output dtype is INT32. All four require a tmp scratch tile (unlike
col_max/col_min, the column argmax/argmin variants also need one).

Inputs use a continuous random distribution so each row/column has a unique
extremum — this avoids tie-break ambiguity between the golden (torch first-index)
and the device, and keeps the integer index comparison exact (rtol/atol default).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


def _rand_init(shape):
    """No-arg init callable returning continuous values with unique extrema."""
    return lambda: torch.randn(shape)


# =============================================================================
# Programs — row_argmax / row_argmin (require tmp_tile, like row_max)
# =============================================================================


@pl.program
class RowArgmax_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.INT32] = pl.tile.row_argmax(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowArgmin_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 1], pl.INT32]],
    ) -> pl.Tensor[[16, 1], pl.INT32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
            [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[16, 1], pl.INT32] = pl.tile.row_argmin(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 1], pl.INT32]],
    ) -> pl.Tensor[[16, 1], pl.INT32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowArgmax_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP16] = pl.tile.create(
            [32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.INT32] = pl.tile.row_argmax(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        output = self.kernel(input_tensor, output)
        return output


# =============================================================================
# Programs — col_argmax / col_argmin (also require tmp_tile)
# =============================================================================


@pl.program
class ColArgmax_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    ) -> pl.Tensor[[1, 64], pl.INT32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 64], pl.INT32] = pl.tile.col_argmax(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    ) -> pl.Tensor[[1, 64], pl.INT32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColArgmin_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
    ) -> pl.Tensor[[1, 128], pl.INT32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        tmp: pl.Tile[[8, 128], pl.FP32] = pl.tile.create(
            [8, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 128], pl.INT32] = pl.tile.col_argmin(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
    ) -> pl.Tensor[[1, 128], pl.INT32]:
        output = self.kernel(input_tensor, output)
        return output


# =============================================================================
# Programs — tensor-level (lowered via tensor->tile, tmp injected by conversion)
# =============================================================================


@pl.program
class TensorRowArgmax_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        result: pl.Tensor[[32, 1], pl.INT32] = pl.row_argmax(a)
        return pl.assemble(output, result, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.INT32]],
    ) -> pl.Tensor[[32, 1], pl.INT32]:
        output = self.kernel(a, output)
        return output


@pl.program
class TensorColArgmax_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    ) -> pl.Tensor[[1, 64], pl.INT32]:
        result: pl.Tensor[[1, 64], pl.INT32] = pl.col_argmax(a)
        return pl.assemble(output, result, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    ) -> pl.Tensor[[1, 64], pl.INT32]:
        output = self.kernel(a, output)
        return output


# =============================================================================
# Test Cases — row argmax/argmin
# =============================================================================


class RowArgmax32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_argmax_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_rand_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowArgmax_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmax(tensors["input_tensor"], dim=1, keepdim=True).to(torch.int32)


class RowArgmin16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_argmin_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=_rand_init([16, 16])),
            TensorSpec("output", [16, 1], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowArgmin_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmin(tensors["input_tensor"], dim=1, keepdim=True).to(torch.int32)


class RowArgmax32x64FP16(PTOTestCase):
    def get_name(self) -> str:
        return "row_argmax_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=_rand_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowArgmax_32x64_FP16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmax(tensors["input_tensor"], dim=1, keepdim=True).to(torch.int32)


# =============================================================================
# Test Cases — col argmax/argmin
# =============================================================================


class ColArgmax32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_argmax_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_rand_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColArgmax_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmax(tensors["input_tensor"], dim=0, keepdim=True).to(torch.int32)


class ColArgmin8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_argmin_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=_rand_init([8, 128])),
            TensorSpec("output", [1, 128], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColArgmin_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmin(tensors["input_tensor"], dim=0, keepdim=True).to(torch.int32)


# =============================================================================
# Test Cases — tensor level
# =============================================================================


class TensorRowArgmax32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "tensor_row_argmax_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 64], DataType.FP32, init_value=_rand_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorRowArgmax_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmax(tensors["a"], dim=1, keepdim=True).to(torch.int32)


class TensorColArgmax32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "tensor_col_argmax_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 64], DataType.FP32, init_value=_rand_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorColArgmax_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.argmax(tensors["a"], dim=0, keepdim=True).to(torch.int32)


# =============================================================================
# pytest entry points
# =============================================================================


class TestTileArgReduce:
    """Tile-level row/col argmax/argmin."""

    def test_row_argmax_fp32(self, test_runner):
        result = test_runner.run(RowArgmax32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_row_argmin_fp32(self, test_runner):
        result = test_runner.run(RowArgmin16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_row_argmax_fp16(self, test_runner):
        result = test_runner.run(RowArgmax32x64FP16())
        assert result.passed, f"Test failed: {result.error}"

    def test_col_argmax_fp32(self, test_runner):
        result = test_runner.run(ColArgmax32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_col_argmin_fp32(self, test_runner):
        result = test_runner.run(ColArgmin8x128FP32())
        assert result.passed, f"Test failed: {result.error}"


class TestTensorArgReduce:
    """Tensor-level pl.row_argmax / pl.col_argmax (lowered via tensor->tile)."""

    def test_tensor_row_argmax_fp32(self, test_runner):
        result = test_runner.run(TensorRowArgmax32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_tensor_col_argmax_fp32(self, test_runner):
        result = test_runner.run(TensorColArgmax32x64FP32())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
