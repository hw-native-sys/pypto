# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test product reduction operations: row_prod (TROWPROD), col_prod (TCOLPROD).

Covers multiple shapes and dtypes:
- Shapes: [32, 64] (tall), [16, 16] (square), [8, 128] (wide)
- Dtypes: FP32, FP16

row_prod reduces along axis=1 ([M, N] -> [M, 1]) and requires a tmp_tile scratch
buffer, mirroring row_sum. col_prod reduces along axis=0 ([M, N] -> [1, N]) and
takes a single argument, mirroring col_max / col_min.

Inputs use a small magnitude (uniform around 1.0) so the running product stays in
a numerically stable range across the reduced dimension.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


def _prod_init(shape):
    """Return a no-arg init callable (harness calls init_value with no args).

    Values in [0.8, 1.2) keep the reduced product well-conditioned across the
    reduced dimension; the harness casts the result to the spec dtype.
    """
    return lambda: torch.rand(shape) * 0.4 + 0.8


# =============================================================================
# Programs — row_prod (requires tmp_tile, like row_sum)
# =============================================================================


@pl.program
class RowProd_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.FP32] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowProd_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
            [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowProd_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 1], pl.FP32]],
    ) -> pl.Tensor[[8, 1], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        tmp: pl.Tile[[8, 128], pl.FP32] = pl.tile.create(
            [8, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[8, 1], pl.FP32] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 1], pl.FP32]],
    ) -> pl.Tensor[[8, 1], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowProd_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP16]],
    ) -> pl.Tensor[[32, 1], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP16] = pl.tile.create(
            [32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.FP16] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP16]],
    ) -> pl.Tensor[[32, 1], pl.FP16]:
        output = self.kernel(input_tensor, output)
        return output


# =============================================================================
# Programs — col_prod (single arg, like col_max / col_min)
# =============================================================================


@pl.program
class ColProd_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColProd_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColProd_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        result: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColProd_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP16] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        output = self.kernel(input_tensor, output)
        return output


# =============================================================================
# Test Cases — row_prod
# =============================================================================


class RowProd32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_prod_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_prod_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=1, keepdim=True)


class RowProd16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_prod_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=_prod_init([16, 16])),
            TensorSpec("output", [16, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=1, keepdim=True)


class RowProd8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_prod_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=_prod_init([8, 128])),
            TensorSpec("output", [8, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=1, keepdim=True)


class RowProd32x64FP16(PTOTestCase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FP16 product over 64 elements accumulates per-element rounding;
        # the default 1e-5 tolerance is far too tight for a running product.
        self.config.rtol = 2e-2
        self.config.atol = 2e-2

    def get_name(self) -> str:
        return "row_prod_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=_prod_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_32x64_FP16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=1, keepdim=True)


# =============================================================================
# Test Cases — col_prod
# =============================================================================


class ColProd32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_prod_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_prod_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=0, keepdim=True)


class ColProd16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_prod_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=_prod_init([16, 16])),
            TensorSpec("output", [1, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=0, keepdim=True)


class ColProd8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_prod_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=_prod_init([8, 128])),
            TensorSpec("output", [1, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=0, keepdim=True)


class ColProd32x64FP16(PTOTestCase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FP16 product accumulates per-element rounding; loosen the tolerance.
        self.config.rtol = 2e-2
        self.config.atol = 2e-2

    def get_name(self) -> str:
        return "col_prod_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=_prod_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_32x64_FP16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=0, keepdim=True)


# =============================================================================
# Tests
# =============================================================================


class TestRowProd:
    """row_prod: row-wise product across different shapes and dtypes."""

    def test_32x64_fp32(self, test_runner):
        result = test_runner.run(RowProd32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32(self, test_runner):
        result = test_runner.run(RowProd16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32(self, test_runner):
        result = test_runner.run(RowProd8x128FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16(self, test_runner):
        result = test_runner.run(RowProd32x64FP16())
        assert result.passed, f"Test failed: {result.error}"


class TestColProd:
    """col_prod: column-wise product across different shapes and dtypes."""

    def test_32x64_fp32(self, test_runner):
        result = test_runner.run(ColProd32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32(self, test_runner):
        result = test_runner.run(ColProd16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32(self, test_runner):
        result = test_runner.run(ColProd8x128FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16(self, test_runner):
        result = test_runner.run(ColProd32x64FP16())
        assert result.passed, f"Test failed: {result.error}"


# =============================================================================
# Coverage — narrow valid_shape (aligned + non-aligned) and BF16
#
# row_prod with a narrow valid_col keeps the [M, 1] output fully valid (only the
# reduced extent shrinks); col_prod with a narrow valid_row keeps [1, N] output
# fully valid. This exercises the valid-region codegen (validRow/validCol
# propagation and the row-reduction tmp stride) which the full-tile cases above
# do not. A non-32B-aligned valid_col (50) additionally exercises the unaligned
# stride path. BF16 is col_prod-only (TROWPROD does not list bfloat16).
# =============================================================================


@pl.program
class RowProd_ValidCol48_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64], valid_shapes=[32, 48])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.FP32] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class RowProd_ValidCol50_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64], valid_shapes=[32, 50])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[32, 1], pl.FP32] = pl.tile.row_prod(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColProd_ValidRow20_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64], valid_shapes=[20, 64])
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColProd_32x64_BF16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.BF16],
        output: pl.Out[pl.Tensor[[1, 64], pl.BF16]],
    ) -> pl.Tensor[[1, 64], pl.BF16]:
        tile: pl.Tile[[32, 64], pl.BF16] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.BF16] = pl.tile.col_prod(tile)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.BF16],
        output: pl.Out[pl.Tensor[[1, 64], pl.BF16]],
    ) -> pl.Tensor[[1, 64], pl.BF16]:
        output = self.kernel(input_tensor, output)
        return output


class RowProdValidCol48FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_prod_valid_col48_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_prod_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_ValidCol48_FP32

    def compute_expected(self, tensors, params=None):
        # Only the first 48 columns are valid → reduce that window.
        tensors["output"][:] = torch.prod(tensors["input_tensor"][:, :48], dim=1, keepdim=True)


class RowProdValidCol50FP32(PTOTestCase):
    def get_name(self) -> str:
        return "row_prod_valid_col50_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_prod_init([32, 64])),
            TensorSpec("output", [32, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowProd_ValidCol50_FP32

    def compute_expected(self, tensors, params=None):
        # Non-32B-aligned valid_col: reduce the first 50 columns.
        tensors["output"][:] = torch.prod(tensors["input_tensor"][:, :50], dim=1, keepdim=True)


class ColProdValidRow20FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_prod_valid_row20_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=_prod_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_ValidRow20_FP32

    def compute_expected(self, tensors, params=None):
        # Only the first 20 rows are valid → reduce that window.
        tensors["output"][:] = torch.prod(tensors["input_tensor"][:20, :], dim=0, keepdim=True)


class ColProdBF16(PTOTestCase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BF16 product accumulates per-element rounding; loosen the tolerance.
        self.config.rtol = 2e-2
        self.config.atol = 2e-2

    def get_name(self) -> str:
        return "col_prod_32x64_bf16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.BF16, init_value=_prod_init([32, 64])),
            TensorSpec("output", [1, 64], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColProd_32x64_BF16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.prod(tensors["input_tensor"], dim=0, keepdim=True)


class TestProdCoverage:
    """Narrow valid_shape (aligned + non-aligned) and BF16 coverage."""

    def test_row_prod_valid_col48_fp32(self, test_runner):
        result = test_runner.run(RowProdValidCol48FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_row_prod_valid_col50_fp32(self, test_runner):
        result = test_runner.run(RowProdValidCol50FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_col_prod_valid_row20_fp32(self, test_runner):
        result = test_runner.run(ColProdValidRow20FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_col_prod_32x64_bf16(self, test_runner):
        result = test_runner.run(ColProdBF16())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
