# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test column-wise reduction operations: col_sum, col_max, col_min.

Covers multiple shapes and dtypes:
- Shapes: [32, 64] (tall), [16, 16] (square), [8, 128] (wide)
- Dtypes: FP32, FP16

col_sum accepts an optional tmp_tile argument. Passing tmp_tile activates
the binary-tree reduction path (TCOLSUM 4-arg form); omitting it uses the
sequential reduction path (TCOLSUM 2-arg form).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# =============================================================================
# Programs — col_sum (tmp_tile optional; provide it for binary-tree reduction)
# =============================================================================


@pl.program
class ColSum_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.tile.create(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_sum(tile, tmp)
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
class ColSum_32x64_FP32_Sequential:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_sum(tile)
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
class ColSum_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
            [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_sum(tile, tmp)
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
class ColSum_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        tmp: pl.Tile[[8, 128], pl.FP32] = pl.tile.create(
            [8, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_sum(tile, tmp)
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
class ColSum_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        tmp: pl.Tile[[32, 64], pl.FP16] = pl.tile.create(
            [32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, 64], pl.FP16] = pl.tile.col_sum(tile, tmp)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        output = self.kernel(input_tensor, output)
        return output


@pl.program
class ColSum_16x16_FP32_Sequential:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_sum(tile)
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
class ColSum_8x128_FP32_Sequential:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        result: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_sum(tile)
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
class ColSum_32x64_FP16_Sequential:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP16] = pl.tile.col_sum(tile)
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
# Programs — col_max
# =============================================================================


@pl.program
class ColMax_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_max(tile)
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
class ColMax_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_max(tile)
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
class ColMax_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        result: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_max(tile)
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
class ColMax_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP16] = pl.tile.col_max(tile)
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
# Programs — col_min
# =============================================================================


@pl.program
class ColMin_32x64_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        tile: pl.Tile[[32, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP32] = pl.tile.col_min(tile)
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
class ColMin_16x16_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_min(tile)
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
class ColMin_8x128_FP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        tile: pl.Tile[[8, 128], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 128])
        result: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_min(tile)
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
class ColMin_32x64_FP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[32, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[1, 64], pl.FP16]],
    ) -> pl.Tensor[[1, 64], pl.FP16]:
        tile: pl.Tile[[32, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [32, 64])
        result: pl.Tile[[1, 64], pl.FP16] = pl.tile.col_min(tile)
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
# Test Cases — col_sum
# =============================================================================


class ColSum32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum32x64FP32Sequential(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_32x64_fp32_sequential"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_32x64_FP32_Sequential

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum32x64FP16(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_32x64_FP16

    def compute_expected(self, tensors, params=None):
        # Simulate binary-tree reduction in FP16 to match hardware TCOLSUM behavior
        inp = tensors["input_tensor"]
        rows = inp.shape[0]
        buf = inp.clone()
        cnt = rows
        while cnt > 1:
            half = cnt // 2
            buf[:half] = (buf[0 : 2 * half : 2] + buf[1 : 2 * half : 2]).half()
            if cnt % 2 == 1:
                buf[0] = (buf[0] + buf[cnt - 1]).half()
            cnt = half
        tensors["output"][:] = buf[0:1]


class ColSum16x16FP32Sequential(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_16x16_fp32_sequential"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_16x16_FP32_Sequential

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum8x128FP32Sequential(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_8x128_fp32_sequential"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_8x128_FP32_Sequential

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.sum(tensors["input_tensor"], dim=0, keepdim=True)


class ColSum32x64FP16Sequential(PTOTestCase):
    def get_name(self) -> str:
        return "col_sum_32x64_fp16_sequential"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSum_32x64_FP16_Sequential

    def compute_expected(self, tensors, params=None):
        # Sequential reduction in FP16: accumulate rows one by one
        inp = tensors["input_tensor"]
        acc = inp[0].clone()
        for i in range(1, inp.shape[0]):
            acc = (acc + inp[i]).half()
        tensors["output"][:] = acc.unsqueeze(0)


# =============================================================================
# Test Cases — col_max
# =============================================================================


class ColMax32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_max_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMax_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.max(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMax16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_max_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMax_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.max(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMax8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_max_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMax_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.max(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMax32x64FP16(PTOTestCase):
    def get_name(self) -> str:
        return "col_max_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMax_32x64_FP16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.max(tensors["input_tensor"], dim=0, keepdim=True)[0]


# =============================================================================
# Test Cases — col_min
# =============================================================================


class ColMin32x64FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_min_32x64_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMin_32x64_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.min(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMin16x16FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_min_16x16_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMin_16x16_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.min(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMin8x128FP32(PTOTestCase):
    def get_name(self) -> str:
        return "col_min_8x128_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [8, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [1, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMin_8x128_FP32

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.min(tensors["input_tensor"], dim=0, keepdim=True)[0]


class ColMin32x64FP16(PTOTestCase):
    def get_name(self) -> str:
        return "col_min_32x64_fp16"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [32, 64], DataType.FP16, init_value=torch.randn),
            TensorSpec("output", [1, 64], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMin_32x64_FP16

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.min(tensors["input_tensor"], dim=0, keepdim=True)[0]


# =============================================================================
# Tests
# =============================================================================


class TestColSum:
    """col_sum: column-wise sum across different shapes and dtypes."""

    def test_32x64_fp32(self, test_runner):
        result = test_runner.run(ColSum32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32(self, test_runner):
        result = test_runner.run(ColSum16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32(self, test_runner):
        result = test_runner.run(ColSum8x128FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16(self, test_runner):
        result = test_runner.run(ColSum32x64FP16())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp32_sequential(self, test_runner):
        result = test_runner.run(ColSum32x64FP32Sequential())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32_sequential(self, test_runner):
        result = test_runner.run(ColSum16x16FP32Sequential())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32_sequential(self, test_runner):
        result = test_runner.run(ColSum8x128FP32Sequential())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16_sequential(self, test_runner):
        result = test_runner.run(ColSum32x64FP16Sequential())
        assert result.passed, f"Test failed: {result.error}"


class TestColMax:
    """col_max: column-wise maximum across different shapes and dtypes."""

    def test_32x64_fp32(self, test_runner):
        result = test_runner.run(ColMax32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32(self, test_runner):
        result = test_runner.run(ColMax16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32(self, test_runner):
        result = test_runner.run(ColMax8x128FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16(self, test_runner):
        result = test_runner.run(ColMax32x64FP16())
        assert result.passed, f"Test failed: {result.error}"


class TestColMin:
    """col_min: column-wise minimum across different shapes and dtypes."""

    def test_32x64_fp32(self, test_runner):
        result = test_runner.run(ColMin32x64FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_16x16_fp32(self, test_runner):
        result = test_runner.run(ColMin16x16FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_8x128_fp32(self, test_runner):
        result = test_runner.run(ColMin8x128FP32())
        assert result.passed, f"Test failed: {result.error}"

    def test_32x64_fp16(self, test_runner):
        result = test_runner.run(ColMin32x64FP16())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
