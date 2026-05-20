# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime checks for 2D scalar writes with unrolled column indices."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

ROWS = 16
COLS = 32
TOPK = 2
VALUE_COUNT = ROWS * TOPK


def make_fp32_values() -> torch.Tensor:
    return (torch.arange(VALUE_COUNT, dtype=torch.float32) + 0.5).contiguous()


def make_int32_values() -> torch.Tensor:
    return (torch.arange(VALUE_COUNT, dtype=torch.int32) + 100).contiguous()


@pl.program
class TensorScalarWriteTransposeFP32Program:
    @pl.function(type=pl.FunctionType.InCore)
    def init(
        self,
        table: pl.InOut[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        tile: pl.Tile[[ROWS, COLS], pl.FP32] = pl.tile.full([ROWS, COLS], dtype=pl.FP32, value=-1.0)
        return pl.store(tile, [0, 0], table)

    @pl.function(type=pl.FunctionType.InCore)
    def write_table(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.FP32],
        table: pl.InOut[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        for k in pl.unroll(TOPK):
            for t in pl.unroll(ROWS):
                val: pl.Scalar[pl.FP32] = pl.read(vals, [k * ROWS + t])
                table = pl.write(table, [t, k], val)
        return table

    @pl.function(type=pl.FunctionType.InCore)
    def copy_out(
        self,
        table: pl.Tensor[[ROWS, COLS], pl.FP32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        tile: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(table, [0, 0], [ROWS, COLS])
        return pl.store(tile, [0, 0], dst)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.FP32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        table = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
        table = self.init(table)
        table = self.write_table(vals, table)
        dst = self.copy_out(table, dst)
        return dst


@pl.program
class TileScalarWriteTransposeFP32Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.FP32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        tile: pl.Tile[[ROWS, COLS], pl.FP32] = pl.tile.full([ROWS, COLS], dtype=pl.FP32, value=-1.0)
        for k in pl.unroll(TOPK):
            for t in pl.unroll(ROWS):
                val: pl.Scalar[pl.FP32] = pl.read(vals, [k * ROWS + t])
                pl.write(tile, [t, k], val)
        return pl.store(tile, [0, 0], dst)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.FP32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        dst = self.kernel(vals, dst)
        return dst


@pl.program
class TensorScalarWriteTransposeINT32Program:
    @pl.function(type=pl.FunctionType.InCore)
    def init(
        self,
        table: pl.InOut[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        tile: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.full([ROWS, COLS], dtype=pl.INT32, value=-7)
        return pl.store(tile, [0, 0], table)

    @pl.function(type=pl.FunctionType.InCore)
    def write_table(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.INT32],
        table: pl.InOut[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        for k in pl.unroll(TOPK):
            for t in pl.unroll(ROWS):
                val: pl.Scalar[pl.INT32] = pl.read(vals, [k * ROWS + t])
                table = pl.write(table, [t, k], val)
        return table

    @pl.function(type=pl.FunctionType.InCore)
    def copy_out(
        self,
        table: pl.Tensor[[ROWS, COLS], pl.INT32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        tile: pl.Tile[[ROWS, COLS], pl.INT32] = pl.load(table, [0, 0], [ROWS, COLS])
        return pl.store(tile, [0, 0], dst)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.INT32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        table = pl.create_tensor([ROWS, COLS], dtype=pl.INT32)
        table = self.init(table)
        table = self.write_table(vals, table)
        dst = self.copy_out(table, dst)
        return dst


@pl.program
class TileScalarWriteTransposeINT32Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.INT32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        tile: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.full([ROWS, COLS], dtype=pl.INT32, value=-7)
        for k in pl.unroll(TOPK):
            for t in pl.unroll(ROWS):
                val: pl.Scalar[pl.INT32] = pl.read(vals, [k * ROWS + t])
                pl.write(tile, [t, k], val)
        return pl.store(tile, [0, 0], dst)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        vals: pl.Tensor[[VALUE_COUNT], pl.INT32],
        dst: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        dst = self.kernel(vals, dst)
        return dst


class _FP32WriteBase(PTOTestCase):
    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("vals", [VALUE_COUNT], DataType.FP32, init_value=make_fp32_values),
            TensorSpec("dst", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full((ROWS, COLS), -1.0, dtype=torch.float32)
        vals = tensors["vals"].reshape(TOPK, ROWS)
        for k in range(TOPK):
            expected[:, k] = vals[k]
        tensors["dst"][:] = expected


class _INT32WriteBase(PTOTestCase):
    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("vals", [VALUE_COUNT], DataType.INT32, init_value=make_int32_values),
            TensorSpec("dst", [ROWS, COLS], DataType.INT32, is_output=True),
        ]

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.full((ROWS, COLS), -7, dtype=torch.int32)
        vals = tensors["vals"].reshape(TOPK, ROWS)
        for k in range(TOPK):
            expected[:, k] = vals[k]
        tensors["dst"][:] = expected


class TensorScalarWriteTransposeFP32TestCase(_FP32WriteBase):
    def get_name(self) -> str:
        return "tensor_scalar_write_transpose_fp32"

    def get_program(self) -> Any:
        return TensorScalarWriteTransposeFP32Program


class TileScalarWriteTransposeFP32TestCase(_FP32WriteBase):
    def get_name(self) -> str:
        return "tile_scalar_write_transpose_fp32"

    def get_program(self) -> Any:
        return TileScalarWriteTransposeFP32Program


class TensorScalarWriteTransposeINT32TestCase(_INT32WriteBase):
    def get_name(self) -> str:
        return "tensor_scalar_write_transpose_int32"

    def get_program(self) -> Any:
        return TensorScalarWriteTransposeINT32Program


class TileScalarWriteTransposeINT32TestCase(_INT32WriteBase):
    def get_name(self) -> str:
        return "tile_scalar_write_transpose_int32"

    def get_program(self) -> Any:
        return TileScalarWriteTransposeINT32Program


class TestScalarWriteTranspose:
    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_tensor_scalar_write_transpose_fp32(self, test_runner):
        result = test_runner.run(TensorScalarWriteTransposeFP32TestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_tile_scalar_write_transpose_fp32(self, test_runner):
        result = test_runner.run(TileScalarWriteTransposeFP32TestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_tensor_scalar_write_transpose_int32(self, test_runner):
        result = test_runner.run(TensorScalarWriteTransposeINT32TestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_tile_scalar_write_transpose_int32(self, test_runner):
        result = test_runner.run(TileScalarWriteTransposeINT32TestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
