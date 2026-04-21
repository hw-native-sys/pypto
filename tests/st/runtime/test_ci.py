# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test tile.ci / tensor.ci (arange) contiguous integer sequence generation.

Covers:
1. Ascending INT32 sequence (start=0).
2. Ascending INT32 sequence with non-zero start.
3. Descending INT32 sequence (tile.ci).
4. tensor.ci ascending (lowers to tile.ci via conversion pass).
5. tensor.ci descending.
6. pl.tile.arange alias.
7. pl.tensor.arange alias.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

ROWS = 1
COLS = 32
N = COLS


# --- Programs ---


@pl.program
class CiAscendStart0Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        seq: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.ci(0, [ROWS, COLS], dtype=pl.INT32)
        out: pl.Tensor[[ROWS, COLS], pl.INT32] = pl.store(seq, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        output = self.kernel(output)
        return output


@pl.program
class CiAscendStart10Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        seq: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.ci(10, [ROWS, COLS], dtype=pl.INT32)
        out: pl.Tensor[[ROWS, COLS], pl.INT32] = pl.store(seq, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        output = self.kernel(output)
        return output


@pl.program
class CiDescendingProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        seq: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.ci(
            N - 1, [ROWS, COLS], dtype=pl.INT32, descending=True
        )
        out: pl.Tensor[[ROWS, COLS], pl.INT32] = pl.store(seq, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        output = self.kernel(output)
        return output


@pl.program
class CiTensorAscendProgram:
    """tensor.ci — Opaque main + pl.at(CORE_GROUP) + pl.assemble writes result into Out."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            seq = pl.tensor.ci(0, [ROWS, COLS], dtype=pl.INT32)
            output = pl.assemble(output, seq, [0, 0])
        return output


@pl.program
class CiTensorDescendingProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            seq = pl.tensor.ci(N - 1, [ROWS, COLS], dtype=pl.INT32, descending=True)
            output = pl.assemble(output, seq, [0, 0])
        return output


@pl.program
class TileArangeAliasProgram:
    """pl.tile.arange should be the alias of pl.tile.ci."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        seq: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.arange(0, [ROWS, COLS], dtype=pl.INT32)
        out: pl.Tensor[[ROWS, COLS], pl.INT32] = pl.store(seq, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        output = self.kernel(output)
        return output


@pl.program
class TileArangeDescendingProgram:
    """pl.tile.arange descending — alias of pl.tile.ci with descending=True."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        seq: pl.Tile[[ROWS, COLS], pl.INT32] = pl.tile.arange(
            N - 1, [ROWS, COLS], dtype=pl.INT32, descending=True
        )
        out: pl.Tensor[[ROWS, COLS], pl.INT32] = pl.store(seq, offsets=[0, 0], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        output = self.kernel(output)
        return output


@pl.program
class TensorArangeAliasProgram:
    """pl.tensor.arange should be the alias of pl.tensor.ci."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            seq = pl.tensor.arange(N - 1, [ROWS, COLS], dtype=pl.INT32, descending=True)
            output = pl.assemble(output, seq, [0, 0])
        return output


# --- Test Cases ---


class _CiBaseTestCase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class CiAscendStart0TestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "ci_ascend_start0"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return CiAscendStart0Program

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(0, N, dtype=torch.int32).reshape(ROWS, COLS)


class CiAscendStart10TestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "ci_ascend_start10"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return CiAscendStart10Program

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(10, 10 + N, dtype=torch.int32).reshape(ROWS, COLS)


class CiDescendingTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "ci_descending"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return CiDescendingProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(N - 1, -1, -1, dtype=torch.int32).reshape(ROWS, COLS)


class CiTensorAscendTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "ci_tensor_ascend"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return CiTensorAscendProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(0, N, dtype=torch.int32).reshape(ROWS, COLS)


class CiTensorDescendingTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "ci_tensor_descending"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return CiTensorDescendingProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(N - 1, -1, -1, dtype=torch.int32).reshape(ROWS, COLS)


class TileArangeAliasTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "tile_arange_alias"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return TileArangeAliasProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(0, N, dtype=torch.int32).reshape(ROWS, COLS)


class TileArangeDescendingTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "tile_arange_descending"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return TileArangeDescendingProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(N - 1, -1, -1, dtype=torch.int32).reshape(ROWS, COLS)


class TensorArangeAliasTestCase(_CiBaseTestCase):
    def get_name(self) -> str:
        return "tensor_arange_alias"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True)]

    def get_program(self) -> Any:
        return TensorArangeAliasProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.arange(N - 1, -1, -1, dtype=torch.int32).reshape(ROWS, COLS)


# --- Tests ---


class TestCi:
    """Verify tile.ci / tensor.ci produce correct integer sequences on device."""

    def test_ci_ascend_start0(self, test_runner):
        result = test_runner.run(CiAscendStart0TestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_ci_ascend_start10(self, test_runner):
        result = test_runner.run(CiAscendStart10TestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_ci_descending(self, test_runner):
        result = test_runner.run(CiDescendingTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_ci_tensor_ascend(self, test_runner):
        result = test_runner.run(CiTensorAscendTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_ci_tensor_descending(self, test_runner):
        result = test_runner.run(CiTensorDescendingTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_arange_alias(self, test_runner):
        result = test_runner.run(TileArangeAliasTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_arange_descending(self, test_runner):
        result = test_runner.run(TileArangeDescendingTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_tensor_arange_alias(self, test_runner):
        result = test_runner.run(TensorArangeAliasTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
