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
3. Descending INT32 sequence.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
