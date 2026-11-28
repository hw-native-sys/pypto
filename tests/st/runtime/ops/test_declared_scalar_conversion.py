# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Executable values for the declared-slot scalar conversion.

The conversion contract is a statement about *values*
(`docs/en/user/language/00-types.md`, "Declared scalar dtypes convert"), so it is
checked by running, not by inspecting IR. Both directions of the
admitted lattice are covered, and the first case sits on the exact wrap boundary:

  IndexToInt32WrapTestCase : an INDEX value of 2**31 reaches an ``INT32``
                              parameter as -2**31 (truncate, two's complement).
  Int32ToIndexTestCase     : an ``INT32`` value reaches an ``INDEX`` parameter
                              (zero/sign extension) and reads back unchanged.

The scalar is always derived at the call site — a loop induction variable in the
first case, a typed literal in the second — so the argument's dtype genuinely
differs from the declaration and the conversion is what puts it right.
"""

from typing import Any

import pypto.language as pl
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

N = 8
WRAP_BOUNDARY = 2**31


def _ones() -> torch.Tensor:
    return torch.ones(1, N, dtype=torch.int32)


@pl.program
class IndexToInt32WrapProgram:
    """``r + 2**31`` is INDEX; the declared parameter is INT32."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[1, N], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT32]],
    ) -> pl.Tensor[[1, N], pl.INT32]:
        tile = pl.load(x, [0, 0], [1, N])
        out = pl.store(pl.add(tile, rank), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT32]],
    ) -> pl.Tensor[[1, N], pl.INT32]:
        for r in pl.range(1):
            out = self.kernel(x, r + WRAP_BOUNDARY, out)
        return out


@pl.program
class Int32ToIndexProgram:
    """A typed INT32 literal crosses into a declared INDEX slot."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[1, N], pl.INT32],
        offset: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[1, N], pl.INT32]],
    ) -> pl.Tensor[[1, N], pl.INT32]:
        tile = pl.load(x, [0, 0], [1, N])
        out = pl.store(pl.add(tile, pl.cast(offset, pl.INT32)), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT32]],
    ) -> pl.Tensor[[1, N], pl.INT32]:
        out = self.kernel(x, pl.const(5, pl.INT32), out)
        return out


class IndexToInt32WrapTestCase(PTOTestCase):
    """INDEX(2**31) into an INT32 parameter is -2**31 on the device."""

    def get_name(self) -> str:
        return "declared_scalar_index_to_int32_wrap"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [1, N], DataType.INT32, init_value=_ones),
            TensorSpec("out", [1, N], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return IndexToInt32WrapProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None) -> None:
        wrapped = torch.tensor(-WRAP_BOUNDARY, dtype=torch.int32)
        tensors["out"][:] = tensors["x"] + wrapped


class Int32ToIndexTestCase(PTOTestCase):
    """An INT32 literal into a declared INDEX parameter keeps its value."""

    def get_name(self) -> str:
        return "declared_scalar_int32_to_index"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [1, N], DataType.INT32, init_value=_ones),
            TensorSpec("out", [1, N], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Int32ToIndexProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None) -> None:
        tensors["out"][:] = tensors["x"] + 5


class TestDeclaredScalarConversion:
    """The conversion produces the specified values on the device path."""

    def test_index_into_int32_wraps_at_the_boundary(self, test_runner):
        result = test_runner.run(IndexToInt32WrapTestCase())
        assert result.passed, f"INDEX -> INT32 conversion failed: {result.error}"

    def test_int32_into_index_keeps_its_value(self, test_runner):
        result = test_runner.run(Int32ToIndexTestCase())
        assert result.passed, f"INT32 -> INDEX conversion failed: {result.error}"
