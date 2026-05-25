# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for on-core arrays (ArrayType) inside an InCore kernel.

ArrayType lives on the scalar register file / C stack. In the InCore (PTO)
codegen path it lowers to PTOAS's stack-local array triad:

  array.create        -> pto.declare_local_array -> !pto.local_array<NxT>
  array.update_element -> pto.local_array_set arr[i], v : !pto.local_array<NxT>, T
  array.get_element    -> pto.local_array_get arr[i]    : !pto.local_array<NxT> -> T

The kernel below round-trips a row index through an on-core array: it reads a
row index from ``index_t`` into a scalar, stores it into the array, copies it
to a second slot (exercising the SSA-functional update_element -> in-place
alias), reads it back, and uses it as the dynamic store row. The visible effect
is ``dst[row] = src[0]`` — directly verifiable against ``compute_expected``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


@pl.program
class IncoreArrayRoundTripProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        index_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[1, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        index_tile: pl.Tile[[1, 8], pl.INT32] = pl.load(index_t, [0, 0], [1, 8])
        src_tile: pl.Tile[[1, 32], pl.FP32] = pl.load(src_t, [0, 0], [1, 32])
        # Read the target row index off the index tile into a scalar.
        row: pl.Scalar[pl.INT32] = pl.tile.read(index_tile, [0, 0])
        # Round-trip it through an on-core array: set slot 0, copy 0 -> 1
        # (get + in-place set on the SAME backing array), then read slot 1 back.
        arr = pl.array.create(4, pl.INT32)
        arr[0] = row
        arr[1] = arr[0]
        stored: pl.Scalar[pl.INT32] = arr[1]
        row_idx: pl.Scalar[pl.INDEX] = pl.cast(stored, pl.INDEX)
        return pl.store(src_tile, [row_idx, 0], dst_t)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        index_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[1, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        dst_t = self.kernel(index_t, src_t, dst_t)
        return dst_t


class IncoreArrayRoundTripTestCase(PTOTestCase):
    """An on-core array carries a row index from tile.read to the store offset.

    ``dst[row] = src[0]`` where ``row = index_t[0, 0]``, routed through
    declare_local_array / local_array_set / local_array_get.
    """

    def get_name(self) -> str:
        return "incore_array_round_trip"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "index_t",
                [1, 8],
                DataType.INT32,
                init_value=torch.tensor([[7, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32),
            ),
            TensorSpec(
                "src_t",
                [1, 32],
                DataType.FP32,
                init_value=lambda: torch.arange(32, dtype=torch.float32).reshape(1, 32),
            ),
            TensorSpec("dst_t", [32, 32], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return IncoreArrayRoundTripProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.zeros_like(tensors["dst_t"])
        row = int(tensors["index_t"][0, 0].item())
        expected[row] = tensors["src_t"][0]
        tensors["dst_t"][:] = expected


@pl.program
class IncoreArrayConditionalProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        cond_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[1, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        cond_tile: pl.Tile[[1, 8], pl.INT32] = pl.load(cond_t, [0, 0], [1, 8])
        src_tile: pl.Tile[[1, 32], pl.FP32] = pl.load(src_t, [0, 0], [1, 32])
        c: pl.Scalar[pl.INT32] = pl.tile.read(cond_tile, [0, 0])
        # Write the on-core array in both branches — both mutate the same
        # backing storage; the read after the IfStmt sees whichever branch ran.
        arr = pl.array.create(4, pl.INT32)
        if c > 0:
            arr[0] = c
        else:
            arr[0] = 1
        sel: pl.Scalar[pl.INT32] = arr[0]
        row_idx: pl.Scalar[pl.INDEX] = pl.cast(sel, pl.INDEX)
        return pl.store(src_tile, [row_idx, 0], dst_t)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        cond_t: pl.Tensor[[1, 8], pl.INT32],
        src_t: pl.Tensor[[1, 32], pl.FP32],
        dst_t: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        dst_t = self.kernel(cond_t, src_t, dst_t)
        return dst_t


class IncoreArrayConditionalTestCase(PTOTestCase):
    """An array written in both if/else branches drives the store offset.

    ``dst[sel] = src[0]`` where ``sel = c if c > 0 else 1`` and
    ``c = cond_t[0, 0]``. Exercises the IfStmt path where an ArrayType return
    var is merged in place (kept out of the scf.if results).
    """

    def get_name(self) -> str:
        return "incore_array_conditional"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "cond_t",
                [1, 8],
                DataType.INT32,
                init_value=torch.tensor([[5, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32),
            ),
            TensorSpec(
                "src_t",
                [1, 32],
                DataType.FP32,
                init_value=lambda: torch.arange(32, dtype=torch.float32).reshape(1, 32),
            ),
            TensorSpec("dst_t", [32, 32], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return IncoreArrayConditionalProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        expected = torch.zeros_like(tensors["dst_t"])
        c = int(tensors["cond_t"][0, 0].item())
        sel = c if c > 0 else 1
        expected[sel] = tensors["src_t"][0]
        tensors["dst_t"][:] = expected


class TestIncoreArray:
    """System test suite for on-core arrays in InCore kernels."""

    def test_incore_array_round_trip(self, test_runner):
        """A row index round-tripped through an on-core array drives the store offset."""
        result = test_runner.run(IncoreArrayRoundTripTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore_array_conditional(self, test_runner):
        """An array written in both if/else branches merges in place to drive the store."""
        result = test_runner.run(IncoreArrayConditionalTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
