# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for dynamic shape kernels using the PyPTO frontend with PTO backend.

Three scenarios are covered, each parametrized over [(128, 128)]:
- Dynamic M×N tensor shape dims: trailing index args in codegen, resolved at runtime
  from the concrete input tensors passed by the orchestration function.
- Static R×C tensors with M, N scalar valid_shapes: shape baked in via closure
  variables captured by @pl.function; M and N read via pl.tensor.dim.
- Dynamic M dim with scf.for loop (step=2, tile rows=2): col count from shape param.

All tests use OptimizationStrategy.Default and BackendType.Ascend910B_PTO.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level DynVar names.
# pyright: reportUndefinedVariable=false

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

M = pl.dynamic("M")
N = pl.dynamic("N")

_SHAPES = [(128, 128)]


class DynShapeAddTestCase(PTOTestCase):
    """Test add kernel with fully dynamic M×N tensor shapes.

    Shape (rows, cols) is provided at construction time. The orchestration
    binds M=rows, N=cols by passing concrete tensors of that size.
    Expected result: c = a + b over the full rows×cols tile.
    """

    __test__ = False

    def __init__(self, shape: tuple[int, int], config: RunConfig | None = None):
        super().__init__(config)
        self._rows, self._cols = shape

    def get_name(self) -> str:
        return f"dyn_shape_add_{self._rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Captured as closure variables by @pl.function / @pl.program decorators.
        rows = self._rows
        cols = self._cols

        @pl.program
        class DynShapeAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                """Add two dynamic-shape tensors element-wise."""
                a_tile = pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)
                b_tile = pl.load(b, [0, 0], [rows, cols])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                c_out = self.add_kernel(a, b, c)
                return c_out

        return DynShapeAddProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class ValidShapeAddTestCase(PTOTestCase):
    """Test add kernel with static tensors where valid_shapes are read from an input tensor.

    Shape (rows, cols) is the full tile size; valid_shape (valid_rows, valid_cols)
    is the live data region passed at runtime via a [2] INT64 tensor. The kernel
    loads with valid_shapes=[m, n] so elements outside the valid region are zero.
    Expected result: c[:valid_rows, :valid_cols] = a + b, c elsewhere = 0.
    """

    __test__ = False

    def __init__(
        self,
        shape: tuple[int, int],
        valid_shape: tuple[int, int],
        config: RunConfig | None = None,
    ):
        super().__init__(config)
        self._rows, self._cols = shape
        self._valid_rows, self._valid_cols = valid_shape

    def get_name(self) -> str:
        return f"valid_shape_add_{self._rows}x{self._cols}_valid_{self._valid_rows}x{self._valid_cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec(
                "valid_shape",
                [2],
                DataType.INT64,
                init_value=torch.tensor([self._valid_rows, self._valid_cols], dtype=torch.int64),
            ),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Captured as closure variables by @pl.function / @pl.program decorators.
        rows = self._rows
        cols = self._cols

        @pl.program
        class ValidShapeAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                m: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                """Add two tiles with dynamic valid_shapes [m, n]."""
                a_tile = pl.load(a, [0, 0], [rows, cols], valid_shapes=[m, n])
                b_tile = pl.load(b, [0, 0], [rows, cols], valid_shapes=[m, n])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                vs: pl.Tensor[[2], pl.INDEX],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                m: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [0])
                n: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [1])
                c = self.add_kernel(a, b, c, m, n)
                return c

        return ValidShapeAddProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        vr = tensors["valid_shape"][0]
        vc = tensors["valid_shape"][1]
        tensors["c"][:vr, :vc] = tensors["a"][:vr, :vc] + tensors["b"][:vr, :vc]


class LoopDynShapeAddTestCase(PTOTestCase):
    """Test add kernel with dynamic M dim and scf.for loop (step=2, tile rows=2).

    Shape (rows, cols) is provided at construction time. rows must be divisible by
    2 — the loop processes pairs of rows per iteration. The InCore function reads M
    from the first tensor dimension via pl.tensor.dim.
    Expected result: c = a + b over the full rows×cols tile.
    """

    __test__ = False

    def __init__(self, shape: tuple[int, int], config: RunConfig | None = None):
        super().__init__(config)
        self._rows, self._cols = shape

    def get_name(self) -> str:
        return f"loop_dyn_shape_add_{self._rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Captured as closure variables by @pl.function / @pl.program decorators.
        rows = self._rows
        cols = self._cols

        @pl.program
        class LoopDynShapeAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, cols], pl.FP32],
                b: pl.Tensor[[M, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[M, cols], pl.FP32]],
            ) -> pl.Tensor[[M, cols], pl.FP32]:
                """Iterate over M rows in pairs and add tiles element-wise."""
                M_dim = pl.tensor.dim(a, 0)
                for i in pl.range(0, M_dim, 2):
                    offset = i
                    a_tile = pl.load(a, [offset, 0], [2, cols], target_memory=pl.MemorySpace.Vec)
                    b_tile = pl.load(b, [offset, 0], [2, cols], target_memory=pl.MemorySpace.Vec)
                    result = pl.add(a_tile, b_tile)
                    out = pl.store(result, [offset, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                c_out = self.add_kernel(a, b, c)
                return c_out

        return LoopDynShapeAddProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


# =============================================================================
# pytest test suite
# =============================================================================


class TestDynamicShapeOperations:
    """Test suite for dynamic shape kernel operations."""

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_dyn_shape_add(self, test_runner, shape):
        """Test add with fully dynamic M×N tensor shapes."""
        test_case = DynShapeAddTestCase(shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}: {result.error}"

    @pytest.mark.parametrize("shape,valid_shape", [((128, 128), (64, 64))])
    def test_valid_shape_add(self, test_runner, shape, valid_shape):
        """Test add with static tensors and valid_shapes read from an input tensor."""
        test_case = ValidShapeAddTestCase(shape, valid_shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}, valid_shape {valid_shape}: {result.error}"

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_loop_dyn_shape_add(self, test_runner, shape):
        """Test add with dynamic M dim iterated in pairs via scf.for."""
        test_case = LoopDynShapeAddTestCase(shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
