# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Runtime test for composite (non-bare-Var) shape dimensions in function parameter
types, executed end-to-end via the task-submit runtime.

The parameter types carry a composite dim ``M * 2`` (a ``Mul(Var, ConstInt)``
expression, not a bare ``pl.dynamic`` Var). This exercises two fixes together:

- SSA verifier: dynamic shape vars nested inside a composite parameter dim are
  registered in the outermost scope (otherwise the inner ``M`` reports "used
  outside its defining scope").
- PTO codegen: the constant factor that appears only inside a composite shape
  expression (the ``2`` in ``M * 2``) is declared in the emitted MLIR before the
  ``arith.muli`` that consumes it (otherwise the kernel references an undeclared
  ``%c2_index``).

The kernel adds two tensors element-wise over a static tile, so the inner ``M``
is referenced solely by the parameter type — the realistic shape that surfaced
the bugs. ``rows = 2 * half_rows`` so the declared ``M * 2`` matches the actual
tensor extent.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level DynVar names.
# pyright: reportUndefinedVariable=false

from typing import Any

import pypto.language as pl
import pytest
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M = pl.dynamic("M")
N = pl.dynamic("N")

# (half_rows, cols): the tensor has rows = 2 * half_rows, so the composite
# parameter dim ``M * 2`` resolves with M = half_rows.
_COMPOSITE_SHAPES = [(16, 16)]


class CompositeDimAddTestCase(PTOTestCase):
    """Add kernel whose parameter shapes use a composite dim ``M * 2``."""

    __test__ = False

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        platform: str | None = None,
        config: RunConfig | None = None,
    ):
        super().__init__(config, platform=platform)
        self._half_rows, self._cols = shape

    def get_name(self) -> str:
        return f"composite_dim_add_{self._half_rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        rows = self._half_rows * 2
        return [
            TensorSpec("a", [rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        rows = self._half_rows * 2
        cols = self._cols

        @pl.program
        class CompositeDimAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M * 2, N], pl.FP32],
                b: pl.Tensor[[M * 2, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
            ) -> pl.Tensor[[M * 2, N], pl.FP32]:
                """Add two tensors whose row dim is the composite ``M * 2``."""
                a_tile = pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)
                b_tile = pl.load(b, [0, 0], [rows, cols])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M * 2, N], pl.FP32],
                b: pl.Tensor[[M * 2, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
            ) -> pl.Tensor[[M * 2, N], pl.FP32]:
                c_out = self.add_kernel(a, b, c)
                return c_out

        return CompositeDimAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestCompositeShapeDim:
    """Task-submit execution of composite parameter shape dims."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("shape", _COMPOSITE_SHAPES)
    def test_composite_dim_add(self, test_runner, shape, platform):
        """Add kernel whose parameter row dim is the composite ``M * 2``."""
        result = test_runner.run(CompositeDimAddTestCase(shape, platform=platform))
        assert result.passed, f"Test failed for shape {shape}: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
