# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test broadcast expand max/min/expdif ops:
  row_expand_max/min/expdif (TROWEXPAND*) — per-row scalar from a [M, 1] vector
  col_expand_max/min/expdif (TCOLEXPAND*) — per-column scalar from a [1, N] vector

Semantics (a = tile, b = broadcast vector):
  *_max     -> torch.maximum(a, b)
  *_min     -> torch.minimum(a, b)
  *_expdif  -> torch.exp(a - b)

Inputs use a near-1 magnitude so the exp-diff stays well-conditioned (a - b is
small) and dtype rounding is bounded.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

M, N = 32, 64


def _near_one(shape):
    # Values in [0.8, 1.2): keeps max/min meaningful and exp(a - b) bounded.
    return lambda: torch.rand(shape) * 0.4 + 0.8


# =============================================================================
# Programs — row expand (vector is [M, 1])
# =============================================================================


@pl.program
class RowExpandMax:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[M, 1], pl.FP32] = pl.load(v, [0, 0], [M, 1])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand_max(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


@pl.program
class RowExpandMin:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[M, 1], pl.FP32] = pl.load(v, [0, 0], [M, 1])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand_min(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


@pl.program
class RowExpandExpdif:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[M, 1], pl.FP32] = pl.load(v, [0, 0], [M, 1])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand_expdif(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[M, 1], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


# =============================================================================
# Programs — col expand (vector is [1, N])
# =============================================================================


@pl.program
class ColExpandMax:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[1, N], pl.FP32] = pl.load(v, [0, 0], [1, N])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand_max(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


@pl.program
class ColExpandMin:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[1, N], pl.FP32] = pl.load(v, [0, 0], [1, N])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand_min(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


@pl.program
class ColExpandExpdif:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tv: pl.Tile[[1, N], pl.FP32] = pl.load(v, [0, 0], [1, N])
        tc: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand_expdif(ta, tv)
        return pl.store(tc, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


# =============================================================================
# Test cases
# =============================================================================


class _ExpandCase(PTOTestCase):
    # NOTE: the @pl.program kernels below are FP32-typed, so these cases are
    # FP32-only. FP16 coverage would require separate FP16 @pl.program classes
    # (dtype is fixed at parse time and cannot be parametrized at runtime).
    program: Any = None
    vec_shape: list = [M, 1]
    op_name: str = ""

    def get_name(self) -> str:
        return f"{self.op_name}_fp32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_near_one([M, N])),
            TensorSpec("v", self.vec_shape, DataType.FP32, init_value=_near_one(self.vec_shape)),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return self.program

    def _torch_op(self, a, b):
        raise NotImplementedError

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = self._torch_op(tensors["a"], tensors["v"])


class RowMaxCase(_ExpandCase):
    program = RowExpandMax
    vec_shape = [M, 1]
    op_name = "row_expand_max"

    def _torch_op(self, a, b):
        return torch.maximum(a, b)


class RowMinCase(_ExpandCase):
    program = RowExpandMin
    vec_shape = [M, 1]
    op_name = "row_expand_min"

    def _torch_op(self, a, b):
        return torch.minimum(a, b)


class RowExpdifCase(_ExpandCase):
    program = RowExpandExpdif
    vec_shape = [M, 1]
    op_name = "row_expand_expdif"

    def _torch_op(self, a, b):
        return torch.exp(a - b)


class ColMaxCase(_ExpandCase):
    program = ColExpandMax
    vec_shape = [1, N]
    op_name = "col_expand_max"

    def _torch_op(self, a, b):
        return torch.maximum(a, b)


class ColMinCase(_ExpandCase):
    program = ColExpandMin
    vec_shape = [1, N]
    op_name = "col_expand_min"

    def _torch_op(self, a, b):
        return torch.minimum(a, b)


class ColExpdifCase(_ExpandCase):
    program = ColExpandExpdif
    vec_shape = [1, N]
    op_name = "col_expand_expdif"

    def _torch_op(self, a, b):
        return torch.exp(a - b)


# =============================================================================
# Tests
# =============================================================================

_CASES = [RowMaxCase, RowMinCase, RowExpdifCase, ColMaxCase, ColMinCase, ColExpdifCase]


class TestExpandOps:
    """row/col expand max/min/expdif on a2a3 (FP32)."""

    @pytest.mark.parametrize("case", _CASES, ids=[c.op_name for c in _CASES])
    def test_fp32(self, test_runner, case):
        result = test_runner.run(case())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
