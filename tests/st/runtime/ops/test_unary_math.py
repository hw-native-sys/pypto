# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level unary math ops: sin, cos, sqrt.

sin/cos are FP32-only and interpret their input as radians. Inputs are kept
positive so sqrt is well-defined on the same data.

Each op is exercised twice:
  * aligned   — valid_shape == static tile shape [M, N]
  * narrow    — valid_shape [VALID_M, VALID_N] < static [M, N]; the store writes
                only the valid sub-region, so the rest of the zero-init output
                must stay 0.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16
VALID_M = 8
VALID_N = 12


def _positive_input() -> torch.Tensor:
    """Positive FP32 input in roughly [0.1, 3.1] — valid for sin/cos/sqrt alike."""
    return torch.rand(M, N, dtype=torch.float32) * 3.0 + 0.1


class TileUnaryMathTestCase(PTOTestCase):
    """Tile sin/cos/sqrt on FP32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(self, *, valid_shapes: tuple[int, int] | None = None, config=None):
        super().__init__(config)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_unary_math_narrow" if self._valid else "tile_unary_math"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_positive_input),
            TensorSpec("sin_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("cos_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("sqrt_o", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class TileUnaryMathProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                sin_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                cos_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                sqrt_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                sin_o = pl.store(pl.tile.sin(a_tile), [0, 0], sin_o)
                cos_o = pl.store(pl.tile.cos(a_tile), [0, 0], cos_o)
                sqrt_o = pl.store(pl.tile.sqrt(a_tile), [0, 0], sqrt_o)
                return sin_o, cos_o, sqrt_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                sin_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                cos_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                sqrt_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                sin_o, cos_o, sqrt_o = self.kernel(a, sin_o, cos_o, sqrt_o)
                return sin_o, cos_o, sqrt_o

        return TileUnaryMathProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        fns = {"sin_o": torch.sin, "cos_o": torch.cos, "sqrt_o": torch.sqrt}
        if self._valid:
            vm, vn = self._valid
            for name, fn in fns.items():
                out = torch.zeros_like(tensors[name])
                out[:vm, :vn] = fn(a[:vm, :vn])
                tensors[name][:] = out
        else:
            for name, fn in fns.items():
                tensors[name][:] = fn(a)


class TestUnaryMath:
    """Tile-level sin/cos/sqrt on a2a3."""

    @pytest.mark.platforms("a2a3")
    def test_tile_unary_math(self, test_runner):
        """Aligned: valid_shape == static [M, N]."""
        result = test_runner.run(TileUnaryMathTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_unary_math_narrow(self, test_runner):
        """Narrow valid_shape [VALID_M, VALID_N]; invalid region stays zero."""
        result = test_runner.run(TileUnaryMathTestCase(valid_shapes=(VALID_M, VALID_N)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
