# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level activation ops: relu, lrelu.

Both need signed inputs to exercise the negative branch. lrelu computes
``max(x, slope * x)`` (leaky ReLU) with a scalar slope.

Each op is exercised aligned (valid_shape == [M, N]) and narrow
(valid_shape [VALID_M, VALID_N] < [M, N]; invalid output region stays zero).

(prelu is intentionally omitted: its 3-arg DSL form ``prelu(tile, slope, tmp)``
mismatches the codegen ``pto.tprelu`` which expects 2 arguments — tracked as a
known gap rather than worked around here.)
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
LRELU_SLOPE = 0.1

# This test set targets a2a3 only; a5 coverage is handled in a separate PR.
A2A3 = [pytest.param("a2a3", id="a2a3")]


def _signed_input() -> torch.Tensor:
    return torch.randn(M, N, dtype=torch.float32)


class TileActivationTestCase(PTOTestCase):
    """Tile relu/lrelu on signed FP32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(
        self, *, valid_shapes: tuple[int, int] | None = None, platform: str | None = None, config=None
    ):
        super().__init__(config, platform=platform)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_activation_narrow" if self._valid else "tile_activation"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_signed_input()),
            TensorSpec("relu_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("lrelu_o", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class TileActivationProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                relu_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                lrelu_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                relu_o = pl.store(pl.tile.relu(a_tile), [0, 0], relu_o)
                lrelu_o = pl.store(pl.tile.lrelu(a_tile, LRELU_SLOPE), [0, 0], lrelu_o)
                return relu_o, lrelu_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                relu_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                lrelu_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                relu_o, lrelu_o = self.kernel(a, relu_o, lrelu_o)
                return relu_o, lrelu_o

        return TileActivationProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        fns = {"relu_o": torch.relu, "lrelu_o": lambda t: torch.maximum(t, LRELU_SLOPE * t)}
        if self._valid:
            vm, vn = self._valid
            for name, fn in fns.items():
                out = torch.zeros_like(tensors[name])
                out[:vm, :vn] = fn(a[:vm, :vn])
                tensors[name][:] = out
        else:
            for name, fn in fns.items():
                tensors[name][:] = fn(a)


class TestActivation:
    """Tile-level relu/lrelu on a2a3."""

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_activation(self, test_runner, platform):
        """Aligned: valid_shape == static [M, N]."""
        result = test_runner.run(TileActivationTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_activation_narrow(self, test_runner, platform):
        """Narrow valid_shape [VALID_M, VALID_N]; invalid region stays zero."""
        result = test_runner.run(TileActivationTestCase(valid_shapes=(VALID_M, VALID_N), platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
