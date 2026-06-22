# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level fused ternary ops.

addc(a, b, c)  = a + b + c    (TADDC)
subc(a, b, c)  = a - b - c    (TSUBC)
addsc(a, s, c) = a + s + c    (TADDSC, middle operand is a scalar)
subsc(a, s, c) = a - s - c    (TSUBSC, middle operand is a scalar)

Each op is exercised aligned (valid_shape == [M, N]) and narrow
(valid_shape [VALID_M, VALID_N] < [M, N]; invalid output region stays zero).

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
SCALAR = 2.5


class FusedTernaryTestCase(PTOTestCase):
    """Tile addc/subc/addsc/subsc on FP32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(self, *, valid_shapes: tuple[int, int] | None = None, config=None):
        super().__init__(config)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_fused_ternary_narrow" if self._valid else "tile_fused_ternary"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("addc_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("subc_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("addsc_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("subsc_o", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class FusedTernaryProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Tensor[[M, N], pl.FP32],
                addc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                subc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                addsc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                subsc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                b_tile = pl.load(b, [0, 0], [M, N], valid_shapes=vshape)
                c_tile = pl.load(c, [0, 0], [M, N], valid_shapes=vshape)
                addc_o = pl.store(pl.tile.addc(a_tile, b_tile, c_tile), [0, 0], addc_o)
                subc_o = pl.store(pl.tile.subc(a_tile, b_tile, c_tile), [0, 0], subc_o)
                addsc_o = pl.store(pl.tile.addsc(a_tile, SCALAR, c_tile), [0, 0], addsc_o)
                subsc_o = pl.store(pl.tile.subsc(a_tile, SCALAR, c_tile), [0, 0], subsc_o)
                return addc_o, subc_o, addsc_o, subsc_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Tensor[[M, N], pl.FP32],
                addc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                subc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                addsc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                subsc_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                addc_o, subc_o, addsc_o, subsc_o = self.kernel(a, b, c, addc_o, subc_o, addsc_o, subsc_o)
                return addc_o, subc_o, addsc_o, subsc_o

        return FusedTernaryProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, b, c = tensors["a"], tensors["b"], tensors["c"]
        fns = {
            "addc_o": lambda: a + b + c,
            "subc_o": lambda: a - b - c,
            "addsc_o": lambda: a + SCALAR + c,
            "subsc_o": lambda: a - SCALAR - c,
        }
        if self._valid:
            vm, vn = self._valid
            for name, fn in fns.items():
                out = torch.zeros_like(tensors[name])
                out[:vm, :vn] = fn()[:vm, :vn]
                tensors[name][:] = out
        else:
            for name, fn in fns.items():
                tensors[name][:] = fn()


class TestFusedTernary:
    """Tile-level fused ternary ops on a2a3."""

    @pytest.mark.platforms("a2a3")
    def test_tile_fused_ternary(self, test_runner):
        """Aligned: valid_shape == static [M, N]."""
        result = test_runner.run(FusedTernaryTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_fused_ternary_narrow(self, test_runner):
        """Narrow valid_shape [VALID_M, VALID_N]; invalid region stays zero."""
        result = test_runner.run(FusedTernaryTestCase(valid_shapes=(VALID_M, VALID_N)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
