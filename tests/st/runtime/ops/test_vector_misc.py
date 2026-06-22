# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for assorted tile-level vector ops without a dedicated home:

muls            tile * scalar
divs            tile / scalar
col_expand_add  tile + broadcast(col_vec[1, N]) over rows

Each op is exercised aligned (valid_shape == [M, N]) and narrow
(valid_shape [VALID_M, VALID_N] < [M, N]; invalid output region stays zero).

(expands and sum are intentionally omitted: ``tile.expands`` and ``tile.sum``
have no codegen registered — tracked as known gaps rather than worked around
here.)
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
MULS_RHS = 2.5
DIVS_RHS = 2.0

# This test set targets a2a3 only; a5 coverage is handled in a separate PR.
A2A3 = [pytest.param("a2a3", id="a2a3")]


class VectorMiscTestCase(PTOTestCase):
    """muls/divs/col_expand_add on FP32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(
        self, *, valid_shapes: tuple[int, int] | None = None, platform: str | None = None, config=None
    ):
        super().__init__(config, platform=platform)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_vector_misc_narrow" if self._valid else "tile_vector_misc"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("muls_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("divs_o", [M, N], DataType.FP32, is_output=True),
            TensorSpec("col_add_o", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]
        col_vshape = [1, self._valid[1]] if self._valid else [1, N]

        @pl.program
        class VectorMiscProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                col_vec: pl.Tensor[[1, N], pl.FP32],
                muls_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                divs_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                col_add_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                col_tile = pl.load(col_vec, [0, 0], [1, N], valid_shapes=col_vshape)
                muls_o = pl.store(pl.tile.muls(a_tile, MULS_RHS), [0, 0], muls_o)
                divs_o = pl.store(pl.tile.divs(a_tile, DIVS_RHS), [0, 0], divs_o)
                col_add_o = pl.store(pl.tile.col_expand_add(a_tile, col_tile), [0, 0], col_add_o)
                return muls_o, divs_o, col_add_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                col_vec: pl.Tensor[[1, N], pl.FP32],
                muls_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                divs_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                col_add_o: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
                pl.Tensor[[M, N], pl.FP32],
            ]:
                muls_o, divs_o, col_add_o = self.kernel(a, col_vec, muls_o, divs_o, col_add_o)
                return muls_o, divs_o, col_add_o

        return VectorMiscProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, col_vec = tensors["a"], tensors["col_vec"]
        fns = {
            "muls_o": lambda: a * MULS_RHS,
            "divs_o": lambda: a / DIVS_RHS,
            "col_add_o": lambda: a + col_vec,  # [1, N] broadcast over rows
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


class TestVectorMisc:
    """Assorted tile-level vector ops on a2a3."""

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_vector_misc(self, test_runner, platform):
        """Aligned: valid_shape == static [M, N]."""
        result = test_runner.run(VectorMiscTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_vector_misc_narrow(self, test_runner, platform):
        """Narrow valid_shape [VALID_M, VALID_N]; invalid region stays zero."""
        result = test_runner.run(VectorMiscTestCase(valid_shapes=(VALID_M, VALID_N), platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
