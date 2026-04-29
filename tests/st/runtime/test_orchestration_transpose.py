# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for orchestration-level ``pl.transpose`` followed by ``pl.slice``.

Regression coverage for issue #1209: an Orchestration-level transpose used to
lose its swapped strides during type deduction, causing downstream
``tile.load`` to read GM bytes with the wrong strides and return incorrect
data when a transposed tensor was sliced inside a ``CORE_GROUP`` block.

The fix records swapped strides on the result ``TensorType`` of
``tensor.transpose`` so the kernel-side ``pto.make_tensor_view`` emits the
correct DMA strides.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig


class TransposeSliceReshapeAssembleTestCase(PTOTestCase):
    """Issue #1209 reproducer: ``transpose`` then ``slice`` then ``reshape`` then ``assemble``.

    Mathematically this writes ``out[:, :N] = x[:, :N]``: each loop iteration
    slices one row of the transposed tensor (which is one column of ``x``),
    reshapes it to a column tile, and assembles it into the corresponding
    output column.

    Before the fix the orchestration tensor view's swapped strides were dropped,
    so the kernel's ``tile.load`` from the transposed view read with row-major
    strides for the swapped shape — producing garbage at non-zero ``h``.
    """

    __test__ = False

    # Use static shapes for bit-exact validation. T=8 / PAD=8 keeps the tile
    # ``[1, T]`` 32-byte aligned (8 FP32) on Ascend 910B.
    _T = 8
    _PAD = 8
    _N = 4

    def __init__(
        self,
        *,
        platform: str | None = None,
        config: RunConfig | None = None,
    ):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"transpose_slice_reshape_assemble_{self._T}x{self._PAD}_n{self._N}"

    def define_tensors(self) -> list[TensorSpec]:
        # x is filled with arange so wrong reads are immediately visible.
        def init_x():
            return torch.arange(self._T * self._PAD, dtype=torch.float32).reshape(self._T, self._PAD)

        return [
            TensorSpec("x", [self._T, self._PAD], DataType.FP32, init_value=init_x),
            TensorSpec("out", [self._T, self._N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        T = self._T
        PAD = self._PAD
        N = self._N

        @pl.program
        class TransposeSliceReproProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[T, PAD], pl.FP32],
                out: pl.Out[pl.Tensor[[T, N], pl.FP32]],
            ):
                xt = pl.transpose(x, axis1=0, axis2=1)
                for h in pl.range(N):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="slice_transposed_row"):
                        col = pl.reshape(pl.slice(xt, [1, T], [h, 0]), [T, 1])
                        out = pl.assemble(out, col, [0, h])
                return out

        return TransposeSliceReproProgram

    def compute_expected(self, tensors, params=None):
        # out[:, :N] == x[:, :N] (the first N columns of x).
        tensors["out"][:] = tensors["x"][:, : self._N]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_transpose_slice_reshape_assemble(test_runner, platform):
    """Static-shape ``transpose -> slice -> reshape -> assemble`` round-trip.

    Verifies issue #1209: tensor.transpose at orchestration must preserve
    stride semantics so downstream tile.load reads the correct GM bytes.
    """
    result = test_runner.run(TransposeSliceReshapeAssembleTestCase(platform=platform))
    assert result.passed, f"Test failed on {platform}: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
