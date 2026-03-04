# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Column-offset pl.view system test for PyPTO.

Verifies that pl.view supports a non-zero column offset [0, col_offset].
All existing examples use row offsets [cur_offset, 0]; this test exercises
the orthogonal case.

Program:
  Input  x : [16, 64]  FP32
  left   = pl.view(x, [16, 32], [0,  0])   — column offset 0  (baseline)
  right  = pl.view(x, [16, 32], [0, 32])   — column offset 32 (under test)
  out    = left + right                     — InCore add kernel
  Output : [16, 32]  FP32,  out[i][j] = x[i][j] + x[i][j+32]
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# ---------------------------------------------------------------------------
# Program definition
# ---------------------------------------------------------------------------


@pl.program
class ColOffsetViewProgram:
    """Split [16,64] column-wise via pl.view, then add both halves."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        """Element-wise add of two [16,32] tiles."""
        tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 32])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[16, 32])
        tile_sum = pl.add(tile_a, tile_b)
        return pl.store(tile_sum, offsets=[0, 0], output_tensor=out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[16, 64], pl.FP32],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        """View x column-wise and add the two halves.

        left  = pl.view(x, [16, 32], [0,  0])  — zero column offset
        right = pl.view(x, [16, 32], [0, 32])  — non-zero column offset (under test)
        """
        left: pl.Tensor[[16, 32], pl.FP32] = pl.view(x, [16, 32], [0, 0])
        right: pl.Tensor[[16, 32], pl.FP32] = pl.view(x, [16, 32], [0, 32])
        out: pl.Tensor[[16, 32], pl.FP32] = pl.create_tensor([16, 32], dtype=pl.FP32)
        out = self.kernel_add(left, right, out)
        return out


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


class ColOffsetView(PTOTestCase):
    """pl.view column-offset test: out[i][j] = x[i][j] + x[i][j+32]."""

    __test__ = False

    def get_name(self) -> str:
        return "col_offset_view_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColOffsetViewProgram

    def compute_expected(self, tensors, params=None):
        """out[i][j] = x[i][j] + x[i][j+32]"""
        x = tensors["x"]  # [16, 64]
        left = x[:, :32]  # [16, 32]
        right = x[:, 32:]  # [16, 32]
        tensors["out"][:] = left + right


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


class TestColOffsetView:
    """System test: pl.view with non-zero column offset [0, col_offset]."""

    def test_col_offset_view(self, test_runner):
        """Verify pl.view([0, 32]) column offset produces correct results."""
        test_case = ColOffsetView()
        result = test_runner.run(test_case)
        assert result.passed, f"Column-offset pl.view failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
