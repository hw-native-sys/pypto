# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for matrix multiplication operation using PyPTO frontend.

This test validates the matmul operation implementation through the
pto-testing-framework, ensuring correct code generation and execution.
"""

from typing import Any, List

import numpy as np
import pypto.language as pl
import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestMatmul(PTOTestCase):
    __test__ = False  # Not a pytest test class

    def __init__(self, rows: int = 64, cols: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"matmul_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.block.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=2)
                tile_b_l1 = pl.block.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=2)
                tile_a_l0a = pl.block.move(tile_a_l1, target_memory=3)
                tile_b_l0b = pl.block.move(tile_b_l1, target_memory=4)
                tile_c_l0c = pl.block.matmul(tile_a_l0a, tile_b_l0b)
                # store can support l0c -> GM directly
                out_c = pl.block.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.matmul(a, b)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.matmul(tensors["a"], tensors["b"])


class TestMatmulOperations:
    """Test suite for elementwise operations."""

    @pytest.mark.parametrize("rows,cols", [(64, 64)])
    def test_matmul_shapes(self, test_runner, rows, cols):
        """Test tile addition with various shapes."""
        test_case = TestMatmul(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"
