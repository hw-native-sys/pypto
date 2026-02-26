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

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestMatmul(PTOTestCase):
    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "matmul_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
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
                tile_a_l1 = pl.block.load(
                    a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat
                )
                tile_b_l1 = pl.block.load(
                    b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a = pl.block.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.block.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
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
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmulOperations:
    """Test suite for matrix multiplication (matmul) operations."""

    def test_matmul_64x64(self, test_runner):
        """Test matmul with 64x64 matrices."""
        test_case = TestMatmul()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
