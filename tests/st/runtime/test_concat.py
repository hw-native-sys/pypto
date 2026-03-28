# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.concat (column-wise concatenation).
"""

from typing import Any

import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.language.beginner.concat import TileConcat32x32Program


class TileConcatTestCase(PTOTestCase):
    """Test case for tile column-wise concatenation (32x16 + 32x16 -> 32x32)."""

    def get_name(self) -> str:
        return "tile_concat_32x32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 16], DataType.FP32, init_value=1.0),
            TensorSpec("b", [32, 16], DataType.FP32, init_value=2.0),
            TensorSpec("c", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileConcat32x32Program

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        tensors["c"][:, :16] = tensors["a"]
        tensors["c"][:, 16:] = tensors["b"]


class TestConcatOperations:
    """Test suite for concat operations."""

    @pytest.mark.skip(reason="PTOAS doesn't support tconcat now.")
    def test_tile_concat_32x32(self, test_runner):
        """Test tile concatenation: 32x16 + 32x16 -> 32x32."""
        test_case = TileConcatTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
