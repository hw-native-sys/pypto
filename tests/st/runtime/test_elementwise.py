# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based elementwise operations using the PyPTO frontend.

This module defines integration tests for elementwise add and multiply
kernels implemented with the internal PTOTestCase harness.  Each test case
accepts an optional ``platform`` parameter so a single class can run
on multiple platforms via ``@pytest.mark.parametrize``.
"""

from typing import Any

import pytest
import torch
from examples.kernels.elementwise import (
    TileAdd64Program,
    TileAdd128Program,
    TileMul64Program,
    TileMul128Program,
)
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


class TileAddTestCase(PTOTestCase):
    """Test case for tile element-wise addition."""

    __test__ = False

    def __init__(self, size: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.size = size

    def get_name(self) -> str:
        return f"tile_add_{self.size}x{self.size}"

    def define_tensors(self) -> list[TensorSpec]:
        s = self.size
        return [
            TensorSpec("a", [s, s], DataType.FP32, init_value=2.0),
            TensorSpec("b", [s, s], DataType.FP32, init_value=3.0),
            TensorSpec("c", [s, s], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAdd128Program if self.size == 128 else TileAdd64Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TileMulTestCase(PTOTestCase):
    """Test case for tile element-wise multiplication."""

    __test__ = False

    def __init__(self, size: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.size = size

    def get_name(self) -> str:
        return f"tile_mul_{self.size}x{self.size}"

    def define_tensors(self) -> list[TensorSpec]:
        s = self.size
        return [
            TensorSpec("a", [s, s], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [s, s], DataType.FP32, init_value=3.0),
            TensorSpec("c", [s, s], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileMul128Program if self.size == 128 else TileMul64Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


# =============================================================================
# pytest test functions
# =============================================================================

_SIZES = [64, 128]


class TestElementwiseOperations:
    """Test suite for elementwise operations across all platforms."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("size", _SIZES)
    def test_tile_add(self, test_runner, platform, size):
        """Test tile addition with configurable shape and platform."""
        result = test_runner.run(TileAddTestCase(size=size, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("size", _SIZES)
    def test_tile_mul(self, test_runner, platform, size):
        """Test tile multiplication with configurable shape and platform."""
        result = test_runner.run(TileMulTestCase(size=size, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
