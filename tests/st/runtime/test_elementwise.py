# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based elementwise operations using the @pl.jit frontend.

Verifies that the migrated tile_add_64/tile_add_128/tile_mul_64/tile_mul_128 kernels
from ``examples.kernels.elementwise`` produce results matching torch references on
the platform configured via ``test_config``.
"""

import pytest
import torch
from examples.kernels.elementwise import (
    tile_add_64,
    tile_add_128,
    tile_mul_64,
    tile_mul_128,
)

_ADD_KERNELS = {64: tile_add_64, 128: tile_add_128}
_MUL_KERNELS = {64: tile_mul_64, 128: tile_mul_128}


class TestElementwiseOperations:
    """Test suite for elementwise operations on the configured platform."""

    @pytest.mark.parametrize("size", [64, 128])
    def test_tile_add(self, test_config, size):
        """Test tile addition: c = a + b at the given square size."""
        kernel = _ADD_KERNELS[size]
        kernel._cache.clear()
        a = torch.full((size, size), 2.0, dtype=torch.float32)
        b = torch.full((size, size), 3.0, dtype=torch.float32)
        c = torch.zeros((size, size), dtype=torch.float32)
        kernel(a, b, c, config=test_config)
        expected = a + b
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"tile_add_{size} failed: max diff = {(c - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("size", [64, 128])
    def test_tile_mul(self, test_config, size):
        """Test tile multiplication: c = a * b at the given square size."""
        kernel = _MUL_KERNELS[size]
        kernel._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(size, size, dtype=torch.float32)
        b = torch.full((size, size), 3.0, dtype=torch.float32)
        c = torch.zeros((size, size), dtype=torch.float32)
        kernel(a, b, c, config=test_config)
        expected = a * b
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"tile_mul_{size} failed: max diff = {(c - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
