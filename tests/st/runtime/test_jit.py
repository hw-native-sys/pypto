# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for the @pl.jit decorator end-to-end execution.

Verifies that ``@pl.jit``-decorated functions compile on first call,
serve from cache on subsequent calls, and execute correctly on device.
"""

import pypto.language as pl
import pytest
import torch
from pypto.ir.compiled_program import CompiledProgram


@pl.jit
def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        M, N = a.shape
        tile_a = pl.load(a, [0, 0], [M, N])
        tile_b = pl.load(b, [0, 0], [M, N])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


class TestJITExecution:
    """End-to-end tests for @pl.jit compile + execute on device."""

    def test_inplace_add(self, test_config):
        """@pl.jit: first call compiles and executes correctly on device."""
        add_kernel._cache.clear()

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        add_kernel(a, b, c, config=test_config)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"JIT add failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_cache_hit_reuses_compiled_program(self, test_config):
        """Second call with same shape hits L1 cache and still produces correct output."""
        add_kernel._cache.clear()

        a1 = torch.full((128, 128), 1.0, dtype=torch.float32)
        b1 = torch.full((128, 128), 2.0, dtype=torch.float32)
        c1 = torch.zeros((128, 128), dtype=torch.float32)
        add_kernel(a1, b1, c1, config=test_config)

        assert len(add_kernel._cache) == 1
        cached = next(iter(add_kernel._cache.values()))
        assert isinstance(cached, CompiledProgram)

        # Second call — same shape → cache hit
        a2 = torch.full((128, 128), 10.0, dtype=torch.float32)
        b2 = torch.full((128, 128), 20.0, dtype=torch.float32)
        c2 = torch.zeros((128, 128), dtype=torch.float32)
        add_kernel(a2, b2, c2, config=test_config)

        assert len(add_kernel._cache) == 1, "Cache should still have exactly one entry"
        assert torch.allclose(c2, torch.full((128, 128), 30.0), rtol=1e-5, atol=1e-5), (
            f"Cache-hit execution failed: max diff = {(c2 - torch.full((128, 128), 30.0)).abs().max().item()}"
        )

    def test_cache_miss_different_shape(self, test_config):
        """Different shape triggers recompilation and executes correctly."""
        add_kernel._cache.clear()

        a1 = torch.full((128, 128), 1.0, dtype=torch.float32)
        b1 = torch.full((128, 128), 1.0, dtype=torch.float32)
        c1 = torch.zeros((128, 128), dtype=torch.float32)
        add_kernel(a1, b1, c1, config=test_config)

        a2 = torch.full((64, 64), 3.0, dtype=torch.float32)
        b2 = torch.full((64, 64), 4.0, dtype=torch.float32)
        c2 = torch.zeros((64, 64), dtype=torch.float32)
        add_kernel(a2, b2, c2, config=test_config)

        assert len(add_kernel._cache) == 2, "Different shape should produce a second cache entry"
        expected = torch.full((64, 64), 7.0, dtype=torch.float32)
        assert torch.allclose(c2, expected, rtol=1e-5, atol=1e-5), (
            f"Recompiled execution failed: max diff = {(c2 - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
