# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for the @pl.jit decorator (compile + execute on device).

Verifies that ``@pl.jit`` kernels compile, cache, and execute correctly on
NPU hardware via ``CompiledProgram.__call__()``.

Tests cover:
- In-place execution: output tensor modified on device
- Cache hit: same shape reuses compiled kernel
- Cache miss: different shape triggers new compilation
- Multi-shape: different shapes produce correct results
- Dynamic dimensions: ``bind_dynamic`` + ``pl.dynamic`` share one compiled kernel
"""

import pypto.language as pl
import pytest
import torch
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit import jit


class TestJitExecute:
    """Test @pl.jit compile + execute on device."""

    def test_inplace_add(self, test_config):
        """In-place call: jit kernel modifies output tensor on device."""

        @jit.incore
        def _add_incore(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _add_incore(a, b, c)
            return c

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        add_kernel(a, b, c, config=test_config)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"In-place add failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_cache_stores_compiled_program(self, test_config):
        """After __call__, cache contains a CompiledProgram."""

        @jit.incore
        def _copy_incore(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def copy_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore(a, c)
            return c

        a = torch.randn(64, 64)
        c = torch.zeros(64, 64)

        copy_kernel(a, c, config=test_config)

        assert len(copy_kernel._cache) == 1
        cached = list(copy_kernel._cache.values())[0]
        assert isinstance(cached, CompiledProgram)

    def test_cache_hit_same_shape(self, test_config):
        """Second call with same shape reuses cached CompiledProgram."""

        @jit.incore
        def _add_incore_hit(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_kernel_hit(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _add_incore_hit(a, b, c)
            return c

        a = torch.full((128, 128), 1.0, dtype=torch.float32)
        b = torch.full((128, 128), 1.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        add_kernel_hit(a, b, c, config=test_config)
        assert len(add_kernel_hit._cache) == 1

        c2 = torch.zeros((128, 128), dtype=torch.float32)
        add_kernel_hit(a, b, c2, config=test_config)
        assert len(add_kernel_hit._cache) == 1  # cache hit — no new compilation

    def test_cache_miss_different_shape(self, test_config):
        """Different shape triggers new compilation and new cache entry."""

        @jit.incore
        def _add_incore_miss(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_kernel_miss(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _add_incore_miss(a, b, c)
            return c

        a128 = torch.full((128, 128), 1.0, dtype=torch.float32)
        b128 = torch.full((128, 128), 2.0, dtype=torch.float32)
        c128 = torch.zeros((128, 128), dtype=torch.float32)

        a64 = torch.full((64, 64), 1.0, dtype=torch.float32)
        b64 = torch.full((64, 64), 2.0, dtype=torch.float32)
        c64 = torch.zeros((64, 64), dtype=torch.float32)

        add_kernel_miss(a128, b128, c128, config=test_config)
        assert len(add_kernel_miss._cache) == 1

        add_kernel_miss(a64, b64, c64, config=test_config)
        assert len(add_kernel_miss._cache) == 2  # different shape — new entry

        expected128 = torch.full((128, 128), 3.0, dtype=torch.float32)
        expected64 = torch.full((64, 64), 3.0, dtype=torch.float32)
        assert torch.allclose(c128, expected128, rtol=1e-5, atol=1e-5)
        assert torch.allclose(c64, expected64, rtol=1e-5, atol=1e-5)

    def test_dynamic_dim_cache_hit(self, test_config):
        """bind_dynamic: different M values reuse the same compiled kernel."""

        @jit.incore
        def _copy_dyn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def copy_dyn_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_dyn(a, c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.zeros(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.zeros(512, 128)

        copy_dyn_kernel(a256, c256, config=test_config)
        assert len(copy_dyn_kernel._cache) == 1

        copy_dyn_kernel(a512, c512, config=test_config)
        assert len(copy_dyn_kernel._cache) == 1  # M is dynamic — same entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
