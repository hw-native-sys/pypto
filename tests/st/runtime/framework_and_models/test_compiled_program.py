# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for the @pl.jit -> CompiledProgram callable API.

Verifies that ``@pl.jit`` decorated functions specialize on first call,
populate the per-function ``_cache`` with a ``CompiledProgram``, and execute
correctly on the configured platform. The exposed call style is in-place
(``kernel(a, b, c, config=...)`` writes the result into ``c``); the
underlying ``CompiledProgram`` object (cached on first call) is also
inspected to verify metadata and the ability to call it directly in
return-style.
"""

import pytest
import torch
from examples.kernels.elementwise import tile_add_128, tile_mul_128
from pypto.ir.compiled_program import CompiledProgram


def _get_cached_compiled(jit_fn) -> CompiledProgram:
    """Return the single CompiledProgram cached on a JITFunction.

    Asserts that exactly one entry is present so the helper is unambiguous.
    """
    assert len(jit_fn._cache) == 1, f"expected one cache entry, got {len(jit_fn._cache)}"
    return next(iter(jit_fn._cache.values()))


class TestJitCompiledProgram:
    """Test the @pl.jit -> CompiledProgram pipeline (in-place + return-style)."""

    def test_inplace_add(self, test_config):
        """In-place call: tile_add_128(a, b, c) modifies c on device."""
        tile_add_128._cache.clear()

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        tile_add_128(a, b, c, config=test_config)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"In-place add failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_first_call_populates_cache(self, test_config):
        """First @pl.jit invocation specializes and caches a CompiledProgram."""
        tile_add_128._cache.clear()
        assert len(tile_add_128._cache) == 0

        a = torch.full((128, 128), 1.0, dtype=torch.float32)
        b = torch.full((128, 128), 2.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        tile_add_128(a, b, c, config=test_config)

        assert len(tile_add_128._cache) == 1
        compiled = _get_cached_compiled(tile_add_128)
        assert isinstance(compiled, CompiledProgram)

    def test_return_style_via_compiled(self, test_config):
        """Return-style call on the cached CompiledProgram allocates the output."""
        tile_add_128._cache.clear()

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        # Trigger specialization + caching via in-place call.
        tile_add_128(a, b, c, config=test_config)

        compiled = _get_cached_compiled(tile_add_128)
        # Return-style: omit the output tensor; CompiledProgram allocates it.
        c_out = compiled(a, b, config=test_config)

        assert c_out is not None, "Return-style call should return a tensor"
        assert isinstance(c_out, torch.Tensor)
        assert c_out.shape == (128, 128)
        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c_out, expected, rtol=1e-5, atol=1e-5), (
            f"Return-style add failed: max diff = {(c_out - expected).abs().max().item()}"
        )

    def test_inplace_mul(self, test_config):
        """In-place multiplication: tile_mul_128(a, b, c) writes c = a * b."""
        tile_mul_128._cache.clear()

        a = torch.full((128, 128), 4.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        tile_mul_128(a, b, c, config=test_config)

        expected = torch.full((128, 128), 12.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"In-place mul failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_compile_once_run_twice(self, test_config):
        """Two calls with the same shape/dtype hit the cache once and run twice."""
        tile_add_128._cache.clear()

        a1 = torch.full((128, 128), 1.0, dtype=torch.float32)
        b1 = torch.full((128, 128), 2.0, dtype=torch.float32)
        c1 = torch.zeros((128, 128), dtype=torch.float32)
        tile_add_128(a1, b1, c1, config=test_config)
        assert torch.allclose(c1, torch.full((128, 128), 3.0), rtol=1e-5, atol=1e-5)

        # Second execution: 10 + 20 = 30. Cache entry must already exist.
        cache_size_before = len(tile_add_128._cache)
        a2 = torch.full((128, 128), 10.0, dtype=torch.float32)
        b2 = torch.full((128, 128), 20.0, dtype=torch.float32)
        c2 = torch.zeros((128, 128), dtype=torch.float32)
        tile_add_128(a2, b2, c2, config=test_config)
        assert torch.allclose(c2, torch.full((128, 128), 30.0), rtol=1e-5, atol=1e-5)
        assert len(tile_add_128._cache) == cache_size_before, (
            "Second call with same spec should reuse the cached CompiledProgram"
        )

    def test_metadata_extraction(self, test_config):
        """The cached CompiledProgram exposes correct param/output metadata."""
        tile_add_128._cache.clear()
        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        tile_add_128(a, b, c, config=test_config)

        compiled = _get_cached_compiled(tile_add_128)
        # tile_add_128 has params (a, b, c-as-Out); only c is auto-allocated.
        assert "a" in compiled.param_names
        assert "b" in compiled.param_names
        assert len(compiled.output_indices) == 1
        assert compiled.has_return is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
