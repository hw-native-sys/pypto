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

import ast

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


# Dynamic user-batch dim declared directly in the annotation (no bind_dynamic),
# matching @pl.program style. user_batch is read at runtime via pl.tensor.dim,
# so one compiled artifact serves any batch. Reproduces gh#1508.
USER_BATCH = pl.dynamic("USER_BATCH")
_COLS = 128
_BATCH_CAP = 32
_ROW_TILE = 16


@pl.jit
def copy_dyn_batch(
    a: pl.Tensor[[USER_BATCH, _COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[USER_BATCH, _COLS], pl.FP32]],
):
    user_batch = pl.tensor.dim(a, 0)
    for b0 in pl.parallel(0, _BATCH_CAP, _ROW_TILE):
        cur_valid = pl.min(_ROW_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP):
            chunk = pl.slice(a, [_ROW_TILE, _COLS], [b0, 0], valid_shape=[cur_valid, _COLS])
            out = pl.assemble(out, chunk, [b0, 0])
    return out


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

    def test_emits_debug_run_script(self, test_config):
        """JIT compile must emit a self-contained ``debug/run.py`` re-runner.

        This is the JIT-side guarantee for the unified debug-replay workflow:
        any ``build_output/<jit_dir>/`` ships with a runnable script so the
        user does not have to choose between the replay CLI and hand-written
        Python — see ``pypto.runtime.debug.run_script_writer``.
        """
        add_kernel._cache.clear()

        a = torch.full((128, 128), 1.0, dtype=torch.float32)
        b = torch.full((128, 128), 2.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        add_kernel(a, b, c, config=test_config)

        (compiled,) = add_kernel._cache.values()
        run_script = compiled.output_dir / "debug" / "run.py"
        assert run_script.exists(), f"Missing auto-emitted debug runner at {run_script}"

        text = run_script.read_text()
        # Syntactic validity — a broken file would surprise the user on first try.
        ast.parse(text)
        # JIT has no golden.py, so inline inputs must be present.
        assert "_inline_inputs" in text
        # Shape / dtype derived from the kernel signature.
        assert "torch.randn((128, 128)" in text
        assert "torch.zeros((128, 128)" in text


class TestJITDynamicBatch:
    """A pl.dynamic() user-batch dim in the annotation serves variable batch (gh#1508)."""

    def test_one_artifact_serves_multiple_batches(self, test_config):
        """Compiling at one batch must not specialize the dynamic dim to a constant.

        Two different runtime batches (both <= BATCH_CAP) hit a single compiled
        artifact and copy correctly — pre-fix, codegen pinned user_batch to the
        first concrete value, blocking smaller batches.
        """
        copy_dyn_batch._cache.clear()

        a32 = torch.randn(_BATCH_CAP, _COLS, dtype=torch.float32)
        out32 = torch.zeros(_BATCH_CAP, _COLS, dtype=torch.float32)
        copy_dyn_batch(a32, out32, config=test_config)
        assert torch.allclose(out32, a32, rtol=1e-5, atol=1e-5)
        assert len(copy_dyn_batch._cache) == 1

        a16 = torch.randn(16, _COLS, dtype=torch.float32)
        out16 = torch.zeros(16, _COLS, dtype=torch.float32)
        copy_dyn_batch(a16, out16, config=test_config)
        assert torch.allclose(out16, a16, rtol=1e-5, atol=1e-5)
        # Same dynamic artifact serves the smaller batch — no recompilation.
        assert len(copy_dyn_batch._cache) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
