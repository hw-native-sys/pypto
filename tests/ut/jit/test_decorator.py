# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit decorator: decoration, cache hit/miss, and bind_dynamic."""

import pypto.language as pl
import pytest
from pypto.jit.decorator import JITFunction, _discover_deps, jit
from pypto.pypto_core import ir

# ---------------------------------------------------------------------------
# Decoration tests (no torch needed)
# ---------------------------------------------------------------------------


class TestJitDecoration:
    def test_plain_jit_creates_jitfunction(self):
        @jit
        def my_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(my_kernel, JITFunction)

    def test_jit_preserves_name(self):
        @jit
        def my_kernel(a: pl.Tensor):
            return a

        assert my_kernel.__name__ == "my_kernel"

    def test_jit_incore_creates_jitfunction(self):
        @jit.incore
        def sub_fn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(sub_fn, JITFunction)
        assert sub_fn._func_type == "incore"

    def test_jit_incore_with_level(self):
        @jit.incore(level=pl.Level.AIC)
        def aic_fn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(aic_fn, JITFunction)
        assert aic_fn._func_type == "incore"
        assert aic_fn._level == pl.Level.AIC

    def test_jit_entry_function_type(self):
        @jit
        def entry(a: pl.Tensor):
            return a

        assert entry._func_type == "orchestration"

    def test_jit_pl_access(self):
        """pl.jit should work the same as jit."""

        @pl.jit
        def kernel(a: pl.Tensor):
            return a

        assert isinstance(kernel, JITFunction)


# ---------------------------------------------------------------------------
# Tensor.bind_dynamic no-op
# ---------------------------------------------------------------------------


class TestBindDynamic:
    def test_bind_dynamic_is_noop(self):
        """Tensor.bind_dynamic() must not raise at runtime."""
        # Annotation-only Tensor
        t = pl.Tensor[[128, 64], pl.FP32]
        M = pl.dynamic("M")
        t.bind_dynamic(0, M)  # should not raise

    def test_bind_dynamic_returns_none(self):
        t = pl.Tensor[[128, 64], pl.FP32]
        M = pl.dynamic("M")
        result = t.bind_dynamic(0, M)
        assert result is None


# ---------------------------------------------------------------------------
# Cache tests (torch-dependent, skipped if torch not available)
# ---------------------------------------------------------------------------


class TestJitCaching:
    def test_cache_hit_same_shape(self):
        """Second call with same shape returns cached program without recompilation."""
        torch = pytest.importorskip("torch")

        @jit
        def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)

        prog1 = add_kernel(a, b, c)
        prog2 = add_kernel(a, b, c)
        assert prog1 is prog2  # same object from cache

    def test_cache_miss_different_shape(self):
        """Different shape causes new compilation."""
        torch = pytest.importorskip("torch")

        @jit
        def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        a128 = torch.randn(128, 128)
        b128 = torch.randn(128, 128)
        c128 = torch.empty(128, 128)

        a64 = torch.randn(64, 64)
        b64 = torch.randn(64, 64)
        c64 = torch.empty(64, 64)

        prog128 = add_kernel(a128, b128, c128)
        prog64 = add_kernel(a64, b64, c64)
        assert prog128 is not prog64

    def test_dynamic_dim_cache_hit_different_concrete_value(self):
        """With bind_dynamic, different M values should hit the same cache entry."""
        torch = pytest.importorskip("torch")

        @jit
        def dyn_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            with pl.at(level=pl.Level.CORE_GROUP):
                K = a.shape[1]
                tile_a = pl.load(a, [0, 0], [M, K])
                pl.store(tile_a, [0, 0], c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.empty(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.empty(512, 128)

        prog256 = dyn_kernel(a256, c256)
        prog512 = dyn_kernel(a512, c512)
        # Both M values → same cache entry (M is dynamic)
        assert prog256 is prog512

    def test_dynamic_dim_cache_miss_on_static_dim_change(self):
        """Changing a non-dynamic dim should miss the cache."""
        torch = pytest.importorskip("torch")

        @jit
        def dyn_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            with pl.at(level=pl.Level.CORE_GROUP):
                K = a.shape[1]
                tile_a = pl.load(a, [0, 0], [M, K])
                pl.store(tile_a, [0, 0], c)
            return c

        a128 = torch.randn(256, 128)
        c128 = torch.empty(256, 128)
        a256 = torch.randn(256, 256)
        c256 = torch.empty(256, 256)

        prog128 = dyn_kernel(a128, c128)
        prog256 = dyn_kernel(a256, c256)
        # K changed (128 → 256), should be different compilations
        assert prog128 is not prog256

    def test_returns_ir_program(self):
        """JIT call should return an ir.Program."""
        torch = pytest.importorskip("torch")

        @jit
        def simple(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                pl.store(tile_a, [0, 0], c)
            return c

        a = torch.randn(64, 64)
        c = torch.empty(64, 64)
        prog = simple(a, c)
        assert isinstance(prog, ir.Program)


class TestStyleBDepDiscovery:
    """Style B: @pl.jit.incore deps are auto-discovered from entry function globals."""

    def test_dep_discovered_from_globals(self):
        """JITFunction.get_deps() finds @pl.jit.incore callees via lazy discovery."""

        # Define both at module scope so that inner is in entry's __globals__
        @jit.incore
        def _inner_dep(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        # Patch _inner_dep into a fresh function's globals to simulate module scope
        import types  # noqa: PLC0415

        def _entry_raw(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _inner_dep(a, c)
            return c

        # Make _inner_dep visible in the function's globals
        new_globals = {**_entry_raw.__globals__, "_inner_dep": _inner_dep}
        entry_raw = types.FunctionType(
            _entry_raw.__code__,
            new_globals,
            _entry_raw.__name__,
            _entry_raw.__defaults__,
            _entry_raw.__closure__,
        )

        entry_fn = JITFunction(entry_raw, func_type="orchestration")
        deps = entry_fn._get_deps()
        dep_names = [d.__name__ for d in deps]
        assert "_inner_dep" in dep_names

    def test_incore_func_type_preserved(self):
        @jit.incore
        def sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert sub._func_type == "incore"

    def test_incore_level_preserved(self):
        @jit.incore(level=pl.Level.AIC)
        def aic_sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert aic_sub._level == pl.Level.AIC

    def test_non_jit_callees_not_in_deps(self):
        """Regular Python functions called from entry are not added as deps."""

        def plain_func(x):
            return x

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = plain_func(c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 0


class TestStyleBIntegration:
    """End-to-end Style B: multi-function @pl.jit compilation."""

    def test_style_b_parseable(self):
        """@pl.jit with an @pl.jit.incore dep produces a valid ir.Program."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            N = x.shape[0]
            tile_x = pl.load(x, [0], [N])
            pl.store(tile_x, [0], out)
            return out

        @jit
        def copy_entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = copy_incore(x, out)
            return out

        x = torch.randn(64)
        out = torch.empty(64)
        prog = copy_entry(x, out)
        assert isinstance(prog, ir.Program)

    def test_style_b_contains_both_functions(self):
        """Generated ir.Program should contain both the dep and entry functions."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def add_incore(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_incore(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        prog = add_entry(a, b, c)
        assert isinstance(prog, ir.Program)
        func_names = [f.name for f in prog.functions]
        assert "add_incore" in func_names
        assert "add_entry" in func_names

    def test_style_b_cache_hit(self):
        """Two Style B calls with same shapes reuse the cached program."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def relu_entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = relu_incore(x, out)
            return out

        x = torch.randn(16, 16)
        out = torch.empty(16, 16)
        p1 = relu_entry(x, out)
        p2 = relu_entry(x, out)
        assert p1 is p2

    def test_style_b_structural_equal_to_program(self):
        """Style B JIT output matches hand-written @pl.program structurally."""
        torch = pytest.importorskip("torch")

        # Hand-written equivalent
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def add_sub(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                M = 32
                N = 32
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def add_entry(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                c = self.add_sub(a, b, c)
                return c

        @jit.incore
        def add_sub(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_sub(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        got = add_entry(a, b, c)
        ir.assert_structural_equal(got, Expected)


class TestRoundTrip:
    """Round-trip: @pl.jit equivalent of examples/ programs must match structural_equal."""

    def test_elementwise_add_128x128(self):
        """JIT version of examples/kernels/01_elementwise.py TileAddProgram."""
        torch = pytest.importorskip("torch")

        @jit
        def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = tile_add(a, b, c)
        # TileAddProgram uses tile_add (InCore) + orchestrator — JIT generates
        # a single Orchestration function wrapping the incore scope; structural
        # equality holds after OutlineIncoreScopes runs in the pass pipeline.
        # We just verify it is a parseable Program here; full pass-level
        # structural equality is deferred to integration tests.
        assert isinstance(got, ir.Program)

    def test_elementwise_add_roundtrip_structural_equal(self):
        """JIT-generated program for 128x128 add is structurally equal to @pl.program."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    M = 128
                    N = 128
                    tile_a = pl.load(a, [0, 0], [M, N])
                    tile_b = pl.load(b, [0, 0], [M, N])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c

        @jit
        def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = tile_add(a, b, c)
        ir.assert_structural_equal(got, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
