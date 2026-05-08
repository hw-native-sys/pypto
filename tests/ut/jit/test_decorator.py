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
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import JITFunction, _discover_deps, _rewrite_jit_error, jit
from pypto.jit.specializer import TensorMeta
from pypto.language.parser.diagnostics.exceptions import ParserTypeError
from pypto.pypto_core import DataType, ir

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
    """Cache behavior tests.

    These tests verify L1 cache hit/miss logic by inspecting the internal
    ``_cache`` dict.  They do NOT execute on device (no NPU required).
    """

    def test_cache_hit_same_shape(self):
        """Second call with same shape returns cached program without recompilation."""
        torch = pytest.importorskip("torch")

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

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)

        add_kernel.compile_for_test(a, b, c)
        assert len(add_kernel._cache) == 1
        add_kernel.compile_for_test(a, b, c)
        assert len(add_kernel._cache) == 1  # no new entry — cache hit

    def test_cache_miss_different_shape(self):
        """Different shape causes new compilation."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _add_incore2(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_kernel2(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _add_incore2(a, b, c)
            return c

        a128 = torch.randn(128, 128)
        b128 = torch.randn(128, 128)
        c128 = torch.empty(128, 128)

        a64 = torch.randn(64, 64)
        b64 = torch.randn(64, 64)
        c64 = torch.empty(64, 64)

        add_kernel2.compile_for_test(a128, b128, c128)
        assert len(add_kernel2._cache) == 1
        add_kernel2.compile_for_test(a64, b64, c64)
        assert len(add_kernel2._cache) == 2  # different shape — cache miss

    def test_dynamic_dim_cache_hit_different_concrete_value(self):
        """With bind_dynamic, different M values should hit the same cache entry."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_dyn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def dyn_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_dyn(a, c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.empty(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.empty(512, 128)

        dyn_kernel.compile_for_test(a256, c256)
        assert len(dyn_kernel._cache) == 1
        dyn_kernel.compile_for_test(a512, c512)
        # Both M values → same cache entry (M is dynamic)
        assert len(dyn_kernel._cache) == 1

    def test_dynamic_dim_cache_miss_on_static_dim_change(self):
        """Changing a non-dynamic dim should miss the cache."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_dyn2(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def dyn_kernel2(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_dyn2(a, c)
            return c

        a128 = torch.randn(256, 128)
        c128 = torch.empty(256, 128)
        a256 = torch.randn(256, 256)
        c256 = torch.empty(256, 256)

        dyn_kernel2.compile_for_test(a128, c128)
        assert len(dyn_kernel2._cache) == 1
        dyn_kernel2.compile_for_test(a256, c256)
        # K changed (128 → 256), should be different compilations
        assert len(dyn_kernel2._cache) == 2

    def test_returns_compiled_program(self):
        """JIT compilation should produce a CompiledProgram in the cache."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_simple(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def simple(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_simple(a, c)
            return c

        a = torch.randn(64, 64)
        c = torch.empty(64, 64)
        simple.compile_for_test(a, c)
        cached_values = list(simple._cache.values())
        assert len(cached_values) == 1
        assert isinstance(cached_values[0], CompiledProgram)


class TestMultiFuncDepDiscovery:
    """Multi-function JIT: @pl.jit.incore deps are auto-discovered from entry function globals."""

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


class TestMultiFuncIntegration:
    """End-to-end multi-function @pl.jit compilation with @pl.jit.incore deps."""

    def test_multi_func_parseable(self):
        """@pl.jit with an @pl.jit.incore dep compiles successfully."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = x.shape
            tile_x = pl.load(x, [0, 0], [M, N])
            pl.store(tile_x, [0, 0], out)
            return out

        @jit
        def copy_entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = copy_incore(x, out)
            return out

        x = torch.randn(64, 64)
        out = torch.empty(64, 64)
        copy_entry.compile_for_test(x, out)
        cached_values = list(copy_entry._cache.values())
        assert len(cached_values) == 1
        assert isinstance(cached_values[0], CompiledProgram)

    def test_multi_func_contains_both_functions(self):
        """Compiled program contains both the @jit.incore dep and the @jit entry functions."""
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
        add_entry.compile_for_test(a, b, c)
        compiled = list(add_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)
        func_names = [f.name for f in compiled.program.functions.values()]
        assert "add_incore" in func_names
        assert "add_entry" in func_names

    def test_multi_func_cache_hit(self):
        """Two multi-function JIT calls with same shapes reuse the cached program."""
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
        relu_entry.compile_for_test(x, out)
        assert len(relu_entry._cache) == 1
        relu_entry.compile_for_test(x, out)
        assert len(relu_entry._cache) == 1  # cache hit

    def test_multi_func_structural_equal_to_program(self):
        """Multi-function JIT output matches hand-written @pl.program structurally."""
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
                tile_a = pl.load(a, [0, 0], [32, 32])
                tile_b = pl.load(b, [0, 0], [32, 32])
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
        got = add_entry.compile_for_test(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


class TestInlineFuncIntegration:
    """End-to-end @pl.jit.inline: dep body is spliced into entry by the IR pass."""

    def test_inline_dep_discovered(self):
        """@pl.jit.inline functions are picked up by dep discovery."""

        @jit.inline
        def helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = helper(a, c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 1
        assert deps[0]._func_type == "inline"

    def test_inline_compiled_program_drops_dep_function(self):
        """After compilation, the Inline dep is gone (spliced + removed).

        Inline bodies must include their own ``pl.at`` scope: unlike InCore,
        Inline doesn't provide an implicit scope. The body is spliced into the
        caller's lexical context as-is, then ``OutlineIncoreScopes`` extracts
        the spliced ``pl.at`` block normally.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def add_inline(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_inline(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        # compile_for_test() returns the post-pass IR (after PassManager.Default).
        # CompiledProgram.program is the *pre-pass* IR, so the cache entry would
        # still contain "add_inline"; we must inspect the post-pass return value.
        post_pass = add_entry.compile_for_test(a, b, c)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "add_inline" not in func_names, (
            f"Inline function should have been spliced and removed, got {func_names}"
        )
        assert "add_entry" in func_names

    def test_inline_body_spliced_into_entry(self):
        """The inlined body's tile ops appear inside the (post-outline) entry.

        Detailed structural equality between JIT-with-inline and hand-written
        @pl.program is intentionally not asserted here: when the call site is
        `c = inline(a, b, c)`, the parser SSA-renames the LHS to `c_v1`, and
        the inline pass's substituted return Var does not match `c_v1` at Var
        identity — so a redundant `c_v1 = c` survives. A side-effect call
        (`inline(a, b, c)` — no LHS) avoids the rename but trips the
        InOutUseDiscipline verifier (must read the post-call return). Both
        equivalents are valid IRs, just structurally distinct from a fully
        hand-written equivalent.

        The pass-level tests in tests/ut/ir/transforms/test_inline_functions.py
        cover detailed structural correctness (Phase 0d).
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def add_inline(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_inline(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        post_pass = add_entry.compile_for_test(a, b, c)
        # After OutlineIncoreScopes: entry is Orchestration, inline body became
        # an InCore-class function (AIV/AIC/InCore) named *_incore_*.
        func_names = [f.name for f in post_pass.functions.values()]
        assert "add_inline" not in func_names
        assert "add_entry" in func_names
        assert any("incore" in n for n in func_names), (
            f"Expected an *_incore_* outlined function from the spliced pl.at body, got {func_names}"
        )

    def test_nested_inline_dep_graph(self):
        """A @pl.jit.inline that calls another @pl.jit.inline: both deps must be
        discovered transitively in leaf-first topological order, with the call
        graph recording who calls whom (regression for issue #1302)."""

        @jit.inline
        def leaf(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            return out

        @jit.inline
        def mid(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = leaf(a, out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = mid(a, out)
            return out

        deps_topo, callers_by_id, callees_by_id, _, _ = entry._get_dep_graph()
        assert [d.__name__ for d in deps_topo] == ["leaf", "mid"]
        assert callers_by_id[id(leaf._func)] == [mid._func]
        assert callers_by_id[id(mid._func)] == [entry._func]
        assert callees_by_id[id(entry._func)] == ["mid"]
        assert callees_by_id[id(mid._func)] == ["leaf"]
        assert callees_by_id[id(leaf._func)] == []

    def test_nested_inline_compiles(self):
        """End-to-end repro from issue #1302: an entry that calls an inline
        that calls another inline must compile without raising and the post-pass
        IR must have spliced both inline bodies."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def leaf(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit.inline
        def mid(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = leaf(a, out)
            return out

        @jit
        def entry_nested(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = mid(a, out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry_nested.compile_for_test(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "leaf" not in func_names, f"leaf should be spliced, got {func_names}"
        assert "mid" not in func_names, f"mid should be spliced, got {func_names}"
        assert "entry_nested" in func_names
        assert any("incore" in n for n in func_names), (
            f"Expected an *_incore_* outlined function from the spliced pl.at body, got {func_names}"
        )

    def test_nested_inline_diamond(self):
        """Diamond: entry -> {a_helper, b_helper}, both call the same shared
        leaf. The shared leaf must be deduplicated in the dep graph."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def shared(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit.inline
        def a_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = shared(a, out)
            return out

        @jit.inline
        def b_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = shared(a, out)
            return out

        @jit
        def entry_diamond(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = a_helper(a, out)
            out = b_helper(a, out)
            return out

        deps_topo, callers_by_id, _, _, _ = entry_diamond._get_dep_graph()
        assert len(deps_topo) == 3, f"expected dedup'd shared leaf + a_helper + b_helper, got {deps_topo}"
        assert {d.__name__ for d in deps_topo} == {"shared", "a_helper", "b_helper"}

        # Both diamond branches must record themselves as callers of the
        # shared leaf, so dyn-dim / dynvar propagation visits every branch
        # rather than just the first DFS path.
        shared_callers = callers_by_id[id(shared._func)]
        assert {c.__name__ for c in shared_callers} == {"a_helper", "b_helper"}
        assert callers_by_id[id(a_helper._func)] == [entry_diamond._func]
        assert callers_by_id[id(b_helper._func)] == [entry_diamond._func]

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        # Compilation must succeed without "Unsupported function call" errors,
        # and post-pass IR must have spliced both helpers and the shared leaf.
        post_pass = entry_diamond.compile_for_test(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        for spliced in ("shared", "a_helper", "b_helper"):
            assert spliced not in func_names, f"{spliced} should be spliced, got {func_names}"
        assert "entry_diamond" in func_names

    def test_mixed_positional_and_keyword_call(self):
        """Mixed ``dep(a, out=out)`` calls must preserve both positional and
        keyword bindings so tensor metadata propagates correctly to the dep
        (regression for CodeRabbit review on PR #1314)."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry_mixed(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            # Positional ``a`` plus keyword ``out=out`` — the keyword binding
            # used to be silently dropped by _extract_call_args_for_dep.
            out = helper(a, out=out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry_mixed.compile_for_test(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "helper" not in func_names, f"helper should be spliced, got {func_names}"
        assert "entry_mixed" in func_names


class TestOpaqueFuncIntegration:
    """End-to-end @pl.jit.opaque: dep is emitted as a separate Opaque IR function."""

    def test_opaque_dep_discovered(self):
        """@pl.jit.opaque functions are picked up by dep discovery."""

        @jit.opaque
        def helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = helper(a, c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 1
        assert deps[0]._func_type == "opaque"

    def test_opaque_structural_equal_to_program(self):
        """JIT(@opaque + @jit) ≡ hand-written @pl.program with Opaque sub-function."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Opaque)
            def add_op(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile_a = pl.load(a, [0, 0], [32, 32])
                    tile_b = pl.load(b, [0, 0], [32, 32])
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
                c = self.add_op(a, b, c)
                return c

        @jit.opaque
        def add_op(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_op(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        got = add_entry.compile_for_test(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


class TestRoundTrip:
    """Round-trip: @pl.jit output must be structurally equal to a hand-written @pl.program."""

    def test_elementwise_add_128x128(self):
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
                    tile_a = pl.load(a, [0, 0], [128, 128])
                    tile_b = pl.load(b, [0, 0], [128, 128])
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
        got = tile_add.compile_for_test(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


# ---------------------------------------------------------------------------
# Variable rebinding (Issue #1121)
# ---------------------------------------------------------------------------


class TestVariableRebinding:
    """Tests for Python-style variable rebinding in @pl.jit (Issue #1121)."""

    def test_rebind_same_type_compiles(self):
        """Rebinding a Tile variable to a new Tile value must compile without error."""

        @jit
        def kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                t = pl.mul(t, t)  # rebind: Tile → Tile (same type)
                pl.store(t, [0, 0], out)
            return out

        result = kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_values={},
            scalar_dtypes={},
            per_func_dyn={id(kernel._func): set()},
            pl=pl,
        )
        assert result is not None

    def test_rebind_error_shows_original_name(self):
        """When a JIT compilation error occurs, error messages must show the
        user's original variable name, not the internal renamed alias."""
        rename_map = {"t_v1": "t", "x_v2": "x"}
        exc = ValueError("Variable 't_v1' has type Tile but expected Scalar")
        rewritten = _rewrite_jit_error(exc, rename_map)
        assert "t_v1" not in str(rewritten)
        assert "'t'" in str(rewritten)

    def test_no_rename_map_returns_original_exception(self):
        """With an empty rename map, the original exception object is returned."""
        exc = ValueError("some error")
        result = _rewrite_jit_error(exc, {})
        assert result is exc

    def test_rebind_longer_alias_replaced_first(self):
        """Longer aliases are replaced before shorter ones to avoid partial matches."""
        rename_map = {"t_v1": "t", "t_v10": "t"}
        exc = ValueError("'t_v10' and 't_v1' are both invalid")
        rewritten = _rewrite_jit_error(exc, rename_map)
        assert "t_v10" not in str(rewritten)
        assert "t_v1" not in str(rewritten)

    def test_rewrite_preserves_exception_fields(self):
        """copy.copy preserves extra fields (e.g. message) for ParserError-style exceptions."""
        exc = ParserTypeError("Variable 'x_v1' has wrong type", hint="use x instead")
        result = _rewrite_jit_error(exc, {"x_v1": "x"})
        assert "x_v1" not in str(result)
        assert "x" in str(result)
        # Extra fields are preserved via copy.copy
        assert isinstance(result, ParserTypeError)
        assert result.hint == "use x instead"  # type: ignore[attr-defined]

    def test_rewrite_non_standard_exception_falls_back(self):
        """Exceptions where copy.copy fails fall back to plain Exception."""

        class WeirdError(Exception):
            def __init__(self, code: int, msg: str) -> None:
                super().__init__(msg)
                self.code = code

        exc = WeirdError(42, "Variable 'x_v1' is invalid")
        result = _rewrite_jit_error(exc, {"x_v1": "x"})
        assert "x_v1" not in str(result)
        assert "x" in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
