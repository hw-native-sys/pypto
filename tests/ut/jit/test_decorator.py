# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit decorator: decoration, cache hit/miss, and bind_dynamic."""

import importlib
import inspect
import warnings

import pypto.language as pl
import pytest
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import (
    JITFunction,
    _discover_deps,
    _extract_local_tensor_metas,
    _rewrite_jit_error,
    _run_config_compile_kwargs,
    _scan_dynamic_dims,
    jit,
)
from pypto.jit.specializer import TensorMeta
from pypto.language.parser.diagnostics.exceptions import ParserTypeError
from pypto.pypto_core import DataType, ir
from pypto.runtime.runner import RunConfig

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
# @pl.jit.host decoration
# ---------------------------------------------------------------------------


class TestJitHostDecoration:
    """@pl.jit.host produces a HOST-Orchestrator JITFunction."""

    def test_jit_host_creates_jitfunction(self):
        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"

    def test_jit_host_parens_form(self):
        @jit.host()
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"

    def test_jit_host_rejects_level_kwarg(self):
        with pytest.raises(TypeError, match="does not accept a level= argument"):

            @jit.host(level=pl.Level.HOST)
            def host_orch(a: pl.Tensor):
                return a

    def test_jit_host_preserves_name(self):
        @jit.host
        def my_host(a: pl.Tensor):
            return a

        assert my_host.__name__ == "my_host"

    def test_jit_host_pl_access(self):
        @pl.jit.host
        def host_orch(a: pl.Tensor):
            return a

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"


class TestHostDiscoversOrchestrationDep:
    """Host entries discover chip-level orchestrators as deps; other entries don't."""

    def test_host_discovers_orchestration_dep(self):
        @jit
        def chip_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return chip_orch(a, c)

        deps = host_orch._get_deps()
        assert len(deps) == 1
        assert deps[0]._func_type == "orchestration"
        assert deps[0].__name__ == "chip_orch"

    def test_orchestration_entry_does_not_discover_orchestration_dep(self):
        """A plain @pl.jit entry must still ignore @pl.jit deps — only
        sub-functions (incore/inline/opaque) are discovered. Otherwise two
        top-level kernels would silently fold into one program."""

        @jit
        def other_orch(a: pl.Tensor):
            return a

        @jit
        def entry(a: pl.Tensor):
            return other_orch(a)

        deps = entry._get_deps()
        assert deps == []

    def test_host_still_discovers_subfunction_deps(self):
        """Sub-function discovery is unchanged for host entries."""

        @jit.incore
        def sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return sub(a, c)

        deps = host_orch._get_deps()
        assert len(deps) == 1
        assert deps[0]._func_type == "incore"


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

    def test_annotation_dynvar_scanned_without_bind_dynamic(self):
        """A pl.dynamic() var in the annotation marks the dim dynamic — no bind_dynamic.

        Mirrors @pl.program semantics so the same kernel works in either style.
        """
        M = pl.dynamic("M")

        @jit
        def ann_kernel(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            c = a
            return c

        dims = _scan_dynamic_dims(ann_kernel._func, ann_kernel._param_names())
        assert ("a", 0) in dims
        assert ("c", 0) in dims
        # Static dim (128) must stay static.
        assert ("a", 1) not in dims

    def test_annotation_dynamic_dim_cache_hit_different_concrete_value(self):
        """Annotation-declared dynamic dim: different M values hit the same cache entry."""
        torch = pytest.importorskip("torch")

        M = pl.dynamic("M")

        @jit.incore
        def _copy_incore_ann(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def ann_dyn_kernel(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            c = _copy_incore_ann(a, c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.empty(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.empty(512, 128)

        ann_dyn_kernel.compile_for_test(a256, c256)
        assert len(ann_dyn_kernel._cache) == 1
        ann_dyn_kernel.compile_for_test(a512, c512)
        # Both M values → same cache entry (M is dynamic via annotation alone).
        assert len(ann_dyn_kernel._cache) == 1

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


# Module-level @pl.jit.incore kernel reused by the metadata-tracking tests below.
# Defined at module level (not inside a test method) so it can be imported by
# the unit test for ``_extract_local_tensor_metas``; the integration tests
# below redefine their own deps inside the method to keep each test isolated.
@jit.incore
def _relu_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    M, N = x.shape
    t = pl.load(x, [0, 0], [M, N])
    r = pl.relu(t)
    pl.store(r, [0, 0], out)
    return out


def _slice_then_dep_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function used only by the _extract_local_tensor_metas unit test."""
    view = pl.slice(src, [16, 8], [0, 0])
    buf = pl.create_tensor([16, 8], dtype=pl.FP32)
    mid = _relu_kernel(view, buf)
    out = _relu_kernel(mid, out)
    return out


def _runtime_slice_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function: a pl.slice whose width is a runtime scalar."""
    valid_len = pl.tensor.read(cfg, [0])
    view = pl.slice(src, [16, valid_len], [0, 0])
    out = _relu_kernel(view, out)
    return out


def _annotated_slice_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function: pl.slice / pl.create_tensor / dep-call locals
    written with annotated assignments (``v: T = ...``), the common DSL style."""
    view: pl.Tensor = pl.slice(src, [16, 8], [0, 0])
    buf: pl.Tensor = pl.create_tensor([16, 8], dtype=pl.FP32)
    mid: pl.Tensor = _relu_kernel(view, buf)
    out = _relu_kernel(mid, out)
    return out


class TestSliceAndDepReturnMetadata:
    """Regression tests: the JIT specializer must track tensor metadata for
    ``pl.slice`` views and ``@pl.jit.incore`` return values when they flow into
    subsequent kernels (KNOWN_ISSUES: "JIT specializer doesn't track pl.slice
    results or @pl.jit.incore return-value tensor metadata")."""

    def test_extract_local_tensor_metas_slice_and_dep_return(self):
        """``_extract_local_tensor_metas`` infers metas for pl.slice views,
        pl.create_tensor locals, and @pl.jit.incore call results."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_slice_then_dep_body, seed_meta=seed)
        # pl.slice view: shape from the literal list, dtype inherited from src.
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # pl.create_tensor: unchanged behaviour.
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # @pl.jit.incore call result: inherits the dep's pl.Out param meta,
        # which here maps to ``buf``.
        assert metas["mid"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_extract_local_tensor_metas_runtime_slice_uses_parent_dim(self):
        """A pl.slice dim that isn't a static int falls back to the parent
        tensor's static dim (the slice is bounded above by its parent)."""
        seed = {
            "src": TensorMeta(shape=(32, 64), dtype=DataType.FP16),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 64), dtype=DataType.FP16),
        }
        metas = _extract_local_tensor_metas(_runtime_slice_body, seed_meta=seed)
        # Runtime-scalar 2nd dim → falls back to src's dim 1 = 64; dtype from src.
        assert metas["view"] == TensorMeta(shape=(16, 64), dtype=DataType.FP16)

    def test_extract_local_tensor_metas_annotated_assignments(self):
        """Annotated assignments (``v: T = ...``) are tracked just like plain ones."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_annotated_slice_body, seed_meta=seed)
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert metas["mid"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_slice_view_flows_into_incore_dep(self):
        """A pl.slice view of an entry parameter can be passed into an @pl.jit.incore dep."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            pl.store(t, [0, 0], out)
            return out

        @jit
        def slice_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            out = copy_incore(view, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        slice_entry.compile_for_test(src, out)
        compiled = list(slice_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)
        func_names = [f.name for f in compiled.program.functions.values()]
        assert "copy_incore" in func_names
        assert "slice_entry" in func_names

    def test_dep_return_value_flows_into_next_dep(self):
        """The return value of one @pl.jit.incore dep can feed the next dep."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def chain_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            mid = relu_incore(view, buf)
            out = relu_incore(mid, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        chain_entry.compile_for_test(src, out)
        compiled = list(chain_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)

    def test_multi_value_dep_return_flows_into_next_dep(self):
        """A tuple-returning @pl.jit.incore dep's results inherit their Out params' metas."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def split_incore(
            x: pl.Tensor, lo: pl.Out[pl.Tensor], hi: pl.Out[pl.Tensor]
        ) -> tuple[pl.Tensor, pl.Tensor]:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            a = pl.relu(t)
            b = pl.abs(t)
            pl.store(a, [0, 0], lo)
            pl.store(b, [0, 0], hi)
            return lo, hi

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def split_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            lo_buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            hi_buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            a, b = split_incore(view, lo_buf, hi_buf)
            out = relu_incore(a, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        split_entry.compile_for_test(src, out)
        compiled = list(split_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)

    def test_runtime_sized_slice_uses_static_parent_dim(self):
        """A pl.slice with a runtime-scalar width is advertised to the consuming
        kernel using the parent tensor's static dim, matching how hand-written
        @pl.program code annotates kernels that consume narrowed views (see
        examples/models/04_paged_attention.py)."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def softmax_incore(sij: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = sij.shape
            t = pl.load(sij, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def attn_entry(big: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            valid_len = pl.tensor.read(cfg, [0])
            sij_valid = pl.slice(big, [16, valid_len], [0, 0])
            out = softmax_incore(sij_valid, out)
            return out

        big = torch.randn(16, 128)
        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 128)
        attn_entry.compile_for_test(big, cfg, out)
        compiled = list(attn_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)

    def test_dep_return_then_runtime_slice_then_dep(self):
        """The paged-attention shape: a dep return value is sliced to a runtime
        width, and that view feeds the next dep — both inferences must hold."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def fill_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            pl.store(t, [0, 0], out)
            return out

        @jit.incore
        def softmax_incore(sij: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = sij.shape
            t = pl.load(sij, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def attn_entry(big: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            sij_buf = pl.create_tensor([16, 128], dtype=pl.FP32)
            sij = fill_incore(big, sij_buf)  # dep return → sij_buf's meta
            valid_len = pl.tensor.read(cfg, [0])
            sij_valid = pl.slice(sij, [16, valid_len], [0, 0])  # runtime slice of a dep return
            out = softmax_incore(sij_valid, out)
            return out

        big = torch.randn(16, 128)
        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 128)
        attn_entry.compile_for_test(big, cfg, out)
        compiled = list(attn_entry._cache.values())[0]
        assert isinstance(compiled, CompiledProgram)

    def test_unresolvable_create_tensor_dim_still_raises_clear_error(self):
        """A pl.create_tensor with a non-static dim has no parent to fall back
        to, so the view is untracked and the existing clear ValueError fires
        for the downstream dep parameter."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            pl.store(t, [0, 0], out)
            return out

        @jit
        def bad_entry(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            n = pl.tensor.read(cfg, [0])
            buf = pl.create_tensor([16, n], dtype=pl.FP32)
            out = copy_incore(buf, out)
            return out

        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 32)
        with pytest.raises(ValueError, match="missing inferred tensor metadata"):
            bad_entry.compile_for_test(cfg, out)


# Module-level dynvar + constant for TestDynamicLocalTensorMetadata.
# Module-level so the generated @pl.program source sees them in the
# originating module's globals when it's parsed.
_M_1524 = pl.dynamic("M_1524")
_HIDDEN_1524 = 128


class TestDynamicLocalTensorMetadata:
    """Regression tests for issue #1524: `pl.create_tensor` whose shape is
    derived from `pl.tensor.dim` on a dynamic-bound parameter, or from a
    bind_dynamic'd DynVar, must inherit the matching :class:`DynDim` so the
    local can flow into a sub-function's annotation as `pl.Tensor[[M, ...], ...]`.
    """

    def test_dim_alias_propagates_dyndim_to_local(self):
        """`tokens = pl.tensor.dim(P, 0)` then `pl.create_tensor([tokens, K], ...)`
        stamps the parent's DynDim onto the local's shape."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {
            "hidden_states": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16),
            "out": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16),
        }

        def body(hidden_states, out):
            tokens = pl.tensor.dim(hidden_states, 0)
            current = pl.create_tensor([tokens, 128], dtype=pl.BF16)
            nxt = pl.create_tensor([tokens, 128], dtype=pl.BF16)
            return current, nxt

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        assert "current" in metas
        assert "nxt" in metas
        assert metas["current"].shape == (m_dim, 128)
        assert metas["nxt"].shape == (m_dim, 128)

    def test_dim_alias_static_dim_stays_int(self):
        """Aliasing a static parent dim resolves to the plain int (no DynDim)."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {"x": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16)}

        def body(x):
            hidden = pl.tensor.dim(x, 1)  # static dim 1 = 128
            buf = pl.create_tensor([4, hidden], dtype=pl.FP32)
            return buf

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        assert metas["buf"].shape == (4, 128)

    def test_dynvar_in_create_tensor_substituted(self):
        """``pl.create_tensor([M, HIDDEN], ...)`` — M is a DynVar bound to a
        param dim. The body transformer rewrites the runtime ``M`` reference
        to ``pl.tensor.dim(P, k)`` so the generated IR doesn't leak the
        annotation-only DynVar past SSA conversion."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def layer_dv(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            return out

        @jit
        def fwd_dv(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            # Direct DynVar in the allocation shape — the issue's "alternative
            # workaround" form that previously failed with "Variable 'M' used
            # outside its defining scope".
            current = pl.create_tensor([_M_1524, _HIDDEN_1524], dtype=pl.BF16)
            nxt = pl.create_tensor([_M_1524, _HIDDEN_1524], dtype=pl.BF16)
            current = layer_dv(current, nxt)
            return current

        hidden = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        fwd_dv.compile_for_test(hidden, out)

    def test_shape_attribute_emits_anchor_not_dynvar(self):
        """``M, N = a.shape`` for a dynamic-bound param emits
        ``pl.tensor.dim(a, 0)`` rather than the bare DynVar — protects against
        the leak reported by Copilot / CodeRabbit (per-helper shape emission
        bypasses ``visit_Name``)."""
        torch = pytest.importorskip("torch")

        @jit
        def shape_unpack(
            a: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            a.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            M, _N = a.shape
            buf = pl.create_tensor([M, _HIDDEN_1524], dtype=pl.BF16)
            return buf

        a = torch.empty(5, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(5, _HIDDEN_1524, dtype=torch.bfloat16)
        shape_unpack.compile_for_test(a, out)

    def test_dim_alias_rebind_is_safe(self):
        """An alias rebound to a non-``pl.tensor.dim`` value must not stamp
        the parent's DynDim onto downstream shape resolution — regression for
        the flow-insensitive alias bug raised by CodeRabbit."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=5)
        seed = {"x": TensorMeta(shape=(m_dim, 128), dtype=DataType.FP32)}

        def body(x):
            tokens = pl.tensor.dim(x, 0)
            tokens = tokens - 1  # rebound — alias must be dropped
            buf = pl.create_tensor([tokens, 128], dtype=pl.FP32)
            return buf

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        # ``buf`` must NOT be recorded — ``tokens`` is no longer a clean dim
        # alias, so pl.create_tensor's shape can't be statically resolved.
        assert "buf" not in metas

    def test_issue_1524_repro_compiles(self):
        """The exact failing pattern from issue #1524."""
        torch = pytest.importorskip("torch")

        # Both M_1524 and _HIDDEN_1524 are module-level — the generated
        # @pl.program source picks them up from the originating module's
        # globals when it's parsed.
        @jit.inline
        def layer_1524(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            return out

        @jit
        def fwd_1524(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            tokens = pl.tensor.dim(hidden_states, 0)
            current = pl.create_tensor([tokens, _HIDDEN_1524], dtype=pl.BF16)
            nxt = pl.create_tensor([tokens, _HIDDEN_1524], dtype=pl.BF16)
            current = layer_1524(current, nxt)
            return current

        hidden = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        # Should not raise — previously failed with
        # "missing inferred tensor metadata for parameter 'hidden_states'".
        fwd_1524.compile_for_test(hidden, out)


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

        deps_topo, callers_by_id, callees_by_id, _ = entry._get_dep_graph()
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

        deps_topo, callers_by_id, _, _ = entry_diamond._get_dep_graph()
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

    def test_inline_multi_return_tuple_unpack(self):
        """Multi-return @pl.jit.inline + tuple-unpack at call site (issue #1304).

        Pre-fix: ParserSyntaxError ('TupleGetItemExpr requires tuple to have
        TupleType, got TensorType') because the specializer emitted a single-
        tensor return annotation.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            y0, y1 = two_returns(a, o0, o1)
            return y0, y1

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        post_pass = entry.compile_for_test(a, o0, o1)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns" not in func_names
        assert "entry" in func_names

    def test_inline_multi_return_direct_return(self):
        """Multi-return @pl.jit.inline + ``return inline_call(...)`` (issue #1304).

        Pre-fix: RuntimeError reporting the inline function as undefined, because
        InlineCallsMutator only handled AssignStmt/EvalStmt — ReturnStmt-shaped
        call sites slipped past splicing.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            return two_returns(a, o0, o1)

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        post_pass = entry.compile_for_test(a, o0, o1)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns" not in func_names
        assert "entry" in func_names

    def test_inline_with_bare_tensor_no_pl_out(self):
        """@pl.jit.inline helpers with bare ``pl.Tensor`` params (no ``pl.Out``)
        compile cleanly without DeprecationWarning. The body is spliced
        identically — `pl.Out` is redundant on inline helpers because the
        splice happens before SSA.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns_no_out(
            a: pl.Tensor,
            o0: pl.Tensor,  # bare — no pl.Out
            o1: pl.Tensor,  # bare — no pl.Out
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            y0, y1 = two_returns_no_out(a, o0, o1)
            return y0, y1

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            post_pass = entry.compile_for_test(a, o0, o1)
        # Bare pl.Tensor inline params must not emit the deprecation warning.
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught), (
            f"Unexpected DeprecationWarning for bare pl.Tensor inline params: "
            f"{[str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]}"
        )
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns_no_out" not in func_names
        assert "entry" in func_names


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
            per_func_dyn={id(kernel._func): {}},
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


# ---------------------------------------------------------------------------
# ir.compile() kwarg forwarding (Issue #1405)
# ---------------------------------------------------------------------------


class TestCompileKwargForwarding:
    """``ir.compile()`` kwargs are forwarded through ``JITFunction._compile``.

    Before this fix, ``_compile`` only forwarded ``skip_ptoas`` and
    ``platform`` — every other compile knob a user set on ``RunConfig``
    (``strategy``, ``dump_passes``, ...) was silently dropped on the JIT path.
    """

    def test_run_config_compile_kwargs_maps_fields(self, tmp_path):
        """Compile-side RunConfig fields map onto the ir.compile() parameter names."""
        artifacts_dir = tmp_path / "jit_artifacts"
        cfg = RunConfig(
            strategy=OptimizationStrategy.DebugTileOptimization,
            dump_passes=True,
            compile_profiling=True,
            save_kernels_dir=str(artifacts_dir),
            block_dim=8,
        )
        kwargs = _run_config_compile_kwargs(cfg)
        assert kwargs["strategy"] == OptimizationStrategy.DebugTileOptimization
        assert kwargs["dump_passes"] is True
        assert kwargs["profiling"] is True  # mapped from RunConfig.compile_profiling
        assert kwargs["output_dir"] == str(artifacts_dir)  # from RunConfig.save_kernels_dir
        assert "diagnostic_phase" in kwargs
        assert "disabled_diagnostics" in kwargs
        # backend_type is derived from `platform` by ir.compile(); not forwarded.
        assert "backend_type" not in kwargs
        # block_dim is a runtime dispatch param — execute_compiled re-supplies
        # RunConfig.block_dim and overrides the baked value, so it is not a
        # compile input and must not be forwarded (would split the cache key).
        assert "block_dim" not in kwargs

    def test_run_config_compile_kwargs_omits_unset_output_dir(self):
        """save_kernels_dir left unset omits output_dir so ir.compile()'s default applies."""
        kwargs = _run_config_compile_kwargs(RunConfig())
        assert "output_dir" not in kwargs

    def test_run_config_compile_kwargs_forwards_distributed_config(self):
        """A RunConfig.distributed_config is forwarded so @pl.jit.host kernels go distributed."""
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dc = DistributedConfig(device_ids=[0, 1])
        kwargs = _run_config_compile_kwargs(RunConfig(distributed_config=dc))
        # Forwarded verbatim (same object) so ir.compile() emits a
        # DistributedCompiledProgram for the HOST-level entry.
        assert kwargs["distributed_config"] is dc

    def test_run_config_compile_kwargs_omits_unset_distributed_config(self):
        """distributed_config left unset is omitted so ir.compile()'s single-chip default applies."""
        kwargs = _run_config_compile_kwargs(RunConfig())
        assert "distributed_config" not in kwargs

    def test_make_cache_key_splits_on_distributed_config(self):
        """distributed_config participates in the cache key (distinct device_ids ≠ collide)."""
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        base = dict(
            source_hash="h",
            param_names=["x"],
            tensor_shapes={"x": (128, 128)},
            tensor_dtypes={"x": DataType.FP32},
            dynamic_dims=set(),
            scalar_values={},
            platform="a2a3",
            strategy=OptimizationStrategy.Default,
        )
        key_none = make_cache_key(**base)
        key_01 = make_cache_key(**base, distributed_config=DistributedConfig(device_ids=[0, 1]))
        key_23 = make_cache_key(**base, distributed_config=DistributedConfig(device_ids=[2, 3]))
        key_01_again = make_cache_key(**base, distributed_config=DistributedConfig(device_ids=[0, 1]))

        # Distinct device_ids must not collide, and a distributed config must
        # not collide with the single-chip (None) default.
        assert len({key_none, key_01, key_23}) == 3
        # Equal configs yield equal keys, so a genuine re-call still hits the
        # cache; the key stays hashable (usable in a set / as a dict key).
        assert key_01 == key_01_again

    def test_resolve_compiled_splits_cache_on_distributed_config(self, monkeypatch):
        """Two calls differing only in distributed_config compile distinct artifacts.

        Regression for the JIT cache key omitting ``distributed_config``: the
        config is baked into the ``DistributedCompiledProgram`` and drives
        per-rank dispatch, so reusing the first artifact for a second call with
        different ``device_ids`` would silently target the wrong ranks.
        """
        torch = pytest.importorskip("torch")
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        # Stub out the actual compile so the test stays device-free and only
        # exercises the cache-key / cache-miss logic in _resolve_compiled.
        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(device_ids):
            cfg = RunConfig(distributed_config=DistributedConfig(device_ids=device_ids))
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve([0, 1])
        second = resolve([2, 3])  # different device_ids → cache miss
        third = resolve([0, 1])  # same as first → cache hit

        assert compile_calls["n"] == 2  # only two compiles, not three
        assert len(cfg_kernel._cache) == 2  # two distinct cached artifacts
        assert first != second  # not the same cached object
        assert third == first  # re-uses the first artifact

    def test_compile_forwards_run_config_kwargs(self, monkeypatch):
        """_compile forwards ir_compile_kwargs verbatim to ir.compile()."""
        # `pypto.ir.compile` the attribute is the re-exported function, so
        # import the submodule explicitly to patch the name _compile reads.
        ir_compile_mod = importlib.import_module("pypto.ir.compile")

        @jit
        def fwd_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                pl.store(t, [0, 0], out)
            return out

        captured: dict = {}

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return "fake-compiled-program"

        # _compile re-imports `compile` from pypto.ir.compile on each call,
        # so patching the module attribute intercepts the real compilation.
        monkeypatch.setattr(ir_compile_mod, "compile", fake_compile)

        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dc = DistributedConfig(device_ids=[0, 1])
        cfg = RunConfig(
            strategy=OptimizationStrategy.DebugTileOptimization,
            dump_passes=True,
            compile_profiling=True,
            distributed_config=dc,
        )
        result = fwd_kernel._compile(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_values={},
            scalar_dtypes={},
            per_func_dyn={id(fwd_kernel._func): {}},
            pl=pl,
            platform="a2a3sim",
            **_run_config_compile_kwargs(cfg),
        )
        assert result == "fake-compiled-program"
        assert captured["strategy"] == OptimizationStrategy.DebugTileOptimization
        assert captured["dump_passes"] is True
        assert captured["profiling"] is True
        assert captured["platform"] == "a2a3sim"
        assert "skip_ptoas" in captured
        # distributed_config reaches ir.compile() so a @pl.jit.host entry can
        # compile to a DistributedCompiledProgram and dispatch per-rank.
        assert captured["distributed_config"] is dc

    def test_compile_without_kwargs_forwards_only_defaults(self, monkeypatch):
        """_compile with no extra kwargs forwards only skip_ptoas + platform."""
        ir_compile_mod = importlib.import_module("pypto.ir.compile")

        @jit
        def plain_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                pl.store(t, [0, 0], out)
            return out

        captured: dict = {}

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return "fake"

        monkeypatch.setattr(ir_compile_mod, "compile", fake_compile)

        plain_kernel._compile(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_values={},
            scalar_dtypes={},
            per_func_dyn={id(plain_kernel._func): {}},
            pl=pl,
        )
        assert set(captured) == {"skip_ptoas", "platform"}
        assert captured["platform"] is None


# ---------------------------------------------------------------------------
# Source provenance: diagnostics map back to the user's real .py (Issue #1612)
# ---------------------------------------------------------------------------


@jit
def _provenance_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [128, 128])
        t = pl.mul(t, t)
        pl.store(t, [0, 0], out)
    return out


def _walk_spans(stmts):
    """Collect spans of all statements (recursing into nested bodies)."""
    out = []
    for s in stmts:
        span = getattr(s, "span", None)
        if span is not None:
            out.append(span)
        body = getattr(s, "body", None)
        if body is not None:
            try:
                out.extend(_walk_spans(list(body)))
            except TypeError:
                pass
    return out


class TestJitSourceProvenance:
    """JIT parse/compile diagnostics point at the user's real source (#1612).

    @pl.jit re-derives the kernel into a generated @pl.program string and
    reparses it, so spans used to land on a synthesized ``<string>`` source.
    A source map now remaps them back to the user's .py at statement
    granularity.
    """

    def test_diagnostic_filename_names_kernel(self):
        """The fallback synthetic filename names the kernel, not ``<string>``."""
        assert _provenance_kernel._diagnostic_filename == "<jit:_provenance_kernel>"

    def test_body_spans_point_at_real_file(self):
        """Every (user-written) body statement's span resolves to this file."""
        prog = _provenance_kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_values={},
            scalar_dtypes={},
            per_func_dyn={id(_provenance_kernel._func): {}},
            pl=pl,
        )
        func = prog.get_function("_provenance_kernel")
        assert func is not None

        spans = _walk_spans(list(func.body))
        assert spans, "expected body statement spans"
        # The kernel body is entirely user-written (no synthesized statements),
        # so every body span must resolve to this real source file.
        assert all(s.filename == __file__ for s in spans), [s.filename for s in spans]

    def test_span_line_matches_the_real_source_line(self):
        """The remapped span line equals the statement's actual line in this file."""
        src_lines, start = inspect.getsourcelines(_provenance_kernel._func)
        with_offset = next(i for i, ln in enumerate(src_lines) if "with pl.at" in ln)
        expected_with_line = start + with_offset

        prog = _provenance_kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_values={},
            scalar_dtypes={},
            per_func_dyn={id(_provenance_kernel._func): {}},
            pl=pl,
        )
        incore = list(prog.get_function("_provenance_kernel").body)[0]
        assert incore.span.filename == __file__
        assert incore.span.begin_line == expected_with_line


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
