# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit decorator: decoration, cache hit/miss, and bind_dynamic."""

import ast
import importlib
import inspect
import re
import warnings

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import (
    _SYNTHESIZED_DYN_PREFIX,
    JITFunction,
    _arg_ref,
    _build_param_mapping,
    _compute_per_func_dyndim_maps,
    _discover_deps,
    _extract_call_args_for_dep,
    _extract_local_tensor_metas,
    _extract_tensor_meta,
    _resolve_dep_call_metadata,
    _rewrite_jit_error,
    _run_config_compile_kwargs,
    _scan_dep_io,
    _scan_dynamic_dims,
    _SlicedArg,
    _synthesized_dyn_dim,
    jit,
)
from pypto.jit.specializer import DynDim, Specializer, TensorMeta
from pypto.language.parser.diagnostics.exceptions import ParserTypeError
from pypto.pypto_core import DataType, InternalError, ir
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

    def test_torch_fp4_x2_shape_becomes_logical_ir_shape(self):
        torch = pytest.importorskip("torch")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            pytest.skip("torch.float4_e2m1fn_x2 required")
        packed = torch.empty((128, 32), dtype=fp4_dtype)
        meta = _extract_tensor_meta(packed)
        assert meta.dtype == DataType.FP4
        assert meta.static_shape() == (128, 64)

    def test_torch_fp4_x2_rejects_empty_packed_dimension(self):
        torch = pytest.importorskip("torch")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            pytest.skip("torch.float4_e2m1fn_x2 required")
        packed = torch.empty((128, 0), dtype=fp4_dtype)
        with pytest.raises(TypeError, match="positive runtime x2 carrier last dimension"):
            _extract_tensor_meta(packed)

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

    def test_jit_default_auto_scope_true(self):
        @jit
        def entry(a: pl.Tensor):
            return a

        assert entry._auto_scope is True

    def test_jit_auto_scope_false(self):
        @jit(auto_scope=False)
        def entry(a: pl.Tensor):
            return a

        assert isinstance(entry, JITFunction)
        assert entry._func_type == "orchestration"
        assert entry._auto_scope is False

    def test_jit_empty_parens_form(self):
        """@pl.jit() (bare parens) is equivalent to @pl.jit."""

        @jit()
        def entry(a: pl.Tensor):
            return a

        assert isinstance(entry, JITFunction)
        assert entry._auto_scope is True


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

    def test_jit_host_accepts_auto_scope_false(self):
        @jit.host(auto_scope=False)
        def host_orch(a: pl.Tensor):
            return a

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"
        assert host_orch._auto_scope is False

    def test_jit_incore_rejects_auto_scope_kwarg(self):
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.incore(auto_scope=False)
            def sub_fn(a: pl.Tensor):
                return a

    def test_jit_inline_accepts_auto_scope_false(self):
        """Inline bodies are spliced into the caller, so hand-placed scopes land
        there — @pl.jit.inline must accept auto_scope=False (#1733)."""

        @jit.inline(auto_scope=False)
        def sub_fn(a: pl.Tensor):
            return a

        assert isinstance(sub_fn, JITFunction)
        assert sub_fn._func_type == "inline"
        assert sub_fn._auto_scope is False

    def test_jit_opaque_rejects_auto_scope_kwarg(self):
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.opaque(auto_scope=False)
            def sub_fn(a: pl.Tensor):
                return a

    def test_jit_incore_rejects_auto_scope_true(self):
        """Sub-decorators reject auto_scope= even when explicitly True — the
        kwarg is not part of their API surface, so passing any value is an
        error, not just a non-True value."""
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.incore(auto_scope=True)
            def sub_fn(a: pl.Tensor):
                return a


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

    def test_host_forwards_dep_auto_scope(self):
        """An @pl.jit(auto_scope=False) chip orchestrator discovered as a dep of
        an @pl.jit.host entry keeps its auto_scope=False when its
        SpecializeContext is built — the flag must be forwarded to the dep
        context, not defaulted to True."""
        torch = pytest.importorskip("torch")

        @jit(auto_scope=False)
        def chip_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return chip_orch(a, c)

        a = torch.empty(128, 128)
        c = torch.empty(128, 128)
        _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = host_orch._bind_args((a, c), {})
        contexts = host_orch._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "chip_orch")
        assert dep_ctx.auto_scope is False

    def test_entry_forwards_inline_dep_auto_scope(self):
        """An @pl.jit.inline(auto_scope=False) dep of a plain @pl.jit entry keeps
        auto_scope=False in its SpecializeContext, while the entry's own flag
        stays at its default True (#1733)."""
        torch = pytest.importorskip("torch")

        @jit.inline(auto_scope=False)
        def inline_fn(a: pl.Tensor, c: pl.Tensor):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return inline_fn(a, c)

        a = torch.empty(128, 128)
        c = torch.empty(128, 128)
        _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "inline_fn")
        assert dep_ctx.auto_scope is False
        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.auto_scope is True


# ---------------------------------------------------------------------------
# Tensor.bind_dynamic no-op
# ---------------------------------------------------------------------------


class TestScanDepIo:
    """_scan_dep_io records both Out and InOut params as output-like (so meta
    propagates to captured dep results), in declaration order."""

    def test_includes_inout_params_in_declaration_order(self):
        @jit.inline
        def sub(a: pl.Tensor, cache: pl.InOut[pl.Tensor], out: pl.Out[pl.Tensor]):
            return out

        def caller(a, cache, out):
            out = sub(a, cache, out)
            return out

        io = _scan_dep_io(caller)
        assert "sub" in io
        param_names, output_params = io["sub"]
        assert param_names == ["a", "cache", "out"]
        # Both InOut (cache) and Out (out) are output-like; declaration order,
        # so the positional target<->param mapping stays aligned.
        assert output_params == ["cache", "out"]

    def test_out_only_dep_unchanged(self):
        @jit.inline
        def sub(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            return out

        def caller(a, out):
            out = sub(a, out)
            return out

        _, output_params = _scan_dep_io(caller)["sub"]
        assert output_params == ["out"]


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

    @pytest.fixture(autouse=True)
    def _disable_ptoas_for_source_only_tests(self, monkeypatch, tmp_path):
        """Keep cache compilation source-only on hosts with an unusable ptoas."""
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

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

        add_kernel.compile(a, b, c)
        assert len(add_kernel._cache) == 1
        add_kernel.compile(a, b, c)
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

        add_kernel2.compile(a128, b128, c128)
        assert len(add_kernel2._cache) == 1
        add_kernel2.compile(a64, b64, c64)
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

        dyn_kernel.compile(a256, c256)
        assert len(dyn_kernel._cache) == 1
        dyn_kernel.compile(a512, c512)
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

        dyn_kernel2.compile(a128, c128)
        assert len(dyn_kernel2._cache) == 1
        dyn_kernel2.compile(a256, c256)
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

        ann_dyn_kernel.compile(a256, c256)
        assert len(ann_dyn_kernel._cache) == 1
        ann_dyn_kernel.compile(a512, c512)
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
        simple.compile(a, c)
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
        program = copy_entry.lower(x, out)
        assert isinstance(program, ir.Program)

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
        program = add_entry.lower(a, b, c)
        func_names = [f.name for f in program.functions.values()]
        assert "add_incore" in func_names
        assert "add_entry" in func_names

    def test_multi_func_cache_hit(self, monkeypatch, tmp_path):
        """Two multi-function JIT calls with same shapes reuse the cached program."""
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

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
        relu_entry.compile(x, out)
        assert len(relu_entry._cache) == 1
        relu_entry.compile(x, out)
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
        got = add_entry.lower(a, b, c)
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


def _reshape_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function for _extract_local_tensor_metas unit test: reshape tracking."""
    x_flat = pl.reshape(src, [128, 128])  # noqa: F841 — tracked by JIT AST metadata extraction
    out_flat = pl.reshape(out, [128, 128])
    return out_flat


@jit.incore
def _callsite_metadata_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Dependency used by point-in-time call-site metadata tests."""
    return out


# --- Runtime-sized local extents (synthesized DynDim) ------------------------
# Fixtures for the tests covering a local tensor whose extent is only known at
# runtime: one decorated dep, then plain (undecorated) caller bodies fed straight
# to ``_extract_local_tensor_metas`` / ``_resolve_dep_call_metadata``.


@jit.incore
def _synth_dim_kernel(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Dependency for the synthesized-dim overlay tests."""
    return out


# Module global (not a closure var) for the folding-precedence test.
_CLOSURE_FOLD_ROWS = 64


def _runtime_create_body(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """create_tensor whose leading extent is read out of a tensor at runtime."""
    n = pl.tensor.read(cfg, [0])
    tmp = pl.create_tensor([n, 8], dtype=pl.FP32)  # noqa: F841 — tracked by metadata extraction
    return out


def _dyn_arith_create_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """create_tensor sized by arithmetic over a DynDim — not statically foldable."""
    rows = pl.tensor.dim(src, 0)
    tmp = pl.create_tensor([rows * 2, 8], dtype=pl.FP32)  # noqa: F841 — tracked by extraction
    return out


def _runtime_window_body(out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """pld.window whose leading extent is the runtime world size."""
    buf = pld.alloc_window_buffer([2, 8], dtype=pl.INT32)
    win = pld.window(buf, [pld.world_size(), 8], dtype=pl.INT32)  # noqa: F841 — tracked
    return out


def _runtime_reshape_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """pl.reshape with a runtime extent — stays strict, no meta."""
    n = pl.tensor.read(cfg, [0])
    flat = pl.reshape(src, [n, 8])  # noqa: F841 — deliberately untracked
    return out


def _runtime_create_then_dep(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """A runtime-sized local crossing a dep call boundary (issue #2450's shape)."""
    n = pl.tensor.read(cfg, [0])
    tmp = pl.create_tensor([n, 8], dtype=pl.FP32)
    out = _synth_dim_kernel(tmp, out)
    return out


def _dyn_alias_create_then_dep(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """A local carrying a caller-derived (not synthesized) DynDim into a dep."""
    rows = pl.tensor.dim(src, 0)
    tmp = pl.create_tensor([rows, 8], dtype=pl.FP32)
    out = _synth_dim_kernel(tmp, out)
    return out


def _plain_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    src = pl.reshape(src, [4, 8])
    out = _callsite_metadata_kernel(src, out)
    src = pl.reshape(src, [2, 16])  # noqa: F841 — must not affect the earlier call
    return out


def _annotated_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    view: pl.Tensor = pl.reshape(src, [4, 8])
    out: pl.Tensor = _callsite_metadata_kernel(view, out)
    view = pl.reshape(view, [2, 16])  # noqa: F841 — must not affect the earlier call
    return out


def _chained_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    src = view = src[4:]
    out = _callsite_metadata_kernel(view, out)
    return out


def _direct_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    out = _callsite_metadata_kernel(src, out)
    return out


def _nested_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    with pl.scope():
        src = pl.reshape(src, [4, 8])
        out = _callsite_metadata_kernel(src, out)
        src = pl.reshape(src, [2, 16])  # noqa: F841 — after the nested call
    return out


def _subscript_slice_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function for _extract_local_tensor_metas unit test: subscript-slice
    sugar tracking (issue #1836). All locals are tracked by JIT AST metadata
    extraction, hence the noqa F841 on the unused names."""
    view = src[0:16, 0:8]  # static bounds → (16, 8)  # noqa: F841
    row = src[4, 0:8]  # scalar index drops dim 0 → (8,)  # noqa: F841
    partial = src[0:16]  # trailing implicit ``:`` keeps parent dim 1  # noqa: F841
    open_lo = src[4:]  # open upper bound → parent_dim - start = (28, 32)  # noqa: F841
    out_view = out[0:16, 0:8]  # noqa: F841
    return out


def _subscript_runtime_slice_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function: a subscript slice whose upper bound is a runtime scalar."""
    valid_len = pl.tensor.read(cfg, [0])
    view = src[0:16, 0:valid_len]  # 2nd dim runtime → parent-dim fallback
    out = _relu_kernel(view, out)
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
        program = slice_entry.lower(src, out)
        func_names = [f.name for f in program.functions.values()]
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
        program = chain_entry.lower(src, out)
        assert isinstance(program, ir.Program)

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
        program = split_entry.lower(src, out)
        assert isinstance(program, ir.Program)

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
        program = attn_entry.lower(big, cfg, out)
        assert isinstance(program, ir.Program)

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
        program = attn_entry.lower(big, cfg, out)
        assert isinstance(program, ir.Program)

    def test_runtime_create_tensor_dim_defers_to_the_shared_pipeline(self):
        """A runtime-sized ``pl.create_tensor`` dim no longer fails in the JIT
        frontend — it becomes a synthesized DynDim and the shared pass pipeline
        diagnoses whatever the program actually does with it.

        Here the kernel loads the WHOLE tensor as a tile, so the dynamic dim
        reaches the tile shape, which InitMemRef rejects. That is the same
        error a hand-written ``@pl.program`` gets for the same code — the point
        of this test is that ``@pl.jit`` no longer front-runs it with a
        frontend-only "missing inferred tensor metadata".
        """
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
        with pytest.raises(InternalError, match="InitMemRef requires static shape") as excinfo:
            bad_entry.lower(cfg, out)
        assert "missing inferred tensor metadata" not in str(excinfo.value)

    def test_extract_local_tensor_metas_reshape(self):
        """``_extract_local_tensor_metas`` infers metas for pl.reshape results."""
        seed = {
            "src": TensorMeta(shape=(2, 64, 128), dtype=DataType.BF16),
            "out": TensorMeta(shape=(2, 64, 128), dtype=DataType.BF16),
        }
        metas = _extract_local_tensor_metas(_reshape_body, seed_meta=seed)
        # reshape changes rank but keeps element count; dtype inherited from src.
        assert metas["x_flat"] == TensorMeta(shape=(128, 128), dtype=DataType.BF16)
        assert metas["out_flat"] == TensorMeta(shape=(128, 128), dtype=DataType.BF16)

    @staticmethod
    def _resolve_callsite_metadata(caller):
        seed = {
            "src": TensorMeta(shape=(32,), dtype=DataType.FP32),
            "out": TensorMeta(shape=(4, 8), dtype=DataType.FP32),
        }
        tensor_meta, scalar_values, scalar_dtypes = _resolve_dep_call_metadata(
            _callsite_metadata_kernel,
            caller,
            seed,
            {},
            {},
            {},
        )
        assert scalar_values == {}
        assert scalar_dtypes == {}
        return tensor_meta

    def test_dep_metadata_uses_plain_rebinding_at_callsite(self):
        """A plain assignment shadows the parameter only until the selected call."""
        metas = self._resolve_callsite_metadata(_plain_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)
        assert metas["out"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_dep_metadata_uses_annotated_rebinding_at_callsite(self):
        """Annotated and plain assignments expose the same point-in-time metadata."""
        metas = self._resolve_callsite_metadata(_annotated_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_dep_metadata_tracks_all_chained_assignment_targets(self):
        """Every target gets RHS metadata from the pre-assignment state."""
        metas = self._resolve_callsite_metadata(_chained_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(28,), dtype=DataType.FP32)

    def test_dep_metadata_direct_call_keeps_seed_metadata(self):
        """Direct dependency calls without local rebindings remain unchanged."""
        metas = self._resolve_callsite_metadata(_direct_callsite)
        assert metas["x"] == TensorMeta(shape=(32,), dtype=DataType.FP32)

    def test_dep_metadata_nested_scope_preserves_source_order(self):
        """Nested DSL scopes stop at their consumer after processing prior producers."""
        metas = self._resolve_callsite_metadata(_nested_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_extract_local_tensor_metas_subscript_slice(self):
        """``_extract_local_tensor_metas`` tracks subscript-slice sugar
        ``v = src[a:b, ...]`` the same way it tracks ``pl.slice`` (issue #1836):
        static extents, scalar-index rank reduction, and trailing implicit ``:``."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_subscript_slice_body, seed_meta=seed)
        # Two static slices → (stop - start) per dim; dtype inherited from src.
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # Scalar index at dim 0 drops it (numpy-style rank reduction).
        assert metas["row"] == TensorMeta(shape=(8,), dtype=DataType.FP32)
        # Only dim 0 indexed → dim 1 is implicit ``:``, keeps the parent extent.
        assert metas["partial"] == TensorMeta(shape=(16, 32), dtype=DataType.FP32)
        # Open upper bound with a static lower bound → parent_dim - start on the
        # sliced dim (matching the parser), parent extent on the trailing dim.
        assert metas["open_lo"] == TensorMeta(shape=(28, 32), dtype=DataType.FP32)
        # Subscript of an Out parameter resolves just like an In parameter.
        assert metas["out_view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_extract_local_tensor_metas_subscript_runtime_bound_uses_parent_dim(self):
        """A subscript slice dim with a non-static upper bound falls back to the
        parent tensor's static dim (the slice is bounded above by its parent)."""
        seed = {
            "src": TensorMeta(shape=(32, 64), dtype=DataType.FP16),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 64), dtype=DataType.FP16),
        }
        metas = _extract_local_tensor_metas(_subscript_runtime_slice_body, seed_meta=seed)
        # Runtime-scalar 2nd bound → falls back to src's dim 1 = 64; dtype from src.
        assert metas["view"] == TensorMeta(shape=(16, 64), dtype=DataType.FP16)

    def test_extract_local_tensor_metas_subscript_propagates_dyndim(self):
        """A full ``:`` subscript over a ``DynDim`` parent dim flows the DynDim
        through transparently (matching ``pl.slice`` behaviour)."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {"src": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16)}

        def body(src):
            view = src[:, 0:64]  # dim 0 full ``:`` keeps DynDim; dim 1 → 64  # noqa: F841
            return view

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        assert metas["view"].shape == (m_dim, 64)
        assert metas["view"].dtype == DataType.BF16


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
        fwd_dv.lower(hidden, out)

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
        shape_unpack.lower(a, out)

    def test_dim_alias_rebind_is_safe(self):
        """An alias rebound to a non-``pl.tensor.dim`` value must not stamp
        the parent's DynDim onto downstream shape resolution — regression for
        the flow-insensitive alias bug raised by CodeRabbit."""
        m_dim = DynDim(name="M", literal="M", static_bound=5)
        seed = {"x": TensorMeta(shape=(m_dim, 128), dtype=DataType.FP32)}

        def body(x):
            tokens = pl.tensor.dim(x, 0)
            tokens = tokens - 1  # rebound — alias must be dropped
            buf = pl.create_tensor([tokens, 128], dtype=pl.FP32)
            return buf

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        # ``tokens`` is no longer a clean dim alias, so the dim cannot be
        # statically resolved: it gets a synthesized placeholder. What must NOT
        # happen is inheriting the parent's ``M`` — that is the flow-insensitive
        # bug this guards.
        dim = metas["buf"].shape[0]
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert dim.name != m_dim.name

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
        fwd_1524.lower(hidden, out)

    def test_reshape_propagates_dyndim_via_dim_alias(self):
        """pl.reshape with a dim-aliased DynDim in shape propagates the DynDim.

        ``tokens = pl.tensor.dim(x, 2)`` extracts the 3rd dim's DynDim; using
        it in ``pl.reshape(x, [128, tokens])`` propagates the DynDim to the
        result's 2nd dim.
        """
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        t_dim = DynDim(name="T", literal="T", static_bound=128)
        seed = {"x": TensorMeta(shape=(2, 64, t_dim), dtype=DataType.BF16)}

        def body(x):
            tokens = pl.tensor.dim(x, 2)  # aliases x's dim 2 → DynDim
            flat = pl.reshape(x, [128, tokens])
            return flat

        metas = _extract_local_tensor_metas(body, seed_meta=seed)
        assert metas["flat"].shape == (128, t_dim)
        assert metas["flat"].dtype == DataType.BF16


# ---------------------------------------------------------------------------
# Per-rank sliced dispatch (chip_orch(x[r], ...)) and pld.window metadata.
# Module-level dynvar + functions so inspect.getsource / inspect.signature
# see real source and annotations (the host-orchestration distributed shape).
# ---------------------------------------------------------------------------
_M_SLICE = pl.dynamic("M_SLICE")


@jit.incore
def _sliced_chip(data: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Chip orchestrator dep reached via a per-rank ``_sliced_chip(x[r], ...)``
    dispatch from a host orchestrator body."""
    M, N = data.shape
    t = pl.load(data, [0, 0], [M, N])
    pl.store(t, [0, 0], out)
    return out


@jit.incore
def _dyn_sliced_chip(data: pl.Tensor[[_M_SLICE, 128], pl.FP32], out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Chip orchestrator dep whose leading dim is dynamic — used to verify the
    DynDim cascade does not flow through a per-rank sliced (dropped) dim."""
    M, N = data.shape
    t = pl.load(data, [0, 0], [M, N])
    pl.store(t, [0, 0], out)
    return out


def _per_rank_dispatch_body(inputs: pl.Tensor, outputs: pl.Out[pl.Tensor]) -> None:
    """Host orchestrator body: dispatch one chip orchestrator per rank by
    subscripting the leading (rank) dim — ``_sliced_chip(inputs[r], outputs[r])``."""
    for r in pl.range(2):
        _sliced_chip(inputs[r], outputs[r])


def _window_local_body(data_buf, signal_buf):
    """Host orchestrator body: per-rank window views over window buffers.
    The body is parsed from source only (never executed) — matching the
    existing ``_slice_then_dep_body`` style."""
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
    return data, signal


class TestArgRef:
    """Unit tests for ``_arg_ref`` — caller-side reference classification."""

    @staticmethod
    def _ref(expr_src: str):
        return _arg_ref(ast.parse(expr_src, mode="eval").body)

    def test_plain_name(self):
        assert self._ref("x") == "x"

    def test_single_integer_index_drops_one_dim(self):
        assert self._ref("x[r]") == _SlicedArg("x", 1)

    def test_multi_integer_index_drops_each_dim(self):
        assert self._ref("x[r, 0]") == _SlicedArg("x", 2)

    def test_slice_index_keeps_dim(self):
        # ``x[r:r+1]`` selects a range — the dim survives, so drop == 0 → None.
        assert self._ref("x[r:r+1]") is None

    def test_mixed_integer_and_slice_counts_only_integers(self):
        # One integer index (dropped) + one slice (kept) → drop == 1.
        assert self._ref("x[r, 0:2]") == _SlicedArg("x", 1)

    def test_literal_returns_none(self):
        assert self._ref("3") is None

    def test_attribute_returns_none(self):
        assert self._ref("obj.attr") is None

    def test_subscript_of_non_name_returns_none(self):
        # ``f()[0]`` — base is a call, not a Name.
        assert self._ref("f()[0]") is None


class TestExtractCallArgsSlicedDispatch:
    """``_extract_call_args_for_dep`` + ``_build_param_mapping`` must carry a
    per-rank subscript (``x[r]``) through as a ``_SlicedArg``."""

    def test_sliced_positional_args_extracted(self):
        call_args = _extract_call_args_for_dep(_per_rank_dispatch_body, "_sliced_chip")
        assert call_args == [
            (None, _SlicedArg("inputs", 1)),
            (None, _SlicedArg("outputs", 1)),
        ]

    def test_param_mapping_pairs_sliced_args_by_position(self):
        call_args = _extract_call_args_for_dep(_per_rank_dispatch_body, "_sliced_chip")
        assert call_args is not None
        mapping = _build_param_mapping(["data", "out"], call_args)
        assert mapping == {
            "data": _SlicedArg("inputs", 1),
            "out": _SlicedArg("outputs", 1),
        }


class TestWindowLocalMetadata:
    """``_extract_local_tensor_metas`` must infer metas for ``pld.window`` views
    so a host orchestrator's per-rank window locals propagate into the chip
    orchestrator's ``pld.DistributedTensor`` parameters."""

    def test_window_view_meta_inferred(self):
        metas = _extract_local_tensor_metas(_window_local_body, seed_meta={})
        # Shape from the 2nd positional arg, dtype from the ``dtype=`` keyword.
        assert metas["data"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)
        assert metas["signal"] == TensorMeta(shape=(1, 1), dtype=DataType.INT32)

    def test_window_missing_dtype_untracked(self):
        def body(buf):
            data = pld.window(buf, [1, 256])  # no dtype= kw
            return data

        metas = _extract_local_tensor_metas(body, seed_meta={})
        assert "data" not in metas

    def test_window_missing_shape_untracked(self):
        def body(buf):
            data = pld.window(buf, dtype=pl.FP32)  # no shape arg
            return data

        metas = _extract_local_tensor_metas(body, seed_meta={})
        assert "data" not in metas


class TestSlicedDispatchMetadata:
    """``_resolve_dep_call_metadata`` must give a per-rank chip orchestrator dep
    the base tensor's meta with the subscripted leading dims removed, and the
    DynDim cascade must not flow through a dropped dim."""

    def test_sliced_arg_drops_leading_dim(self):
        # Host passes ``inputs[r]`` / ``outputs[r]`` — each drops the rank dim.
        seed = {
            "inputs": TensorMeta(shape=(2, 1, 256), dtype=DataType.FP32),
            "outputs": TensorMeta(shape=(2, 1, 256), dtype=DataType.FP32),
        }
        tensor_meta, _, _ = _resolve_dep_call_metadata(
            _sliced_chip,
            _per_rank_dispatch_body,
            seed,
            {},
            {},
            {},
            caller_func_type="host",
        )
        assert tensor_meta["data"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)
        assert tensor_meta["out"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)

    def test_sliced_arg_drop_at_or_past_rank_is_untracked(self):
        # Base is 1-D; dropping a leading dim leaves nothing meaningful, so the
        # guard ``drop < len(shape)`` skips it rather than producing an empty meta.
        seed = {
            "inputs": TensorMeta(shape=(2,), dtype=DataType.FP32),
            "outputs": TensorMeta(shape=(2,), dtype=DataType.FP32),
        }
        tensor_meta, _, _ = _resolve_dep_call_metadata(
            _sliced_chip,
            _per_rank_dispatch_body,
            seed,
            {},
            {},
            {},
            caller_func_type="host",
        )
        assert "data" not in tensor_meta
        assert "out" not in tensor_meta

    def test_dyndim_cascade_skips_sliced_arg(self):
        # The dep declares a dynamic leading dim; the host reaches it via
        # ``_dyn_sliced_chip(inputs[r], ...)``. The DynDim must NOT cascade onto
        # a ``_SlicedArg`` key — doing so would corrupt the caller's dim map with
        # a non-string key (and is semantically wrong: the rank dim is dropped).
        call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | _SlicedArg | None]] | None] = {
            (id(_per_rank_dispatch_body), "_dyn_sliced_chip"): [
                (None, _SlicedArg("inputs", 1)),
                (None, _SlicedArg("outputs", 1)),
            ]
        }
        maps = _compute_per_func_dyndim_maps(
            entry_func=_per_rank_dispatch_body,
            entry_param_names=["inputs", "outputs"],
            deps=[_dyn_sliced_chip],
            callers_by_dep_id={id(_dyn_sliced_chip._func): [_per_rank_dispatch_body]},
            call_args_cache=call_args_cache,
        )
        host_map = maps[id(_per_rank_dispatch_body)]
        # Sanity: the dep itself carries the dynamic dim.
        assert maps[id(_dyn_sliced_chip._func)]["data"][0].literal == "M_SLICE"
        # The host map must only ever be keyed by parameter name strings.
        assert all(isinstance(key, str) for key in host_map)
        assert _SlicedArg("inputs", 1) not in host_map
        assert _SlicedArg("outputs", 1) not in host_map


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
        # lower() returns the post-pass IR (after PassManager.Default).
        # CompiledProgram.program is the *pre-pass* IR, so the cache entry would
        # still contain "add_inline"; we must inspect the post-pass return value.
        post_pass = add_entry.lower(a, b, c)
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
        post_pass = add_entry.lower(a, b, c)
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
        post_pass = entry_nested.lower(a, out)
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
        post_pass = entry_diamond.lower(a, out)
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
        post_pass = entry_mixed.lower(a, out)
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
        post_pass = entry.lower(a, o0, o1)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns" not in func_names
        assert "entry" in func_names

    def test_inline_tensor_and_task_id_tuple_return(self):
        """Inline multi-return preserves an explicitly typed TASK_ID element."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def produce(
            a: pl.Tensor,
            out: pl.Tensor,
        ) -> tuple[pl.Tensor[[32, 32], pl.FP32], pl.Scalar[pl.TASK_ID]]:
            with pl.at(level=pl.Level.CORE_GROUP) as producer_tid:
                tile = pl.load(a, [0, 0], [32, 32])
                out = pl.store(tile, [0, 0], out)
            return out, producer_tid

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            result = produce(a, out)
            produced = result[0]
            producer_tid = result[1]
            with pl.at(level=pl.Level.CORE_GROUP, deps=[producer_tid]):
                tile = pl.load(produced, [0, 0], [32, 32])
                out = pl.store(tile, [0, 0], out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry.lower(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "produce" not in func_names
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
        post_pass = entry.lower(a, o0, o1)
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
            post_pass = entry.lower(a, o0, o1)
        # Bare pl.Tensor inline params must not emit the deprecation warning.
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught), (
            f"Unexpected DeprecationWarning for bare pl.Tensor inline params: "
            f"{[str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]}"
        )
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns_no_out" not in func_names
        assert "entry" in func_names

    def test_inline_with_reshape_compiles(self):
        """Regression: @pl.jit.inline + pl.reshape in caller compiles.

        Previously failed because ``_extract_local_tensor_metas`` did not track
        ``pl.reshape``, so the specializer couldn't find metadata for the
        reshaped tensor passed to the inline dep.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def copy_inline(
            x: pl.Tensor[[128, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [128, 128])
                pl.store(tile, [0, 0], y)
            return y

        @jit
        def reshape_caller(
            x: pl.Tensor[[2, 64, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[2, 64, 128], pl.BF16]],
        ) -> pl.Tensor[[2, 64, 128], pl.BF16]:
            x_flat = pl.reshape(x, [128, 128])
            y = pl.reshape(y, [128, 128])
            y = copy_inline(x_flat, y)
            y = pl.reshape(y, [2, 64, 128])
            return y

        x = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        y = torch.empty(2, 64, 128, dtype=torch.bfloat16)
        post_pass = reshape_caller.lower(x, y)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "copy_inline" not in func_names, f"Inline should be spliced, got {func_names}"
        assert "reshape_caller" in func_names

    def test_inline_with_subscript_slice_compiles(self):
        """Regression for #1836: a subscript-slice view ``src[a:b]`` forwarded into
        an @pl.jit.inline dep compiles.

        Previously failed because ``_extract_local_tensor_metas`` did not track the
        subscript-slice sugar (an ``ast.Subscript``, unlike ``pl.slice``'s Call),
        so the inline dep's param got no inferred metadata and ``_build_params``
        raised ``missing inferred tensor metadata``. Mirrors
        ``test_inline_with_reshape_compiles`` (the #1755 sibling) with the local
        view produced by subscript sugar instead of ``pl.reshape``.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def copy_inline(
            x: pl.Tensor[[128, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [128, 128])
                pl.store(tile, [0, 0], y)
            return y

        @jit
        def subscript_caller(
            src: pl.Tensor[[256, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ) -> pl.Tensor[[128, 128], pl.BF16]:
            x_view = src[0:128]  # subscript-slice sugar → (128, 128)
            y = copy_inline(x_view, y)
            return y

        src = torch.randn(256, 128, dtype=torch.bfloat16)
        y = torch.empty(128, 128, dtype=torch.bfloat16)
        post_pass = subscript_caller.lower(src, y)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "copy_inline" not in func_names, f"Inline should be spliced, got {func_names}"
        assert "subscript_caller" in func_names

    def test_jit_inline_split_aiv_underscore_loop_var(self):
        """`for _ in pl.split_aiv(...)` is the documented lane-agnostic idiom and
        must compile inside an inline callee.

        The inliner alpha-renames callee locals by appending `_inline<N>`; with a
        loop variable named `_` that used to concatenate into `__inline<N>`, and
        the `__` there is the delimiter reserved by the IR auto-naming utility,
        so ConvertToSSA rejected a name the author never wrote.
        """
        torch = pytest.importorskip("torch")

        @jit.inline(auto_scope=False)
        def region(x: pl.Tensor, y: pl.Tensor):
            M, N = x.shape
            with pl.spmd(1, name_hint="usc") as _tid:
                _blk = pl.tile.get_block_idx()
                for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    y = pl.assemble(y, pl.mul(x[0:M, 0:N], 2.0), [0, 0])
            return y

        @jit
        def entry(x: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = region(x, y)
            return y

        entry.compile(torch.zeros(16, 64), torch.zeros(16, 64))


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
        got = add_entry.lower(a, b, c)
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
        got = tile_add.lower(a, b, c)
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
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            dump_ptoas_passes=True,
            compile_profiling=True,
            save_kernels_dir=str(artifacts_dir),
            analyze_auto_scopes_for_deps=True,
        )
        kwargs = _run_config_compile_kwargs(cfg)
        assert kwargs["strategy"] == OptimizationStrategy.Default
        assert kwargs["dump_passes"] is True
        assert kwargs["dump_ptoas_passes"] is True
        assert kwargs["profiling"] is True  # mapped from RunConfig.compile_profiling
        assert kwargs["output_dir"] == str(artifacts_dir)  # from RunConfig.save_kernels_dir
        assert kwargs["analyze_auto_scopes_for_deps"] is True
        assert "diagnostic_phase" in kwargs
        assert "disabled_diagnostics" in kwargs
        # backend_type is derived from `platform` by ir.compile(); not forwarded.
        assert "backend_type" not in kwargs

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
        assert kwargs["analyze_auto_scopes_for_deps"] is False

    def test_make_cache_key_splits_on_distributed_config(self):
        """distributed_config participates in the cache key (distinct device_ids ≠ collide)."""
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        def key_for(distributed_config):
            return make_cache_key(
                source_hash="h",
                param_names=["x"],
                tensor_shapes={"x": (128, 128)},
                tensor_dtypes={"x": DataType.FP32},
                dynamic_dims=set(),
                scalar_values={},
                platform="a2a3",
                strategy=OptimizationStrategy.Default,
                distributed_config=distributed_config,
            )

        key_none = key_for(None)
        key_01 = key_for(DistributedConfig(device_ids=[0, 1]))
        key_23 = key_for(DistributedConfig(device_ids=[2, 3]))
        key_01_again = key_for(DistributedConfig(device_ids=[0, 1]))

        # Distinct device_ids must not collide, and a distributed config must
        # not collide with the single-chip (None) default.
        assert len({key_none, key_01, key_23}) == 3
        # Equal configs yield equal keys, so a genuine re-call still hits the
        # cache; the key stays hashable (usable in a set / as a dict key).
        assert key_01 == key_01_again

    def test_make_cache_key_splits_on_auto_scope_deps_switch(self):
        """AUTO-scope dependency analysis changes codegen, so it splits cache."""
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        def key_for(enabled):
            return make_cache_key(
                source_hash="h",
                param_names=["x"],
                tensor_shapes={"x": (128, 128)},
                tensor_dtypes={"x": DataType.FP32},
                dynamic_dims=set(),
                scalar_values={},
                platform="a2a3",
                strategy=OptimizationStrategy.Default,
                analyze_auto_scopes_for_deps=enabled,
            )

        assert key_for(False) != key_for(True)

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

        def resolve(device_ids, analyze_auto_scopes_for_deps=False):
            cfg = RunConfig(
                distributed_config=DistributedConfig(device_ids=device_ids),
                analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            )
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve([0, 1])
        second = resolve([2, 3])  # different device_ids → cache miss
        third = resolve([0, 1])  # same as first → cache hit

        assert compile_calls["n"] == 2  # only two compiles, not three
        assert len(cfg_kernel._cache) == 2  # two distinct cached artifacts
        assert first != second  # not the same cached object
        assert third == first  # re-uses the first artifact

    def test_resolve_compiled_splits_cache_on_auto_scope_deps_switch(self, monkeypatch):
        """The same JIT call compiles separately when the AUTO-scope deps switch changes."""
        torch = pytest.importorskip("torch")

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(enabled):
            cfg = RunConfig(analyze_auto_scopes_for_deps=enabled)
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve(False)
        second = resolve(True)
        third = resolve(False)

        assert compile_calls["n"] == 2
        assert len(cfg_kernel._cache) == 2
        assert first != second
        assert third == first

    def test_resolve_compiled_splits_cache_on_ptoas_pass_dump(self, monkeypatch):
        """Enabling ptoas pass dumps cannot reuse an artifact compiled without them."""
        torch = pytest.importorskip("torch")

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(dump_ptoas_passes):
            cfg = RunConfig(dump_ptoas_passes=dump_ptoas_passes)
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve(False)
        second = resolve(True)
        third = resolve(False)

        assert compile_calls["n"] == 2
        assert len(cfg_kernel._cache) == 2
        assert first != second
        assert third == first

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
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            compile_profiling=True,
            distributed_config=dc,
            analyze_auto_scopes_for_deps=True,
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
        assert captured["strategy"] == OptimizationStrategy.Default
        assert captured["dump_passes"] is True
        assert captured["profiling"] is True
        assert captured["platform"] == "a2a3sim"
        assert captured["analyze_auto_scopes_for_deps"] is True
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


class TestClosureConstantFolding:
    """A ``@pl.jit`` function defined inside a factory (issue #2449).

    The generated ``@pl.program`` source is ``exec``'d in a fresh module holding
    only ``pl`` and ``pld``, so every free name in a body must be folded to a
    literal first. Folding used to read ``__globals__`` alone, where a closure
    free var never appears — so a factory constant survived verbatim and the
    parser raised ``Undefined variable``. Annotations always worked (they are
    rendered from ``tensor_meta``, not from the name), which made the gap
    invisible from outside.
    """

    @staticmethod
    def _specialize(entry, *args) -> str:
        _pn, _, tmeta, sv, sd, pfd = entry._bind_args(args, {})
        contexts = entry._build_contexts(tmeta, sv, sd, pfd)
        return Specializer(f"_jit_{entry.__name__}", contexts).specialize()

    @staticmethod
    def _build_factory_entry(rows: int):
        """Return a @pl.jit entry whose body references the factory's ``rows``."""

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            # ``rows`` is a closure free var, not a module global.
            tmp = pl.create_tensor([rows, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        return entry

    def test_closure_constant_folds_into_generated_body(self):
        """A factory constant referenced in a body is inlined as a literal."""
        torch = pytest.importorskip("torch")

        entry = self._build_factory_entry(64)
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        source = self._specialize(entry, a, out)

        assert "pl.create_tensor([64, 64]" in source
        # The free name must not survive — it is undefined in the generated module.
        assert not re.search(r"\brows\b", source)

    def test_closure_constant_specializes_per_value(self):
        """Two factory instantiations fold their own constant, not a shared one."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        assert "pl.create_tensor([32, 64]" in self._specialize(self._build_factory_entry(32), a, out)
        assert "pl.create_tensor([96, 64]" in self._specialize(self._build_factory_entry(96), a, out)

    @staticmethod
    def _build_mutable_factory_entry():
        """Return (entry, setter) sharing one closure cell, so the setter
        rebinds the *same* ``JITFunction``'s captured constant."""

        rows = 64

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tmp = pl.create_tensor([rows, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        def set_rows(value: int) -> None:
            nonlocal rows
            rows = value

        return entry, set_rows

    def test_rebound_closure_cell_changes_the_generated_source(self):
        """Rebinding the cell on one JITFunction changes what gets emitted.

        This is the premise of the cache-key test below: if the artifact did not
        depend on the cell, keying on it would be pointless.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        before = self._specialize(entry, a, out)
        set_rows(96)
        after = self._specialize(entry, a, out)

        assert "pl.create_tensor([64, 64]" in before
        assert "pl.create_tensor([96, 64]" in after

    def test_rebound_closure_cell_splits_the_cache(self):
        """A rebound cell must not silently reuse the previous artifact.

        ``_get_source_hash`` hashes the function *text*, which a ``nonlocal``
        rebind leaves byte-identical, so the closure values have to enter the
        key separately. Two distinct factory instances would not catch this —
        they are different ``JITFunction`` objects with their own caches.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        del a, out
        before = entry._folded_closure_constants()
        source_hash_before = entry._get_source_hash()
        set_rows(96)
        after = entry._folded_closure_constants()

        # The text-based hash cannot see the rebind — that is the whole problem.
        assert entry._get_source_hash() == source_hash_before
        # The closure component does, so the composed key differs.
        assert before != after
        assert ("entry", "rows", "64") in before
        assert ("entry", "rows", "96") in after

    def test_rebound_closure_cell_recompiles_instead_of_reusing(self):
        """The observable consequence: ``compile()`` must not hand back the
        artifact built from the previous cell value.

        Kept separate from the key-component test so this assertion is reached
        on its own — a regression in the key wiring has to fail *here*, not be
        masked by an earlier assertion.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        compiled_before = entry.compile(a, out, config=RunConfig(platform="a2a3sim"))
        set_rows(128)
        compiled_after = entry.compile(a, out, config=RunConfig(platform="a2a3sim"))
        assert compiled_before is not compiled_after, "stale artifact reused after rebind"
        assert len(entry._cache) == 2
        # An unchanged cell must still hit the cache, or the key would be useless.
        assert entry.compile(a, out, config=RunConfig(platform="a2a3sim")) is compiled_after
        assert len(entry._cache) == 2

    def test_closure_constants_key_component_is_stable_and_typed(self):
        """Equal state yields an equal component (so a genuine re-call still
        hits the cache), and ``repr`` keeps look-alike values apart."""
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        def key_for(closure_constants):
            return make_cache_key(
                source_hash="h",
                param_names=["x"],
                tensor_shapes={"x": (64, 64)},
                tensor_dtypes={"x": DataType.FP32},
                dynamic_dims=set(),
                scalar_values={},
                closure_constants=closure_constants,
            )

        base = key_for((("entry", "rows", "1"),))
        assert base == key_for((("entry", "rows", "1"),))
        # 1 / 1.0 / True all fold, and all produce different literals.
        assert len({base, key_for((("entry", "rows", "1.0"),)), key_for((("entry", "rows", "True"),))}) == 3

    def test_module_globals_still_fold(self):
        """Closure bindings are merged on top of globals, not instead of them."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tmp = pl.create_tensor([_CLOSURE_FOLD_ROWS, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        source = self._specialize(entry, a, out)
        assert "pl.create_tensor([64, 64]" in source
        assert not re.search(r"\b_CLOSURE_FOLD_ROWS\b", source)


class TestRuntimeSizedLocalExtents:
    """A local tensor whose extent is only known at runtime (issue #2450).

    ``pld.window(buf, [pld.world_size(), 1], ...)`` and
    ``pl.create_tensor([pl.tensor.read(cfg, [0]), 128], ...)`` have no static
    extent. Before this behaviour the whole meta was dropped, and the moment the
    local was passed to a dep ``_build_params`` raised "missing inferred tensor
    metadata". Now the unresolved dim becomes a synthesized ``DynDim``, which the
    generated program declares via ``pl.dynamic`` and the kernel binds from the
    actual argument's descriptor — the same IR a hand-written ``@pl.program``
    produces for this pattern.
    """

    @staticmethod
    def _leading_dim(metas: dict[str, TensorMeta], name: str):
        assert name in metas, f"{name!r} has no meta: {sorted(metas)}"
        return metas[name].shape[0]

    def test_runtime_create_extent_synthesizes_dyn_dim(self):
        """A create_tensor dim read out of a tensor becomes a synthesized DynDim."""
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_runtime_create_body, seed_meta=seed)
        dim = self._leading_dim(metas, "tmp")
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert dim.name.startswith(_SYNTHESIZED_DYN_PREFIX)
        # Only the unresolved dim is synthesized; the literal one stays an int.
        assert metas["tmp"].shape[1] == 8
        assert metas["tmp"].dtype == DataType.FP32

    def test_runtime_window_extent_synthesizes_dyn_dim(self):
        """pld.window sized by pld.world_size() gets a synthesized DynDim."""
        seed = {"out": TensorMeta(shape=(16, 8), dtype=DataType.FP32)}
        metas = _extract_local_tensor_metas(_runtime_window_body, seed_meta=seed)
        dim = self._leading_dim(metas, "win")
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert metas["win"].shape[1] == 8
        assert metas["win"].dtype == DataType.INT32

    def test_arithmetic_over_dyn_dim_synthesizes(self):
        """``M * 2`` is not statically foldable, so that dim is synthesized too."""
        seed = {
            "src": TensorMeta(shape=(DynDim(name="M", literal="M", static_bound=32), 8), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_dyn_arith_create_body, seed_meta=seed)
        dim = self._leading_dim(metas, "tmp")
        assert isinstance(dim, DynDim)
        assert dim.synthesized

    def test_statically_resolvable_dims_are_not_synthesized(self):
        """Existing static resolution is untouched — no placeholder is invented."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_slice_then_dep_body, seed_meta=seed)
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert all(not isinstance(d, DynDim) for d in metas["buf"].shape)

    def test_reshape_stays_strict(self):
        """A reshape dim is constrained by the source's element count, which an
        invented symbol cannot express — it still declines the meta."""
        seed = {
            "src": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = _extract_local_tensor_metas(_runtime_reshape_body, seed_meta=seed)
        assert "flat" not in metas

    def test_synthesized_dim_reaches_dep_parameter(self):
        """The dep parameter is typed instead of raising "missing inferred
        tensor metadata" — issue #2450's failure."""
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas, _, _ = _resolve_dep_call_metadata(
            _synth_dim_kernel, _runtime_create_then_dep, seed, {}, {}, {}
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.synthesized

    def test_dep_declared_symbol_replaces_synthesized_placeholder(self):
        """A dep that declares the dim ``pl.dynamic`` gets its own symbol back.

        The dep's body may reference that name (``for i in pl.range(NR)``);
        keeping the invented one would leave the reference unbound.
        """
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        dep_dyn_map = {"t": {0: DynDim(name="NR", literal="NR", static_bound=0)}}
        metas, _, _ = _resolve_dep_call_metadata(
            _synth_dim_kernel, _runtime_create_then_dep, seed, {}, {}, dep_dyn_map
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.name == "NR"
        assert not dim.synthesized

    def test_runtime_sized_local_lowers_end_to_end(self):
        """The headline claim: a runtime-sized local crossing a dep boundary
        reaches a real ``ir.Program``, with the synthesized symbol declared via
        ``pl.dynamic`` and left dynamic in the kernel's parameter type."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def scale_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(pl.add(tile, tile), [0, 0], out)
            return out

        @jit
        def runtime_sized_entry(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            n = pl.tensor.read(cfg, [0])
            tmp = pl.create_tensor([n, 64], dtype=pl.FP32)
            return scale_incore(tmp, out)

        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.zeros(64, 64, dtype=torch.float32)
        program = runtime_sized_entry.lower(cfg, out)
        assert isinstance(program, ir.Program)

        # The kernel parameter kept a dynamic leading dim rather than being
        # frozen to the placeholder extent.
        kernel = program.get_function("scale_incore")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        leading = param_type.shape[0]
        assert not isinstance(leading, ir.ConstInt), f"leading dim froze to {leading}"

    def test_synthesized_symbol_is_qualified_by_owning_function(self):
        """Two functions each holding a runtime-sized local of the same name get
        distinct symbols, so a dumped program never shows one name on two
        unrelated tensors."""
        a = _synthesized_dyn_dim("host_entry", "tmp", 0)
        b = _synthesized_dyn_dim("mid", "tmp", 0)
        assert a.name != b.name
        assert a.name.startswith(_SYNTHESIZED_DYN_PREFIX)
        assert b.name.startswith(_SYNTHESIZED_DYN_PREFIX)

    def test_caller_derived_dyn_dim_outranks_dep_declaration(self):
        """A DynDim the caller actually derived is NOT replaced — its symbol is
        the one bound at the call site (pre-existing precedence)."""
        seed = {
            "src": TensorMeta(shape=(DynDim(name="M", literal="M", static_bound=32), 8), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        dep_dyn_map = {"t": {0: DynDim(name="NR", literal="NR", static_bound=0)}}
        metas, _, _ = _resolve_dep_call_metadata(
            _synth_dim_kernel, _dyn_alias_create_then_dep, seed, {}, {}, dep_dyn_map
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.name == "M"
        assert not dim.synthesized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
