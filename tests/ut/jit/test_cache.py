# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for python/pypto/jit/cache.py."""

import ast
import importlib
import inspect
import types
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pypto.language as pl
import pytest
from pypto.ir import DistributedConfig, OptimizationStrategy
from pypto.jit._source import capture_namespaces, function_namespace
from pypto.jit.cache import (
    compute_source_hash,
    make_cache_key,
)
from pypto.jit.decorator import (
    _resolve_enable_pypto_l0c_double_buffer,
    _resolve_memory_planner,
    _resolve_runtime,
)
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError
from pypto.pypto_core import DataType, ir, passes
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig


class TestComputeSourceHash:
    def test_deterministic(self):
        h1 = compute_source_hash(["def f(): pass"])
        h2 = compute_source_hash(["def f(): pass"])
        assert h1 == h2

    def test_different_sources_differ(self):
        h1 = compute_source_hash(["def f(): pass"])
        h2 = compute_source_hash(["def g(): pass"])
        assert h1 != h2

    def test_multiple_sources_combined(self):
        h_combined = compute_source_hash(["def f(): pass", "def g(): pass"])
        h_single_f = compute_source_hash(["def f(): pass"])
        assert h_combined != h_single_f

    def test_order_matters(self):
        h1 = compute_source_hash(["aaa", "bbb"])
        h2 = compute_source_hash(["bbb", "aaa"])
        assert h1 != h2

    def test_returns_string(self):
        h = compute_source_hash(["source"])
        assert isinstance(h, str)
        assert len(h) > 0


class TestMakeCacheKey:
    def _make_key(  # noqa: PLR0913 — mirrors make_cache_key's per-dimension args
        self,
        source_hash="abc",
        param_names=None,
        tensor_shapes=None,
        tensor_dtypes=None,
        dynamic_dims=None,
        scalar_values=None,
        platform=None,
        strategy=None,
        distributed_config=None,
        analyze_auto_scopes_for_deps=False,
        memory_planner=None,
        enable_pypto_l0c_double_buffer=False,
        tensor_layouts=None,
        dep_layouts=(),
        runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    ):
        return make_cache_key(
            source_hash=source_hash,
            param_names=param_names or [],
            tensor_shapes=tensor_shapes or {},
            tensor_dtypes=tensor_dtypes or {},
            dynamic_dims=dynamic_dims or set(),
            scalar_values=scalar_values or {},
            platform=platform,
            strategy=strategy,
            distributed_config=distributed_config,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            memory_planner=memory_planner,
            enable_pypto_l0c_double_buffer=enable_pypto_l0c_double_buffer,
            tensor_layouts=tensor_layouts,
            dep_layouts=dep_layouts,
            runtime=runtime,
        )

    def test_basic_key_structure(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 128)},
            tensor_dtypes={"a": DataType.FP32},
        )
        assert isinstance(key, tuple)
        assert len(key) == 7
        source_hash, platform, strategy, tensor_part, scalar_part, dist_part, compile_opts = key
        assert source_hash == "abc"
        assert platform is None
        assert strategy is None
        assert isinstance(tensor_part, tuple)
        assert isinstance(scalar_part, tuple)
        assert dist_part is None  # single-chip default
        assert compile_opts == (
            ("analyze_auto_scopes_for_deps", False),
            ("dump_ptoas_passes", False),
            ("memory_planner", None),
            ("enable_pypto_l0c_double_buffer", False),
            ("dep_layouts", ()),
            ("runtime", "tensormap_and_ringbuffer"),
        )

    def test_tensor_shape_in_key(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 64)},
            tensor_dtypes={"a": DataType.FP32},
        )
        _, _, _, tensor_part, _, _, _ = key
        assert len(tensor_part) == 1
        info = tensor_part[0]
        assert info.name == "a"
        assert info.shape == (128, 64)
        assert info.dtype == DataType.FP32

    def test_dynamic_dim_becomes_none(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (256, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
        )
        _, _, _, tensor_part, _, _, _ = key
        assert tensor_part[0].shape == (None, 128)

    def test_dynamic_dim_cache_hit_on_different_concrete_value(self):
        """Two calls with different values for a dynamic dim should produce the same key."""
        key_256 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (256, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
            scalar_values={},
        )
        key_512 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (512, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
            scalar_values={},
        )
        assert key_256 == key_512

    def test_static_dim_change_causes_miss(self):
        """Changing a non-dynamic dim should produce a different key."""
        key_128 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (256, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
            scalar_values={},
        )
        key_256 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (256, 256)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
            scalar_values={},
        )
        assert key_128 != key_256

    def test_scalar_values_in_key(self):
        key = self._make_key(
            param_names=["BLOCK_M"],
            scalar_values={"BLOCK_M": 64},
        )
        _, _, _, _, scalar_part, _, _ = key
        assert len(scalar_part) == 1
        assert scalar_part[0].name == "BLOCK_M"
        assert scalar_part[0].value == 64

    def test_different_scalar_values_cause_miss(self):
        k1 = self._make_key(param_names=["B"], scalar_values={"B": 64})
        k2 = self._make_key(param_names=["B"], scalar_values={"B": 128})
        assert k1 != k2

    def test_param_order_preserved(self):
        """Tensor infos should follow param_names order."""
        key = make_cache_key(
            source_hash="h",
            param_names=["b", "a"],
            tensor_shapes={"a": (16,), "b": (32,)},
            tensor_dtypes={"a": DataType.FP16, "b": DataType.FP32},
            dynamic_dims=set(),
            scalar_values={},
        )
        _, _, _, tensor_part, _, _, _ = key
        assert tensor_part[0].name == "b"
        assert tensor_part[1].name == "a"

    def test_key_is_hashable(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.INT32},
        )
        d = {key: "value"}
        assert d[key] == "value"

    def test_source_hash_change_causes_miss(self):
        k1 = self._make_key(
            source_hash="hash1",
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
        )
        k2 = self._make_key(
            source_hash="hash2",
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
        )
        assert k1 != k2

    def test_different_tensor_layouts_cause_miss(self):
        """A layout can reach the annotation through a variable, leaving the
        source text — and so ``source_hash`` — identical. It must split the key
        on its own."""
        common = {"param_names": ["a"], "tensor_shapes": {"a": (8, 8)}, "tensor_dtypes": {"a": DataType.FP32}}
        k1 = self._make_key(**common, tensor_layouts={"a": ir.TensorLayout.MX_A_ZZ})
        k2 = self._make_key(**common, tensor_layouts={"a": ir.TensorLayout.MX_B_NN})
        assert k1 != k2

    def test_different_dep_layouts_cause_miss(self):
        """Same, one call deeper: a layout a *dep* declares appears in no entry
        parameter meta, so it needs its own key component."""
        common = {"param_names": ["a"], "tensor_shapes": {"a": (8, 8)}, "tensor_dtypes": {"a": DataType.FP32}}
        k1 = self._make_key(**common, dep_layouts=(("dep", "x", "TensorLayout.MX_A_ZZ"),))
        k2 = self._make_key(**common, dep_layouts=(("dep", "x", "TensorLayout.MX_B_NN"),))
        assert k1 != k2

    def test_different_platforms_cause_miss(self):
        """Same shapes/dtypes compiled for different platforms must not collide."""
        k1 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform="a2a3sim",
        )
        k2 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform="a3",
        )
        assert k1 != k2

    def test_same_platform_is_cache_hit(self):
        k1 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform="a2a3sim",
        )
        k2 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform="a2a3sim",
        )
        assert k1 == k2

    def test_distributed_config_in_key(self):
        """distributed_config is baked into the artifact, so it must split the key.

        Different ``device_ids`` (and the single-chip ``None`` default) produce
        distinct keys; equal configs collide so a genuine re-call still hits the
        cache.
        """

        def key_for(distributed_config):
            return self._make_key(
                param_names=["a"],
                tensor_shapes={"a": (8, 8)},
                tensor_dtypes={"a": DataType.FP32},
                distributed_config=distributed_config,
            )

        k_none = key_for(None)
        k_01 = key_for(DistributedConfig(device_ids=[0, 1]))
        k_23 = key_for(DistributedConfig(device_ids=[2, 3]))
        k_01_again = key_for(DistributedConfig(device_ids=[0, 1]))

        assert len({k_none, k_01, k_23}) == 3  # all distinct, and key stays hashable
        assert k_01 == k_01_again  # equal config → cache hit

    def test_none_platform_differs_from_named_platform(self):
        """platform=None and platform='a2a3sim' must not collide."""
        k_none = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform=None,
        )
        k_named = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            platform="a2a3sim",
        )
        assert k_none != k_named

    def test_same_strategy_is_cache_hit(self):
        k1 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=OptimizationStrategy.Default,
        )
        k2 = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=OptimizationStrategy.Default,
        )
        assert k1 == k2

    def test_none_strategy_differs_from_named_strategy(self):
        """strategy=None (JIT default) and an explicit strategy must not collide."""
        k_none = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=None,
        )
        k_named = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=OptimizationStrategy.Default,
        )
        assert k_none != k_named

    def test_key_with_strategy_is_hashable(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.INT32},
            strategy=OptimizationStrategy.Default,
        )
        d = {key: "value"}
        assert d[key] == "value"

    def test_analyze_auto_scopes_for_deps_splits_key(self):
        """AUTO-scope auto-deps changes generated code, so it must split cache."""
        k_off = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            analyze_auto_scopes_for_deps=False,
        )
        k_on = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            analyze_auto_scopes_for_deps=True,
        )
        assert k_off != k_on

    def test_memory_planner_splits_key(self):
        """The planner changes both placement and address ownership, so it must
        split the cache even when two modes both use ptoas level3."""
        keys = [
            self._make_key(
                param_names=["a"],
                tensor_shapes={"a": (8, 8)},
                tensor_dtypes={"a": DataType.FP32},
                memory_planner=planner,
            )
            for planner in (
                None,
                MemoryPlanner.PYPTO,
                MemoryPlanner.DSA_RP,
                MemoryPlanner.PTOAS,
            )
        ]
        assert len(set(keys)) == len(keys), f"planner must split the cache key, got {keys}"

    def test_dbc_double_buffer_flag_splits_legacy_pypto_key(self):
        """The legacy-PyPTO dbC=2 opt-in changes AutoTileMatmulL0/MemoryReuse output,
        so a kernel compiled with it off must not reuse that artifact when later
        called with it on."""
        key_off = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            memory_planner=MemoryPlanner.PYPTO,
            enable_pypto_l0c_double_buffer=False,
        )
        key_on = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            memory_planner=MemoryPlanner.PYPTO,
            enable_pypto_l0c_double_buffer=True,
        )
        assert key_off != key_on, "dbC=2 opt-in must split the cache key"

    @pytest.mark.parametrize("planner", [MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_dbc_double_buffer_flag_does_not_split_automatic_planner_key(self, planner):
        """DSA_RP and PTOAS enable dbC automatically, so the legacy-PyPTO flag is inert."""
        kwargs = {
            "param_names": ["a"],
            "tensor_shapes": {"a": (8, 8)},
            "tensor_dtypes": {"a": DataType.FP32},
            "memory_planner": planner,
        }
        key_off = self._make_key(**kwargs, enable_pypto_l0c_double_buffer=False)
        key_on = self._make_key(**kwargs, enable_pypto_l0c_double_buffer=True)
        assert key_off == key_on

    def test_runtime_splits_key(self):
        """The runtime is baked into the artifact's ``kernel_config.py`` and decides
        which worker can bind it, so a ``host_build_graph`` call must not reuse a
        ``tensormap_and_ringbuffer`` artifact."""
        kwargs = {
            "param_names": ["a"],
            "tensor_shapes": {"a": (8, 8)},
            "tensor_dtypes": {"a": DataType.FP32},
        }
        key_tmrb = self._make_key(**kwargs, runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        key_hbg = self._make_key(**kwargs, runtime=passes.RuntimeKind.HOST_BUILD_GRAPH)
        assert key_tmrb != key_hbg, "runtime must split the cache key"


class TestResolveRuntime:
    """The runtime the JIT keys on must match the one ``ir.compile()`` will use."""

    def test_defaults_to_tensormap_and_ringbuffer(self):
        assert _resolve_runtime() == passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER

    def test_reads_the_active_pass_context(self):
        # The runtime is PassContext-only — RunConfig does not carry it — so the
        # context is the sole source the cache key can consult.
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            assert _resolve_runtime() == passes.RuntimeKind.HOST_BUILD_GRAPH


class TestResolveMemoryPlanner:
    """The planner the JIT keys on must match the one ``ir.compile()`` will use."""

    def test_defaults_to_pypto(self):
        assert _resolve_memory_planner(None) == MemoryPlanner.PYPTO

    @pytest.mark.parametrize("planner", [MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_reads_the_active_pass_context(self, planner):
        """The planner is usually selected by wrapping the call in a PassContext,
        which never reaches RunConfig. Keying only on RunConfig would let such a
        call reuse a PYPTO-compiled artifact."""
        with passes.PassContext([], memory_planner=planner):
            assert _resolve_memory_planner(None) == planner
        assert _resolve_memory_planner(None) == MemoryPlanner.PYPTO


class TestResolveEnablePyptoL0cDoubleBuffer:
    """The legacy-PyPTO dbC=2 opt-in must match what ``ir.compile()`` inherits."""

    def test_defaults_to_off(self):
        assert _resolve_enable_pypto_l0c_double_buffer() is False

    def test_reads_the_active_pass_context(self):
        """The flag is set by wrapping the call in a PassContext, which never
        reaches RunConfig; keying only on RunConfig would let a flag-on call
        reuse the flag-off artifact."""
        with passes.PassContext([], enable_pypto_l0c_double_buffer=True):
            assert _resolve_enable_pypto_l0c_double_buffer() is True
        assert _resolve_enable_pypto_l0c_double_buffer() is False

    def test_run_config_field_wins_over_default(self):
        cfg = RunConfig(platform="a2a3", memory_planner=MemoryPlanner.PTOAS)
        assert _resolve_memory_planner(cfg) == MemoryPlanner.PTOAS

    def test_unset_run_config_field_defers_to_context(self):
        cfg = RunConfig(platform="a2a3")
        assert cfg.memory_planner is None
        with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
            assert _resolve_memory_planner(cfg) == MemoryPlanner.PTOAS


_CACHE_BLOCK = 32
_CACHE_UNUSED = 0
_CACHE_LAYOUT = pl.NZ


def _global_slice(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[_CACHE_BLOCK, 128], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        y = pl.slice(x, [_CACHE_BLOCK, 128], [0, 0])
    return y


def _with_globals(func, **values):
    cloned = types.FunctionType(func.__code__, {**func.__globals__, **values}, func.__name__)
    cloned.__annotations__ = func.__annotations__.copy()
    return cloned


class TestGlobalDependencies:
    @pytest.fixture
    def compile_programs(self, monkeypatch):
        """Exercise real specialization and parsing without invoking toolchains."""
        programs = []

        def compile_program(program, **kwargs):
            programs.append(program)
            return program

        monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "compile", compile_program)
        return programs

    def test_global_change_invalidates_memory_cache(self, compile_programs, monkeypatch):
        kernel = pl.jit(_global_slice)
        first = kernel.compile()
        assert kernel.compile() is first
        monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
        second = kernel.compile()
        assert second is not first
        assert len(compile_programs) == 2
        assert "[32, 128]" in first.as_python()
        assert "[64, 128]" in second.as_python()

    def test_hash_available_before_specialization(self):
        first = pl.jit(_with_globals(_global_slice, _CACHE_BLOCK=32))
        second = pl.jit(_with_globals(_global_slice, _CACHE_BLOCK=64))
        assert first._get_source_hash() != second._get_source_hash()

    def test_unused_global_does_not_invalidate(self, monkeypatch):
        kernel = pl.jit(_global_slice)
        initial = kernel._get_source_hash()
        monkeypatch.setitem(_global_slice.__globals__, "_CACHE_UNUSED", 123)
        assert kernel._get_source_hash() == initial

    @pytest.mark.parametrize("first,second", [(True, 1), (1, 1.0), (0.0, -0.0)])
    def test_constant_encoding_preserves_type_and_float_bits(self, first, second):
        def add_constant(x: pl.Tensor[[128], pl.FP32]):
            return pl.add(x, _CACHE_BLOCK)

        a = pl.jit(_with_globals(add_constant, _CACHE_BLOCK=first))
        b = pl.jit(_with_globals(add_constant, _CACHE_BLOCK=second))
        assert a._get_source_hash() != b._get_source_hash()

    def test_closure_constant_uses_current_snapshot(self, compile_programs):
        block = 32

        @pl.jit
        def kernel(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = pl.add(x, block)
            return y

        first = kernel.compile()
        block = 64
        second = kernel.compile()
        assert second is not first
        assert ", 32.0)" in first.as_python()
        assert ", 64.0)" in second.as_python()

    def test_source_hash_and_specialization_share_closure_snapshot(self, compile_programs, monkeypatch):
        block = 32

        @pl.jit
        def kernel(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = pl.add(x, block)
            return y

        source_hash = kernel._get_source_hash

        def mutate_after_source_hash():
            nonlocal block
            result = source_hash()
            block = 64
            return result

        monkeypatch.setattr(kernel, "_get_source_hash", mutate_after_source_hash)
        first = kernel.compile()
        assert ", 32.0)" in first.as_python()
        monkeypatch.setattr(kernel, "_get_source_hash", source_hash)
        block = 32
        assert kernel.compile() is first
        block = 64
        assert ", 64.0)" in kernel.compile().as_python()
        assert len(compile_programs) == 2

    def test_same_global_name_in_distinct_namespaces(self):
        def left(x):
            return pl.add(x, _CACHE_BLOCK)

        def right(x):
            return pl.mul(x, _CACHE_BLOCK)

        left_dep = pl.jit.inline(_with_globals(left, _CACHE_BLOCK=32))
        right_dep = pl.jit.inline(_with_globals(right, _CACHE_BLOCK=64))

        @pl.jit
        def entry(x: pl.Tensor[[128], pl.FP32]):
            return right_dep(left_dep(x))

        initial = entry._get_source_hash()
        left_dep._func.__globals__["_CACHE_BLOCK"] = 16
        left_changed = entry._get_source_hash()
        assert left_changed != initial
        right_dep._func.__globals__["_CACHE_BLOCK"] = 16
        assert entry._get_source_hash() != left_changed

    def test_local_shadow_does_not_invalidate(self, monkeypatch):
        @pl.jit
        def kernel(x: pl.Tensor[[128], pl.FP32]):
            _CACHE_BLOCK = 16
            with pl.at(level=pl.Level.CORE_GROUP):
                y = pl.slice(x, [_CACHE_BLOCK], [0])
            return y

        initial = kernel._get_source_hash()
        monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
        assert kernel._get_source_hash() == initial

    def test_return_annotation_global_is_keyed(self, monkeypatch):
        @pl.jit
        def kernel(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[_CACHE_BLOCK], pl.FP32]:
            return x

        initial = kernel._get_source_hash()
        monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
        assert kernel._get_source_hash() != initial

    def test_transitive_global_change(self, monkeypatch):
        leaf = pl.jit.inline(_global_slice)

        @pl.jit.inline
        def helper(x):
            return leaf(x)

        @pl.jit
        def entry(x: pl.Tensor[[128], pl.FP32]):
            return helper(x)

        initial = entry._get_source_hash()
        monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
        assert entry._get_source_hash() != initial

    def test_rebound_helper_changes_dependency_graph(self):
        @pl.jit.inline
        def helper(x):
            return x

        @pl.jit
        def entry(x: pl.Tensor[[128], pl.FP32]):
            return helper(x)

        initial = entry._get_source_hash()
        assert entry._get_deps() == [helper]

        @pl.jit.inline
        def helper(x):
            return pl.add(x, x)

        assert entry._get_source_hash() != initial
        assert entry._get_deps() == [helper]

    def test_compilation_uses_key_snapshot(self, compile_programs, monkeypatch):
        kernel = pl.jit(_global_slice)
        make_key = importlib.import_module("pypto.jit.decorator").make_cache_key

        def mutate_after_key(**kwargs):
            key = make_key(**kwargs)
            monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
            return key

        monkeypatch.setattr(
            importlib.import_module("pypto.jit.decorator"), "make_cache_key", mutate_after_key
        )
        first = kernel.compile()
        assert "[32, 128]" in first.as_python()
        second = kernel.compile()
        assert "[64, 128]" in second.as_python()
        assert len(compile_programs) == 2

    def test_concurrent_helper_rebinding_keeps_each_call_snapshot(self, compile_programs, monkeypatch):
        @pl.jit.inline
        def add_impl(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            return pl.add(x, x)

        @pl.jit.inline
        def mul_impl(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            return pl.mul(x, x)

        helper = add_impl

        @pl.jit
        def entry(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = helper(x)
            return y

        initial_hash = entry._get_source_hash()
        expected_first = entry.specialize().as_python()
        get_deps = entry._get_deps
        captured, resume = Event(), Event()

        def pause_after_dependency_capture():
            deps = get_deps()
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            if (
                current_thread().name.startswith("jit-request")
                and frame.f_back.f_code.co_name == "_get_static_source_hash"
            ):
                captured.set()
                assert resume.wait(10), "Timed out waiting for the second compilation"
            return deps

        monkeypatch.setattr(entry, "_get_deps", pause_after_dependency_capture)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jit-request") as executor:
            first_call = executor.submit(entry.compile)
            try:
                assert captured.wait(10), "First call did not capture its dependency graph"
                helper = mul_impl
                second = entry.compile()
                assert entry._get_source_hash() != initial_hash
            finally:
                resume.set()
            first = first_call.result(timeout=10)

        assert first is not second
        assert first.as_python() == expected_first
        assert second.as_python() == entry.specialize().as_python()
        assert len(compile_programs) == 2
        assert entry.compile() is second
        helper = add_impl
        assert entry.compile() is first

    @pytest.mark.parametrize(
        "before,after",
        [
            (pl.jit.inline, pl.jit.opaque),
            (pl.jit.inline(auto_scope=True), pl.jit.inline(auto_scope=False)),
        ],
        ids=["function-type", "auto-scope"],
    )
    def test_rebound_helper_attributes_invalidate_cache(self, compile_programs, before, after):
        def implementation(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            return pl.add(x, x)

        helper = before(implementation)

        @pl.jit
        def entry(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = helper(x)
            return y

        initial_hash = entry._get_source_hash()
        first = entry.compile()
        helper = after(implementation)
        assert entry._get_source_hash() != initial_hash
        second = entry.compile()
        assert second is not first
        assert second.as_python() == entry.specialize().as_python()
        assert len(compile_programs) == 2
        assert entry.compile() is second

    def test_rebound_helper_level_does_not_hide_invalid_ir(self, compile_programs):
        def implementation(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            return pl.add(x, x)

        helper = pl.jit.incore(level=pl.Level.CHIP_DIE)(implementation)

        @pl.jit
        def entry(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = helper(x)
            return y

        initial_hash = entry._get_source_hash()
        entry.compile()
        helper = pl.jit.incore(level=pl.Level.AIC)(implementation)
        assert entry._get_source_hash() != initial_hash
        with pytest.raises(ParserSyntaxError, match="explicit level=AIC"):
            entry.compile()
        assert len(compile_programs) == 1

    def test_snapshot_is_released_after_compile_failure(self, compile_programs, monkeypatch):
        kernel = pl.jit(_global_slice)
        compile_kernel = kernel._compile

        def fail_once(*args, **kwargs):
            monkeypatch.setitem(_global_slice.__globals__, "_CACHE_BLOCK", 64)
            raise RuntimeError("injected compilation failure")

        monkeypatch.setattr(kernel, "_compile", fail_once)
        with pytest.raises(RuntimeError, match="injected compilation failure"):
            kernel.compile()
        monkeypatch.setattr(kernel, "_compile", compile_kernel)
        assert "[64, 128]" in kernel.compile().as_python()

    def test_warm_key_reuses_source_and_layout_resolution(self, monkeypatch):
        helper = pl.jit.inline(_global_slice)

        @pl.jit
        def entry(x: pl.Tensor[[128, 128], pl.FP32]):
            return helper(x)

        with capture_namespaces():
            initial_hash = entry._get_source_hash()
            initial_layouts = entry._dep_declared_layouts()

        def unexpected_recomputation(*args, **kwargs):
            pytest.fail("Warm key rebuilt unchanged source structure or parameter layouts")

        monkeypatch.setattr(entry, "_compute_static_source_hash", unexpected_recomputation)
        monkeypatch.setattr(
            importlib.import_module("pypto.jit.decorator"), "_param_layouts", unexpected_recomputation
        )
        with capture_namespaces():
            assert entry._get_source_hash() == initial_hash
            assert entry._dep_declared_layouts() == initial_layouts

    def test_layout_cache_tracks_postponed_annotation_binding(self):
        def implementation(x: "pl.Tensor[[128, 128], pl.FP32, _CACHE_LAYOUT]"):
            return x

        func = _with_globals(implementation, _CACHE_LAYOUT=pl.NZ)
        helper = pl.jit.inline(func)

        @pl.jit
        def entry(x: pl.Tensor[[128, 128], pl.FP32]):
            return helper(x)

        first = entry._dep_declared_layouts()
        assert first == (("implementation", "x", str(pl.NZ)),)
        with capture_namespaces():
            assert entry._dep_declared_layouts() == first
            func.__globals__["_CACHE_LAYOUT"] = pl.DN
            assert entry._dep_declared_layouts() == first
        assert entry._dep_declared_layouts() == (("implementation", "x", str(pl.DN)),)

    def test_empty_closure_cell_shadows_module_global(self, monkeypatch):
        rows = 32

        def implementation(x):
            return pl.add(x, rows)

        assert implementation.__closure__ is not None
        del implementation.__closure__[0].cell_contents
        monkeypatch.setitem(implementation.__globals__, "rows", 5)
        kernel = pl.jit(implementation)
        initial_hash = kernel._get_source_hash()
        monkeypatch.setitem(implementation.__globals__, "rows", 9)
        assert kernel._get_source_hash() == initial_hash
        assert function_namespace(implementation).get("rows") != 9

    def test_namespace_view_keeps_closure_precedence_and_module_snapshot(self, monkeypatch):
        rows = 32

        def implementation(x):
            return pl.add(x, rows + _CACHE_BLOCK)

        monkeypatch.setitem(implementation.__globals__, "rows", 5)
        with capture_namespaces():
            namespace = function_namespace(implementation)
            rows = 64
            monkeypatch.setitem(implementation.__globals__, "_CACHE_BLOCK", 96)
            assert namespace["rows"] == 32
            assert namespace["_CACHE_BLOCK"] == 32
            assert dict(namespace)["rows"] == 32
        assert function_namespace(implementation)["rows"] == 64
        assert function_namespace(implementation)["_CACHE_BLOCK"] == 96

    def test_invalid_string_annotation_does_not_break_source_hash(self, monkeypatch):
        def implementation(x):
            return pl.add(x, _CACHE_BLOCK)

        # Construct invalid annotation syntax as data so type checkers can
        # still validate the test module itself.
        definition = ast.parse(
            "def implementation(x: 'not a valid expression'):\n    return pl.add(x, _CACHE_BLOCK)\n"
        ).body[0]
        monkeypatch.setattr(
            importlib.import_module("pypto.jit.decorator"), "_get_func_def", lambda _: definition
        )
        kernel = pl.jit(implementation)
        assert kernel._get_source_hash() == kernel._get_source_hash()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
