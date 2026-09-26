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
import re
import types
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from typing import Any

import pypto.language as pl
import pytest
from pypto.ir import DistributedConfig, OptimizationStrategy
from pypto.jit._source import capture_namespaces, function_namespace
from pypto.jit.cache import (
    SCALAR_SEMANTICS,
    compute_source_hash,
    make_cache_key,
)
from pypto.jit.decorator import (
    _resolve_enable_buffer_ir,
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
        platform=None,
        strategy=None,
        distributed_config=None,
        analyze_auto_scopes_for_deps=False,
        memory_planner=None,
        enable_pypto_l0c_double_buffer=False,
        tensor_layouts=None,
        dep_layouts=(),
        runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        enable_buffer_ir=False,
    ):
        return make_cache_key(
            source_hash=source_hash,
            param_names=param_names or [],
            tensor_shapes=tensor_shapes or {},
            tensor_dtypes=tensor_dtypes or {},
            dynamic_dims=dynamic_dims or set(),
            platform=platform,
            strategy=strategy,
            distributed_config=distributed_config,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            memory_planner=memory_planner,
            enable_pypto_l0c_double_buffer=enable_pypto_l0c_double_buffer,
            tensor_layouts=tensor_layouts,
            dep_layouts=dep_layouts,
            runtime=runtime,
            enable_buffer_ir=enable_buffer_ir,
        )

    def test_basic_key_structure(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 128)},
            tensor_dtypes={"a": DataType.FP32},
        )
        assert isinstance(key, tuple)
        assert len(key) == 6
        source_hash, platform, strategy, tensor_part, dist_part, compile_opts = key
        assert source_hash == "abc"
        assert platform is None
        assert strategy is None
        assert isinstance(tensor_part, tuple)
        assert dist_part is None  # single-chip default
        assert compile_opts == (
            ("scalar_semantics", 2),
            ("analyze_auto_scopes_for_deps", False),
            ("emit_source_loc", True),
            ("memory_planner", None),
            ("enable_pypto_l0c_double_buffer", False),
            ("dep_layouts", ()),
            ("runtime", "tensormap_and_ringbuffer"),
            ("enable_buffer_ir", False),
        )

    def test_tensor_shape_in_key(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 64)},
            tensor_dtypes={"a": DataType.FP32},
        )
        _, _, _, tensor_part, _, _ = key
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
        _, _, _, tensor_part, _, _ = key
        assert tensor_part[0].shape == (None, 128)

    def test_dynamic_dim_cache_hit_on_different_concrete_value(self):
        """Two calls with different values for a dynamic dim should produce the same key."""
        key_256 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (256, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
        )
        key_512 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (512, 128)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
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
        )
        key_256 = make_cache_key(
            source_hash="x",
            param_names=["a"],
            tensor_shapes={"a": (256, 256)},
            tensor_dtypes={"a": DataType.FP32},
            dynamic_dims={("a", 0)},
        )
        assert key_128 != key_256

    def test_scalar_semantics_version_is_in_key(self):
        """The key states which scalar contract built it (issue #2751).

        Without the stamp, an artifact compiled when a numeric argument was
        folded into the body could be served to a request that expects the
        parameter to stay symbolic.
        """
        _, _, _, _, _, compile_opts = self._make_key(param_names=["B"])
        assert ("scalar_semantics", SCALAR_SEMANTICS) in compile_opts

    def test_param_order_preserved(self):
        """Tensor infos should follow param_names order."""
        key = make_cache_key(
            source_hash="h",
            param_names=["b", "a"],
            tensor_shapes={"a": (16,), "b": (32,)},
            tensor_dtypes={"a": DataType.FP16, "b": DataType.FP32},
            dynamic_dims=set(),
        )
        _, _, _, tensor_part, _, _ = key
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

    @pytest.mark.parametrize("planner", [MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_buffer_ir_splits_key_for_every_planner(self, planner):
        assert self._make_key(memory_planner=planner) != self._make_key(
            memory_planner=planner, enable_buffer_ir=True
        )

    def test_dbc_double_buffer_flag_does_not_split_unresolved_default_key(self):
        """A None planner means DSA_RP, so the legacy-only flag is inert."""
        kwargs = {
            "param_names": ["a"],
            "tensor_shapes": {"a": (8, 8)},
            "tensor_dtypes": {"a": DataType.FP32},
            "memory_planner": None,
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

    def test_defaults_to_dsa_rp(self):
        assert _resolve_memory_planner(None) == MemoryPlanner.DSA_RP

    @pytest.mark.parametrize("planner", [MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_reads_the_active_pass_context(self, planner):
        """The planner is usually selected by wrapping the call in a PassContext,
        which never reaches RunConfig. Keying only on RunConfig would let such a
        call reuse a PYPTO-compiled artifact."""
        with passes.PassContext([], memory_planner=planner):
            assert _resolve_memory_planner(None) == planner
        assert _resolve_memory_planner(None) == MemoryPlanner.DSA_RP


def test_buffer_ir_cache_option_follows_active_context():
    assert _resolve_enable_buffer_ir() is False
    with passes.PassContext([], enable_buffer_ir=True):
        assert _resolve_enable_buffer_ir() is True
        with passes.PassContext([]):
            assert _resolve_enable_buffer_ir() is False
        assert _resolve_enable_buffer_ir() is True
    assert _resolve_enable_buffer_ir() is False


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


_CACHE_MODE = "trunc"
_CACHE_DTYPE = pl.INT8
_CACHE_MEM = pl.Mem.Vec
_CACHE_SHAPE = [1, 64]
# Deliberately typed ``Any``: this stands for any value the specializer cannot
# render as source, and the point is what the *cache key* does with it.
_CACHE_OPAQUE: Any = object()


def _global_cast(x: pl.Tensor[[1, 64], pl.FP16], out: pl.Out[pl.Tensor[[1, 64], pl.INT8]]):
    """Names a str, a DataType, an enum and a list constant — every foldable kind."""
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], _CACHE_SHAPE, target_memory=_CACHE_MEM)
        q = pl.cast(t, _CACHE_DTYPE, mode=_CACHE_MODE)
        y = pl.store(q, [0, 0], out)
    return y


def _global_opaque(x: pl.Tensor[[1, 64], pl.FP16]):
    """Names a value with no source form, so specializing it would fail."""
    with pl.at(level=pl.Level.CORE_GROUP):
        y = pl.load(x, [0, 0], [1, 64], target_memory=_CACHE_OPAQUE)
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

    @pytest.mark.parametrize(
        "name, first, second",
        [
            ("_CACHE_MODE", "trunc", "round"),
            ("_CACHE_DTYPE", pl.INT8, pl.INT16),
            ("_CACHE_MEM", pl.Mem.Vec, pl.Mem.Mat),
            ("_CACHE_SHAPE", [1, 64], [1, 32]),
        ],
        ids=["str", "dtype", "enum", "list"],
    )
    def test_every_foldable_constant_kind_invalidates(self, name, first, second):
        """A constant the specializer folds must also move the key.

        These four kinds only became foldable alongside this test; before that a
        body could not name them at all. Had the key not been extended with them,
        rebinding one would silently hand back an artifact built from the old
        value — the failure mode the int/float/bool tracking already prevents.
        """
        a = pl.jit(_with_globals(_global_cast, **{name: first}))
        b = pl.jit(_with_globals(_global_cast, **{name: second}))
        assert a._get_source_hash() != b._get_source_hash()

    def test_unfoldable_constant_does_not_invalidate(self):
        """A value with no source form cannot change the generated source, so it cannot change the key."""
        a = pl.jit(_with_globals(_global_opaque, _CACHE_OPAQUE=object()))
        b = pl.jit(_with_globals(_global_opaque, _CACHE_OPAQUE=object()))
        assert a._get_source_hash() == b._get_source_hash()

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


def _scalar_kernel(
    x: pl.Tensor[[16, 16], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    row: pl.Scalar[pl.INT32],
    bias: pl.Scalar[pl.FP32],
    flag: pl.Scalar[pl.BOOL],
):
    """One parameter per supported scalar kind; ``flag`` is carried, not read."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [row, 0], [16, 16])
        pl.store(pl.add(tile, bias), [0, 0], out)
    return out


class TestRuntimeScalarParameters:
    """A scalar parameter is a runtime value, not a specialization (issue #2751).

    Changing a token count, an offset, or a scale must reuse the compiled
    artifact and pass the new value at execution time.
    """

    @pytest.fixture
    def compile_programs(self, monkeypatch):
        """Run real specialization and parsing without invoking toolchains."""
        programs = []

        def compile_program(program, **kwargs):
            programs.append(program)
            return program

        monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "compile", compile_program)
        return programs

    def test_one_compilation_serves_every_value(self, compile_programs):
        torch = pytest.importorskip("torch")

        kernel = pl.jit(_scalar_kernel)
        x = torch.zeros(16, 16, dtype=torch.float32)
        out = torch.zeros_like(x)

        first = kernel.compile(x, out, 0, 1.0, False)
        for row, bias, flag in ((1, 2.0, True), (2, -0.0, False), (3, 0.5, True)):
            assert kernel.compile(x, out, row, bias, flag) is first
        assert len(compile_programs) == 1

    def test_generated_program_keeps_each_scalar_symbolic(self):
        """Every scalar survives as a parameter and every use reads the symbol.

        A folded value would leave the declared parameter unused and bake one
        call site's number into the artifact.
        """
        torch = pytest.importorskip("torch")

        kernel = pl.jit(_scalar_kernel)
        x = torch.zeros(16, 16, dtype=torch.float32)
        source = kernel.specialize(x, torch.zeros_like(x), 7, 3.5, True).as_python()

        assert "row: pl.Scalar[pl.INT32]" in source
        assert "bias: pl.Scalar[pl.FP32]" in source
        assert "flag: pl.Scalar[pl.BOOL]" in source
        assert "pl.tile.load(x, [row, 0]" in source
        assert "pl.tile.adds(tile, bias)" in source


def _constexpr_kernel(
    x: pl.Tensor[[32, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    scale: pl.Scalar[pl.FP32],
    BLOCK: pl.constexpr,
):
    """One parameter of each kind, so the two contracts are exercised together."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [BLOCK, BLOCK])
        pl.store(pl.add(tile, scale), [0, 0], out)
    return out


_CONSTEXPR_SETTINGS = types.SimpleNamespace(BLOCK=16)
# A module global deliberately sharing a name with a runtime parameter below.
n = 16


@pl.jit.incore
def _constexpr_tile_dep(
    a: pl.Tensor[[32, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    N: pl.constexpr,
):
    pl.store(pl.load(a, [0, 0], [N, N]), [0, 0], out)
    return out


@pl.jit
def _attr_cfg_entry(a: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pass
    return _constexpr_tile_dep(a, out, _CONSTEXPR_SETTINGS.BLOCK)


# Three call sites, two of them pinned to literals and the third steered by a
# config attribute. Flipping the attribute leaves the binding *set* at {16, 32}
# and only moves which specialization the third site reaches.
_TARGET_SWITCH_SETTINGS = types.SimpleNamespace(N=16)


@pl.jit
def _target_switch_entry(a: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pass
    out = _constexpr_tile_dep(a, out, 16)
    out = _constexpr_tile_dep(a, out, 32)
    return _constexpr_tile_dep(a, out, _TARGET_SWITCH_SETTINGS.N)


@pl.jit
def _shadowed_entry(
    a: pl.Tensor[[32, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pass
    return _constexpr_tile_dep(a, out, n)


class TestConstexprParameters:
    """``pl.constexpr`` selects a specialization; ``pl.Scalar`` does not (issue #2759).

    The pair has to be tested together: the point of the annotation is that two
    parameters of the same kernel sit on opposite sides of the compile-time /
    run-time line.
    """

    @pytest.fixture
    def compile_programs(self, monkeypatch):
        """Run real specialization and parsing without invoking toolchains."""
        programs = []

        def compile_program(program, **kwargs):
            programs.append(program)
            return program

        monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "compile", compile_program)
        return programs

    @pytest.fixture
    def samples(self):
        torch = pytest.importorskip("torch")
        x = torch.zeros(32, 32, dtype=torch.float32)
        return x, torch.zeros_like(x)

    def test_constant_selects_a_specialization_and_scalar_does_not(self, compile_programs, samples):
        x, out = samples
        kernel = pl.jit(_constexpr_kernel)

        first = kernel.compile(x, out, 1.0, 16)
        assert kernel.compile(x, out, 2.0, 16) is first, "a runtime Scalar must not split the cache"
        assert kernel.compile(x, out, 9.5, 16) is first
        second = kernel.compile(x, out, 1.0, 32)
        assert second is not first, "a constexpr value must select its own specialization"
        assert kernel.compile(x, out, 7.0, 32) is second
        assert len(compile_programs) == 2

    def test_value_reaches_the_body_and_leaves_the_signature(self, samples):
        """The constant folds into the IR; the parameter is gone from the ABI.

        Leaving it declared would demand an argument at dispatch for something
        the artifact already decided.
        """
        x, out = samples
        kernel = pl.jit(_constexpr_kernel)
        source = kernel.specialize(x, out, 1.0, 16).as_python()

        assert "pl.tile.load(x, [0, 0], [16, 16]" in source
        assert "scale: pl.Scalar[pl.FP32]" in source
        assert "BLOCK" not in source

    def test_nothing_is_dispatched_for_a_constexpr_parameter(self, samples, monkeypatch):
        """The runtime ABI, not just the generated source, drops the parameter.

        Asserts the argument list ``_resolve_compiled`` actually returns — the
        one ``__call__`` dispatches — rather than rebuilding it here from
        ``param_names`` and ``arguments``. Rebuilding it would restate the
        production rule and pass even if the dispatch path regressed to
        forwarding the raw call arguments. Compilation is stubbed because only
        the argument list is under test.
        """
        x, out = samples
        kernel = pl.jit(_constexpr_kernel)
        monkeypatch.setattr(kernel, "_compile", lambda *a, **k: object())

        specialization, _ = kernel._resolve_specialization((x, out, 1.0, 16), {})
        assert "BLOCK" in specialization.param_names, "it is still a declared parameter"
        assert specialization.constexpr_values["BLOCK"] == "16"

        _compiled, ordered_args, _config = kernel._resolve_compiled((x, out, 1.0, 16), {})
        assert ordered_args == [x, out, 1.0], "the constexpr value must not reach dispatch"

    def test_lower_folds_the_constant_and_requires_a_value(self, samples):
        """``lower()`` honours the same binding rules as ``compile()``.

        It reaches specialization by the same path, but nothing covered it, so
        a regression that reached only this entry point would have shipped.
        """
        x, out = samples
        kernel = pl.jit(_constexpr_kernel)
        source = kernel.lower(x, out, 1.0, 16).as_python()

        # Post-pipeline, so assert on the tile the constant sized rather than
        # the pre-SSA call text.
        assert "pl.Tile[[16, 16], pl.FP32" in source
        assert "BLOCK" not in source
        assert "pl.Scalar[pl.FP32]" in source, "the runtime scalar still survives as a parameter"

        with pytest.raises(TypeError, match=r"constexpr parameter 'BLOCK'.*no source form"):
            kernel.lower(x, out, 1.0, object())

    def test_a_value_with_no_source_form_is_rejected(self, samples):
        x, out = samples
        kernel = pl.jit(_constexpr_kernel)

        with pytest.raises(TypeError, match=r"constexpr parameter 'BLOCK'.*no source form"):
            kernel.specialize(x, out, 1.0, object())

    def test_signature_mode_requires_a_value(self):
        """Unlike a scalar, a constexpr has no runtime slot to fall back on."""

        @pl.jit
        def sig_kernel(
            x: pl.Tensor[[32, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            BLOCK: pl.constexpr,
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.store(pl.load(x, [0, 0], [BLOCK, BLOCK]), [0, 0], out)
            return out

        with pytest.raises(TypeError, match=r"constexpr parameter 'BLOCK' has no value"):
            sig_kernel.specialize()
        assert "[8, 8]" in sig_kernel.specialize(BLOCK=8).as_python()

    def test_a_kernel_without_constexpr_keeps_its_identity(self):
        """The request hash is untouched when no constexpr parameter is bound.

        Layering the binding onto ``source_hash`` must not invalidate every
        existing artifact just for existing.
        """
        from pypto.jit.decorator import _request_source_hash  # noqa: PLC0415

        at_16 = [(0, "m", "k", "BLOCK", "16")]
        at_32 = [(0, "m", "k", "BLOCK", "32")]
        assert _request_source_hash("abc", []) == "abc"
        assert _request_source_hash("abc", at_16) != "abc"
        assert _request_source_hash("abc", at_16) != _request_source_hash("abc", at_32)

    def test_a_dep_only_constant_reaches_the_cache_key(self):
        """A value only the dep call site names must still move the key.

        ``_get_source_hash`` can render free *names* only, so an attribute on a
        config object was invisible to it: the generated source changed while
        the key stood still, and the artifact built for the old value was served.
        """
        from pypto.jit.decorator import _request_source_hash  # noqa: PLC0415

        torch = pytest.importorskip("torch")
        x = torch.zeros(32, 32, dtype=torch.float32)
        out = torch.zeros_like(x)

        def request_hash():
            bindings = _attr_cfg_entry._resolve_constexpr_bindings({})
            return _request_source_hash(
                _attr_cfg_entry._get_source_hash(),
                _attr_cfg_entry._constexpr_identity_records(bindings),
            )

        _CONSTEXPR_SETTINGS.BLOCK = 16
        before, source_before = request_hash(), _attr_cfg_entry.specialize(x, out).as_python()
        _CONSTEXPR_SETTINGS.BLOCK = 32
        after, source_after = request_hash(), _attr_cfg_entry.specialize(x, out).as_python()

        assert "[16, 16]" in source_before
        assert "[32, 32]" in source_after
        assert before != after

    def test_a_shadowed_global_does_not_fold_a_runtime_value(self):
        """A caller local wins over a same-named global, as everywhere else.

        With a module-level ``n = 16`` beside a runtime scalar parameter also
        called ``n``, the dep was bound to 16 and compiled against a constant
        the caller never passed.
        """
        torch = pytest.importorskip("torch")
        x = torch.zeros(32, 32, dtype=torch.float32)

        with pytest.raises(TypeError, match=r"'N' is bound to 'n'.*no compile-time value"):
            _shadowed_entry.specialize(x, torch.zeros_like(x), 4)

    def test_every_specialization_of_a_split_dep_reaches_the_key(self):
        """A dep emitted twice contributes both bindings to the identity.

        One record per *function* was enough while a dep had one binding. Now
        that two call sites compile it separately, a key carrying only the
        first would let a program calling the dep at 16 and 32 collide with one
        calling it twice at 16 — different programs, and the second would be
        served the first's artifact.
        """

        def folded(entry):
            plan = entry._resolve_constexpr_bindings({})
            return sorted(text for *_, name, text in entry._constexpr_identity_records(plan) if name == "N")

        @pl.jit
        def split(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            o = _constexpr_tile_dep(a, o, 16)
            return _constexpr_tile_dep(a, o, 32)

        @pl.jit
        def same(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            o = _constexpr_tile_dep(a, o, 16)
            return _constexpr_tile_dep(a, o, 16)

        assert folded(split) == ["16", "32"]
        assert folded(same) == ["16"]

    def test_retargeting_a_call_site_moves_the_key(self):
        """Which specialization a site reaches is identity, not just which exist.

        With calls at ``16``, ``32`` and ``cfg.N``, flipping ``cfg.N`` between
        the two literals leaves the binding set at ``{16, 32}`` and only moves
        the third call's target. Recording the bindings alone left the key
        still while the generated program changed, and an attribute is
        invisible to ``_get_source_hash``, so the artifact compiled for the
        other target was served.
        """
        from pypto.jit.decorator import _request_source_hash  # noqa: PLC0415

        torch = pytest.importorskip("torch")
        x = torch.zeros(32, 32, dtype=torch.float32)
        out = torch.zeros_like(x)

        def probe():
            entry = _target_switch_entry
            plan = entry._resolve_constexpr_bindings({})
            key = _request_source_hash(entry._get_source_hash(), entry._constexpr_identity_records(plan))
            source = entry.specialize(x, out).as_python()
            return key, re.findall(r"self\.(_constexpr_tile_dep(?:__\d+)?)\(", source)

        original = _TARGET_SWITCH_SETTINGS.N
        try:
            _TARGET_SWITCH_SETTINGS.N = 16
            key_16, calls_16 = probe()
            _TARGET_SWITCH_SETTINGS.N = 32
            key_32, calls_32 = probe()
        finally:
            _TARGET_SWITCH_SETTINGS.N = original

        # Only the third site moves; the first two stay pinned to their literals.
        assert calls_16 == ["_constexpr_tile_dep", "_constexpr_tile_dep__2", "_constexpr_tile_dep"]
        assert calls_32 == ["_constexpr_tile_dep", "_constexpr_tile_dep__2", "_constexpr_tile_dep__2"]
        assert key_16 != key_32

    def test_an_unsplit_program_records_no_call_wiring(self):
        """The wiring is redundant until something splits, so it is not emitted.

        With one binding per function each call name has a single possible
        target, so adding wiring records there would invalidate every existing
        key and its artifacts to say nothing new.
        """
        plan = _attr_cfg_entry._resolve_constexpr_bindings({})
        records = _attr_cfg_entry._constexpr_identity_records(plan)

        assert records, "the dep's own binding must still be recorded"
        assert all(len(record) == 5 for record in records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
