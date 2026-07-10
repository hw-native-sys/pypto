# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for python/pypto/jit/cache.py."""

import pytest
from pypto.ir import OptimizationStrategy
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.jit.cache import (
    compute_source_hash,
    make_cache_key,
)
from pypto.jit.decorator import _resolve_enable_pypto_l0c_double_buffer, _resolve_memory_planner
from pypto.pypto_core import DataType, passes
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
            ("memory_planner", None),
            ("enable_pypto_l0c_double_buffer", False),
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

    def test_different_strategy_causes_miss(self):
        """Same shapes/dtypes compiled with different strategies must not collide.

        Keeps an A/B comparison honest: calling one kernel with two
        strategies must compile twice, not serve the first artifact twice.
        """
        k_default = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=OptimizationStrategy.Default,
        )
        k_debug = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            strategy=OptimizationStrategy.DebugTileOptimization,
        )
        assert k_default != k_debug

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
        """The planner decides whether physical addresses are baked into the
        artifact (ptoas level3 vs level2), so it must split the cache."""
        keys = [
            self._make_key(
                param_names=["a"],
                tensor_shapes={"a": (8, 8)},
                tensor_dtypes={"a": DataType.FP32},
                memory_planner=planner,
            )
            for planner in (None, MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        ]
        assert len(set(keys)) == len(keys), f"planner must split the cache key, got {keys}"

    def test_dbc_double_buffer_flag_splits_key(self):
        """The PyPTO dbC=2 opt-in changes the AutoTileMatmulL0/MemoryReuse output,
        so a kernel compiled with it off must not reuse that artifact when later
        called with it on."""
        key_off = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            enable_pypto_l0c_double_buffer=False,
        )
        key_on = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (8, 8)},
            tensor_dtypes={"a": DataType.FP32},
            enable_pypto_l0c_double_buffer=True,
        )
        assert key_off != key_on, "dbC=2 opt-in must split the cache key"


class TestResolveMemoryPlanner:
    """The planner the JIT keys on must match the one ``ir.compile()`` will use."""

    def test_defaults_to_pypto(self):
        assert _resolve_memory_planner(None) == MemoryPlanner.PYPTO

    def test_reads_the_active_pass_context(self):
        """The planner is usually selected by wrapping the call in a PassContext,
        which never reaches RunConfig. Keying only on RunConfig would let such a
        call reuse a PYPTO-compiled artifact."""
        with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
            assert _resolve_memory_planner(None) == MemoryPlanner.PTOAS
        assert _resolve_memory_planner(None) == MemoryPlanner.PYPTO


class TestResolveEnablePyptoL0cDoubleBuffer:
    """The dbC=2 opt-in the JIT keys on must match the one ``ir.compile()`` inherits."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
