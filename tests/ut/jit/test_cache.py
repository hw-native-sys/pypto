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
from pypto.jit.cache import (
    compute_source_hash,
    make_cache_key,
)
from pypto.pypto_core import DataType


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
    def _make_key(
        self,
        source_hash="abc",
        param_names=None,
        tensor_shapes=None,
        tensor_dtypes=None,
        dynamic_dims=None,
        scalar_values=None,
        platform=None,
    ):
        return make_cache_key(
            source_hash=source_hash,
            param_names=param_names or [],
            tensor_shapes=tensor_shapes or {},
            tensor_dtypes=tensor_dtypes or {},
            dynamic_dims=dynamic_dims or set(),
            scalar_values=scalar_values or {},
            platform=platform,
        )

    def test_basic_key_structure(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 128)},
            tensor_dtypes={"a": DataType.FP32},
        )
        assert isinstance(key, tuple)
        assert len(key) == 4
        source_hash, platform, tensor_part, scalar_part = key
        assert source_hash == "abc"
        assert platform is None
        assert isinstance(tensor_part, tuple)
        assert isinstance(scalar_part, tuple)

    def test_tensor_shape_in_key(self):
        key = self._make_key(
            param_names=["a"],
            tensor_shapes={"a": (128, 64)},
            tensor_dtypes={"a": DataType.FP32},
        )
        _, _, tensor_part, _ = key
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
        _, _, tensor_part, _ = key
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
        _, _, _, scalar_part = key
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
        _, _, tensor_part, _ = key
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
