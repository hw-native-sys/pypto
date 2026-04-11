# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compilation cache for @pl.jit functions.

L1 cache: in-memory dict on each JITFunction instance.
Cache key encodes source hash, tensor shapes/dtypes, and scalar values.
Dynamic dimensions (marked via bind_dynamic) are stored as None in the key
so different concrete values for that dimension share the same cache entry.
"""

import hashlib
from dataclasses import dataclass

from pypto.pypto_core import DataType


@dataclass(frozen=True)
class TensorCacheInfo:
    """Per-tensor component of a cache key.

    Attributes:
        name: Parameter name.
        shape: Shape tuple with None for dynamic dimensions.
        dtype: DataType of the tensor.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: DataType


@dataclass(frozen=True)
class ScalarCacheInfo:
    """Per-scalar-param component of a cache key.

    Attributes:
        name: Parameter name.
        value: Concrete scalar value passed at this call site.
    """

    name: str
    value: int | float | bool


# A cache key is a tuple of (source_hash, tensor_infos, scalar_infos).
# Using a plain tuple keeps it hashable without a custom __hash__.
CacheKey = tuple[str, tuple[TensorCacheInfo, ...], tuple[ScalarCacheInfo, ...]]


def compute_source_hash(sources: list[str]) -> str:
    """Compute a stable hash over one or more source strings.

    Args:
        sources: List of source code strings (main function + all deps).

    Returns:
        Hex digest string (SHA-256, first 16 chars for brevity).
    """
    h = hashlib.sha256()
    for src in sources:
        h.update(src.encode())
    return h.hexdigest()[:16]


def make_cache_key(
    source_hash: str,
    param_names: list[str],
    tensor_shapes: dict[str, tuple[int, ...]],
    tensor_dtypes: dict[str, DataType],
    dynamic_dims: set[tuple[str, int]],
    scalar_values: dict[str, int | float | bool],
) -> CacheKey:
    """Build a cache key for a JIT call site.

    Args:
        source_hash: Hash of function source code (and all dep sources).
        param_names: Ordered list of all parameter names (preserves arg order).
        tensor_shapes: Concrete shape per tensor parameter name.
        tensor_dtypes: DataType per tensor parameter name.
        dynamic_dims: Set of (param_name, dim_index) pairs that are dynamic.
            Dynamic dims are stored as None in the cache key so different
            concrete values for that dimension produce the same cache entry.
        scalar_values: Concrete value per scalar parameter name.

    Returns:
        Hashable CacheKey tuple.
    """
    tensor_infos = []
    for name in param_names:
        if name not in tensor_shapes:
            continue
        concrete_shape = tensor_shapes[name]
        keyed_shape = tuple(
            None if (name, i) in dynamic_dims else dim for i, dim in enumerate(concrete_shape)
        )
        tensor_infos.append(TensorCacheInfo(name=name, shape=keyed_shape, dtype=tensor_dtypes[name]))

    scalar_infos = []
    for name in param_names:
        if name not in scalar_values:
            continue
        scalar_infos.append(ScalarCacheInfo(name=name, value=scalar_values[name]))

    return (source_hash, tuple(tensor_infos), tuple(scalar_infos))


__all__ = [
    "CacheKey",
    "ScalarCacheInfo",
    "TensorCacheInfo",
    "compute_source_hash",
    "make_cache_key",
]
