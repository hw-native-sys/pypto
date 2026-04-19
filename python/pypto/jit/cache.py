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
L2 cache: on-disk under ~/.cache/pypto/jit/<key_hash>/ (survives process restarts).

Cache key encodes source hash, tensor shapes/dtypes, scalar values, and the
PyPTO version so that a version upgrade automatically invalidates stale entries.
Dynamic dimensions (marked via bind_dynamic) are stored as None in the key
so different concrete values for that dimension share the same cache entry.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from pypto.pypto_core import DataType

# Stable PyPTO version stamp included in every cache key so that upgrading
# PyPTO (which may change the pass pipeline or codegen) automatically
# invalidates all previously cached compilation artifacts.
try:
    from pypto import __version__ as _PYPTO_VERSION
except Exception:
    _PYPTO_VERSION = "unknown"

# Root directory for L2 on-disk cache.
_L2_CACHE_ROOT = Path.home() / ".cache" / "pypto" / "jit"


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

    The PyPTO version is mixed in so that upgrading PyPTO invalidates all
    previously cached compilation artifacts without requiring manual cache
    clearing (addresses issue #878 Q3).

    Args:
        sources: List of source code strings (main function + all deps).

    Returns:
        Hex digest string (SHA-256, first 16 chars for brevity).
    """
    h = hashlib.sha256()
    h.update(_PYPTO_VERSION.encode())
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


def _key_to_hash(key: CacheKey) -> str:
    """Return a SHA-256 hex digest of the cache key (for L2 directory naming)."""
    h = hashlib.sha256()
    h.update(json.dumps(key, default=str).encode())
    return h.hexdigest()


def l2_lookup(key: CacheKey) -> str | None:
    """Look up a compiled output_dir from the L2 on-disk cache.

    The L2 cache stores the path to the compiled artifacts directory produced
    by ``ir.compile()``.  The path is written to ``manifest.json`` inside the
    cache slot.  Returns ``None`` on a cache miss or if the stored path no
    longer exists on disk.

    Args:
        key: Cache key for this specialization.

    Returns:
        Absolute path string to the compiled output directory, or ``None``
        on a miss.
    """
    slot = _L2_CACHE_ROOT / _key_to_hash(key)
    manifest = slot / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
        output_dir = data.get("output_dir", "")
        if output_dir and Path(output_dir).exists():
            return output_dir
    except Exception:
        pass
    return None


def l2_store(key: CacheKey, output_dir: str) -> None:
    """Store a compiled output_dir in the L2 on-disk cache.

    Copies the entire ``output_dir`` tree into the cache slot so the artifacts
    survive even if the original directory is cleaned up.  Writes a
    ``manifest.json`` pointing to the cached copy.

    Args:
        key: Cache key for this specialization.
        output_dir: Path to the directory produced by ``ir.compile()``.
    """
    slot = _L2_CACHE_ROOT / _key_to_hash(key)
    try:
        slot.mkdir(parents=True, exist_ok=True)
        artifacts_dir = slot / "artifacts"
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        shutil.copytree(output_dir, artifacts_dir)
        manifest = slot / "manifest.json"
        manifest.write_text(json.dumps({"output_dir": str(artifacts_dir)}))
    except Exception:
        # L2 cache write failure is non-fatal; L1 cache will still be used.
        pass


__all__ = [
    "CacheKey",
    "ScalarCacheInfo",
    "TensorCacheInfo",
    "compute_source_hash",
    "l2_lookup",
    "l2_store",
    "make_cache_key",
]
