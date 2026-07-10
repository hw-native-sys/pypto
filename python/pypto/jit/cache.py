# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compilation cache support for @pl.jit functions.

The active cache used by JITFunction is an in-memory dict on each instance (L1).
This module also defines the cache-key construction utilities and an on-disk
L2 cache implementation (l2_lookup / l2_store) that can be wired in by callers
to persist compiled artifacts across process restarts.

Cache keys encode source hash, tensor shapes/dtypes, scalar values, and the
PyPTO version so that a version upgrade automatically invalidates stale entries.
Dynamic dimensions (marked via bind_dynamic) are stored as None in the key
so different concrete values for that dimension share the same cache entry.
"""

import dataclasses
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypto.pypto_core import DataType

if TYPE_CHECKING:
    from pypto.ir.pass_manager import OptimizationStrategy

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


# A cache key is a tuple of
# (source_hash, platform, strategy, tensor_infos, scalar_infos, dist_config, compile_opts).
# Using a plain tuple keeps it hashable without a custom __hash__.
CacheKey = tuple[
    str,
    str | None,
    "OptimizationStrategy | None",
    tuple[TensorCacheInfo, ...],
    tuple[ScalarCacheInfo, ...],
    tuple[Any, ...] | None,
    tuple[Any, ...] | None,
]


def _freeze(value: Any) -> Any:
    """Recursively convert a value into a hashable, deterministic form.

    Dataclasses become a tuple of ``(field_name, frozen_value)`` pairs and
    lists/tuples become tuples, so a ``DistributedConfig`` (which has a mutable
    ``device_ids`` list and is therefore unhashable) can participate in the
    cache key. Field-name pairing keeps the key stable and self-describing if
    new fields are added.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple((f.name, _freeze(getattr(value, f.name))) for f in dataclasses.fields(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


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


def make_cache_key(  # noqa: PLR0913
    source_hash: str,
    param_names: list[str],
    tensor_shapes: dict[str, tuple[int, ...]],
    tensor_dtypes: dict[str, DataType],
    dynamic_dims: set[tuple[str, int]],
    scalar_values: dict[str, int | float | bool],
    platform: str | None = None,
    strategy: "OptimizationStrategy | None" = None,
    distributed_config: Any = None,
    analyze_auto_scopes_for_deps: bool = False,
    use_ptoas_multi_buffer: bool | None = None,
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
        platform: Target platform string (e.g. "a2a3sim"). Included in the key
            because compiled artifacts are platform-specific; a cache entry
            compiled for one platform must not be reused for another.
        strategy: Optimization strategy applied during compilation (an
            ``OptimizationStrategy`` member, or ``None`` for the JIT default).
            Included in the key because the strategy changes the compiled
            artifact; without it, calling the same kernel with two strategies
            (an A/B comparison) would return the first-compiled artifact for
            both.
        distributed_config: Optional ``DistributedConfig`` forwarded to
            ``ir.compile()`` on the ``@pl.jit.host`` path. Included in the key
            because it is baked into the resulting
            ``DistributedCompiledProgram`` and drives per-rank dispatch; without
            it, calling the same host kernel with two different ``device_ids``
            would silently reuse the first-compiled artifact. ``None`` (the
            single-chip default) leaves the key unchanged for non-distributed
            callers.
        analyze_auto_scopes_for_deps: Compile-side switch for deriving explicit
            task dependencies from AUTO runtime scopes. Included in the key
            because it changes generated orchestration dependencies.
        use_ptoas_multi_buffer: Compile-side switch for ptoas multi-buffer
            lowering. Included in the key because it changes the emitted IR /
            kernel (alloc_multi_tile vs replicated alloc_tile); without it the
            same kernel compiled with the switch on and off would collide.

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

    dist_key = _freeze(distributed_config) if distributed_config is not None else None
    compile_opts = (
        ("analyze_auto_scopes_for_deps", analyze_auto_scopes_for_deps),
        ("use_ptoas_multi_buffer", use_ptoas_multi_buffer),
    )
    return (
        source_hash,
        platform,
        strategy,
        tuple(tensor_infos),
        tuple(scalar_infos),
        dist_key,
        compile_opts,
    )


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
