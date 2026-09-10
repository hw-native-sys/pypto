# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public persistent-cache policy and inexpensive process-local statistics."""

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter_ns


@dataclass(frozen=True)
class CacheConfig:
    """Complete cache policy; explicit objects replace lower-precedence defaults.

    Installation files must remain immutable for the process lifetime. Extra
    application sources are read on each request. Cached code requires trusted
    writers. Read-only stores permit private builds outside the cache root.
    """

    enabled: bool = False
    root: Path | None = None
    readonly: bool = False
    extra_source_paths: tuple[Path, ...] = ()
    extra_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.readonly) is not bool:
            raise TypeError("CacheConfig.enabled and readonly must be bool")
        if self.root is not None and not isinstance(self.root, Path):
            raise TypeError(f"CacheConfig.root must be Path or None, got {self.root!r}")
        if type(self.extra_source_paths) is not tuple or any(
            not isinstance(p, Path) for p in self.extra_source_paths
        ):
            raise TypeError("CacheConfig.extra_source_paths must be a tuple of Path objects")
        if self.extra_fingerprint is not None and type(self.extra_fingerprint) is not str:
            raise TypeError("CacheConfig.extra_fingerprint must be str or None")


_DEFAULT_CONFIG = CacheConfig()


@dataclass(frozen=True)
class CacheStats:
    """Cumulative process-local counts and nanosecond totals, excluding execution."""

    requests: int = 0
    object_hits: int = 0
    ready_hits: int = 0
    generated_hits: int = 0
    misses: int = 0
    disabled_requests: int = 0
    forced_rebuilds: int = 0
    bypasses: int = 0
    last_bypass_reason: str | None = None
    invalid_entries: int = 0
    storage_errors: int = 0
    generation_builds: int = 0
    binary_builds: int = 0
    lookup_ns: int = 0
    build_ns: int = 0


_lock = threading.Lock()


@dataclass
class _Policy:
    override: CacheConfig | None = None


_policy = _Policy()
_totals = asdict(CacheStats())


def configure_cache(config: CacheConfig | None) -> None:
    """Set process defaults; None restores environment/default precedence."""
    if config is not None and not isinstance(config, CacheConfig):
        raise TypeError(f"Expected CacheConfig or None, got {type(config).__name__}")
    with _lock:
        _policy.override = config


def cache_stats() -> CacheStats:
    """Return an immutable snapshot without scanning storage or resetting counts."""
    with _lock:
        return CacheStats(**_totals)


def record_stats(**increments: int) -> None:
    with _lock:
        for name, value in increments.items():
            _totals[name] += value


def record_bypass(reason: str) -> None:
    """Record an enabled-cache fallback and its latest diagnostic atomically."""
    with _lock:
        _totals["bypasses"] += 1
        _totals["last_bypass_reason"] = reason


@contextmanager
def time_stage(stage: str) -> Iterator[None]:
    start = perf_counter_ns()
    try:
        yield
    finally:
        record_stats(**{stage: perf_counter_ns() - start})


def _boolean(name: str, value: str) -> bool:
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {value!r}")
    return value == "1"


def capture_cache_config(per_call: CacheConfig | None) -> CacheConfig:
    with _lock:
        config = per_call if per_call is not None else _policy.override
    if config is None:
        root = os.environ.get("PYPTO_CACHE_DIR")
        enabled = os.environ.get("PYPTO_CACHE")
        readonly = os.environ.get("PYPTO_CACHE_READONLY")
        if root is None and enabled is None and readonly is None:
            return _DEFAULT_CONFIG
        config = CacheConfig(
            enabled=_boolean("PYPTO_CACHE", "0" if enabled is None else enabled),
            root=Path(root) if root else None,
            readonly=_boolean("PYPTO_CACHE_READONLY", "0" if readonly is None else readonly),
        )
    if not isinstance(config, CacheConfig):
        raise TypeError(f"RunConfig.cache_config must be CacheConfig or None, got {type(config).__name__}")
    if not config.enabled:
        return config
    return replace(
        config,
        root=(config.root or Path.home() / ".cache/pypto/jit").resolve(),
        extra_source_paths=tuple(Path.cwd() / p for p in config.extra_source_paths),
    )
