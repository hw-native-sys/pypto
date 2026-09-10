# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""JIT request identity, immutable publication and compatible object reuse."""

import logging
import os
import struct
import tempfile
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import cache, lru_cache
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from pypto._cache_config import CacheConfig, record_bypass, record_stats, time_stage
from pypto._identity import digest_record, fingerprint_extra_sources
from pypto.pypto_core import DataType

from ._artifact_manifest import ArtifactKey, ArtifactSpec, ArtifactState, BuildKind
from ._toolchain import capture_toolchain
from .artifact_cache import ArtifactLookup, ArtifactStore, BuildFailure, LookupStatus
from .cache import CacheKey

logger = logging.getLogger(__name__)
_count_initial_lookup: ContextVar[bool] = ContextVar("jit_count_initial_lookup", default=False)


class _PrivateBuild(Exception):
    def __init__(self, compiled: Any, reason: str):
        self.compiled = compiled
        self.reason = reason


def _semantic_environment() -> tuple[str | None, ...]:
    locale = os.environ.get("LC_ALL")
    return (
        locale or os.environ.get("LC_CTYPE") or os.environ.get("LANG"),
        os.environ.get("SOURCE_DATE_EPOCH"),
        os.environ.get("PYTHONHASHSEED"),
        os.environ.get("PYTHONOPTIMIZE"),
    )


def _record(value: Any) -> Any:
    """Normalize known key types explicitly; never fall back to repr/string."""
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, DataType):
        return ("pypto.DataType", value.code())
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, value.name)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__qualname__, {f.name: _record(getattr(value, f.name)) for f in fields(value)})
    if type(value) in (list, tuple, CacheKey):
        return tuple(_record(item) for item in value)
    # nanobind enum types do not derive from Python's enum.Enum.
    if type(value).__module__.startswith("pypto.pypto_core") and isinstance(
        getattr(value, "name", None), str
    ):
        return (type(value).__module__, type(value).__qualname__, getattr(value, "name"))
    raise TypeError(f"Unsupported specialization identity type: {type(value).__name__}")


@lru_cache(maxsize=1024)
def _specialization_digest(key: Any, scalar_tags: tuple[Any, ...]) -> str:
    # scalar_tags guards Python's True == 1 == 1.0 and signed-zero equality.
    return digest_record(_record(key))


def _typed_specialization(key: CacheKey) -> str:
    tags = tuple(
        (s.name, type(s.value).__name__, struct.pack(">d", s.value) if type(s.value) is float else s.value)
        for s in key.scalar_infos
    )
    return _specialization_digest(key, tags)


class JITArtifactStore(ArtifactStore):
    """Count actual lazy binary transactions, including ordinary execution."""

    def lookup(self, key: ArtifactKey, spec: ArtifactSpec) -> ArtifactLookup:
        result = super().lookup(key, spec)
        if _count_initial_lookup.get():
            _count_initial_lookup.set(False)
            _event(result.status)
        return result

    def get_or_build(self, key: ArtifactKey, spec: ArtifactSpec, builder: Callable[[Path], Any]) -> Any:
        binary = spec.state is ArtifactState.BINARY_READY
        builder_ns = 0

        def measured(directory: Path) -> Any:
            nonlocal builder_ns
            start = perf_counter_ns()
            if binary:
                record_stats(binary_builds=1)
            try:
                return builder(directory)
            finally:
                builder_ns = perf_counter_ns() - start
                if binary:
                    record_stats(build_ns=builder_ns)

        token = _count_initial_lookup.set(binary)
        start = perf_counter_ns()
        try:
            result = super().get_or_build(key, spec, measured)
            if binary and result.failure in (BuildFailure.LOCK, BuildFailure.PUBLICATION):
                record_stats(storage_errors=1)
            return result
        finally:
            _count_initial_lookup.reset(token)
            record_stats(lookup_ns=perf_counter_ns() - start - builder_ns)


def _event(status: LookupStatus) -> None:
    if status is LookupStatus.INVALID:
        record_stats(invalid_entries=1)
    elif status is LookupStatus.STORAGE_ERROR:
        record_stats(storage_errors=1)


def _bypass(reason: str) -> None:
    record_bypass(reason)
    logger.info(f"Persistent JIT cache bypass: {reason}")


@cache
def _fallback_private_root(cache_root: Path, process_id: int) -> Path:
    """Allocate a retained, unpredictable 0700 parent without TMPDIR probes.

    Include the PID in memoization so forked children allocate their own parent.
    The build and runtime objects own these paths; there is no online cleanup.
    """
    for candidate in (Path("/tmp"), Path("/var/tmp")):
        parent = candidate.resolve()
        if parent != cache_root and cache_root not in parent.parents:
            return Path(tempfile.mkdtemp(prefix=f"pypto-jit-{os.getuid()}-", dir=parent))
    raise OSError("No private temporary directory outside the artifact cache root")


def resolve_persistent(
    owner: Any,
    object_key: Any,
    config: CacheConfig,
    build: Callable[..., Any],
    source: Callable[[], str],
    *,
    platform: str,
    runtime_name: str,
    distributed: bool,
) -> Any:
    """Capture identity before object lookup; keep policy out of content keys."""
    from pypto.runtime._artifact_runtime import bind_artifact, restore_artifact  # noqa: PLC0415
    from pypto.runtime._artifact_sources import package_generated_sources  # noqa: PLC0415
    from pypto.runtime._extern_includes import UnsupportedArtifactInput  # noqa: PLC0415
    from pypto.runtime._prebuilt import ready_spec  # noqa: PLC0415

    with time_stage("lookup_ns"):
        identity = capture_toolchain(platform, runtime_name)
        extra = fingerprint_extra_sources(config.extra_source_paths, config.extra_fingerprint)
    if not identity.usable or extra.digest is None:
        reason = extra.failure or "; ".join(f"{f.component}: {f.reason}" for f in identity.failures)
        _bypass(reason)
        return build()
    with time_stage("lookup_ns"):
        kind = BuildKind.DISTRIBUTED if distributed else BuildKind.SINGLE_CHIP
        source_before = source()
        try:
            specialization = _typed_specialization(object_key)
        except Exception as exc:
            identity_failure = f"Specialization identity unavailable: {exc}"
        else:
            identity_failure = None
    if identity_failure is not None:
        _bypass(identity_failure)
        return build()
    with time_stage("lookup_ns"):
        semantic = _semantic_environment()
        compatible = (
            identity,
            source_before,
            extra.digest,
            specialization,
            kind,
            semantic,
            config.root,
            config.readonly,
        )
        cached = owner._artifact_objects.get(compatible)
        if cached is not None:
            record_stats(object_hits=1)
            return cached
        key = ArtifactKey(
            identity,
            digest_record((source_before, extra.digest)),
            digest_record((specialization, kind.value, semantic)),
        )
        assert config.root is not None
        # Avoid tempfile's write probes under a readonly TMPDIR. The usual
        # build_output parent is created lazily; the exceptional temporary
        # parent is allocated securely before any runtime output can use it.
        private_root = Path.cwd() / "build_output"
        if private_root == config.root or config.root in private_root.parents:
            private_root = _fallback_private_root(config.root, os.getpid())
        store = JITArtifactStore(config.root, readonly=config.readonly, private_root=private_root)
        spec = ArtifactSpec(
            ArtifactState.GENERATED,
            kind,
            ("distributed_meta.json", "orchestration/host_orch.py")
            if distributed
            else ("compiled_meta.json", "kernel_config.py"),
        )
        initial = store.lookup(key, spec)
        _event(initial.status)
        handle = initial.handle
        if handle is not None:
            ready = store.lookup(key, ready_spec(handle.directory, spec))
            _event(ready.status)
            record_stats(**{"ready_hits" if ready.handle is not None else "generated_hits": 1})
            handle = ready.handle or handle
            compiled = restore_artifact(store, handle, private_root / f"run-{uuid.uuid4().hex}")
            owner._artifact_objects[compatible] = compiled
            return compiled
        record_stats(misses=1)

    def generate(directory: Path) -> Any:
        compiled = build(output_dir=str(directory))
        with time_stage("build_ns"):
            try:
                package_generated_sources(directory, kind)
            except UnsupportedArtifactInput as exc:
                raise _PrivateBuild(compiled, str(exc)) from exc
            # Mutable sources may change while compilation or a competing
            # process runs. Such a result must remain private.
            current_extra = fingerprint_extra_sources(config.extra_source_paths, config.extra_fingerprint)
            if current_extra != extra or source() != source_before:
                raise _PrivateBuild(compiled, "Application sources changed during compilation")
        return compiled

    try:
        result = store.get_or_build(key, spec, generate)
    except _PrivateBuild as exc:
        _bypass(exc.reason)
        return exc.compiled
    with time_stage("lookup_ns"):
        if result.handle is None:
            if result.failure in (BuildFailure.LOCK, BuildFailure.PUBLICATION) or (
                result.failure is BuildFailure.STORAGE and initial.status is not LookupStatus.STORAGE_ERROR
            ):
                record_stats(storage_errors=1)
                logger.info(f"Persistent JIT publication unavailable: {result.reason}")
            compiled = result.value
        elif result.value is not None:
            compiled = result.value
            bind_artifact(compiled, store, result.handle, private_root / f"run-{uuid.uuid4().hex}")
        else:
            compiled = restore_artifact(store, result.handle, private_root / f"run-{uuid.uuid4().hex}")
        owner._artifact_objects[compatible] = compiled
        return compiled
