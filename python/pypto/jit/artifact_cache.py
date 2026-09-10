# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal immutable artifact store, not yet connected to JIT dispatch.

Cooperating writers use persistent per-key locks and no-replace publication.
Readers never write. Cache errors preserve a usable private build; compiler
exceptions propagate. The cache is executable code from trusted writers.
"""

import ctypes
import errno
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pypto._fslock import file_lock

from ._artifact_manifest import (
    MANIFEST_NAME,
    ArtifactKey,
    ArtifactSpec,
    check_directory,
    encode_manifest,
    make_manifest,
    read_manifest,
)

T = TypeVar("T")


def _check_private_path(path: Path, cache_root: Path) -> None:
    """Keep private writes outside the entire canonical shared cache root."""
    if path == cache_root or cache_root in path.parents:
        raise ValueError(f"Artifact private directory must be outside the cache root: {path}")


class LookupStatus(Enum):
    """Observable lookup outcomes without mutation or silent repair."""

    HIT = "hit"
    MISS = "miss"
    INVALID = "invalid"
    STORAGE_ERROR = "storage_error"


class BuildDisposition(Enum):
    """Whether a request reused, published, or retained private output."""

    HIT = "hit"
    PUBLISHED = "published"
    PRIVATE = "private"


class BuildFailure(Enum):
    """Typed cache failure retaining a private build; independent of diagnostics."""

    INVALID = "invalid"
    STORAGE = "storage"
    LOCK = "lock"
    PUBLICATION = "publication"


@dataclass(frozen=True)
class ArtifactHandle:
    """A validated shared directory; consumers must treat its contents as immutable."""

    directory: Path
    key: ArtifactKey
    spec: ArtifactSpec
    cache_root: Path

    def materialize(self, destination: Path) -> None:
        """Copy validated payload to an empty private directory, without hardlinks.

        Use this when building BINARY_READY from GENERATED. The old completion
        marker is excluded. Copies are owner-writable and preserve owner execute permission.
        On failure, the caller owns the partial destination.
        """
        destination = destination.resolve()
        _check_private_path(destination, self.cache_root)
        manifest = read_manifest(self.directory, self.key, self.spec)
        check_directory(destination)
        if any(destination.iterdir()):
            raise ValueError(f"Artifact materialization requires an empty directory: {destination}")
        _copy_payload(self.directory, destination, manifest)
        if make_manifest(destination, self.key, self.spec) != manifest:
            raise ValueError(f"Artifact changed during materialization: {self.directory}")


@dataclass(frozen=True)
class ArtifactLookup:
    """A lookup result with a diagnostic reason for invalid/unavailable storage."""

    status: LookupStatus
    handle: ArtifactHandle | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ArtifactBuild(Generic[T]):
    """Result retaining the compiler value and any paths it may still reference.

    A hit has no private directory or builder value. A fresh build retains both
    even after publication: only its adapter knows when paths have been rebound
    and the original directory can be removed. There is no automatic cleanup.
    """

    disposition: BuildDisposition
    handle: ArtifactHandle | None = None
    private_directory: Path | None = None
    value: T | None = None
    reason: str | None = None
    failure: BuildFailure | None = None


_BuildRequest = tuple[Path, Path | None, bool, str, str]


def _lookup_failure(lookup: ArtifactLookup) -> BuildFailure | None:
    if lookup.status is LookupStatus.INVALID:
        return BuildFailure.INVALID
    if lookup.status is LookupStatus.STORAGE_ERROR:
        return BuildFailure.STORAGE
    return None


@dataclass(frozen=True)
class _BuildFlight:
    owner: int
    result: Future[ArtifactBuild[Any]]


class _BuildFlights:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """A forked child cannot wait on builders that only exist in its parent."""
        self.lock = threading.Lock()
        self.pending: dict[_BuildRequest, _BuildFlight] = {}

    def run(
        self, request: _BuildRequest, operation: Callable[[], ArtifactBuild[T]]
    ) -> tuple[ArtifactBuild[T], bool]:
        """Share one in-flight result or exception, retaining no completed entries."""
        with self.lock:
            flight = self.pending.get(request)
            leader = flight is None
            if flight is None:
                flight = _BuildFlight(threading.get_ident(), Future())
                self.pending[request] = flight
            elif flight.owner == threading.get_ident():
                raise RuntimeError("Artifact builder recursively requested its own in-flight build")
        if not leader:
            return flight.result.result(), False
        try:
            result = operation()
        except BaseException as exc:
            # Wake every waiter even for cancellation, then preserve the original
            # compiler exception. This is coordination, not a cache-error fallback.
            flight.result.set_exception(exc)
            raise
        else:
            flight.result.set_result(result)
            return result, True
        finally:
            with self.lock:
                if self.pending.get(request) is flight:
                    del self.pending[request]


_build_flights = _BuildFlights()
os.register_at_fork(after_in_child=_build_flights.reset)


def _mkdir(path: Path) -> None:
    # Check existing ancestors before mkdir can follow a corrupt cache symlink.
    for ancestor in (*reversed(path.parents), path):
        try:
            check_directory(ancestor)
        except FileNotFoundError:
            break
    path.mkdir(parents=True, exist_ok=True)
    check_directory(path)


def _copy_payload(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / entry["path"], target, follow_symlinks=False)
        # Read/write policy may change when sealing a prewarmed cache. Only
        # owner execute permission affects the contract; private copies remain
        # writable for the next compilation stage.
        target.chmod(0o700 if entry["executable"] else 0o600)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _complete_staging(directory: Path, manifest: dict[str, Any]) -> None:
    # Files and nested directory entries precede the completion marker.
    for entry in manifest["files"]:
        with (directory / entry["path"]).open("rb") as stream:
            os.fsync(stream.fileno())
    directories = {directory}
    for entry in manifest["files"]:
        parent = (directory / entry["path"]).parent
        while parent != directory:
            directories.add(parent)
            parent = parent.parent
    for parent in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _sync_directory(parent)
    with (directory / MANIFEST_NAME).open("xb") as stream:
        stream.write(encode_manifest(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(directory)


def _rename_noreplace(source: Path, destination: Path) -> None:
    # POSIX rename can replace an existing empty directory, including a corrupt
    # slot. Linux RENAME_NOREPLACE preserves even that slot for offline cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "Atomic no-replace artifact publication is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


class ArtifactStore:
    """Store validated stages under a complete environment and compilation key.

    Args:
        root: Trusted cache root; captured as a canonical absolute path.
        readonly: Never create locks, staging, or any other cache-root output.
        private_root: Writable parent for retained private builds, outside the
            cache root. Without one, hits work but requests needing a build fail.

    Writer support requires Linux no-replace rename, POSIX flock, and a
    filesystem honoring both. Concurrent cache mutation/deletion is unsupported.
    """

    def __init__(self, root: Path, *, readonly: bool = False, private_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.readonly = readonly
        self.private_root = private_root.resolve() if private_root is not None else None
        if self.private_root is not None:
            _check_private_path(self.private_root, self.root)

    def _slot(self, key: ArtifactKey, spec: ArtifactSpec) -> Path:
        return (
            self.root
            / "artifacts"
            / str(key.environment.digest)
            / key.digest
            / spec.digest
            / spec.state.value
        )

    def lookup(self, key: ArtifactKey, spec: ArtifactSpec) -> ArtifactLookup:
        """Validate all bytes and metadata, without creating directories or locks."""
        directory = self._slot(key, spec)
        try:
            check_directory(directory)
        except FileNotFoundError:
            return ArtifactLookup(LookupStatus.MISS)
        except ValueError as exc:
            return ArtifactLookup(LookupStatus.INVALID, reason=str(exc))
        except OSError as exc:
            return ArtifactLookup(LookupStatus.STORAGE_ERROR, reason=str(exc))
        try:
            read_manifest(directory, key, spec)
        except (ValueError, FileNotFoundError, UnicodeError, RecursionError) as exc:
            return ArtifactLookup(LookupStatus.INVALID, reason=str(exc))
        except OSError as exc:
            return ArtifactLookup(LookupStatus.STORAGE_ERROR, reason=str(exc))
        return ArtifactLookup(LookupStatus.HIT, ArtifactHandle(directory, key, spec, self.root))

    def get_or_build(
        self, key: ArtifactKey, spec: ArtifactSpec, builder: Callable[[Path], T]
    ) -> ArtifactBuild[T]:
        """Coalesce overlapping calls, recheck under a lock, and publish output.

        Builder and invalid-build errors propagate, including builder OSError.
        Storage failures return the completed private build with a reason.
        Within one process, matching root/private-root/readonly/key/spec calls
        share a private result, including its value. Their builders must be
        interchangeable and their values safe to share between waiting callers.
        Completed private results are not memoized for subsequent requests.
        Builders must not recursively acquire the same key's lock.
        """
        request = (self.root, self.private_root, self.readonly, key.digest, spec.digest)
        result, leader = _build_flights.run(request, lambda: self._get_or_build(key, spec, builder))
        if not leader and result.handle is not None:
            return ArtifactBuild(BuildDisposition.HIT, handle=result.handle)
        return result

    def _get_or_build(
        self, key: ArtifactKey, spec: ArtifactSpec, builder: Callable[[Path], T]
    ) -> ArtifactBuild[T]:
        lookup = self.lookup(key, spec)
        if lookup.status is LookupStatus.HIT:
            return ArtifactBuild(BuildDisposition.HIT, handle=lookup.handle)
        if self.readonly:
            return self._build_private(
                key, spec, builder, lookup.reason or "Artifact cache is read-only", _lookup_failure(lookup)
            )
        with ExitStack() as stack:
            try:
                lock_directory = self.root / "locks"
                _mkdir(lock_directory)
                stack.enter_context(file_lock(lock_directory / f"{key.digest}.lock"))
            except (OSError, ValueError) as exc:
                return self._build_private(
                    key, spec, builder, f"Artifact lock unavailable: {exc}", BuildFailure.LOCK
                )
            lookup = self.lookup(key, spec)
            if lookup.status is LookupStatus.HIT:
                return ArtifactBuild(BuildDisposition.HIT, handle=lookup.handle)
            if lookup.status is not LookupStatus.MISS:
                return self._build_private(key, spec, builder, lookup.reason, _lookup_failure(lookup))
            built = self._build_private(key, spec, builder, None)
            return self._publish(key, spec, built)

    def _build_private(
        self,
        key: ArtifactKey,
        spec: ArtifactSpec,
        builder: Callable[[Path], T],
        reason: str | None,
        failure: BuildFailure | None = None,
    ) -> ArtifactBuild[T]:
        # tempfile.gettempdir() probes candidate directories by writing files.
        # A candidate could be inside the read-only cache, so let the adapter
        # explicitly select a private build root instead of probing implicitly.
        private_root = self.private_root
        if private_root is None:
            raise OSError(errno.ENOENT, "Artifact build requires a private_root outside the cache root")
        _check_private_path(private_root, self.root)
        _mkdir(private_root)
        directory = Path(tempfile.mkdtemp(prefix="pypto-build-", dir=private_root))
        # Do not catch compiler failures or delete paths a returned object owns.
        value = builder(directory)
        make_manifest(directory, key, spec)
        return ArtifactBuild(
            BuildDisposition.PRIVATE, private_directory=directory, value=value, reason=reason, failure=failure
        )

    def _publish(self, key: ArtifactKey, spec: ArtifactSpec, built: ArtifactBuild[T]) -> ArtifactBuild[T]:
        directory = built.private_directory
        if directory is None:
            raise ValueError("Artifact publication requires a completed private build")
        staging: Path | None = None
        destination = self._slot(key, spec)
        try:
            _mkdir(destination.parent)
            staging = Path(tempfile.mkdtemp(prefix=".tmp.", dir=destination.parent))
            manifest = make_manifest(directory, key, spec)
            _copy_payload(directory, staging, manifest)
            if make_manifest(staging, key, spec) != manifest:
                raise ValueError(f"Artifact changed while copying to publication staging: {directory}")
            _complete_staging(staging, manifest)
            _rename_noreplace(staging, destination)
            _sync_directory(destination.parent)
        except (OSError, ValueError) as exc:
            return ArtifactBuild(
                BuildDisposition.PRIVATE,
                private_directory=directory,
                value=built.value,
                reason=f"Artifact publication unavailable: {exc}",
                failure=BuildFailure.PUBLICATION,
            )
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
        return ArtifactBuild(
            BuildDisposition.PUBLISHED,
            ArtifactHandle(destination, key, spec, self.root),
            directory,
            built.value,
        )
