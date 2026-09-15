# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compatibility stamps for generated kernel and orchestration binaries."""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BINARY_CONTEXT_SCHEMA = 2
_BINARY_CONTEXT_FILENAME = "binary_context.json"


@dataclass(frozen=True)
class BinaryCacheContext:
    """Runtime and PTO-ISA inputs checked before reusing generated binaries."""

    platform: str
    runtime_name: str
    runtime_revision: str
    pto_isa_revision: str
    schema: int = _BINARY_CONTEXT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ABI context stored in a binary-cache stamp."""
        return asdict(self)


def binary_context_path(work_dir: Path | str) -> Path:
    """Return the compatibility-stamp path for one complete chip sub-build."""
    return Path(work_dir) / "cache" / _BINARY_CONTEXT_FILENAME


@contextlib.contextmanager
def binary_context_lock(work_dir: Path | str) -> Iterator[None]:
    """Serialize cache validation, compilation, and stamping for *work_dir*.

    The lock file is intentionally persistent: removing it while another
    process is waiting would let callers lock different inodes and enter the
    same cache transaction concurrently.
    """
    lock_path = Path(work_dir) / "cache" / ".binary_context.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _binary_artifacts(work_dir: Path | str) -> list[Path]:
    """List reusable binaries without following paths supplied by a stamp."""
    work_dir = Path(work_dir)
    candidates: list[Path] = []
    cache_dir = work_dir / "cache"
    if cache_dir.is_dir():
        candidates.extend(cache_dir.glob("*.bin"))
    for subdir in ("kernels", "orchestration"):
        root = work_dir / subdir
        if not root.is_dir():
            continue
        for extension in ("*.so", "*.o"):
            candidates.extend(root.rglob(extension))

    return candidates


def _invalidate_binary_artifacts(work_dir: Path | str) -> int:
    """Delete reusable binaries for one chip sub-build, preserving all sources."""
    removed = 0
    for path in _binary_artifacts(work_dir):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
    return removed


def invalidate_binary_context(work_dir: Path | str) -> int:
    """Delete reusable binaries and discard the context that authorized them.

    Returns the number of binary files removed.  The compatibility stamp is
    deliberately not included in that count because it is metadata, not a
    generated binary.
    """
    with contextlib.suppress(FileNotFoundError):
        binary_context_path(work_dir).unlink()
    return _invalidate_binary_artifacts(work_dir)


def _read_binary_context(path: Path) -> dict[str, Any] | None:
    """Read a JSON stamp, treating unreadable or malformed metadata as a cache miss."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def prepare_binary_context(work_dir: Path | str, context: BinaryCacheContext | None) -> int:
    """Begin a cache transaction, invalidating binaries with a mismatched context.

    ``None`` means that the current runtime identity cannot be established. In
    that case cached binaries are never trusted.  A matching stamp preserves
    the binaries for this transaction, but the stamp itself is always discarded
    before compilation starts.  A caller must write a fresh stamp only after all
    binaries assemble successfully so an interrupted transaction fails closed.

    Returns the number of binary files removed.
    """
    stamp_path = binary_context_path(work_dir)
    cached = _read_binary_context(stamp_path)
    if (
        context is not None
        and cached is not None
        and cached.get("context") == context.to_dict()
        and isinstance(cached.get("files"), dict)
    ):
        with contextlib.suppress(FileNotFoundError):
            stamp_path.unlink()
        # Missing files are normal partial hits. Reject altered or unrecorded
        # files independently so the remaining verified binaries can be reused.
        removed = 0
        for path in _binary_artifacts(work_dir):
            expected = cached["files"].get(path.relative_to(work_dir).as_posix())
            try:
                valid = not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
            except OSError:
                valid = False
            if not valid:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    return invalidate_binary_context(work_dir)


def record_binary_context(work_dir: Path | str, context: BinaryCacheContext | None) -> None:
    """Atomically record *context* after all binaries assemble successfully."""
    if context is None:
        return

    payload = {
        "context": context.to_dict(),
        "files": {
            path.relative_to(work_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in _binary_artifacts(work_dir)
            if not path.is_symlink()
        },
    }
    path = binary_context_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        file = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


__all__ = [
    "BinaryCacheContext",
    "binary_context_lock",
    "binary_context_path",
    "invalidate_binary_context",
    "prepare_binary_context",
    "record_binary_context",
]
