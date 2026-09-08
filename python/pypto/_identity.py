# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Content identity primitives for the planned persistent artifact store.

These helpers are deliberately not called by JIT dispatch. A toolchain adapter
must supply a complete dependency inventory before its component is usable;
hashing a compiler executable alone is not evidence of a complete toolchain.
No version string, Git revision, file timestamp, or build ID substitutes for
file contents. Loaded installations are immutable during a process's lifetime.
"""

import hashlib
import json
import os
import stat
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA = 1
_COMPONENTS = ("pypto", "runtime", "pto_isa", "ptoas", "device_toolchain")
_IGNORED_DIRECTORIES = frozenset({".git", "__pycache__"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def _typed_record(value: Any, ancestors: frozenset[int] = frozenset()) -> list[Any]:
    """Encode only protocol values, preserving types and IEEE-754 bits."""
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float64", struct.pack(">d", value).hex()]
    if type(value) is str:
        # JSON escapes a non-BMP character and its explicit surrogate pair
        # identically. Preserve Python code points before entering JSON.
        return ["str", value.encode("utf-8", errors="surrogatepass").hex()]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if id(value) in ancestors:
        raise ValueError("Identity records cannot contain cycles")
    nested = ancestors | {id(value)}
    if type(value) in (list, tuple):
        return [type(value).__name__, [_typed_record(item, nested) for item in value]]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Identity record dictionary keys must be strings")
        return [
            "dict",
            [[_typed_record(key, nested), _typed_record(value[key], nested)] for key in sorted(value)],
        ]
    raise TypeError(f"Unsupported identity record type: {type(value).__module__}.{type(value).__qualname__}")


def encode_record(value: Any) -> bytes:
    """Encode a versioned identity record without coercing unsupported objects.

    Accepted values are None, bool, int, float, str, bytes, lists, tuples, and
    dictionaries with string keys. Callers must explicitly normalize enums,
    paths, and effective configuration objects into these protocol types.
    """
    return json.dumps(
        ["pypto.identity", IDENTITY_SCHEMA, _typed_record(value)],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def digest_record(value: Any) -> str:
    """Return the full SHA-256 digest of a typed, versioned record."""
    return hashlib.sha256(encode_record(value)).hexdigest()


@dataclass(frozen=True)
class ContentRoot:
    """One ordered content input, with its path captured at construction.

    A file contributes its exact bytes. A directory contributes every regular
    file recursively, except Git metadata and generated Python bytecode.
    ``python_only`` filters directory entries to ``.py`` files for additional
    application source roots; it never filters a directly supplied file.
    Paths participate in identity because source locations and includes are
    not yet remapped to a relocatable namespace.
    """

    path: Path
    python_only: bool = False

    def __post_init__(self) -> None:
        # abspath/normpath would collapse symlink/.. before the filesystem can
        # resolve it, potentially selecting a different file.
        object.__setattr__(self, "path", Path.cwd() / self.path)


@dataclass(frozen=True)
class IdentityFailure:
    """Why an input cannot be assigned a reusable content identity."""

    component: str
    reason: str


@dataclass(frozen=True)
class ContentIdentity:
    """A complete content digest or an explicit failure, never an UNKNOWN key."""

    digest: str | None
    failure: str | None = None


def _file_digest(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Identity input is not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    # Metadata is a race detector, not an identity or a memoization key.
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Identity input changed while being read: {path}")
    current = path.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ):
        raise ValueError(f"Identity input was replaced while being read: {path}")
    return after.st_size, digest.hexdigest()


def _content_entries(
    path: Path,
    relative: str,
    python_only: bool,
    ancestors: frozenset[Path],
) -> list[tuple[Any, ...]]:
    resolved = path.resolve(strict=True)
    mode = resolved.stat().st_mode
    if stat.S_ISDIR(mode):
        if resolved in ancestors:
            raise ValueError(f"Identity input contains a directory symlink cycle: {path}")
        # scandir propagates unreadable-directory errors; glob may silently
        # omit them and give a smaller, apparently valid dependency set.
        with os.scandir(path) as scan:
            names = sorted(entry.name for entry in scan)
        entries: list[tuple[Any, ...]] = [] if python_only else [("directory", relative, str(resolved))]
        for name in names:
            if name in _IGNORED_DIRECTORIES:
                continue
            child = path / name
            child_relative = f"{relative}/{name}" if relative else name
            # is_dir() returns False for dangling links, which must not turn
            # an unavailable source subtree into an excluded non-Python file.
            child_mode = child.resolve(strict=True).stat().st_mode
            if not (stat.S_ISDIR(child_mode) or stat.S_ISREG(child_mode)):
                raise ValueError(f"Identity input is not a regular file or directory: {child}")
            if stat.S_ISREG(child_mode) and (
                child.suffix in _IGNORED_SUFFIXES or (python_only and child.suffix != ".py")
            ):
                continue
            entries.extend(_content_entries(child, child_relative, python_only, ancestors | {resolved}))
        with os.scandir(path) as scan:
            after = sorted(entry.name for entry in scan)
        if names != after or resolved != path.resolve(strict=True):
            raise ValueError(f"Identity directory changed while being read: {path}")
        return entries
    if not stat.S_ISREG(mode):
        raise ValueError(f"Identity input is not a regular file or directory: {path}")
    size, digest = _file_digest(path)
    if resolved != path.resolve(strict=True):
        raise ValueError(f"Identity symlink changed while being read: {path}")
    return [("file", relative, str(resolved), size, digest)]


def fingerprint_content(roots: tuple[ContentRoot, ...]) -> ContentIdentity:
    """Read every declared input and return its digest, or explain failure.

    Input order and root boundaries are preserved, including roots with equal
    basenames. Empty inventories are unavailable. This does not discover
    dependencies or certify that a caller's inventory is complete.
    """
    if not roots:
        return ContentIdentity(None, "No content inputs were supplied")
    records = []
    try:
        for root in roots:
            records.append(
                (
                    str(root.path),
                    root.python_only,
                    _content_entries(root.path, "", root.python_only, frozenset()),
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        return ContentIdentity(None, str(exc))
    return ContentIdentity(digest_record(("content", records)))


def fingerprint_extra_sources(
    paths: tuple[Path, ...], extra_fingerprint: str | None = None
) -> ContentIdentity:
    """Refresh application sources on every request, outside installation memoization."""
    roots = tuple(ContentRoot(path, python_only=True) for path in paths)
    content = fingerprint_content(roots) if roots else ContentIdentity(digest_record(("content", [])))
    if content.digest is None:
        return content
    return ContentIdentity(digest_record(("extra_sources", content.digest, extra_fingerprint)))


@dataclass(frozen=True)
class ComponentInputs:
    """Inventory supplied by a dependency-aware compiler/toolchain adapter.

    An adapter must leave ``unavailable_reason`` set until it has accounted
    for all resources and dynamic dependencies, even if it knows some files.
    A caller-supplied application fingerprint cannot complete this inventory.
    """

    roots: tuple[ContentRoot, ...] = ()
    unavailable_reason: str | None = "Dependency inventory has not been established"


@dataclass(frozen=True)
class ToolchainInputs:
    """Required component inventories, selected by the actual compilation path."""

    pypto: ComponentInputs
    runtime: ComponentInputs
    pto_isa: ComponentInputs
    ptoas: ComponentInputs
    device_toolchain: ComponentInputs


@dataclass(frozen=True)
class ToolchainIdentity:
    """Content identities and actionable reasons for every unavailable component."""

    pypto: str | None
    runtime: str | None
    pto_isa: str | None
    ptoas: str | None
    device_toolchain: str | None
    failures: tuple[IdentityFailure, ...] = ()
    schema: int = IDENTITY_SCHEMA

    @property
    def usable(self) -> bool:
        """Whether all required inventories could be read completely."""
        return not self.failures and all(getattr(self, name) is not None for name in _COMPONENTS)

    @property
    def digest(self) -> str | None:
        """Return a combined identity only when all components are available."""
        if not self.usable:
            return None
        return digest_record(("toolchain", self.schema, {name: getattr(self, name) for name in _COMPONENTS}))


class InstallationIdentityCache:
    """Memoize successful reads of a process's immutable installation inputs.

    The complete resolved inventory selects the entry; there is no unkeyed
    singleton identity. Adapters must resolve tools again when their selection
    configuration changes. Replacing installed code or libraries at the same
    paths requires a process restart. Per-request application sources must use
    ``fingerprint_extra_sources`` instead of this cache.
    """

    def __init__(self) -> None:
        self._components: dict[ComponentInputs, str] = {}
        self._lock = threading.Lock()

    def capture(self, inputs: ToolchainInputs) -> ToolchainIdentity:
        """Hash complete component inventories, preserving every failure reason."""
        digests: dict[str, str | None] = {}
        failures = []
        with self._lock:
            for name in _COMPONENTS:
                component: ComponentInputs = getattr(inputs, name)
                if component.unavailable_reason is not None:
                    result = ContentIdentity(None, component.unavailable_reason)
                elif component in self._components:
                    result = ContentIdentity(self._components[component])
                else:
                    result = fingerprint_content(component.roots)
                    if result.digest is not None:
                        self._components[component] = result.digest
                digests[name] = result.digest
                if result.digest is None:
                    failures.append(IdentityFailure(name, result.failure or "Content identity unavailable"))
        return ToolchainIdentity(
            pypto=digests["pypto"],
            runtime=digests["runtime"],
            pto_isa=digests["pto_isa"],
            ptoas=digests["ptoas"],
            device_toolchain=digests["device_toolchain"],
            failures=tuple(failures),
        )
