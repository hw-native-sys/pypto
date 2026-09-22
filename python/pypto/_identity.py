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
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

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


# Sections a debugger reads and an execution never does. .dynsym and .dynstr
# drive dynamic linking and are deliberately absent: they decide what a library
# resolves at load time, so they are part of what it does.
_DEBUG_SECTIONS = (".debug", ".zdebug", ".comment", ".gnu_debuglink")
_DEBUGGER_SECTIONS = (*_DEBUG_SECTIONS, ".symtab", ".strtab")
_ET_REL = 1
_ELF_MAGIC = b"\x7fELF"
_SHT_NOBITS = 8


def _elf_sections(stream: BinaryIO) -> tuple[int, list[tuple[str, int, int]]] | None:
    """Return the ELF64 type and every section holding file bytes.

    None whenever the header cannot be read the way this expects -- a different
    class, a truncated table, an unreadable string section, or a declared size
    the file cannot hold. The caller then reads the whole file, so a file this
    does not understand is covered exactly as it was before.

    Sizes come from the file's own header, so every read is clamped to the
    file's real length first: a corrupt or truncated header can declare a
    section table of gigabytes, and attempting that allocation would raise out
    of here instead of falling back the way this promises.
    """
    length = os.fstat(stream.fileno()).st_size
    stream.seek(0)
    header = stream.read(64)
    if len(header) < 64 or header[:4] != _ELF_MAGIC or header[4] != 2:
        return None
    endian = "<" if header[5] == 1 else ">"
    (kind,) = struct.unpack_from(endian + "H", header, 16)
    (offset,) = struct.unpack_from(endian + "Q", header, 40)
    entry_size, count, names_index = struct.unpack_from(endian + "HHH", header, 58)
    if count == 0 or offset == 0 or entry_size < 64 or names_index >= count:
        return None
    stream.seek(offset)
    # entry_size * count cannot exceed ~4 GiB, which read absorbs by returning
    # short; the 64-bit sizes below cannot, and are bounded before they are used.
    table = stream.read(entry_size * count)
    if len(table) != entry_size * count:
        return None

    def field(index: int, at: int, code: str) -> int:
        return struct.unpack_from(endian + code, table, index * entry_size + at)[0]

    names_offset, names_size = field(names_index, 24, "Q"), field(names_index, 32, "Q")
    if names_offset > length or names_size > length - names_offset:
        return None
    stream.seek(names_offset)
    names = stream.read(names_size)
    sections = []
    for index in range(count):
        start = field(index, 0, "I")
        end = names.find(b"\0", start)
        if start >= len(names) or end < 0:
            return None
        if field(index, 4, "I") == _SHT_NOBITS:
            continue
        section_offset, section_size = field(index, 24, "Q"), field(index, 32, "Q")
        if section_offset > length or section_size > length - section_offset:
            return None
        sections.append((names[start:end].decode("ascii", "replace"), section_offset, section_size))
    return kind, sections


def _executable_digest(stream: BinaryIO) -> tuple[int, str] | None:
    """Digest what an ELF file does, skipping what only a debugger reads.

    Debug information cannot change the artifact a compiler produces from this
    installation, but it dominates an unstripped build: on the development tree
    measured here it is 69% of every byte the inventory reads, and 367 MB of a
    single 376 MB extension module. Reading it makes a rebuild that changed
    only source paths or -g level invalidate every cached artifact.

    The section name and size enter the digest beside the bytes, so removing a
    section, renaming it, or moving its bytes into another cannot leave the
    result unchanged. None whenever the sections cannot be read.
    """
    parsed = _elf_sections(stream)
    if parsed is None:
        return None
    kind, found = parsed
    # In a relocatable object the symbol table is what a linker resolves against,
    # not something only a debugger reads: two objects with identical code and
    # relocations but a rebuilt .symtab link differently. Only a finished
    # executable or shared object can spare it.
    skipped = _DEBUGGER_SECTIONS if kind != _ET_REL else _DEBUG_SECTIONS
    digest = hashlib.sha256()
    covered = 0
    for name, offset, size in sorted(found):
        if name.startswith(skipped):
            continue
        covered += size
        digest.update(f"{name}\0{size}\0".encode())
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("Identity input ended inside a declared ELF section")
            digest.update(chunk)
            remaining -= len(chunk)
    # The size recorded beside the digest has to be the size of what was
    # digested. Reporting the whole file would put the debug sections back into
    # the identity through the record, undoing what skipping them achieved.
    #
    # Tagged, so a section digest can never be read as the whole-file digest of
    # some other file that happens to hash to the same value.
    return covered, f"elf64-sections:{digest.hexdigest()}"


def _file_digest(path: Path, *, sections: bool = False) -> tuple[int, str]:
    """Digest a file's bytes, and detect a change or replacement during the read.

    ``sections`` selects the identity reading, which skips the parts of an ELF
    file only a debugger reads; the artifact manifest leaves it off, because it
    is verifying that stored bytes are intact rather than asking what an
    installation would compile with, and its record is a plain SHA-256 that
    other code recomputes over the whole file.
    """
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Identity input is not a regular file: {path}")
        executable = _executable_digest(stream) if sections else None
        recorded = None if executable is None else executable[1]
        if recorded is None:
            digest = hashlib.sha256()
            stream.seek(0)
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            recorded = digest.hexdigest()
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
    return (after.st_size if executable is None else executable[0]), recorded


def _content_entries(
    path: Path,
    relative: str,
    python_only: bool,
    ancestors: frozenset[Path],
    resolved: Path | None = None,
    via_symlink: bool = True,
) -> list[tuple[Any, ...]]:
    # An entry that is not itself a symlink inherits its parent's resolution,
    # so only a symlinked entry needs a full readlink walk of every component.
    # ``via_symlink`` records which case produced ``resolved``, selecting the
    # cheapest post-read check that still detects replacement of this entry.
    if resolved is None:
        resolved = path.resolve(strict=True)
    mode = resolved.stat().st_mode
    if stat.S_ISDIR(mode):
        if resolved in ancestors:
            raise ValueError(f"Identity input contains a directory symlink cycle: {path}")
        # scandir propagates unreadable-directory errors; glob may silently
        # omit them and give a smaller, apparently valid dependency set.
        with os.scandir(path) as scan:
            children = sorted(scan, key=lambda entry: entry.name)
        names = [entry.name for entry in children]
        entries: list[tuple[Any, ...]] = [] if python_only else [("directory", relative, str(resolved))]
        for entry in children:
            name = entry.name
            if name in _IGNORED_DIRECTORIES:
                continue
            child = path / name
            child_relative = f"{relative}/{name}" if relative else name
            # is_dir() returns False for dangling links, which must not turn
            # an unavailable source subtree into an excluded non-Python file.
            # resolve(strict=True) still raises for a dangling symlink here.
            if entry.is_symlink():
                child_resolved = child.resolve(strict=True)
                child_mode = child_resolved.stat().st_mode
            else:
                child_resolved = resolved / name
                child_mode = entry.stat(follow_symlinks=False).st_mode
            if not (stat.S_ISDIR(child_mode) or stat.S_ISREG(child_mode)):
                raise ValueError(f"Identity input is not a regular file or directory: {child}")
            if stat.S_ISREG(child_mode) and (
                child.suffix in _IGNORED_SUFFIXES or (python_only and child.suffix != ".py")
            ):
                continue
            entries.extend(
                _content_entries(
                    child,
                    child_relative,
                    python_only,
                    ancestors | {resolved},
                    child_resolved,
                    entry.is_symlink(),
                )
            )
        with os.scandir(path) as scan:
            after = sorted(entry.name for entry in scan)
        if names != after or resolved != path.resolve(strict=True):
            raise ValueError(f"Identity directory changed while being read: {path}")
        return entries
    if not stat.S_ISREG(mode):
        raise ValueError(f"Identity input is not a regular file or directory: {path}")
    size, digest = _file_digest(path, sections=True)
    # An inherited resolution only has to prove that this entry did not become
    # a symlink while its bytes were read: replacement of an ancestor component
    # is caught by that directory's own post-read check. _file_digest already
    # rejects a same-path regular file swapped in during the read.
    if via_symlink:
        replaced = resolved != path.resolve(strict=True)
    else:
        replaced = stat.S_ISLNK(os.lstat(path).st_mode)
    if replaced:
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


@lru_cache(maxsize=128)
def _empty_extra_sources(extra_fingerprint: str | None) -> ContentIdentity:
    return ContentIdentity(
        digest_record(("extra_sources", digest_record(("content", [])), extra_fingerprint))
    )


def fingerprint_extra_sources(
    paths: tuple[Path, ...], extra_fingerprint: str | None = None
) -> ContentIdentity:
    """Refresh application sources on every request, outside installation memoization."""
    if not paths:
        return _empty_extra_sources(extra_fingerprint)
    roots = tuple(ContentRoot(path, python_only=True) for path in paths)
    content = fingerprint_content(roots)
    if content.digest is None:
        return content
    return ContentIdentity(digest_record(("extra_sources", content.digest, extra_fingerprint)))


@dataclass(frozen=True)
class ComponentInputs:
    """Inventory supplied by a dependency-aware compiler/toolchain adapter.

    An adapter must leave ``unavailable_reason`` set until it has accounted
    for all resources and dynamic dependencies, even if it knows some files.
    A caller-supplied application fingerprint cannot complete this inventory.

    ``verified_revision`` is the narrow exception to reading contents: an
    adapter may supply a revision when some *other* mechanism has already
    proven, in the same resolution, that the component's bytes are exactly that
    revision -- not merely that it claims to be. A version an installation
    reports about itself is not such a proof. The adapter owns that proof and
    must document what it does not cover; without one, leave this unset so the
    contents are read.

    ``reported_version`` is weaker on purpose and is named for what it is: a
    version the installation states about *itself*, with nothing verifying it.
    It distinguishes installations that say they differ; it cannot detect bytes
    that changed while the version stayed put, so a rebuild or a patch applied
    in place is invisible to it. Use it only where the deployment establishes
    that the component arrives as an unmodified published build, and record
    that reasoning where the adapter sets it. It is not interchangeable with
    ``verified_revision`` and must not be treated as precedent for another
    component.

    ``reported_version`` may accompany ``roots``. A component whose files come
    from more than one source -- a vendor package that states its own build
    identity, alongside host files that state nothing -- covers each part with
    the evidence that part actually has, and the identity records both. Do not
    use this to let a version stand in for files the vendor does not publish.
    """

    roots: tuple[ContentRoot, ...] = ()
    unavailable_reason: str | None = "Dependency inventory has not been established"
    verified_revision: str | None = None
    reported_version: str | None = None


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

    def _evidence(self, name: str, component: ComponentInputs) -> ContentIdentity:
        """Record every kind of evidence a component carries, keyed by its name.

        Each kind keeps its own tag, so a self-reported version can never
        produce the digest a verified revision would, and neither can collide
        with a content digest or with another component's. Declared roots are
        always read: a version never stands in for files the component lists.
        Caller holds ``self._lock``.
        """
        if component.unavailable_reason is not None:
            return ContentIdentity(None, component.unavailable_reason)
        evidence: dict[str, Any] = {}
        if component.verified_revision is not None:
            evidence["verified_revision"] = component.verified_revision
        if component.reported_version is not None:
            evidence["reported_version"] = component.reported_version
        if component.roots or not evidence:
            # No evidence at all still reads the (empty) inventory, so an
            # adapter that supplies nothing stays unavailable rather than
            # acquiring an identity by omission.
            if component in self._components:
                content = ContentIdentity(self._components[component])
            else:
                content = fingerprint_content(component.roots)
                if content.digest is not None:
                    self._components[component] = content.digest
            if content.digest is None:
                return content
            evidence["content"] = content.digest
        return ContentIdentity(digest_record(("component", name, evidence)))

    def capture(self, inputs: ToolchainInputs) -> ToolchainIdentity:
        """Hash complete component inventories, preserving every failure reason."""
        digests: dict[str, str | None] = {}
        failures = []
        with self._lock:
            for name in _COMPONENTS:
                component: ComponentInputs = getattr(inputs, name)
                result = self._evidence(name, component)
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
