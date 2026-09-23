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
from bisect import bisect_right
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

    A file contributes its bytes except verified ELF debug payloads.
    A directory contributes every regular file recursively, except Git metadata
    and generated Python bytecode.
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


# Only non-allocated debug payloads may be omitted. All ELF metadata, symbol
# tables, segment contents, and bytes outside sections remain identity inputs.
_ELF_MAGIC = b"\x7fELF"
_SHT_NOBITS = 8


def _elf_debug_ranges(stream: BinaryIO) -> list[tuple[int, int]] | None:
    """Find safely skippable ELF64 debug payloads, or request whole-file hashing.

    Support ordinary ELF64 relocatables, executables, and shared objects in
    either byte order. Extended numbering, unknown layouts, overlapping
    sections, and debug payloads referenced by program headers fall back.
    Reads are bounded independently of the file's declared or sparse size.
    """
    length = os.fstat(stream.fileno()).st_size
    stream.seek(0)
    header = stream.read(64)
    if (
        len(header) != 64
        or header[:4] != _ELF_MAGIC
        or header[4] != 2
        or header[5] not in (1, 2)
        or header[6] != 1
    ):
        return None
    endian = "<" if header[5] == 1 else ">"
    kind, _, version, _, phoff, shoff, _, ehsize, phsize, phnum, shsize, shnum, names_index = (
        struct.unpack_from(endian + "HHIQQQIHHHHHH", header, 16)
    )
    if (
        kind not in (1, 2, 3)
        or version != 1
        or ehsize != 64
        or shsize != 64
        or not 0 < names_index < shnum < 0xFF00
        or shoff < 64
        or phnum == 0xFFFF
        or (phnum and (phsize != 56 or phoff < 64))
        or (not phnum and phoff != 0)
    ):
        return None

    def in_file(offset: int, size: int) -> bool:
        """Check a declared byte range without reading or allocating its contents."""
        return offset <= length and size <= length - offset

    if not in_file(shoff, shsize * shnum) or not in_file(phoff, phsize * phnum):
        return None
    stream.seek(shoff)
    table = stream.read(shsize * shnum)  # At most 0xFEFF fixed-size entries.
    if len(table) != shsize * shnum or table[:64] != bytes(64):
        return None
    sections = list(struct.iter_unpack(endian + "IIQQQQIIQQ", table))
    _, names_type, _, _, names_offset, names_size, _, _, _, _ = sections[names_index]
    if names_type != 3 or not names_size or not in_file(names_offset, names_size):
        return None

    occupied = [(0, 64), (shoff, shoff + len(table))]
    if phnum:
        occupied.append((phoff, phoff + phsize * phnum))
    skipped = []
    for index, (name, section_type, flags, _, offset, size, _, _, _, _) in enumerate(sections[1:], 1):
        if name >= names_size:
            return None
        stream.seek(names_offset + name)
        # ELF imposes no short name limit. Unusually long names are supported
        # by whole-file hashing instead of allocating an unbounded string table.
        raw_name = stream.read(min(256, names_size - name))
        end = raw_name.find(b"\0")
        if end < 0:
            return None
        section_name = raw_name[:end]
        if section_type in (0, _SHT_NOBITS):
            continue  # Their full headers are still hashed, including BSS size.
        if not in_file(offset, size):
            return None
        if size:
            occupied.append((offset, offset + size))
        debug = section_name in (b".debug", b".zdebug", b".comment", b".gnu_debuglink") or (
            section_name.startswith((b".debug_", b".zdebug_"))
        )
        # SHF_COMPRESSED is safe; any other flag or non-PROGBITS type may carry
        # semantics beyond debug information. Never omit the section-name table.
        if debug and section_type == 1 and flags & ~0x800 == 0 and index != names_index and size:
            skipped.append((offset, offset + size))
    occupied.sort()
    if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
        return None

    skipped.sort()
    if not _elf_segments_avoid_ranges(stream, endian, (phoff, phnum), skipped, length):
        return None
    return skipped


def _elf_segments_avoid_ranges(
    stream: BinaryIO,
    endian: str,
    headers: tuple[int, int],
    ranges: list[tuple[int, int]],
    length: int,
) -> bool:
    """Check segment bounds and reject references to sorted debug ranges."""
    phoff, phnum = headers
    ends = [end for _, end in ranges]
    for index in range(phnum):
        stream.seek(phoff + index * 56)
        entry = stream.read(56)
        if len(entry) != 56:
            return False
        _, _, offset, _, _, size, _, _ = struct.unpack(endian + "IIQQQQQQ", entry)
        if offset > length or size > length - offset:
            return False
        # The first debug range ending after this segment starts is the only
        # candidate needed. Avoid a quadratic scan of two file-declared tables.
        candidate = bisect_right(ends, offset)
        if size and candidate < len(ranges) and ranges[candidate][0] < offset + size:
            return False
    return True


def _executable_digest(stream: BinaryIO) -> tuple[int, str] | None:
    """Hash all ELF bytes except verified debug payloads, retaining all headers.

    In particular, section attributes and SHT_NOBITS sizes, program headers,
    and segment bytes outside sections must affect identity. Keeping offsets
    is conservative: debug rebuilds that change layout can invalidate the
    cache, while same-layout debug edits still avoid hashing their payloads.
    """
    skipped = _elf_debug_ranges(stream)
    if skipped is None:
        return None
    length = os.fstat(stream.fileno()).st_size
    digest = hashlib.sha256()
    covered = 0
    offset = 0
    for start, end in [*skipped, (length, length)]:
        size = start - offset
        if size < 0:
            raise ValueError(
                f"Identity input changed while being read: ELF range end {start} precedes offset {offset}"
            )
        digest.update(struct.pack(">QQ", offset, size))
        covered += size
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("Identity input ended inside a declared ELF range")
            digest.update(chunk)
            remaining -= len(chunk)
        offset = end
    # Version the tag: the old section-only digest omitted semantic metadata.
    return covered, f"elf64-debug-filtered-v2:{digest.hexdigest()}"


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
    """Inventory one input recursively, retaining path semantics and detecting replacement."""
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
