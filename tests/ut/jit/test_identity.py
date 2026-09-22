# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Protocol and content identity tests; no toolchain or device is required."""

import hashlib
import io
import os
import shutil
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from enum import IntEnum
from pathlib import Path

import pytest
from pypto import _identity
from pypto._identity import (
    ComponentInputs,
    ContentRoot,
    InstallationIdentityCache,
    ToolchainInputs,
    digest_record,
    encode_record,
    fingerprint_content,
    fingerprint_extra_sources,
)


def test_protocol_is_typed_and_preserves_float_bits():
    values = [None, False, True, 0, 1, 0.0, -0.0, 1.0, "1", b"1", [1], (1,), {"x": 1}]
    assert len({digest_record(value) for value in values}) == len(values)
    nans = [struct.unpack(">d", bytes.fromhex(bits))[0] for bits in ("7ff8000000000001", "7ff8000000000002")]
    assert digest_record(nans[0]) != digest_record(nans[1])
    assert digest_record(nans[0]) == digest_record(nans[0])
    assert digest_record(float("inf")) != digest_record(float("-inf"))


def test_record_order_boundaries_and_schema(monkeypatch):
    assert encode_record({"b": [2], "a": 1}) == encode_record({"a": 1, "b": [2]})
    assert digest_record(["ab", "c"]) != digest_record(["a", "bc"])
    assert digest_record([1, 2]) != digest_record([2, 1])
    before = digest_record({"a": 1})
    assert len(before) == 64
    monkeypatch.setattr(_identity, "IDENTITY_SCHEMA", _identity.IDENTITY_SCHEMA + 1)
    assert digest_record({"a": 1}) != before


def test_records_are_stable_across_process_hash_seeds():
    code = (
        "from pypto._identity import digest_record; "
        'print(digest_record({key: key for key in {"alpha", "beta", "gamma"}}))'
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        for seed in ("1", "42")
    ]
    assert outputs[0] == outputs[1] == digest_record({key: key for key in ("alpha", "beta", "gamma")})


def test_unicode_code_points_do_not_collide_with_explicit_surrogates():
    character = "\U0001f600"
    surrogates = "\ud83d\ude00"
    assert character != surrogates
    assert encode_record(character) != encode_record(surrogates)
    assert digest_record({character: 1}) != digest_record({surrogates: 1})
    assert digest_record({character: 1, surrogates: 2}) == digest_record({surrogates: 2, character: 1})


@pytest.mark.parametrize("value", [object(), Path("input"), {1: "value"}, {1, 2}])
def test_unsupported_values_are_not_stringified(value):
    with pytest.raises(TypeError, match="identity record|Identity record"):
        encode_record(value)


def test_int_enum_is_not_silently_encoded_as_an_integer():
    class Mode(IntEnum):
        FAST = 1

    with pytest.raises(TypeError, match="Unsupported identity record type"):
        digest_record(Mode.FAST)


def test_cycles_are_rejected_but_shared_subrecords_are_allowed():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cycles"):
        encode_record(value)
    shared = [1]
    assert encode_record([shared, shared]) == encode_record([[1], [1]])


def test_content_changes_with_unchanged_size_timestamp_and_build_id(tmp_path):
    library = tmp_path / "compiler.so"
    library.write_bytes(b"build-id:fixed;code:AAAA")
    roots = (ContentRoot(library),)
    before = fingerprint_content(roots)
    metadata = library.stat()
    library.write_bytes(b"build-id:fixed;code:BBBB")
    os.utime(library, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    after = fingerprint_content(roots)
    assert before.digest is not None
    assert after.digest is not None
    assert before.digest != after.digest
    assert fingerprint_content(roots) == after


def test_timestamp_only_change_keeps_content_identity(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("rows = 32\n")
    roots = (ContentRoot(source),)
    before = fingerprint_content(roots)
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    assert fingerprint_content(roots) == before


def test_tree_tracks_resources_and_ignores_only_declared_metadata(tmp_path):
    source = tmp_path / "compiler.py"
    source.write_text("rows = 32\n")
    resource = tmp_path / "resource.json"
    resource.write_text('{"option": 1}')
    roots = (ContentRoot(tmp_path),)
    before = fingerprint_content(roots)
    for directory in (".git", "__pycache__"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "ignored").write_bytes(b"metadata")
    (tmp_path / "compiler.pyc").write_bytes(b"bytecode")
    assert fingerprint_content(roots) == before
    resource.write_text('{"option": 2}')
    assert fingerprint_content(roots).digest != before.digest


def test_extra_directories_refresh_python_sources_and_preserve_roots(tmp_path):
    roots = tuple(tmp_path / name for name in ("one", "two"))
    for root in roots:
        root.mkdir()
        (root / "kernel.py").write_text("rows = 32\n")
    before = fingerprint_extra_sources(roots)
    (roots[1] / "kernel.py").write_text("rows = 64\n")
    after = fingerprint_extra_sources(roots)
    assert after.digest != before.digest
    assert after.digest != fingerprint_extra_sources(tuple(reversed(roots))).digest
    (roots[1] / "notes.txt").write_text("not an additional Python dependency")
    (roots[1] / "empty-doc-directory").mkdir()
    assert fingerprint_extra_sources(roots) == after
    assert fingerprint_extra_sources((roots[1] / "notes.txt",)).digest is not None
    assert fingerprint_extra_sources(roots, "config-a") != fingerprint_extra_sources(roots, "config-b")


def test_paths_remain_semantic_until_codegen_has_stable_path_mapping(tmp_path):
    first, second = tmp_path / "one.py", tmp_path / "two.py"
    first.write_text("rows = 32\n")
    second.write_bytes(first.read_bytes())
    assert fingerprint_extra_sources((first,)).digest != fingerprint_extra_sources((second,)).digest


def test_relative_roots_capture_the_callers_working_directory(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "kernel.py").write_text("rows = 32\n")
    monkeypatch.chdir(first)
    root = ContentRoot(Path("kernel.py"))
    before = fingerprint_content((root,))
    monkeypatch.chdir(second)
    assert root.path == first / "kernel.py"
    assert fingerprint_content((root,)) == before


@pytest.mark.parametrize("relative", [False, True])
def test_parent_component_after_symlink_preserves_filesystem_meaning(tmp_path, monkeypatch, relative):
    actual = tmp_path / "actual"
    (actual / "child").mkdir(parents=True)
    (tmp_path / "link").symlink_to(actual / "child", target_is_directory=True)
    source = actual / "kernel.py"
    source.write_text("rows = 32\n")
    decoy = tmp_path / "kernel.py"
    decoy.write_text("rows = 99\n")
    monkeypatch.chdir(tmp_path)
    supplied = Path("link/../kernel.py")
    root = ContentRoot(supplied if relative else tmp_path / supplied)
    assert root.path.read_bytes() == source.read_bytes()
    before = fingerprint_content((root,))
    assert before.digest is not None
    decoy.write_text("rows = 88\n")
    assert fingerprint_content((root,)) == before
    source.write_text("rows = 64\n")
    assert fingerprint_content((root,)).digest != before.digest


def test_missing_inputs_do_not_shrink_an_inventory(tmp_path):
    present = tmp_path / "present.py"
    missing = tmp_path / "missing.py"
    present.write_text("rows = 32\n")
    identity = fingerprint_extra_sources((present, missing), "cannot-replace-missing-evidence")
    assert identity.digest is None
    assert identity.failure is not None and "missing.py" in identity.failure
    assert fingerprint_content(()).digest is None
    assert fingerprint_extra_sources(()).digest is not None


def test_unreadable_directory_is_not_silently_omitted(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("rows = 32\n")

    def denied(_path):
        raise PermissionError("source directory is not readable")

    monkeypatch.setattr(os, "scandir", denied)
    identity = fingerprint_content((ContentRoot(tmp_path),))
    assert identity.digest is None
    assert identity.failure is not None and "not readable" in identity.failure


def test_file_size_change_during_hashing_is_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "compiler.bin"
    source.write_bytes(b"original")
    original_sha256 = _identity.hashlib.sha256

    class EditingDigest:
        def __init__(self, data=b""):
            self.digest = original_sha256(data)

        def update(self, chunk):
            self.digest.update(chunk)
            source.write_bytes(b"modified-and-longer")

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(_identity.hashlib, "sha256", EditingDigest)
    identity = fingerprint_content((ContentRoot(source),))
    assert identity.digest is None
    assert identity.failure is not None and "changed while being read" in identity.failure


def test_symlinks_track_target_contents_and_reject_cycles(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    source = target / "kernel.py"
    source.write_text("rows = 32\n")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    roots = (ContentRoot(link),)
    before = fingerprint_content(roots)
    source.write_text("rows = 64\n")
    assert fingerprint_content(roots).digest != before.digest
    (target / "cycle").symlink_to(target, target_is_directory=True)
    identity = fingerprint_content(roots)
    assert identity.digest is None
    assert identity.failure is not None and "cycle" in identity.failure


@pytest.mark.parametrize("name", ["plugins", "plugin.py", "plugin.data"])
def test_extra_source_filter_cannot_hide_broken_symlinks(tmp_path, name):
    (tmp_path / name).symlink_to(tmp_path / "missing-directory", target_is_directory=True)
    identity = fingerprint_extra_sources((tmp_path,))
    assert identity.digest is None
    assert identity.failure is not None and "missing-directory" in identity.failure


def test_extra_source_filter_cannot_hide_unreadable_subtrees(tmp_path, monkeypatch):
    directory = tmp_path / "plugins"
    directory.mkdir()
    original_stat = Path.stat

    def stat_with_unreadable_directory(path, *args, **kwargs):
        if path == directory:
            raise PermissionError("Cannot inspect plugins")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_unreadable_directory)
    identity = fingerprint_extra_sources((tmp_path,))
    assert identity.digest is None
    assert identity.failure is not None and "Cannot inspect plugins" in identity.failure


def test_nested_entries_record_their_true_resolved_paths(tmp_path):
    # Entries inherit their parent's resolution instead of resolving every
    # component again; the recorded path must still be the real path.
    actual = tmp_path / "actual"
    (actual / "nested").mkdir(parents=True)
    (actual / "nested/kernel.py").write_text("rows = 32\n")
    (actual / "plain.py").write_text("rows = 16\n")
    (actual / "linked.py").symlink_to(actual / "nested/kernel.py")
    root = tmp_path / "link"
    root.symlink_to(actual, target_is_directory=True)

    entries = _identity._content_entries(root, "", False, frozenset())
    recorded = {entry[1]: entry[2] for entry in entries}
    assert recorded
    for relative, resolved in recorded.items():
        assert resolved == os.path.realpath(root / relative if relative else root)
    assert recorded["linked.py"] == str((actual / "nested/kernel.py").resolve())


def _elf64(
    path: Path, sections: dict[str, bytes], *, endian: str = "<", program_headers: bytes = b""
) -> Path:
    """Write a minimal ELF64 whose named sections carry the given bytes."""
    names = bytearray(b"\0")
    name_offset = {}
    for name in (*sections, ".shstrtab"):
        name_offset[name] = len(names)
        names += name.encode() + b"\0"

    body = bytearray(program_headers)
    placed = []
    for name, payload in sections.items():
        placed.append((name, 64 + len(body), len(payload)))
        body += payload
    shstrtab_offset = 64 + len(body)
    body += names
    table_offset = 64 + len(body)

    header = bytearray(64)
    header[0:7] = b"\x7fELF\x02\x01\x01"
    header[5] = 1 if endian == "<" else 2
    struct.pack_into(endian + "HHI", header, 16, 3, 62, 1)  # ET_DYN, EM_X86_64, EV_CURRENT
    struct.pack_into(endian + "H", header, 52, 64)
    if program_headers:
        struct.pack_into(endian + "Q", header, 32, 64)
        struct.pack_into(endian + "HH", header, 54, 56, len(program_headers) // 56)
    struct.pack_into(endian + "Q", header, 40, table_offset)
    struct.pack_into(endian + "HHH", header, 58, 64, len(placed) + 2, len(placed) + 1)

    entries = bytearray(64)  # index 0 is SHT_NULL
    for name, offset, size in placed:
        entry = bytearray(64)
        struct.pack_into(endian + "I", entry, 0, name_offset[name])
        struct.pack_into(endian + "I", entry, 4, 1)  # SHT_PROGBITS
        struct.pack_into(endian + "Q", entry, 24, offset)
        struct.pack_into(endian + "Q", entry, 32, size)
        entries += entry
    entry = bytearray(64)
    struct.pack_into(endian + "I", entry, 0, name_offset[".shstrtab"])
    struct.pack_into(endian + "I", entry, 4, 3)  # SHT_STRTAB
    struct.pack_into(endian + "Q", entry, 24, shstrtab_offset)
    struct.pack_into(endian + "Q", entry, 32, len(names))
    entries += entry

    path.write_bytes(bytes(header) + bytes(body) + bytes(entries))
    return path


def test_debug_information_does_not_reach_an_executable_identity(tmp_path):
    # Debug sections cannot change what a compiler built from this
    # installation produces, and on an unstripped build they are most of it.
    before = _elf64(tmp_path / "a.so", {".text": b"code", ".debug_info": b"aaaa"})
    after = _elf64(tmp_path / "b.so", {".text": b"code", ".debug_info": b"bbbb"})

    assert _identity._file_digest(before, sections=True)[1] == _identity._file_digest(after, sections=True)[1]


@pytest.mark.parametrize("section", [".text", ".rodata", ".dynsym", ".shstrtab"])
def test_everything_execution_depends_on_still_reaches_the_identity(tmp_path, section):
    # .dynsym decides what the loader resolves, so it is not debugger-only;
    # .shstrtab names the sections the digest is built from.
    payload = {".text": b"code", ".rodata": b"data", ".dynsym": b"syms"}
    before = _elf64(tmp_path / "a.so", payload)
    changed = dict(payload)
    if section == ".shstrtab":
        changed[".note"] = b"x"
    else:
        changed[section] = b"XXXX"
    after = _elf64(tmp_path / "b.so", changed)

    assert _identity._file_digest(before, sections=True)[1] != _identity._file_digest(after, sections=True)[1]


@pytest.mark.parametrize(
    "before,after",
    [
        # .data and .rodata sort adjacently, so these two files present the
        # digest with the very same byte stream: identical section names (and
        # so an identical .shstrtab), the same bytes, in the same order. Only
        # the boundary between the two sections moves.
        ({".data": b"ab", ".rodata": b"cd"}, {".data": b"abcd", ".rodata": b""}),
        ({".data": b"", ".rodata": b"abcd"}, {".data": b"abcd", ".rodata": b""}),
        # The same bytes under a different section name.
        ({".text": b"code"}, {".rodata": b"code"}),
    ],
)
def test_section_boundaries_are_part_of_the_identity(tmp_path, before, after):
    """Bytes alone do not say what a file does; which section holds them does.

    Feeding each section's name and size in beside its bytes is what keeps
    moving bytes between sections, renaming one, or dropping an empty one from
    producing the same digest as the file that did none of those.
    """
    first = _identity._file_digest(_elf64(tmp_path / "a.so", before), sections=True)[1]
    second = _identity._file_digest(_elf64(tmp_path / "b.so", after), sections=True)[1]

    assert first != second


def _as_relocatable(path: Path) -> Path:
    raw = bytearray(path.read_bytes())
    struct.pack_into("<H", raw, 16, 1)  # ET_REL
    path.write_bytes(bytes(raw))
    return path


def test_a_relocatable_object_keeps_the_symbol_table_a_linker_reads(tmp_path):
    """In a .o the symbol table is a linker input, not debugger-only.

    Two objects with identical code and relocations but a rebuilt .symtab --
    a renamed symbol, a changed binding or visibility -- link differently, so
    the skip that is safe for a finished shared object is not safe here. GCC's
    startup objects reach the inventory through the compiler's own resources.
    """
    first = _as_relocatable(_elf64(tmp_path / "a.o", {".text": b"code", ".symtab": b"symA"}))
    second = _as_relocatable(_elf64(tmp_path / "b.o", {".text": b"code", ".symtab": b"symB"}))

    assert _identity._file_digest(first, sections=True)[1] != _identity._file_digest(second, sections=True)[1]


def test_a_shared_object_preserves_non_debug_symbol_tables(tmp_path):
    # Keep non-debug tables conservatively, even in a finished ET_DYN.
    first = _elf64(tmp_path / "a.so", {".text": b"code", ".symtab": b"symA"})
    second = _elf64(tmp_path / "b.so", {".text": b"code", ".symtab": b"symB"})

    assert _identity._file_digest(first, sections=True)[1] != _identity._file_digest(second, sections=True)[1]


def test_debug_layout_changes_conservatively_invalidate_identity(tmp_path):
    """Debug payloads are omitted, but layout metadata remains authoritative."""
    path = _elf64(tmp_path / "x.so", {".text": b"code", ".debug_info": b"a"})
    size, digest = _identity._file_digest(path, sections=True)
    assert size == path.stat().st_size - 1
    before = fingerprint_content((ContentRoot(path),))
    _elf64(path, {".text": b"code", ".debug_info": b"a" * 64})
    after_size, after_digest = _identity._file_digest(path, sections=True)
    assert after_size == size
    assert after_digest != digest
    assert fingerprint_content((ContentRoot(path),)) != before


def _corrupt(path: Path, offset: int, value: int, code: str = "<Q") -> Path:
    raw = bytearray(path.read_bytes())
    struct.pack_into(code, raw, offset, value)
    path.write_bytes(bytes(raw))
    return path


@pytest.mark.parametrize("field", ["names_size", "names_offset", "section_size"])
def test_a_declared_size_the_file_cannot_hold_falls_back(tmp_path, field):
    """A corrupt header must degrade to a whole-file read, never raise.

    Every size here is read out of the file itself, so a truncated download or
    a fuzzed artifact can declare a section of exabytes. ``read`` of a 64-bit
    size raises MemoryError rather than returning short, so without a bound
    against the file's real length that propagates out of the digest and the
    whole identity fails instead of covering the file the slow way.

    A declared size the read *can* absorb is caught differently -- the short
    read is noticed -- but a section whose bytes run past the end would then be
    reported as truncated content rather than as a file this cannot parse.
    """
    path = _elf64(tmp_path / "x.so", {".text": b"code", ".rodata": b"data"})
    table_offset = struct.unpack_from("<Q", path.read_bytes(), 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", path.read_bytes(), 58)
    names_entry = table_offset + names_index * entry_size
    if field == "names_size":
        _corrupt(path, names_entry + 32, 1 << 60)
    elif field == "names_offset":
        _corrupt(path, names_entry + 24, 1 << 60)
    else:
        _corrupt(path, table_offset + entry_size + 32, 1 << 60)

    size, digest = _identity._file_digest(path, sections=True)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert size == path.stat().st_size


def test_a_section_digest_is_tagged_apart_from_a_whole_file_digest(tmp_path):
    # The two coexist in one record, so a reader must never take one for the
    # other -- an unparsable file falls back to hashing all of its bytes.
    elf = _elf64(tmp_path / "a.so", {".text": b"code"})
    plain = tmp_path / "b.bin"
    plain.write_bytes(b"code")

    assert _identity._file_digest(elf, sections=True)[1].startswith("elf64-debug-filtered-v2:")
    assert not _identity._file_digest(plain, sections=True)[1].startswith("elf64-debug-filtered-v2:")


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    "field,code,value",
    [
        (4, "I", 8),
        (8, "Q", 4),
        (16, "Q", 4096),
        (32, "Q", 2),
        (40, "I", 2),
        (44, "I", 1),
        (48, "Q", 32),
        (56, "Q", 8),
    ],
)
def test_section_metadata_changes_identity(tmp_path, endian, field, code, value):
    """Every section attribute, including NOBITS, is an identity input."""
    path = _elf64(tmp_path / "x.o", {".text": b"code"}, endian=endian)
    before = _identity._file_digest(path, sections=True)
    table = struct.unpack_from(endian + "Q", path.read_bytes(), 40)[0]
    _corrupt(path, table + 64 + field, value, endian + code)
    assert _identity._file_digest(path, sections=True) != before


@pytest.mark.parametrize("field,code,value", [(16, "H", 1), (18, "H", 183), (24, "Q", 4096), (48, "I", 1)])
def test_elf_header_changes_identity(tmp_path, field, code, value):
    path = _elf64(tmp_path / "x.so", {".text": b"code"})
    before = _identity._file_digest(path, sections=True)
    _corrupt(path, field, value, "<" + code)
    assert _identity._file_digest(path, sections=True) != before


@pytest.mark.parametrize(
    "field,code,value",
    [
        (0, "I", 0x6474E551),
        (4, "I", 7),
        (16, "Q", 8192),
        (24, "Q", 8192),
        (32, "Q", 3),
        (40, "Q", 32),
        (48, "Q", 8192),
    ],
)
def test_program_header_changes_identity(tmp_path, field, code, value):
    program = struct.pack("<IIQQQQQQ", 1, 5, 120, 4096, 4096, 4, 4, 4096)
    path = _elf64(tmp_path / "x.so", {".text": b"code"}, program_headers=program)
    before = _identity._file_digest(path, sections=True)
    _corrupt(path, 64 + field, value, "<" + code)
    assert _identity._file_digest(path, sections=True) != before


@pytest.mark.parametrize("name", [".debug_info", ".debugger", ".comment.extra", ".symtab.extra"])
def test_only_unallocated_debug_payloads_are_omitted(tmp_path, name):
    path = _elf64(tmp_path / "x.so", {name: b"aaaa"})
    table = struct.unpack_from("<Q", path.read_bytes(), 40)[0]
    _corrupt(path, table + 64 + 8, 2)  # SHF_ALLOC
    before = _identity._file_digest(path, sections=True)
    _corrupt(path, 64, 0x62626262, "<I")
    assert _identity._file_digest(path, sections=True) != before


@pytest.mark.parametrize("name", [".debugger", ".comment.extra", ".symtab.extra"])
def test_debug_name_prefix_does_not_hide_other_sections(tmp_path, name):
    first = _elf64(tmp_path / "a.so", {name: b"aaaa"})
    second = _elf64(tmp_path / "b.so", {name: b"bbbb"})
    assert _identity._file_digest(first, sections=True) != _identity._file_digest(second, sections=True)


def test_segment_backed_debug_payload_falls_back(tmp_path):
    program = struct.pack("<IIQQQQQQ", 1, 5, 120, 4096, 4096, 4, 4, 4096)
    path = _elf64(tmp_path / "x.so", {".debug_info": b"aaaa"}, program_headers=program)
    assert _identity._file_digest(path, sections=True) == _identity._file_digest(path)


def test_bytes_outside_sections_are_preserved(tmp_path):
    path = _elf64(tmp_path / "x.so", {".text": b"code"})
    with path.open("ab") as stream:
        stream.write(b"extra segment data")
    before = _identity._file_digest(path, sections=True)
    _corrupt(path, path.stat().st_size - 1, ord("X"), "B")
    assert _identity._file_digest(path, sections=True) != before


@pytest.mark.parametrize(
    "offset,code,value",
    [
        (5, "B", 0),
        (5, "B", 3),
        (6, "B", 0),
        (16, "H", 4),
        (20, "I", 2),
        (40, "Q", 1 << 60),
        (52, "H", 65),
        (58, "H", 65535),
        (60, "H", 65535),
        (60, "H", 0),
        (62, "H", 65535),
        (54, "H", 65535),
        (56, "H", 65535),
    ],
)
def test_unsupported_elf_headers_fall_back(tmp_path, offset, code, value):
    path = _elf64(tmp_path / "x.so", {".debug_info": b"aaaa"})
    # A nonzero program count makes its entry size meaningful.
    if offset == 54:
        _corrupt(path, 56, 1, "<H")
    _corrupt(path, offset, value, "<" + code)
    assert _identity._file_digest(path, sections=True) == _identity._file_digest(path)


def test_overlapping_debug_section_falls_back(tmp_path):
    path = _elf64(tmp_path / "x.so", {".text": b"code", ".debug_info": b"aaaa"})
    table = struct.unpack_from("<Q", path.read_bytes(), 40)[0]
    _corrupt(path, table + 128 + 24, 64)
    assert _identity._file_digest(path, sections=True) == _identity._file_digest(path)


def test_sparse_string_table_does_not_cause_large_reads(tmp_path):
    path = _elf64(tmp_path / "x.so", {".debug_info": b"aaaa"})
    raw = path.read_bytes()
    table = struct.unpack_from("<Q", raw, 40)[0]
    names_offset = struct.unpack_from("<Q", raw, table + 128 + 24)[0]
    _corrupt(path, table + 128 + 32, 1 << 32)
    with path.open("r+b") as stream:
        stream.truncate(names_offset + (1 << 32))

    class BoundedReader(io.BufferedReader):
        def read(self, size=-1):
            assert 0 <= size <= 1024 * 1024
            return super().read(size)

    with path.open("rb", buffering=0) as stream, BoundedReader(stream) as bounded:
        assert _identity._elf_debug_ranges(bounded) is None


@pytest.fixture(scope="module")
def clang_elf_object(tmp_path_factory):
    """Exercise genuine Clang sections rather than only synthetic ELF layouts."""
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("Clang is required for the real ELF regression")
    directory = tmp_path_factory.mktemp("clang-identity")
    source = directory / "input.c"
    source.write_text("char buffer[16]; int read_buffer(void) { return buffer[0]; }\n")
    path = directory / "input.o"
    subprocess.run([clang, "-g", "-c", str(source), "-o", str(path)], check=True, capture_output=True)
    return path.read_bytes()


def _section_header(raw: bytes, name: bytes) -> int:
    """Locate a section in the real little-endian ELF64 test object."""
    table = struct.unpack_from("<Q", raw, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", raw, 58)
    names = struct.unpack_from("<Q", raw, table + names_index * entry_size + 24)[0]
    for index in range(count):
        entry = table + index * entry_size
        start = names + struct.unpack_from("<I", raw, entry)[0]
        if raw[start : raw.index(b"\0", start)] == name:
            return entry
    raise AssertionError(f"Missing ELF section {name!r}")


@pytest.mark.parametrize("name,field,value", [(b".bss", 32, 32), (b".bss", 48, 32), (b".text", 8, 2)])
def test_real_clang_elf_metadata_invalidates_identity(tmp_path, clang_elf_object, name, field, value):
    path = tmp_path / "input.o"
    path.write_bytes(clang_elf_object)
    before = _identity._file_digest(path, sections=True)
    entry = _section_header(clang_elf_object, name)
    if name == b".bss" and field == 32:
        assert struct.unpack_from("<Q", clang_elf_object, entry + field)[0] == 16
    _corrupt(path, entry + field, value)
    assert _identity._file_digest(path, sections=True) != before


def test_real_clang_debug_payload_is_skipped(tmp_path, clang_elf_object):
    path = tmp_path / "input.o"
    path.write_bytes(clang_elf_object)
    before = _identity._file_digest(path, sections=True)
    entry = _section_header(clang_elf_object, b".debug_info")
    offset = struct.unpack_from("<Q", clang_elf_object, entry + 24)[0]
    _corrupt(path, offset, clang_elf_object[offset] ^ 1, "B")
    assert _identity._file_digest(path, sections=True) == before
    assert before[0] < len(clang_elf_object)


def test_a_file_that_is_not_elf64_is_read_whole(tmp_path):
    for name, payload in (
        ("short.bin", b"\x7fELF"),
        ("elf32.bin", b"\x7fELF\x01" + bytes(59)),
        ("text.bin", b"not an elf at all"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        size, digest = _identity._file_digest(path, sections=True)
        assert digest == hashlib.sha256(payload).hexdigest()
        assert size == len(payload)


def test_the_artifact_manifest_still_gets_a_whole_file_sha256(tmp_path):
    # _prebuilt recomputes this over the stored bytes to prove they are intact,
    # so the manifest reading must stay a plain digest of the whole file.
    elf = _elf64(tmp_path / "a.so", {".text": b"code", ".debug_info": b"aaaa"})

    size, digest = _identity._file_digest(elf)

    assert digest == hashlib.sha256(elf.read_bytes()).hexdigest()
    assert size == elf.stat().st_size


def test_child_replaced_by_symlink_after_hashing_is_unavailable(tmp_path, monkeypatch):
    # Whether _file_digest's own metadata comparison notices this swap depends
    # on the filesystem: replacing the name changes st_nlink, but not every
    # filesystem reports that as a ctime change. Swap after the read returns so
    # the entry's post-read check is the only thing that can reject it.
    source = tmp_path / "compiler.bin"
    source.write_bytes(b"original")
    target = tmp_path / "target.bin"
    target.write_bytes(b"original")
    original_file_digest = _identity._file_digest

    def replace_after_reading(path, **options):
        result = original_file_digest(path, **options)
        if path == source and not source.is_symlink():
            source.unlink()
            source.symlink_to(target)
        return result

    monkeypatch.setattr(_identity, "_file_digest", replace_after_reading)
    identity = fingerprint_content((ContentRoot(tmp_path),))
    assert identity.digest is None
    assert identity.failure is not None and "symlink changed while being read" in identity.failure


def test_special_files_are_rejected_without_opening_them(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    identity = fingerprint_content((ContentRoot(fifo),))
    assert identity.digest is None
    assert identity.failure is not None and "regular file" in identity.failure


@pytest.fixture
def inventories(tmp_path):
    components = {}
    for name in ("pypto", "runtime", "pto_isa", "ptoas", "device_toolchain"):
        path = tmp_path / name
        path.mkdir()
        (path / "input.bin").write_bytes(name.encode())
        components[name] = ComponentInputs((ContentRoot(path),), unavailable_reason=None)
    return ToolchainInputs(**components)


@pytest.mark.parametrize("component", ["pypto", "runtime", "pto_isa", "ptoas", "device_toolchain"])
def test_every_required_component_must_have_a_complete_inventory(inventories, component):
    cache = InstallationIdentityCache()
    complete = cache.capture(inventories)
    assert complete.usable
    assert complete.digest is not None and len(complete.digest) == 64
    partial = replace(inventories, **{component: ComponentInputs(getattr(inventories, component).roots)})
    identity = cache.capture(partial)
    assert not identity.usable
    assert identity.digest is None
    assert [failure.component for failure in identity.failures] == [component]


def test_all_missing_components_are_reported(inventories):
    inputs = replace(inventories, ptoas=ComponentInputs(), device_toolchain=ComponentInputs())
    identity = InstallationIdentityCache().capture(inputs)
    assert {failure.component for failure in identity.failures} == {"ptoas", "device_toolchain"}
    assert identity.digest is None


def test_failed_component_reads_can_be_retried(inventories):
    path = inventories.ptoas.roots[0].path / "input.bin"
    contents = path.read_bytes()
    path.unlink()
    missing = replace(inventories, ptoas=ComponentInputs((ContentRoot(path),), unavailable_reason=None))
    cache = InstallationIdentityCache()
    assert not cache.capture(missing).usable
    path.write_bytes(contents)
    assert cache.capture(missing).usable


def test_a_verified_revision_identifies_a_component_without_reading_it(inventories):
    # The adapter supplying the revision owns the proof that the contents are
    # that revision; the cache must then not read the component at all.
    verified = replace(inventories.pto_isa, roots=(), verified_revision="a" * 40)
    identity = InstallationIdentityCache().capture(replace(inventories, pto_isa=verified))
    assert identity.usable
    assert identity.pto_isa is not None
    # An empty root tuple is otherwise unavailable, so this cannot be the
    # content path returning a digest by accident.
    assert fingerprint_content(()).digest is None


def test_a_verified_revision_tracks_the_revision(inventories):
    def capture(revision):
        verified = replace(inventories.pto_isa, roots=(), verified_revision=revision)
        return InstallationIdentityCache().capture(replace(inventories, pto_isa=verified)).pto_isa

    assert capture("a" * 40) != capture("b" * 40)


def test_a_verified_revision_cannot_collide_across_schemes(inventories):
    revision = "c" * 40
    as_pto_isa = replace(inventories.pto_isa, roots=(), verified_revision=revision)
    as_runtime = replace(inventories.runtime, roots=(), verified_revision=revision)
    identity = InstallationIdentityCache().capture(
        replace(inventories, pto_isa=as_pto_isa, runtime=as_runtime)
    )
    # The same revision under two components must not produce one digest, and
    # neither may equal a content digest of the component's own inputs.
    assert identity.pto_isa != identity.runtime
    assert identity.pto_isa != fingerprint_content(inventories.pto_isa.roots).digest


def test_a_reported_version_is_not_a_verified_revision(inventories):
    # The same string under the two fields records different evidence, so the
    # digests must differ; otherwise a weaker claim could impersonate a proof.
    claim = "0.61"
    reported = replace(inventories.ptoas, roots=(), reported_version=claim)
    verified = replace(inventories.ptoas, roots=(), verified_revision=claim)
    cache = InstallationIdentityCache()
    as_reported = cache.capture(replace(inventories, ptoas=reported)).ptoas
    as_verified = cache.capture(replace(inventories, ptoas=verified)).ptoas
    assert as_reported is not None and as_verified is not None
    assert as_reported != as_verified


def test_a_reported_version_tracks_the_whole_string(inventories):
    def capture(reported):
        component = replace(inventories.ptoas, roots=(), reported_version=reported)
        return InstallationIdentityCache().capture(replace(inventories, ptoas=component)).ptoas

    # A dev build and the release it came from share a parsed number; the
    # identity must still separate them.
    assert capture("ptoas 0.61") != capture("ptoas 0.61.dev3")


def test_one_component_can_carry_a_version_and_contents(inventories):
    # A component whose files come from a vendor package and from the host OS
    # covers each part with the evidence that part has; the identity must move
    # when either moves.
    def capture(reported, payload):
        (inventories.device_toolchain.roots[0].path / "input.bin").write_text(payload)
        component = replace(inventories.device_toolchain, reported_version=reported)
        return (
            InstallationIdentityCache()
            .capture(replace(inventories, device_toolchain=component))
            .device_toolchain
        )

    base = capture("B250", "aaa")
    assert base is not None
    assert capture("B251", "aaa") != base
    assert capture("B250", "bbb") != base


def test_a_version_never_stands_in_for_declared_contents(inventories):
    # With roots present, an unreadable inventory must fail the component even
    # though a version is available.
    missing = replace(
        inventories.device_toolchain,
        roots=(ContentRoot(inventories.device_toolchain.roots[0].path.parent / "absent"),),
        reported_version="B250",
    )
    identity = InstallationIdentityCache().capture(replace(inventories, device_toolchain=missing))
    assert not identity.usable
    assert identity.device_toolchain is None


def test_a_component_supplying_nothing_stays_unavailable(inventories):
    empty = replace(inventories.device_toolchain, roots=())
    identity = InstallationIdentityCache().capture(replace(inventories, device_toolchain=empty))
    assert not identity.usable
    assert identity.device_toolchain is None


def test_an_unavailable_component_outranks_a_verified_revision(inventories):
    blocked = replace(
        inventories.pto_isa, roots=(), verified_revision="d" * 40, unavailable_reason="probe failed"
    )
    identity = InstallationIdentityCache().capture(replace(inventories, pto_isa=blocked))
    assert not identity.usable
    assert [failure.reason for failure in identity.failures] == ["probe failed"]


def test_installation_cache_is_keyed_by_resolved_inputs(inventories, tmp_path):
    cache = InstallationIdentityCache()
    first = cache.capture(inventories)
    tool = tmp_path / "other-compiler"
    tool.write_bytes(b"different compiler")
    selected = replace(
        inventories, device_toolchain=ComponentInputs((ContentRoot(tool),), unavailable_reason=None)
    )
    assert cache.capture(selected).digest != first.digest
    assert cache.capture(inventories) == first


def test_replacing_an_installation_requires_a_new_snapshot(inventories):
    cache = InstallationIdentityCache()
    first = cache.capture(inventories)
    library = inventories.pypto.roots[0].path / "input.bin"
    metadata = library.stat()
    library.write_bytes(b"other")
    os.utime(library, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert cache.capture(inventories) == first
    assert InstallationIdentityCache().capture(inventories).digest != first.digest


def test_new_process_reads_replaced_native_file_contents(tmp_path):
    library = tmp_path / "compiler.so"
    library.write_bytes(b"same-build-id;old-code")
    code = (
        "import sys; from pathlib import Path; "
        "from pypto._identity import ContentRoot, fingerprint_content; "
        "print(fingerprint_content((ContentRoot(Path(sys.argv[1])),)).digest)"
    )

    def child_digest():
        return subprocess.run(
            [sys.executable, "-c", code, str(library)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()

    before = child_digest()
    metadata = library.stat()
    library.write_bytes(b"same-build-id;new-code")
    os.utime(library, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert child_digest() != before


def test_threads_share_one_successful_inventory_read(inventories, monkeypatch):
    cache = InstallationIdentityCache()
    original = _identity.fingerprint_content
    seen = []

    def count(roots):
        seen.append(roots)
        return original(roots)

    monkeypatch.setattr(_identity, "fingerprint_content", count)
    barrier = threading.Barrier(4)

    def capture(inputs):
        barrier.wait(timeout=10)
        return cache.capture(inputs)

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(capture, [inventories] * 8))
    assert all(identity == identities[0] for identity in identities)
    assert len(seen) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
