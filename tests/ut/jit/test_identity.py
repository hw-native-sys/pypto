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
import os
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


def _elf64(path: Path, sections: dict[str, bytes]) -> Path:
    """Write a minimal ELF64 whose named sections carry the given bytes."""
    names = bytearray(b"\0")
    name_offset = {}
    for name in (*sections, ".shstrtab"):
        name_offset[name] = len(names)
        names += name.encode() + b"\0"

    body = bytearray()
    placed = []
    for name, payload in sections.items():
        placed.append((name, 64 + len(body), len(payload)))
        body += payload
    shstrtab_offset = 64 + len(body)
    body += names
    table_offset = 64 + len(body)

    header = bytearray(64)
    header[0:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", header, 40, table_offset)
    struct.pack_into("<HHH", header, 58, 64, len(placed) + 2, len(placed) + 1)

    entries = bytearray(64)  # index 0 is SHT_NULL
    for name, offset, size in placed:
        entry = bytearray(64)
        struct.pack_into("<I", entry, 0, name_offset[name])
        struct.pack_into("<I", entry, 4, 1)  # SHT_PROGBITS
        struct.pack_into("<Q", entry, 24, offset)
        struct.pack_into("<Q", entry, 32, size)
        entries += entry
    entry = bytearray(64)
    struct.pack_into("<I", entry, 0, name_offset[".shstrtab"])
    struct.pack_into("<I", entry, 4, 3)  # SHT_STRTAB
    struct.pack_into("<Q", entry, 24, shstrtab_offset)
    struct.pack_into("<Q", entry, 32, len(names))
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


def test_a_section_digest_is_tagged_apart_from_a_whole_file_digest(tmp_path):
    # The two coexist in one record, so a reader must never take one for the
    # other -- an unparsable file falls back to hashing all of its bytes.
    elf = _elf64(tmp_path / "a.so", {".text": b"code"})
    plain = tmp_path / "b.bin"
    plain.write_bytes(b"code")

    assert _identity._file_digest(elf, sections=True)[1].startswith("elf64-sections:")
    assert not _identity._file_digest(plain, sections=True)[1].startswith("elf64-sections:")


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
