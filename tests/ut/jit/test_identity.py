# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Protocol and content identity tests; no toolchain or device is required."""

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
