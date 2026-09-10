# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free acceptance tests for immutable artifact publication and recovery."""

import json
import multiprocessing
import os
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pypto._fslock import file_lock
from pypto._identity import ToolchainIdentity, digest_record
from pypto.jit import artifact_cache
from pypto.jit._artifact_manifest import MANIFEST_NAME, ArtifactKey, ArtifactSpec, ArtifactState, BuildKind
from pypto.jit.artifact_cache import (
    ArtifactLookup,
    ArtifactStore,
    BuildDisposition,
    BuildFailure,
    LookupStatus,
)


def _key(source="source"):
    identity = ToolchainIdentity(
        pypto=digest_record("pypto"),
        runtime=digest_record("runtime"),
        pto_isa=digest_record("isa"),
        ptoas=digest_record("ptoas"),
        device_toolchain=digest_record("cxx"),
    )
    return ArtifactKey(identity, digest_record(source), digest_record({"rows": 16}))


def _spec(state=ArtifactState.GENERATED, build_kind=BuildKind.SINGLE_CHIP):
    return ArtifactSpec(state, build_kind, ("kernel/source.pto", "kernel_config.py"))


def _builder(directory):
    (directory / "kernel").mkdir()
    (directory / "kernel/source.pto").write_bytes(b"generated code")
    (directory / "kernel_config.py").write_bytes(b"KERNELS = []\n")
    return directory / "kernel/source.pto"


def _unexpected_builder(directory):
    pytest.fail(f"A hit must not call its builder: {directory}")


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")


@pytest.mark.parametrize("state", list(ArtifactState))
@pytest.mark.parametrize("kind", list(BuildKind))
def test_publish_then_hit_preserves_private_value_and_records(store, state, kind):
    key, spec = _key(), _spec(state, kind)
    result = store.get_or_build(key, spec, _builder)
    assert result.disposition is BuildDisposition.PUBLISHED
    assert result.value.read_bytes() == b"generated code"
    assert result.private_directory in result.value.parents
    handle = result.handle
    assert (
        handle.directory
        == store.root / "artifacts" / key.environment.digest / key.digest / spec.digest / state.value
    )
    marker = json.loads((handle.directory / MANIFEST_NAME).read_text())
    assert marker["components"] == key.record()
    assert marker["build_kind"] == kind.value
    assert marker["files"][0]["sha256"]
    hit = store.get_or_build(key, spec, _unexpected_builder)
    assert hit.disposition is BuildDisposition.HIT
    assert hit.handle == handle
    assert hit.value is None and hit.private_directory is None


def test_changed_spec_gets_an_independent_reusable_slot(store):
    key = _key()
    original = _spec()
    extended = replace(original, required_files=(*original.required_files, "extra.json"))
    distributed = replace(original, build_kind=BuildKind.DISTRIBUTED)

    def build(directory):
        value = _builder(directory)
        (directory / "extra.json").write_text("{}")
        return value

    handles = []
    for spec in (original, extended, distributed):
        assert store.lookup(key, spec).status is LookupStatus.MISS
        result = store.get_or_build(key, spec, build)
        assert result.disposition is BuildDisposition.PUBLISHED
        handles.append(result.handle)
    assert len({handle.directory for handle in handles}) == 3
    for spec, handle in zip((original, extended, distributed), handles):
        assert store.get_or_build(key, spec, _unexpected_builder).handle == handle
    assert (
        replace(original, required_files=tuple(reversed(original.required_files))).digest == original.digest
    )


def test_invalid_slot_does_not_poison_another_spec(store):
    original = _spec()
    changed = replace(original, build_kind=BuildKind.DISTRIBUTED)
    directory = store.get_or_build(_key(), original, _builder).handle.directory
    (directory / MANIFEST_NAME).write_bytes(b"damaged marker")
    assert store.lookup(_key(), original).status is LookupStatus.INVALID
    assert store.lookup(_key(), changed).status is LookupStatus.MISS
    assert store.get_or_build(_key(), changed, _builder).disposition is BuildDisposition.PUBLISHED
    assert (directory / MANIFEST_NAME).read_bytes() == b"damaged marker"


def test_key_requires_complete_full_digests():
    key = _key()
    with pytest.raises(ValueError, match="usable toolchain"):
        replace(key, environment=replace(key.environment, ptoas=None))
    with pytest.raises(ValueError, match="current schema"):
        replace(key, environment=replace(key.environment, schema=99))
    for invalid in ("unknown", "a" * 16, "A" * 64, "../escape", None):
        with pytest.raises(ValueError, match="SHA-256"):
            replace(key, source_digest=invalid)
    assert key.digest != replace(key, specialization_digest=digest_record({"rows": 32})).digest
    assert (
        key.digest != replace(key, environment=replace(key.environment, runtime=digest_record("new"))).digest
    )


@pytest.mark.parametrize(
    "path", ["../escape", "/abs", "a/../b", "a//b", "a/./b", "a/", "a\\b", "", ".", MANIFEST_NAME]
)
def test_spec_rejects_unsafe_relative_paths(path):
    with pytest.raises(ValueError, match="path"):
        ArtifactSpec(ArtifactState.GENERATED, BuildKind.SINGLE_CHIP, (path,))


@pytest.mark.parametrize("files", [(), ("a", "a")])
def test_spec_requires_nonempty_unique_files(files):
    with pytest.raises(ValueError, match="nonempty and unique"):
        ArtifactSpec(ArtifactState.GENERATED, BuildKind.SINGLE_CHIP, files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", 99),
        ("schema", True),
        ("schema", 1.0),
        ("key", "a" * 64),
        ("components", {}),
        ("state", "ready"),
        ("build_kind", "distributed"),
        ("required_files", []),
        ("files", []),
        ("unexpected", 1),
    ],
)
def test_invalid_manifest_is_not_repaired(store, field, value):
    key, spec = _key(), _spec()
    directory = store.get_or_build(key, spec, _builder).handle.directory
    marker = directory / MANIFEST_NAME
    manifest = json.loads(marker.read_text())
    manifest[field] = value
    marker.write_text(json.dumps(manifest))
    before = marker.read_bytes()
    assert store.lookup(key, spec).status is LookupStatus.INVALID
    rebuilt = store.get_or_build(key, spec, _builder)
    assert rebuilt.disposition is BuildDisposition.PRIVATE
    assert rebuilt.reason and rebuilt.value.exists()
    assert marker.read_bytes() == before


@pytest.mark.parametrize(
    "damage", ["missing", "changed", "extra", "mode", "symlink", "fifo", "marker", "duplicate"]
)
def test_payload_and_completion_marker_are_verified(store, damage):
    key, spec = _key(), _spec()
    directory = store.get_or_build(key, spec, _builder).handle.directory
    source = directory / "kernel/source.pto"
    marker = directory / MANIFEST_NAME
    if damage == "missing":
        source.unlink()
    elif damage == "changed":
        # Same-size corruption must fail too.
        source.write_bytes(b"GENERATED CODE")
    elif damage == "extra":
        (directory / "unlisted.so").write_bytes(b"extra executable code")
    elif damage == "mode":
        source.chmod(0o700)
    elif damage == "symlink":
        source.unlink()
        source.symlink_to(directory / "kernel_config.py")
    elif damage == "fifo":
        source.unlink()
        os.mkfifo(source)
    elif damage == "marker":
        marker.unlink()
    else:
        marker.write_text(marker.read_text().replace('"schema":1', '"schema":1,"schema":1', 1))
    assert store.lookup(key, spec).status is LookupStatus.INVALID


@pytest.mark.parametrize("field,value", [("path", "../../outside"), ("size", True), ("sha256", "a" * 64)])
def test_file_records_cannot_redirect_or_coerce_validation(store, field, value):
    key, spec = _key(), _spec()
    directory = store.get_or_build(key, spec, _builder).handle.directory
    marker = directory / MANIFEST_NAME
    manifest = json.loads(marker.read_text())
    manifest["files"][0][field] = value
    marker.write_text(json.dumps(manifest))
    assert store.lookup(key, spec).status is LookupStatus.INVALID


def test_generated_to_ready_copies_without_mutating_shared_source(store):
    key, generated = _key(), _spec()
    handle = store.get_or_build(key, generated, _builder).handle
    ready = ArtifactSpec(
        ArtifactState.BINARY_READY, BuildKind.SINGLE_CHIP, (*generated.required_files, "obj.so")
    )

    def compile_binary(directory):
        handle.materialize(directory)
        copied = directory / "kernel/source.pto"
        assert copied.stat().st_ino != (handle.directory / "kernel/source.pto").stat().st_ino
        assert not (directory / MANIFEST_NAME).exists()
        copied.write_bytes(b"private adaptation")
        (directory / "obj.so").write_bytes(b"synthetic binary")
        return copied

    result = store.get_or_build(key, ready, compile_binary)
    assert result.disposition is BuildDisposition.PUBLISHED
    assert store.lookup(key, generated).status is LookupStatus.HIT
    assert store.lookup(key, ready).status is LookupStatus.HIT
    assert (handle.directory / "kernel/source.pto").read_bytes() == b"generated code"


def test_readonly_hit_and_miss_make_no_cache_writes(store, monkeypatch):
    key, spec = _key(), _spec()
    handle = store.get_or_build(key, spec, _builder).handle
    before = {path.relative_to(store.root): path.stat().st_mtime_ns for path in store.root.rglob("*")}
    readonly = ArtifactStore(store.root, readonly=True, private_root=store.private_root)

    def forbidden_lock(path):
        pytest.fail(f"Read-only access attempted a lock: {path}")

    monkeypatch.setattr(artifact_cache, "file_lock", forbidden_lock)
    assert readonly.get_or_build(key, spec, _unexpected_builder).handle == handle
    missing = readonly.get_or_build(_key("other"), spec, _builder)
    assert missing.disposition is BuildDisposition.PRIVATE
    assert missing.value is not None
    assert missing.value.exists()
    after = {path.relative_to(store.root): path.stat().st_mtime_ns for path in store.root.rglob("*")}
    assert after == before


def test_readonly_missing_root_is_never_created(tmp_path):
    store = ArtifactStore(tmp_path / "absent", readonly=True, private_root=tmp_path / "private")
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.value is not None
    assert result.value.exists()
    assert not store.root.exists()
    with pytest.raises(ValueError, match="outside"):
        ArtifactStore(store.root, private_root=store.root / "private")


@pytest.mark.parametrize("state", list(ArtifactState))
def test_removing_write_permissions_preserves_cache_hits(store, state):
    key, spec = _key(), _spec(state)
    handle = store.get_or_build(key, spec, _builder).handle
    paths = [store.root, *store.root.rglob("*")]
    original_modes = {path: path.stat().st_mode & 0o777 for path in paths}
    try:
        for path in reversed(paths):
            path.chmod(original_modes[path] & ~0o222)
        sealed_modes = {path: path.stat().st_mode for path in paths}
        readonly = ArtifactStore(store.root, readonly=True)
        hit = readonly.get_or_build(key, spec, _unexpected_builder)
        assert hit.disposition is BuildDisposition.HIT
        assert hit.handle == handle
        assert {path: path.stat().st_mode for path in paths} == sealed_modes
    finally:
        for path in paths:
            path.chmod(original_modes[path])


def test_readonly_payload_materializes_writable_private_files(store, tmp_path):
    def build(directory):
        value = _builder(directory)
        value.chmod(0o444)
        (directory / "kernel_config.py").chmod(0o555)
        return value

    result = store.get_or_build(_key(), _spec(), build)
    destination = tmp_path / "promotion"
    destination.mkdir()
    result.handle.materialize(destination)
    copied = destination / "kernel/source.pto"
    assert copied.stat().st_mode & 0o200
    assert (destination / "kernel_config.py").stat().st_mode & 0o111 == 0o100
    copied.write_bytes(b"adapted source")
    assert (result.handle.directory / "kernel/source.pto").read_bytes() == b"generated code"


@pytest.mark.parametrize("destination_kind", ["sibling", "other_key", "symlink"])
def test_materialization_rejects_all_shared_cache_destinations(store, tmp_path, destination_kind):
    key, spec = _key(), _spec()
    handle = store.get_or_build(key, spec, _builder).handle
    shared = handle.directory.parent / "ready"
    if destination_kind == "other_key":
        shared = store._slot(_key("other"), spec)
    shared.mkdir(parents=True)
    destination = shared
    if destination_kind == "symlink":
        destination = tmp_path / "private-looking-link"
        destination.symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the.*cache"):
        handle.materialize(destination)
    assert list(shared.iterdir()) == []
    assert store.lookup(key, spec).status is LookupStatus.HIT


@pytest.mark.parametrize("source_mode", [0o700, 0o755])
def test_sharing_chmod_preserves_owner_execute_contract(store, source_mode):
    def build(directory):
        value = _builder(directory)
        value.chmod(source_mode)
        return value

    key, spec = _key(), _spec()
    handle = store.get_or_build(key, spec, build).handle
    assert handle.directory.stat().st_mode & 0o777 == 0o700
    assert (handle.directory / "kernel/source.pto").stat().st_mode & 0o777 == 0o700
    assert (handle.directory / "kernel_config.py").stat().st_mode & 0o777 == 0o600
    for permissions in ("a+rX", "go-x"):
        subprocess.run(["chmod", "-R", permissions, "--", str(handle.directory)], check=True)
        assert store.lookup(key, spec).status is LookupStatus.HIT
    source = handle.directory / "kernel/source.pto"
    source.chmod(source.stat().st_mode & ~0o100)
    assert store.lookup(key, spec).status is LookupStatus.INVALID


def test_missing_private_root_allows_hits_without_probing_temporary_directories(store, monkeypatch):
    key, spec = _key(), _spec()
    store.get_or_build(key, spec, _builder)

    def forbidden_probe():
        pytest.fail("Temporary-directory discovery can write into a cache-root candidate")

    monkeypatch.setattr(artifact_cache.tempfile, "gettempdir", forbidden_probe)
    readonly = ArtifactStore(store.root, readonly=True)
    assert readonly.get_or_build(key, spec, _unexpected_builder).disposition is BuildDisposition.HIT
    with pytest.raises(OSError, match="requires a private_root"):
        readonly.get_or_build(_key("missing"), spec, _unexpected_builder)


def test_orphan_staging_and_empty_final_slot_never_hit(store):
    key, spec = _key(), _spec()
    slot = store._slot(key, spec)
    orphan = slot.parent / ".tmp.abandoned"
    orphan.mkdir(parents=True)
    _builder(orphan)
    assert store.lookup(key, spec).status is LookupStatus.MISS
    slot.mkdir()
    assert store.lookup(key, spec).status is LookupStatus.INVALID
    result = store.get_or_build(key, spec, _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert list(slot.iterdir()) == []
    assert orphan.exists()


@pytest.mark.parametrize(
    "operation", ["_copy_payload", "_complete_staging", "_rename_noreplace", "_sync_directory"]
)
def test_publication_failures_retain_private_build(store, monkeypatch, operation):
    def fail(*args):
        raise OSError("injected storage failure")

    monkeypatch.setattr(artifact_cache, operation, fail)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert "injected storage failure" in result.reason
    assert result.value.read_bytes() == b"generated code"
    assert store.lookup(_key(), _spec()).status is LookupStatus.MISS
    assert not list(store.root.rglob(".tmp.*"))


def test_failed_parent_sync_after_rename_keeps_both_valid_outputs(store, monkeypatch):
    sync = artifact_cache._sync_directory
    slot = store._slot(_key(), _spec())

    def fail_after_rename(path):
        if path == slot.parent:
            raise OSError("injected parent sync failure")
        sync(path)

    monkeypatch.setattr(artifact_cache, "_sync_directory", fail_after_rename)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert result.value.exists()
    assert store.lookup(_key(), _spec()).status is LookupStatus.HIT


def test_no_replace_preserves_even_an_empty_destination_created_during_build(store, monkeypatch):
    rename = artifact_cache._rename_noreplace

    def race(source, destination):
        destination.mkdir()
        rename(source, destination)

    monkeypatch.setattr(artifact_cache, "_rename_noreplace", race)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert result.value.exists()
    assert list(store._slot(_key(), _spec()).iterdir()) == []


def test_lock_failure_falls_back_but_builder_oserror_propagates(store, monkeypatch):
    def fail_lock(path):
        raise OSError("injected lock failure")

    monkeypatch.setattr(artifact_cache, "file_lock", fail_lock)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert "injected lock failure" in result.reason

    def fail_build(directory):
        raise OSError("compiler failure")

    with pytest.raises(OSError, match="compiler failure"):
        store.get_or_build(_key(), _spec(), fail_build)


def test_builder_failure_releases_lock_and_missing_outputs_propagate(store):
    def fail(directory):
        raise RuntimeError("compiler failure")

    with pytest.raises(RuntimeError, match="compiler failure"):
        store.get_or_build(_key(), _spec(), fail)
    with pytest.raises(ValueError, match="missing required files"):
        store.get_or_build(_key(), _spec(), lambda directory: None)
    assert store.get_or_build(_key(), _spec(), _builder).disposition is BuildDisposition.PUBLISHED


def test_symlinked_cache_ancestor_is_invalid_and_not_written(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    store.root.mkdir()
    (store.root / "artifacts").symlink_to(outside, target_is_directory=True)
    assert store.lookup(_key(), _spec()).status is LookupStatus.INVALID
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert list(outside.iterdir()) == []


def test_symlinked_lock_directory_is_not_followed(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    store.root.mkdir()
    (store.root / "locks").symlink_to(outside, target_is_directory=True)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert "lock unavailable" in result.reason
    assert list(outside.iterdir()) == []


def test_unreadable_manifest_falls_back_without_repair(store, monkeypatch):
    directory = store.get_or_build(_key(), _spec(), _builder).handle.directory
    original_open = Path.open

    def fail_manifest(path, *args, **kwargs):
        if path == directory / MANIFEST_NAME:
            raise PermissionError("injected unreadable marker")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_manifest)
    assert store.lookup(_key(), _spec()).status is LookupStatus.STORAGE_ERROR
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert "unreadable marker" in result.reason
    assert result.value.exists()


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"null", b"[]", b'{"files":NaN}'])
def test_malformed_marker_is_invalid(store, raw):
    directory = store.get_or_build(_key(), _spec(), _builder).handle.directory
    (directory / MANIFEST_NAME).write_bytes(raw)
    assert store.lookup(_key(), _spec()).status is LookupStatus.INVALID


def test_copy_corruption_never_publishes(store, monkeypatch):
    copy = artifact_cache._copy_payload

    def corrupt(source, destination, manifest):
        copy(source, destination, manifest)
        (destination / "kernel/source.pto").write_bytes(b"corrupt copy")

    monkeypatch.setattr(artifact_cache, "_copy_payload", corrupt)
    result = store.get_or_build(_key(), _spec(), _builder)
    assert result.disposition is BuildDisposition.PRIVATE
    assert result.value.read_bytes() == b"generated code"
    assert store.lookup(_key(), _spec()).status is LookupStatus.MISS


def test_same_key_threads_build_once(store):
    start = threading.Barrier(4)
    calls = []

    def build(directory):
        calls.append(directory)
        return _builder(directory)

    def invoke():
        start.wait(timeout=20)
        return store.get_or_build(_key(), _spec(), build).disposition

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: invoke(), range(4)))
    assert len(calls) == 1
    assert results.count(BuildDisposition.PUBLISHED) == 1
    assert results.count(BuildDisposition.HIT) == 3


def _join_build_waiters(monkeypatch):
    joined = threading.Barrier(4)

    class WaitingFuture(Future):
        def result(self, timeout=None):
            joined.wait(timeout=20)
            return super().result(timeout)

    monkeypatch.setattr(artifact_cache, "Future", WaitingFuture)
    return joined


@pytest.mark.parametrize("failure", ["invalid", "storage_error", "readonly", "lock", "publication"])
def test_private_builds_coalesce_across_store_instances(store, monkeypatch, failure):
    key, spec = _key(), _spec()
    if failure == "invalid":
        store._slot(key, spec).mkdir(parents=True)
    elif failure == "storage_error":
        monkeypatch.setattr(
            ArtifactStore,
            "lookup",
            lambda self, key, spec: ArtifactLookup(LookupStatus.STORAGE_ERROR, reason="unreadable cache"),
        )

    def fail(*args):
        raise OSError("storage failure")

    if failure == "lock":
        monkeypatch.setattr(artifact_cache, "file_lock", fail)
    elif failure == "publication":
        monkeypatch.setattr(artifact_cache, "_rename_noreplace", fail)
    joined = _join_build_waiters(monkeypatch)
    calls = []

    def build(directory):
        calls.append(directory)
        joined.wait(timeout=20)
        return _builder(directory)

    def invoke():
        instance = ArtifactStore(store.root, private_root=store.private_root, readonly=failure == "readonly")
        return instance.get_or_build(key, spec, build)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: invoke(), range(4)))
    assert len(calls) == 1
    assert all(result is results[0] for result in results)
    assert results[0].disposition is BuildDisposition.PRIVATE
    assert (
        results[0].failure
        is {
            "invalid": BuildFailure.INVALID,
            "storage_error": BuildFailure.STORAGE,
            "readonly": None,
            "lock": BuildFailure.LOCK,
            "publication": BuildFailure.PUBLICATION,
        }[failure]
    )
    assert results[0].value is not None
    assert results[0].value.read_bytes() == b"generated code"
    assert not artifact_cache._build_flights.pending
    if failure == "readonly":
        assert not store.root.exists()
    if failure == "invalid":
        assert list(store._slot(key, spec).iterdir()) == []
    # A completed cohort is not an unbounded private-object cache.
    monkeypatch.setattr(artifact_cache, "Future", Future)
    instance = ArtifactStore(store.root, private_root=store.private_root, readonly=failure == "readonly")
    again = instance.get_or_build(key, spec, _builder)
    assert again.private_directory != results[0].private_directory


class _BuildCancelled(BaseException):
    pass


@pytest.mark.parametrize("error_type", [OSError, _BuildCancelled])
def test_coalesced_compiler_errors_wake_waiters_and_allow_retry(store, monkeypatch, error_type):
    joined = _join_build_waiters(monkeypatch)
    failure = error_type("compiler cancelled or failed")
    calls = []

    def build(directory):
        calls.append(directory)
        joined.wait(timeout=20)
        raise failure

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(store.get_or_build, _key(), _spec(), build) for _ in range(4)]
        for future in futures:
            with pytest.raises(error_type) as error:
                future.result(timeout=30)
            assert error.value is failure
    assert len(calls) == 1
    assert not artifact_cache._build_flights.pending
    monkeypatch.setattr(artifact_cache, "Future", Future)
    assert store.get_or_build(_key(), _spec(), _builder).disposition is BuildDisposition.PUBLISHED


def test_coalescing_rejects_recursive_requests_without_leaking_state(store):
    def recurse(directory):
        return store.get_or_build(_key(), _spec(), _builder)

    with pytest.raises(RuntimeError, match="recursively requested"):
        store.get_or_build(_key(), _spec(), recurse)
    assert not artifact_cache._build_flights.pending
    assert store.get_or_build(_key(), _spec(), _builder).disposition is BuildDisposition.PUBLISHED


@pytest.mark.parametrize("change", ["spec", "private_root", "readonly"])
def test_coalescing_keeps_distinct_requests_independent(store, tmp_path, change):
    joined = threading.Barrier(2)
    first = ArtifactStore(store.root, private_root=store.private_root, readonly=True)
    second = ArtifactStore(
        store.root,
        private_root=tmp_path / "other-private" if change == "private_root" else store.private_root,
        readonly=change != "readonly",
    )
    other_spec = replace(_spec(), build_kind=BuildKind.DISTRIBUTED) if change == "spec" else _spec()

    def build(directory):
        joined.wait(timeout=20)
        return _builder(directory)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.get_or_build, _key(), _spec(), build)
        b = pool.submit(second.get_or_build, _key(), other_spec, build)
        assert a.result(timeout=30).private_directory != b.result(timeout=30).private_directory


def _fork_private_build(root, private_root, sender):
    store = ArtifactStore(Path(root), private_root=Path(private_root), readonly=True)
    result = store.get_or_build(_key(), _spec(), _builder)
    sender.send(result.private_directory)
    sender.close()


def test_forked_child_does_not_wait_for_parent_build(store):
    started, release = threading.Event(), threading.Event()
    readonly = ArtifactStore(store.root, private_root=store.private_root, readonly=True)

    def build(directory):
        started.set()
        assert release.wait(timeout=30)
        return _builder(directory)

    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_fork_private_build, args=(str(store.root), str(store.private_root), sender))
    with ThreadPoolExecutor(max_workers=1) as pool:
        parent = pool.submit(readonly.get_or_build, _key(), _spec(), build)
        try:
            assert started.wait(timeout=20)
            child.start()
            sender.close()
            assert receiver.poll(timeout=20)
            child_directory = receiver.recv()
            child.join(timeout=10)
            assert child.exitcode == 0
            release.set()
            assert child_directory != parent.result(timeout=10).private_directory
        finally:
            release.set()
            if child.is_alive():
                child.kill()
                child.join(timeout=10)
            receiver.close()
            sender.close()


def test_different_keys_build_concurrently(store):
    inside_build = threading.Barrier(2)

    def build(directory):
        inside_build.wait(timeout=20)
        return _builder(directory)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(store.get_or_build, _key(name), _spec(), build) for name in ("a", "b")]
        assert all(future.result(timeout=30).disposition is BuildDisposition.PUBLISHED for future in futures)


def _process_build(root, private_root, start, outcomes):
    store = ArtifactStore(Path(root), private_root=Path(private_root))
    start.wait(timeout=30)
    result = store.get_or_build(_key(), _spec(), _builder)
    outcomes.put(result.disposition.value)


def _exit_with_lock(lock_path, acquired):
    with file_lock(Path(lock_path)):
        acquired.set()
        os._exit(7)


def test_spawned_processes_deduplicate_and_dead_process_releases_lock(store):
    ctx = multiprocessing.get_context("spawn")
    start, outcomes = ctx.Barrier(3), ctx.Queue()
    processes = [
        ctx.Process(target=_process_build, args=(str(store.root), str(store.private_root), start, outcomes))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        results = [outcomes.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert results.count("published") == 1
        assert results.count("hit") == 2
        assert len(list(store.private_root.iterdir())) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        outcomes.close()

    lock_path = store.root / "locks" / f"{_key('after-death').digest}.lock"
    acquired = ctx.Event()
    process = ctx.Process(target=_exit_with_lock, args=(str(lock_path), acquired))
    try:
        process.start()
        assert acquired.wait(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 7
        inode = lock_path.stat().st_ino
        result = store.get_or_build(_key("after-death"), _spec(), _builder)
        assert result.disposition is BuildDisposition.PUBLISHED
        assert lock_path.stat().st_ino == inode
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
