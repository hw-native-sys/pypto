# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for generated-binary runtime compatibility stamps."""

import fcntl
import json
import multiprocessing
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pypto.runtime._binary_cache as binary_cache
import pytest
from pypto.runtime._binary_cache import (
    BinaryCacheContext,
    binary_context_path,
    prepare_binary_context,
    record_binary_context,
)

_RUNTIME_OLD = "a" * 40
_RUNTIME_NEW = "b" * 40
_PTO_ISA = "c" * 40


def _context(**changes) -> BinaryCacheContext:
    context = BinaryCacheContext(
        platform="a2a3sim",
        runtime_name="tensormap_and_ringbuffer",
        runtime_revision=_RUNTIME_OLD,
        pto_isa_revision=_PTO_ISA,
    )
    return replace(context, **changes)


def _touch(path: Path, content: bytes = b"binary") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    (path / "tracked.hpp").write_text("// committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.hpp"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PyPTO Tests",
            "-c",
            "user.email=pypto-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_runtime_revision_change_invalidates_both_orchestration_caches(tmp_path: Path) -> None:
    """A runtime-header revision change invalidates prebuild and sibling binaries."""
    old_context = _context()
    record_binary_context(tmp_path, old_context)
    orch_prebuild = _touch(tmp_path / "cache" / "orch_main.bin")
    orch_sidecar = _touch(tmp_path / "orchestration" / "main.so")
    kernel_sidecar = _touch(tmp_path / "kernels" / "aiv" / "kernel.so")
    kernel_object = _touch(tmp_path / "kernels" / "aic" / "kernel.o")
    orch_source = _touch(tmp_path / "orchestration" / "main.cpp", b"// source\n")
    kernel_source = _touch(tmp_path / "kernels" / "aiv" / "kernel.cpp", b"// source\n")

    removed = prepare_binary_context(
        tmp_path,
        replace(old_context, runtime_revision=_RUNTIME_NEW),
    )

    assert removed == 4
    assert not orch_prebuild.exists()
    assert not orch_sidecar.exists()
    assert not kernel_sidecar.exists()
    assert not kernel_object.exists()
    assert orch_source.exists()
    assert kernel_source.exists()
    assert not binary_context_path(tmp_path).exists()


def test_matching_context_preserves_binaries_but_discards_stamp(tmp_path: Path) -> None:
    context = _context()
    record_binary_context(tmp_path, context)
    orch_prebuild = _touch(tmp_path / "cache" / "orch_main.bin")
    orch_sidecar = _touch(tmp_path / "orchestration" / "main.so")

    assert prepare_binary_context(tmp_path, context) == 0
    assert orch_prebuild.exists()
    assert orch_sidecar.exists()
    assert not binary_context_path(tmp_path).exists()


def test_legacy_unstamped_cache_is_invalidated_once(tmp_path: Path) -> None:
    orch_prebuild = _touch(tmp_path / "cache" / "orch_main.bin")
    orch_sidecar = _touch(tmp_path / "orchestration" / "main.so")

    assert prepare_binary_context(tmp_path, _context()) == 2
    assert not orch_prebuild.exists()
    assert not orch_sidecar.exists()


@pytest.mark.parametrize(
    "changed_context",
    [
        _context(platform="a2a3"),
        _context(runtime_name="host_build_graph"),
        _context(pto_isa_revision="d" * 40),
    ],
)
def test_any_abi_context_change_invalidates_cache(
    tmp_path: Path,
    changed_context: BinaryCacheContext,
) -> None:
    record_binary_context(tmp_path, _context())
    binary = _touch(tmp_path / "cache" / "orch_main.bin")

    assert prepare_binary_context(tmp_path, changed_context) == 1
    assert not binary.exists()


def test_malformed_stamp_never_authorizes_binary_reuse(tmp_path: Path) -> None:
    stamp = binary_context_path(tmp_path)
    stamp.parent.mkdir(parents=True)
    stamp.write_text("{not json", encoding="utf-8")
    binary = _touch(tmp_path / "cache" / "orch_main.bin")

    assert prepare_binary_context(tmp_path, _context()) == 1
    assert not binary.exists()
    assert not stamp.exists()


def test_unknown_current_identity_never_authorizes_binary_reuse(tmp_path: Path) -> None:
    record_binary_context(tmp_path, _context())
    binary = _touch(tmp_path / "cache" / "orch_main.bin")

    assert prepare_binary_context(tmp_path, None) == 1
    assert not binary.exists()
    assert not binary_context_path(tmp_path).exists()
    record_binary_context(tmp_path, None)
    assert not binary_context_path(tmp_path).exists()


def test_invalidation_discards_stamp_before_removing_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context()
    stamp = binary_context_path(tmp_path)
    record_binary_context(tmp_path, context)

    def fail_artifact_removal(_work_dir: Path | str) -> int:
        assert not stamp.exists()
        raise RuntimeError("artifact removal failed")

    monkeypatch.setattr(binary_cache, "_invalidate_binary_artifacts", fail_artifact_removal)

    with pytest.raises(RuntimeError, match="artifact removal failed"):
        binary_cache.invalidate_binary_context(tmp_path)

    assert not stamp.exists()


def test_record_binary_context_writes_schema_and_cleans_temp_file(tmp_path: Path) -> None:
    context = _context()

    record_binary_context(tmp_path, context)

    stamp = binary_context_path(tmp_path)
    assert json.loads(stamp.read_text(encoding="utf-8")) == context.to_dict()
    assert not list(stamp.parent.glob(f"{stamp.name}.*.tmp"))


def test_dirty_runtime_checkout_disables_binary_reuse(device_runner, monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    pto_isa_root = tmp_path / "pto-isa"
    _init_git_repo(runtime_root)
    _init_git_repo(pto_isa_root)
    (runtime_root / "tracked.hpp").write_text("// locally modified\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "_task_interface",
        SimpleNamespace(__build_commit__=_RUNTIME_OLD),
    )

    context = device_runner._current_binary_context(
        SimpleNamespace(project_root=runtime_root),
        platform="a2a3sim",
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=str(pto_isa_root),
    )

    assert context is None


def test_dirty_pto_isa_checkout_disables_binary_reuse(device_runner, monkeypatch, tmp_path: Path) -> None:
    pto_isa_root = tmp_path / "pto-isa"
    _init_git_repo(pto_isa_root)
    (pto_isa_root / "tracked.hpp").write_text("// locally modified\n", encoding="utf-8")
    monkeypatch.setattr(device_runner, "_runtime_revision", Mock(return_value=_RUNTIME_OLD))

    context = device_runner._current_binary_context(
        object(),
        platform="a2a3sim",
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=str(pto_isa_root),
    )

    assert context is None


def test_non_git_pto_isa_root_does_not_use_pin_as_actual_identity(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A non-git root has no verifiable revision, so binary reuse must be off.

    The identity has to come from the checkout's own HEAD. Substituting the pin
    (what the tree is *supposed* to be) would let stale binaries survive.
    """
    pto_isa_root = tmp_path / "pto-isa"
    pto_isa_root.mkdir()
    monkeypatch.setattr(device_runner, "_runtime_revision", Mock(return_value=_RUNTIME_OLD))

    context = device_runner._current_binary_context(
        object(),
        platform="a2a3sim",
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=str(pto_isa_root),
    )

    assert context is None


def test_installed_runtime_uses_embedded_revision_with_clean_pto_isa(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_assets = tmp_path / "wheel-assets"
    runtime_assets.mkdir()
    pto_isa_root = tmp_path / "pto-isa"
    pto_isa_revision = _init_git_repo(pto_isa_root)
    monkeypatch.setitem(
        sys.modules,
        "_task_interface",
        SimpleNamespace(__build_commit__=_RUNTIME_OLD),
    )

    context = device_runner._current_binary_context(
        SimpleNamespace(project_root=runtime_assets),
        platform="a2a3sim",
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=str(pto_isa_root),
    )

    assert context == _context(pto_isa_revision=pto_isa_revision)


def _write_minimal_artifact(work_dir: Path) -> None:
    source = work_dir / "orchestration" / "main.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("// orchestration\n", encoding="utf-8")
    (work_dir / "kernel_config.py").write_text(
        "KERNELS = []\n"
        f"ORCHESTRATION = {{'source': {str(source)!r}, 'function_name': 'entry'}}\n"
        "RUNTIME_CONFIG = {'runtime': 'tensormap_and_ringbuffer'}\n",
        encoding="utf-8",
    )


def _stub_assembly(device_runner, monkeypatch, tmp_path: Path, chip_build: Mock) -> BinaryCacheContext:
    _write_minimal_artifact(tmp_path)
    context = _context()
    monkeypatch.setattr(device_runner, "ensure_pto_isa_root", Mock(return_value="/pto-isa"))
    monkeypatch.setattr(device_runner, "KernelCompiler", Mock(return_value=object()))
    monkeypatch.setattr(device_runner, "_current_binary_context", Mock(return_value=context))
    monkeypatch.setattr(
        device_runner,
        "_compile_single_orchestration",
        Mock(return_value=b"orchestration binary"),
    )
    monkeypatch.setattr(device_runner, "ChipCallable", SimpleNamespace(build=chip_build))
    monkeypatch.setattr(device_runner, "register_callable_identity", Mock())
    return context


def test_compile_and_assemble_records_context_after_success(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    chip = object()
    context = _stub_assembly(device_runner, monkeypatch, tmp_path, Mock(return_value=chip))
    record = Mock()
    monkeypatch.setattr(device_runner, "record_binary_context", record)

    assembled, _, _ = device_runner._compile_and_assemble(tmp_path, "a2a3sim")

    assert assembled is chip
    record.assert_called_once_with(tmp_path, context)


def test_compile_and_assemble_does_not_stamp_failed_assembly(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_assembly(device_runner, monkeypatch, tmp_path, Mock(side_effect=RuntimeError("assemble failed")))
    record = Mock()
    monkeypatch.setattr(device_runner, "record_binary_context", record)

    with pytest.raises(RuntimeError, match="assemble failed"):
        device_runner._compile_and_assemble(tmp_path, "a2a3sim")

    record.assert_not_called()


def test_failed_matching_cache_is_invalidated_before_retry(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context()
    record_binary_context(tmp_path, context)
    cached_binary = _touch(tmp_path / "cache" / "orch_main.bin", b"cached orchestration")
    chip = object()
    build = Mock(side_effect=[RuntimeError("assemble failed"), chip])
    _stub_assembly(device_runner, monkeypatch, tmp_path, build)
    compile_orchestration = device_runner._compile_single_orchestration

    with pytest.raises(RuntimeError, match="assemble failed"):
        device_runner._compile_and_assemble(tmp_path, "a2a3sim")

    assert cached_binary.exists()
    assert not binary_context_path(tmp_path).exists()
    compile_orchestration.assert_not_called()

    assembled, _, _ = device_runner._compile_and_assemble(tmp_path, "a2a3sim")

    assert assembled is chip
    assert not cached_binary.exists()
    compile_orchestration.assert_called_once()
    assert json.loads(binary_context_path(tmp_path).read_text(encoding="utf-8")) == context.to_dict()


def test_compile_and_assemble_serializes_same_work_dir(
    device_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_minimal_artifact(tmp_path)
    process_context = multiprocessing.get_context("fork")
    entered = process_context.Event()
    release = process_context.Event()

    def assemble_locked(_work_dir, _platform):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("assembly was not released")
        return object(), "runtime", {}

    monkeypatch.setattr(device_runner, "_compile_and_assemble_locked", assemble_locked)
    process = process_context.Process(
        target=device_runner._compile_and_assemble,
        args=(tmp_path, "a2a3sim"),
    )
    process.start()
    try:
        assert entered.wait(timeout=5)
        lock_path = tmp_path / "cache" / ".binary_context.lock"
        with lock_path.open("a+b") as lock_file:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        release.set()
        process.join(timeout=5)
        assert process.exitcode == 0

        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
