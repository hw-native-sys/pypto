# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Artifact promotion and read-only reconstruction without device dependencies."""

import importlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pypto.language as pl
import pytest
from pypto import ir
from pypto._identity import ToolchainIdentity, digest_record
from pypto.ir.compiled_program import _COMPILED_META_SCHEMA, CompiledProgram
from pypto.ir.distributed_compiled_program import _META_SCHEMA, DistributedCompiledProgram
from pypto.jit import _artifact_manifest
from pypto.jit._artifact_manifest import ArtifactKey, ArtifactSpec, ArtifactState, BuildKind
from pypto.jit.artifact_cache import ArtifactStore, BuildDisposition, LookupStatus
from pypto.runtime import _prebuilt
from pypto.runtime._artifact_runtime import ArtifactRuntime, bind_artifact, restore_artifact
from pypto.runtime._artifact_sources import (
    UnsupportedArtifactInput,
    package_generated_sources,
    read_kernel_config,
)
from pypto.runtime.distributed_runner import (
    _assemble_chip_callables,
    _load_generated_module,
    _write_dispatch_name_map,
)
from pypto.runtime.runner import DfxOptions, _execute_compiled, _write_name_map


class Direction(Enum):
    SCALAR = 0
    IN = 1
    OUT = 2
    INOUT = 3


def _key():
    identity = ToolchainIdentity(
        pypto=digest_record("pypto"),
        runtime=digest_record("runtime"),
        pto_isa=digest_record("pto_isa"),
        ptoas=digest_record("ptoas"),
        device_toolchain=digest_record("device_toolchain"),
    )
    return ArtifactKey(identity, digest_record("source"), digest_record("spec"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _chip(root: Path) -> None:
    _write(root / "kernels/kernel.cpp", "// kernel")
    _write(root / "orchestration/main.cpp", "// orchestration")
    _write(
        root / "kernel_config.py",
        "\n".join(
            [
                "from pathlib import Path",
                "from simpler.task_interface import ArgDirection as D",
                "ROOT = Path(__file__).parent",
                "KERNELS = [dict(func_id=7, name='kernel', core_type='aiv', "
                "source=str(ROOT / 'kernels/kernel.cpp'), signature=[D.IN, D.OUT])]",
                "ORCHESTRATION = dict(function_name='entry', source=str(ROOT / 'orchestration/main.cpp'), "
                "signature=[D.INOUT])",
                "RUNTIME_CONFIG = dict(runtime='test_runtime', enable_sdma=True, aicpu_thread_num=2)",
            ]
        ),
    )


def _generated(root: Path, kind: BuildKind) -> None:
    meta: dict[str, Any] = dict(
        schema=_COMPILED_META_SCHEMA,
        params=[],
        num_return_types=0,
        platform="a2a3sim",
        backend_type="Ascend910B",
    )
    if kind is BuildKind.SINGLE_CHIP:
        _chip(root)
        _write(root / "compiled_meta.json", json.dumps(meta))
    else:
        for name in ("left", "right"):
            _chip(root / "next_levels" / name)
        meta.update(
            schema=_META_SCHEMA,
            distributed_config=dict(
                device_ids=[0, 1],
                num_sub_workers=0,
                runtime="test_runtime",
                aicpu_thread_num=2,
            ),
        )
        _write(root / "distributed_meta.json", json.dumps(meta))
        _write(
            root / "orchestration/host_orch.py", "def entry(): pass\nentry._pypto_distributed_entry = True"
        )


def _spec(kind=BuildKind.SINGLE_CHIP):
    metadata = "compiled_meta.json" if kind is BuildKind.SINGLE_CHIP else "distributed_meta.json"
    configs = (
        ("kernel_config.py",)
        if kind is BuildKind.SINGLE_CHIP
        else ("next_levels/left/kernel_config.py", "next_levels/right/kernel_config.py")
    )
    return ArtifactSpec(ArtifactState.GENERATED, kind, (metadata, *configs))


@pytest.fixture
def fake_runtime(monkeypatch):
    task_interface: Any = ModuleType("simpler.task_interface")
    task_interface.ArgDirection = Direction
    task_interface.CoreCallable = SimpleNamespace(build=Mock(side_effect=lambda **kwargs: kwargs))
    task_interface.ChipCallable = SimpleNamespace(build=Mock(side_effect=lambda **kwargs: kwargs))
    simpler: Any = ModuleType("simpler")
    simpler.__path__ = []
    simpler.task_interface = task_interface
    monkeypatch.setitem(sys.modules, "simpler", simpler)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", task_interface)
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", task_interface)
    runner: Any = ModuleType("pypto.runtime.device_runner")
    runner.register_callable_identity = Mock()
    monkeypatch.setattr("pypto.runtime._callable_identity.register_callable_identity", Mock())
    runner._execute_on_device = Mock()

    def compile_(root, platform, *, save_prebuilt=False):
        assert save_prebuilt
        config = read_kernel_config(root / "kernel_config.py")
        _prebuilt.write_chip_binaries(
            root,
            platform,
            config.ORCHESTRATION,
            [(k, b"kernel bytes") for k in config.KERNELS],
            b"orchestration bytes",
            "test_runtime",
            config.RUNTIME_CONFIG,
        )

    runner._compile_and_assemble = Mock(side_effect=compile_)
    monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", runner)
    return SimpleNamespace(runner=runner, interface=task_interface)


def _publish(tmp_path, kind):
    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    result = store.get_or_build(_key(), _spec(kind), lambda root: _generated(root, kind))
    assert result.disposition is BuildDisposition.PUBLISHED and result.handle is not None
    return store, result.handle


@pytest.mark.parametrize("kind", list(BuildKind))
def test_promote_all_children_and_restore_readonly(tmp_path, fake_runtime, monkeypatch, kind):
    store, generated = _publish(tmp_path, kind)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "runs")
    chips = runtime.load()
    expected = 1 if kind is BuildKind.SINGLE_CHIP else 2
    assert len(chips) == expected
    assert fake_runtime.runner._compile_and_assemble.call_count == expected
    assert runtime.handle.spec.state is ArtifactState.BINARY_READY
    assert runtime.load() is chips
    assert store.lookup(generated.key, generated.spec).status is LookupStatus.HIT
    # All compilation and source access becomes forbidden for ready restoration.
    fake_runtime.runner._compile_and_assemble.side_effect = AssertionError("unexpected compilation")
    monkeypatch.setattr(
        "pypto.runtime._artifact_sources.read_kernel_config", Mock(side_effect=AssertionError)
    )
    monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", None)
    monkeypatch.setitem(sys.modules, "pypto.runtime.kernel_compiler", None)
    monkeypatch.setitem(sys.modules, "simpler_setup", None)
    readonly = ArtifactStore(store.root, readonly=True)
    restored = restore_artifact(readonly, runtime.handle, tmp_path / "readonly-runs")
    assert restored.program is None
    assert restored._artifact_runtime.load() == chips
    assert not list(store.root.rglob("__pycache__"))
    assert not (tmp_path / "readonly-runs").exists()


@pytest.mark.parametrize("kind", list(BuildKind))
def test_jit_warmup_promotes_and_reuses_readonly_ready(tmp_path, fake_runtime, monkeypatch, kind):
    store, generated = _publish(tmp_path, kind)
    compiled = restore_artifact(store, generated, tmp_path / "runs")

    @pl.jit
    def kernel():
        pass

    # Automatic JIT artifact lookup remains separate; exercise the explicit
    # adapter when it supplies the compiled object to the public warmup path.
    monkeypatch.setattr(kernel, "compile", lambda *args, **kwargs: compiled)
    monkeypatch.setitem(sys.modules, "simpler.worker", None)
    assert kernel.warmup() is compiled
    runtime = compiled._artifact_runtime
    assert runtime is not None
    assert runtime.handle.spec.state is ArtifactState.BINARY_READY
    expected = 1 if kind is BuildKind.SINGLE_CHIP else 2
    assert fake_runtime.runner._compile_and_assemble.call_count == expected
    assert len(runtime.load()) == expected
    assert kernel.warmup() is compiled
    assert fake_runtime.runner._compile_and_assemble.call_count == expected
    fake_runtime.runner._execute_on_device.assert_not_called()

    monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", None)
    monkeypatch.setitem(sys.modules, "pypto.runtime.kernel_compiler", None)
    monkeypatch.setitem(sys.modules, "simpler_setup", None)
    readonly = ArtifactStore(store.root, readonly=True)
    compiled = restore_artifact(readonly, runtime.handle, tmp_path / "readonly-runs")
    before = {path: path.read_bytes() for path in store.root.rglob("*") if path.is_file()}
    assert kernel.warmup() is compiled
    assert compiled.program is None
    assert len(compiled._artifact_runtime.load()) == expected
    after = {path: path.read_bytes() for path in store.root.rglob("*") if path.is_file()}
    assert after == before
    assert not (tmp_path / "readonly-runs").exists()


def test_ready_relocation_does_not_need_original_sources(tmp_path, fake_runtime):
    store, generated = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "run")
    runtime.load()
    relocated = tmp_path / "relocated"
    shutil.copytree(runtime.directory, relocated)
    shutil.rmtree(store.root)
    fake_runtime.runner._compile_and_assemble.side_effect = AssertionError("compiler called")
    result = _prebuilt.load_prebuilt(relocated, "a2a3sim", BuildKind.SINGLE_CHIP)
    assert result["."][0]["children"][0][0] == 7
    assert result["."][0]["signature"] == [Direction.INOUT]


@pytest.mark.parametrize(
    "damage", ["bytes", "missing", "size", "path", "symlink", "duplicate", "signature", "platform"]
)
def test_invalid_prebuilt_fails_before_any_callable(tmp_path, fake_runtime, damage):
    _chip(tmp_path)
    _prebuilt.prepare_prebuilt(tmp_path, "a2a3sim", BuildKind.SINGLE_CHIP)
    manifest = tmp_path / _prebuilt.BINARY_MANIFEST
    data = json.loads(manifest.read_text())
    kernel = data["kernels"][0]
    binary = tmp_path / kernel["binary"]["path"]
    if damage == "bytes":
        binary.write_bytes(b"x" * binary.stat().st_size)
    elif damage == "missing":
        binary.unlink()
    elif damage == "size":
        kernel["binary"]["size"] = True
    elif damage == "path":
        kernel["binary"]["path"] = "../outside.bin"
    elif damage == "symlink":
        outside = tmp_path / "outside.bin"
        binary.rename(outside)
        binary.symlink_to(outside)
    elif damage == "duplicate":
        data["kernels"].append(dict(kernel))
    elif damage == "signature":
        kernel["signature"] = ["__dict__"]
    else:
        data["platform"] = "a5"
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        _prebuilt.load_prebuilt(tmp_path, "a2a3sim", BuildKind.SINGLE_CHIP)
    fake_runtime.interface.CoreCallable.build.assert_not_called()
    fake_runtime.interface.ChipCallable.build.assert_not_called()


def test_distributed_manifest_must_cover_every_child(tmp_path, fake_runtime):
    _generated(tmp_path, BuildKind.DISTRIBUTED)
    _prebuilt.prepare_prebuilt(tmp_path, "a2a3sim", BuildKind.DISTRIBUTED)
    manifest = tmp_path / _prebuilt.BINARY_MANIFEST
    data = json.loads(manifest.read_text())
    data["chips"].pop()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="all chip"):
        _prebuilt.load_prebuilt(tmp_path, "a2a3sim", BuildKind.DISTRIBUTED)
    fake_runtime.interface.ChipCallable.build.assert_not_called()


def test_failed_child_never_publishes_ready_and_can_retry(tmp_path, fake_runtime):
    store, handle = _publish(tmp_path, BuildKind.DISTRIBUTED)
    runtime = ArtifactRuntime(store, handle, "a2a3sim", tmp_path / "runs")
    compile_ = fake_runtime.runner._compile_and_assemble.side_effect

    def fail(root, platform, **kwargs):
        if root.name == "right":
            raise OSError("compiler failure")
        compile_(root, platform, **kwargs)

    fake_runtime.runner._compile_and_assemble.side_effect = fail
    with pytest.raises(OSError, match="compiler failure"):
        runtime.load()
    assert not list(store.root.rglob("ready"))
    assert store.lookup(handle.key, handle.spec).status is LookupStatus.HIT
    fake_runtime.runner._compile_and_assemble.side_effect = compile_
    assert len(runtime.load()) == 2


@pytest.mark.parametrize("readonly", [False, True])
def test_storage_failure_retains_usable_private_output(tmp_path, fake_runtime, monkeypatch, readonly):
    store, handle = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    store = ArtifactStore(store.root, private_root=tmp_path / "fallback", readonly=readonly)
    if not readonly:
        monkeypatch.setattr(
            "pypto.jit.artifact_cache._rename_noreplace", Mock(side_effect=OSError("storage failure"))
        )
    runtime = ArtifactRuntime(store, handle, "a2a3sim", tmp_path / "runs")
    result = runtime.load()
    assert runtime.directory.is_relative_to(tmp_path / "fallback")
    assert runtime.load() is result
    assert fake_runtime.runner._compile_and_assemble.call_count == 1
    assert (runtime.directory / _prebuilt.BINARY_MANIFEST).is_file()


def test_concurrent_runtime_loads_compile_once(tmp_path, fake_runtime):
    store, handle = _publish(tmp_path, BuildKind.DISTRIBUTED)
    runtimes = [ArtifactRuntime(store, handle, "a2a3sim", tmp_path / f"run{i}") for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda runtime: runtime.load(), runtimes))
    assert all(result == results[0] for result in results)
    assert fake_runtime.runner._compile_and_assemble.call_count == 2


def test_binding_retains_ir_and_path_hash_during_promotion(tmp_path, fake_runtime):
    store, handle = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    compiled = CompiledProgram.from_dir(handle.directory)
    program = ir.Program([], "Retained", ir.Span.unknown())
    compiled._program = program
    bind_artifact(compiled, store, handle, tmp_path / "runs")
    before = hash(compiled)
    compiled.load()
    assert hash(compiled) == before
    assert compiled.program is program
    assert compiled.output_dir == handle.directory
    assert compiled._artifact_runtime is not None
    assert compiled._artifact_runtime.handle.spec.state is ArtifactState.BINARY_READY


def test_runtime_outputs_must_be_outside_cache(tmp_path, fake_runtime):
    store, handle = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    with pytest.raises(ValueError, match="outside"):
        ArtifactRuntime(store, handle, "a2a3sim", store.root / "runs")


def test_generated_python_loader_never_creates_bytecode(tmp_path):
    path = tmp_path / "host_orch.py"
    path.write_text("def entry(): return 7")
    assert _load_generated_module(path).entry() == 7
    assert not list(tmp_path.rglob("*.pyc"))


def test_extern_packaging_preserves_nested_includes_after_relocation(tmp_path, fake_runtime):
    root = tmp_path / "generated"
    _chip(root)
    external = tmp_path / "external"
    _write(external / "src/kernel.cpp", '#include "../include/local.hpp"\n')
    _write(external / "include/local.hpp", "#include <other.hpp>\n")
    _write(external / "extra/other.hpp", "// external header")
    config = root / "kernel_config.py"
    with config.open("a") as stream:
        stream.write(
            f"\nKERNELS[0].update(external=True, source={str(external / 'src/kernel.cpp')!r}, "
            f"extra_include_dirs=[{str(external / 'extra')!r}])\n"
        )
    package_generated_sources(root, BuildKind.SINGLE_CHIP)
    relocated = tmp_path / "relocated"
    shutil.move(root, relocated)
    shutil.rmtree(external)
    config = read_kernel_config(relocated / "kernel_config.py")
    source = Path(config.KERNELS[0]["source"])
    assert (source.parent / "../include/local.hpp").is_file()
    assert (Path(config.KERNELS[0]["extra_include_dirs"][0]) / "other.hpp").is_file()
    _prebuilt.prepare_prebuilt(relocated, "a2a3sim", BuildKind.SINGLE_CHIP)


@pytest.mark.parametrize("include", ["#include HEADER", '#include "/outside.h"', '#include "missing.h"'])
def test_unsupported_extern_include_fails_closed(tmp_path, fake_runtime, include):
    _chip(tmp_path)
    _write(tmp_path / "external.cpp", include)
    with (tmp_path / "kernel_config.py").open("a") as stream:
        stream.write(f"\nKERNELS[0].update(external=True, source={str(tmp_path / 'external.cpp')!r})\n")
    with pytest.raises(UnsupportedArtifactInput):
        package_generated_sources(tmp_path, BuildKind.SINGLE_CHIP)


@pytest.mark.parametrize("failure", [False, True])
def test_attached_execution_uses_ready_bytes_and_never_retries(tmp_path, fake_runtime, monkeypatch, failure):
    store, generated = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "runs")
    runtime.load()
    headers = Mock(side_effect=AssertionError("header rewrite"))
    monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "_ensure_orchestration_headers", headers)
    fake_runtime.runner._compile_and_assemble.reset_mock()
    fake_runtime.runner._compile_and_assemble.side_effect = AssertionError("compilation on ready hit")
    execute = fake_runtime.runner._execute_on_device
    if failure:
        execute.side_effect = RuntimeError("device failure")
        with pytest.raises(RuntimeError, match="device failure"):
            _execute_compiled(
                generated.directory, [], platform="a2a3sim", device_id=0, artifact_runtime=runtime
            )
    else:
        _execute_compiled(
            generated.directory,
            [],
            platform="a2a3sim",
            device_id=0,
            artifact_runtime=runtime,
            dfx=DfxOptions(enable_dump_args=True),
        )
        assert execute.call_args.kwargs["output_prefix"] == str(tmp_path / "runs/dfx_outputs")
    execute.assert_called_once()
    headers.assert_not_called()
    fake_runtime.runner._compile_and_assemble.assert_not_called()
    assert not (store.root / "dfx_outputs").exists()


def test_distributed_assembly_uses_attached_runtime(tmp_path, fake_runtime):
    store, generated = _publish(tmp_path, BuildKind.DISTRIBUTED)
    compiled = restore_artifact(store, generated, tmp_path / "runs")
    chips, runtime_name, sdma = _assemble_chip_callables(compiled)
    assert set(chips) == {"left", "right"}
    assert runtime_name == "test_runtime" and sdma
    assert compiled._artifact_runtime is not None
    assert compiled._artifact_runtime.handle.spec.state is ArtifactState.BINARY_READY


def test_ready_diagnostic_labels_do_not_execute_config(tmp_path, fake_runtime, monkeypatch):
    _chip(tmp_path / "ready")
    _prebuilt.prepare_prebuilt(tmp_path / "ready", "a2a3sim", BuildKind.SINGLE_CHIP)
    (tmp_path / "ready/kernel_config.py").write_text("raise AssertionError('config executed')")
    run = tmp_path / "run"
    run.mkdir()
    # The ready label path must not import the optional compiler-tool package.
    monkeypatch.setitem(sys.modules, "simpler_setup.tools.swimlane_converter", None)
    for name_map in (
        _write_name_map(tmp_path / "ready", run, prebuilt=True),
        _write_dispatch_name_map(run, tmp_path / "ready", {}, prebuilt=True),
    ):
        assert name_map is not None and name_map.parent == run
        assert json.loads(name_map.read_text())["callable_id_to_name"] == {"7": "kernel"}
    assert not list((tmp_path / "ready").rglob("*.pyc"))


def test_ready_spec_enumerates_all_child_binaries(tmp_path, fake_runtime):
    store, generated = _publish(tmp_path, BuildKind.DISTRIBUTED)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "run")
    runtime.load()
    required = set(runtime.handle.spec.required_files)
    assert "binary_manifest.json" in required
    for name in ("left", "right"):
        for file in ("binary_manifest.json", "prebuilt/kernel_0.bin", "prebuilt/orchestration.bin"):
            assert f"next_levels/{name}/{file}" in required


def test_altered_handle_fails_before_source_execution(tmp_path, fake_runtime):
    store, handle = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    runtime = ArtifactRuntime(store, handle, "a2a3sim", tmp_path / "run")
    (handle.directory / "kernel_config.py").write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="manifest"):
        runtime.load()
    fake_runtime.runner._compile_and_assemble.assert_not_called()


def test_distributed_missing_declared_child_config_cannot_be_published(tmp_path, fake_runtime):
    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")

    def incomplete(root):
        _generated(root, BuildKind.DISTRIBUTED)
        (root / "next_levels/right/kernel_config.py").unlink()

    with pytest.raises(ValueError, match="missing required files.*right/kernel_config"):
        store.get_or_build(_key(), _spec(BuildKind.DISTRIBUTED), incomplete)
    fake_runtime.runner._compile_and_assemble.assert_not_called()


def test_distributed_auxiliary_directories_are_not_chip_builds(tmp_path, fake_runtime):
    def generated(root):
        _generated(root, BuildKind.DISTRIBUTED)
        _write(root / "next_levels/scratch/notes.txt", "auxiliary data")
        package_generated_sources(root, BuildKind.DISTRIBUTED)

    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    result = store.get_or_build(_key(), _spec(BuildKind.DISTRIBUTED), generated)
    assert result.handle is not None
    runtime = ArtifactRuntime(store, result.handle, "a2a3sim", tmp_path / "run")
    assert set(runtime.load()) == {"left", "right"}
    assert fake_runtime.runner._compile_and_assemble.call_count == 2
    assert (runtime.directory / "next_levels/scratch/notes.txt").is_file()


@pytest.mark.parametrize("include_dirs", ["empty", "none"])
def test_packaged_extern_promotes_after_store_drops_empty_directories(tmp_path, fake_runtime, include_dirs):
    external = tmp_path / "extern-source"
    _write(external / "kernel.cpp", "// external kernel")
    empty = external / "empty"
    empty.mkdir()
    configured = [str(empty)] if include_dirs == "empty" else None

    def generated(root):
        _generated(root, BuildKind.SINGLE_CHIP)
        with (root / "kernel_config.py").open("a") as stream:
            stream.write(
                f"\nKERNELS[0].update(external=True, source={str(external / 'kernel.cpp')!r}, "
                f"extra_include_dirs={configured!r})\n"
            )
        package_generated_sources(root, BuildKind.SINGLE_CHIP)

    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    result = store.get_or_build(_key(), _spec(), generated)
    assert result.handle is not None
    shutil.rmtree(external)
    runtime = ArtifactRuntime(store, result.handle, "a2a3sim", tmp_path / "run")
    assert runtime.load()["."][1] == "test_runtime"
    assert runtime.handle.spec.state is ArtifactState.BINARY_READY


@pytest.mark.parametrize("link_kind", ["header", "include_dir", "parent_traversal"])
def test_extern_symlinks_fail_before_generated_publication(tmp_path, fake_runtime, link_kind):
    external = tmp_path / "extern-source"
    _write(external / "kernel.cpp", '#include "alias/header.hpp"')
    _write(external / "real/header.hpp", "// header")
    if link_kind == "parent_traversal":
        (external / "real/deep").mkdir()
        (external / "alias").symlink_to(external / "real/deep", target_is_directory=True)
        _write(external / "header.hpp", "// wrong lexical parent")
        _write(external / "kernel.cpp", '#include "alias/../header.hpp"')
    elif link_kind == "include_dir":
        (external / "alias").symlink_to(external / "real", target_is_directory=True)
    else:
        (external / "alias").mkdir()
        (external / "alias/header.hpp").symlink_to(external / "real/header.hpp")

    def generated(root):
        _generated(root, BuildKind.SINGLE_CHIP)
        with (root / "kernel_config.py").open("a") as stream:
            stream.write(f"\nKERNELS[0].update(external=True, source={str(external / 'kernel.cpp')!r})\n")
        package_generated_sources(root, BuildKind.SINGLE_CHIP)

    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    with pytest.raises(UnsupportedArtifactInput, match="Symbolic links in extern inputs"):
        store.get_or_build(_key(), _spec(), generated)
    assert store.lookup(_key(), _spec()).status is LookupStatus.MISS


@pytest.mark.parametrize("kind", list(BuildKind))
@pytest.mark.parametrize("attach", ["restore", "bind"])
def test_ready_attachment_hashes_each_payload_once(tmp_path, fake_runtime, kind, attach):
    store, generated = _publish(tmp_path, kind)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "run")
    runtime.load()
    handle = runtime.handle
    cls = CompiledProgram if kind is BuildKind.SINGLE_CHIP else DistributedCompiledProgram
    compiled = cls.from_dir(handle.directory) if attach == "bind" else None
    with (
        patch.object(_artifact_manifest, "_file_digest", wraps=_artifact_manifest._file_digest) as digest,
        patch.object(cls, "from_dir", wraps=cls.from_dir) as from_dir,
        patch.object(_prebuilt.hashlib, "sha256", wraps=_prebuilt.hashlib.sha256) as sha,
    ):
        if attach == "restore":
            compiled = restore_artifact(store, handle, tmp_path / "restored-run")
        else:
            bind_artifact(compiled, store, handle, tmp_path / "bound-run")
        assert compiled is not None and compiled._artifact_runtime is not None
        payload_hashes = sha.call_count
        compiled._artifact_runtime.load()
        compiled._artifact_runtime.load()
        paths = [call.args[0] for call in digest.call_args_list]
        expected = {
            p for p in handle.directory.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"
        }
        assert set(paths) == expected and len(paths) == len(expected)
        # No additional SHA pass over binaries while reconstructing callables.
        assert sha.call_count == payload_hashes
        from_dir.assert_called_once_with(handle.directory)


@pytest.mark.parametrize("damage", ["payload", "inner_digest"])
def test_ready_attachment_rejects_corruption_before_callables(tmp_path, fake_runtime, damage):
    store, generated = _publish(tmp_path, BuildKind.SINGLE_CHIP)
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "run")
    runtime.load()
    handle = runtime.handle
    if damage == "payload":
        (handle.directory / "prebuilt/kernel_0.bin").write_bytes(b"wrong kernel")
    else:
        # Even an outer inventory certifying these bytes cannot override a
        # contradictory digest in the binary loader's inner contract.
        marker = handle.directory / _prebuilt.BINARY_MANIFEST
        record = json.loads(marker.read_bytes())
        record["kernels"][0]["binary"]["sha256"] = "0" * 64
        marker.write_text(json.dumps(record))
        outer = _artifact_manifest.make_manifest(handle.directory, handle.key, handle.spec)
        (handle.directory / "artifact_manifest.json").write_bytes(_artifact_manifest.encode_manifest(outer))
    fake_runtime.interface.CoreCallable.build.reset_mock()
    fake_runtime.interface.ChipCallable.build.reset_mock()
    with pytest.raises(ValueError):
        restored = restore_artifact(store, handle, tmp_path / "restored-run")
        restored.load()
    fake_runtime.interface.CoreCallable.build.assert_not_called()
    fake_runtime.interface.ChipCallable.build.assert_not_called()


@pytest.mark.parametrize("with_includes", [False, True])
def test_extern_workspace_ancestor_symlink_is_relocatable(tmp_path, fake_runtime, with_includes):
    workspace = tmp_path / "workspace"
    _write(workspace / "extern/src/kernel.cpp", '#include "../include/header.hpp"')
    _write(workspace / "extern/include/header.hpp", "// header")
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    root = tmp_path / "generated"
    _chip(root)
    source = alias / "extern/src/kernel.cpp"
    includes = [str(alias / "extern/include")] if with_includes else []
    with (root / "kernel_config.py").open("a") as stream:
        stream.write(
            f"\nKERNELS[0].update(external=True, source={str(source)!r}, extra_include_dirs={includes!r})\n"
        )
    package_generated_sources(root, BuildKind.SINGLE_CHIP)
    shutil.rmtree(workspace)
    config = read_kernel_config(root / "kernel_config.py")
    packaged = Path(config.KERNELS[0]["source"])
    assert (packaged.parent / "../include/header.hpp").read_bytes() == b"// header"


@pytest.mark.parametrize(
    "source",
    [
        b'/*\n#include MACRO\n*/\n#include "header.hpp"\n',
        b'#if 0\n#include "gone.h"\n#endif\n#include "header.hpp"\n',
        b'// caf\xe9\n#include "header.hpp"\n',
    ],
)
def test_extern_scanning_preserves_original_bytes(tmp_path, fake_runtime, source):
    root = tmp_path / "generated"
    _chip(root)
    external = tmp_path / "external"
    external.mkdir()
    (external / "kernel.cpp").write_bytes(source)
    (external / "header.hpp").write_bytes(b"// caf\xe9")
    with (root / "kernel_config.py").open("a") as stream:
        stream.write(f"\nKERNELS[0].update(external=True, source={str(external / 'kernel.cpp')!r})\n")
    package_generated_sources(root, BuildKind.SINGLE_CHIP)
    shutil.rmtree(external)
    packaged = Path(read_kernel_config(root / "kernel_config.py").KERNELS[0]["source"])
    assert packaged.read_bytes() == source
    assert (packaged.parent / "header.hpp").read_bytes() == b"// caf\xe9"


@pytest.mark.parametrize("kind", list(BuildKind))
def test_ready_publication_prunes_all_chip_caches_and_sidecars(tmp_path, fake_runtime, kind):
    store, generated = _publish(tmp_path, kind)
    compile_ = fake_runtime.runner._compile_and_assemble.side_effect

    def with_legacy_outputs(root, platform, **kwargs):
        compile_(root, platform, **kwargs)
        _write(root / "cache/.binary_context.lock", "")
        _write(root / "cache/binary_context.json", "{}")
        _write(root / "cache/kernel.bin", "duplicate kernel")
        _write(root / "kernels/kernel.o", "intermediate kernel")
        _write(root / "kernels/kernel.so", "duplicate kernel")
        _write(root / "orchestration/main.so", "duplicate orchestration")
        _write(root / "extern/input.o", "extern input must survive")

    fake_runtime.runner._compile_and_assemble.side_effect = with_legacy_outputs
    runtime = ArtifactRuntime(store, generated, "a2a3sim", tmp_path / "run")
    runtime.load()
    for chip in _prebuilt.chip_directories(runtime.directory, kind).values():
        assert not (chip / "cache").exists()
        assert not (chip / "kernels/kernel.o").exists()
        assert not (chip / "kernels/kernel.so").exists()
        assert not (chip / "orchestration/main.so").exists()
        assert (chip / "extern/input.o").read_text() == "extern input must survive"
        assert (chip / "kernels/kernel.cpp").is_file()
        assert (chip / "orchestration/main.cpp").is_file()
    assert store.lookup(runtime.handle.key, runtime.handle.spec).status is LookupStatus.HIT


def test_ready_spec_drops_declared_legacy_outputs_but_retains_sources(tmp_path, fake_runtime):
    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    legacy = ("cache/binary_context.json", "kernels/kernel.o", "orchestration/main.so")
    source = "kernels/kernel.cpp"
    spec = ArtifactSpec(
        ArtifactState.GENERATED, BuildKind.SINGLE_CHIP, (*_spec().required_files, *legacy, source)
    )

    def generated(root):
        _generated(root, BuildKind.SINGLE_CHIP)
        for name in legacy:
            _write(root / name, "inherited output")

    built = store.get_or_build(_key(), spec, generated)
    assert built.handle is not None
    runtime = ArtifactRuntime(store, built.handle, "a2a3sim", tmp_path / "run")
    runtime.load()
    assert not set(legacy).intersection(runtime.handle.spec.required_files)
    assert source in runtime.handle.spec.required_files
    assert (runtime.directory / source).is_file()
    assert store.lookup(runtime.handle.key, runtime.handle.spec).status is LookupStatus.HIT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
