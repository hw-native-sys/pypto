# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct JIT eager calls: real compiler/metadata/cache, fake execution boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig
from pypto.runtime.kernel.context import KernelState


@pytest.fixture
def copy_kernel(monkeypatch, tmp_path):
    from pypto.backend import pto_backend  # noqa: PLC0415
    from pypto.jit import decorator  # noqa: PLC0415

    # Simulate only external PTOAS bytes. Keep real MLIR, wrappers/config,
    # compiler sidecars and cache publication instead of using source-only mode.
    monkeypatch.setattr(decorator, "_ptoas_available", lambda: True)
    monkeypatch.setattr(
        pto_backend,
        "_run_ptoas",
        lambda source, target, **kwargs: Path(target).write_text("// fake external PTOAS output\n"),
    )

    @pl.jit.incore
    def copy_core(x: pl.Tensor, out: pl.Out[pl.Tensor]):
        rows, cols = x.shape
        value = pl.load(x, [0, 0], [rows, cols])
        pl.store(value, [0, 0], out)
        return out

    @pl.jit
    def copy(x: pl.Tensor, step: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor]):
        out = copy_core(x, out)
        return out, out

    return copy


def test_uninitialized_fails_before_binding_or_compilation(eager_env, copy_kernel, monkeypatch):
    monkeypatch.setattr(copy_kernel, "_bind_args", lambda *a: pytest.fail("bound before init"))
    with pytest.raises(RuntimeError, match="pypto.torch.init"):
        copy_kernel()
    assert not eager_env.workers


def test_real_compiler_cache_callback_and_retirement(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    assert copy_kernel(x, 3, out) == (out, out)
    assert not env.runs and len(env.state._submissions) == 1
    artifact = next(iter(copy_kernel._cache.values()))
    assert isinstance(artifact, CompiledArtifact)
    assert artifact.execution_capabilities.modes == (ArtifactExecutionMode.KERNEL,)
    assert artifact.metadata is not None and artifact.metadata.return_aliases == (2, 2)
    metadata = json.loads((artifact.output_dir / "compiled_meta.json").read_text())
    assert metadata["supported_execution_modes"] == ["kernel"]
    assert (artifact.output_dir / "kernel_config.py").is_file()
    assert list(artifact.output_dir.glob("orchestration/*.cpp"))
    assert list(artifact.output_dir.glob("ptoas/*.pto"))
    env.run_queue()
    submitted = [e for e in env.events if isinstance(e, tuple) and e[0] == "submit"]
    assert submitted[0][4] == env.rts_stream != env.stream_id
    assert submitted[0][2].scalars[0].value == 3
    assert [t[1] for t in submitted[0][2].tensors] == ["IN", "OUT"]
    assert not env.runs[0].done and not env.state._submissions[0].done()
    copy_kernel(x, 9, out)
    env.run_queue()
    assert len(copy_kernel._cache) == 1 and env.events.count("register") == 1
    assert len(env.buffers) == 2  # Sources are reused across outstanding runs.
    assert "wait" not in env.events and "synchronize" not in env.events
    first, second = env.state._submissions
    env.runs[0].complete = True
    assert first.done() and not any(b.closed for b in env.buffers)
    env.runs[1].complete = True
    assert second.done() and all(b.closed for b in env.buffers)
    assert env.events.index("wait") < env.events.index("source.close")
    env.state.close()
    assert env.state.state is KernelState.CLOSED
    stub_device_runner._compile_and_assemble.assert_called_once()


def test_compile_keeps_program_contract_and_separate_cache(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    x, out = env.tensor(), env.tensor()
    options = CompileOptions(execution_mode=ExecutionMode.ONBOARD)
    program = copy_kernel.compile(x, 1, out, config=options)
    assert program.execution_capabilities.modes == (ArtifactExecutionMode.PROGRAM,)
    assert not env.workers
    env.init()
    copy_kernel(x, 1, out, config=options)
    assert len(copy_kernel._cache) == 2
    assert copy_kernel.compile(x, 2, out, config=options) is program
    env.state.close()  # Explicit close may synchronize the queued callback.


@pytest.mark.parametrize("warm", [False, True])
def test_capture_rejected_before_binding_even_when_prepared(
    eager_env, copy_kernel, stub_device_runner, monkeypatch, warm
):
    env = eager_env
    env.init()
    if warm:
        copy_kernel(env.tensor(), 1, env.tensor())
        env.state.drain()
    env.capture = True
    monkeypatch.setattr(copy_kernel, "_bind_args", lambda *a: pytest.fail("capture entered binding"))
    with pytest.raises(NotImplementedError, match="capture"):
        copy_kernel()
    env.capture = False
    env.state.close()


def test_fake_capable_capture_only_uses_prepared_entries(
    eager_env, copy_kernel, stub_device_runner, monkeypatch
):
    from pypto.torch import launch  # noqa: PLC0415

    env = eager_env
    env.init()
    monkeypatch.setattr(launch, "_capture_supported", lambda: True)
    x, out = env.tensor(), env.tensor()
    env.capture = True
    with pytest.raises(RuntimeError, match="not prepared"):
        copy_kernel(x, 1, out)
    assert not copy_kernel._cache and "register" not in env.events
    env.capture = False
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = True
    monkeypatch.setattr(copy_kernel, "_compile", lambda *a, **kw: pytest.fail("compiled during capture"))
    monkeypatch.setattr(
        env.state, "ensure_callable", lambda *a: pytest.fail("registered or loaded during capture")
    )
    assert copy_kernel(x, 2, out) == (out, out)
    with pytest.raises(RuntimeError, match="not prepared"):
        copy_kernel(env.tensor((32, 16)), 2, env.tensor((32, 16)))
    registrations = env.state._registrations.copy()
    env.state._registrations.clear()
    with pytest.raises(RuntimeError, match="not been prepared"):
        copy_kernel(x, 2, out)
    env.state._registrations.update(registrations)
    env.capture = False
    env.state.close()


@pytest.mark.parametrize("options", [RunConfig(), CompileOptions()])
def test_call_config_cannot_change_execution_mode(eager_env, copy_kernel, options):
    eager_env.init()
    with pytest.raises((TypeError, ValueError), match="CompileOptions|initialized"):
        copy_kernel(config=options)
    eager_env.state.close()


def test_invalid_arguments_fail_before_passes(eager_env, copy_kernel, monkeypatch):
    from pypto.ir.pass_manager import PassManager  # noqa: PLC0415

    eager_env.init()
    monkeypatch.setattr(PassManager, "run_passes", lambda *a: pytest.fail("invalid call compiled"))
    with pytest.raises(TypeError, match="NPU"):
        copy_kernel(torch.empty(16, 16), 1, torch.empty(16, 16))
    with pytest.raises(TypeError, match="out"):
        copy_kernel(eager_env.tensor(), 1)
    with pytest.raises(ValueError, match="INT32|int32"):
        copy_kernel(eager_env.tensor(), 1 << 40, eager_env.tensor())
    eager_env.state.close()


def test_registry_busy_is_preserved_without_drain(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    env.init()
    env.register_error = RuntimeError("registry busy")
    with pytest.raises(RuntimeError, match="registry busy"):
        copy_kernel(env.tensor(), 1, env.tensor())
    assert not env.runs and "wait" not in env.events and "synchronize" not in env.events
    env.state.close()


def test_compile_options_change_rebuilds(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    options = CompileOptions(execution_mode=ExecutionMode.ONBOARD)
    copy_kernel(x, 1, out, config=options)
    env.state.drain()
    copy_kernel(x, 1, out, config=replace(options, analyze_auto_scopes_for_deps=True))
    assert len(copy_kernel._cache) == 2
    env.state.close()


def test_persistent_kernel_build_restore_and_binary_validation(
    eager_env, copy_kernel, device_runner, monkeypatch, tmp_path
):
    from enum import Enum  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from pypto import _cache_config  # noqa: PLC0415
    from pypto._identity import ToolchainIdentity, digest_record  # noqa: PLC0415
    from pypto.jit import _persistent  # noqa: PLC0415
    from pypto.runtime import _callable_identity, _prebuilt  # noqa: PLC0415
    from pypto.runtime._binary_cache import BinaryCacheContext  # noqa: PLC0415

    env = eager_env
    native = env.api

    class Direction(Enum):
        IN = 1
        OUT = 2
        INOUT = 3
        SCALAR = 4

    native.ArgDirection = Direction
    native.CoreCallable = SimpleNamespace(build=lambda **kw: kw)
    native.ChipCallable = SimpleNamespace(build=lambda **kw: kw)
    monkeypatch.setattr(_prebuilt, "_native_callable_interface", lambda: native)
    monkeypatch.setattr(_callable_identity, "register_callable_identity", lambda *a: None)
    monkeypatch.setattr(device_runner, "register_callable_identity", lambda *a: None)
    monkeypatch.setattr(device_runner, "CoreCallable", native.CoreCallable)
    monkeypatch.setattr(device_runner, "ChipCallable", native.ChipCallable)
    monkeypatch.setattr(device_runner, "ensure_pto_isa_root", lambda: str(tmp_path / "isa"))
    monkeypatch.setattr(
        device_runner,
        "KernelCompiler",
        lambda **kw: SimpleNamespace(
            get_incore_include_dirs=lambda: [], get_kernel_include_dirs=lambda _: []
        ),
    )
    monkeypatch.setattr(
        device_runner,
        "_current_binary_context",
        lambda *a, **kw: BinaryCacheContext(
            platform="a2a3",
            runtime_name="tensormap_and_ringbuffer",
            runtime_revision="a" * 40,
            pto_isa_revision="b" * 40,
        ),
    )
    built = []

    def kernel_binary(kernel, *args, **kwargs):
        built.append(kernel)
        assert Path(kernel["source"]).is_file()
        return b"fake ELF", b"fake device bytes"

    monkeypatch.setattr(device_runner, "_compile_single_kernel", kernel_binary)
    monkeypatch.setattr(device_runner, "_compile_single_orchestration", lambda *a, **kw: b"fake orch bytes")
    identity = ToolchainIdentity(
        pypto=digest_record("pypto"),
        runtime=digest_record("runtime"),
        pto_isa=digest_record("pto_isa"),
        ptoas=digest_record("ptoas"),
        device_toolchain=digest_record("device_toolchain"),
    )
    monkeypatch.setattr(_persistent, "capture_toolchain", lambda *a: identity)
    monkeypatch.setattr(
        _cache_config, "_DEFAULT_CONFIG", _cache_config.CacheConfig(enabled=True, root=tmp_path / "cache")
    )
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    artifact = next(iter(copy_kernel._artifact_objects.values()))
    handle = artifact._artifact_runtime.handle
    assert handle.key.contract is ArtifactExecutionMode.KERNEL
    assert handle.spec.execution_capabilities.modes == (ArtifactExecutionMode.KERNEL,)
    from pypto.jit._artifact_manifest import ArtifactState  # noqa: PLC0415

    assert handle.spec.state is ArtifactState.BINARY_READY
    assert len(built) == 1
    from pypto.torch import launch  # noqa: PLC0415

    monkeypatch.setattr(launch, "_capture_supported", lambda: True)
    env.capture = True
    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(copy_kernel, "_compile", lambda *a, **kw: pytest.fail("capture compiled"))
        capture_patch.setattr(
            env.state, "ensure_callable", lambda *a: pytest.fail("capture registered or loaded")
        )
        copy_kernel(x, 2, out)
        copy_kernel._artifact_objects.clear()
        with pytest.raises(RuntimeError, match="not prepared"):
            copy_kernel(x, 3, out)  # A disk hit must not load/restore during capture.
    env.capture = False
    copy_kernel._artifact_objects.clear()
    copy_kernel(x, 7, out)
    restored = next(iter(copy_kernel._artifact_objects.values()))
    assert isinstance(restored, CompiledArtifact) and restored is not artifact
    restored.load()
    assert len(built) == 1 and env.events.count("register") == 1
    assert restored.metadata == artifact.metadata
    env.state.close()
    # Existing public validation must still catch corrupted device bytes.
    manifest = json.loads((handle.directory / "artifact_manifest.json").read_text())
    binary = next(item["path"] for item in manifest["files"] if item["path"].endswith(".bin"))
    (handle.directory / binary).write_bytes(b"corrupt")
    from pypto.jit._artifact_manifest import read_manifest  # noqa: PLC0415

    with pytest.raises(ValueError, match="does not match the request or payload"):
        read_manifest(handle.directory, handle.key, handle.spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
