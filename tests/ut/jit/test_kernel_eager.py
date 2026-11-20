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
from pypto.backend import BackendType
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.pypto_core import passes
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


def test_bool_tensors_fail_before_compilation_and_packing(eager_env, copy_kernel, monkeypatch):
    from pypto.runtime.kernel import arguments  # noqa: PLC0415

    env = eager_env
    env.init()
    # Exercise the production converter too: the shared fixture normally stubs
    # runtime dtype conversion, which previously hid the missing BOOL mapping.
    monkeypatch.setattr(arguments, "_tensor_dtype", env.tensor_dtype)
    with pytest.raises(ValueError, match=r"do not support torch.bool tensors"):
        arguments._tensor_dtype(torch.bool)
    with pytest.raises(ValueError, match=r"do not support torch.bool tensors"):
        copy_kernel(env.tensor(dtype=torch.bool), 1, env.tensor(dtype=torch.bool))
    assert not copy_kernel._cache and not env.buffers and not env.queue
    assert "register" not in env.events
    env.state.close()


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
    config_source = (artifact.output_dir / "kernel_config.py").read_text()
    assert '"signature": [_D.IN, _D.OUT, _D.SCALAR]' in config_source
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


def test_capture_uses_only_eager_warmed_entries(eager_env, copy_kernel, stub_device_runner, monkeypatch):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    env.capture = 42
    with pytest.raises(RuntimeError, match="not prepared"):
        copy_kernel(x, 1, out)
    assert not copy_kernel._cache and "register" not in env.events
    env.capture = False
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 42
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
    env.run_queue()  # The original framework capture_end drains its host queue.
    env.capture = False
    env.models[42].destroyed = True
    env.state.close()


def test_capture_admission_polls_warmup_but_defers_its_finalization(
    eager_env, copy_kernel, stub_device_runner, monkeypatch
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    warmup = env.state._submissions[0]
    original_done = env.api.PreparedSubmission.done

    def poll(submission):
        if submission is warmup.submission:
            env.events.append("warmup.poll")
        return original_done.__get__(submission)

    monkeypatch.setattr(env.api.PreparedSubmission, "done", property(poll))
    env.run_queue()
    env.runs[0].complete = True  # The user's Torch synchronization completed warmup.
    env.events.clear()
    env.capture = 42
    copy_kernel(x, 2, out)
    assert not warmup.retired and len(env.state._submissions) == 2 and len(env.queue) == 1
    assert "warmup.poll" in env.events and "wait" not in env.events
    assert "synchronize" not in env.events and "device.synchronize" not in env.events
    env.run_queue()
    env.capture = False
    env.models[42].destroyed = True
    env.state.close()


def test_capture_returns_before_callbacks_and_shares_model_notice(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    baseline = len(env.runs)

    # Observe the original methods: PyPTO must never wrap any NPUGraph method.
    class Graph:
        def capture_end(self):
            env.run_queue()
            env.capture = False

        def replay(self):
            pass

        def reset(self):
            env.models[42].destroyed = True

    env.framework.npu.NPUGraph = Graph
    methods = (Graph.capture_end, Graph.replay, Graph.reset)
    env.capture = 42
    env.events.clear()
    copy_kernel(x, 2, out)
    copy_kernel(out, 3, x)
    first, second = env.state._submissions
    assert len(env.queue) == 2 and len(env.runs) == baseline
    assert "synchronize" not in env.events and "device.synchronize" not in env.events
    assert not first.done() and not second.done()
    assert first.capture.native is second.capture.anchors[42]
    assert first.submission.model_notice is second.submission.model_notice
    graph = Graph()
    graph.capture_end()
    copy_kernel(x, 4, out)  # A live graph retains every owner during eager admission.
    assert env.state.state is KernelState.READY
    assert len(env.state._submissions) == 3 and first._owners and second._owners
    assert "capture.release" not in env.events and "device.synchronize" not in env.events
    graph.replay()
    graph.reset()
    env.events.clear()
    copy_kernel(x, 5, out)  # Reclaim the destroyed graph before enqueueing this call.
    assert first.retired and second.retired
    assert len(env.queue) == 1 and env.state._submissions[-1].submission.run is None
    assert len(env.state._submissions) == 1
    assert not env.state._worker.capture_anchors
    assert env.events.index("device.synchronize") < env.events.index("capture.release")
    assert env.events.index("capture.release") < env.events.index("prepare")
    assert methods == (Graph.capture_end, Graph.replay, Graph.reset)
    env.state.close()


def test_capture_close_refuses_live_graph_and_retries_after_destroy(
    eager_env, copy_kernel, stub_device_runner
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 23
    copy_kernel(x, 2, out)
    env.run_queue()
    env.capture = False
    ticket = env.state._submissions[0]
    with pytest.raises(RuntimeError, match="caller-owned NPUGraph"):
        env.state.close()
    assert ticket._owners and not ticket.retired and "close" not in env.events
    env.models[23].destroyed = True
    env.state.close()
    assert ticket.retired and env.state.state is KernelState.CLOSED


@pytest.mark.parametrize("failure", ["device_sync_error", "capture_release_error"])
def test_capture_cleanup_failure_preserves_owners_for_close_retry(
    eager_env, copy_kernel, stub_device_runner, failure
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 42
    copy_kernel(x, 2, out)
    env.run_queue()
    env.capture = False
    env.models[42].destroyed = True
    error = RuntimeError("cleanup refused")
    setattr(env, failure, error)
    ticket = env.state._submissions[0]
    with pytest.raises(RuntimeError) as caught:
        env.state.close()
    assert caught.value is error
    assert ticket._owners and not ticket.retired and not ticket.submission.safe_to_release
    setattr(env, failure, None)
    env.state.close()
    assert ticket.retired and env.state.state is KernelState.CLOSED


def test_capture_model_address_reuse_does_not_reuse_destroyed_anchor(
    eager_env, copy_kernel, stub_device_runner
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 42
    copy_kernel(x, 2, out)
    env.run_queue()
    old = env.state._submissions[0]
    env.models[42].destroyed = True
    env.events.clear()
    copy_kernel(x, 3, out)  # New recording must defer the old model's cleanup.
    assert not old.retired and "device.synchronize" not in env.events
    new = env.state._submissions[1]
    assert new.submission.model_notice is not old.submission.model_notice
    env.run_queue()
    env.capture = False
    env.state.drain()
    assert old.retired and not new.retired
    assert env.state._worker.capture_anchors[42] is new.capture.native
    env.models[42].destroyed = True
    env.state.close()


@pytest.mark.parametrize("after_notice", [False, True])
def test_capture_partial_retention_failure_keeps_resources_until_checked_cleanup(
    eager_env, copy_kernel, stub_device_runner, monkeypatch, after_notice
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    original = env.api.PreparedSubmission._retain_model_resources
    error = ValueError("retention failed")

    def fail(submission, anchor=None):
        if after_notice:
            original(submission, anchor)
        raise error

    monkeypatch.setattr(env.api.PreparedSubmission, "_retain_model_resources", fail)
    env.capture = 42
    with pytest.raises(ValueError) as caught:
        copy_kernel(x, 2, out)
    assert caught.value is error and not env.queue
    ticket = env.state._submissions[0]
    assert not ticket.retired and ticket._owners
    env.capture = False
    if after_notice:
        with pytest.raises(RuntimeError, match="destroy notice"):
            env.state.close()
        assert ticket._owners
        env.models[42].destroyed = True
    with pytest.raises(ValueError) as caught:
        env.state.close()
    assert caught.value is error and ticket.retired and env.state.state is KernelState.CLOSED


def test_capture_enqueue_failure_preserves_model_owners(eager_env, copy_kernel, stub_device_runner):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 42
    env.enqueue_error = ValueError("enqueue failed")
    with pytest.raises(ValueError, match="enqueue failed"):
        copy_kernel(x, 2, out)
    ticket = env.state._submissions[0]
    assert ticket.submission.cancelled and ticket._owners and not ticket.retired
    env.capture = False
    env.models[42].destroyed = True
    with pytest.raises(ValueError, match="enqueue failed"):
        env.state.close()
    assert ticket.retired and env.state.state is KernelState.CLOSED


@pytest.mark.parametrize("failure", ["device_sync_error", "capture_release_error"])
def test_automatic_capture_cleanup_failure_retains_owners_and_refuses_new_enqueue(
    eager_env, copy_kernel, stub_device_runner, failure
):
    env = eager_env
    env.init()
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    env.capture = 42
    copy_kernel(x, 2, out)
    env.run_queue()
    env.capture = False
    env.models[42].destroyed = True
    ticket = env.state._submissions[0]
    error = RuntimeError("retry release")
    setattr(env, failure, error)
    env.events.clear()
    with pytest.raises(RuntimeError, match="earlier asynchronous launch error") as caught:
        copy_kernel(x, 3, out)
    assert caught.value.__cause__ is error
    assert env.state.state is KernelState.FAILED and env.state._submissions == [ticket]
    assert ticket._owners and not ticket.retired
    assert "prepare" not in env.events and not env.queue
    setattr(env, failure, None)
    env.state.close()
    assert ticket.retired and env.state.state is KernelState.CLOSED


def test_capture_rejects_diagnostics_before_compilation(eager_env, copy_kernel, tmp_path):
    env = eager_env
    env.init(enable_chip_swimlane=1, output_dir=tmp_path)
    env.capture = 42
    with pytest.raises(ValueError, match="diagnostics disabled"):
        copy_kernel(env.tensor(), 1, env.tensor())
    assert not copy_kernel._cache and not env.queue
    env.capture = False
    env.state.close()


@pytest.mark.parametrize("options", [RunConfig(), CompileOptions(backend_type=BackendType.Ascend950)])
def test_call_config_cannot_change_execution_target(eager_env, copy_kernel, options):
    eager_env.init()
    with pytest.raises((TypeError, ValueError), match="CompileOptions.*conflicts|require CompileOptions"):
        copy_kernel(eager_env.tensor(), 1, eager_env.tensor(), config=options)
    eager_env.state.close()


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("default_options", [False, True])
def test_default_compile_target_uses_initialized_config(
    eager_env, copy_kernel, stub_device_runner, runtime, default_options
):
    env = eager_env
    env.init(runtime=runtime)
    callable_, _, config = stub_device_runner._compile_and_assemble.return_value
    stub_device_runner._compile_and_assemble.return_value = callable_, runtime, config
    options = CompileOptions(memory_planner=passes.MemoryPlanner.PYPTO)
    kwargs = {"config": options} if default_options else {}
    x, out = env.tensor(), env.tensor()
    assert passes.PassContext.current() is None
    assert copy_kernel(x, 1, out, **kwargs) == (out, out)
    key, artifact = next(iter(copy_kernel._cache.items()))
    assert key.platform == artifact.platform == "a2a3"
    assert dict(key.compile_opts)["runtime"] == runtime
    assert f'"{runtime}"' in (artifact.output_dir / "kernel_config.py").read_text()
    assert options.execution_mode is ExecutionMode.SIM  # Do not mutate caller options.
    assert passes.PassContext.current() is None
    copy_kernel(x, 2, out, config=CompileOptions(execution_mode=ExecutionMode.ONBOARD))
    assert len(copy_kernel._cache) == 1 and env.events.count("register") == 1
    env.state.close()


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("matching", [False, True])
def test_active_context_runtime_must_match_initialized_config(
    eager_env, copy_kernel, stub_device_runner, monkeypatch, runtime, matching
):
    env = eager_env
    env.init(runtime=runtime)
    callable_, _, config = stub_device_runner._compile_and_assemble.return_value
    stub_device_runner._compile_and_assemble.return_value = callable_, runtime, config
    other = "host_build_graph" if runtime == "tensormap_and_ringbuffer" else "tensormap_and_ringbuffer"
    context = passes.PassContext([], runtime=passes.runtime_kind_from_name(runtime if matching else other))
    with context:
        if matching:
            copy_kernel(env.tensor(), 1, env.tensor(), config=CompileOptions())
            key = next(iter(copy_kernel._cache))
            assert dict(key.compile_opts)["runtime"] == runtime
        else:
            monkeypatch.setattr(copy_kernel, "_compile", lambda *a, **kw: pytest.fail("compiled mismatch"))
            with pytest.raises(ValueError, match="Active PassContext runtime.*conflicts"):
                copy_kernel(env.tensor(), 1, env.tensor())
            assert not copy_kernel._cache and "register" not in env.events
    env.state.close()


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


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_persistent_kernel_build_restore_and_binary_validation(
    eager_env, copy_kernel, device_runner, monkeypatch, tmp_path, runtime
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
            runtime_name=runtime,
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
    env.init(runtime=runtime)
    x, out = env.tensor(), env.tensor()
    copy_kernel(x, 1, out)
    env.state.drain()
    artifact = next(iter(copy_kernel._artifact_objects.values()))
    handle = artifact._artifact_runtime.handle
    assert artifact.runtime_name == runtime
    assert handle.key.contract is ArtifactExecutionMode.KERNEL
    assert handle.spec.execution_capabilities.modes == (ArtifactExecutionMode.KERNEL,)
    from pypto.jit._artifact_manifest import ArtifactState  # noqa: PLC0415

    assert handle.spec.state is ArtifactState.BINARY_READY
    assert len(built) == 1
    generated_config = json.loads((artifact.output_dir / "kernel_config.json").read_text())
    assert generated_config["orchestration"]["signature"] == ["IN", "OUT", "SCALAR"]
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
    env.run_queue()
    env.models[1].destroyed = True
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
