# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public JIT eager dispatch, validated without native device execution."""

import ctypes
import importlib
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.ir import DistributedConfig
from pypto.jit.decorator import JITFunction
from pypto.pypto_core import passes
from pypto.runtime import CompileOptions, RunConfig
from pypto.runtime.kernel import context
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import KernelState
from pypto.torch import interop, launch

from tests.ut.torch.test_interop import _tensor


@pl.jit
def scale_eager(x: pl.Tensor, scale: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(x, [0, 0], [4, 4]), scale), [0, 0], out)
    return out


def _bind(monkeypatch, config):
    """Emulate a completed pypto.torch.init without a native Worker."""
    state = context._ProcessKernelState()
    if config is not None:
        state.config = config
        state.state = KernelState.READY
    monkeypatch.setattr(context, "_process", SimpleNamespace(state=state))
    return state


@pytest.fixture
def eager(monkeypatch):
    scale_eager._cache.clear()
    scale_eager._kernel_contracts.clear()
    stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=12)
    events = SimpleNamespace(
        builds=0, bindings=0, frontends=0, captures=[], frames=[], stream=stream, mutate=None, device=0
    )
    npu = SimpleNamespace(
        get_npu_format=lambda tensor: 2,
        npu=SimpleNamespace(
            current_device=lambda: events.device, current_stream=lambda device: events.stream
        ),
    )
    monkeypatch.setattr(interop, "_load_torch_npu", lambda: npu)
    monkeypatch.setattr(
        launch,
        "_load_native",
        lambda: SimpleNamespace(
            check_call=lambda stream_id, device: events.captures.append((stream_id, device))
        ),
    )

    def build(program, **kwargs):
        events.builds += 1
        events.build_kwargs = kwargs
        if events.mutate is not None:
            events.mutate.value = 99
        image = f"image-{events.builds}".encode()
        return SimpleNamespace(
            kernel_abi=kwargs["_kernel_abi"],
            identity=lambda: image,
            loaded_identity=lambda: image,
            load=lambda: image,
        )

    frontend = JITFunction._compile_to_program
    specialize = JITFunction._resolve_specialization

    def counted_frontend(self, *args, **kwargs):
        events.frontends += 1
        return frontend(self, *args, **kwargs)

    def counted_specialization(self, *args, **kwargs):
        events.bindings += 1
        return specialize(self, *args, **kwargs)

    def invoke(artifact, frame, bound, *, specialization=None):
        assert bound is events.state.config
        events.frames.append(frame)
        return frame.alias_result()

    monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "_compile_impl", build)
    monkeypatch.setattr(JITFunction, "_compile_to_program", counted_frontend)
    monkeypatch.setattr(JITFunction, "_resolve_specialization", counted_specialization)
    events.real_invoke = launch.invoke
    monkeypatch.setattr(launch, "invoke", invoke)
    monkeypatch.setattr(
        importlib.import_module("pypto._cache_config")._policy, "override", CacheConfig(enabled=False)
    )
    events.state = _bind(monkeypatch, KernelConfig("a2a3", "tensormap_and_ringbuffer", 0))
    return events


def test_eager_reuses_artifact_and_snapshots_each_scalar_and_stream(eager):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scalar = ctypes.c_float(2)
    assert scale_eager(x, scalar, out) is out
    scalar.value = 3
    eager.stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=13)
    assert scale_eager(out=out, x=x, scale=scalar) is out
    assert eager.builds == 1
    assert len(scale_eager._kernel_contracts) == 1
    assert [frame.scalars[0].value for frame in eager.frames] == [2, 3]
    assert [frame.stream.stream_id for frame in eager.frames] == [12, 13]
    assert eager.captures == [(12, 0), (13, 0)]


def test_eager_compiles_for_the_init_bound_target(eager):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    assert scale_eager(x, 2, out) is out
    abi = eager.build_kwargs["_kernel_abi"]
    assert (abi.platform, abi.runtime) == ("a2a3", "tensormap_and_ringbuffer")
    assert eager.build_kwargs["platform"] == "a2a3"
    assert eager.build_kwargs["runtime"] == passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
    assert eager.frames[0].device_index == 0


def test_eager_requires_init_before_binding_or_native_queries(eager, monkeypatch):
    _bind(monkeypatch, None)
    with pytest.raises(RuntimeError, match=r"call pypto\.torch\.init"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)))
    # The missing init is reported before argument binding would reject the missing output.
    with pytest.raises(RuntimeError, match=r"call pypto\.torch\.init"):
        scale_eager(_tensor((4, 4)), 2)
    assert eager.bindings == eager.frontends == eager.builds == 0
    assert not eager.frames and not eager.captures


def test_eager_snapshots_mutable_scalar_before_compilation(eager):
    scalar = ctypes.c_float(2)
    eager.mutate = scalar
    scale_eager(_tensor((4, 4)), scalar, _tensor((4, 4)))
    assert scalar.value == 99
    assert eager.frames[0].scalars[0].value == 2


def test_eager_requires_outputs_before_compilation(eager):
    with pytest.raises(TypeError, match="out"):
        scale_eager(_tensor((4, 4)), 2)
    assert eager.builds == 0 and not eager.frames and not eager.captures


def test_eager_rejects_cpu_without_compiler_or_native_queries(eager):
    with pytest.raises(TypeError, match="NPU torch.Tensor"):
        scale_eager(torch.ones(4, 4), 2, torch.empty(4, 4))
    assert eager.builds == 0 and not eager.frames and not eager.captures


@pytest.mark.parametrize(
    "config",
    [RunConfig(platform="a2a3", cache_config=CacheConfig(enabled=False)), RunConfig(device_id=1), object()],
)
def test_eager_rejects_execution_config_before_frontend(eager, config):
    with pytest.raises(TypeError, match=r"CompileOptions"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)), config=config)
    assert eager.bindings == eager.frontends == eager.builds == 0 and not eager.frames


def test_eager_rejects_distributed_compile_options_before_binding(eager):
    with pytest.raises(ValueError, match="distributed_config"):
        config = CompileOptions(distributed_config=DistributedConfig())
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)), config=config)
    assert eager.bindings == eager.builds == 0 and not eager.frames


def test_eager_compile_options_join_the_specialization(eager):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    scale_eager(x, 2, out, config=CompileOptions())
    assert eager.builds == 1
    scale_eager(x, 2, out, config=CompileOptions(platform="a2a3"))
    assert eager.builds == 1
    scale_eager(x, 2, out, config=CompileOptions(analyze_auto_scopes_for_deps=True))
    assert eager.builds == 2 and len(eager.frames) == 4


def test_eager_rejects_conflicting_compile_platform_before_build(eager):
    with pytest.raises(ValueError, match=r"platform 'a2a3' bound by pypto\.torch\.init"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)), config=CompileOptions(platform="a5"))
    assert eager.builds == 0 and not eager.frames


def test_eager_rejects_conflicting_pass_context_runtime(eager):
    with (
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
        pytest.raises(ValueError, match="conflicts with runtime 'tensormap_and_ringbuffer'"),
    ):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)))
    assert eager.builds == 0 and not eager.frames


def test_eager_uses_a_matching_pass_context_runtime(eager):
    with passes.PassContext([], runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)))
    # The active context already names the runtime, so none is passed to ir.compile.
    assert eager.builds == 1 and "runtime" not in eager.build_kwargs
    assert eager.build_kwargs["_kernel_abi"].runtime == "tensormap_and_ringbuffer"


def test_eager_rejects_tensors_off_the_init_bound_device(eager):
    eager.device = 1
    eager.stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=1), stream_id=12)
    with pytest.raises(ValueError, match=r"bound NPU 0"):
        scale_eager(_tensor((4, 4), device=1), 2, _tensor((4, 4), device=1))
    assert eager.builds == 0 and not eager.frames and not eager.captures


def test_invalidated_capture_is_rejected_before_build(eager, monkeypatch):
    def captured(stream, device):
        raise ValueError("graph capture was invalidated")

    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=captured))
    with pytest.raises(ValueError, match="graph capture"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)))
    assert eager.builds == 0 and not eager.frames


def test_cold_capture_rejected_before_build(eager, monkeypatch):
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    with pytest.raises(RuntimeError, match="requires warmup outside capture"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)))
    assert eager.builds == 0 and not eager.frames


def test_capture_reuses_warm_artifact_and_snapshots_current_scalar(eager, monkeypatch):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    assert scale_eager(x, 3, out) is out
    assert eager.builds == 1
    assert eager.frames[-1].capture_id == 42
    assert eager.frames[-1].scalars[0].value == 3


def test_explicit_compile_keeps_program_path_and_separate_cache(eager, monkeypatch):
    calls = []

    def program(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(scale_eager, "_compile", lambda *args, **kwargs: program)
    x, out = _tensor((4, 4)), _tensor((4, 4))
    config = RunConfig(platform="a2a3", cache_config=CacheConfig(enabled=False))
    compiled = scale_eager.compile(x, 2, out, config=config)
    assert compiled is program
    assert not eager.frames and not eager.captures
    compiled(x, 3, out)
    assert calls == [(x, 3, out)]
    assert scale_eager(x, 4, out) is out
    assert eager.builds == 1 and len(scale_eager._cache) == 2


@pytest.fixture
def prepared(eager, monkeypatch):
    """Keep real preparation/publication and stub only the native submission boundary."""
    eager.prepares = []
    eager.registrations = []
    eager.fail_prepare = False

    def prepare(image):
        eager.prepares.append(image)
        if eager.fail_prepare:
            raise RuntimeError("prepare failed")
        return object()

    def enqueue(registration, frame):
        registration.require_live()
        eager.registrations.append(registration)
        eager.frames.append(frame)
        return frame.alias_result()

    eager.state._worker = SimpleNamespace(prepare=prepare, close=lambda: None)
    monkeypatch.setattr(launch, "invoke", eager.real_invoke)
    monkeypatch.setattr(launch, "_enqueue_frame", enqueue)
    return eager


@pytest.mark.parametrize("bypass", ["none", "environment", "options"])
def test_capture_finds_prepared_callable_without_compilation_caches(prepared, monkeypatch, tmp_path, bypass):
    kwargs = {}
    if bypass == "environment":
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))
    elif bypass == "options":
        kwargs["config"] = CompileOptions(output_dir=str(tmp_path))
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out, **kwargs)
    registration = prepared.registrations[-1]
    scale_eager._cache.clear()
    scale_eager._kernel_contracts.clear()
    warmed = prepared.builds, prepared.frontends, len(prepared.prepares)
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    assert scale_eager(x, 3, out, **kwargs) is out
    assert prepared.registrations[-1] is registration
    assert prepared.frames[-1].scalars[0].value == 3
    assert (prepared.builds, prepared.frontends, len(prepared.prepares)) == warmed
    assert not scale_eager._cache and not scale_eager._kernel_contracts


def test_bypassed_eager_rebuild_publishes_latest_prepared_callable(prepared, monkeypatch, tmp_path):
    monkeypatch.setenv("PYPTO_COMPILE_PROFILING", "1")
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    scale_eager(x, 3, out)
    first, latest = prepared.registrations
    assert first is not latest and prepared.builds == len(prepared.prepares) == 2
    assert not scale_eager._cache and not scale_eager._kernel_contracts
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    scale_eager(x, 4, out)
    assert prepared.registrations[-1] is latest
    assert prepared.builds == len(prepared.prepares) == 2


def test_build_directory_reuses_eager_preparation(prepared, monkeypatch, tmp_path):
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    scale_eager(x, 3, out)
    assert prepared.builds == len(prepared.prepares) == 1
    assert prepared.registrations[0] is prepared.registrations[1]
    assert [frame.scalars[0].value for frame in prepared.frames] == [2, 3]


def test_capture_selects_each_prepared_specialization(prepared, monkeypatch):
    shapes = [(4, 4), (8, 8)]
    tensors = [(_tensor(shape), _tensor(shape)) for shape in shapes]
    for x, out in tensors:
        scale_eager(x, 2, out)
    expected = list(prepared.registrations)
    scale_eager._cache.clear()
    scale_eager._kernel_contracts.clear()
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    for (x, out), registration in zip(tensors, expected):
        scale_eager(x, 3, out)
        assert prepared.registrations[-1] is registration
    with pytest.raises(RuntimeError, match="requires warmup"):
        scale_eager(_tensor((16, 16)), 2, _tensor((16, 16)))
    assert prepared.builds == len(prepared.prepares) == 2


def test_capture_rejects_unwarmed_compile_configuration(prepared, monkeypatch):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    with pytest.raises(RuntimeError, match="requires warmup"):
        scale_eager(x, 3, out, config=CompileOptions(analyze_auto_scopes_for_deps=True))
    assert prepared.builds == len(prepared.prepares) == 1


def test_failed_prepare_does_not_publish_capture_warmup(prepared, monkeypatch, tmp_path):
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))
    prepared.fail_prepare = True
    x, out = _tensor((4, 4)), _tensor((4, 4))
    with pytest.raises(RuntimeError, match="prepare failed"):
        scale_eager(x, 2, out)
    assert not prepared.state._specializations
    with monkeypatch.context() as capture:
        capture.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
        with pytest.raises(RuntimeError, match="requires warmup"):
            scale_eager(x, 3, out)
    assert prepared.builds == len(prepared.prepares) == 1
    prepared.fail_prepare = False
    scale_eager(x, 2, out)
    assert len(prepared.state._specializations) == 1


def test_new_worker_generation_requires_new_warmup(prepared, monkeypatch):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager(x, 2, out)
    old_registration = prepared.registrations[-1]
    worker = prepared.state._worker
    prepared.state.close()
    assert not prepared.state._specializations
    state = _bind(monkeypatch, prepared.state.config)
    state._worker = worker
    with monkeypatch.context() as capture:
        capture.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
        with pytest.raises(RuntimeError, match="requires warmup"):
            scale_eager(x, 3, out)
    scale_eager(x, 2, out)
    new_registration = prepared.registrations[-1]
    assert new_registration.generation != old_registration.generation
    with pytest.raises(RuntimeError, match="closed"):
        old_registration.require_live()
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    scale_eager(x, 3, out)
    assert prepared.registrations[-1] is new_registration
    assert prepared.builds == 1 and len(prepared.prepares) == 2


def test_program_diagnostics_still_rebuild_without_key_lookup(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("PYPTO_COMPILE_PROFILING", "1")
    compiled = []
    monkeypatch.setattr(scale_eager, "_compile", lambda *args, **kwargs: compiled.append(object()))
    monkeypatch.setattr(
        scale_eager, "_get_source_hash", lambda: pytest.fail("program bypass must stay fresh")
    )
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scale_eager.compile(x, 2, out)
    scale_eager.compile(x, 2, out)
    assert len(compiled) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
