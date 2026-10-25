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
import gc
import importlib
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pypto.language as pl
import pypto.torch as pypto_torch
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


@pl.jit
def host_read(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    scale: pl.Scalar[pl.FP32] = pl.tensor.read(x, [0, 0])
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

    compiler = importlib.import_module("pypto.ir.compile")
    events.real_compile = compiler._compile_impl
    monkeypatch.setattr(compiler, "_compile_impl", build)
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


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_eager_compiles_for_the_init_bound_target(eager, monkeypatch, runtime):
    eager.state = _bind(monkeypatch, KernelConfig("a2a3", runtime, 0))
    x, out = _tensor((4, 4)), _tensor((4, 4))
    assert scale_eager(x, 2, out) is out
    abi = eager.build_kwargs["_kernel_abi"]
    assert (abi.platform, abi.runtime) == ("a2a3", runtime)
    assert eager.build_kwargs["platform"] == "a2a3"
    assert eager.build_kwargs["runtime"] == passes.runtime_kind_from_name(runtime)
    assert eager.frames[0].device_index == 0


def test_hbg_rejects_host_read_before_pipeline_or_launch(eager, monkeypatch):
    eager.state = _bind(monkeypatch, KernelConfig("a2a3", "host_build_graph", 0))
    host_read._cache.clear()
    host_read._kernel_contracts.clear()
    compiler = importlib.import_module("pypto.ir.compile")
    monkeypatch.setattr(compiler, "_compile_impl", eager.real_compile)

    def unexpected_pipeline(*args, **kwargs):
        pytest.fail("Invalid Host access reached the optimization pipeline")

    monkeypatch.setattr(compiler, "_run_pass_pipeline", unexpected_pipeline)
    with pytest.raises(ValueError, match="HBG kernel Host orchestration.*cannot use tensor.read"):
        host_read(_tensor((4, 4)), _tensor((4, 4)))
    assert not eager.frames


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

    def enqueue(registration, frame, *, plan=None):
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


def test_public_prepare_does_not_execute_and_hot_calls_skip_jit(prepared, monkeypatch):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    ready = pypto_torch.prepare(scale_eager, x, 2.0, out)
    assert prepared.builds == len(prepared.prepares) == 1
    assert not prepared.frames and not prepared.registrations

    def unexpected(*args, **kwargs):
        pytest.fail("A prepared call repeated JIT analysis or signature construction")

    # A timing assertion is noisy; forbidding the work proves it is absent
    # even when the compiled entry contains a much deeper dependency graph.
    for name in ("_resolve_compiled", "_resolve_specialization", "_get_source_hash"):
        monkeypatch.setattr(JITFunction, name, unexpected)
    monkeypatch.setattr(interop.CallSignature, "__init__", unexpected)
    assert ready(x, 3.0, out) is out
    assert ready(out=out, scale=4.0, x=x) is out
    assert [frame.scalars[0].value for frame in prepared.frames] == [3.0, 4.0]
    assert prepared.registrations[0] is prepared.registrations[1]


def test_public_prepare_reuses_registration_and_publishes_warmup(prepared):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    first = pypto_torch.prepare(scale_eager, x, 2.0, out)
    second = pypto_torch.prepare(scale_eager, x, 3.0, out)
    assert prepared.builds == len(prepared.prepares) == 1
    assert not prepared.frames
    assert first(x, 4.0, out) is second(x, 5.0, out) is out
    scale_eager(x, 6.0, out)
    assert all(reg is prepared.registrations[0] for reg in prepared.registrations)
    assert len(prepared.state._specializations) == 1


def test_public_prepare_rejects_capture_before_build_or_registration(prepared, monkeypatch):
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    with pytest.raises(RuntimeError, match="prepare.*outside.*capture"):
        pypto_torch.prepare(scale_eager, _tensor((4, 4)), 2.0, _tensor((4, 4)))
    assert prepared.builds == 0 and not prepared.prepares and not prepared.frames


@pytest.mark.parametrize(
    "kernel,config,message", [(lambda: None, None, "@pl.jit"), (scale_eager, RunConfig(), "CompileOptions")]
)
def test_public_prepare_requires_jit_and_compile_options(prepared, kernel, config, message):
    with pytest.raises(TypeError, match=message):
        pypto_torch.prepare(kernel, _tensor((4, 4)), 2.0, _tensor((4, 4)), config=config)
    assert prepared.builds == 0 and not prepared.prepares and not prepared.frames


def test_public_prepare_requires_initialized_worker(prepared, monkeypatch):
    _bind(monkeypatch, None)
    with pytest.raises(RuntimeError, match=r"call pypto\.torch\.init"):
        pypto_torch.prepare(scale_eager, _tensor((4, 4)), 2.0, _tensor((4, 4)))
    assert prepared.builds == 0 and not prepared.prepares and not prepared.frames


def test_public_prepared_call_rejects_stale_registration_before_arguments(prepared):
    ready = pypto_torch.prepare(scale_eager, _tensor((4, 4)), 2.0, _tensor((4, 4)))
    prepared.state.close()
    with pytest.raises(RuntimeError, match="closed"):
        ready()
    assert not prepared.frames


_PREPARED_T = pl.dynamic("T")
_PREPARED_OFFSET = 1.0


@pl.jit
def dynamic_prepared(
    x: pl.Tensor[[_PREPARED_T, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[_PREPARED_T, 32], pl.FP32]],
    value: pl.constexpr,
):
    with pl.at(level=pl.Level.CORE_GROUP):
        rows = pl.tensor.dim(x, 0)
        tile = pl.load(x, [0, 0], [32, 32], valid_shape=[rows, 32])
        pl.store(pl.add(tile, value + _PREPARED_OFFSET), [0, 0], out)
    return out


def test_prepared_freezes_constants_but_accepts_new_dynamic_tensors(prepared, monkeypatch):
    dynamic_prepared._cache.clear()
    dynamic_prepared._kernel_contracts.clear()
    ready = pypto_torch.prepare(dynamic_prepared, _tensor((32, 32)), _tensor((32, 32)), value=2.0)
    assert not prepared.frames
    first_registration = ready._registration
    monkeypatch.setitem(dynamic_prepared._func.__globals__, "_PREPARED_OFFSET", 7.0)
    for rows in (16, 32):
        x, out = _tensor((rows, 32)), _tensor((rows, 32))
        assert ready(x, out) is out
        assert prepared.registrations[-1] is first_registration
        assert prepared.frames[-1].tensors[0].metadata.shape == (rows, 32)
    assert prepared.builds == 1
    # Explicit freezing is observable: ordinary JIT and a new prepare still
    # resolve changed constants, while the old callable retains its artifact.
    fresh = pypto_torch.prepare(dynamic_prepared, x, out, value=2.0)
    assert fresh._registration is not first_registration
    assert prepared.builds == 2
    dynamic_prepared(x, out, value=2.0)
    assert prepared.registrations[-1] is fresh._registration
    changed_constexpr = pypto_torch.prepare(dynamic_prepared, x, out, value=3.0)
    assert changed_constexpr._registration is not fresh._registration
    with pytest.raises(TypeError, match="unexpected keyword"):
        ready(x, out, value=3.0)


def test_prepare_does_not_retain_sample_tensors_or_scalars(prepared):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scalar = ctypes.c_float(2)
    references = [weakref.ref(value) for value in (x, out, scalar)]
    ready = pypto_torch.prepare(scale_eager, x, scalar, out)
    del x, out, scalar
    gc.collect()
    assert all(reference() is None for reference in references)
    current_out = _tensor((4, 4))
    assert ready(_tensor((4, 4)), 5.0, current_out) is current_out


@pytest.mark.parametrize("invalid", ["missing", "extra", "unknown", "duplicate", "config"])
def test_prepared_binding_rejects_invalid_calls(prepared, invalid):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    ready = pypto_torch.prepare(scale_eager, x, 2.0, out)
    args, kwargs = (x, 2.0, out), {}
    if invalid == "missing":
        args = (x, 2.0)
    elif invalid == "extra":
        args = (x, 2.0, out, 3)
    elif invalid == "unknown":
        kwargs = {"unknown": 3}
    elif invalid == "duplicate":
        kwargs = {"x": x}
    else:
        kwargs = {"config": CompileOptions()}
    with pytest.raises(TypeError):
        ready(*args, **kwargs)
    assert not prepared.frames


@pytest.mark.parametrize("invalid", ["shape", "dtype", "strides", "alias", "device", "grad"])
def test_prepared_revalidates_tensor_contract(prepared, invalid):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    ready = pypto_torch.prepare(scale_eager, x, 2.0, out)
    if invalid == "shape":
        x = _tensor((8, 4))
    elif invalid == "dtype":
        x = _tensor((4, 4), dtype=torch.int32)
    elif invalid == "strides":
        x = x.t()
    elif invalid == "alias":
        base = _tensor((5, 4))
        x, out = base[:4], base[1:]
    elif invalid == "device":
        prepared.device = 1
    else:
        x.requires_grad_(True)
    with pytest.raises((TypeError, ValueError)):
        ready(x, 2.0, out)
    assert not prepared.frames


def test_prepared_rechecks_current_stream_capture_and_scalar(prepared, monkeypatch):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scalar = ctypes.c_float(2)
    ready = pypto_torch.prepare(scale_eager, x, scalar, out)
    assert ready(x, scalar, out) is out
    scalar.value = 9
    prepared.stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=37)
    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_call=lambda *args: 42))
    assert ready(x, scalar, out) is out
    assert [frame.scalars[0].value for frame in prepared.frames] == [2, 9]
    assert [frame.stream.stream_id for frame in prepared.frames] == [12, 37]
    assert [frame.capture_id for frame in prepared.frames] == [0, 42]
    assert prepared.builds == len(prepared.prepares) == 1


def test_prepared_rejects_zero_dynamic_extent_before_submission(prepared):
    ready = pypto_torch.prepare(dynamic_prepared, _tensor((32, 32)), _tensor((32, 32)), value=2.0)
    with pytest.raises(ValueError, match="positive u32"):
        ready(_tensor((0, 32)), _tensor((0, 32)))
    assert not prepared.frames


@pytest.mark.parametrize("stale", ["pid", "generation"])
def test_prepared_checks_registration_identity(prepared, stale):
    ready = pypto_torch.prepare(scale_eager, _tensor((4, 4)), 2.0, _tensor((4, 4)))
    registration = ready._registration
    invalid = replace(registration, **{stale: getattr(registration, stale) + 1})
    ready = replace(ready, _registration=invalid)
    with pytest.raises(RuntimeError, match="live Worker generation"):
        ready()
    assert not prepared.frames


def test_prepare_failure_does_not_publish_and_can_retry(prepared):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    prepared.fail_prepare = True
    with pytest.raises(RuntimeError, match="prepare failed"):
        pypto_torch.prepare(scale_eager, x, 2.0, out)
    assert not prepared.state._specializations and not prepared.frames
    prepared.fail_prepare = False
    ready = pypto_torch.prepare(scale_eager, x, 2.0, out)
    assert ready(x, 2.0, out) is out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
