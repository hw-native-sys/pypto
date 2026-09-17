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
        return SimpleNamespace(kernel_abi=kwargs["_kernel_abi"])

    frontend = JITFunction._compile_to_program
    specialize = JITFunction._resolve_specialization

    def counted_frontend(self, *args, **kwargs):
        events.frontends += 1
        return frontend(self, *args, **kwargs)

    def counted_specialization(self, *args, **kwargs):
        events.bindings += 1
        return specialize(self, *args, **kwargs)

    def invoke(artifact, frame, bound):
        assert bound is events.state.config
        events.frames.append(frame)
        return frame.alias_result()

    monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "_compile_impl", build)
    monkeypatch.setattr(JITFunction, "_compile_to_program", counted_frontend)
    monkeypatch.setattr(JITFunction, "_resolve_specialization", counted_specialization)
    monkeypatch.setattr(launch, "invoke", invoke)
    monkeypatch.setattr(importlib.import_module("pypto._cache_config")._policy, "override", CacheConfig())
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
