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
from pypto.runtime import RunConfig
from pypto.torch import interop, launch

from tests.ut.torch.test_interop import _tensor


@pl.jit
def scale_eager(x: pl.Tensor, scale: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(x, [0, 0], [4, 4]), scale), [0, 0], out)
    return out


@pytest.fixture
def eager(monkeypatch):
    scale_eager._cache.clear()
    scale_eager._kernel_contracts.clear()
    stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=12)
    events = SimpleNamespace(builds=0, captures=[], frames=[], stream=stream, mutate=None)
    npu = SimpleNamespace(
        get_npu_format=lambda tensor: 2,
        npu=SimpleNamespace(current_device=lambda: 0, current_stream=lambda device: events.stream),
    )
    monkeypatch.setattr(interop, "_load_torch_npu", lambda: npu)
    monkeypatch.setattr(
        launch,
        "_load_native",
        lambda: SimpleNamespace(
            check_eager=lambda stream_id, device: events.captures.append((stream_id, device))
        ),
    )

    def build(program, **kwargs):
        events.builds += 1
        if events.mutate is not None:
            events.mutate.value = 99
        return SimpleNamespace(kernel_abi=kwargs["_kernel_abi"])

    def invoke(artifact, frame, config):
        events.frames.append(frame)
        return frame.alias_result()

    monkeypatch.setattr(importlib.import_module("pypto.ir.compile"), "_compile_impl", build)
    monkeypatch.setattr(launch, "invoke", invoke)
    events.config = RunConfig(platform="a2a3", cache_config=CacheConfig(enabled=False))
    return events


def test_eager_reuses_artifact_and_snapshots_each_scalar_and_stream(eager):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    scalar = ctypes.c_float(2)
    assert scale_eager(x, scalar, out, config=eager.config) is out
    scalar.value = 3
    eager.stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=13)
    assert scale_eager(out=out, x=x, scale=scalar, config=eager.config) is out
    assert eager.builds == 1
    assert len(scale_eager._kernel_contracts) == 1
    assert [frame.scalars[0].value for frame in eager.frames] == [2, 3]
    assert [frame.stream.stream_id for frame in eager.frames] == [12, 13]
    assert eager.captures == [(12, 0), (13, 0)]


def test_eager_defaults_to_current_device_without_program_config(eager):
    x, out = _tensor((4, 4)), _tensor((4, 4))
    assert scale_eager(x, 2, out) is out
    assert eager.builds == 1
    assert eager.frames[0].device_index == 0
    assert eager.captures == [(12, 0)]


def test_eager_snapshots_mutable_scalar_before_compilation(eager):
    scalar = ctypes.c_float(2)
    eager.mutate = scalar
    scale_eager(_tensor((4, 4)), scalar, _tensor((4, 4)), config=eager.config)
    assert scalar.value == 99
    assert eager.frames[0].scalars[0].value == 2


def test_eager_requires_outputs_before_compilation(eager):
    with pytest.raises(TypeError, match="out"):
        scale_eager(_tensor((4, 4)), 2, config=eager.config)
    assert eager.builds == 0 and not eager.frames and not eager.captures


def test_eager_rejects_cpu_without_compiler_or_native_queries(eager):
    with pytest.raises(TypeError, match="NPU torch.Tensor"):
        scale_eager(torch.ones(4, 4), 2, torch.empty(4, 4), config=eager.config)
    assert eager.builds == 0 and not eager.frames and not eager.captures


@pytest.mark.parametrize(
    "change",
    [
        {"platform": "a2a3sim"},
        {"platform": "a5"},
        {"device_id": 1},
        {"enable_pmu": 1},
        {"codegen_only": True},
        {"aicpu_thread_num": 1},
    ],
)
def test_eager_rejects_unsupported_runtime_options_before_build(eager, change):
    config = RunConfig(**({"platform": "a2a3", "cache_config": CacheConfig(enabled=False)} | change))
    with pytest.raises(ValueError):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)), config=config)
    assert eager.builds == 0 and not eager.frames


def test_eager_rejects_capture_before_build(eager, monkeypatch):
    def captured(stream, device):
        raise ValueError("graph capture is unsupported")

    monkeypatch.setattr(launch, "_load_native", lambda: SimpleNamespace(check_eager=captured))
    with pytest.raises(ValueError, match="graph capture"):
        scale_eager(_tensor((4, 4)), 2, _tensor((4, 4)), config=eager.config)
    assert eager.builds == 0 and not eager.frames


def test_explicit_compile_keeps_program_path_and_separate_cache(eager, monkeypatch):
    calls = []

    def program(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(scale_eager, "_compile", lambda *args, **kwargs: program)
    x, out = _tensor((4, 4)), _tensor((4, 4))
    compiled = scale_eager.compile(x, 2, out, config=eager.config)
    assert compiled is program
    assert not eager.frames and not eager.captures
    compiled(x, 3, out)
    assert calls == [(x, 3, out)]
    assert scale_eager(x, 4, out, config=eager.config) is out
    assert eager.builds == 1 and len(scale_eager._cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
