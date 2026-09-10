# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free JIT warmup shares specialization and prepares every chip build."""

import sys
from types import ModuleType
from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto.ir.compiled_program import CompiledProgram
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.runtime import RunConfig


@pytest.fixture
def kernel():
    # DSL scalar annotations describe IR values; defaults are Python literals.
    default_factor: Any = 2.0

    @pl.jit
    def scale(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        factor: pl.Scalar[pl.FP32] = default_factor,
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(x, [0, 0], [16, 16])
            pl.store(pl.mul(tile, factor), [0, 0], out)
        return out

    return scale


@pytest.fixture
def assembly(monkeypatch, tmp_path):
    # Unit tests run the real JIT passes/codegen but stub the external toolchain.
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    calls = []
    runner: Any = ModuleType("pypto.runtime.device_runner")

    def build(directory, platform):
        assert directory.is_dir()
        result = (object(), "test_runtime", {})
        calls.append((directory, platform, result))
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("warmup must not initialize a worker or execute a program")

    runner._compile_and_assemble = build
    monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", runner)
    monkeypatch.setitem(sys.modules, "simpler.worker", None)
    monkeypatch.setattr(CompiledProgram, "__call__", forbidden)
    monkeypatch.setattr(DistributedCompiledProgram, "__call__", forbidden)
    monkeypatch.setattr(DistributedCompiledProgram, "prepare", forbidden)
    return calls


def test_warmup_prepares_existing_compilation_once(kernel, assembly):
    compiled = kernel.compile()
    assert assembly == []
    assert compiled._chip_callable is None
    assert kernel.warmup() is compiled
    assert compiled.program is not None
    assert compiled.chip_callable is assembly[0][2][0]
    assert kernel.warmup() is compiled
    assert kernel.compile() is compiled
    assert len(assembly) == 1


def test_sample_arguments_share_annotation_specialization(kernel, assembly):
    x = torch.zeros(16, 16)
    out = torch.zeros_like(x)
    compiled = kernel.warmup(x=x, out=out)
    assert kernel.warmup() is compiled
    assert torch.count_nonzero(out) == 0
    assert len(assembly) == 1


def test_warmup_preserves_scalar_specialization_and_runtime_marker(kernel, assembly):
    default = kernel.warmup()
    specialized = kernel.warmup(factor=3.0)
    dynamic = kernel.warmup(factor=pl.RUNTIME)
    assert len({id(default), id(specialized), id(dynamic)}) == 3
    assert kernel.compile(factor=pl.RUNTIME) is dynamic
    assert len(assembly) == 3


def test_warmup_preserves_dynamic_extents(assembly):
    rows = pl.dynamic("rows")

    @pl.jit
    def copy(x: pl.Tensor[[rows, 16], pl.FP32], out: pl.Out[pl.Tensor[[rows, 16], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.store(pl.load(x, [0, 0], [16, 16]), [0, 0], out)
        return out

    first = copy.warmup(torch.zeros(16, 16), torch.zeros(16, 16))
    second = copy.warmup(torch.zeros(32, 16), torch.zeros(32, 16))
    assert first is second
    assert len(assembly) == 1


def test_warmup_honors_fresh_output_and_platform(kernel, assembly, tmp_path):
    cached = kernel.warmup()
    config = RunConfig(platform="a2a3", save_kernels_dir=str(tmp_path / "fresh"))
    first = kernel.warmup(config=config)
    second = kernel.warmup(config=config)
    assert first is not second and first is not cached
    assert first.output_dir == tmp_path / "fresh"
    assert assembly[-1][1] == "a2a3"
    assert kernel.compile() is cached
    assert len(assembly) == 3


def test_binary_failure_leaves_retryable_compiled_object(kernel, assembly, monkeypatch):
    runner = sys.modules["pypto.runtime.device_runner"]
    build = runner._compile_and_assemble
    error = RuntimeError("device compiler failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "_compile_and_assemble", fail)
    with pytest.raises(RuntimeError, match="device compiler failed") as raised:
        kernel.warmup()
    assert raised.value is error
    compiled = kernel.compile()
    assert compiled._chip_callable is None
    monkeypatch.setattr(runner, "_compile_and_assemble", build)
    assert kernel.warmup() is compiled
    assert len(assembly) == 1


def test_invalid_arguments_fail_before_binary_preparation(kernel, assembly):
    with pytest.raises(TypeError, match="out"):
        kernel.warmup(torch.zeros(16, 16))
    assert assembly == []


def test_distributed_warmup_prepares_all_children(kernel, assembly, tmp_path, monkeypatch):
    for name in ("left", "right"):
        child = tmp_path / "next_levels" / name
        child.mkdir(parents=True)
        (child / "kernel_config.py").write_text("")
    (tmp_path / "next_levels" / "auxiliary").mkdir()
    compiled = DistributedCompiledProgram(None, str(tmp_path), platform="a2a3")
    monkeypatch.setattr(kernel, "compile", lambda *args, **kwargs: compiled)
    assert kernel.warmup() is compiled
    assert [path.name for path, _, _ in assembly] == ["left", "right"]
    assert all(platform == "a2a3" for _, platform, _ in assembly)


def test_distributed_warmup_rejects_empty_build(kernel, assembly, tmp_path, monkeypatch):
    compiled = DistributedCompiledProgram(None, str(tmp_path))
    monkeypatch.setattr(kernel, "compile", lambda *args, **kwargs: compiled)
    with pytest.raises(RuntimeError, match="No chip-level tasks"):
        kernel.warmup()
    assert assembly == []


def test_multi_orchestration_warmup_prepares_each_subbuild(kernel, assembly, tmp_path, monkeypatch):
    @pl.program
    class Multi:
        @pl.function(type=pl.FunctionType.Orchestration)
        def left(self, x: pl.Tensor[[16, 16], pl.FP32]):
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def right(self, x: pl.Tensor[[16, 16], pl.FP32]):
            return x

    for name in ("left", "right"):
        (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
    compiled = CompiledProgram(Multi, str(tmp_path))
    monkeypatch.setattr(kernel, "compile", lambda *args, **kwargs: compiled)
    assert kernel.warmup() is compiled
    assert [path.name for path, _, _ in assembly] == ["left", "right"]


def test_unsupported_result_has_context(kernel, monkeypatch):
    monkeypatch.setattr(kernel, "compile", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="scale.*warmup.*object"):
        kernel.warmup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
