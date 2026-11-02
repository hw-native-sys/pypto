# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Selecting the target Simpler runtime ABI for a compilation.

The runtime name travels through ``PassContext`` rather than being a
codegen-only argument, so passes that must legalize IR for a specific runtime
can read it. These tests pin the whole path: the context round-trip, the
precedence ``ir.compile()`` applies, and the value that lands in the generated
``kernel_config.py``.

Note that ``tests/ut/conftest.py`` wraps every unit test in a ``PassContext``,
so ``ir.compile(runtime=...)`` is always a conflict here — selecting the runtime
by wrapping the call in a context is the path these tests exercise, and the one
users take in practice.
"""

import pypto.language as pl
import pytest
from pypto.ir.compile import compile as ir_compile
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.pypto_core import passes

DEFAULT_RUNTIME = passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
GRAPH_EXECUTION_RUNTIME = passes.RuntimeKind.HOST_BUILD_GRAPH


@pl.program
class VectorAdd:
    """Minimal orchestration entry, so compilation emits a ``kernel_config.py``."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
            tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
        return c


# ---------------------------------------------------------------------------
# PassContext round-trip
# ---------------------------------------------------------------------------


def test_pass_context_default_runtime_is_tensormap_and_ringbuffer():
    ctx = passes.PassContext([])
    assert ctx.get_runtime() == DEFAULT_RUNTIME


@pytest.mark.parametrize("runtime", [DEFAULT_RUNTIME, GRAPH_EXECUTION_RUNTIME])
def test_pass_context_runtime_round_trip(runtime):
    ctx = passes.PassContext([], runtime=runtime)
    assert ctx.get_runtime() == runtime


def test_pass_context_runtime_is_independent_of_memory_planner():
    ctx = passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, runtime=GRAPH_EXECUTION_RUNTIME)
    assert ctx.get_memory_planner() == passes.MemoryPlanner.PTOAS
    assert ctx.get_runtime() == GRAPH_EXECUTION_RUNTIME


# ---------------------------------------------------------------------------
# Validation and precedence
# ---------------------------------------------------------------------------


def test_compile_rejects_runtime_when_a_context_is_active(tmp_path):
    # Same rule as memory_planner: an explicit argument alongside an active
    # context is ambiguous, so it is refused rather than silently ranked.
    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        with pytest.raises(RuntimeError, match="runtime while a PassContext is already active"):
            ir_compile(
                VectorAdd,
                runtime=DEFAULT_RUNTIME,
                skip_ptoas=True,
                platform="a2a3",
                output_dir=str(tmp_path / "conflict"),
            )


# ---------------------------------------------------------------------------
# The value that lands in the artifact
# ---------------------------------------------------------------------------

# ``kernel_config.py`` is only written when ptoas is not skipped, so these tests
# need ``skip_ptoas=False`` — but they are about the runtime name, not about
# kernel compilation. Stub the ptoas invocation, as the neighbouring codegen
# tests do, so the assertions do not depend on a working ptoas toolchain.
_STUB_PTOAS_OUTPUT = """\
#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void stub_kernel(__gm__ float* v1) {}
"""


@pytest.fixture
def stub_ptoas(monkeypatch):
    monkeypatch.setattr(
        "pypto.backend.pto_backend._compile_pto_module",
        lambda _pto_code, _module_name, _output_dir, _memory_planner=None: _STUB_PTOAS_OUTPUT,
    )


def _compile_and_read_config(out_dir) -> str:
    ir_compile(
        VectorAdd,
        skip_ptoas=False,
        platform="a2a3",
        output_dir=str(out_dir),
        dump_passes=False,
    )
    return (out_dir / "kernel_config.py").read_text()


@pytest.mark.parametrize("runtime", [DEFAULT_RUNTIME, GRAPH_EXECUTION_RUNTIME])
def test_context_runtime_reaches_kernel_config(tmp_path, runtime, stub_ptoas):
    with passes.PassContext([], runtime=runtime):
        config = _compile_and_read_config(tmp_path / passes.runtime_kind_to_name(runtime))
    name = passes.runtime_kind_to_name(runtime)
    assert f'"runtime": "{name}"' in config
    assert f"# Runtime configuration for {name}." in config


def test_compile_without_explicit_runtime_defaults_to_tensormap(tmp_path, stub_ptoas):
    config = _compile_and_read_config(tmp_path / "default")
    assert f'"runtime": "{passes.runtime_kind_to_name(DEFAULT_RUNTIME)}"' in config


@pytest.mark.parametrize("runtime", [DEFAULT_RUNTIME, GRAPH_EXECUTION_RUNTIME])
@pytest.mark.parametrize("contract", ["PROGRAM", "KERNEL"])
def test_program_and_artifact_share_real_pipeline(tmp_path, runtime, contract, monkeypatch):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir._artifact_metadata import ArtifactMetadata  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415

    # The compiler persists each artifact once. Adding the program execution
    # wrapper must not rewrite metadata that the shared producer already owns.
    writes = []
    write = ArtifactMetadata.write

    def record_write(metadata, directory):
        writes.append(directory)
        write(metadata, directory)

    monkeypatch.setattr(ArtifactMetadata, "write", record_write)
    with passes.PassContext([], runtime=runtime):
        program = ir_compile(VectorAdd, output_dir=str(tmp_path / "program"), skip_ptoas=True)
        artifact = _compile_artifact(
            VectorAdd,
            output_dir=str(tmp_path / "artifact"),
            skip_ptoas=True,
            contract=ArtifactExecutionMode[contract],
        )
    assert isinstance(program, CompiledProgram)
    assert isinstance(artifact, CompiledArtifact)
    assert writes == [program.output_dir, artifact.output_dir]
    assert artifact.metadata is not None
    assert program._artifact.metadata is not None
    assert artifact.metadata.return_aliases == (2,)
    assert artifact.metadata.parameters == program._artifact.metadata.parameters
    assert artifact.metadata.lowered_parameters == program._artifact.metadata.lowered_parameters
    assert program.execution_capabilities.record() == ["program"]
    assert artifact.execution_capabilities.record() == [contract.lower()]
    assert artifact._chip_callable is None
    # Real DSL -> IR -> passes -> codegen, with no device/compiler boundary mocked.
    program_sources = {
        p.relative_to(program.output_dir): p.read_text() for p in program.output_dir.rglob("*.pto")
    }
    artifact_sources = {
        p.relative_to(artifact.output_dir): p.read_text() for p in artifact.output_dir.rglob("*.pto")
    }
    assert program_sources and program_sources == artifact_sources
    restored = CompiledArtifact.from_dir(artifact.output_dir)
    assert restored.metadata == artifact.metadata


def test_kernel_rejects_implicit_output_but_program_compiles(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class FreshOutput:
        @pl.function
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.add(a, a)
            return out

    ir_compile(FreshOutput, output_dir=str(tmp_path / "program"), skip_ptoas=True)
    with pytest.raises(ValueError, match="explicit Out/InOut"):
        _compile_artifact(
            FreshOutput,
            output_dir=str(tmp_path / "kernel"),
            skip_ptoas=True,
            contract=ArtifactExecutionMode.KERNEL,
        )


def test_kernel_rejects_hbg_host_data_access_but_program_compiles(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class HostControl:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, control: pl.InOut[pl.Tensor[[2], pl.INT32]], n: pl.Scalar[pl.INT32]):
            value = pl.tensor.read(control, [0])
            pl.tensor.write(control, [1], value + n)

    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        ir_compile(HostControl, output_dir=str(tmp_path / "program"), skip_ptoas=True)
        with pytest.raises(ValueError, match="HBG Host"):
            _compile_artifact(
                HostControl,
                output_dir=str(tmp_path / "kernel"),
                skip_ptoas=True,
                contract=ArtifactExecutionMode.KERNEL,
            )


def test_kernel_persists_logical_and_lowered_nz_signatures(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class NzMatmul:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64, 128], pl.FP16, pl.NZ],
            b: pl.Tensor[[16, 128], pl.FP16],
            c: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                weight = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
                tile = pl.load(b, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
                acc = pl.matmul(tile, pl.tile.transpose_view(weight), out_dtype=pl.FP32)
                pl.store(acc, [0, 0], c)
            return c

    kernel = _compile_artifact(
        NzMatmul, output_dir=str(tmp_path), skip_ptoas=True, contract=ArtifactExecutionMode.KERNEL
    )
    assert isinstance(kernel, CompiledArtifact) and kernel.metadata is not None
    assert kernel.metadata.parameters[0].shape == (64, 128)
    assert kernel.metadata.lowered_parameters[0].shape == (1, 8, 4, 16, 16)
    assert kernel.metadata.return_aliases == (2,)


def test_kernel_multiple_returns_keep_full_parameter_indices(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class MultipleReturns:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            n: pl.Scalar[pl.INT32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [16, 16])
                pl.store(tile, [0, 0], c)
            return c, c

    kernel = _compile_artifact(
        MultipleReturns, output_dir=str(tmp_path), skip_ptoas=True, contract=ArtifactExecutionMode.KERNEL
    )
    assert isinstance(kernel, CompiledArtifact) and kernel.metadata is not None
    assert kernel.metadata.parameters[1].shape is None
    assert kernel.metadata.return_aliases == (2, 2)


@pytest.mark.parametrize("contract", ["PROGRAM", "KERNEL"])
def test_single_chip_artifact_rejects_multiple_entries_before_codegen(tmp_path, contract):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class MultiEntry:
        @pl.function(type=pl.FunctionType.Orchestration)
        def first(self, a: pl.Tensor[[16, 16], pl.FP32]):
            pass

        @pl.function(type=pl.FunctionType.Orchestration)
        def second(self, a: pl.Tensor[[16, 16], pl.FP32]):
            pass

    with pytest.raises(ValueError, match="exactly one single-chip entry"):
        _compile_artifact(
            MultiEntry, output_dir=str(tmp_path), skip_ptoas=True, contract=ArtifactExecutionMode[contract]
        )
    assert not (tmp_path / "compiled_meta.json").exists()


@pytest.mark.parametrize("contract", ["PROGRAM", "KERNEL"])
def test_single_chip_artifact_rejects_distributed_entry_before_codegen(tmp_path, contract):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class DistributedEntry:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]):
            pass

    with pytest.raises(ValueError, match="single chip"):
        _compile_artifact(
            DistributedEntry,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            contract=ArtifactExecutionMode[contract],
        )
    assert not (tmp_path / "compiled_meta.json").exists()


def test_kernel_preserves_dynamic_shape_and_scalar_slots(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    rows = pl.dynamic("rows")

    @pl.program
    class DynamicCopy:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[rows, 16], pl.FP32],
            n: pl.Scalar[pl.INT32],
            c: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
        ) -> pl.Tensor[[rows, 16], pl.FP32]:
            for i in pl.range(0, n):
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile = pl.load(a, [i * 16, 0], [16, 16])
                    pl.store(tile, [i * 16, 0], c)
            return c

    kernel = _compile_artifact(
        DynamicCopy, output_dir=str(tmp_path), skip_ptoas=True, contract=ArtifactExecutionMode.KERNEL
    )
    assert isinstance(kernel, CompiledArtifact) and kernel.metadata is not None
    assert kernel.metadata.parameters[0].shape == (-1, 16)
    assert kernel.metadata.parameters[1].shape is None
    assert kernel.metadata.parameters == kernel.metadata.lowered_parameters
    assert kernel.metadata.return_aliases == (2,)


@pytest.mark.parametrize("multi_entry", [False, True])
def test_program_preserves_original_mx_parameters(tmp_path, multi_entry):
    import torch  # noqa: PLC0415
    from pypto import ir  # noqa: PLC0415
    from pypto.backend import BackendType  # noqa: PLC0415
    from pypto.ir.compiled_program import (  # noqa: PLC0415
        CompiledProgram,
        _coerce_args,
        _extract_func_param_infos,
    )
    from pypto.runtime import DeviceTensor  # noqa: PLC0415

    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is required for MX DeviceTensor arguments")

    @pl.program
    class MxInputs:
        @pl.function(type=pl.FunctionType.InCore)
        def load_scales(
            self,
            a: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
        ):
            _a_tile = pl.load(a, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
            _b_tile = pl.load(b, [0, 0], [2, 16], target_memory=pl.Mem.Mat)

        @pl.function(type=pl.FunctionType.Orchestration)
        def first(
            self,
            a: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
        ):
            self.load_scales(a, b)

        @pl.function(type=pl.FunctionType.Orchestration)
        def second(
            self,
            a: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ],
            b: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
        ):
            self.load_scales(a, b)

    source = MxInputs
    if not multi_entry:
        source = ir.Program(
            [func for func in source.functions.values() if func.name != "second"],
            source.name,
            source.span,
        )
    compiled = ir_compile(
        source,
        output_dir=str(tmp_path),
        backend_type=BackendType.Ascend950,
        skip_ptoas=True,
        dump_passes=False,
    )
    assert isinstance(compiled, CompiledProgram)
    entries = [compiled["first"], compiled["second"]] if multi_entry else [compiled]
    original_entry = source.get_function("first")
    assert original_entry is not None
    expected_params, expected_outputs, expected_returns = _extract_func_param_infos(original_entry)
    for entry in entries:
        metadata = entry._artifact.metadata
        assert metadata is not None
        assert [param.shape for param in metadata.lowered_parameters] == [(1, 4, 2, 16, 2)] * 2
        for program in (entry, CompiledProgram.from_dir(entry.output_dir)):
            params, outputs, returns = program._get_metadata()
            assert params == expected_params
            assert outputs == expected_outputs
            assert len(returns) == len(expected_returns)
            assert program.param_names == ["a", "b"]
            args = (
                DeviceTensor(0x1000, (64, 4), scale_dtype),
                DeviceTensor(0x2000, (4, 64), scale_dtype),
            )
            coerced, return_style = _coerce_args(args, params, outputs, returns, caller_name="program")
            assert all(actual is original for actual, original in zip(coerced, args, strict=True))
            assert not return_style


def test_kernel_contract_allows_omitted_internal_out():
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir._compile_contract import analyze_contract  # noqa: PLC0415

    @pl.program
    class InternalOutput:
        @pl.function(type=pl.FunctionType.InCore)
        def copy(
            self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile = pl.load(a, [0, 0], [16, 16])
            pl.store(tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                scratch, first = pl.submit(self.copy, a)
                out, second = pl.submit(self.copy, scratch, out, deps=[first])
            return out

    params, aliases = analyze_contract(InternalOutput, ArtifactExecutionMode.KERNEL)
    assert params is not None
    assert [param.name for param in params] == ["a", "out"]
    assert aliases == (1,)


def test_kernel_contract_preserves_single_submit_return_alias():
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir._compile_contract import analyze_contract  # noqa: PLC0415

    @pl.program
    class SubmittedAlias:
        @pl.function(type=pl.FunctionType.InCore)
        def forward(self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                result, task_id = pl.submit(self.forward, out)
            return result

    params, aliases = analyze_contract(SubmittedAlias, ArtifactExecutionMode.KERNEL)
    assert params is not None
    assert [param.name for param in params] == ["out"]
    assert aliases == (0,)


def test_kernel_cannot_return_a_narrowed_external_view(tmp_path):
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    @pl.program
    class NarrowedReturn:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
            return pl.slice(a, [8, 16], [0, 0])

    with pytest.raises(ValueError, match="returns must alias external"):
        _compile_artifact(
            NarrowedReturn, output_dir=str(tmp_path), skip_ptoas=True, contract=ArtifactExecutionMode.KERNEL
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
