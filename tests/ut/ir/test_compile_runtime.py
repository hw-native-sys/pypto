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

import ast
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir.compile import compile as ir_compile
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.ir.compiled_program import CompiledProgram
from pypto.ir.param_info import block_nz_shape
from pypto.pypto_core import ir, passes
from pypto.runtime import DeviceTensor
from pypto.runtime._binary_cache import BinaryCacheContext

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


def test_verified_waiter_and_sync_kernel_reach_real_registration_path(
    tmp_path, stub_ptoas, device_runner, monkeypatch
):
    """Exercise the complete IR/config/assembly chain; only native compilers are stubbed."""

    @pl.program
    class Program:
        @pl.function
        def main(
            self,
            signal: pld.DistributedTensor[[4, 1], pl.INT32],
            data: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                for peer in pl.range(4):
                    pld.system.defer_wait(signal, offsets=[peer, 0], expected=1, cmp=pld.WaitCmp.Ge)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                tile = pl.load(data, [0, 0], [16, 16])
                out = pl.store(tile, [0, 0], out)
            return out

    out_dir = tmp_path / "qualified"
    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        ir_compile(Program, output_dir=str(out_dir), skip_ptoas=False, platform="a2a3", dump_passes=False)
    native = SimpleNamespace(
        ArgDirection=SimpleNamespace(IN="IN", OUT="OUT", INOUT="INOUT", SCALAR="SCALAR"),
        CoreAsyncMode=SimpleNamespace(SYNCHRONOUS="native-sync", DEFERRED="native-deferred"),
        CoreAsyncCapability=lambda **fields: fields,
    )
    monkeypatch.setitem(sys.modules, "simpler.task_interface", native)
    monkeypatch.setattr(device_runner, "ensure_pto_isa_root", lambda: "/test-pto-isa")
    monkeypatch.setattr(device_runner, "KernelCompiler", lambda **kwargs: object())
    monkeypatch.setattr(
        device_runner,
        "_current_binary_context",
        lambda *args, **kwargs: BinaryCacheContext(
            platform="a2a3sim",
            runtime_name="host_build_graph",
            runtime_revision="a" * 40,
            pto_isa_revision="b" * 40,
        ),
    )
    monkeypatch.setattr(device_runner, "_compile_single_orchestration", lambda *args, **kwargs: b"orch")
    monkeypatch.setattr(device_runner, "_compile_single_kernel", lambda *args, **kwargs: (b"raw", b"kernel"))
    monkeypatch.setattr(device_runner, "register_callable_identity", lambda *args: None)
    monkeypatch.setattr(device_runner, "CoreCallable", SimpleNamespace(build=lambda **fields: fields))
    monkeypatch.setattr(device_runner, "ChipCallable", SimpleNamespace(build=lambda **fields: fields))
    callable_, _, _ = device_runner._compile_and_assemble_locked(out_dir, "a2a3sim")
    capabilities = [child["async_capability"] for _, child in callable_["children"]]
    assert sorted((c["mode"], c["max_conditions"], c["condition_types"]) for c in capabilities) == [
        ("native-deferred", 4, 1),
        ("native-sync", 0, 0),
    ]


def _three_moe_waiters(n_ranks: int):
    """Keep the three waiter loops and DAG from PyPTO-Lib decode_moe.py.

    Source: 26ef3e865733747a89c3356ec145e567851ad985, dispatch and
    combine_scattered. Payload computation is reduced to one row; the meta
    waiter intentionally has no dependency on the local meta publisher.
    """

    @pl.program
    class ThreeMoeWaiters:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            inp: pl.Tensor[[1, 8], pl.INT32],
            payload: pl.InOut[pld.DistributedTensor[[n_ranks, 8], pl.INT32]],
            arrived: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
            data_arrived: pl.InOut[pld.DistributedTensor[[n_ranks, 1, 1], pl.INT32]],
            combine_arrived: pl.InOut[pld.DistributedTensor[[n_ranks, 1, 1], pl.INT32]],
            out: pl.Out[pl.Tensor[[3, 8], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
            moe_epoch: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[3, 8], pl.INT32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_push") as meta_push:
                    row = pl.load(inp, [0, 0], [1, 8])
                    for dst in pl.range(n_ranks):
                        pld.tile.remote_store(row, target=payload, peer=dst, offsets=[my_rank, 0])
                        if dst != my_rank:
                            pld.system.notify(
                                arrived, peer=dst, offsets=[my_rank, 0], value=moe_epoch, op=pld.NotifyOp.Set
                            )
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as meta_wait:
                    for src in pl.range(n_ranks):
                        if src != my_rank:
                            pld.system.defer_wait(arrived, [src, 0], moe_epoch, cmp=pld.WaitCmp.Ge)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="dispatch_meta_finalize",
                    deps=[meta_push, meta_wait],
                    allow_early_resolve=True,
                ) as meta_finalize:
                    row = pl.load(payload, [my_rank, 0], [1, 8])
                    out = pl.store(row, [0, 0], out)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_push") as push:
                    row = pl.load(inp, [0, 0], [1, 8])
                    for dst in pl.range(n_ranks):
                        pld.tile.remote_store(row, target=payload, peer=dst, offsets=[my_rank, 0])
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="dispatch_publish_epoch",
                    deps=[push],
                    allow_early_resolve=False,
                ) as publish:
                    for dst in pl.range(n_ranks):
                        if dst != my_rank:
                            pld.system.notify(
                                data_arrived,
                                peer=dst,
                                offsets=[my_rank, 0, 0],
                                value=moe_epoch,
                                op=pld.NotifyOp.Set,
                            )
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[publish]) as wait:
                    for src in pl.range(n_ranks):
                        if src != my_rank:
                            pld.system.defer_wait(data_arrived, [src, 0, 0], moe_epoch, cmp=pld.WaitCmp.Ge)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="dispatch_gather",
                    deps=[wait, push, meta_finalize],
                ) as scatter_done:
                    row = pl.load(payload, [my_rank, 0], [1, 8])
                    out = pl.store(row, [1, 0], out)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="combine_notify",
                    deps=[scatter_done],
                    allow_early_resolve=False,
                ) as notify:
                    for peer in pl.range(n_ranks):
                        if peer != my_rank:
                            pld.system.notify(
                                combine_arrived,
                                peer=peer,
                                offsets=[my_rank, 0, 0],
                                value=moe_epoch,
                                op=pld.NotifyOp.Set,
                            )
                with pl.at(
                    level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify]
                ) as combine_wait:
                    for src in pl.range(n_ranks):
                        if src != my_rank:
                            pld.system.defer_wait(combine_arrived, [src, 0, 0], moe_epoch, cmp=pld.WaitCmp.Ge)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_reduce", deps=[combine_wait]):
                    row = pl.load(payload, [my_rank, 0], [1, 8])
                    out = pl.store(row, [2, 0], out)
            return out

    return ThreeMoeWaiters


def _waiter_registrations(function, rank: int, epoch: int):
    """Evaluate only scalar control in the compiled registration-only waiter IR."""
    inputs = {"my_rank": rank, "moe_epoch": epoch}
    values = {
        param.unique_id: inputs[param.name_hint.split("__ssa", 1)[0]]
        for param in function.params
        if param.name_hint.split("__ssa", 1)[0] in inputs
    }
    records = []

    def scalar(expr):
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.Var):
            return values[expr.unique_id]
        if isinstance(expr, ir.Cast):
            return scalar(expr.operand)
        if isinstance(expr, ir.Ne):
            return scalar(expr.left) != scalar(expr.right)
        raise AssertionError(f"Unexpected waiter scalar expression: {expr}")

    def visit(stmt):
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                visit(child)
        elif isinstance(stmt, ir.ForStmt):
            assert not stmt.iter_args
            for index in range(scalar(stmt.start), scalar(stmt.stop), scalar(stmt.step)):
                values[stmt.loop_var.unique_id] = index
                visit(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            branch = stmt.then_body if scalar(stmt.condition) else stmt.else_body
            if branch is not None:
                visit(branch)
        elif isinstance(stmt, ir.AssignStmt):
            values[stmt.var.unique_id] = scalar(stmt.value)
        elif isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            assert isinstance(call, ir.Call)
            assert call.op.name == ir.get_op("pld.system.defer_wait").name
            offsets = call.args[1]
            assert isinstance(offsets, ir.MakeTuple)
            records.append((tuple(scalar(v) for v in offsets.elements), scalar(call.args[2])))
        elif isinstance(stmt, (ir.ReturnStmt, ir.YieldStmt)):
            assert not stmt.value
        else:
            raise AssertionError(f"Unexpected waiter statement: {stmt}")

    visit(function.body)
    return records


@pytest.mark.parametrize("n_ranks", [1, 2, 8, 64])
def test_three_moe_waiters_keep_bounds_registration_counts_and_dag(tmp_path, stub_ptoas, n_ranks):
    """Real passes/config/codegen; PTOAS alone is stubbed in this portable UT."""
    from pypto.ir.compile import _compile_to_files  # noqa: PLC0415

    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        compiled = _compile_to_files(
            _three_moe_waiters(n_ranks), output_dir=str(tmp_path), platform="a5", dump_passes=False
        )
    functions = {f.name: f for f in compiled.lowered_program.functions.values()}
    waiters = {name: f for name, f in functions.items() if f.attrs.get("deferred_completion_waiter", False)}
    # Identical dispatch/combine waiter code may be deduplicated. The contract
    # is three independent task invocations, not three different machine codes.
    assignments = {}

    class Assignments(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            assignments[op.var.unique_id] = op.value
            super().visit_assign_stmt(op)

    Assignments().visit_stmt(functions["main"].body)
    submits = [value for value in assignments.values() if isinstance(value, ir.Submit)]
    waiter_calls = [call for call in submits if call.op.name in waiters]
    assert len(waiter_calls) == 3

    def signal_name(call):
        signals = [arg for arg in call.args if isinstance(arg.type, ir.DistributedTensorType)]
        assert len(signals) == 1
        return signals[0].name_hint.split("__ssa", 1)[0]

    assert {signal_name(call) for call in waiter_calls} == {"arrived", "data_arrived", "combine_arrived"}
    for call in waiter_calls:
        function = waiters[call.op.name]
        rank_dims = 2 if signal_name(call) == "arrived" else 3
        assert function.func_type == ir.FunctionType.AIV
        assert function.attrs["deferred_completion_max_count"] == n_ranks
        for rank in range(n_ranks):
            assert _waiter_registrations(function, rank, 17) == [
                ((peer, *([0] * (rank_dims - 1))), 17) for peer in range(n_ranks) if peer != rank
            ]

    # Read only emitted literal metadata: importing config would require the
    # optional native Simpler binding and is outside this portable compiler UT.
    config = ast.parse((tmp_path / "kernel_config.py").read_text())
    kernels = next(
        node.value
        for node in config.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets)
    )
    assert isinstance(kernels, ast.List)
    qualified = {}
    for node in kernels.elts:
        assert isinstance(node, ast.Dict)
        fields = {}
        for key, value in zip(node.keys, node.values, strict=True):
            assert key is not None
            fields[ast.literal_eval(key)] = value
        qualified[ast.literal_eval(fields["name"])] = ast.literal_eval(fields["async_capability"])
    for name in waiters:
        assert qualified[name] == dict(version=1, mode="DEFERRED", max_conditions=n_ranks, condition_types=1)
    assert all(c["mode"] == "SYNCHRONOUS" for name, c in qualified.items() if name not in waiters)

    def role(call):
        if call.op.name in waiters:
            return {
                "arrived": "dispatch_meta_wait",
                "data_arrived": "dispatch_wait",
                "combine_arrived": "combine_wait",
            }[signal_name(call)]
        if call.op.name in {"dispatch_publish_epoch", "combine_notify"}:
            return {"data_arrived": "dispatch_publish_epoch", "combine_arrived": "combine_notify"}[
                signal_name(call)
            ]
        return call.op.name

    def producer(expr):
        if isinstance(expr, ir.Var):
            return producer(assignments[expr.unique_id])
        if isinstance(expr, ir.TupleGetItemExpr):
            return producer(expr.tuple)
        assert isinstance(expr, ir.Submit), expr
        return role(expr)

    launches = {role(call): call for call in submits}
    dependencies = {name: {producer(dep) for dep in call.deps} for name, call in launches.items()}
    assert {"dispatch_meta_wait", "dispatch_wait", "combine_wait"}.issubset(launches)
    assert dependencies["dispatch_meta_wait"] == set()
    assert dependencies["dispatch_meta_finalize"] == {"dispatch_meta_push", "dispatch_meta_wait"}
    assert dependencies["dispatch_wait"] == {"dispatch_publish_epoch"}
    assert dependencies["dispatch_gather"] == {"dispatch_wait", "dispatch_push", "dispatch_meta_finalize"}
    assert dependencies["combine_wait"] == {"combine_notify"}
    assert dependencies["combine_reduce"] == {"combine_wait"}


def test_three_moe_waiters_reject_static_bound_above_capacity(tmp_path, stub_ptoas):
    """Sixty-four actual peers do not make a static bound of 65 admissible."""
    from pypto.ir.compile import _compile_to_files  # noqa: PLC0415

    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        with pytest.raises(ValueError, match="at most 64 conditions"):
            _compile_to_files(
                _three_moe_waiters(65), output_dir=str(tmp_path), platform="a5", dump_passes=False
            )


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


@pytest.mark.parametrize(
    ("pre_lowered", "multi_entry"),
    [(False, False), (True, False), (False, True), (True, True), ("mixed", True)],
)
def test_program_validates_nz_against_lowered_signature(tmp_path, pre_lowered, multi_entry):
    from pypto import ir  # noqa: PLC0415

    @pl.program
    class NzEntries:
        @pl.function(type=pl.FunctionType.Orchestration)
        def first(self, weight: pl.Tensor[[64, 128], pl.FP16, pl.NZ]):
            return

        @pl.function(type=pl.FunctionType.Orchestration)
        def second(self, weight: pl.Tensor[[64, 128], pl.FP16, pl.NZ]):
            return

    source = NzEntries
    if not multi_entry:
        source = ir.Program(
            [func for func in source.functions.values() if func.name != "second"],
            source.name,
            source.span,
        )
    if pre_lowered:
        lowered = passes.block_nz_tensor_views()(source)
        if pre_lowered == "mixed":
            first = source.get_function("first")
            second = lowered.get_function("second")
            assert first is not None and second is not None
            source = ir.Program([first, second], source.name, source.span)
        else:
            source = lowered
    compiled = ir_compile(source, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    entries = [compiled["first"], compiled["second"]] if multi_entry else [compiled]
    for index, entry in enumerate(entries):
        blocked_entry = index == 1 if pre_lowered == "mixed" else pre_lowered
        for program in (entry, CompiledProgram.from_dir(entry.output_dir)):
            metadata = program._artifact.metadata
            assert metadata is not None
            assert metadata.parameters[0].shape == ((1, 8, 4, 16, 16) if blocked_entry else (64, 128))
            assert metadata.lowered_parameters[0].shape == (1, 8, 4, 16, 16)
            # Exercise public calls and TaskArgs construction without native
            # execution. Static logical ranks may differ when storage agrees.
            with (
                patch("pypto.runtime.runner._execute_compiled") as execute,
                patch.object(type(program), "_host_tensor_indices", return_value=()),
                patch("pypto.runtime.runner._coerced_to_orch_args", return_value="args"),
            ):
                for shape in [(64, 128), (1, 64, 128)]:
                    arg = DeviceTensor(0x1000, shape, torch.float16)
                    assert program(arg) is None
                    _, coerced, _ = program._build_orch_args(arg, worker=object())
                    assert coerced == [arg]
                assert execute.call_count == 2
                with pytest.raises(TypeError, match="expects shape"):
                    program(DeviceTensor(0x1000, (128, 64), torch.float16))
                with pytest.raises(TypeError, match="expects dtype"):
                    program(DeviceTensor(0x1000, (64, 128), torch.int8))
                assert execute.call_count == 2


@pytest.mark.parametrize(
    ("shape", "dtype", "torch_dtype"),
    [
        ((5120, 1280), pl.FP8E4M3FN, getattr(torch, "float8_e4m3fn", None)),
        ((2, 3, 4, 256, 512), pl.INT8, torch.int8),
        ((2, 3, 4, 5, 256, 512), pl.INT8, torch.int8),
        ((1, 4, 16, 16, 32), pl.INT8, torch.int8),
    ],
)
def test_program_uses_pass_produced_nz_shape(tmp_path, shape, dtype, torch_dtype):
    if torch_dtype is None:
        pytest.skip("Torch does not provide this optional dtype")

    @pl.program
    class NzEntry:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, weight: pl.Tensor[shape, dtype, pl.NZ]):
            return

    compiled = ir_compile(NzEntry, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    for program in (compiled, CompiledProgram.from_dir(tmp_path)):
        metadata = program._artifact.metadata
        assert metadata is not None
        assert metadata.parameters[0].shape == shape
        assert metadata.lowered_parameters[0].shape == tuple(block_nz_shape(shape, torch_dtype))
        with patch("pypto.runtime.runner._execute_compiled") as execute:
            assert program(DeviceTensor(0x1000, shape, torch_dtype)) is None
            execute.assert_called_once()


@pytest.mark.parametrize(
    "dtype,carrier_dtype,c0",
    [
        (pl.HF8, torch.uint8, 32),
        (pl.FP8E4M3FN, torch.uint8, 32),
        (pl.UINT16, torch.int16, 16),
        (pl.UINT32, torch.int32, 8),
        (pl.UINT64, torch.int64, 4),
    ],
)
def test_program_validates_nz_without_torch_dtype_mapping(tmp_path, monkeypatch, dtype, carrier_dtype, c0):
    from pypto.ir import param_info  # noqa: PLC0415

    # HF8 has no Torch equivalent. Remove optional mappings to reproduce
    # older supported Torch versions without requiring those versions in CI.
    monkeypatch.delitem(param_info._DATATYPE_TO_TORCH, str(dtype), raising=False)

    @pl.program
    class NzEntry:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, weight: pl.Tensor[[32, 64], dtype, pl.NZ]):
            return

    compiled = ir_compile(NzEntry, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    for program in (compiled, CompiledProgram.from_dir(tmp_path)):
        metadata = program._artifact.metadata
        assert metadata is not None
        assert metadata.parameters[0].shape == (32, 64)
        assert metadata.lowered_parameters[0].shape == (1, 64 // c0, 2, 16, c0)
        arg = DeviceTensor(0x1000, (32, 64), carrier_dtype)
        with (
            patch("pypto.runtime.runner._execute_compiled") as execute,
            patch.object(type(program), "_host_tensor_indices", return_value=()),
            patch("pypto.runtime.runner._coerced_to_orch_args", return_value="args"),
        ):
            assert program(arg) is None
            _, coerced, _ = program._build_orch_args(arg, worker=object())
            assert coerced == [arg]
            # Missing a Torch dtype mapping must not disable shape checks.
            with pytest.raises(TypeError, match="expects shape"):
                program(DeviceTensor(0x1000, (16, 128), carrier_dtype))
            # Equal logical shapes do not make a carrier with a different
            # element width compatible with the compiler's NZ storage view.
            wrong_dtype = torch.float16 if c0 == 32 else torch.uint8
            wrong_carrier = DeviceTensor(0x1000, (32, 64), wrong_dtype)
            with pytest.raises(TypeError, match=f"expects {256 // c0}-bit storage"):
                program(wrong_carrier)
            with pytest.raises(TypeError, match=f"expects {256 // c0}-bit storage"):
                program._build_orch_args(wrong_carrier, worker=object())
            execute.assert_called_once()


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
