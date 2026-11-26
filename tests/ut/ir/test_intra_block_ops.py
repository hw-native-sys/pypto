# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Named intra-block contract, round-trip, ownership, and real PTOAS lowering."""

import re
import subprocess
import sys

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import system_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def a5_backend():
    """Use A5 independently of the global backend selected by other tests."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


@pytest.mark.parametrize("action", ["set", "wait"])
@pytest.mark.parametrize("event", [0, 15, 16, 17, 31])
@pytest.mark.parametrize(
    "pipe",
    [pl.PipeType.FIX, pl.PipeType.MTE1, pl.PipeType.MTE2, pl.PipeType.MTE3, pl.PipeType.V, pl.PipeType.S],
)
def test_static_contract(action, event, pipe):
    call = getattr(system_ops, f"{action}_intra_block")(event, pipe=pipe)
    assert isinstance(call.type, ir.UnknownType)
    assert call.args == []
    assert call.kwargs == {"pipe": int(pipe), "event_id": event}


@pytest.mark.parametrize("action", ["set", "wait"])
@pytest.mark.parametrize("event", [-1, 32])
def test_static_range(action, event):
    with pytest.raises(ValueError, match=r"static event_id must be in \[0, 31\]"):
        getattr(system_ops, f"{action}_intra_block")(event, pipe=pl.PipeType.S)


@pytest.mark.parametrize("pipe", [pl.PipeType.M, pl.PipeType.ALL, -1, 99])
@pytest.mark.parametrize("action", ["set", "wait"])
def test_invalid_pipe(action, pipe):
    with pytest.raises(ValueError, match="pipe must be FIX, MTE1, MTE2, MTE3, V, or S"):
        getattr(system_ops, f"{action}_intra_block")(17, pipe=pipe)


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64])
@pytest.mark.parametrize("action", ["set", "wait"])
def test_dynamic_scalar_unwrap(action, dtype):
    var = ir.Var("event", ir.ScalarType(dtype), ir.Span.unknown())
    call = getattr(pl.system, f"{action}_intra_block")(pl.Scalar(dtype, var), pipe=pl.PipeType.S)
    assert call.args == [var]
    assert "event_id" not in call.kwargs


@pytest.mark.parametrize("dtype", [DataType.INDEX, DataType.INT16, DataType.UINT32, DataType.FP32])
@pytest.mark.parametrize("action", ["set", "wait"])
def test_invalid_dynamic_type(action, dtype):
    var = ir.Var("event", ir.ScalarType(dtype), ir.Span.unknown())
    with pytest.raises(ValueError, match="cast INDEX explicitly"):
        getattr(system_ops, f"{action}_intra_block")(var, pipe=pl.PipeType.S)


@pytest.mark.parametrize("action", ["set", "wait"])
@pytest.mark.parametrize("event", [True, 1.5, "17", None])
def test_invalid_python_event(action, event):
    with pytest.raises(TypeError, match="event_id must be int or Expr"):
        getattr(system_ops, f"{action}_intra_block")(event, pipe=pl.PipeType.S)


@pytest.mark.parametrize("action", ["set", "wait"])
def test_raw_ir_validates_exclusive_event_forms(action):
    var = ir.Var("event", ir.ScalarType(DataType.INT32), ir.Span.unknown())
    name = f"system.{action}_intra_block"
    for args, kwargs in [
        ([], {"pipe": int(pl.PipeType.S)}),
        ([var], {"pipe": int(pl.PipeType.S), "event_id": 17}),
    ]:
        with pytest.raises(ValueError, match="exactly one"):
            ir.create_op_call(name, args, kwargs, ir.Span.unknown())
    with pytest.raises(ValueError, match="requires a pipe"):
        ir.create_op_call(name, [], {"event_id": 17}, ir.Span.unknown())
    with pytest.raises(ValueError, match="ffts_mode"):
        ir.create_op_call(
            name, [], {"pipe": int(pl.PipeType.S), "event_id": 17, "ffts_mode": 4}, ir.Span.unknown()
        )


@pytest.mark.parametrize("action", ["set", "wait"])
def test_kernel_domain(action):
    op = getattr(pl.system, f"{action}_intra_block")
    with pytest.raises(TypeError, match="KernelType member"):
        op(17, pipe=pl.PipeType.S, core_type="aiv")
    with pytest.raises(ValueError, match="Omit core_type only in an explicitly typed"):
        op(17, pipe=pl.PipeType.S, core_type=pl.KernelType.MIX)


@pl.program
class NamedEvents:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, event32: pl.Scalar[pl.INT32], event64: pl.Scalar[pl.INT64]):
        pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
        pl.system.wait_intra_block(17, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
        pl.system.set_intra_block(event32, pipe=pl.PipeType.S, core_type=pl.KernelType.AIV)
        pl.system.wait_intra_block(event64, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
        pl.system.set_intra_block(31, pipe=pl.PipeType.S, core_type=pl.KernelType.AIC)
        pl.system.wait_intra_block(31, pipe=pl.PipeType.S, core_type=pl.KernelType.AIV)


def _calls(function):
    return [
        stmt.expr
        for stmt in ir.flatten_to_stmts(function.body)
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]


def test_printer_round_trip():
    printed = NamedEvents.as_python()
    assert "pipe=pl.PipeType.S" in printed
    assert "core_type=pl.KernelType.AIC" in printed
    assert "core_type=pl.KernelType.AIV" in printed
    ir.assert_structural_equal(NamedEvents, pl.parse_program(printed))


@pytest.mark.parametrize("default_passes", [False, True])
def test_mixed_ownership_and_order(default_passes):
    Before = NamedEvents

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def main_aic(self, event32: pl.Scalar[pl.INT32], event64: pl.Scalar[pl.INT64]):
            pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(event64, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.set_intra_block(31, pipe=pl.PipeType.S, core_type=pl.KernelType.AIC)

        @pl.function(type=pl.FunctionType.AIV)
        def main_aiv(self, event32: pl.Scalar[pl.INT32], event64: pl.Scalar[pl.INT64]):
            pl.system.wait_intra_block(17, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
            pl.system.set_intra_block(event32, pipe=pl.PipeType.S, core_type=pl.KernelType.AIV)
            pl.system.wait_intra_block(31, pipe=pl.PipeType.S, core_type=pl.KernelType.AIV)

        @pl.function(type=pl.FunctionType.Group)
        def main(self, event32: pl.Scalar[pl.INT32], event64: pl.Scalar[pl.INT64]):
            self.main_aic(event32, event64)
            self.main_aiv(event32, event64)

    if default_passes:
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        after = manager.run_passes(Before)
        expected = manager.run_passes(Expected)
        # Typed reference peers already existed during tensor conversion;
        # generated peers did not. Canonicalize their implicit void returns.
        comparison_after = passes.convert_tensor_to_tile_ops()(after)
        comparison_expected = passes.convert_tensor_to_tile_ops()(expected)
    else:
        before = passes.infer_tile_memory_space()(passes.convert_to_ssa()(Before))
        after = passes.expand_mixed_kernel()(before)
        comparison_after = after
        comparison_expected = passes.infer_tile_memory_space()(passes.convert_to_ssa()(Expected))
    # Explicit peers keep the reference independent of mixed-kernel splitting.
    # Compare full signatures, operands, pipes, attributes and group calls.
    ir.assert_structural_equal(comparison_after, comparison_expected)
    aic = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIC)
    aiv = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert [(c.op.name, c.kwargs.get("event_id")) for c in _calls(aic)] == [
        ("system.set_intra_block", 17),
        ("system.wait_intra_block", None),
        ("system.set_intra_block", 31),
    ]
    assert [(c.op.name, c.kwargs.get("event_id")) for c in _calls(aiv)] == [
        ("system.wait_intra_block", 17),
        ("system.set_intra_block", None),
        ("system.wait_intra_block", 31),
    ]


def test_unpinned_event_cannot_be_duplicated_in_mixed_kernel():
    @pl.program
    class Unpinned:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self):
            pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(17, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
            pl.system.set_intra_block(3, pipe=pl.PipeType.S)

    with pytest.raises(ValueError, match="unpinned named event cannot be duplicated"):
        passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(Unpinned)))


@pl.program
class TypedNamedEvents:
    @pl.function(type=pl.FunctionType.AIV)
    def named_events(self, event32: pl.Scalar[pl.INT32], event64: pl.Scalar[pl.INT64]):
        pl.system.set_intra_block(0, pipe=pl.PipeType.MTE3)
        pl.system.wait_intra_block(17, pipe=pl.PipeType.MTE2)
        pl.system.set_intra_block(event32, pipe=pl.PipeType.MTE3)
        pl.system.wait_intra_block(event64, pipe=pl.PipeType.V)
        pl.system.set_intra_block(31, pipe=pl.PipeType.MTE3)


def test_named_event_emission(tmp_path, validate_native=False):
    """Check emission; the codegen companion opts into real native verification."""
    text = codegen.PTOCodegen().generate(TypedNamedEvents)
    assert "pto.set_intra_block <PIPE_MTE3>, 0" in text
    assert "pto.wait_intra_block <PIPE_MTE2>, 17" in text
    assert "pto.set_intra_block <PIPE_MTE3>, %arg0 : i32" in text
    assert "pto.wait_intra_block <PIPE_V>, %arg1 : i64" in text
    assert "pto.set_intra_block <PIPE_MTE3>, 31" in text
    if not validate_native:
        return
    compiler = find_ptoas_binary()
    if not compiler:
        pytest.skip("PTOAS not installed: emission assertions passed, compiler validation unavailable")
    source = tmp_path / "named.pto"
    source.write_text(text)
    result = subprocess.run(
        [compiler, "--pto-arch=a5", str(source)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    (tmp_path / "named.cpp").write_text(result.stdout)
    assert "set_intra_block(PIPE_MTE3, 0)" in result.stdout
    assert "wait_intra_block(PIPE_MTE2, 17)" in result.stdout
    assert "set_intra_block(PIPE_MTE3, 31)" in result.stdout
    signature = re.search(r"named_events\(int32_t (\w+), int64_t (\w+)\)", result.stdout)
    assert signature is not None, result.stdout
    event32, event64 = signature.groups()
    assert f"set_intra_block(PIPE_MTE3, {event32})" in result.stdout
    wait = re.search(r"wait_intra_block\(PIPE_V, (\w+)\)", result.stdout)
    assert wait is not None, result.stdout
    wait_operand = wait.group(1)
    assert wait_operand == event64 or re.search(
        rf"int32_t\s+{re.escape(wait_operand)}\s*=\s*(?:\(int32_t\)\s*)?{re.escape(event64)}\s*;",
        result.stdout,
    ), result.stdout
    assert "ffts_cross_core_sync" not in result.stdout
    # The compiler's own PTO printer must preserve the typed event operands.
    printed = subprocess.run(
        [compiler, "--pto-arch=a5", "--emit-pto-ir", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    signature = re.search(r"@named_events\((%\w+): i32, (%\w+): i64\)", printed.stdout)
    assert signature is not None, printed.stdout
    event32, event64 = signature.groups()
    expected_events = [
        "pto.set_intra_block <PIPE_MTE3>, 0",
        "pto.wait_intra_block <PIPE_MTE2>, 17",
        f"pto.set_intra_block <PIPE_MTE3>, {event32} : i32",
        f"pto.wait_intra_block <PIPE_V>, {event64} : i64",
        "pto.set_intra_block <PIPE_MTE3>, 31",
    ]
    positions = [printed.stdout.index(event) for event in expected_events]
    assert positions == sorted(positions)
    assert printed.stdout.count("pto.set_intra_block") == 3
    assert printed.stdout.count("pto.wait_intra_block") == 2
    source.write_text(printed.stdout)
    reparsed = subprocess.run(
        [compiler, "--pto-arch=a5", str(source)], capture_output=True, text=True, timeout=60, check=False
    )
    assert reparsed.returncode == 0, reparsed.stderr


def test_old_ffts_contract_is_unchanged():
    for mode in (0, 1, 2):
        assert system_ops.sync_set(13, pipe=pl.PipeType.MTE3, ffts_mode=mode).kwargs["ffts_mode"] == mode
    with pytest.raises(ValueError, match="ffts_mode"):
        system_ops.sync_set(3, pipe=pl.PipeType.MTE3, ffts_mode=4)
    with pytest.raises(ValueError, match="event_id"):
        system_ops.sync_set(17, pipe=pl.PipeType.MTE3)
    event = ir.Var("event", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    assert system_ops.sync_wait(event, pipe=pl.PipeType.MTE2).args == [event]


def test_scalar_pipe_emission(tmp_path, validate_native=False):
    """Check scalar-pipe emission; the companion checks assembler capability."""

    @pl.program
    class ScalarPipe:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            pl.system.set_intra_block(17, pipe=pl.PipeType.S)
            pl.system.wait_intra_block(17, pipe=pl.PipeType.S)

    text = codegen.PTOCodegen().generate(ScalarPipe)
    assert "pto.set_intra_block <PIPE_S>, 17" in text
    assert "pto.wait_intra_block <PIPE_S>, 17" in text
    if not validate_native:
        return
    compiler = find_ptoas_binary()
    if not compiler:
        pytest.skip("PTOAS not installed")
    source = tmp_path / "scalar_pipe.pto"
    source.write_text(text)
    result = subprocess.run(
        [compiler, "--pto-arch=a5", str(source)], capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode and "expects pipe to be one of" in result.stderr and "<PIPE_S>" not in result.stderr:
        pytest.xfail("Installed PTOAS lacks A5 intra-block PIPE_S support: " + result.stderr)
    assert result.returncode == 0, result.stderr
    assert "set_intra_block(PIPE_S, 17)" in result.stdout
    assert "wait_intra_block(PIPE_S, 17)" in result.stdout


def test_nested_unpinned_event_cannot_be_copied_as_a_shared_loop():
    @pl.program
    class Nested:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self):
            pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(17, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
            for _ in pl.range(2):
                pl.system.set_intra_block(3, pipe=pl.PipeType.S)

    with pytest.raises(ValueError, match="unpinned named event cannot be duplicated"):
        passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(Nested)))


@pytest.mark.parametrize(
    "pipe",
    [pl.PipeType.FIX, pl.PipeType.MTE1, pl.PipeType.MTE2, pl.PipeType.MTE3, pl.PipeType.V, pl.PipeType.S],
)
@pytest.mark.parametrize(
    "wait_pipe",
    [pl.PipeType.FIX, pl.PipeType.MTE1, pl.PipeType.MTE2, pl.PipeType.MTE3, pl.PipeType.V, pl.PipeType.S],
)
def test_different_pipe_pairs_emit_without_equating_pipes(pipe, wait_pipe):
    span = ir.Span.unknown()
    stmts: list[ir.Stmt] = [
        ir.EvalStmt(system_ops.set_intra_block(17, pipe=pipe), span),
        ir.EvalStmt(system_ops.wait_intra_block(17, pipe=wait_pipe), span),
        ir.ReturnStmt([], span),
    ]
    function = ir.Function("pair", [], [], ir.SeqStmts(stmts, span), span, type=ir.FunctionType.AIV)
    text = codegen.PTOCodegen().generate(ir.Program([function], "Pair", span))
    assert f"pto.set_intra_block <PIPE_{pipe.name}>, 17" in text
    assert f"pto.wait_intra_block <PIPE_{wait_pipe.name}>, 17" in text


@pytest.mark.parametrize(
    "event,pipe,message",
    [
        (17, pl.PipeType.MTE1, "static event_id on A2/A3"),
        (3, pl.PipeType.S, "PIPE_S requires A5"),
    ],
)
def test_a3_codegen_rejects_a5_only_forms(event, pipe, message):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    span = ir.Span.unknown()
    stmt = ir.EvalStmt(system_ops.set_intra_block(event, pipe=pipe), span)
    function = ir.Function(
        "a3", [], [], ir.SeqStmts([stmt, ir.ReturnStmt([], span)], span), span, type=ir.FunctionType.AIV
    )
    with pytest.raises(ValueError, match=message):
        codegen.PTOCodegen().generate(ir.Program([function], "A3", span))


def test_default_passes_keep_payload_between_wait_and_signal():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, data: pl.Tensor[[16], pl.FP32], out: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            tile = pl.load(data, [0], [16])
            tagged = pl.add(tile, 1000.0)
            out = pl.store(tagged, [0], out)
            pl.system.set_intra_block(2, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)
            pl.system.wait_intra_block(18, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def main_aic(self, data: pl.Tensor[[16], pl.FP32], out: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(18, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)

        @pl.function(type=pl.FunctionType.AIV)
        def main_aiv(
            self, data: pl.Tensor[[16], pl.FP32], out: pl.InOut[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            pl.system.wait_intra_block(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            tile = pl.load(data, [0], [16])
            tagged = pl.add(tile, 1000.0)
            out = pl.store(tagged, [0], out)
            pl.system.set_intra_block(2, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)
            return out

        @pl.function(type=pl.FunctionType.Group)
        def main(
            self, data: pl.Tensor[[16], pl.FP32], out: pl.InOut[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            self.main_aic(data, out)
            self.main_aiv(data, out)
            return out

    manager = PassManager.get_strategy(OptimizationStrategy.Default)
    after = manager.run_passes(Before)
    expected = manager.run_passes(Expected)
    # Use the same implicit-return form for generated and explicitly typed peers.
    comparison_after = passes.convert_tensor_to_tile_ops()(after)
    comparison_expected = passes.convert_tensor_to_tile_ops()(expected)
    ir.assert_structural_equal(comparison_after, comparison_expected)
    aiv = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV)
    names = []
    for stmt in ir.flatten_to_stmts(aiv.body):
        value = (
            stmt.expr
            if isinstance(stmt, ir.EvalStmt)
            else stmt.value
            if isinstance(stmt, ir.AssignStmt)
            else None
        )
        if isinstance(value, ir.Call):
            names.append(value.op.name)
    assert names.index("system.wait_intra_block") < names.index("tile.load")
    assert names.index("tile.load") < names.index("tile.store")
    assert names.index("tile.store") < names.index("system.set_intra_block")
    assert "tile.tpush_to_aic" not in names


@pytest.mark.parametrize("action", ["set", "wait"])
def test_parser_index_literals_stay_static_without_casting_dynamic_index(action):
    literal = ir.ConstInt(17, DataType.INDEX, ir.Span.unknown())
    call = getattr(system_ops, f"{action}_intra_block")(literal, pipe=pl.PipeType.S)
    assert call.args == []
    assert call.kwargs["event_id"] == 17


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_pipeline_preserves_named_event_fences(stage, nested, dynamic, tmp_path, validate_native=False):
    """No SSA edge links these events to IO; pipeline clustering used to cross both fences.

    Nested control also fences the enclosing pipeline: demoting only the inner
    loop would still let outer-stage replication reuse the same event early.
    """
    event = "event" if dynamic else "4"
    body = f"""pl.system.wait_intra_block({event}, pipe=pl.PipeType.MTE2)
value = pl.tile.load(data, [i, 0], [1, 16])
tagged = pl.tile.add(value, 1000.0)
pl.tile.store(tagged, [i, 0], out)
pl.system.set_intra_block({event}, pipe=pl.PipeType.MTE3)"""
    if nested:
        body = "for j in pl.range(1):\n" + "\n".join("    " + line for line in body.splitlines())
    body = "\n".join("            " + line for line in body.splitlines())
    Before = pl.parse_program(
        f"""import pypto.language as pl
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main(self, data: pl.Tensor[[4, 16], pl.FP32],
             out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], event: pl.Scalar[pl.INT32]):
        for i in pl.pipeline(4, stage={stage}):
{body}
        return out
"""
    )
    # This reference starts sequential and never exercises pipeline demotion.
    # Its authored event IDs, operands, pipes and payload accesses stay explicit.
    Expected = pl.parse_program(
        f"""import pypto.language as pl
@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.AIV)
    def main(self, data: pl.Tensor[[4, 16], pl.FP32],
             out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], event: pl.Scalar[pl.INT32]):
        for i in pl.range(4):
{body}
        return out
"""
    )
    manager = PassManager.get_strategy(OptimizationStrategy.Default)
    after = manager.run_passes(Before)
    expected = manager.run_passes(Expected)
    ir.assert_structural_equal(after, expected)
    target_ops = {
        ir.get_op("system.wait_intra_block").name,
        ir.get_op("tile.load").name,
        ir.get_op("tile.store").name,
        ir.get_op("system.set_intra_block").name,
    }
    owners: dict[str, list[ir.ForStmt | None]] = {}

    def record_owners(stmt: ir.Stmt, loop: ir.ForStmt | None = None):
        for child in ir.flatten_to_stmts(stmt):
            if isinstance(child, ir.ForStmt):
                record_owners(child.body, child)
                continue
            call = (
                child.expr
                if isinstance(child, ir.EvalStmt)
                else child.value
                if isinstance(child, ir.AssignStmt)
                else None
            )
            if isinstance(call, ir.Call) and call.op.name in target_ops:
                owners.setdefault(call.op.name, []).append(loop)

    function = next(f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV)
    record_owners(function.body)
    assert set(owners) == target_ops
    assert all(len(loops) == 1 for loops in owners.values())
    payload_loop = owners["tile.load"][0]
    assert payload_loop is not None
    assert all(loops[0] is payload_loop for loops in owners.values())
    printed = after.as_python()
    assert "pl.pipeline(" not in printed
    # Each authored event remains once in the kept loop body, without cloned
    # prologue/epilogue events or new synchronization instructions.
    assert printed.count("pl.system.wait_intra_block(") == 1
    assert printed.count("pl.system.set_intra_block(") == 1
    pto = codegen.PTOCodegen().generate(after)
    assert pto.index("pto.wait_intra_block") < pto.index("pto.tload")
    assert pto.index("pto.tload") < pto.index("pto.tstore")
    assert pto.index("pto.tstore") < pto.index("pto.set_intra_block")
    if not validate_native:
        return
    compiler = find_ptoas_binary()
    if not compiler:
        pytest.skip("PTOAS not installed: pipeline order checked, compilation unavailable")
    source = tmp_path / "pipeline.pto"
    source.write_text(pto)
    # Default passes assign physical tile addresses; compile the emitted level-3
    # program with the same assembler mode used by the device build path.
    result = subprocess.run(
        [compiler, "--pto-arch=a5", "--pto-level=level3", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("action", ["set", "wait"])
@pytest.mark.parametrize("stage", [1, 3])
def test_skew_preserves_nested_named_event_structure(action, stage):
    """A single event inside a branch suffices; no tpush/tpop pair is needed."""
    text = f"""import pypto.language as pl
@pl.program
class NestedEvent:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def main(self):
        for i in pl.pipeline(4, stage={stage}):
            for j in pl.pipeline(2, stage={stage}):
                if i >= j:
                    pl.system.{action}_intra_block(4, pipe=pl.PipeType.MTE2)
"""
    before = pl.parse_program(text)
    expected = pl.parse_program(
        text.replace(f"pl.pipeline(4, stage={stage})", "pl.range(4)").replace(
            f"pl.pipeline(2, stage={stage})", "pl.range(2)"
        )
    )
    after = passes.skew_cross_core_pipeline()(before)
    ir.assert_structural_equal(after, expected)
    ir.assert_structural_equal(passes.skew_cross_core_pipeline()(after), after)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
