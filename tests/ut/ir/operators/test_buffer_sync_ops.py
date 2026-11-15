# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual A5 Buffer-ID contract, round-trip, placement and lowering regressions."""

import os
import re
import subprocess

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import system_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

SPAN = ir.Span.unknown()


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("buf_id", [0, 31])
def test_static_contract(name, buf_id):
    call = getattr(system_ops, name)(buf_id, pipe=ir.PipeType.MTE2)
    assert call.op.name == f"system.{name}"
    assert not call.args
    assert call.kwargs["buf_id"] == buf_id
    assert call.kwargs["mode"] == 0
    assert isinstance(call.type, ir.UnknownType)
    assert getattr(pl.system, name)
    assert getattr(ir.system, name)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("constant", [False, True])
def test_dynamic_contract(name, constant):
    id_expr = ir.ConstInt(1, pl.INDEX, SPAN) if constant else ir.Var("id", ir.ScalarType(pl.INDEX), SPAN)
    call = getattr(system_ops, name)(id_expr, pipe=ir.PipeType.V)
    assert len(call.args) == 1
    assert "buf_id" not in call.kwargs
    assert isinstance(call.type, ir.UnknownType)
    wrapped = pl.Scalar(pl.INDEX, id_expr)
    assert len(getattr(pl.system, name)(wrapped, pipe=pl.PipeType.V).args) == 1


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("value", [-1, 32, True, False, 1.5, "1"])
def test_invalid_id(name, value):
    with pytest.raises((TypeError, ValueError), match="buf_id"):
        getattr(system_ops, name)(value, pipe=ir.PipeType.MTE2)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("dtype", [pl.INT32, pl.INT64, pl.FP32, pl.BOOL])
def test_invalid_dynamic_dtype(name, dtype):
    with pytest.raises(TypeError, match="INDEX"):
        getattr(system_ops, name)(ir.Var("id", ir.ScalarType(dtype), SPAN), pipe=ir.PipeType.V)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("pipe", [ir.PipeType.ALL, 1, True, "MTE2", None])
def test_invalid_pipe(name, pipe):
    with pytest.raises((TypeError, ValueError), match="pipe"):
        getattr(system_ops, name)(0, pipe=pipe)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize("mode", [-1, 1, 4, True, 0.0, "0"])
def test_invalid_mode(name, mode):
    with pytest.raises((TypeError, ValueError), match="mode"):
        getattr(system_ops, name)(0, pipe=ir.PipeType.V, mode=mode)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
def test_missing_pipe(name):
    with pytest.raises(TypeError, match="pipe"):
        getattr(system_ops, name)(0)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize(
    "args,kwargs,diagnostic",
    [
        ([], {"pipe": ir.PipeType.V}, "exactly one"),
        ([ir.Var("id", ir.ScalarType(pl.INDEX), SPAN)], {"buf_id": 0, "pipe": ir.PipeType.V}, "exactly one"),
        ([ir.ConstInt(0, pl.INDEX, SPAN)] * 2, {"pipe": ir.PipeType.V}, "at most one"),
        ([], {"buf_id": 32, "pipe": ir.PipeType.V}, "buf_id"),
        ([], {"buf_id": True, "pipe": ir.PipeType.V}, "buf_id"),
        ([], {"buf_id": 0}, "pipe"),
        ([], {"buf_id": 0, "pipe": 999}, "pipe"),
        ([], {"buf_id": 0, "pipe": ir.PipeType.ALL}, "pipe"),
        ([], {"buf_id": 0, "pipe": ir.PipeType.V, "mode": 4}, "mode"),
        ([], {"buf_id": 0, "pipe": ir.PipeType.V, "mode": True}, "mode"),
        ([], {"buf_id": 0, "pipe": ir.PipeType.V, "unexpected": 0}, "Unknown kwarg"),
        ([None], {"pipe": ir.PipeType.V}, "buf_id"),
        ([ir.ConstInt(0, pl.INT32, SPAN)], {"pipe": ir.PipeType.V}, "INDEX"),
        ([ir.ConstInt(32, pl.INDEX, SPAN)], {"pipe": ir.PipeType.V}, "buf_id"),
    ],
)
def test_direct_ir_validation(name, args, kwargs, diagnostic):
    with pytest.raises((TypeError, ValueError), match=diagnostic):
        ir.create_op_call(f"system.{name}", args, kwargs, SPAN)


@pl.program
class BufferSyncLoop:
    @pl.function(type=pl.FunctionType.AIV)
    def main(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        for i in pl.range(4):
            buf_id: pl.Scalar[pl.INDEX] = i % 2
            pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
            tile = pl.load(x, [0, 0], [16, 16])
            pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
            pl.system.get_buf(buf_id, pipe=pl.PipeType.MTE3)
            pl.store(tile, [0, 0], out)
            pl.system.rls_buf(buf_id, pipe=pl.PipeType.MTE3)


def test_round_trip():
    printed = BufferSyncLoop.as_python()
    assert "pipe=pl.PipeType.MTE2" in printed
    assert "mode=0" in printed
    assert "buf_id=0" in printed
    ir.assert_structural_equal(BufferSyncLoop, pl.parse_program(printed))


def _generate(program=BufferSyncLoop, target=BackendType.Ascend950):
    backend.reset_for_testing()
    backend.set_backend_type(target)
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    return codegen.PTOCodegen().generate(lowered)


def test_default_pipeline_loop_order():
    pto = _generate()
    operations = [
        "scf.for",
        "arith.remsi",
        'pto.get_buf "PIPE_MTE2", 0, 0',
        "pto.tload",
        'pto.rls_buf "PIPE_MTE2", 0, 0',
        'pto.get_buf_dyn "PIPE_MTE3",',
        "pto.tstore",
        'pto.rls_buf_dyn "PIPE_MTE3",',
    ]
    positions = [pto.index(op) for op in operations]
    assert positions == sorted(positions), pto
    assert len(re.findall(r"pto\.(?:get|rls)_buf(?:_dyn)? ", pto)) == 4
    assert not re.search(r"%\w+\s*=\s*pto\.(get|rls)_buf", pto)


@pytest.mark.parametrize("target", [BackendType.Ascend910B])
def test_unsupported_target(target):
    with pytest.raises((TypeError, ValueError), match=r"system.get_buf.*A5/Ascend950.*target"):
        _generate(target=target)


def test_incore_placement_rejected():
    @pl.program
    class Untyped:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self):
            pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
            pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises((TypeError, ValueError), match="system.get_buf.*explicitly typed AIC/AIV"):
        passes.expand_mixed_kernel()(Untyped)


def test_ptoas_compilation(tmp_path):
    """Use an explicitly selected PTOAS built from the sibling worktree."""
    ptoas = os.environ.get("T1_PTOAS")
    if not ptoas:
        pytest.skip("Set T1_PTOAS to the sibling worktree build/tools/ptoas/ptoas")
    source = tmp_path / "buffer_sync.pto"
    source.write_text(_generate())
    output = tmp_path / "buffer_sync.cpp"
    result = subprocess.run(
        [
            ptoas,
            "--pto-arch=a5",
            "--pto-level=level3",
            "--enable-insert-sync",
            str(source),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    cpp = output.read_text()
    assert "get_buf(PIPE_MTE2, 0, 0)" in cpp
    assert "rls_buf(PIPE_MTE3," in cpp


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
def test_dynamic_constant_round_trip(name):
    call = getattr(system_ops, name)(ir.ConstInt(3, pl.INDEX, SPAN), pipe=pl.PipeType.V)
    body = ir.EvalStmt(call, SPAN)
    func = ir.Function("main", [], [], body, SPAN, ir.FunctionType.AIV)
    program = ir.Program([func], "DynamicConstant", SPAN)
    printed = program.as_python()
    assert "pl.const(3, pl.INDEX)" in printed
    ir.assert_structural_equal(program, pl.parse_program(printed))


def test_no_result_cannot_be_bound():
    @pl.program
    class InvalidBinding:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            _unused = pl.system.get_buf(0, pipe=pl.PipeType.V)

    with pytest.raises(ValueError, match="no result"):
        _generate(program=InvalidBinding)


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize(
    "pipe",
    [
        pl.PipeType.MTE1,
        pl.PipeType.MTE2,
        pl.PipeType.MTE3,
        pl.PipeType.M,
        pl.PipeType.V,
        pl.PipeType.S,
        pl.PipeType.FIX,
    ],
)
def test_concrete_pipe_codegen(name, pipe):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    call = getattr(system_ops, name)(31, pipe=pipe)
    func = ir.Function(
        "main",
        [],
        [],
        ir.SeqStmts([ir.EvalStmt(call, SPAN)], SPAN),
        SPAN,
        ir.FunctionType.AIC
        if pipe in (pl.PipeType.M, pl.PipeType.MTE1, pl.PipeType.FIX)
        else ir.FunctionType.AIV,
    )
    pto = codegen.PTOCodegen().generate(ir.Program([func], "ConcretePipe", SPAN))
    assert f'pto.{name} "PIPE_{pipe.name}", 31, 0' in pto


def test_pipeline_keeps_manual_regions_in_order():
    @pl.program
    class Pipelined:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            for i in pl.pipeline(0, 4, 1, stage=2):
                pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
                tile = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
                pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
                pl.system.get_buf(i % 2, pipe=pl.PipeType.MTE3)
                pl.tile.store(tile, [0, 0], out)
                pl.system.rls_buf(i % 2, pipe=pl.PipeType.MTE3)

    after = passes.canonicalize_io_order()(passes.lower_pipeline_loops()(Pipelined))
    operations = re.findall(r"pl\.(?:system\.(?:get_buf|rls_buf)|tile\.(?:load|store))", after.as_python())
    assert (
        operations
        == [
            "pl.system.get_buf",
            "pl.tile.load",
            "pl.system.rls_buf",
            "pl.system.get_buf",
            "pl.tile.store",
            "pl.system.rls_buf",
        ]
        * 2
    )


@pytest.mark.parametrize("value", [0, 31])
@pytest.mark.parametrize("keyword", [False, True])
def test_dsl_static_layout(value, keyword):
    spelling = f"buf_id={value}" if keyword else str(value)
    source = f"""
@pl.program
class StaticLayout:
    @pl.function(type=pl.FunctionType.AIV)
    def main(self):
        pl.system.get_buf({spelling}, pipe=pl.PipeType.MTE2)
        pl.system.rls_buf({spelling}, pipe=pl.PipeType.MTE2)
"""
    parsed = pl.parse_program(source)
    printed = parsed.as_python()
    assert printed.count(f"buf_id={value}") == 2
    ir.assert_structural_equal(parsed, pl.parse_program(printed))


def test_folded_invalid_dynamic_id():
    id_expr = ir.Add(ir.ConstInt(31, pl.INDEX, SPAN), ir.ConstInt(1, pl.INDEX, SPAN), pl.INDEX, SPAN)
    with pytest.raises(ValueError, match=r"system.get_buf.*buf_id.*\[0, 31\]"):
        system_ops.get_buf(id_expr, pipe=pl.PipeType.V)


def test_cast_and_buffer_modes_round_trip():
    @pl.program
    class MixedModes:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self, x: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP16]]):
            tile = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
            pl.system.get_buf(0, pipe=pl.PipeType.V, mode=0)
            converted = pl.tile.cast(tile, pl.FP16, mode="round")
            pl.system.rls_buf(0, pipe=pl.PipeType.V, mode=0)
            pl.tile.store(converted, [0, 0], out)

    printed = MixedModes.as_python()
    assert re.search(r"mode=['\"]round['\"]", printed)
    assert printed.count("mode=0") == 2
    ir.assert_structural_equal(MixedModes, pl.parse_program(printed))


@pytest.mark.parametrize("name", ["get_buf", "rls_buf"])
@pytest.mark.parametrize(
    "dynamic,kwargs,message",
    [
        (True, {"buf_id": 0, "pipe": pl.PipeType.V}, "exactly one"),
        (False, {"pipe": pl.PipeType.V}, "exactly one"),
        (False, {"buf_id": 32, "pipe": pl.PipeType.V}, "buf_id"),
        (False, {"buf_id": 0, "pipe": pl.PipeType.V, "mode": 4}, "mode=0"),
    ],
)
def test_raw_call_cannot_bypass_contract(name, dynamic, kwargs, message):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    identifier = ir.Var("id", ir.ScalarType(pl.INDEX), SPAN)
    call = ir.Call(ir.Op(f"system.{name}"), [identifier] if dynamic else [], kwargs, ir.UnknownType(), SPAN)
    func = ir.Function(
        "main", [(identifier, ir.ParamDirection.In)], [], ir.EvalStmt(call, SPAN), SPAN, ir.FunctionType.AIV
    )
    with pytest.raises(ValueError, match=message):
        codegen.PTOCodegen().generate(ir.Program([func], "RawCall", SPAN))


def test_aic_nested_loop_keeps_protected_mat_load():
    @pl.program
    class MatLoop:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            b: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            for _outer in pl.range(2):
                for _inner in pl.range(2):
                    pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
                    lhs = pl.tile.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
                    pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
                    pl.system.get_buf(0, pipe=pl.PipeType.MTE1)
                    left = pl.tile.move(lhs, target_memory=pl.Mem.Left)
                    pl.system.rls_buf(0, pipe=pl.PipeType.MTE1)
                    rhs = pl.tile.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
                    right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
                    result = pl.tile.matmul(left, right)
                    pl.tile.store(result, [0, 0], out)

    pto = _generate(program=MatLoop)
    inner_loop = pto.index("scf.for", pto.index("scf.for") + 1)
    positions = [
        inner_loop,
        pto.index('pto.get_buf "PIPE_MTE2"'),
        pto.index("pto.tload"),
        pto.index('pto.rls_buf "PIPE_MTE2"'),
        pto.index('pto.get_buf "PIPE_MTE1"'),
        pto.index("pto.tmov"),
        pto.index('pto.rls_buf "PIPE_MTE1"'),
    ]
    assert positions == sorted(positions), pto
    assert pto.count('pto.get_buf "PIPE_MTE2"') == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
