# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical debugging of the default Buffer pipeline and its pass dumps."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.debug import torch_codegen
from pypto.jit.decorator import jit
from pypto.runtime.runner import RunConfig

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


@jit.incore
def _arithmetic(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    rows, cols = a.shape
    x = pl.load(a, [0, 0], [rows, cols])
    y = pl.load(b, [0, 0], [rows, cols])
    pl.store(pl.mul(pl.add(x, y), y), [0, 0], out)
    return out


@jit
def buffer_debug(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _arithmetic(a, b, out)


@jit
def matrix_debug(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:, :] = pl.matmul(a[:, :], b[:, :], b_trans=True, out_dtype=pl.FP32)
    return out


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("planner", _PLANNERS)
def test_public_lower_and_compiled_pass_dumps_remain_numerically_debuggable(
    tmp_path, monkeypatch, planner, pass_verification_instruments, matrix
):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32) / 32
    b = torch.linspace(-1, 1, 512).reshape(16, 32)
    out = torch.empty((16, 16)) if matrix else torch.empty_like(a)
    expected = a @ b.T if matrix else (a + b) * b
    kernel = matrix_debug if matrix else buffer_debug
    config = RunConfig(
        platform="a2a3sim",
        save_kernels_dir=str(tmp_path / "compiled"),
        dump_passes=True,
    )
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        program = kernel.lower(a, b, out, config=config)
    assert any(function.ir_stage == ir.FunctionIRStage.Buffer for function in program.functions.values())
    entry = next(
        function for function in program.functions.values() if ir.is_orchestration_like(function.func_type)
    )
    namespace = {}
    exec(torch_codegen(program, check_shapes=True), namespace)  # noqa: S102
    namespace[entry.name](a, b, out)
    torch.testing.assert_close(out, expected)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        compiled = kernel.compile(a, b, out, config=config)
    final_dump = next((compiled.output_dir / "passes_dump").glob("*LowerTileToBuffer.py"))
    final_program = ir.deserialize(final_dump.with_suffix(".msgpack").read_bytes())
    assert isinstance(final_program, ir.Program)
    assert any(
        function.ir_stage == ir.FunctionIRStage.Buffer for function in final_program.functions.values()
    )
    compiled.validate_ir({"a": a, "b": b, "out": torch.empty_like(out)}, {"out": expected})


def test_buffer_storage_views_alias_bytes_and_destination_writes():
    span = ir.Span.unknown()
    byte_type = ir.BufferType([4, 32], pl.UINT8, ir.MemorySpace.Vec)
    window_type = ir.BufferType([2, 32], pl.UINT8, ir.MemorySpace.Vec)
    value_type = ir.BufferType([2, 8], pl.FP32, ir.MemorySpace.Vec)
    storage = ir.Var("storage", byte_type, span)
    window = ir.Var("window", window_type, span)
    value = ir.Var("value", value_type, span)
    alias = ir.Var("alias", value_type, span)
    src = ir.Var("src", ir.TensorType([2, 8], pl.FP32), span)
    out = ir.Var("out", ir.TensorType([2, 8], pl.FP32), span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, pl.INDEX, span) for value in values], span)

    def call(name, args, type_):
        return ir.Call(ir.get_op(name), args, {}, type_, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(storage, call("buffer.alloc", [dims()], byte_type), span),
            ir.AssignStmt(window, call("buffer.subview", [storage, dims(2, 0)], window_type), span),
            ir.AssignStmt(value, call("buffer.reshape", [window], value_type), span),
            ir.AssignStmt(alias, call("buffer.reshape", [window], value_type), span),
            ir.EvalStmt(call("buffer.load", [src, dims(0, 0), dims(2, 8), value], ir.VoidType()), span),
            ir.EvalStmt(call("buffer.add", [value, value, value], ir.VoidType()), span),
            ir.EvalStmt(call("buffer.store", [alias, dims(0, 0), dims(2, 8), out], ir.VoidType()), span),
        ],
        span,
    )
    function = ir.Function("kernel", [src, out], [], body, span, ir_stage=ir.FunctionIRStage.Buffer)
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    source = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    output = torch.empty_like(source)
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, source * 2)
