# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit byte subviews and reshapes serialize directly to native alias instructions."""

import re
from pathlib import Path

import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as _ir

_SPAN = ir.Span.unknown()


def _int(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _SPAN)


def _tuple(*values: int) -> ir.MakeTuple:
    return ir.MakeTuple([_int(value) for value in values], _SPAN)


def _program(addressed: bool, byte_offset: int) -> ir.Program:
    root_type = ir.BufferType([128, 32], DataType.UINT8, ir.MemorySpace.Vec)
    window_type = ir.BufferType([64, 32], DataType.UINT8, ir.MemorySpace.Vec)
    tile_type = ir.BufferType([16, 32], DataType.FP32, ir.MemorySpace.Vec)
    result_type = ir.BufferType([8, 64], DataType.FP32, ir.MemorySpace.Vec)
    root = ir.Var("root", root_type, _SPAN)
    # Repeated hints still require separate SSA definitions and operand bindings.
    window = ir.Var("view", window_type, _SPAN)
    typed = ir.Var("view", tile_type, _SPAN)
    reshaped = ir.Var("view", result_type, _SPAN)
    output = ir.Var("output", ir.TensorType([8, 64], DataType.FP32), _SPAN)
    alloc_args: list[ir.Expr] = [_tuple()]
    if addressed:
        alloc_args.append(_int(4096))

    def assign(var: ir.Var, name: str, args: list[ir.Expr]) -> ir.AssignStmt:
        call = _ir._create_internal_op_call(name, args, {}, var.type, _SPAN)
        return ir.AssignStmt(var, call, _SPAN)

    fill = _ir._create_internal_op_call(
        "buffer.full", [ir.ConstFloat(1.0, DataType.FP32, _SPAN), typed], {}, _SPAN
    )
    store = _ir._create_internal_op_call(
        "buffer.store", [reshaped, _tuple(0, 0), _tuple(8, 64), output], {}, _SPAN
    )
    statements = [
        assign(root, "buffer.alloc", alloc_args),
        assign(window, "buffer.subview", [root, _tuple(byte_offset // 32, 0)]),
        assign(typed, "buffer.reshape", [window]),
        ir.EvalStmt(fill, _SPAN),
        assign(reshaped, "buffer.reshape", [typed]),
        ir.EvalStmt(store, _SPAN),
    ]
    function = ir.Function(
        "kernel",
        [(output, ir.ParamDirection.Out)],
        [],
        ir.SeqStmts(statements, _SPAN),
        _SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "BufferViews", _SPAN)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("addressed", [False, True])
@pytest.mark.parametrize("byte_offset", [0, 64])
def test_buffer_views_round_trip_and_emit_one_native_operation_per_call(
    tmp_path: Path, ascend_backend: BackendType, addressed: bool, byte_offset: int
) -> None:
    program = _program(addressed, byte_offset)
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    text = codegen.PTOCodegen().generate(restored, emit_tile_addr=not addressed, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 1
    assert text.count("pto.subview") == 1
    assert text.count("pto.treshape") == 2
    assert "pto.tmov" not in text and "pto.textract" not in text
    assert (" addr = " in text) == addressed
    aliases = re.findall(r"(%\w+) = pto.treshape (%\w+)", text)
    assert len(aliases) == 2 and aliases[1][1] == aliases[0][0]
    assert f"outs({aliases[0][0]} : " in text
    assert f"pto.tstore ins({aliases[1][0]} : " in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "buffer_views.pto"
    output = tmp_path / "buffer_views.cpp"
    source.write_text(text)
    arch = "a2" if ascend_backend == BackendType.Ascend910B else "a5"
    _run_ptoas(
        str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={'level3' if addressed else 'level2'}"]
    )
    cpp = output.read_text()
    assert cpp.count("TRESHAPE(") == 2
    assert "TMOV(" not in cpp and "TEXTRACT(" not in cpp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
