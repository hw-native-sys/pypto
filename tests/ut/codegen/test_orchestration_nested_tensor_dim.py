# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Late physical-shape expressions keep nested calls; do not run Simplify."""

import re

import pytest
from _orchestration_codegen_common import _assert_orchestration_compiles, _generate_orch_code
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.pypto_core import DataType, ir


def _generate_without_roundtrip(program):
    # This deliberately hand-built late IR is not frontend-roundtrippable.
    with passes.PassContext([]):
        return _generate_orch_code(program)


@pytest.mark.parametrize(
    "source_kind,axis",
    [
        ("static", 0),
        ("static", -1),
        ("input_dynamic", 0),
        ("local_dynamic", 0),
        ("invalid_axis", 2),
        ("other_call", 0),
    ],
)
def test_nested_dim_in_allocation_extent(source_kind, axis, tmp_path):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    span = ir.Span.unknown()
    index = DataType.INDEX
    n = ir.Var("N", ir.ScalarType(index), span)
    # Normalize explicitly for the native all-int/all-Expr overloads; the public
    # Python TensorType wrapper also accepts and normalizes mixed shapes.
    first = ir.ConstInt(64, index, span) if source_kind in ("static", "invalid_axis", "other_call") else n
    shape = [first, ir.ConstInt(16, index, span)]
    src = ir.Var("src", ir.TensorType(shape, DataType.FP32), span)
    stmts = []
    params = [src]
    if source_kind == "local_dynamic":
        params = [n]
        stmts.append(ir.AssignStmt(src, ir.Call(ir.Op("tensor.create"), [], src.type, span), span))
    dim = ir.Call(ir.Op("tensor.dim"), [src, ir.ConstInt(axis, index, span)], ir.ScalarType(index), span)
    if source_kind == "other_call":
        read = ir.Call(
            ir.Op("tensor.read"),
            [src, ir.MakeTuple([ir.ConstInt(0, index, span), ir.ConstInt(0, index, span)], span)],
            ir.ScalarType(DataType.FP32),
            span,
        )
        dim = ir.Cast(read, index, span)
    extent = ir.Mul(dim, ir.ConstInt(2, index, span), index, span)
    output_type = ir.TensorType([extent, ir.ConstInt(8, index, span)], DataType.FP32)
    output = ir.Var("workspace", output_type, span)
    stmts.extend(
        [
            ir.AssignStmt(output, ir.Call(ir.Op("tensor.create"), [], output_type, span), span),
            ir.ReturnStmt(span),
        ]
    )
    func = ir.Function(
        "orch_nested_dim", params, [], ir.SeqStmts(stmts, span), span, type=ir.FunctionType.Orchestration
    )
    program = ir.Program([func], "nested_dim", span)
    # The bug is a Call nested under Mul in a TensorType extent, not a standalone
    # scalar assignment. No pass which flattens/folds that expression is used.
    assert isinstance(extent.left, ir.Call) or source_kind == "other_call"
    if source_kind in ("invalid_axis", "other_call"):
        match = "axis out of range" if source_kind == "invalid_axis" else "expression type.*Call"
        with pytest.raises((ValueError, RuntimeError), match=match):
            _generate_without_roundtrip(program)
        return
    code = _generate_without_roundtrip(program)
    if source_kind == "static":
        assert re.search(rf"{64 if axis == 0 else 16}\s*\*\s*2", code), code
    elif source_kind == "input_dynamic":
        assert "orch_args.tensor(0).ref().shapes[0]" in code, code
    else:
        assert "src.shapes[0]" in code, code
        assert code.index("const Tensor& src") < code.index("workspace_ci_shapes"), code
    cpp = tmp_path / "nested_dim.cpp"
    cpp.write_text(code)
    _assert_orchestration_compiles(cpp, arch="a5")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
