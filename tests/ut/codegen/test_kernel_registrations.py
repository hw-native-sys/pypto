# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Registration identity follows the final payload, including wrapper dispatch."""

import pytest
from _orchestration_codegen_common import _finalize_for_codegen
from pypto import DataType as D
from pypto import backend, codegen, ir
from pypto.backend import BackendType

S = ir.Span.unknown()


def parameter(name):
    return ir.Var(name, ir.TensorType([16], D.FP32), S)


def function(name, kind, param, body):
    return ir.Function(name, [param], [], body, S, type=kind)


def call(name, argument, direction):
    return ir.Call(ir.GlobalVar(name), [argument], {}, {"arg_directions": [direction]}, ir.TupleType([]), S)


@pytest.mark.parametrize("wrapper", [None, ir.FunctionType.Spmd, ir.FunctionType.Group, "mixed"])
def test_same_code_different_tensor_directions_get_distinct_ids(wrapper):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    x = parameter("kernel_input")
    functions = [function("vector", ir.FunctionType.AIV, x, ir.ReturnStmt([], S))]
    target = "vector"
    if wrapper is not None:
        w = parameter("wrapper_input")
        body: list[ir.Stmt] = [ir.EvalStmt(call("vector", w, ir.ArgDirection.Input), S)]
        if wrapper == "mixed":
            cube_param = parameter("cube_input")
            functions.append(function("cube", ir.FunctionType.AIC, cube_param, ir.ReturnStmt([], S)))
            body.insert(0, ir.EvalStmt(call("cube", w, ir.ArgDirection.Input), S))
        body.append(ir.ReturnStmt([], S))
        functions.append(
            function(
                "wrapper", ir.FunctionType.Group if wrapper == "mixed" else wrapper, w, ir.SeqStmts(body, S)
            )
        )
        target = "wrapper"
    actual = parameter("actual")
    body = ir.SeqStmts(
        [
            ir.EvalStmt(call(target, actual, ir.ArgDirection.Input), S),
            ir.EvalStmt(call(target, actual, ir.ArgDirection.NoDep), S),
            ir.EvalStmt(call(target, actual, ir.ArgDirection.Input), S),
            ir.ReturnStmt([], S),
        ],
        S,
    )
    functions.append(function("main", ir.FunctionType.Orchestration, actual, body))
    source = _finalize_for_codegen(ir.Program(functions, "test", S))
    main = source.get_function("main")
    assert main is not None
    result = codegen.generate_orchestration(source, main)
    registrations = result.kernel_registrations
    expected_names = {"vector", "cube"} if wrapper == "mixed" else {"vector"}
    assert len(registrations) == 2 * len(expected_names)
    assert len({r.func_id for r in registrations}) == len(registrations)
    for name in expected_names:
        assert {tuple(r.signature) for r in registrations if r.code_name == name} == {("IN",), ("INOUT",)}
    assert result.code.count("add_no_dep") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
