# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tile-only postcondition covers exactly the kernels selected for conversion."""

import pytest
from pypto import ir, passes
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops

_CORE_TYPES = [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV]


def _program(function_type, *, eval_stmt=False, tile=False, **options):
    span = ir.Span.unknown()
    operand_type = (
        ir.TileType([16, 16], ir.DataType.FP32) if tile else ir.TensorType([16, 16], ir.DataType.FP32)
    )
    source = ir.Var("source", operand_type, span)
    call = tile_ops.add(source, source) if tile else tensor_ops.add(source, source)
    statement = (
        ir.EvalStmt(call, span) if eval_stmt else ir.AssignStmt(ir.Var("result", call.type, span), call, span)
    )
    function = ir.Function(
        "kernel",
        [source],
        [],
        ir.SeqStmts([statement, ir.ReturnStmt([], span)], span),
        span,
        type=function_type,
        **options,
    )
    return ir.Program([function], "tile_op_property", span)


def _verify(program):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.IncoreTileOps)
    return passes.PropertyVerifierRegistry.verify(properties, program)


@pytest.mark.parametrize("function_type", _CORE_TYPES)
@pytest.mark.parametrize("eval_stmt", [False, True], ids=["assignment", "eval"])
def test_rejects_convertible_tensor_ops_in_all_functional_core_types(function_type, eval_stmt):
    """AIC/AIV must diagnose the same surviving tensor.add as generic InCore."""
    diagnostics = _verify(_program(function_type, eval_stmt=eval_stmt))
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "IncoreTileOps"
    assert "tensor.add" in diagnostics[0].message
    assert "should have been converted" in diagnostics[0].message


@pytest.mark.parametrize("function_type", _CORE_TYPES)
def test_accepts_tile_ops_in_all_functional_core_types(function_type):
    assert not _verify(_program(function_type, tile=True))


@pytest.mark.parametrize("function_type", _CORE_TYPES)
@pytest.mark.parametrize("excluded", ["buffer", "external", "runtime_bound"])
def test_excludes_the_same_core_bodies_as_conversion(function_type, excluded):
    """The isolated property must not impose a functional contract on excluded bodies."""
    options = {}
    if excluded == "buffer":
        options["ir_stage"] = ir.FunctionIRStage.Buffer
    elif excluded == "external":
        options["attrs"] = {"external_source": "kernel.cpp"}
    else:
        options["requires_runtime_binding"] = True
    assert not _verify(_program(function_type, **options))


@pytest.mark.parametrize(
    "function_type",
    [ir.FunctionType.Orchestration, ir.FunctionType.Opaque, ir.FunctionType.Group, ir.FunctionType.Spmd],
)
def test_leaves_noncore_tensor_ops_outside_the_postcondition(function_type):
    assert not _verify(_program(function_type))
