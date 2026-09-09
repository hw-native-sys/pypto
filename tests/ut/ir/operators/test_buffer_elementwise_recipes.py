# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typed elementwise recipes preserve physical operands, effects, and precision."""

from typing import Any

import pytest
from pypto import DataType, backend, ir, passes
from pypto.pypto_core import ir as _ir

_SPAN = ir.Span.unknown()
_RECIPES = [
    ("add", 2),
    ("mul", 2),
    ("sub", 2),
    ("div", 2),
    ("maximum", 2),
    ("minimum", 2),
    ("abs", 1),
    ("exp", 1),
    ("sqrt", 1),
    ("neg", 1),
    ("relu", 1),
    ("log", 1),
    ("recip", 1),
]


def _buffer(name, **changes):
    options: dict[str, Any] = dict(shape=[16, 32], dtype=DataType.FP32, memory_space=ir.MemorySpace.Vec)
    options.update(changes)
    return ir.Var(name, ir.BufferType(**options), _SPAN)


def _call(suffix, args, **kwargs):
    return _ir._create_internal_op_call(f"buffer.{suffix}", args, kwargs, _SPAN)


def test_reported_recipes_are_the_actual_sorted_independent_snapshot():
    expected = sorted(ir.get_op(f"tile.{suffix}").name for suffix, _ in _RECIPES)
    names = backend.get_buffer_elementwise_recipe_names()
    assert names == expected
    names.clear()
    assert backend.get_buffer_elementwise_recipe_names() == expected


@pytest.mark.parametrize("suffix,input_count", _RECIPES)
def test_recipe_operand_contract_and_effects(suffix, input_count):
    args = [_buffer(f"arg_{i}") for i in range(input_count + 1)]
    call = _call(suffix, args)
    assert isinstance(call.type, ir.VoidType)
    destination = call.args[-1]
    assert isinstance(destination, ir.Var)
    assert destination.unique_id == args[-1].unique_id
    assert ir.get_op_output_arity(call.op.name) == 0
    assert ir.get_op_ir_stage(call.op.name) == ir.OpIRStage.Buffer
    for i in range(input_count + 1):
        effect = ir.get_op_buffer_arg_effect(call.op.name, i)
        assert effect.data == (ir.BufferAccess.Write if i == input_count else ir.BufferAccess.Read)
        assert effect.metadata == ir.BufferAccess.Read
        assert not effect.non_memory and not ir.op_arg_is_workspace(call.op.name, i)
    with pytest.raises(ValueError, match="buffer operands"):
        _call(suffix, args[:-1])
    with pytest.raises(ValueError, match="buffer operands"):
        _call(suffix, [*args, args[-1]])
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(call.op.name, args, _SPAN)


@pytest.mark.parametrize("suffix,input_count", [(s, n) for s, n in _RECIPES if s != "recip"])
def test_exact_inplace_recipe_reuses_the_explicit_handle(suffix, input_count):
    shared = _buffer("shared")
    call = _call(suffix, [shared] * (input_count + 1))
    restored = ir.deserialize(ir.serialize(ir.EvalStmt(call, _SPAN)))
    assert isinstance(restored, ir.EvalStmt) and isinstance(restored.expr, ir.Call)
    identities = set()
    for argument in restored.expr.args:
        assert isinstance(argument, ir.Var)
        identities.add(argument.unique_id)
    assert len(identities) == 1
    ir.assert_structural_equal(ir.EvalStmt(call, _SPAN), restored, enable_auto_mapping=True)


@pytest.mark.parametrize("suffix,input_count", [(s, n) for s, n in _RECIPES if s not in {"add", "mul"}])
def test_new_recipes_do_not_infer_fp16_legality_from_descriptor_support(suffix, input_count):
    with pytest.raises(ValueError, match="requires FP32"):
        _call(suffix, [_buffer(f"arg_{i}", dtype=DataType.FP16) for i in range(input_count + 1)])


@pytest.mark.parametrize("suffix,input_count", [("exp", 1), ("sub", 2)])
@pytest.mark.parametrize(
    "changes",
    [
        {"valid_shape": [-1, 32]},
        {"blayout": ir.TileLayout.col_major},
        {"shape": [2, 16, 32]},
    ],
)
def test_fixed_descriptor_recipes_reject_unimplemented_physical_forms(suffix, input_count, changes):
    with pytest.raises(ValueError, match="static dense rank-1/rank-2"):
        _call(suffix, [_buffer(f"arg_{i}", **changes) for i in range(input_count + 1)])


@pytest.mark.parametrize("suffix,input_count", [("log", 1), ("minimum", 2)])
def test_recipe_requires_matching_descriptors_and_real_buffers(suffix, input_count):
    args = [_buffer(f"arg_{i}") for i in range(input_count)]
    with pytest.raises(ValueError, match="identical physical descriptors"):
        _call(suffix, [*args, _buffer("dst", valid_shape=[15, 32])])
    with pytest.raises(ValueError, match="must have BufferType"):
        _call(suffix, [*args, ir.ConstFloat(1.0, DataType.FP32, _SPAN)])
    with pytest.raises(ValueError, match="Unknown kwarg 'transpose'"):
        _call(suffix, [*args, _buffer("dst")], transpose=True)


def test_reciprocal_rejects_input_destination_alias():
    shared = _buffer("shared")
    with pytest.raises(ValueError, match="requires a distinct destination"):
        _call("recip", [shared, shared])


@pytest.mark.parametrize("suffix,input_count", [("div", 2), ("log", 1), ("recip", 1)])
@pytest.mark.parametrize("high_precision", [False, True])
def test_precision_survives_verification_and_serialization(suffix, input_count, high_precision):
    args = [_buffer(f"arg_{i}") for i in range(input_count + 1)]
    call = _call(suffix, args, high_precision=high_precision)
    allocations = [
        ir.AssignStmt(
            argument,
            ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], _SPAN)], {}, None, argument.type, _SPAN),
            _SPAN,
        )
        for argument in args
    ]
    program = ir.Program(
        [
            ir.Function(
                "kernel",
                [],
                [],
                ir.SeqStmts([*allocations, ir.EvalStmt(call, _SPAN)], _SPAN),
                _SPAN,
                type=ir.FunctionType.InCore,
                ir_stage=ir.FunctionIRStage.Buffer,
            )
        ],
        "Precision",
        _SPAN,
    )
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    function = next(iter(restored.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    statement = function.body.stmts[-1]
    assert isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call)
    assert statement.expr.kwargs["high_precision"] is high_precision


def test_unmodeled_precision_kwarg_is_rejected():
    with pytest.raises(ValueError, match="Unknown kwarg 'high_precision'"):
        _call("exp", [_buffer("src"), _buffer("dst")], high_precision=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
