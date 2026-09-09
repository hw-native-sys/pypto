# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Internal buffer writes expose destination operands and zero SSA results."""

from collections.abc import Sequence
from typing import Any

import pytest
from pypto import DataType, ir
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import testing


def buffer_var(name: str, **descriptor: Any) -> ir.Var:
    options: dict[str, Any] = dict(shape=[16, 32], dtype=DataType.FP32, memory_space=ir.MemorySpace.Vec)
    options.update(descriptor)
    return ir.Var(name, ir.BufferType(**options), ir.Span.unknown())


def internal_call(name: str, args: Sequence[ir.Expr], **kwargs: Any) -> ir.Call:
    return _ir._create_internal_op_call(name, args, kwargs, ir.Span.unknown())


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_buffer_write_has_explicit_destination_and_void_result(op_name, arg_count):
    args = [buffer_var(f"arg_{i}") for i in range(arg_count)]
    call = internal_call(op_name, args)
    assert isinstance(call.type, ir.VoidType)
    assert len(call.args) == arg_count
    ir.assert_structural_equal(call.args[-1], args[-1])
    assert isinstance(ir.EvalStmt(call, ir.Span.unknown()), ir.EvalStmt)
    assert ir.get_op_ir_stage(op_name) == ir.OpIRStage.Buffer
    assert ir.get_op_output_arity(op_name) == 0
    assert ir.get_op_buffer_result_spec(op_name).behavior == ir.BufferResultBehavior.None_
    assert ir.get_op_buffer_result_spec(op_name).alias_arg is None
    assert not ir.op_arg_is_workspace(op_name, arg_count - 1)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_data_and_metadata_effects_are_explicit(op_name, arg_count):
    for index in range(arg_count):
        effect = ir.get_op_buffer_arg_effect(op_name, index)
        assert not effect.non_memory
        assert effect.metadata == ir.BufferAccess.Read
        assert effect.data == (ir.BufferAccess.Write if index == arg_count - 1 else ir.BufferAccess.Read)
    with pytest.raises(ValueError, match="no buffer effect"):
        ir.get_op_buffer_arg_effect(op_name, arg_count)
    with pytest.raises(ValueError, match="requires GetBufferArgEffect"):
        ir.get_op_arg_effect(op_name, 0)
    assert testing.get_execution_memory_access_evidence(op_name) == "unknown"


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_buffer_ops_are_internal_only(op_name, arg_count):
    args = [buffer_var(f"arg_{i}") for i in range(arg_count)]
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(op_name, args, ir.Span.unknown())


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_exact_input_destination_alias_is_legal(op_name, arg_count):
    value = buffer_var("shared")
    call = internal_call(op_name, [value] * arg_count)
    assert isinstance(call.type, ir.VoidType)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_dynamic_valid_descriptor_is_preserved(op_name, arg_count):
    args = [buffer_var(f"arg_{i}", valid_shape=[-1, 32]) for i in range(arg_count)]
    call = internal_call(op_name, args)
    destination_type = call.args[-1].type
    assert isinstance(destination_type, ir.BufferType)
    assert list(destination_type.valid_shape) == [-1, 32]


@pytest.mark.parametrize(
    "difference",
    [
        {"shape": [8, 32]},
        {"dtype": DataType.FP16},
        {"valid_shape": [8, 32]},
        {"valid_shape": [-1, 32]},
        {"blayout": ir.TileLayout.col_major},
        {"slayout": ir.TileLayout.row_major},
        {"fractal": 1024},
        {"pad": ir.PadValue.zero},
        {"compact": ir.CompactMode.normal},
    ],
)
@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_every_physical_descriptor_field_must_match(op_name, arg_count, difference):
    args = [buffer_var(f"arg_{i}") for i in range(arg_count - 1)] + [buffer_var("dst", **difference)]
    with pytest.raises(ValueError, match="identical physical descriptors"):
        internal_call(op_name, args)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_non_vector_buffer_is_rejected(op_name, arg_count):
    args = [buffer_var(f"arg_{i}", memory_space=ir.MemorySpace.Mat) for i in range(arg_count)]
    with pytest.raises(ValueError, match="must be in Vec memory"):
        internal_call(op_name, args)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_argument_arity_is_exact(op_name, arg_count):
    for count in (arg_count - 1, arg_count + 1):
        with pytest.raises(ValueError, match="buffer operands"):
            internal_call(op_name, [buffer_var(f"arg_{i}") for i in range(count)])


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_logical_tiles_do_not_satisfy_buffer_schema(op_name, arg_count):
    tile = ir.Var("tile", ir.TileType([16, 32], DataType.FP32), ir.Span.unknown())
    with pytest.raises(ValueError, match="must have BufferType"):
        internal_call(op_name, [tile] * arg_count)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_unknown_kwargs_do_not_silently_change_buffer_semantics(op_name, arg_count):
    with pytest.raises(ValueError, match="Unknown kwarg 'transpose'"):
        internal_call(op_name, [buffer_var(f"arg_{i}") for i in range(arg_count)], transpose=True)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3)])
def test_buffer_statement_round_trip_preserves_destination_identity(op_name, arg_count):
    shared = buffer_var("shared")
    call = internal_call(op_name, [shared] * arg_count)
    statement = ir.EvalStmt(call, ir.Span.unknown())
    restored = ir.deserialize(ir.serialize(statement))
    ir.assert_structural_equal(statement, restored, enable_auto_mapping=True)
    assert isinstance(restored, ir.EvalStmt)
    assert isinstance(restored.expr, ir.Call)
    assert isinstance(restored.expr.type, ir.VoidType)
    assert restored.expr.args[-1].same_as(restored.expr.args[0])
    assert ir.get_op_ir_stage(restored.expr.op.name) == ir.OpIRStage.Buffer


def valid_extents(*values: int | ir.Expr) -> ir.MakeTuple:
    span = ir.Span.unknown()
    return ir.MakeTuple(
        [ir.ConstInt(value, DataType.INDEX, span) if isinstance(value, int) else value for value in values],
        span,
    )


def test_set_validshape_mutates_only_metadata_and_preserves_handle_type():
    buffer = buffer_var("buffer", valid_shape=[16, -1])
    columns = ir.Var("columns", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    call = internal_call("buffer.set_validshape", [buffer, valid_extents(16, columns)])
    assert isinstance(call.type, ir.VoidType)
    assert call.args[0].same_as(buffer)
    assert isinstance(buffer.type, ir.BufferType)
    assert buffer.type.valid_shape == [16, -1]
    effect = ir.get_op_buffer_arg_effect("buffer.set_validshape", 0)
    assert effect.data == ir.BufferAccess.None_
    assert effect.metadata == ir.BufferAccess.Write
    assert not effect.non_memory
    assert ir.get_op_buffer_arg_effect("buffer.set_validshape", 1).non_memory
    assert ir.get_op_output_arity("buffer.set_validshape") == 0
    assert ir.get_op_buffer_result_spec("buffer.set_validshape").behavior == ir.BufferResultBehavior.None_
    assert isinstance(ir.EvalStmt(call, ir.Span.unknown()), ir.EvalStmt)


@pytest.mark.parametrize("valid", [(0, 0), (8, 24), (16, 32)])
def test_set_validshape_accepts_empty_and_full_valid_regions(valid):
    call = internal_call(
        "buffer.set_validshape", [buffer_var("buffer", valid_shape=[-1, -1]), valid_extents(*valid)]
    )
    assert isinstance(call.type, ir.VoidType)


@pytest.mark.parametrize("valid", [(8, 32), (16, 24)])
def test_set_validshape_cannot_change_static_descriptor_dimensions(valid):
    with pytest.raises(ValueError, match="cannot change static valid dimension"):
        internal_call("buffer.set_validshape", [buffer_var("buffer"), valid_extents(*valid)])


def test_set_validshape_cannot_make_a_static_dimension_dynamic():
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    with pytest.raises(ValueError, match="descriptor must already mark changing dimensions dynamic"):
        internal_call("buffer.set_validshape", [buffer_var("buffer"), valid_extents(rows, 32)])


@pytest.mark.parametrize("valid", [(-1, 32), (17, 32), (16, 33)])
def test_set_validshape_checks_constant_extent_bounds(valid):
    with pytest.raises(ValueError, match="valid extent for dimension .* must be between"):
        internal_call(
            "buffer.set_validshape", [buffer_var("buffer", valid_shape=[-1, -1]), valid_extents(*valid)]
        )


@pytest.mark.parametrize("dtype", [DataType.FP32, DataType.BOOL, DataType.TASK_ID])
def test_set_validshape_rejects_non_integer_extents(dtype):
    value = ir.Var("extent", ir.ScalarType(dtype), ir.Span.unknown())
    with pytest.raises(ValueError, match="integer or INDEX scalar"):
        internal_call(
            "buffer.set_validshape", [buffer_var("buffer", valid_shape=[-1, 32]), valid_extents(value, 32)]
        )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_set_validshape_requires_every_descriptor_dimension(count):
    with pytest.raises(ValueError, match="one valid extent per physical dimension"):
        internal_call("buffer.set_validshape", [buffer_var("buffer"), valid_extents(*([16] * count))])


def test_set_validshape_requires_an_explicit_tuple():
    dims = ir.Var("dims", ir.TupleType([ir.ScalarType(DataType.INDEX)] * 2), ir.Span.unknown())
    with pytest.raises(ValueError, match="valid extents must be a MakeTuple"):
        internal_call("buffer.set_validshape", [buffer_var("buffer"), dims])


def test_set_validshape_runtime_operand_survives_rewriting_and_serialization():
    buffer = buffer_var("buffer", valid_shape=[-1, 32])
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    new_rows = ir.Var("new_rows", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    call = internal_call("buffer.set_validshape", [buffer, valid_extents(rows, 32)])
    rewritten = ir.substitute_expr(call, [(rows, new_rows)])
    assert isinstance(rewritten, ir.Call)
    assert rewritten.args[0].same_as(buffer)
    assert isinstance(rewritten.args[1], ir.MakeTuple)
    assert rewritten.args[1].elements[0].same_as(new_rows)
    restored = ir.deserialize(ir.serialize(rewritten))
    ir.assert_structural_equal(restored, rewritten, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
