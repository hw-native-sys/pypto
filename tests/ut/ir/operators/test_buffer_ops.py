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


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
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


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
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


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_buffer_ops_are_internal_only(op_name, arg_count):
    args = [buffer_var(f"arg_{i}") for i in range(arg_count)]
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(op_name, args, ir.Span.unknown())


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_exact_input_destination_alias_is_legal(op_name, arg_count):
    value = buffer_var("shared")
    call = internal_call(op_name, [value] * arg_count)
    assert isinstance(call.type, ir.VoidType)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
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
@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_every_physical_descriptor_field_must_match(op_name, arg_count, difference):
    args = [buffer_var(f"arg_{i}") for i in range(arg_count - 1)] + [buffer_var("dst", **difference)]
    with pytest.raises(ValueError, match="identical physical descriptors"):
        internal_call(op_name, args)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_non_vector_buffer_is_rejected(op_name, arg_count):
    args = [buffer_var(f"arg_{i}", memory_space=ir.MemorySpace.Mat) for i in range(arg_count)]
    with pytest.raises(ValueError, match="must be in Vec memory"):
        internal_call(op_name, args)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_argument_arity_is_exact(op_name, arg_count):
    for count in (arg_count - 1, arg_count + 1):
        with pytest.raises(
            ValueError, match=rf"requires {arg_count} (?:buffer|explicit) operands, got {count}"
        ):
            internal_call(op_name, [buffer_var(f"arg_{i}") for i in range(count)])


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_logical_tiles_do_not_satisfy_buffer_schema(op_name, arg_count):
    tile = ir.Var("tile", ir.TileType([16, 32], DataType.FP32), ir.Span.unknown())
    with pytest.raises(ValueError, match="must have BufferType"):
        internal_call(op_name, [tile] * arg_count)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
def test_unknown_kwargs_do_not_silently_change_buffer_semantics(op_name, arg_count):
    with pytest.raises(ValueError, match="Unknown kwarg 'transpose'"):
        internal_call(op_name, [buffer_var(f"arg_{i}") for i in range(arg_count)], transpose=True)


@pytest.mark.parametrize("op_name,arg_count", [("buffer.copy", 2), ("buffer.mul", 3), ("buffer.add", 3)])
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


def transfer_args(name: str) -> list[ir.Expr]:
    tensor = ir.Var("gm", ir.TensorType([32, 64], DataType.FP32), ir.Span.unknown())
    buffer = buffer_var("buffer")
    source, destination = (tensor, buffer) if name == "buffer.load" else (buffer, tensor)
    return [source, valid_extents(8, 16), valid_extents(16, 32), destination]


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
def test_gm_transfer_contract_exposes_window_and_memory_effects(name):
    args = transfer_args(name)
    call = internal_call(name, args)
    assert isinstance(call.type, ir.VoidType)
    assert len(call.args) == 4
    assert ir.get_op_ir_stage(name) == ir.OpIRStage.Buffer
    assert ir.get_op_output_arity(name) == 0
    assert ir.get_op_buffer_result_spec(name).behavior == ir.BufferResultBehavior.None_
    for index, access in [(0, ir.BufferAccess.Read), (3, ir.BufferAccess.Write)]:
        effect = ir.get_op_buffer_arg_effect(name, index)
        assert effect.data == access
        assert effect.metadata == ir.BufferAccess.Read
        assert not effect.non_memory
    for index in (1, 2):
        assert ir.get_op_buffer_arg_effect(name, index).non_memory
    assert testing.get_execution_memory_access_evidence(name) == "unknown"
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(name, args, ir.Span.unknown())
    with pytest.raises(ValueError, match="Unknown kwarg 'cache'"):
        internal_call(name, args, cache=1)
    restored = ir.deserialize(ir.serialize(call))
    ir.assert_structural_equal(call, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tensor", ir.TensorType([32], DataType.FP32), "rank-2"),
        ("tensor", ir.TensorType([32, 64], DataType.FP16), "matching"),
        ("buffer", ir.BufferType([32], DataType.FP32, ir.Mem.Vec), "rank-2"),
        ("buffer", ir.BufferType([16, 32], DataType.FP16, ir.Mem.Vec), "matching"),
        ("buffer", ir.BufferType([16, 32], DataType.FP32, ir.Mem.Left), "Vec"),
        ("offsets", valid_extents(0), "rank-2 MakeTuple"),
        ("offsets", valid_extents(-1, 0), "nonnegative"),
        ("offsets", valid_extents(17, 0), "exceeds GM"),
        ("offsets", valid_extents(0, 33), "exceeds GM"),
        ("valid", valid_extents(16), "rank-2 MakeTuple"),
        ("valid", valid_extents(15, 32), "static valid dimension"),
        ("valid", valid_extents(-1, 32), "between 0 and"),
        ("valid", valid_extents(17, 32), "between 0 and"),
        ("offsets", valid_extents(ir.ConstFloat(1.0, DataType.FP32, ir.Span.unknown()), 0), "integer"),
    ],
)
def test_gm_transfer_rejects_invalid_types_and_windows(name, field, value, message):
    args = transfer_args(name)
    if field in ("tensor", "buffer"):
        index = (
            (0 if name == "buffer.load" else 3) if field == "tensor" else (3 if name == "buffer.load" else 0)
        )
        args[index] = ir.Var(field, value, ir.Span.unknown())
    else:
        args[1 if field == "offsets" else 2] = value
    with pytest.raises(ValueError, match=message):
        internal_call(name, args)


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
@pytest.mark.parametrize("valid_shape", [[23, 48], [24, 47]])
def test_gm_transfer_rejects_windows_outside_tensor_valid_region(name, valid_shape):
    args = transfer_args(name)
    tensor_type = ir.TensorType(
        [32, 64], DataType.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=valid_shape)
    )
    args[0 if name == "buffer.load" else 3] = ir.Var("gm", tensor_type, ir.Span.unknown())
    with pytest.raises(ValueError, match="exceeds GM tensor physical or valid dimension"):
        internal_call(name, args)


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
@pytest.mark.parametrize("valid_shape", [[24, 48], [32, 64]])
def test_gm_transfer_accepts_windows_within_tensor_valid_region(name, valid_shape):
    args = transfer_args(name)
    tensor_type = ir.TensorType(
        [32, 64], DataType.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=valid_shape)
    )
    args[0 if name == "buffer.load" else 3] = ir.Var("gm", tensor_type, ir.Span.unknown())
    call = internal_call(name, args)
    assert isinstance(call.type, ir.VoidType)
    restored = ir.deserialize(ir.serialize(call))
    ir.assert_structural_equal(call, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
def test_gm_transfer_requires_exact_operand_count_and_tuple_windows(name):
    args = transfer_args(name)
    for malformed in (args[:-1], [*args, args[-1]], [args[0], args[0], *args[2:]]):
        with pytest.raises(ValueError, match="requires"):
            internal_call(name, malformed)


@pytest.mark.parametrize("name", ["buffer.load", "buffer.store"])
def test_gm_transfer_dynamic_valid_extents_are_visible_operands(name):
    args = transfer_args(name)
    rows = ir.Var("rows", ir.ScalarType(DataType.UINT8), ir.Span.unknown())
    args[3 if name == "buffer.load" else 0] = buffer_var("buffer", valid_shape=[-1, 32])
    args[2] = valid_extents(rows, 32)
    call = internal_call(name, args)
    assert isinstance(call.args[2], ir.MakeTuple)
    assert call.args[2].elements[0].same_as(rows)
    # No metadata mutation is hidden in either transfer.
    assert (
        ir.get_op_buffer_arg_effect(name, 3 if name == "buffer.load" else 0).metadata == ir.BufferAccess.Read
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


@pytest.mark.parametrize("suffix", ["get_block_idx", "get_block_num", "get_subblock_idx"])
def test_buffer_spmd_queries_have_scalar_results_and_validate_schema(suffix):
    name = f"buffer.{suffix}"
    span = ir.Span.unknown()
    call = _ir._create_internal_op_call(name, [], {}, span)
    assert isinstance(call.type, ir.ScalarType) and call.type.dtype == DataType.INDEX
    assert ir.get_op_output_arity(name) == 1
    assert ir.get_op_ir_stage(name) == ir.OpIRStage.Buffer
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(name, [], {}, span)
    with pytest.raises(ValueError, match="requires no arguments"):
        _ir._create_internal_op_call(name, [ir.ConstInt(0, DataType.INDEX, span)], {}, span)
    with pytest.raises(ValueError, match="Unknown kwarg"):
        _ir._create_internal_op_call(name, [], {"unknown": True}, span)


_NZ = (ir.TileLayout.col_major, ir.TileLayout.row_major)
_ZN = (ir.TileLayout.row_major, ir.TileLayout.col_major)


def matrix_var(name: str, space: ir.MemorySpace, shape: list[int], dtype: DataType, **fields: Any) -> ir.Var:
    blayout, slayout = _ZN if space == ir.MemorySpace.Right else _NZ
    layout: dict[str, Any] = dict(
        blayout=blayout, slayout=slayout, fractal=1024 if space == ir.MemorySpace.Acc else 512
    )
    layout.update(fields)
    return buffer_var(name, shape=shape, dtype=dtype, memory_space=space, **layout)


def matmul_args(
    lhs_dtype: DataType = DataType.FP16, acc_dtype: DataType = DataType.FP32, **acc: Any
) -> list[ir.Expr]:
    return [
        matrix_var("lhs", ir.MemorySpace.Left, [16, 64], lhs_dtype),
        matrix_var("rhs", ir.MemorySpace.Right, [64, 32], lhs_dtype),
        matrix_var("acc", ir.MemorySpace.Acc, acc.pop("shape", [16, 32]), acc_dtype, **acc),
    ]


@pytest.mark.parametrize(
    "name,destination_access",
    [("buffer.matmul", ir.BufferAccess.Write), ("buffer.matmul_acc", ir.BufferAccess.ReadWrite)],
)
def test_matmul_writes_an_explicit_acc_destination(name, destination_access):
    call = internal_call(name, matmul_args())
    assert isinstance(call.type, ir.VoidType)
    assert ir.get_op_ir_stage(name) == ir.OpIRStage.Buffer
    assert ir.get_op_output_arity(name) == 0
    for index, access in [(0, ir.BufferAccess.Read), (1, ir.BufferAccess.Read), (2, destination_access)]:
        effect = ir.get_op_buffer_arg_effect(name, index)
        assert (effect.data, effect.metadata, effect.non_memory) == (access, ir.BufferAccess.Read, False)
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(name, matmul_args(), ir.Span.unknown())
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)


@pytest.mark.parametrize(
    "lhs_dtype,acc_dtype",
    [
        (DataType.FP16, DataType.FP32),
        (DataType.BF16, DataType.FP32),
        (DataType.FP32, DataType.FP32),
        (DataType.INT8, DataType.INT32),
    ],
)
def test_matmul_accepts_each_cube_accumulator_pairing(lhs_dtype, acc_dtype):
    for name in ("buffer.matmul", "buffer.matmul_acc"):
        assert isinstance(internal_call(name, matmul_args(lhs_dtype, acc_dtype)).type, ir.VoidType)


@pytest.mark.parametrize(
    "index,replacement,message",
    [
        (0, ("lhs", ir.MemorySpace.Mat, [16, 64], DataType.FP16), "rank-2 Left buffer"),
        (1, ("rhs", ir.MemorySpace.Left, [64, 32], DataType.FP16), "rank-2 Right buffer"),
        (2, ("acc", ir.MemorySpace.Vec, [16, 32], DataType.FP32), "rank-2 Acc buffer"),
        (1, ("rhs", ir.MemorySpace.Right, [64, 32], DataType.BF16), "identical lhs and rhs"),
        (2, ("acc", ir.MemorySpace.Acc, [16, 32], DataType.INT32), "fp32 accumulator"),
        (1, ("rhs", ir.MemorySpace.Right, [48, 32], DataType.FP16), r"\[M, K\] x \[K, N\]"),
        (2, ("acc", ir.MemorySpace.Acc, [32, 32], DataType.FP32), r"\[M, K\] x \[K, N\]"),
        (0, ("lhs", ir.MemorySpace.Left, [16, 64], DataType.INT32), "does not support int32 Left"),
    ],
)
def test_matmul_rejects_wrong_spaces_types_and_shapes(index, replacement, message):
    args = matmul_args()
    args[index] = matrix_var(*replacement)
    with pytest.raises(ValueError, match=message):
        internal_call("buffer.matmul", args)


def test_only_the_accumulating_form_may_write_a_wider_valid_rectangle():
    wider = matmul_args()
    wider[1] = matrix_var("rhs", ir.MemorySpace.Right, [64, 32], DataType.FP16, valid_shape=[64, 24])
    assert isinstance(internal_call("buffer.matmul_acc", wider).type, ir.VoidType)
    with pytest.raises(ValueError, match="must equal the product valid extent 24"):
        internal_call("buffer.matmul", wider)
    narrower = matmul_args(valid_shape=[16, 24])
    with pytest.raises(ValueError, match="must contain the product valid extent 32"):
        internal_call("buffer.matmul_acc", narrower)
    uncovered_k = matmul_args()
    uncovered_k[1] = matrix_var("rhs", ir.MemorySpace.Right, [64, 32], DataType.FP16, valid_shape=[48, 32])
    with pytest.raises(ValueError, match="rhs valid K to cover lhs valid K"):
        internal_call("buffer.matmul", uncovered_k)


def extract_args(
    row: int | ir.Expr = 16,
    col: int | ir.Expr = 0,
    space: ir.MemorySpace = ir.MemorySpace.Left,
    dtype: DataType = DataType.FP16,
    **destination: Any,
) -> list[ir.Expr]:
    span = ir.Span.unknown()
    source = matrix_var("mat", ir.MemorySpace.Mat, [64, 64], DataType.FP16)
    target = matrix_var("l0", space, [16, 64], dtype, **destination)
    offsets = [
        value if isinstance(value, ir.Expr) else ir.ConstInt(value, DataType.INDEX, span)
        for value in (row, col)
    ]
    return [source, *offsets, target]


def test_extract_exposes_runtime_offsets_as_non_memory_operands():
    row = ir.Var("row", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    call = internal_call("buffer.extract", extract_args(row=row))
    assert isinstance(call.type, ir.VoidType)
    effects = [ir.get_op_buffer_arg_effect("buffer.extract", index) for index in range(4)]
    assert [effect.non_memory for effect in effects] == [False, True, True, False]
    assert (effects[0].data, effects[3].data) == (ir.BufferAccess.Read, ir.BufferAccess.Write)
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    vec = [buffer_var("src", shape=[32, 32]), *extract_args()[1:3], buffer_var("dst", shape=[16, 32])]
    assert isinstance(internal_call("buffer.extract", vec).type, ir.VoidType)


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(row=49), r"row window \[49, 65\) exceeds the source extent 64"),
        (dict(col=16), r"column window \[16, 80\) exceeds the source extent 64"),
        (
            dict(space=ir.MemorySpace.Acc, dtype=DataType.FP32, fractal=1024),
            "Mat -> Left/Right and Vec -> Vec",
        ),
        (dict(dtype=DataType.BF16), "matching element types"),
        (dict(row=ir.ConstFloat(1.0, DataType.FP32, ir.Span.unknown())), "integer or INDEX"),
    ],
)
def test_extract_rejects_out_of_range_windows_and_unsupported_pairs(options, message):
    with pytest.raises(ValueError, match=message):
        internal_call("buffer.extract", extract_args(**options))


@pytest.mark.parametrize("space", [ir.MemorySpace.Left, ir.MemorySpace.Right])
def test_copy_moves_a_mat_operand_into_l0_with_its_own_layout(space):
    source = matrix_var("mat", ir.MemorySpace.Mat, [16, 64], DataType.FP16)
    assert isinstance(
        internal_call("buffer.copy", [source, matrix_var("l0", space, [16, 64], DataType.FP16)]).type,
        ir.VoidType,
    )
    for changed in (dict(shape=[16, 32]), dict(valid_shape=[16, 48])):
        options = dict(shape=[16, 64])
        options.update(changed)
        target = matrix_var("l0", space, options.pop("shape"), DataType.FP16, **options)
        with pytest.raises(ValueError, match="keep the element type and the physical and valid extents"):
            internal_call("buffer.copy", [source, target])


def test_copy_rejects_other_cross_space_moves():
    acc = matrix_var("acc", ir.MemorySpace.Acc, [16, 64], DataType.FP32)
    mat = matrix_var("mat", ir.MemorySpace.Mat, [16, 64], DataType.FP32)
    with pytest.raises(ValueError, match="got Acc -> Mat"):
        internal_call("buffer.copy", [acc, mat])


def test_mat_load_and_acc_store_are_the_only_matrix_transfers():
    span = ir.Span.unknown()
    window = [valid_extents(0, 0), valid_extents(16, 32)]
    fp16 = ir.Var("gm16", ir.TensorType([32, 64], DataType.FP16), span)
    assert isinstance(
        internal_call(
            "buffer.load", [fp16, *window, matrix_var("mat", ir.MemorySpace.Mat, [16, 32], DataType.FP16)]
        ).type,
        ir.VoidType,
    )
    accumulator = matrix_var("acc", ir.MemorySpace.Acc, [16, 32], DataType.FP32)
    assert isinstance(internal_call("buffer.store", [accumulator, *window, fp16]).type, ir.VoidType)
    integer = matrix_var("acc", ir.MemorySpace.Acc, [16, 32], DataType.INT32)
    with pytest.raises(ValueError, match="cannot convert int32 to fp16 without a scale"):
        internal_call("buffer.store", [integer, *window, fp16])
    mat = matrix_var("mat", ir.MemorySpace.Mat, [16, 32], DataType.FP16)
    with pytest.raises(ValueError, match="does not support Mat buffers"):
        internal_call("buffer.store", [mat, *window, fp16])
    with pytest.raises(ValueError, match="does not support Acc buffers"):
        internal_call(
            "buffer.load", [ir.Var("gm", ir.TensorType([32, 64], DataType.FP32), span), *window, accumulator]
        )


def test_matrix_reshape_relabels_one_window_in_its_space():
    source = matrix_var("nz", ir.MemorySpace.Mat, [64, 32], DataType.FP16)
    transposed = ir.BufferType([32, 64], DataType.FP16, ir.MemorySpace.Mat, [32, 64], *_ZN)
    call = _ir._create_internal_op_call("buffer.reshape", [source], {}, transposed, ir.Span.unknown())
    ir.assert_structural_equal(call.type, transposed)
    for result, message in [
        (ir.BufferType([32, 64], DataType.FP16, ir.MemorySpace.Left, [32, 64], *_NZ), "same memory space"),
        (
            ir.BufferType([32, 32], DataType.FP16, ir.MemorySpace.Mat, [32, 32], *_ZN),
            "equal physical byte sizes",
        ),
        (ir.BufferType([32, 32], DataType.FP32, ir.MemorySpace.Mat, [32, 32], *_ZN), "same element type"),
    ]:
        with pytest.raises(ValueError, match=message):
            _ir._create_internal_op_call("buffer.reshape", [source], {}, result, ir.Span.unknown())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
