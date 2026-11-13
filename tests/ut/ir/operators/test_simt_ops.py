# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import simt

SPAN = ir.Span.unknown()


@pytest.mark.parametrize("query", [simt.thread_idx, simt.block_idx, simt.block_dim, simt.grid_dim])
@pytest.mark.parametrize("axis", [ir.SimtAxis.X, ir.SimtAxis.Y, ir.SimtAxis.Z])
def test_typed_queries_round_trip(query, axis):
    call = query(axis)
    assert call.type.dtype == DataType.INT32
    assert call.kwargs["axis"] == axis
    restored = ir.deserialize(ir.serialize(call))
    ir.assert_structural_equal(call, restored)
    assert ir.structural_hash(call) == ir.structural_hash(restored)


@pytest.mark.parametrize("axis", [0, 1, "X", "x", True])
def test_axis_is_not_an_integer_or_string(axis):
    with pytest.raises(TypeError, match="axis|SimtAxis|type"):
        simt.thread_idx(axis)


def test_query_axis_is_required_and_lane_has_none():
    with pytest.raises(ValueError, match="axis"):
        ir.create_op_call("simt.thread_idx", [], {}, SPAN)
    lane_type = simt.lane_id().type
    assert isinstance(lane_type, ir.ScalarType)
    assert lane_type.dtype == DataType.INT32
    with pytest.raises(ValueError):
        ir.create_op_call("simt.lane_id", [], {"axis": ir.SimtAxis.X}, SPAN)


@pytest.mark.parametrize("storage_type", [ir.TensorType, ir.TileType])
def test_scalar_memory_operations(storage_type):
    storage = ir.Var("storage", storage_type([16, 32], DataType.FP32), SPAN)
    load = simt.load(storage, [0, 1])
    assert isinstance(load.type, ir.ScalarType)
    assert load.type.dtype == DataType.FP32
    store = simt.store(storage, [0, 1], load)
    assert isinstance(store.type, ir.VoidType)
    assert ir.get_op_output_arity("simt.store") == 0
    with pytest.raises(ValueError, match="Void|void|value"):
        ir.AssignStmt(ir.Var("bad", ir.ScalarType(DataType.FP32), SPAN), store, SPAN)
    with pytest.raises(ValueError, match="dimension"):
        simt.load(storage, [1])
    with pytest.raises(ValueError, match="integer"):
        simt.load(storage, [ir.ConstFloat(0.5, DataType.FP32, SPAN), 1])
    with pytest.raises(ValueError, match="dtype"):
        simt.store(storage, [0, 1], ir.ConstInt(0, DataType.INT32, SPAN))


def test_gm_cache_policy_domains():
    tensor = ir.Var("gm", ir.TensorType([32], DataType.FP32), SPAN)
    value = simt.load(tensor, [0], l1cache="uncache", l2cache="nakeep")
    simt.store(tensor, [0], value, l1cache="cache", l2cache="wtsred")
    with pytest.raises(ValueError, match="store.*l2cache"):
        simt.store(tensor, [0], value, l2cache="nakeep")
    with pytest.raises(ValueError, match="load.*l2cache"):
        simt.load(tensor, [0], l2cache="wtsred")
    with pytest.raises(ValueError, match="l1cache"):
        simt.load(tensor, [0], l1cache="invalid")
    ub = ir.Var("ub", ir.TileType([32], DataType.FP32, memory_space=ir.MemorySpace.Vec), SPAN)
    with pytest.raises(ValueError, match="only to GM"):
        simt.load(ub, [0], l2cache="nmfv")


def test_cube_storage_is_not_accessible():
    storage = ir.Var("acc", ir.TileType([32], DataType.FP32, memory_space=ir.MemorySpace.Acc), SPAN)
    with pytest.raises(ValueError, match="Vec"):
        simt.load(storage, [0])


@pytest.mark.parametrize(
    "op,arity",
    [("load", 0), ("load", 1), ("load", 3), ("store", 0), ("store", 1), ("store", 2), ("store", 4)],
)
def test_memory_op_arity_checked_before_type_deduction(op, arity):
    # The canonical parser and low-level IR builder bypass the Python wrappers.
    # Too few operands must produce a diagnostic rather than native out-of-bounds access.
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), SPAN)
    indices = ir.MakeTuple([ir.ConstInt(0, DataType.INT32, SPAN)], SPAN)
    value = ir.ConstFloat(1.0, DataType.FP32, SPAN)
    args = [storage, indices, value, value][:arity]
    expected = 2 if op == "load" else 3
    with pytest.raises(ValueError, match=f"simt.{op} requires exactly {expected} arguments, got {arity}"):
        ir.create_op_call(f"simt.{op}", args, {}, SPAN)


@pytest.mark.parametrize("storage_type", [ir.TensorType, ir.TileType])
@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
def test_packed_fp4_storage_is_not_scalar(storage_type, dtype):
    # Packed FP4 has an eight-bit carrier, but each carrier contains two elements.
    storage = ir.Var("packed", storage_type([16], dtype), SPAN)
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    with pytest.raises(ValueError, match="unpacked byte-addressable"):
        simt.load(storage, [0])
    with pytest.raises(ValueError, match="unpacked byte-addressable"):
        simt.store(storage, [0], value)


VOTES = ("vote_all", "vote_any", "vote_uni", "vote_ballot")
SHUFFLES = ("shuffle_idx", "shuffle_up", "shuffle_down", "shuffle_bfly")
REDUCTIONS = ("redux_add", "redux_min", "redux_max")


@pytest.mark.parametrize("name", VOTES)
def test_vote_predicate_and_round_trip(name):
    pred = ir.Var("pred", ir.ScalarType(DataType.BOOL), SPAN)
    call = getattr(simt, name)(pred)
    assert call.type.dtype == (DataType.INT32 if name == "vote_ballot" else DataType.BOOL)
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    with pytest.raises(ValueError, match="BOOL predicate"):
        getattr(simt, name)(simt.lane_id())


@pytest.mark.parametrize("name", SHUFFLES + REDUCTIONS)
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP32])
def test_numeric_collective_type_and_round_trip(name, dtype):
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    args = [value, 1] if name in SHUFFLES else [value]
    call = getattr(simt, name)(*args)
    assert call.type.dtype == dtype
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    assert ir.op_has_declared_arg_effects(call.op.name)
    assert ir.get_op_arg_effect(call.op.name, 0) == ir.ArgEffect.Read


@pytest.mark.parametrize("name", SHUFFLES + REDUCTIONS)
@pytest.mark.parametrize("dtype", [DataType.BOOL, DataType.INDEX, DataType.UINT32, DataType.FP16])
def test_collective_rejects_unsupported_payload(name, dtype):
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    args = [value, 1] if name in SHUFFLES else [value]
    with pytest.raises(ValueError, match="INT32 or FP32"):
        getattr(simt, name)(*args)


@pytest.mark.parametrize("name", SHUFFLES)
@pytest.mark.parametrize("width", [16, 32])
def test_shuffle_width_and_control(name, width):
    call = getattr(simt, name)(simt.lane_id(), 1, width=width)
    assert call.kwargs["width"] == width
    assert call.args[1].dtype == DataType.INT32
    with pytest.raises(ValueError, match="INT32 lane control"):
        getattr(simt, name)(simt.lane_id(), ir.ConstInt(1, DataType.INDEX, SPAN))


@pytest.mark.parametrize("width", [0, 1, 8, 24, 64])
def test_shuffle_rejects_unsupported_width(width):
    with pytest.raises(ValueError, match="width must be 16 or 32"):
        simt.shuffle_idx(simt.lane_id(), 0, width=width)


@pytest.mark.parametrize("name", VOTES + SHUFFLES + REDUCTIONS)
@pytest.mark.parametrize("arity", [0, 3])
def test_collective_checks_arity_before_accessing_arguments(name, arity):
    with pytest.raises(ValueError, match="requires exactly"):
        ir.create_op_call(f"simt.{name}", [simt.lane_id()] * arity, {}, SPAN)


@pytest.mark.parametrize("name", VOTES + SHUFFLES + REDUCTIONS)
def test_dsl_collectives_match_ir_wrappers(name):
    value = ir.Var("value", ir.ScalarType(DataType.BOOL if name in VOTES else DataType.FP32), SPAN)
    args = [value, 1] if name in SHUFFLES else [value]
    call = getattr(pl.simt, name)(*args).unwrap()
    ir.assert_structural_equal(call, getattr(simt, name)(*args))


@pytest.mark.parametrize("control", [-(1 << 31) - 1, 1 << 31])
def test_shuffle_control_literal_overflow(control):
    with pytest.raises(ValueError, match="must fit INT32"):
        simt.shuffle_idx(simt.lane_id(), control)


@pytest.mark.parametrize("width", [True, 16.0, "16"])
def test_shuffle_width_requires_integer_attribute(width):
    with pytest.raises(TypeError, match="width|type"):
        simt.shuffle_idx(simt.lane_id(), 0, width=width)


@pytest.mark.parametrize("control", [True, False])
def test_shuffle_control_does_not_coerce_bool(control):
    with pytest.raises(TypeError, match="INT32, got bool"):
        simt.shuffle_idx(simt.lane_id(), control)


@pytest.mark.parametrize("name", VOTES)
@pytest.mark.parametrize("predicate", [True, False])
def test_vote_accepts_python_bool_predicate(name, predicate):
    call = getattr(simt, name)(predicate)
    assert isinstance(call.args[0], ir.ConstBool)
    assert call.args[0].value == predicate
    ir.assert_structural_equal(call, getattr(pl.simt, name)(predicate).unwrap())


ATOMICS = ("atomic_add", "atomic_min", "atomic_max", "atomic_exch", "atomic_cas")


@pytest.mark.parametrize("name", ATOMICS)
@pytest.mark.parametrize("storage_type", [ir.TensorType, ir.TileType, ir.DistributedTensorType])
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP32])
def test_atomic_types_effects_and_round_trip(name, storage_type, dtype):
    storage = ir.Var(
        "target", storage_type([ir.ConstInt(d, DataType.INDEX, SPAN) for d in (4, 8)], dtype), SPAN
    )
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    args = [storage, [1, 2], value] + ([value] if name == "atomic_cas" else [])
    call = getattr(simt, name)(*args)
    assert call.type.dtype == dtype
    assert ir.get_op_arg_effect(call.op.name, 0) == ir.ArgEffect.ReadWrite
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    ir.assert_structural_equal(call, getattr(pl.simt, name)(*args).unwrap())
    helper = ir.Function(
        "atomic",
        [(storage, ir.ParamDirection.InOut), value],
        [],
        ir.EvalStmt(call, SPAN),
        SPAN,
        ir.FunctionType.SIMT,
    )
    access = ir.analyze_simt_access(helper)
    assert access[0].reads and access[0].writes
    assert not access[1].reads and not access[1].writes


@pytest.mark.parametrize("name", ATOMICS)
@pytest.mark.parametrize("dtype", [DataType.INT64, DataType.UINT32, DataType.FP16, DataType.BOOL])
def test_atomic_rejects_unsupported_types(name, dtype):
    storage = ir.Var("target", ir.TensorType([8], dtype), SPAN)
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    with pytest.raises(ValueError, match="INT32 or FP32|byte-addressable"):
        getattr(simt, name)(storage, [0], *([value] * (2 if name == "atomic_cas" else 1)))


@pytest.mark.parametrize("name", ATOMICS)
@pytest.mark.parametrize("arity", [0, 1, 2, 5])
def test_atomic_checks_arity_before_accessing_arguments(name, arity):
    with pytest.raises(ValueError, match="requires exactly"):
        ir.create_op_call(f"simt.{name}", [simt.lane_id()] * arity, {}, SPAN)


def test_atomic_validation_and_cache():
    storage = ir.Var("gm", ir.TensorType([8], DataType.INT32), SPAN)
    tile = ir.Var("ub", ir.TileType([8], DataType.INT32), SPAN)
    integer = simt.lane_id()
    floating = ir.ConstFloat(1.0, DataType.FP32, SPAN)
    with pytest.raises(ValueError, match="value dtype"):
        simt.atomic_add(storage, [0], floating)
    with pytest.raises(ValueError, match="compare dtype"):
        simt.atomic_cas(storage, [0], floating, integer)
    with pytest.raises(ValueError, match="one index per"):
        simt.atomic_add(storage, [0, 0], integer)
    with pytest.raises(ValueError, match="only to GM"):
        simt.atomic_add(tile, [0], integer, l2cache="nmfv")
    with pytest.raises(ValueError, match="l2cache"):
        simt.atomic_add(storage, [0], integer, l2cache="nmpref")
    assert simt.atomic_add(storage, [0], integer, l2cache="nmfv").kwargs["l2cache"] == "nmfv"


@pytest.mark.parametrize("name", ["syncthreads", "threadfence", "threadfence_block"])
def test_synchronization_void_and_round_trip(name):
    call = getattr(simt, name)()
    assert isinstance(call.type, ir.VoidType)
    ir.assert_structural_equal(call, getattr(pl.simt, name)())
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    with pytest.raises(ValueError, match="no arguments"):
        ir.create_op_call(f"simt.{name}", [simt.lane_id()], {}, SPAN)


MATH_ARITIES = {
    "abs": 1,
    "min": 2,
    "max": 2,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "sin": 1,
    "cos": 1,
    "ceil": 1,
    "floor": 1,
    "rint": 1,
    "round": 1,
    "pow": 2,
    "fma": 3,
    "prmt": 3,
    "mulhi": 2,
    "ftoi": 1,
    "itof": 1,
}


@pytest.mark.parametrize("name,arity", MATH_ARITIES.items())
def test_scalar_math_types_and_round_trip(name, arity):
    dtype = DataType.INT32 if name in {"prmt", "mulhi", "itof"} else DataType.FP32
    args = [ir.Var(f"x{i}", ir.ScalarType(dtype), SPAN) for i in range(arity)]
    call = getattr(simt, name)(*args)
    expected = DataType.INT32 if name == "ftoi" else DataType.FP32 if name == "itof" else dtype
    assert call.type.dtype == expected
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    ir.assert_structural_equal(call, getattr(pl.simt, name)(*args).unwrap())
    for count in (0, arity + 1):
        with pytest.raises(ValueError, match="requires exactly"):
            ir.create_op_call(f"simt.{name}", (args * 4)[:count], {}, SPAN)
    with pytest.raises(ValueError, match="does not accept|Unknown|unknown"):
        ir.create_op_call(f"simt.{name}", args, {"fastmath": True}, SPAN)


@pytest.mark.parametrize("name,arity", MATH_ARITIES.items())
@pytest.mark.parametrize(
    "dtype", [DataType.FP16, DataType.BF16, DataType.INT64, DataType.UINT32, DataType.BOOL]
)
def test_scalar_math_rejects_unverified_dtypes(name, arity, dtype):
    args = [ir.Var(f"x{i}", ir.ScalarType(dtype), SPAN) for i in range(arity)]
    with pytest.raises(ValueError, match="scalar operands"):
        getattr(simt, name)(*args)


@pytest.mark.parametrize("name", ["ftoi", "itof"])
@pytest.mark.parametrize("rounding", ["r", "a", "f", "c", "z"])
@pytest.mark.parametrize("unsigned", [False, True])
def test_explicit_conversion_controls(name, rounding, unsigned):
    dtype = DataType.FP32 if name == "ftoi" else DataType.INT32
    arg = ir.Var("x", ir.ScalarType(dtype), SPAN)
    call = getattr(simt, name)(arg, rounding=rounding, unsigned=unsigned)
    assert call.kwargs == {"rounding": rounding, "unsigned": unsigned}
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)


@pytest.mark.parametrize("name", ["ftoi", "itof"])
@pytest.mark.parametrize("rounding", ["", "R", "nearest", "o", "h", 1, True])
def test_conversion_rejects_invalid_rounding(name, rounding):
    dtype = DataType.FP32 if name == "ftoi" else DataType.INT32
    arg = ir.Var("x", ir.ScalarType(dtype), SPAN)
    with pytest.raises((ValueError, TypeError), match="rounding|type"):
        getattr(simt, name)(arg, rounding=rounding)


@pytest.mark.parametrize("name", ["ftoi", "itof", "mulhi"])
@pytest.mark.parametrize("unsigned", [1, "unsigned"])
def test_conversion_signedness_is_boolean(name, unsigned):
    dtype = DataType.FP32 if name == "ftoi" else DataType.INT32
    args = [ir.Var("x", ir.ScalarType(dtype), SPAN)] * MATH_ARITIES[name]
    with pytest.raises((ValueError, TypeError), match="unsigned|type"):
        getattr(simt, name)(*args, unsigned=unsigned)


@pytest.mark.parametrize("name,arity", MATH_ARITIES.items())
def test_scalar_math_rejects_storage_and_mixed_operands(name, arity):
    dtype = DataType.INT32 if name in {"prmt", "mulhi", "itof"} else DataType.FP32
    values = [ir.Var(f"x{i}", ir.ScalarType(dtype), SPAN) for i in range(arity)]
    invalid = [
        ir.Var("storage", ir.TensorType([32], dtype), SPAN),
        ir.Var("wrong", ir.ScalarType(DataType.FP32 if dtype == DataType.INT32 else DataType.INT32), SPAN),
    ]
    for value in invalid:
        # Put the mismatch last so a deducer checking only the first operand fails.
        with pytest.raises(ValueError, match="scalar operands"):
            getattr(simt, name)(*values[:-1], value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
