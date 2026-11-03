# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
