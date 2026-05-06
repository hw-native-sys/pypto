# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for cross-core communication ops and MixedKernelExpanded IRProperty."""

import pytest
from pypto import DataType, ir, passes
from pypto.pypto_core.ir import ConstInt


def test_tpush_ops_return_unknown_type():
    """Test tpush ops return UnknownType."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64], DataType.FP32)
    tile_var = ir.Var("t", tile_type, span)

    for op_name in ["tile.tpush_to_aiv", "tile.tpush_to_aic"]:
        call = ir.create_op_call(op_name, [tile_var], {"split": 0}, span)
        assert isinstance(call.type, ir.UnknownType)


def test_frontend_pipe_id_kwarg_is_accepted():
    """Cross-core frontend ops accept an optional PTOAS pipe id."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64], DataType.FP32)
    tile_var = ir.Var("t", tile_type, span)
    z = ConstInt(0, DataType.INT32, span)

    tpush = ir.create_op_call("tile.tpush_to_aiv", [tile_var], {"split": 0, "id": 7}, span)
    assert tpush.kwargs["id"] == 7

    tpop_op = ir.get_op("tile.tpop_from_aic")
    tpop = ir.Call(tpop_op, [], {"split": 0, "id": 7}, tile_type, span)
    assert tpop.kwargs["id"] == 7

    tfree = ir.create_op_call("system.tfree_to_aic", [tile_var], {"id": 7}, span)
    assert tfree.kwargs["id"] == 7

    init = ir.create_op_call(
        "system.aiv_initialize_pipe",
        [z, z],
        {"dir_mask": 1, "slot_size": 256, "id": 7},
        span,
    )
    assert init.kwargs["id"] == 7


def test_tpop_ops_return_tile_type():
    """Test tpop ops return TileType when constructed with explicit type."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64], DataType.FP32)

    for op_name in ["tile.tpop_from_aic", "tile.tpop_from_aiv"]:
        op = ir.get_op(op_name)
        call = ir.Call(op, [], {"split": 0}, tile_type, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.shape == [64]
        assert call.type.dtype == DataType.FP32


def test_initialize_pipe_ops():
    """Test initialize_pipe ops take two i32 buffer operands and return UnknownType.

    dir_mask=1: only C2V is active; c2v_consumer_buf must be concrete, v2c uses placeholder zero.
    """
    span = ir.Span.unknown()
    z = ConstInt(0, DataType.INT32, span)
    c2v_base = ir.Var("c2v_base", ir.ScalarType(DataType.INT32), span)

    for op_name in ["system.aic_initialize_pipe", "system.aiv_initialize_pipe"]:
        call = ir.create_op_call(op_name, [c2v_base, z], {"dir_mask": 1, "slot_size": 256}, span)
        assert isinstance(call.type, ir.UnknownType)


def test_reserve_buffer_op():
    """Test reserve_buffer op accepts no args and returns ScalarType(INT32)."""
    span = ir.Span.unknown()
    call = ir.create_op_call("system.reserve_buffer", [], {"name": "shared_buf", "size": 1024}, span)
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT32


def test_import_peer_buffer_op():
    """Test import_peer_buffer op accepts no args and returns ScalarType(INT32)."""
    span = ir.Span.unknown()
    call = ir.create_op_call(
        "system.import_peer_buffer", [], {"name": "shared_buf", "peer_func": "aic_kernel"}, span
    )
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT32


def test_cross_core_ops_registered():
    """Test all cross-core ops are registered."""
    op_names = [
        "tile.tpush_to_aiv",
        "tile.tpush_to_aic",
        "tile.tpop_from_aic",
        "tile.tpop_from_aiv",
        "system.aic_initialize_pipe",
        "system.aiv_initialize_pipe",
        "system.reserve_buffer",
        "system.import_peer_buffer",
    ]
    for name in op_names:
        assert ir.is_op_registered(name), f"{name} should be registered"


def test_mixed_kernel_expanded_property():
    """Test IRProperty.MixedKernelExpanded works with IRPropertySet."""
    prop_set = passes.IRPropertySet()
    prop_set.insert(passes.IRProperty.MixedKernelExpanded)
    assert prop_set.contains(passes.IRProperty.MixedKernelExpanded)
    assert not prop_set.contains(passes.IRProperty.SSAForm)

    prop_set.remove(passes.IRProperty.MixedKernelExpanded)
    assert not prop_set.contains(passes.IRProperty.MixedKernelExpanded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
