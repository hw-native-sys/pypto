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
from pypto import language as pl
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


def test_aiv_shard_halves_split_axis():
    """tile.aiv_shard halves the split axis: UP_DOWN -> axis0, LEFT_RIGHT -> axis1."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([16, 128], DataType.FP32), span)

    up_down = ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)
    assert isinstance(up_down.type, ir.TileType)
    assert up_down.type.shape == [8, 128]
    assert up_down.type.dtype == DataType.FP32

    left_right = ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 2}, span)
    assert isinstance(left_right.type, ir.TileType)
    assert left_right.type.shape == [16, 64]
    assert left_right.type.dtype == DataType.FP32


def test_aic_gather_doubles_split_axis():
    """tile.aic_gather is the inverse of aiv_shard: it doubles the split axis."""
    span = ir.Span.unknown()

    up_down = ir.create_op_call(
        "tile.aic_gather", [ir.Var("t", ir.TileType([8, 128], DataType.FP32), span)], {"split": 1}, span
    )
    assert isinstance(up_down.type, ir.TileType)
    assert up_down.type.shape == [16, 128]

    left_right = ir.create_op_call(
        "tile.aic_gather", [ir.Var("t", ir.TileType([16, 64], DataType.FP32), span)], {"split": 2}, span
    )
    assert isinstance(left_right.type, ir.TileType)
    assert left_right.type.shape == [16, 128]


def test_split_reshape_rejects_non_2d_tile():
    """aiv_shard / aic_gather require a 2D tile."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([2, 16, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="2D tile"):
        ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)
    with pytest.raises(ValueError, match="2D tile"):
        ir.create_op_call("tile.aic_gather", [tile_var], {"split": 1}, span)


def test_split_reshape_rejects_bad_split_attr():
    """split must be 1 or 2 (0 and out-of-range rejected)."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([16, 128], DataType.FP32), span)

    for bad in (0, 3):
        with pytest.raises(ValueError, match="split must be"):
            ir.create_op_call("tile.aiv_shard", [tile_var], {"split": bad}, span)


def test_aiv_shard_rejects_odd_split_axis():
    """aiv_shard requires the static split-axis extent to be even."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([15, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="must be even"):
        ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)


def test_aiv_shard_rejects_partial_split_axis_valid_shape():
    """A shared boundary type cannot encode different lane-local valid extents."""
    span = ir.Span.unknown()
    tile_view = ir.TileView(
        valid_shape=[ir.ConstInt(13, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)]
    )
    tile_type = ir.TileType([16, 128], DataType.FP32, memref=None, tile_view=tile_view, memory_space=None)
    tile_var = ir.Var("t", tile_type, span)

    with pytest.raises(ValueError, match="split-axis valid extent must be provably equal"):
        ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)


def test_split_reshape_preserves_partial_non_split_axis():
    """Validity on an axis replicated identically across both lanes remains representable."""
    span = ir.Span.unknown()
    tile_view = ir.TileView(
        valid_shape=[ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(80, DataType.INDEX, span)]
    )
    tile_var = ir.Var(
        "t",
        ir.TileType([16, 128], DataType.FP32, memref=None, tile_view=tile_view, memory_space=None),
        span,
    )

    sharded = ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)
    assert isinstance(sharded.type, ir.TileType)
    assert sharded.type.shape == [8, 128]
    assert sharded.type.tile_view is not None
    assert sharded.type.tile_view.valid_shape == [8, 80]


def test_split_reshape_halves_dynamic_physical_extent_symbolically():
    """A dynamic (non-ConstInt) split-axis physical extent is reshaped to a
    symbolic floordiv (shard) / mul (gather), not left as an identity reshape."""
    span = ir.Span.unknown()
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    tile_type = ir.TileType([n, ir.ConstInt(128, DataType.INDEX, span)], DataType.FP32)
    tile_var = ir.Var("t", tile_type, span)

    sharded = ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)
    assert isinstance(sharded.type, ir.TileType)
    half0 = sharded.type.shape[0]
    assert not isinstance(half0, ir.ConstInt)  # symbolic floordiv(n, 2)
    assert half0 is not n  # transformed, not an identity passthrough

    gathered = ir.create_op_call("tile.aic_gather", [tile_var], {"split": 1}, span)
    assert isinstance(gathered.type, ir.TileType)
    assert gathered.type.shape[0] is not n  # symbolic n * 2


def test_split_reshape_rejects_unknown_dynamic_split_axis_valid_shape():
    """Unknown equality is rejected because no runtime lane-local guard is emitted."""
    span = ir.Span.unknown()
    valid_n = ir.Var("valid_n", ir.ScalarType(DataType.INDEX), span)
    tile_view = ir.TileView(valid_shape=[valid_n, ir.ConstInt(128, DataType.INDEX, span)])
    tile_type = ir.TileType([16, 128], DataType.FP32, memref=None, tile_view=tile_view, memory_space=None)
    tile_var = ir.Var("t", tile_type, span)

    with pytest.raises(ValueError, match="symbolic equality cannot be proven"):
        ir.create_op_call("tile.aiv_shard", [tile_var], {"split": 1}, span)


def test_aic_gather_rejects_partial_split_axis_valid_shape():
    """Gather cannot recover one global prefix from one shared partial-half type."""
    span = ir.Span.unknown()
    tile_view = ir.TileView(
        valid_shape=[ir.ConstInt(5, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)]
    )
    tile_var = ir.Var(
        "t",
        ir.TileType([8, 128], DataType.FP32, memref=None, tile_view=tile_view, memory_space=None),
        span,
    )

    with pytest.raises(ValueError, match="split-axis valid extent must be provably equal"):
        ir.create_op_call("tile.aic_gather", [tile_var], {"split": 1}, span)


def test_aic_gather_allows_odd_split_axis():
    """aic_gather doubles the axis, so an odd extent is fine (no even constraint)."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([15, 128], DataType.FP32), span)

    call = ir.create_op_call("tile.aic_gather", [tile_var], {"split": 1}, span)
    assert isinstance(call.type, ir.TileType)
    assert call.type.shape == [30, 128]


def test_tensor_cross_core_ops_registered():
    """The tensor-level split-axis ops are registered alongside the tile ops."""
    for name in ("tensor.aiv_shard", "tensor.aic_gather"):
        assert ir.is_op_registered(name), f"{name} should be registered"


def test_tensor_aiv_shard_halves_split_axis():
    """tensor.aiv_shard halves the split axis: UP_DOWN -> axis0, LEFT_RIGHT -> axis1."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([16, 128], DataType.FP32), span)

    up_down = ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 1}, span)
    assert isinstance(up_down.type, ir.TensorType)
    assert up_down.type.shape == [8, 128]
    assert up_down.type.dtype == DataType.FP32

    left_right = ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 2}, span)
    assert isinstance(left_right.type, ir.TensorType)
    assert left_right.type.shape == [16, 64]
    assert left_right.type.dtype == DataType.FP32


def test_tensor_aic_gather_doubles_split_axis():
    """tensor.aic_gather is the inverse of tensor.aiv_shard: it doubles the split axis."""
    span = ir.Span.unknown()

    up_down = ir.create_op_call(
        "tensor.aic_gather", [ir.Var("t", ir.TensorType([8, 128], DataType.FP32), span)], {"split": 1}, span
    )
    assert isinstance(up_down.type, ir.TensorType)
    assert up_down.type.shape == [16, 128]

    left_right = ir.create_op_call(
        "tensor.aic_gather", [ir.Var("t", ir.TensorType([16, 64], DataType.FP32), span)], {"split": 2}, span
    )
    assert isinstance(left_right.type, ir.TensorType)
    assert left_right.type.shape == [16, 128]


def test_tensor_split_reshape_rejects_non_2d_tensor():
    """tensor.aiv_shard / aic_gather require a 2D tensor (rank-2 gate, with reshape hint)."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([2, 16, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="2D tensor"):
        ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 1}, span)
    with pytest.raises(ValueError, match="Reshape the operand to 2D"):
        ir.create_op_call("tensor.aic_gather", [tensor_var], {"split": 1}, span)


def test_tensor_split_reshape_rejects_tile_type():
    """tensor.aiv_shard rejects a TileType operand — that is the tile op's domain."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([16, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.create_op_call("tensor.aiv_shard", [tile_var], {"split": 1}, span)


def test_tensor_split_reshape_rejects_distributed_tensor():
    """tensor.aiv_shard rejects a DistributedTensorType — distributed is out of scope."""
    span = ir.Span.unknown()
    dist_var = ir.Var("t", ir.DistributedTensorType([16, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="non-distributed"):
        ir.create_op_call("tensor.aiv_shard", [dist_var], {"split": 1}, span)


def test_tensor_split_reshape_rejects_bad_split_attr():
    """split must be 1 or 2 (0 and out-of-range rejected)."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([16, 128], DataType.FP32), span)

    for bad in (0, 3):
        with pytest.raises(ValueError, match="split must be"):
            ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": bad}, span)


def test_tensor_aiv_shard_rejects_odd_split_axis():
    """tensor.aiv_shard requires the static split-axis extent to be even."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([15, 128], DataType.FP32), span)

    with pytest.raises(ValueError, match="must be even"):
        ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 1}, span)


def test_tensor_aiv_shard_rejects_partial_split_axis_valid_shape():
    """Tensor and tile split boundaries share the same fail-closed contract."""
    span = ir.Span.unknown()
    tensor_view = ir.TensorView(
        [],
        ir.TensorLayout.ND,
        [ir.ConstInt(13, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)],
    )
    tensor_type = ir.TensorType([16, 128], DataType.FP32, None, tensor_view)
    tensor_var = ir.Var("t", tensor_type, span)

    with pytest.raises(ValueError, match="split-axis valid extent must be provably equal"):
        ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 1}, span)


def test_tensor_aic_gather_rejects_partial_split_axis_valid_shape():
    """Tensor gather also rejects an unrepresentable partial half."""
    span = ir.Span.unknown()
    tensor_view = ir.TensorView(
        [],
        ir.TensorLayout.ND,
        [ir.ConstInt(5, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)],
    )
    tensor_var = ir.Var("t", ir.TensorType([8, 128], DataType.FP32, None, tensor_view), span)

    with pytest.raises(ValueError, match="split-axis valid extent must be provably equal"):
        ir.create_op_call("tensor.aic_gather", [tensor_var], {"split": 1}, span)


def test_tensor_split_reshape_halves_dynamic_extent_symbolically():
    """A dynamic split-axis physical extent is reshaped to a symbolic floordiv (shard)
    / mul (gather), matching the tile deducer."""
    span = ir.Span.unknown()
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    tensor_type = ir.TensorType([n, ir.ConstInt(128, DataType.INDEX, span)], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    sharded = ir.create_op_call("tensor.aiv_shard", [tensor_var], {"split": 1}, span)
    assert isinstance(sharded.type, ir.TensorType)
    half0 = sharded.type.shape[0]
    assert not isinstance(half0, ir.ConstInt)  # symbolic floordiv(n, 2)
    assert half0 is not n  # transformed, not an identity passthrough

    gathered = ir.create_op_call("tensor.aic_gather", [tensor_var], {"split": 1}, span)
    assert isinstance(gathered.type, ir.TensorType)
    assert gathered.type.shape[0] is not n  # symbolic n * 2


def test_dsl_aiv_shard_aic_gather_require_split_aiv_scope():
    """The eager pl.aiv_shard / pl.aic_gather DSL wrappers raise: the split mode is
    inherited from the enclosing pl.split_aiv scope, which only exists during a
    @pl.program parse, so they cannot be constructed eagerly."""
    span = ir.Span.unknown()
    tile = pl.Tile(expr=ir.Var("t", ir.TileType([16, 128], DataType.FP32), span))

    with pytest.raises(RuntimeError, match="pl.split_aiv"):
        pl.aiv_shard(tile)

    with pytest.raises(RuntimeError, match="pl.split_aiv"):
        pl.aic_gather(tile)


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
