# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for the operator registration system.

Tests cover:
- TileType construction and validation
- TensorAdd and BlockAdd operations
- Type deduction for various input combinations
- Broadcasting behavior
- Dynamic dimension handling
- Error cases
"""

import re
import sys
from pathlib import Path

import pypto
import pytest
from pypto import DataType, ir
from pypto.pypto_core import testing


def test_dynamic_dimension_constant():
    """Test dynamic dimension constant."""
    # Check that DYNAMIC_DIM is -1
    assert ir.DYNAMIC_DIM == -1

    # Can be used in dimension expressions
    span = ir.Span.unknown()
    dynamic_dim = ir.ConstInt(ir.DYNAMIC_DIM, DataType.INT32, span)
    assert dynamic_dim.value == -1


def test_tensor_add_same_shape():
    """Test TensorAdd with identical shapes."""
    span = ir.Span.unknown()

    # Create shape [4, 8]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim4, dim8]

    # Create two tensor variables with same shape
    tensor_type = ir.TensorType(shape, DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_add_broadcasting():
    """Test TensorAdd with broadcasting."""
    span = ir.Span.unknown()

    # Tensor A: [4, 8]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape_a = [dim4, dim8]
    type_a = ir.TensorType(shape_a, DataType.FP32)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8] (should broadcast to [4, 8])
    shape_b = [dim8]
    type_b = ir.TensorType(shape_b, DataType.FP32)
    var_b = ir.Var("b", type_b, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type - should be [4, 8]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert len(result_type.shape) == 2


def test_tensor_add_broadcasting_with_one():
    """Test TensorAdd broadcasting with dimension of size 1."""
    span = ir.Span.unknown()

    # Tensor A: [4, 1]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape_a = [dim4, dim1]
    type_a = ir.TensorType(shape_a, DataType.FP32)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8]
    shape_b = [dim8]
    type_b = ir.TensorType(shape_b, DataType.FP32)
    var_b = ir.Var("b", type_b, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type - should be [4, 8]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert len(result_type.shape) == 2


def test_tensor_add_type_promotion():
    """Test TensorAdd with different data types."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim8]

    # INT32 + FP32 should promote to FP32
    type_int = ir.TensorType(shape, DataType.INT32)
    type_float = ir.TensorType(shape, DataType.FP32)
    var_int = ir.Var("a", type_int, span)
    var_float = ir.Var("b", type_float, span)

    call = ir.create_op_call("tensor.add", [var_int, var_float], span)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_add_wrong_arg_count():
    """Test TensorAdd with wrong number of arguments."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)

    # Too few arguments
    with pytest.raises(ValueError):
        ir.create_op_call("tensor.add", [var_a], span)

    # Too many arguments
    var_b = ir.Var("b", tensor_type, span)
    var_c = ir.Var("c", tensor_type, span)
    with pytest.raises(ValueError):
        ir.create_op_call("tensor.add", [var_a, var_b, var_c], span)


def test_tensor_add_wrong_type():
    """Test TensorAdd with non-tensor arguments."""
    span = ir.Span.unknown()

    # Scalar type instead of tensor
    scalar_type = ir.ScalarType(DataType.FP32)
    var_scalar = ir.Var("s", scalar_type, span)

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_tensor = ir.Var("t", tensor_type, span)

    with pytest.raises(ValueError):
        ir.create_op_call("tensor.add", [var_scalar, var_tensor], span)


def test_operator_registration_status():
    """Test operator registration queries."""
    # Check that our operators are registered
    assert ir.is_op_registered("tensor.add")
    assert ir.is_op_registered("tensor.sub")
    assert ir.is_op_registered("tensor.mul")
    assert ir.is_op_registered("tensor.div")

    # Check that a non-existent operator is not registered
    assert not ir.is_op_registered("nonexistent.op")


def test_get_op():
    """Test getting operator instances."""
    tensor_add_op = ir.get_op("tensor.add")
    assert tensor_add_op.name == "tensor.add"

    # Non-existent operator should raise exception
    with pytest.raises(ValueError):
        ir.get_op("nonexistent.op")


def test_test_op_kwarg_schema():
    """Test that test.op has kwarg schema defined."""
    test_op = ir.get_op("test.op")

    # Check kwarg keys exist in schema
    assert test_op.has_attr("int_attr")
    assert test_op.has_attr("string_attr")
    assert test_op.has_attr("bool_attr")


def test_test_op_all_kwarg_keys():
    """Test all kwarg keys of test.op."""
    test_op = ir.get_op("test.op")

    # Get all kwarg keys from schema
    keys = test_op.get_attr_keys()

    # Check all expected kwargs are present
    assert "int_attr" in keys
    assert "string_attr" in keys
    assert "bool_attr" in keys

    # Verify we have exactly 3 kwargs
    assert len(keys) == 3


def test_test_op_nonexistent_kwarg():
    """Test checking non-existent kwargs."""
    test_op = ir.get_op("test.op")

    # Check that non-existent kwarg is not in schema
    assert not test_op.has_attr("nonexistent")
    assert not test_op.has_attr("device")
    assert not test_op.has_attr("priority")


def test_test_op_kwarg_isolation():
    """Test that test.op kwarg schema is isolated from other operators."""
    test_op = ir.get_op("test.op")
    tensor_add_op = ir.get_op("tensor.add")

    # test.op should have int_attr, string_attr, bool_attr in schema
    assert test_op.has_attr("int_attr")
    assert test_op.has_attr("string_attr")
    assert test_op.has_attr("bool_attr")

    # tensor.add should NOT have these in its schema
    assert not tensor_add_op.has_attr("int_attr")
    assert not tensor_add_op.has_attr("string_attr")
    assert not tensor_add_op.has_attr("bool_attr")


def test_tensor_sub_mul_div():
    """Test other tensor operations (sub, mul, div)."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim8]
    tensor_type = ir.TensorType(shape, DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Test sub
    call_sub = ir.create_op_call("tensor.sub", [var_a, var_b], span)
    assert isinstance(call_sub.type, ir.TensorType)

    # Test mul
    call_mul = ir.create_op_call("tensor.mul", [var_a, var_b], span)
    assert isinstance(call_mul.type, ir.TensorType)

    # Test div
    call_div = ir.create_op_call("tensor.div", [var_a, var_b], span)
    assert isinstance(call_div.type, ir.TensorType)


def test_call_with_explicit_type():
    """Test Call constructor with explicit type parameter."""
    span = ir.Span.unknown()

    # Create a simple operation
    op = ir.get_op("tensor.add")

    # Create arguments
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Create call with explicit type
    result_type = ir.TensorType([dim8], DataType.FP32)
    call = ir.Call(op, [var_a, var_b], {}, result_type, span)

    # Verify type is set correctly
    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


def test_matmul_with_valid_kwargs():
    """Test tensor.matmul with valid kwargs."""
    span = ir.Span.unknown()

    # Create two matrices
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    type_a = ir.TensorType([dim64, dim128], DataType.FP16)
    type_b = ir.TensorType([dim128, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Test with DataType kwarg (passed directly)
    kwargs = {"out_dtype": DataType.FP32, "a_trans": False, "b_trans": False}
    call = ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_matmul_with_transpose_kwargs():
    """Test tensor.matmul with transpose kwargs."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    type_a = ir.TensorType([dim128, dim64], DataType.FP16)  # Will be transposed
    type_b = ir.TensorType([dim128, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Test with a_trans=True
    kwargs = {"a_trans": True, "b_trans": False}
    call = ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Should work without error
    assert isinstance(call.type, ir.TensorType)


def test_tile_batch_matmul_type_deduction():
    """Test tile.batch_matmul type deduction without transpose kwargs."""
    span = ir.Span.unknown()

    dim2 = ir.ConstInt(2, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    type_a = ir.TileType([dim2, dim128, dim64], DataType.FP16)
    type_b = ir.TileType([dim2, dim64, dim32], DataType.FP16)
    var_a = ir.Var("a_tile", type_a, span)
    var_b = ir.Var("b_tile", type_b, span)

    call = ir.create_op_call("tile.batch_matmul", [var_a, var_b], span)

    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3
    assert isinstance(result_type.shape[0], ir.ConstInt)
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert isinstance(result_type.shape[2], ir.ConstInt)
    assert result_type.shape[0].value == 2
    assert result_type.shape[1].value == 128
    assert result_type.shape[2].value == 32


def test_matmul_with_unknown_kwarg():
    """Test tensor.matmul with unknown kwarg should raise error."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    # Unknown kwarg should raise ValueError
    kwargs = {"unknown_param": 123, "a_trans": False}

    with pytest.raises(ValueError) as exc_info:
        ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check error message contains "unknown"
    assert "unknown" in str(exc_info.value).lower() or "Unknown" in str(exc_info.value)


def test_matmul_with_wrong_type_kwarg():
    """Test tensor.matmul with wrong type kwarg should raise error."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    # Wrong type for bool kwarg (passing string instead of bool)
    kwargs = {
        "a_trans": "true"  # Should be bool, not string
    }

    with pytest.raises(TypeError) as exc_info:
        ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check error message indicates type mismatch
    error_msg = str(exc_info.value).lower()
    assert "type" in error_msg or "incompatible" in error_msg


def test_cast_with_datatype_kwarg():
    """Test tensor.cast with DataType kwarg."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    type_fp16 = ir.TensorType([dim8], DataType.FP16)
    var_a = ir.Var("a", type_fp16, span)

    # Cast from FP16 to FP32. `mode` is a declared attr codegen reads unconditionally,
    # so tensor.cast requires it alongside target_type (2 == round, the DSL default).
    kwargs = {"target_type": DataType.FP32, "mode": 2}
    call = ir.create_op_call("tensor.cast", [var_a], kwargs, span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_reduction_with_kwargs():
    """Test tensor reduction operations with kwargs."""
    span = ir.Span.unknown()

    # Create a 2D tensor
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)

    # Test row_max with axis and keep_dim kwargs
    kwargs = {"axis": -1, "keep_dim": True}
    call = ir.create_op_call("tensor.row_max", [var_a], kwargs, span)

    # Should work without error
    assert isinstance(call.type, ir.TensorType)


def test_matmul_kwarg_schema():
    """Test that tensor.matmul has correct kwarg schema."""
    matmul_op = ir.get_op("tensor.matmul")

    # Check that expected kwargs are in schema
    assert matmul_op.has_attr("out_dtype")
    assert matmul_op.has_attr("a_trans")
    assert matmul_op.has_attr("b_trans")
    assert matmul_op.has_attr("c_matrix_nz")

    # Get all kwarg keys
    keys = matmul_op.get_attr_keys()
    assert "out_dtype" in keys
    assert "a_trans" in keys
    assert "b_trans" in keys


def test_tile_batch_matmul_kwarg_schema():
    """Test that tile.batch_matmul does not add custom kwargs."""
    batch_matmul_op = ir.get_op("tile.batch_matmul")

    keys = batch_matmul_op.get_attr_keys()
    assert not keys


def test_cast_kwarg_schema():
    """Test that tensor.cast has correct kwarg schema."""
    cast_op = ir.get_op("tensor.cast")

    # Check that expected kwargs are in schema
    assert cast_op.has_attr("target_type")
    assert cast_op.has_attr("mode")


def test_reduction_kwarg_schema():
    """Test that tensor reduction ops have correct kwarg schema."""
    row_max_op = ir.get_op("tensor.row_max")
    row_sum_op = ir.get_op("tensor.row_sum")

    # Check that expected kwargs are in schema
    assert row_max_op.has_attr("axis")
    assert row_max_op.has_attr("keep_dim")
    assert row_sum_op.has_attr("axis")
    assert row_sum_op.has_attr("keep_dim")


def test_fillpad_kwarg_schema():
    """Test that fillpad ops declare pad_value in their kwarg schemas."""
    tensor_fillpad_op = ir.get_op("tensor.fillpad")
    tile_fillpad_op = ir.get_op("tile.fillpad")
    tile_fillpad_inplace_op = ir.get_op("tile.fillpad_inplace")

    assert tensor_fillpad_op.has_attr("pad_value")
    assert tile_fillpad_op.has_attr("pad_value")
    assert tile_fillpad_inplace_op.has_attr("pad_value")


def test_tile_slice_pad_value_kwarg_schema():
    """Test that tile.slice declares pad_value in its kwarg schema."""
    tile_slice_op = ir.get_op("tile.slice")
    assert tile_slice_op.has_attr("pad_value")


def test_tensor_slice_pad_value_kwarg_schema():
    """Test that tensor.slice declares pad_value in its kwarg schema."""
    tensor_slice_op = ir.get_op("tensor.slice")
    assert tensor_slice_op.has_attr("pad_value")


class TestOpMemorySpecRegistry:
    """Test that op memory specs are correctly registered and queryable."""

    def test_matmul_spec(self):
        """tile.matmul has Left/Right input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.matmul")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.Right]

    def test_matmul_acc_spec(self):
        """tile.matmul_acc has Acc/Left/Right input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.matmul_acc")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 3
        assert constraints[0] == [ir.MemorySpace.Acc]
        assert constraints[1] == [ir.MemorySpace.Left]
        assert constraints[2] == [ir.MemorySpace.Right]

    def test_load_spec(self):
        """tile.load output is retargetable: resolves from target_memory kwarg if
        present, otherwise deferred for InferTileMemorySpace to decide from
        consumer demand."""
        spec = ir.get_op_memory_spec("tile.load")
        assert spec is not None
        assert spec["output_memory"] == "deferred"
        assert spec["input_constraints"] == []

    def test_store_spec(self):
        """tile.store input 0 accepts Vec or Acc."""
        spec = ir.get_op_memory_spec("tile.store")
        assert spec is not None
        constraints = spec["input_constraints"]
        assert len(constraints) == 1
        assert set(constraints[0]) == {ir.MemorySpace.Vec, ir.MemorySpace.Acc}

    @pytest.mark.parametrize(
        "op_name",
        [
            "tile.reshape",
            "tile.slice",
            "tile.transpose",
            "tile.assemble",
            "tile.set_validshape",
        ],
    )
    def test_view_ops_inherit_from_input(self, op_name):
        """View/transform ops inherit output memory from input."""
        spec = ir.get_op_memory_spec(op_name)
        assert spec is not None
        assert spec["output_memory"] == "inherit_from_input"

    def test_matmul_bias_spec(self):
        """tile.matmul_bias has Left/Right/Bias input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.matmul_bias")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 3
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.Right]
        assert constraints[2] == [ir.MemorySpace.Bias]

    def test_gemv_spec(self):
        """tile.gemv has Left/Right input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.gemv")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.Right]

    def test_gemv_acc_spec(self):
        """tile.gemv_acc has Acc/Left/Right input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.gemv_acc")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 3
        assert constraints[0] == [ir.MemorySpace.Acc]
        assert constraints[1] == [ir.MemorySpace.Left]
        assert constraints[2] == [ir.MemorySpace.Right]

    def test_gemv_bias_spec(self):
        """tile.gemv_bias has Left/Right/Bias input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.gemv_bias")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 3
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.Right]
        assert constraints[2] == [ir.MemorySpace.Bias]

    def test_elementwise_vec_spec(self):
        """Elementwise ops (tile.add) have Vec input/output memory spec."""
        spec = ir.get_op_memory_spec("tile.add")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Vec]
        assert constraints[1] == [ir.MemorySpace.Vec]

    def test_unary_vec_spec(self):
        """Unary ops (tile.neg) have Vec input/output memory spec."""
        spec = ir.get_op_memory_spec("tile.neg")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 1
        assert constraints[0] == [ir.MemorySpace.Vec]

    def test_tile_scalar_vec_spec(self):
        """Tile-scalar ops (tile.adds) constrain only the tile input."""
        spec = ir.get_op_memory_spec("tile.adds")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 1
        assert constraints[0] == [ir.MemorySpace.Vec]

    def test_reduction_with_tmp_spec(self):
        """Reduction ops with tmp_tile (tile.row_sum) constrain both tile inputs."""
        spec = ir.get_op_memory_spec("tile.row_sum")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Vec]
        assert constraints[1] == [ir.MemorySpace.Vec]

    def test_broadcast_binary_vec_spec(self):
        """tile.row_expand_add constrains its two sources and optional tmp to Vec."""
        spec = ir.get_op_memory_spec("tile.row_expand_add")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 3
        assert constraints[0] == [ir.MemorySpace.Vec]
        assert constraints[1] == [ir.MemorySpace.Vec]
        assert constraints[2] == [ir.MemorySpace.Vec]

    def test_full_vec_spec(self):
        """tile.full creates tiles in Vec (no tile inputs)."""
        spec = ir.get_op_memory_spec("tile.full")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        assert spec["input_constraints"] == []

    def test_unregistered_op_returns_none(self):
        """Unregistered op returns None."""
        spec = ir.get_op_memory_spec("nonexistent.op")
        assert spec is None

    def test_tensor_op_has_no_memory_spec(self):
        """tensor-level ops (tensor.add) have no memory spec."""
        spec = ir.get_op_memory_spec("tensor.add")
        assert spec is None

    def test_scalar_op_has_no_memory_spec(self):
        """scalar-level ops (scalar.add) have no memory spec."""
        spec = ir.get_op_memory_spec("scalar.add")
        assert spec is None

    def test_batch_matmul_spec(self):
        """tile.batch_matmul has Left/Right input constraints and Acc output."""
        spec = ir.get_op_memory_spec("tile.batch_matmul")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.Right]

    def test_move_spec(self):
        """tile.move output is from kwarg, defaults to Vec."""
        spec = ir.get_op_memory_spec("tile.move")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec

    def test_create_spec(self):
        """tile.create output is retargetable: resolves from target_memory kwarg
        if present, otherwise deferred for InferTileMemorySpace to decide."""
        spec = ir.get_op_memory_spec("tile.create")
        assert spec is not None
        assert spec["output_memory"] == "deferred"


class TestRegistryInfrastructure:
    """Test the op memory spec registry infrastructure (dict structure, types, completeness)."""

    def test_spec_dict_keys(self):
        """All specs have exactly 'input_constraints' and 'output_memory' keys."""
        spec = ir.get_op_memory_spec("tile.matmul")
        assert spec is not None
        assert set(spec.keys()) == {"input_constraints", "output_memory"}

    def test_fixed_output_returns_enum(self):
        """Fixed output memory (Acc) returns a MemorySpace enum."""
        spec = ir.get_op_memory_spec("tile.matmul")
        assert spec is not None
        assert isinstance(spec["output_memory"], ir.MemorySpace)

    def test_kwarg_output_returns_deferred(self):
        """Retargetable ops (tile.load/tile.create) report 'deferred' when the
        target_memory kwarg is absent — InferTileMemorySpace resolves from
        consumer demand."""
        spec = ir.get_op_memory_spec("tile.load")
        assert spec is not None
        assert spec["output_memory"] == "deferred"

    def test_inherit_output_returns_string(self):
        """Inherit-from-input output returns the string 'inherit_from_input'."""
        spec = ir.get_op_memory_spec("tile.reshape")
        assert spec is not None
        assert isinstance(spec["output_memory"], str)
        assert spec["output_memory"] == "inherit_from_input"

    def test_constraints_are_lists_of_enums(self):
        """Each input constraint is a list of MemorySpace enums."""
        spec = ir.get_op_memory_spec("tile.matmul")
        assert spec is not None
        for i, constraint in enumerate(spec["input_constraints"]):
            assert isinstance(constraint, list), f"constraint {i} not a list"
            for ms in constraint:
                assert isinstance(ms, ir.MemorySpace), f"constraint {i} has non-enum"

    @pytest.mark.parametrize(
        "op_name",
        [
            "tile.add",
            "tile.sub",
            "tile.mul",
            "tile.div",
            "tile.neg",
            "tile.exp",
            "tile.recip",
            "tile.sqrt",
            "tile.row_sum",
            "tile.row_max",
            "tile.row_min",
            "tile.row_expand",
            "tile.col_expand",
            "tile.cmp",
            "tile.sel",
            "tile.fillpad",
            "tile.fillpad_inplace",
            "tile.cast",
            "tile.abs",
            "tile.relu",
        ],
    )
    def test_vec_ops_have_vec_output(self, op_name):
        """All Vec tile ops have Vec output memory spec."""
        spec = ir.get_op_memory_spec(op_name)
        assert spec is not None, f"{op_name} missing memory spec"
        assert spec["output_memory"] == ir.MemorySpace.Vec

    @pytest.mark.parametrize(
        "op_name",
        [
            "tile.matmul",
            "tile.matmul_acc",
            "tile.matmul_bias",
            "tile.gemv",
            "tile.gemv_acc",
            "tile.gemv_bias",
            "tile.batch_matmul",
            "tile.load",
            "tile.store",
            "tile.move",
            "tile.create",
            "tile.slice",
            "tile.reshape",
            "tile.transpose",
            "tile.assemble",
            "tile.set_validshape",
            "tile.add",
            "tile.sub",
            "tile.mul",
            "tile.div",
            "tile.neg",
            "tile.exp",
            "tile.recip",
            "tile.sqrt",
            "tile.tquant_mx",
            "tile.row_sum",
            "tile.row_max",
            "tile.row_min",
            "tile.row_expand",
            "tile.col_expand",
            "tile.full",
            "tile.write",
        ],
    )
    def test_all_tile_ops_have_spec(self, op_name):
        """Every standard tile op has a memory spec (completeness check)."""
        spec = ir.get_op_memory_spec(op_name)
        assert spec is not None, f"{op_name} missing memory spec"
        assert "input_constraints" in spec

    @pytest.mark.parametrize(
        "op_name",
        [
            "tile.get_block_idx",
            "tile.alloc",
        ],
    )
    def test_non_tile_output_ops_have_no_spec(self, op_name):
        """Tile ops that don't produce TileType use no_memory_spec() and return None."""
        spec = ir.get_op_memory_spec(op_name)
        assert spec is None

    def test_import_validates_tile_ops(self):
        """Importing pypto succeeds — ValidateTileOps() passed at import time."""
        # If we got here, import succeeded, meaning all tile.* ops either have
        # a memory spec or explicitly opted out via no_memory_spec().
        # Verify at least one tile op exists as a sanity check.
        assert ir.is_op_registered("tile.matmul")


class TestDeclaredCoreAffinity:
    """`set_core_affinity(...)` declarations, read back through the registry."""

    @pytest.mark.parametrize(
        "op_name",
        [
            "pld.tile.put",
            "pld.tile.get",
            "pld.tensor.put",
            "pld.tensor.get",
        ],
    )
    def test_put_get_family_is_vector(self, op_name):
        """The TPUT/TGET family is vector-only: it bounces through a VEC tile.

        pto-isa streams GM -> UB -> remote GM, and ptoas enforces the staging
        tile's address space (``verifyCommStagingTileLike`` requires VEC). The
        tile-level forms would classify VECTOR incidentally via their staging
        tile argument; the tensor-level forms have no tile operand at all, so
        without the declaration they would classify SHARED and be duplicated
        onto the cube lane of a mixed kernel.
        """
        assert testing.get_declared_core_affinity(op_name) == "vector"

    @pytest.mark.parametrize(
        "op_name",
        [
            "pld.system.notify",
            "pld.system.wait",
        ],
    )
    def test_notify_wait_are_core_agnostic(self, op_name):
        """TNOTIFY / TWAIT run on either core, so they declare no affinity.

        Their pto-isa implementations are pure scalar/GM (st_atomic, dcci, dsb)
        and ptoas imposes no core or section constraint, so declaring VECTOR
        here would be a false claim about the ISA.
        """
        assert testing.get_declared_core_affinity(op_name) is None

    def test_declared_affinity_rejects_unknown_op(self):
        with pytest.raises(ValueError):
            testing.get_declared_core_affinity("pld.tile.not_an_op")


class TestNoDuplicateOps:
    """`set_no_duplicate()` declarations, read back through the registry."""

    def test_notify_must_not_run_on_a_second_core(self):
        """A notify copied onto the cube lane can release the peer too early.

        The hazard is premature release from the wrong lane, not
        non-idempotence: a copy on the AIC lane can publish the signal before
        the AIV lane's TPUT has landed the data that signal covers, so the peer
        reads stale bytes. That applies to BOTH ``NotifyOp`` forms, which is why
        the flag is unconditional rather than keyed on the ``op`` kwarg.
        """
        assert testing.is_no_duplicate_op("pld.system.notify") is True

    @pytest.mark.parametrize(
        "op_name",
        [
            # TWAIT's presence on the cube lane is load-bearing: pinning it to
            # AIV would let the matmul race past the peer data it blocks on.
            "pld.system.wait",
            # Placement, not duplication, keeps the put/get family off the cube
            # lane: they declare VECTOR affinity.
            "pld.tile.put",
            "pld.tile.get",
            # Ordinary compute is safe to run on either lane.
            "tile.matmul",
            "tile.add",
        ],
    )
    def test_ops_safe_to_run_on_a_second_core(self, op_name):
        assert testing.is_no_duplicate_op(op_name) is False

    def test_no_duplicate_rejects_unknown_op(self):
        with pytest.raises(ValueError):
            testing.is_no_duplicate_op("pld.system.not_an_op")


class TestDeduceTypeExceptionPassthrough:
    """`OpRegistry::CreateImpl` must not flatten deduction failures to ValueError.

    Every op Call is built through that one funnel. It appends the IR span to the
    message, and used to do so by catching `const std::exception&` and rethrowing a
    fresh `ValueError` - which collapsed InternalError / TypeError / IndexError into
    one class and replaced the stack trace of the real throw site with its own.
    """

    @staticmethod
    def _arg():
        return ir.ConstInt(1, DataType.INT32, ir.Span.unknown())

    def test_internal_error_is_not_flattened(self):
        """An INTERNAL_CHECK inside type deduction stays an InternalError."""
        with pytest.raises(pypto.InternalError):
            ir.create_op_call("test.deduce_raises_internal", [self._arg()], ir.Span.unknown())

    def test_internal_error_is_not_a_value_error(self):
        """Guards the specific regression: InternalError must not be catchable as ValueError."""
        with pytest.raises(pypto.InternalError) as exc_info:
            ir.create_op_call("test.deduce_raises_internal", [self._arg()], ir.Span.unknown())
        assert not isinstance(exc_info.value, ValueError)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="libbacktrace cannot symbolize an MH_BUNDLE on macOS"
    )
    def test_internal_error_keeps_original_throw_site(self):
        """The trace still reaches the deduction function that actually threw.

        Constructing a fresh exception in the registry's catch block would root the trace
        at the catch instead, losing every frame below it - so the presence of the
        deduction site is what distinguishes a preserved trace from a rebuilt one.
        (`op_registry.cpp` appears either way: it is a genuine caller frame.)
        """
        with pytest.raises(pypto.InternalError) as exc_info:
            ir.create_op_call("test.deduce_raises_internal", [self._arg()], ir.Span.unknown())
        assert "src/ir/op/testing.cpp" in str(exc_info.value)

    def test_span_is_still_appended(self):
        """Preserving the type must not cost the IR location the funnel adds."""
        span = ir.Span("kernel.py", 12, 15)
        with pytest.raises(pypto.InternalError) as exc_info:
            ir.create_op_call("test.deduce_raises_internal", [self._arg()], span)
        assert "kernel.py:12:15" in str(exc_info.value)

    def test_unknown_span_appends_no_location(self):
        """An unknown span adds nothing - not a rendered placeholder.

        `Span.unknown()` stringifies to ":-1:-1", so a merely-substring assertion would
        stay green if the guard in LocationSuffix were dropped. Comparing the two messages
        for exact equality pins the behaviour in both directions instead: the located one
        must differ by the suffix and nothing else. Uses the TypeError op because it is
        user-class, so neither message carries a traceback that would need filtering.
        """
        with pytest.raises(TypeError) as unknown_info:
            ir.create_op_call("test.deduce_raises_type", [self._arg()], ir.Span.unknown())
        with pytest.raises(TypeError) as located_info:
            ir.create_op_call("test.deduce_raises_type", [self._arg()], ir.Span("kernel.py", 12, 15))

        unknown_message = str(unknown_info.value)
        assert unknown_message == "test.deduce_raises_type always fails"
        assert str(located_info.value) == f"{unknown_message} at kernel.py:12:15"

    def test_type_error_is_not_flattened(self):
        """The user-error half of the contract: TypeError stays a TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ir.create_op_call("test.deduce_raises_type", [self._arg()], ir.Span.unknown())
        assert not isinstance(exc_info.value, ValueError)

    def test_value_error_still_surfaces_as_value_error(self):
        """The common case is unchanged: a CHECK in deduction stays a ValueError."""
        with pytest.raises(ValueError):
            ir.create_op_call("tile.cast", [self._arg()], ir.Span.unknown())


class TestArgEffects:
    """Per-argument read/write effects declared on the operator registry.

    Every direction and dependency analysis needs one answer to "does this call
    write the buffer this argument names". These tests pin that answer at its
    source, so a new operator cannot quietly join the set of writers nobody
    models — which is how a written parameter keeps direction ``In``, loses its
    RAW edge, and deadlocks or races on device.
    """

    def test_unnamed_argument_defaults_to_read(self):
        """The tile a store copies *from* is read, not written."""
        assert ir.get_op_arg_effect("tile.store", 0) == ir.ArgEffect.Read

    def test_index_past_the_argument_list_is_read(self):
        assert ir.get_op_arg_effect("tile.store", 99) == ir.ArgEffect.Read

    def test_functional_op_is_unclassified(self):
        """`tensor.add` writes through no argument and was never classified;
        `False` here is what lets an analysis tell that apart from a declared
        read-only operator."""
        assert ir.op_has_declared_arg_effects("tensor.add") is False
        assert ir.get_op_arg_effect("tensor.add", 0) == ir.ArgEffect.Read

    def test_declared_read_only_op_is_classified(self):
        """`pld.system.wait` polls a signal it never writes — classified, but
        with no write. That is a decision on record, not an omission."""
        assert ir.op_has_declared_arg_effects("pld.system.wait") is True
        assert ir.get_op_arg_effect("pld.system.wait", 0) == ir.ArgEffect.Read

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            ir.get_op_arg_effect("tile.not_an_op", 0)

    @pytest.mark.parametrize(
        ("op_name", "arg_index", "expected"),
        [
            # A store overwrites the region it lands on; the untouched remainder
            # is neither loaded nor re-stored, so nothing moves into the kernel.
            ("tile.store", 2, ir.ArgEffect.Write),
            # Same contract, and the one that was missing: a GM tensor written
            # only by a scatter used to read as a pure input.
            ("tile.mscatter", 2, ir.ArgEffect.Write),
            ("tensor.assemble", 0, ir.ArgEffect.Write),
            ("tensor.write", 0, ir.ArgEffect.Write),
            # Cross-rank pushes and pulls land in their destination operand.
            ("pld.tile.remote_store", 1, ir.ArgEffect.Write),
            ("pld.tensor.remote_store", 1, ir.ArgEffect.Write),
            ("pld.tile.put", 0, ir.ArgEffect.Write),
            ("pld.tile.get", 0, ir.ArgEffect.Write),
            # Accumulators read the running sum they add to.
            ("tile.matmul_acc", 0, ir.ArgEffect.ReadWrite),
            ("tile.gemv_acc", 0, ir.ArgEffect.ReadWrite),
            # Destination-passing style: the positions the op does not rewrite
            # pass through to the result, so the prior content is read.
            ("tile.scatter", 0, ir.ArgEffect.ReadWrite),
            ("tile.scatter_update", 0, ir.ArgEffect.ReadWrite),
            ("array.update_element", 0, ir.ArgEffect.ReadWrite),
            ("tile.write", 0, ir.ArgEffect.ReadWrite),
            ("tile.assemble", 0, ir.ArgEffect.ReadWrite),
            # Composite collectives update their window and signal in place.
            ("pld.tensor.allreduce", 0, ir.ArgEffect.ReadWrite),
            ("pld.tensor.allreduce", 1, ir.ArgEffect.ReadWrite),
            # A gather/exchange destination is overwritten, not updated: the
            # lowering only pushes into it and never loads from it. `recv_counts`
            # is deposited with NotifyOp::Set, so it is not an accumulate either.
            ("pld.tensor.allgather", 0, ir.ArgEffect.Read),
            ("pld.tensor.allgather", 1, ir.ArgEffect.Write),
            ("pld.tensor.allgather", 2, ir.ArgEffect.ReadWrite),
            ("pld.tensor.all_to_all", 1, ir.ArgEffect.Write),
            ("pld.tensor.all_to_all_v", 1, ir.ArgEffect.Write),
            ("pld.tensor.all_to_all_v", 3, ir.ArgEffect.Read),
            ("pld.tensor.all_to_all_v", 4, ir.ArgEffect.Write),
            # A reduce destination *is* read — its lowering loads the running
            # value back — so the distinction is per operator, not per family.
            ("pld.tensor.allreduce", 0, ir.ArgEffect.ReadWrite),
            ("pld.tensor.reduce_scatter", 0, ir.ArgEffect.ReadWrite),
        ],
    )
    def test_declared_effects(self, op_name, arg_index, expected):
        assert ir.get_op_arg_effect(op_name, arg_index) == expected

    def test_atomic_store_reads_its_destination(self):
        """`out += x` is not an overwrite: the accumulate reads the slot first.
        Declaring it `Write` would let the runtime skip staging the buffer, and
        the sum would start from allocator garbage."""
        plain = ir.get_op_arg_effect("tile.store", 2)
        atomic = ir.get_op_arg_effect("tile.store", 2, atomic=int(ir.AtomicType.Add))
        assert plain == ir.ArgEffect.Write
        assert atomic == ir.ArgEffect.ReadWrite

    def test_atomic_assemble_reads_its_destination(self):
        assert ir.get_op_arg_effect("tensor.assemble", 0) == ir.ArgEffect.Write
        assert (
            ir.get_op_arg_effect("tensor.assemble", 0, atomic=int(ir.AtomicType.Add))
            == ir.ArgEffect.ReadWrite
        )

    def test_notify_defaults_to_accumulating(self):
        """`pld.system.notify`'s `op` kwarg defaults to atomic-add, so an
        unannotated notify reads the slot it adds into; only the set form is a
        pure overwrite."""
        assert ir.get_op_arg_effect("pld.system.notify", 0) == ir.ArgEffect.ReadWrite
        assert ir.get_op_arg_effect("pld.system.notify", 0, op=int(ir.NotifyOp.Set)) == ir.ArgEffect.Write

    def test_mgather_scratch_only_in_mat_elem_mode(self):
        """`tile.mgather`'s argument 2 is a written GM scratch tensor only when
        the gather stages through one.

        `DeduceTileMgatherType` puts `scratch` at that position for Mat *elem*
        mode; Mat row mode holds `valid_shape` there and Vec mode has no third
        operand at all. Declaring the write unconditionally would claim a tuple
        operand is a written buffer, and could promote a read-only parameter to
        an output.
        """
        mat = ir.MemorySpace.Mat
        # `MgatherCoalesceMode` (include/pypto/ir/comm.h) is not bound to Python;
        # the DSL passes the same ints, and the op deducer validates the range.
        elem, row = 1, 0
        assert ir.get_op_arg_effect("tile.mgather", 2, target_memory=mat, coalesce=elem) == (
            ir.ArgEffect.Write
        )
        assert ir.get_op_arg_effect("tile.mgather", 2, target_memory=mat, coalesce=row) == (ir.ArgEffect.Read)
        # Vec is the default output space and carries no third operand.
        assert ir.get_op_arg_effect("tile.mgather", 2) == ir.ArgEffect.Read

    def test_enum_valued_kwargs_reach_the_resolver(self):
        """A resolver may key on any kwarg the operator declares, including an
        enum-valued one. The query converts kwargs the same way every other
        binding does, so a `MemorySpace` argument resolves instead of raising."""
        assert (
            ir.get_op_arg_effect("tile.mgather", 2, target_memory=ir.MemorySpace.Mat, coalesce=1)
            == ir.ArgEffect.Write
        )

    def test_set_ffts_declares_no_write(self):
        """`system.set_ffts` hands the workspace *pointer* to the FFTS unit
        (`pto.set_ffts %ws : !pto.ptr<i64>`); it declares where the hardware's
        scratch lives rather than moving any data. The FFTS unit writes that
        region on its own schedule, which no PyPTO dependency edge models."""
        assert ir.op_has_declared_arg_effects("system.set_ffts") is True
        assert ir.get_op_arg_effect("system.set_ffts", 0) == ir.ArgEffect.Read
        assert ir.get_op_write_channel("system.set_ffts") is None

    def test_in_place_gate_asks_about_the_reused_argument(self):
        """The import-time gate must ask about the argument the operator updates
        in place, not merely whether *some* argument was classified.

        `per_arg` cannot answer that on its own — it is resized to cover the
        highest declared index, so a slot nobody named looks like a declared
        `Read`. Without the distinction, an operator declaring
        `set_output_reuses_input(2)` while classifying argument 1 would pass the
        gate with argument 2 still defaulting to `Read`.
        """
        # tile.store declares set_output_reuses_input(2) and classifies 2.
        assert ir.op_has_declared_arg_effect("tile.store", 2) is True
        # Argument 0 is covered by `per_arg` (it was resized past it) but was
        # never named, so no verdict was reached about it.
        assert ir.op_has_declared_arg_effect("tile.store", 0) is False
        # `no_arg_writes()` is a verdict about every argument at once.
        assert ir.op_has_declared_arg_effect("pld.system.wait", 0) is True
        assert ir.op_has_declared_arg_effect("pld.system.wait", 7) is True
        # An operator nobody classified reaches no verdict about any argument.
        assert ir.op_has_declared_arg_effect("tensor.add", 0) is False

    def test_a_write_channel_alone_is_not_a_verdict(self):
        """Declaring only a write channel must not make an operator look classified.

        `set_write_channel()` creates the effect spec as a side effect, so
        "the spec exists" cannot stand in for "a human decided". Were it allowed
        to, an operator that declared a channel and forgot its `set_arg_effect`
        would pass the in-place gate with the argument it updates still
        defaulting to `Read` — the exact silent default this registry exists to
        remove. `no_arg_writes()` records the verdict explicitly instead.

        Every operator that declares a channel therefore also writes something,
        which `ValidateArgEffects()` enforces at import; this pins the invariant
        that check maintains.
        """
        for op_name, written_index in _CHANNEL_OPS.items():
            assert ir.get_op_write_channel(op_name) is not None, op_name
            assert ir.op_has_declared_arg_effect(op_name, written_index), (
                f"{op_name} declares a write channel but reached no verdict about "
                f"argument {written_index}, the one that channel describes"
            )

    def test_composite_collectives_declare_no_write_channel(self):
        """A composite collective updates a data window and a signal through
        different mechanisms, and one operator-level channel cannot describe
        both. Declaring `Dma` for the pair would let the mixed-store diagnostic
        pair a collective's signal write against a scalar `tensor.write` on the
        same buffer and reject a program that is fine. Recording no channel
        keeps them out of that diagnostic, exactly as before this API existed.
        """
        for op_name in (
            ir.get_op("pld.tensor.allreduce").name,
            ir.get_op("pld.tensor.barrier").name,
            ir.get_op("pld.tensor.allgather").name,
            ir.get_op("builtin.tensor.broadcast").name,
        ):
            assert ir.get_op_write_channel(op_name) is None, op_name

    def test_notify_declares_no_write_channel(self):
        """`pld.system.notify` emits `pto.comm.tnotify`, which is neither the
        MTE3 store path nor the scalar D-cache path the mixed-store diagnostic
        orders against each other. Claiming either would make that diagnostic
        reject a valid program, so it declares the write without a channel."""
        assert ir.get_op_arg_effect("pld.system.notify", 0) == ir.ArgEffect.ReadWrite
        assert ir.get_op_write_channel("pld.system.notify") is None

    def test_hard_syncall_does_not_touch_the_workspace(self):
        """The soft form counts arrivals in the GM workspace; the hard form is
        an FFTS barrier that never reads or writes it."""
        assert ir.get_op_arg_effect("system.syncall", 0) == ir.ArgEffect.Read
        assert ir.get_op_arg_effect("system.syncall", 0, mode="soft") == ir.ArgEffect.ReadWrite

    @pytest.mark.parametrize(
        ("op_name", "expected"),
        [
            ("tile.store", ir.WriteChannel.Dma),
            ("tensor.assemble", ir.WriteChannel.Dma),
            ("tile.mscatter", ir.WriteChannel.Dma),
            # The one scalar D-cache writer. PyPTO cannot order a scalar write
            # against an MTE3 store to the same GM tensor, and rejects a
            # function that mixes them.
            ("tensor.write", ir.WriteChannel.Scalar),
            # Declared classified, writes nothing, so no channel.
            ("pld.system.wait", None),
        ],
    )
    def test_write_channel(self, op_name, expected):
        assert ir.get_op_write_channel(op_name) == expected

    def test_every_in_place_op_is_classified(self):
        """An operator whose result reuses an input's buffer writes through that
        argument. Leaving the effect undeclared is what let `tile.mscatter`
        write a GM output while every direction analysis read it as an input.

        `pypto` fails at import when this is violated (see
        `OpRegistry::ValidateArgEffects`); asserting it here names the operator
        and the fix instead of failing the whole test session on import.
        """
        for op_name in _IN_PLACE_OPS:
            assert ir.op_has_declared_arg_effects(op_name), (
                f"{op_name} updates an argument in place but never declared what it does to it. "
                f"Add .set_arg_effect(<index>, ArgEffect::Write) to its REGISTER_OP block — "
                f"ArgEffect::ReadWrite when it accumulates, or .no_arg_writes() when the slot "
                f"is metadata rather than data."
            )


#: Operators declaring a write channel, mapped to the argument that channel
#: describes. Each must also declare a write there — a channel says *how* an
#: operator writes, so one without a write is either a stray declaration or a
#: missing one. `tile.mgather` reaches its verdict through a kwarg resolver,
#: which still counts: the registration named the argument.
_CHANNEL_OPS = {
    ir.get_op(name).name: index
    for name, index in (
        ("tile.store", 2),
        ("tile.mscatter", 2),
        ("tile.mgather", 2),
        ("tensor.write", 0),
        ("tensor.assemble", 0),
        ("pld.tile.put", 0),
        ("pld.tile.get", 0),
        ("pld.tile.remote_store", 1),
    )
}

#: Operators declaring ``set_output_reuses_input``: their SSA result IS an
#: argument's buffer, so they write through it and must classify that argument.
#: Routed through ``get_op`` so a renamed operator fails at import rather than
#: silently dropping out of the coverage this list asserts.
_IN_PLACE_OPS = [
    ir.get_op(name).name
    for name in (
        "array.update_element",
        # main declared these in-place after this series began; the import gate
        # is what surfaced them, so pin them here too.
        "tensor.assemble",
        "tensor.set_validshape",
        "tile.batch_matmul_acc",
        "tile.fillpad_inplace",
        "tile.gather_row",
        "tile.gemv_acc",
        "tile.matmul_acc",
        "tile.matmul_mx_acc",
        "tile.mscatter",
        "tile.scatter",
        "tile.scatter_mask",
        "tile.scatter_update",
        "tile.store",
        "tile.tget_scale_addr",
    )
]


class TestResultAliasContract:
    """The result-alias contract may only name operators that have a result.

    ``ResultAliasedArgIndex`` answers "which argument's buffer does this call's
    result name". A side-effect-only operator deduces ``UnknownType`` — "no SSA
    result for downstream consumers" — so there is no result to alias, and its
    write target travels through ``ArgEffect`` / ``CallWriteTargets`` instead.
    Listing one would invite a consumer to read a destination alias out of a
    bare side effect.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[4]
    _CONTRACT = _REPO_ROOT / "src/ir/transforms/utils/result_alias_utils.cpp"

    def _contract_ops(self) -> set[str]:
        text = self._CONTRACT.read_text()
        return set(re.findall(r'IsOp\(call, "([^"]+)"\)', text))

    def _side_effect_only_ops(self) -> set[str]:
        """Operators whose registered deduction returns ``GetUnknownType()``."""
        op_sources = sorted((self._REPO_ROOT / "src/ir/op").rglob("*.cpp"))
        joined = "\n".join(f.read_text() for f in op_sources)
        side_effect = set()
        for match in re.finditer(r'REGISTER_OP\("([^"]+)"\)(.*?)(?=REGISTER_OP\(|\Z)', joined, re.S):
            name, body = match.group(1), match.group(2)
            deduce = re.search(r"f_deduce_type\(\s*&?([A-Za-z_][A-Za-z0-9_]*)", body)
            if deduce is None:
                continue
            fn = re.search(r"TypePtr\s+" + deduce.group(1) + r"\s*\([^)]*\)[^{]*\{(.*?)\n\}", joined, re.S)
            if fn is not None and "GetUnknownType()" in fn.group(1):
                side_effect.add(name)
        return side_effect

    def test_contract_names_only_registered_operators(self):
        for name in self._contract_ops():
            # Raises if the literal is not a registered operator.
            assert ir.get_op(name).name == name

    def test_contract_excludes_side_effect_only_operators(self):
        side_effect = self._side_effect_only_ops()
        # Guard the detector itself: these are known side-effect-only ops.
        for known in ("pld.tile.put", "pld.tile.get", "pld.system.notify"):
            assert known in side_effect, f"{known} should be detected as side-effect-only"

        listed = sorted(self._contract_ops() & side_effect)
        assert listed == [], "these operators have no SSA result, so they cannot alias one: " + ", ".join(
            listed
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
