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

import pytest
from pypto import DataType, ir


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
    with pytest.raises(Exception):
        ir.create_op_call("tensor.add", [var_a], span)

    # Too many arguments
    var_b = ir.Var("b", tensor_type, span)
    var_c = ir.Var("c", tensor_type, span)
    with pytest.raises(Exception):
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

    with pytest.raises(Exception):
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
    with pytest.raises(Exception):
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


def test_matmul_with_unknown_kwarg():
    """Test tensor.matmul with unknown kwarg should raise error."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    # Unknown kwarg should raise ValueError
    kwargs = {"unknown_param": 123, "a_trans": False}

    with pytest.raises(Exception) as exc_info:
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

    with pytest.raises(Exception) as exc_info:
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

    # Cast from FP16 to FP32
    kwargs = {"target_type": DataType.FP32}
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
        """tile.load output is from kwarg, defaults to Vec."""
        spec = ir.get_op_memory_spec("tile.load")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
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
        """Broadcast binary ops (tile.row_expand_add) constrain both tile inputs."""
        spec = ir.get_op_memory_spec("tile.row_expand_add")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec
        constraints = spec["input_constraints"]
        assert len(constraints) == 2
        assert constraints[0] == [ir.MemorySpace.Vec]
        assert constraints[1] == [ir.MemorySpace.Vec]

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
        """tile.create output is from kwarg, defaults to Vec."""
        spec = ir.get_op_memory_spec("tile.create")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Vec


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

    def test_kwarg_output_returns_default_enum(self):
        """Kwarg-based output resolves to default MemorySpace enum."""
        spec = ir.get_op_memory_spec("tile.load")
        assert spec is not None
        assert isinstance(spec["output_memory"], ir.MemorySpace)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
