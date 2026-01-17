# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for TupleType and TupleGetItemExpr."""

import pytest
from pypto import DataType, ir


class TestTupleType:
    """Tests for TupleType class."""

    def test_tuple_type_creation_empty(self):
        """Test creating an empty tuple type."""
        tuple_type = ir.TupleType([])
        assert len(tuple_type.types) == 0

    def test_tuple_type_creation_single(self):
        """Test creating a tuple type with a single element."""
        scalar_type = ir.ScalarType(DataType.INT64)
        tuple_type = ir.TupleType([scalar_type])
        assert len(tuple_type.types) == 1
        assert isinstance(tuple_type.types[0], ir.ScalarType)

    def test_tuple_type_creation_multiple(self):
        """Test creating a tuple type with multiple elements."""
        scalar_type = ir.ScalarType(DataType.INT64)
        tensor_type = ir.TensorType(DataType.FP32, [])
        tuple_type = ir.TupleType([scalar_type, tensor_type])
        assert len(tuple_type.types) == 2
        assert isinstance(tuple_type.types[0], ir.ScalarType)
        assert isinstance(tuple_type.types[1], ir.TensorType)

    def test_tuple_type_attributes(self):
        """Test accessing TupleType attributes."""
        scalar_type = ir.ScalarType(DataType.INT64)
        tuple_type = ir.TupleType([scalar_type])
        assert hasattr(tuple_type, "types")
        assert len(tuple_type.types) == 1

    def test_tuple_type_nested(self):
        """Test creating nested tuple types."""
        inner_tuple = ir.TupleType([ir.ScalarType(DataType.INT64)])
        outer_tuple = ir.TupleType([inner_tuple, ir.ScalarType(DataType.FP32)])
        assert len(outer_tuple.types) == 2
        assert isinstance(outer_tuple.types[0], ir.TupleType)
        assert isinstance(outer_tuple.types[1], ir.ScalarType)

    def test_tuple_with_scalar_types(self):
        """Test tuple containing only scalar types."""
        tuple_type = ir.TupleType(
            [ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32), ir.ScalarType(DataType.INT32)]
        )
        assert len(tuple_type.types) == 3
        for t in tuple_type.types:
            assert isinstance(t, ir.ScalarType)

    def test_tuple_with_tensor_types(self):
        """Test tuple containing tensor types."""
        span = ir.Span.unknown()
        dim1 = ir.ConstInt(10, DataType.INT64, span)
        dim2 = ir.ConstInt(20, DataType.INT64, span)
        tuple_type = ir.TupleType(
            [ir.TensorType(DataType.FP32, [dim1]), ir.TensorType(DataType.INT64, [dim2])]
        )
        assert len(tuple_type.types) == 2
        for t in tuple_type.types:
            assert isinstance(t, ir.TensorType)

    def test_tuple_mixed_types(self):
        """Test tuple containing mixed types."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(10, DataType.INT64, span)
        tuple_type = ir.TupleType(
            [
                ir.ScalarType(DataType.INT64),
                ir.TensorType(DataType.FP32, [dim]),
                ir.TileType(DataType.FP16, [dim, dim]),
                ir.UnknownType(),
            ]
        )
        assert len(tuple_type.types) == 4
        assert isinstance(tuple_type.types[0], ir.ScalarType)
        assert isinstance(tuple_type.types[1], ir.TensorType)
        assert isinstance(tuple_type.types[2], ir.TileType)
        assert isinstance(tuple_type.types[3], ir.UnknownType)


class TestTupleGetItemExpr:
    """Tests for TupleGetItemExpr class."""

    def test_tuple_get_item_creation(self):
        """Test creating a tuple element access expression."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)
        get_item = ir.TupleGetItemExpr(tuple_var, 0, span)

        assert get_item is not None
        assert get_item.tuple == tuple_var
        assert get_item.index == 0

    def test_tuple_get_item_type_inference(self):
        """Test that TupleGetItemExpr correctly infers result type."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)

        # Access first element
        first = ir.TupleGetItemExpr(tuple_var, 0, span)
        assert isinstance(first.type, ir.ScalarType)
        assert first.type.dtype == DataType.INT64

        # Access second element
        second = ir.TupleGetItemExpr(tuple_var, 1, span)
        assert isinstance(second.type, ir.ScalarType)
        assert second.type.dtype == DataType.FP32

    def test_tuple_get_item_bounds_check(self):
        """Test that TupleGetItemExpr checks index bounds."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)

        # Valid indices
        ir.TupleGetItemExpr(tuple_var, 0, span)
        ir.TupleGetItemExpr(tuple_var, 1, span)

        # Out of bounds index should raise error
        with pytest.raises(Exception):
            ir.TupleGetItemExpr(tuple_var, 2, span)

    def test_tuple_get_item_negative_index(self):
        """Test that negative indices are rejected."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)

        with pytest.raises(Exception):
            ir.TupleGetItemExpr(tuple_var, -1, span)

    def test_tuple_get_item_wrong_type(self):
        """Test that non-tuple types are rejected."""
        span = ir.Span.unknown()
        # Create a non-tuple variable
        scalar_var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Should raise error when trying to access as tuple
        with pytest.raises(Exception):
            ir.TupleGetItemExpr(scalar_var, 0, span)

    def test_nested_tuple_get_item(self):
        """Test accessing nested tuples."""
        span = ir.Span.unknown()
        inner_tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        outer_tuple_type = ir.TupleType([inner_tuple_type, ir.ScalarType(DataType.INT32)])
        tuple_var = ir.Var("nested", outer_tuple_type, span)

        # Access first element (which is itself a tuple)
        first = ir.TupleGetItemExpr(tuple_var, 0, span)
        assert isinstance(first.type, ir.TupleType)

        # Access nested element
        nested = ir.TupleGetItemExpr(first, 0, span)
        assert isinstance(nested.type, ir.ScalarType)
        assert nested.type.dtype == DataType.INT64

    def test_tuple_get_item_in_expression(self):
        """Test using TupleGetItemExpr in arithmetic expressions."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.INT64)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)

        first = ir.TupleGetItemExpr(tuple_var, 0, span)
        second = ir.TupleGetItemExpr(tuple_var, 1, span)

        # Use in an add expression
        result = ir.Add(first, second, DataType.INT64, span)
        assert result is not None
        assert isinstance(result, ir.Add)

    def test_call_returning_tuple_with_getitem_in_subsequent_call(self):
        """Test Call returns TupleType, extract elements with GetItem for subsequent Call."""
        span = ir.Span.unknown()

        # Define a tuple return type (e.g., function returns (tensor, scalar))
        dim = ir.ConstInt(10, DataType.INT64, span)
        return_tuple_type = ir.TupleType([ir.TensorType(DataType.FP32, [dim]), ir.ScalarType(DataType.INT64)])

        # Create an operation that returns a tuple
        tuple_op = ir.Op("compute_with_multiple_returns")
        input_var = ir.Var("input", ir.TensorType(DataType.FP32, [dim]), span)

        # Call that returns a tuple
        tuple_call = ir.Call(tuple_op, [input_var], return_tuple_type, span)

        # Assign the tuple result to a variable
        tuple_result = ir.Var("result", return_tuple_type, span)
        assign_tuple = ir.AssignStmt(tuple_result, tuple_call, span)

        # Extract first element (tensor) from tuple
        extracted_tensor = ir.TupleGetItemExpr(tuple_result, 0, span)
        assert isinstance(extracted_tensor.type, ir.TensorType)

        # Extract second element (scalar) from tuple
        extracted_scalar = ir.TupleGetItemExpr(tuple_result, 1, span)
        assert isinstance(extracted_scalar.type, ir.ScalarType)

        # Use extracted tensor in a subsequent call
        process_op = ir.Op("process_tensor")
        subsequent_call = ir.Call(process_op, [extracted_tensor], span)

        # Use extracted scalar in another operation
        scalar_add = ir.Add(extracted_scalar, ir.ConstInt(1, DataType.INT64, span), DataType.INT64, span)

        # Create a complete statement sequence
        tensor_var = ir.Var("processed", ir.UnknownType(), span)
        assign_tensor = ir.AssignStmt(tensor_var, subsequent_call, span)

        scalar_var = ir.Var("count", ir.ScalarType(DataType.INT64), span)
        assign_scalar = ir.AssignStmt(scalar_var, scalar_add, span)

        # Build complete sequence
        seq = ir.SeqStmts([assign_tuple, assign_tensor, assign_scalar], span)

        # Verify the structure
        assert isinstance(seq, ir.SeqStmts)
        assert len(seq.stmts) == 3

        # Verify first statement assigns tuple call result
        assert isinstance(seq.stmts[0], ir.AssignStmt)
        assert isinstance(seq.stmts[0].value, ir.Call)
        assert isinstance(seq.stmts[0].value.type, ir.TupleType)

        # Verify second statement uses extracted tensor
        assert isinstance(seq.stmts[1], ir.AssignStmt)
        assert isinstance(seq.stmts[1].value, ir.Call)

        # Verify third statement uses extracted scalar
        assert isinstance(seq.stmts[2], ir.AssignStmt)
        assert isinstance(seq.stmts[2].value, ir.Add)

        # Test serialization of the complete workflow
        data = ir.serialize(seq)
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.SeqStmts)
        assert len(restored.stmts) == 3


class TestTupleSerialization:
    """Tests for TupleType and TupleGetItemExpr serialization."""

    def test_tuple_type_serialization(self):
        """Test serializing and deserializing TupleType."""
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        span = ir.Span.unknown()
        var = ir.Var("t", tuple_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TupleType)
        assert len(restored.type.types) == 2

    def test_tuple_get_item_serialization(self):
        """Test serializing and deserializing TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)
        get_item = ir.TupleGetItemExpr(tuple_var, 1, span)

        # Wrap in a statement for serialization
        result_var = ir.Var("result", get_item.type, span)
        stmt = ir.AssignStmt(result_var, get_item, span)

        # Serialize
        data = ir.serialize(stmt)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.AssignStmt)
        assert isinstance(restored.value, ir.TupleGetItemExpr)
        assert restored.value.index == 1

    def test_nested_tuple_serialization(self):
        """Test serializing nested tuples."""
        inner_tuple = ir.TupleType([ir.ScalarType(DataType.INT64)])
        outer_tuple = ir.TupleType([inner_tuple, ir.ScalarType(DataType.FP32)])
        span = ir.Span.unknown()
        var = ir.Var("nested", outer_tuple, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TupleType)
        assert len(restored.type.types) == 2
        assert isinstance(restored.type.types[0], ir.TupleType)


class TestTupleStructuralComparison:
    """Tests for TupleType and TupleGetItemExpr structural comparison."""

    def test_tuple_structural_equal(self):
        """Test structural equality of TupleType."""
        tuple1 = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple2 = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])

        span = ir.Span.unknown()
        var1 = ir.Var("t1", tuple1, span)
        var2 = ir.Var("t2", tuple2, span)

        # Should be structurally equal
        assert ir.structural_equal(var1, var2, enable_auto_mapping=True)

    def test_tuple_structural_hash(self):
        """Test structural hash consistency for TupleType."""
        tuple1 = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple2 = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])

        span = ir.Span.unknown()
        var1 = ir.Var("t1", tuple1, span)
        var2 = ir.Var("t2", tuple2, span)

        # Structurally equal nodes should have same hash
        hash1 = ir.structural_hash(var1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(var2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_tuple_different_order(self):
        """Test that tuples with different element order are not equal."""
        tuple1 = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple2 = ir.TupleType([ir.ScalarType(DataType.FP32), ir.ScalarType(DataType.INT64)])

        span = ir.Span.unknown()
        var1 = ir.Var("t1", tuple1, span)
        var2 = ir.Var("t2", tuple2, span)

        # Should NOT be structurally equal (different order)
        assert not ir.structural_equal(var1, var2)

    def test_get_item_structural_equal(self):
        """Test structural equality of TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var1 = ir.Var("t1", tuple_type, span)
        tuple_var2 = ir.Var("t2", tuple_type, span)

        get_item1 = ir.TupleGetItemExpr(tuple_var1, 0, span)
        get_item2 = ir.TupleGetItemExpr(tuple_var2, 0, span)

        # Should be structurally equal
        assert ir.structural_equal(get_item1, get_item2, enable_auto_mapping=True)

    def test_get_item_structural_hash(self):
        """Test structural hash consistency for TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var1 = ir.Var("t1", tuple_type, span)
        tuple_var2 = ir.Var("t2", tuple_type, span)

        get_item1 = ir.TupleGetItemExpr(tuple_var1, 0, span)
        get_item2 = ir.TupleGetItemExpr(tuple_var2, 0, span)

        # Structurally equal nodes should have same hash
        hash1 = ir.structural_hash(get_item1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(get_item2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_get_item_different_indices(self):
        """Test that different indices result in different hashes."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("t", tuple_type, span)

        get_item0 = ir.TupleGetItemExpr(tuple_var, 0, span)
        get_item1 = ir.TupleGetItemExpr(tuple_var, 1, span)

        # Different indices should result in different hashes
        hash0 = ir.structural_hash(get_item0)
        hash1 = ir.structural_hash(get_item1)
        assert hash0 != hash1


class TestTuplePythonPrinter:
    """Tests for Python printing of TupleType and TupleGetItemExpr."""

    def test_python_print_tuple_type(self):
        """Test Python printing of TupleType."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        var = ir.Var("my_tuple", tuple_type, span)
        assign = ir.AssignStmt(var, ir.ConstInt(0, DataType.INT64, span), span)

        result = ir.python_print(assign)
        assert "pi.Tuple([" in result
        assert "pi.Int64" in result
        assert "pi.FP32" in result

    def test_python_print_tuple_get_item(self):
        """Test Python printing of TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)
        get_item = ir.TupleGetItemExpr(tuple_var, 0, span)
        result_var = ir.Var("result", get_item.type, span)
        assign = ir.AssignStmt(result_var, get_item, span)

        result = ir.python_print(assign)
        assert "my_tuple[0]" in result

    def test_python_print_nested_tuple_access(self):
        """Test Python printing of nested tuple access."""
        span = ir.Span.unknown()
        inner_tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        outer_tuple_type = ir.TupleType([inner_tuple_type, ir.ScalarType(DataType.INT32)])
        tuple_var = ir.Var("nested", outer_tuple_type, span)

        first = ir.TupleGetItemExpr(tuple_var, 0, span)
        nested = ir.TupleGetItemExpr(first, 1, span)
        result_var = ir.Var("result", nested.type, span)
        assign = ir.AssignStmt(result_var, nested, span)

        result = ir.python_print(assign)
        assert "nested[0][1]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
