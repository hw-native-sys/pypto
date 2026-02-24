# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for tensor variables using Var with TensorType."""

import pytest
from pypto import DataType, ir


class TestTensorVar:
    """Test cases for tensor variables using Var with TensorType."""

    def test_creation_with_constant_shape(self):
        """Test Var with TensorType creation with constant dimensions."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(2, DataType.INT32, span),
            ir.ConstInt(3, DataType.INT32, span),
            ir.ConstInt(4, DataType.INT32, span),
        ]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        tensor_var = ir.Var("A", tensor_type, span)

        assert tensor_var.name == "A"
        assert isinstance(tensor_var.type, ir.TensorType)
        assert tensor_var.type.dtype == DataType.FP32
        assert len(tensor_var.type.shape) == 3
        assert isinstance(tensor_var, ir.Var)
        assert isinstance(tensor_var, ir.Expr)

    def test_creation_with_symbolic_shape(self):
        """Test Var with TensorType with symbolic shape dimensions."""
        span = ir.Span.unknown()
        # Create symbolic shape with Var nodes
        scalar_type = ir.ScalarType(DataType.INT32)
        N = ir.Var("N", scalar_type, span)
        M = ir.Var("M", scalar_type, span)
        K = ir.Var("K", scalar_type, span)
        shape = [N, M, K]

        tensor_type = ir.TensorType(shape, DataType.FP16)
        tensor_var = ir.Var("B", tensor_type, span)

        assert tensor_var.name == "B"
        assert isinstance(tensor_var.type, ir.TensorType)
        assert tensor_var.type.dtype == DataType.FP16
        assert len(tensor_var.type.shape) == 3
        # Check that shape contains expressions
        assert all(isinstance(dim, ir.Expr) for dim in tensor_var.type.shape)

    def test_different_dtypes(self):
        """Test Var with TensorType with different data types."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT32, span)]

        for dtype in [DataType.FP32, DataType.FP16, DataType.INT32, DataType.BOOL]:
            tensor_type = ir.TensorType(shape, dtype)
            tensor = ir.Var("T", tensor_type, span)
            assert isinstance(tensor.type, ir.TensorType) and tensor.type.dtype == dtype

    def test_scalar_shape_dimensions(self):
        """Test tensor with scalar (0-D) shape."""
        span = ir.Span.unknown()
        shape = []  # Scalar tensor

        tensor_type = ir.TensorType(shape, DataType.FP32)
        scalar_tensor = ir.Var("scalar", tensor_type, span)
        assert isinstance(scalar_tensor.type, ir.TensorType) and len(scalar_tensor.type.shape) == 0

    def test_high_dimensional_tensor(self):
        """Test tensor with many dimensions."""
        span = ir.Span.unknown()
        # 5D tensor
        shape = [ir.ConstInt(i + 1, DataType.INT32, span) for i in range(5)]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        tensor = ir.Var("T", tensor_type, span)
        assert isinstance(tensor.type, ir.TensorType) and len(tensor.type.shape) == 5

    def test_mixed_symbolic_constant_shape(self):
        """Test tensor with mixed symbolic and constant dimensions."""
        span = ir.Span.unknown()
        scalar_type = ir.ScalarType(DataType.INT32)
        N = ir.Var("N", scalar_type, span)
        shape = [
            ir.ConstInt(2, DataType.INT32, span),  # Constant
            N,  # Symbolic
            ir.ConstInt(4, DataType.INT32, span),  # Constant
        ]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        tensor = ir.Var("T", tensor_type, span)
        assert isinstance(tensor.type, ir.TensorType) and len(tensor.type.shape) == 3
        assert isinstance(tensor.type.shape[0], ir.ConstInt)
        assert isinstance(tensor.type.shape[1], ir.Var)
        assert isinstance(tensor.type.shape[2], ir.ConstInt)


class TestTensorVarStructuralEqual:
    """Test cases for structural equality of tensor variables."""

    def test_tensor_var_equality_same_instance(self):
        """Test structural equality for same Var instance."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        A = ir.Var("A", tensor_type, span)

        ir.assert_structural_equal(A, A)

    def test_tensor_var_equality_different_instances(self):
        """Test structural equality for different Var instances."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        A1 = ir.Var("A", tensor_type, span)
        A2 = ir.Var("A", tensor_type, span)
        B = ir.Var("B", tensor_type, span)

        # Without auto-mapping, different instances are not equal
        assert not ir.structural_equal(A1, A2, enable_auto_mapping=False)

        # With auto-mapping, same structure should be equal
        ir.assert_structural_equal(A1, B, enable_auto_mapping=True)

    def test_tensor_different_shapes_not_equal(self):
        """Test that tensors with different shapes are not equal."""
        span = ir.Span.unknown()
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(3, DataType.INT32, span)]

        tensor_type1 = ir.TensorType(shape1, DataType.FP32)
        tensor_type2 = ir.TensorType(shape2, DataType.FP32)
        A = ir.Var("A", tensor_type1, span)
        B = ir.Var("A", tensor_type2, span)

        assert not ir.structural_equal(A, B)

    def test_tensor_different_dtypes_not_equal(self):
        """Test that tensors with different dtypes are not equal."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span)]

        tensor_type1 = ir.TensorType(shape, DataType.FP32)
        tensor_type2 = ir.TensorType(shape, DataType.FP16)
        A = ir.Var("A", tensor_type1, span)
        B = ir.Var("A", tensor_type2, span)

        assert not ir.structural_equal(A, B)


class TestTensorVarStructuralHash:
    """Test cases for structural hashing of tensor variables."""

    def test_tensor_var_hash_consistency(self):
        """Test that same Var instance has consistent hash."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        A = ir.Var("A", tensor_type, span)

        hash1 = ir.structural_hash(A)
        hash2 = ir.structural_hash(A)
        assert hash1 == hash2

    def test_tensor_var_hash_different_names(self):
        """Test that different Var names produce different hashes."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span)]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        A = ir.Var("A", tensor_type, span)
        B = ir.Var("B", tensor_type, span)

        hash_a = ir.structural_hash(A, enable_auto_mapping=False)
        hash_b = ir.structural_hash(B, enable_auto_mapping=False)

        # Different names should produce different hashes
        assert hash_a != hash_b

    def test_tensor_var_hash_same_content(self):
        """Test that Var with same content has same hash when auto-mapping is enabled."""
        span = ir.Span.unknown()
        # Create separate shape lists to avoid move semantics issues
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(2, DataType.INT32, span)]

        tensor_type1 = ir.TensorType(shape1, DataType.FP32)
        tensor_type2 = ir.TensorType(shape2, DataType.FP32)
        A1 = ir.Var("A", tensor_type1, span)
        A2 = ir.Var("A", tensor_type2, span)

        # With auto-mapping enabled, same content should have same hash
        hash_a1 = ir.structural_hash(A1, enable_auto_mapping=True)
        hash_a2 = ir.structural_hash(A2, enable_auto_mapping=True)
        assert hash_a1 == hash_a2

    def test_tensor_var_hash_different_shapes(self):
        """Test that different shapes produce different hashes."""
        span = ir.Span.unknown()
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(3, DataType.INT32, span)]

        tensor_type1 = ir.TensorType(shape1, DataType.FP32)
        tensor_type2 = ir.TensorType(shape2, DataType.FP32)
        A = ir.Var("A", tensor_type1, span)
        B = ir.Var("A", tensor_type2, span)

        hash_a = ir.structural_hash(A)
        hash_b = ir.structural_hash(B)

        # Different shapes should have different hashes
        assert hash_a != hash_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
