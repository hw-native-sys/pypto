# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for TensorExpr IR nodes (TensorVar only)."""

import pytest
from pypto import DataType
from pypto.pypto_core import ir


class TestTensorVar:
    """Test cases for TensorVar."""

    def test_creation_with_constant_shape(self):
        """Test TensorVar creation with constant dimensions."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(2, DataType.INT32, span),
            ir.ConstInt(3, DataType.INT32, span),
            ir.ConstInt(4, DataType.INT32, span),
        ]

        tensor_var = ir.TensorVar("A", DataType.FP32, shape, span)

        assert tensor_var.name == "A"
        assert tensor_var.dtype == DataType.FP32
        assert len(tensor_var.shape) == 3
        assert isinstance(tensor_var, ir.TensorExpr)
        assert isinstance(tensor_var, ir.Expr)

    def test_creation_with_symbolic_shape(self):
        """Test TensorVar with symbolic shape dimensions."""
        span = ir.Span.unknown()
        # Create symbolic shape with Var nodes
        N = ir.Var("N", DataType.INT32, span)
        M = ir.Var("M", DataType.INT32, span)
        K = ir.Var("K", DataType.INT32, span)
        shape = [N, M, K]

        tensor_var = ir.TensorVar("B", DataType.FP16, shape, span)

        assert tensor_var.name == "B"
        assert tensor_var.dtype == DataType.FP16
        assert len(tensor_var.shape) == 3
        # Check that shape contains symbolic expressions
        assert all(isinstance(dim, ir.ScalarExpr) for dim in tensor_var.shape)

    def test_different_dtypes(self):
        """Test TensorVar with different data types."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT32, span)]

        for dtype in [DataType.FP32, DataType.FP16, DataType.INT32, DataType.BOOL]:
            tensor = ir.TensorVar("T", dtype, shape, span)
            assert tensor.dtype == dtype

    def test_scalar_shape_dimensions(self):
        """Test tensor with scalar (0-D) shape."""
        span = ir.Span.unknown()
        shape = []  # Scalar tensor

        scalar_tensor = ir.TensorVar("scalar", DataType.FP32, shape, span)
        assert len(scalar_tensor.shape) == 0

    def test_high_dimensional_tensor(self):
        """Test tensor with many dimensions."""
        span = ir.Span.unknown()
        # 5D tensor
        shape = [ir.ConstInt(i + 1, DataType.INT32, span) for i in range(5)]

        tensor = ir.TensorVar("T", DataType.FP32, shape, span)
        assert len(tensor.shape) == 5

    def test_mixed_symbolic_constant_shape(self):
        """Test tensor with mixed symbolic and constant dimensions."""
        span = ir.Span.unknown()
        N = ir.Var("N", DataType.INT32, span)
        shape = [
            ir.ConstInt(2, DataType.INT32, span),  # Constant
            N,  # Symbolic
            ir.ConstInt(4, DataType.INT32, span),  # Constant
        ]

        tensor = ir.TensorVar("T", DataType.FP32, shape, span)
        assert len(tensor.shape) == 3
        assert isinstance(tensor.shape[0], ir.ConstInt)
        assert isinstance(tensor.shape[1], ir.Var)
        assert isinstance(tensor.shape[2], ir.ConstInt)


class TestTensorVarStructuralEqual:
    """Test cases for structural equality of TensorVar."""

    def test_tensor_var_equality_same_instance(self):
        """Test structural equality for same TensorVar instance."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape, span)

        assert ir.structural_equal(A, A)

    def test_tensor_var_equality_different_instances(self):
        """Test structural equality for different TensorVar instances."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        A1 = ir.TensorVar("A", DataType.FP32, shape, span)
        A2 = ir.TensorVar("A", DataType.FP32, shape, span)
        B = ir.TensorVar("B", DataType.FP32, shape, span)

        # Without auto-mapping, different instances are not equal
        assert not ir.structural_equal(A1, A2, enable_auto_mapping=False)

        # With auto-mapping, same structure should be equal
        assert ir.structural_equal(A1, B, enable_auto_mapping=True)

    def test_tensor_different_shapes_not_equal(self):
        """Test that tensors with different shapes are not equal."""
        span = ir.Span.unknown()
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(3, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape1, span)
        B = ir.TensorVar("A", DataType.FP32, shape2, span)

        assert not ir.structural_equal(A, B)

    def test_tensor_different_dtypes_not_equal(self):
        """Test that tensors with different dtypes are not equal."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape, span)
        B = ir.TensorVar("A", DataType.FP16, shape, span)

        assert not ir.structural_equal(A, B)


class TestTensorVarStructuralHash:
    """Test cases for structural hashing of TensorVar."""

    def test_tensor_var_hash_consistency(self):
        """Test that same TensorVar instance has consistent hash."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), ir.ConstInt(3, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape, span)

        hash1 = ir.structural_hash(A)
        hash2 = ir.structural_hash(A)
        assert hash1 == hash2

    def test_tensor_var_hash_different_names(self):
        """Test that different TensorVar names produce different hashes."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape, span)
        B = ir.TensorVar("B", DataType.FP32, shape, span)

        hash_a = ir.structural_hash(A, enable_auto_mapping=False)
        hash_b = ir.structural_hash(B, enable_auto_mapping=False)

        # Different names should produce different hashes
        assert hash_a != hash_b

    def test_tensor_var_hash_same_content(self):
        """Test that TensorVar with same content has same hash when auto-mapping is enabled."""
        span = ir.Span.unknown()
        # Create separate shape lists to avoid move semantics issues
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(2, DataType.INT32, span)]

        A1 = ir.TensorVar("A", DataType.FP32, shape1, span)
        A2 = ir.TensorVar("A", DataType.FP32, shape2, span)

        # With auto-mapping enabled, same content should have same hash
        hash_a1 = ir.structural_hash(A1, enable_auto_mapping=True)
        hash_a2 = ir.structural_hash(A2, enable_auto_mapping=True)
        assert hash_a1 == hash_a2

    def test_tensor_var_hash_different_shapes(self):
        """Test that different shapes produce different hashes."""
        span = ir.Span.unknown()
        shape1 = [ir.ConstInt(2, DataType.INT32, span)]
        shape2 = [ir.ConstInt(3, DataType.INT32, span)]

        A = ir.TensorVar("A", DataType.FP32, shape1, span)
        B = ir.TensorVar("A", DataType.FP32, shape2, span)

        hash_a = ir.structural_hash(A)
        hash_b = ir.structural_hash(B)

        # Different shapes should have different hashes
        assert hash_a != hash_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
