# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IterArg class."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestIterArg:
    """Test IterArg class."""

    def test_iter_arg_creation(self):
        """Test creating an IterArg instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        name = "iter_arg"
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg = ir.IterArg(name, ir.ScalarType(dtype), init_value, span)

        assert iter_arg is not None
        assert iter_arg.span.filename == "test.py"
        assert iter_arg.name == name
        assert iter_arg.initValue is not None
        assert isinstance(iter_arg.initValue, ir.ConstInt)

    def test_iter_arg_has_attributes(self):
        """Test that IterArg has name, initValue, and value attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        name = "iter_arg"
        init_value = ir.ConstInt(5, dtype, span)
        iter_arg = ir.IterArg(name, ir.ScalarType(dtype), init_value, span)

        assert iter_arg.name == name
        assert iter_arg.initValue is not None
        assert cast(ir.ConstInt, iter_arg.initValue).value == 5

    def test_iter_arg_is_var(self):
        """Test that IterArg is an instance of Var."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value, span)

        assert isinstance(iter_arg, ir.Var)
        assert isinstance(iter_arg, ir.Expr)
        assert isinstance(iter_arg, ir.IRNode)

    def test_iter_arg_immutability(self):
        """Test that IterArg attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            iter_arg.name = "new_name"  # type: ignore
        with pytest.raises(AttributeError):
            iter_arg.initValue = ir.ConstInt(1, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            iter_arg.value = ir.Var("new_v", ir.ScalarType(dtype), span)  # type: ignore

    def test_iter_arg_with_different_init_value_types(self):
        """Test IterArg with different expression types for initValue."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64

        # Test with ConstInt
        init_value1 = ir.ConstInt(5, dtype, span)
        iter_arg1 = ir.IterArg("arg1", ir.ScalarType(dtype), init_value1, span)
        assert isinstance(iter_arg1.initValue, ir.ConstInt)

        # Test with Var
        init_value2 = ir.Var("x", ir.ScalarType(dtype), span)
        iter_arg2 = ir.IterArg("arg2", ir.ScalarType(dtype), init_value2, span)
        assert isinstance(iter_arg2.initValue, ir.Var)

        # Test with binary expression
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        init_value3 = ir.Add(x, y, dtype, span)
        iter_arg3 = ir.IterArg("arg3", ir.ScalarType(dtype), init_value3, span)
        assert isinstance(iter_arg3.initValue, ir.Add)


class TestIterArgHash:
    """Tests for IterArg hash function."""

    def test_iter_arg_same_structure_hash(self):
        """Test IterArg nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value1 = ir.ConstInt(0, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value1, span)

        init_value2 = ir.ConstInt(0, dtype, span)
        iter_arg2 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value2, span)

        hash1 = ir.structural_hash(iter_arg1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(iter_arg2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_iter_arg_different_name_hash(self):
        """Test IterArg nodes with different names hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg1", ir.ScalarType(dtype), init_value, span)
        iter_arg2 = ir.IterArg("iter_arg2", ir.ScalarType(dtype), init_value, span)

        hash1 = ir.structural_hash(iter_arg1)
        hash2 = ir.structural_hash(iter_arg2)
        assert hash1 != hash2

    def test_iter_arg_different_init_value_hash(self):
        """Test IterArg nodes with different initValue hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value1 = ir.ConstInt(0, dtype, span)
        init_value2 = ir.ConstInt(1, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value1, span)
        iter_arg2 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value2, span)

        hash1 = ir.structural_hash(iter_arg1)
        hash2 = ir.structural_hash(iter_arg2)
        assert hash1 != hash2

    def test_iter_arg_different_value_hash(self):
        """Test IterArg nodes with different value hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value, span)
        iter_arg2 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value, span)

        hash1 = ir.structural_hash(iter_arg1)
        hash2 = ir.structural_hash(iter_arg2)
        assert hash1 != hash2


class TestIterArgEquality:
    """Tests for IterArg structural equality function."""

    def test_iter_arg_structural_equal(self):
        """Test structural equality of IterArg nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value1 = ir.ConstInt(0, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value1, span)

        init_value2 = ir.ConstInt(0, dtype, span)
        iter_arg2 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value2, span)

        assert ir.structural_equal(iter_arg1, iter_arg2, enable_auto_mapping=True)

    def test_iter_arg_different_name_not_equal(self):
        """Test IterArg nodes with different names are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg1", ir.ScalarType(dtype), init_value, span)
        iter_arg2 = ir.IterArg("iter_arg2", ir.ScalarType(dtype), init_value, span)

        assert not ir.structural_equal(iter_arg1, iter_arg2)

    def test_iter_arg_different_init_value_not_equal(self):
        """Test IterArg nodes with different initValue are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value1 = ir.ConstInt(0, dtype, span)
        init_value2 = ir.ConstInt(1, dtype, span)
        iter_arg1 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value1, span)
        iter_arg2 = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value2, span)

        assert not ir.structural_equal(iter_arg1, iter_arg2)

    # Removed test_iter_arg_different_value_not_equal since value field no longer exists

    def test_iter_arg_different_from_base_var_not_equal(self):
        """Test IterArg and base Var nodes are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        init_value = ir.ConstInt(0, dtype, span)
        iter_arg = ir.IterArg("iter_arg", ir.ScalarType(dtype), init_value, span)
        base_var = ir.Var("iter_arg", ir.ScalarType(dtype), span)

        assert not ir.structural_equal(iter_arg, base_var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
