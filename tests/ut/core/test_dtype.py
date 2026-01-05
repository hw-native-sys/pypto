# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the DataType enum and related utility functions."""

import pypto
import pytest
from pypto import (
    DT_BF16,
    DT_BOOL,
    DT_FP4,
    DT_FP8,
    DT_FP16,
    DT_FP32,
    DT_HF4,
    DT_HF8,
    DT_INT4,
    DT_INT8,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_UINT4,
    DT_UINT8,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
    DataType,
)


class TestDataTypeEnum:
    """Test DataType enumeration values and access patterns."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        # Signed integers
        assert hasattr(DataType, "INT4")
        assert hasattr(DataType, "INT8")
        assert hasattr(DataType, "INT16")
        assert hasattr(DataType, "INT32")
        assert hasattr(DataType, "INT64")

        # Floating point
        assert hasattr(DataType, "FP8")
        assert hasattr(DataType, "FP16")
        assert hasattr(DataType, "FP32")
        assert hasattr(DataType, "BF16")

        # Hybrid float
        assert hasattr(DataType, "HF4")
        assert hasattr(DataType, "HF8")

        # Unsigned integers
        assert hasattr(DataType, "UINT8")
        assert hasattr(DataType, "UINT16")
        assert hasattr(DataType, "UINT32")
        assert hasattr(DataType, "UINT64")

        # Boolean
        assert hasattr(DataType, "BOOL")

    def test_enum_values_are_unique(self):
        """Test that all enum values have unique integer values."""
        values = [
            DataType.INT4,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT4,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.FP4,
            DataType.FP8,
            DataType.FP16,
            DataType.FP32,
            DataType.BF16,
            DataType.HF4,
            DataType.HF8,
            DataType.BOOL,
        ]
        # Convert to int to compare underlying values
        int_values = [v.Code() for v in values]
        assert len(int_values) == len(set(int_values)), "Enum values must be unique"

    def test_convenience_constants(self):
        """Test that convenience constants match DataType enum values."""
        assert DT_INT4 == DataType.INT4
        assert DT_INT8 == DataType.INT8
        assert DT_INT16 == DataType.INT16
        assert DT_INT32 == DataType.INT32
        assert DT_INT64 == DataType.INT64
        assert DT_UINT4 == DataType.UINT4
        assert DT_UINT8 == DataType.UINT8
        assert DT_UINT16 == DataType.UINT16
        assert DT_UINT32 == DataType.UINT32
        assert DT_UINT64 == DataType.UINT64
        assert DT_FP4 == DataType.FP4
        assert DT_FP8 == DataType.FP8
        assert DT_FP16 == DataType.FP16
        assert DT_FP32 == DataType.FP32
        assert DT_BF16 == DataType.BF16
        assert DT_HF4 == DataType.HF4
        assert DT_HF8 == DataType.HF8
        assert DT_BOOL == DataType.BOOL

    def test_convenience_constants_in_pypto_namespace(self):
        """Test that convenience constants are accessible from pypto module."""
        assert hasattr(pypto, "DT_INT4")
        assert hasattr(pypto, "DT_INT8")
        assert hasattr(pypto, "DT_INT16")
        assert hasattr(pypto, "DT_INT32")
        assert hasattr(pypto, "DT_INT64")
        assert hasattr(pypto, "DT_UINT4")
        assert hasattr(pypto, "DT_UINT8")
        assert hasattr(pypto, "DT_UINT16")
        assert hasattr(pypto, "DT_UINT32")
        assert hasattr(pypto, "DT_UINT64")
        assert hasattr(pypto, "DT_FP4")
        assert hasattr(pypto, "DT_FP8")
        assert hasattr(pypto, "DT_FP16")
        assert hasattr(pypto, "DT_FP32")
        assert hasattr(pypto, "DT_BF16")
        assert hasattr(pypto, "DT_HF4")
        assert hasattr(pypto, "DT_HF8")
        assert hasattr(pypto, "DT_BOOL")
        assert pypto.DT_INT32 == DataType.INT32


class TestDataTypeBit:
    """Test GetBit() method."""

    def test_4bit_types(self):
        """Test data types that are 4 bits."""
        assert pypto.DT_INT4.GetBit() == 4
        assert pypto.DT_UINT4.GetBit() == 4
        assert pypto.DT_FP4.GetBit() == 4
        assert pypto.DT_HF4.GetBit() == 4

    def test_8bit_types(self):
        """Test data types that are 8 bits."""
        assert pypto.DT_INT8.GetBit() == 8
        assert pypto.DT_UINT8.GetBit() == 8
        assert pypto.DT_FP8.GetBit() == 8
        assert pypto.DT_HF8.GetBit() == 8
        assert pypto.DT_BOOL.GetBit() == 8

    def test_16bit_types(self):
        """Test data types that are 16 bits."""
        assert pypto.DT_INT16.GetBit() == 16
        assert pypto.DT_UINT16.GetBit() == 16
        assert pypto.DT_FP16.GetBit() == 16
        assert pypto.DT_BF16.GetBit() == 16

    def test_32bit_types(self):
        """Test data types that are 32 bits."""
        assert pypto.DT_INT32.GetBit() == 32
        assert pypto.DT_UINT32.GetBit() == 32
        assert pypto.DT_FP32.GetBit() == 32

    def test_64bit_types(self):
        """Test data types that are 64 bits."""
        assert pypto.DT_INT64.GetBit() == 64
        assert pypto.DT_UINT64.GetBit() == 64


class TestDataTypeString:
    """Test ToString() method."""

    def test_signed_integer_strings(self):
        """Test string representation of signed integer types."""
        assert pypto.DT_INT4.ToString() == "int4"
        assert pypto.DT_INT8.ToString() == "int8"
        assert pypto.DT_INT16.ToString() == "int16"
        assert pypto.DT_INT32.ToString() == "int32"
        assert pypto.DT_INT64.ToString() == "int64"

    def test_unsigned_integer_strings(self):
        """Test string representation of unsigned integer types."""
        assert pypto.DT_UINT4.ToString() == "uint4"
        assert pypto.DT_UINT8.ToString() == "uint8"
        assert pypto.DT_UINT16.ToString() == "uint16"
        assert pypto.DT_UINT32.ToString() == "uint32"
        assert pypto.DT_UINT64.ToString() == "uint64"

    def test_floating_point_strings(self):
        """Test string representation of floating point types."""
        assert pypto.DT_FP4.ToString() == "fp4"
        assert pypto.DT_FP8.ToString() == "fp8"
        assert pypto.DT_FP16.ToString() == "fp16"
        assert pypto.DT_FP32.ToString() == "fp32"
        assert pypto.DT_BF16.ToString() == "bfloat16"

    def test_hybrid_float_strings(self):
        """Test string representation of hybrid float types."""
        assert pypto.DT_HF4.ToString() == "hf4"
        assert pypto.DT_HF8.ToString() == "hf8"

    def test_bool_string(self):
        """Test string representation of boolean type."""
        assert pypto.DT_BOOL.ToString() == "bool"


class TestDataTypePredicates:
    """Test type checking predicate methods."""

    def test_is_float(self):
        """Test IsFloat() correctly identifies floating point types."""
        # Floating point types
        assert pypto.DT_FP4.IsFloat() is True
        assert pypto.DT_FP8.IsFloat() is True
        assert pypto.DT_FP16.IsFloat() is True
        assert pypto.DT_FP32.IsFloat() is True
        assert pypto.DT_BF16.IsFloat() is True
        assert pypto.DT_HF4.IsFloat() is True
        assert pypto.DT_HF8.IsFloat() is True

        # Non-floating point types
        assert pypto.DT_INT8.IsFloat() is False
        assert pypto.DT_INT32.IsFloat() is False
        assert pypto.DT_UINT8.IsFloat() is False
        assert pypto.DT_BOOL.IsFloat() is False

    def test_is_signed_int(self):
        """Test IsSignedInt() correctly identifies signed integer types."""
        # Signed integer types
        assert pypto.DT_INT4.IsSignedInt() is True
        assert pypto.DT_INT8.IsSignedInt() is True
        assert pypto.DT_INT16.IsSignedInt() is True
        assert pypto.DT_INT32.IsSignedInt() is True
        assert pypto.DT_INT64.IsSignedInt() is True

        # Non-signed integer types
        assert pypto.DT_UINT8.IsSignedInt() is False
        assert pypto.DT_FP32.IsSignedInt() is False
        assert pypto.DT_BOOL.IsSignedInt() is False

    def test_is_unsigned_int(self):
        """Test IsUnsignedInt() correctly identifies unsigned integer types."""
        # Unsigned integer types
        assert pypto.DT_UINT4.IsUnsignedInt() is True
        assert pypto.DT_UINT8.IsUnsignedInt() is True
        assert pypto.DT_UINT16.IsUnsignedInt() is True
        assert pypto.DT_UINT32.IsUnsignedInt() is True
        assert pypto.DT_UINT64.IsUnsignedInt() is True

        # Non-unsigned integer types
        assert pypto.DT_INT8.IsUnsignedInt() is False
        assert pypto.DT_FP32.IsUnsignedInt() is False
        assert pypto.DT_BOOL.IsUnsignedInt() is False

    def test_is_int(self):
        """Test IsInt() correctly identifies any integer types."""
        # Integer types (both signed and unsigned)
        assert pypto.DT_INT4.IsInt() is True
        assert pypto.DT_INT8.IsInt() is True
        assert pypto.DT_INT16.IsInt() is True
        assert pypto.DT_INT32.IsInt() is True
        assert pypto.DT_INT64.IsInt() is True
        assert pypto.DT_UINT4.IsInt() is True
        assert pypto.DT_UINT8.IsInt() is True
        assert pypto.DT_UINT16.IsInt() is True
        assert pypto.DT_UINT32.IsInt() is True
        assert pypto.DT_UINT64.IsInt() is True

        # Non-integer types
        assert pypto.DT_FP4.IsInt() is False
        assert pypto.DT_FP8.IsInt() is False
        assert pypto.DT_FP16.IsInt() is False
        assert pypto.DT_FP32.IsInt() is False
        assert pypto.DT_BF16.IsInt() is False
        assert pypto.DT_HF4.IsInt() is False
        assert pypto.DT_HF8.IsInt() is False
        assert pypto.DT_BOOL.IsInt() is False

    def test_type_predicates_mutual_exclusion(self):
        """Test that signed, unsigned, and floating point are mutually exclusive."""
        all_types = [
            DT_INT4,
            DT_INT8,
            DT_INT16,
            DT_INT32,
            DT_INT64,
            DT_FP4,
            DT_FP8,
            DT_FP16,
            DT_FP32,
            DT_BF16,
            DT_HF4,
            DT_HF8,
            DT_UINT4,
            DT_UINT8,
            DT_UINT16,
            DT_UINT32,
            DT_UINT64,
            DT_BOOL,
        ]

        for dtype in all_types:
            # A type should not be both signed integer and unsigned integer
            if dtype.IsSignedInt():
                assert not dtype.IsUnsignedInt()

            # A type should not be both integer and floating point
            if dtype.IsInt():
                assert not dtype.IsFloat()


class TestDataTypeIntegration:
    """Integration tests for DataType system."""

    all_types: list[DataType] = [
        pypto.DT_INT4,
        pypto.DT_INT8,
        pypto.DT_INT16,
        pypto.DT_INT32,
        pypto.DT_INT64,
        pypto.DT_UINT4,
        pypto.DT_UINT8,
        pypto.DT_UINT16,
        pypto.DT_UINT32,
        pypto.DT_UINT64,
        pypto.DT_FP4,
        pypto.DT_FP8,
        pypto.DT_FP16,
        pypto.DT_FP32,
        pypto.DT_BF16,
        pypto.DT_HF4,
        pypto.DT_HF8,
        pypto.DT_BOOL,
    ]

    def test_all_types_have_bit_size(self):
        """Test that all data types have a valid bit size."""

        for dtype in self.all_types:
            bit_size = dtype.GetBit()
            assert bit_size > 0, f"Type {dtype.ToString()} should have positive bit size"
            assert bit_size in [4, 8, 16, 32, 64], f"Type {dtype.ToString()} should have valid bit size"

    def test_all_types_have_string_representation(self):
        """Test that all data types have a valid string representation."""

        for dtype in self.all_types:
            string_repr = dtype.ToString()
            assert string_repr != "unknown", f"Type {dtype} should have valid string representation"
            assert len(string_repr) > 0, f"Type {dtype} should have non-empty string representation"

    def test_all_types_classified(self):
        """Test that all data types are classified as either integer, float, or bool."""

        for dtype in self.all_types:
            is_integer = dtype.IsInt()
            is_floating = dtype.IsFloat()
            is_boolean = dtype == pypto.DT_BOOL

            # Each type should be classified as at least one category
            # (bool is a special case that's neither int nor float in this classification)
            assert is_integer or is_floating or is_boolean, (
                f"Type {dtype.ToString()} should be classified as int, float, or bool"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
