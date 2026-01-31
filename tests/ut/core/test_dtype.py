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
    DT_FP8E4M3FN,
    DT_FP8E5M2,
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
        assert hasattr(DataType, "FP8E4M3FN")
        assert hasattr(DataType, "FP8E5M2")
        assert hasattr(DataType, "FP16")
        assert hasattr(DataType, "FP32")
        assert hasattr(DataType, "BF16")

        # Hisilicon float
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
            DataType.FP8E4M3FN,
            DataType.FP8E5M2,
            DataType.FP16,
            DataType.FP32,
            DataType.BF16,
            DataType.HF4,
            DataType.HF8,
            DataType.BOOL,
        ]
        # Convert to int to compare underlying values
        int_values = [v.code() for v in values]
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
        assert DT_FP8E4M3FN == DataType.FP8E4M3FN
        assert DT_FP8E5M2 == DataType.FP8E5M2
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
        assert hasattr(pypto, "DT_FP8E4M3FN")
        assert hasattr(pypto, "DT_FP8E5M2")
        assert hasattr(pypto, "DT_FP16")
        assert hasattr(pypto, "DT_FP32")
        assert hasattr(pypto, "DT_BF16")
        assert hasattr(pypto, "DT_HF4")
        assert hasattr(pypto, "DT_HF8")
        assert hasattr(pypto, "DT_BOOL")
        assert pypto.DT_INT32 == DataType.INT32


class TestDataTypeBit:
    """Test GetBit() method."""

    def test_1bit_types(self):
        """Test data types that are 1 bit."""
        assert pypto.DT_BOOL.get_bit() == 1

    def test_4bit_types(self):
        """Test data types that are 4 bits."""
        assert pypto.DT_INT4.get_bit() == 4
        assert pypto.DT_UINT4.get_bit() == 4
        assert pypto.DT_FP4.get_bit() == 4
        assert pypto.DT_HF4.get_bit() == 4

    def test_8bit_types(self):
        """Test data types that are 8 bits."""
        assert pypto.DT_INT8.get_bit() == 8
        assert pypto.DT_UINT8.get_bit() == 8
        assert pypto.DT_FP8E4M3FN.get_bit() == 8
        assert pypto.DT_FP8E5M2.get_bit() == 8
        assert pypto.DT_HF8.get_bit() == 8

    def test_16bit_types(self):
        """Test data types that are 16 bits."""
        assert pypto.DT_INT16.get_bit() == 16
        assert pypto.DT_UINT16.get_bit() == 16
        assert pypto.DT_FP16.get_bit() == 16
        assert pypto.DT_BF16.get_bit() == 16

    def test_32bit_types(self):
        """Test data types that are 32 bits."""
        assert pypto.DT_INT32.get_bit() == 32
        assert pypto.DT_UINT32.get_bit() == 32
        assert pypto.DT_FP32.get_bit() == 32

    def test_64bit_types(self):
        """Test data types that are 64 bits."""
        assert pypto.DT_INT64.get_bit() == 64
        assert pypto.DT_UINT64.get_bit() == 64


class TestDataTypeString:
    """Test ToString() method."""

    def test_signed_integer_strings(self):
        """Test string representation of signed integer types."""
        assert pypto.DT_INT4.to_string() == "int4"
        assert pypto.DT_INT8.to_string() == "int8"
        assert pypto.DT_INT16.to_string() == "int16"
        assert pypto.DT_INT32.to_string() == "int32"
        assert pypto.DT_INT64.to_string() == "int64"

    def test_unsigned_integer_strings(self):
        """Test string representation of unsigned integer types."""
        assert pypto.DT_UINT4.to_string() == "uint4"
        assert pypto.DT_UINT8.to_string() == "uint8"
        assert pypto.DT_UINT16.to_string() == "uint16"
        assert pypto.DT_UINT32.to_string() == "uint32"
        assert pypto.DT_UINT64.to_string() == "uint64"

    def test_floating_point_strings(self):
        """Test string representation of floating point types."""
        assert pypto.DT_FP4.to_string() == "fp4"
        assert pypto.DT_FP8E4M3FN.to_string() == "fp8e4m3fn"
        assert pypto.DT_FP8E5M2.to_string() == "fp8e5m2"
        assert pypto.DT_FP16.to_string() == "fp16"
        assert pypto.DT_FP32.to_string() == "fp32"
        assert pypto.DT_BF16.to_string() == "bfloat16"

    def test_hybrid_float_strings(self):
        """Test string representation of Hisilicon float types."""
        assert pypto.DT_HF4.to_string() == "hf4"
        assert pypto.DT_HF8.to_string() == "hf8"

    def test_bool_string(self):
        """Test string representation of boolean type."""
        assert pypto.DT_BOOL.to_string() == "bool"


class TestDataTypePredicates:
    """Test type checking predicate methods."""

    def test_is_float(self):
        """Test is_float() correctly identifies floating point types."""
        # Floating point types
        assert pypto.DT_FP4.is_float() is True
        assert pypto.DT_FP8E4M3FN.is_float() is True
        assert pypto.DT_FP8E5M2.is_float() is True
        assert pypto.DT_FP16.is_float() is True
        assert pypto.DT_FP32.is_float() is True
        assert pypto.DT_BF16.is_float() is True
        assert pypto.DT_HF4.is_float() is True
        assert pypto.DT_HF8.is_float() is True

        # Non-floating point types
        assert pypto.DT_INT8.is_float() is False
        assert pypto.DT_INT32.is_float() is False
        assert pypto.DT_UINT8.is_float() is False
        assert pypto.DT_BOOL.is_float() is False

    def test_is_signed_int(self):
        """Test is_signed_int() correctly identifies signed integer types."""
        # Signed integer types
        assert pypto.DT_INT4.is_signed_int() is True
        assert pypto.DT_INT8.is_signed_int() is True
        assert pypto.DT_INT16.is_signed_int() is True
        assert pypto.DT_INT32.is_signed_int() is True
        assert pypto.DT_INT64.is_signed_int() is True

        # Non-signed integer types
        assert pypto.DT_UINT8.is_signed_int() is False
        assert pypto.DT_FP32.is_signed_int() is False
        assert pypto.DT_BOOL.is_signed_int() is False

    def test_is_unsigned_int(self):
        """Test is_unsigned_int() correctly identifies unsigned integer types."""
        # Unsigned integer types
        assert pypto.DT_UINT4.is_unsigned_int() is True
        assert pypto.DT_UINT8.is_unsigned_int() is True
        assert pypto.DT_UINT16.is_unsigned_int() is True
        assert pypto.DT_UINT32.is_unsigned_int() is True
        assert pypto.DT_UINT64.is_unsigned_int() is True

        # Non-unsigned integer types
        assert pypto.DT_INT8.is_unsigned_int() is False
        assert pypto.DT_FP32.is_unsigned_int() is False
        assert pypto.DT_BOOL.is_unsigned_int() is False

    def test_is_int(self):
        """Test is_int() correctly identifies any integer types."""
        # Integer types (both signed and unsigned)
        assert pypto.DT_INT4.is_int() is True
        assert pypto.DT_INT8.is_int() is True
        assert pypto.DT_INT16.is_int() is True
        assert pypto.DT_INT32.is_int() is True
        assert pypto.DT_INT64.is_int() is True
        assert pypto.DT_UINT4.is_int() is True
        assert pypto.DT_UINT8.is_int() is True
        assert pypto.DT_UINT16.is_int() is True
        assert pypto.DT_UINT32.is_int() is True
        assert pypto.DT_UINT64.is_int() is True

        # Non-integer types
        assert pypto.DT_FP4.is_int() is False
        assert pypto.DT_FP8E4M3FN.is_int() is False
        assert pypto.DT_FP8E5M2.is_int() is False
        assert pypto.DT_FP16.is_int() is False
        assert pypto.DT_FP32.is_int() is False
        assert pypto.DT_BF16.is_int() is False
        assert pypto.DT_HF4.is_int() is False
        assert pypto.DT_HF8.is_int() is False
        assert pypto.DT_BOOL.is_int() is False

    def test_type_predicates_mutual_exclusion(self):
        """Test that signed, unsigned, and floating point are mutually exclusive."""
        all_types = [
            DT_INT4,
            DT_INT8,
            DT_INT16,
            DT_INT32,
            DT_INT64,
            DT_FP4,
            DT_FP8E4M3FN,
            DT_FP8E5M2,
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
            if dtype.is_signed_int():
                assert not dtype.is_unsigned_int()

            # A type should not be both integer and floating point
            if dtype.is_int():
                assert not dtype.is_float()


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
        pypto.DT_FP8E4M3FN,
        pypto.DT_FP8E5M2,
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
            bit_size = dtype.get_bit()
            assert bit_size > 0, f"Type {dtype.to_string()} should have positive bit size"
            assert bit_size in [1, 4, 8, 16, 32, 64], f"Type {dtype.to_string()} should have valid bit size"

    def test_all_types_have_string_representation(self):
        """Test that all data types have a valid string representation."""

        for dtype in self.all_types:
            string_repr = dtype.to_string()
            assert string_repr != "unknown", f"Type {dtype} should have valid string representation"
            assert len(string_repr) > 0, f"Type {dtype} should have non-empty string representation"

    def test_all_types_classified(self):
        """Test that all data types are classified as either integer, float, or bool."""

        for dtype in self.all_types:
            is_integer = dtype.is_int()
            is_floating = dtype.is_float()
            is_boolean = dtype == pypto.DT_BOOL

            # Each type should be classified as at least one category
            # (bool is a special case that's neither int nor float in this classification)
            assert is_integer or is_floating or is_boolean, (
                f"Type {dtype.to_string()} should be classified as int, float, or bool"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
