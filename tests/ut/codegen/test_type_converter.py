# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TypeConverter class."""

from pypto import DataType
from pypto.pypto_core import codegen


class TestDataTypeConversion:
    """Test DataType to C++ type conversion."""

    def test_convert_fp32(self):
        """Test FP32 conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.FP32) == "float"

    def test_convert_fp16(self):
        """Test FP16 conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.FP16) == "half"

    def test_convert_int32(self):
        """Test INT32 conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.INT32) == "int32_t"

    def test_convert_int64(self):
        """Test INT64 conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.INT64) == "int64_t"

    def test_convert_bool(self):
        """Test BOOL conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.BOOL) == "bool"

    def test_convert_bf16(self):
        """Test BF16 conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertDataType(DataType.BF16) == "bfloat16"


class TestShapeGeneration:
    """Test Shape type generation."""

    def test_generate_shape_2d(self):
        """Test 2D shape generation with padding."""
        converter = codegen.TypeConverter()
        shape = converter.GenerateShapeType([128, 64])
        assert shape == "Shape<1, 1, 1, 128, 64>"

    def test_generate_shape_1d(self):
        """Test 1D shape generation with padding."""
        converter = codegen.TypeConverter()
        shape = converter.GenerateShapeType([256])
        assert shape == "Shape<1, 1, 1, 1, 256>"

    def test_generate_shape_3d(self):
        """Test 3D shape generation with padding."""
        converter = codegen.TypeConverter()
        shape = converter.GenerateShapeType([16, 128, 64])
        assert shape == "Shape<1, 1, 16, 128, 64>"


class TestStrideGeneration:
    """Test Stride type generation."""

    def test_generate_stride_2d(self):
        """Test 2D stride generation (row-major)."""
        converter = codegen.TypeConverter()
        stride = converter.GenerateStrideType([128, 64])
        # Row-major: stride[0] = 64, stride[1] = 1
        assert stride == "Stride<1, 1, 1, 64, 1>"

    def test_generate_stride_1d(self):
        """Test 1D stride generation."""
        converter = codegen.TypeConverter()
        stride = converter.GenerateStrideType([256])
        assert stride == "Stride<1, 1, 1, 1, 1>"

    def test_generate_stride_3d(self):
        """Test 3D stride generation (row-major)."""
        converter = codegen.TypeConverter()
        stride = converter.GenerateStrideType([16, 128, 64])
        # Row-major: stride[0] = 128*64, stride[1] = 64, stride[2] = 1
        assert stride == "Stride<1, 1, 8192, 64, 1>"
