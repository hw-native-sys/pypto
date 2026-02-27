# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TypeConverter class."""

import pytest
from pypto import DataType, codegen
from pypto.ir import PipeType


class TestDataTypeConversion:
    """Test DataType to C++ type conversion."""

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (DataType.FP32, "float"),
            (DataType.FP16, "half"),
            (DataType.INT32, "int32_t"),
            (DataType.INT64, "int64_t"),
            (DataType.BOOL, "bool"),
            (DataType.BF16, "bfloat16_t"),
        ],
    )
    def test_convert_data_type(self, dtype, expected):
        """Test DataType to C++ type string conversion."""
        assert dtype.to_c_type_string() == expected


class TestShapeGeneration:
    """Test Shape type generation."""

    @pytest.mark.parametrize(
        "dims,expected",
        [
            ([128, 64], "Shape<1, 1, 1, 128, 64>"),
            ([256], "Shape<1, 1, 1, 1, 256>"),
            ([16, 128, 64], "Shape<1, 1, 16, 128, 64>"),
        ],
    )
    def test_generate_shape(self, dims, expected):
        """Test shape generation with padding to 5D."""
        converter = codegen.TypeConverter()
        assert converter.GenerateShapeType(dims) == expected


class TestStrideGeneration:
    """Test Stride type generation."""

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ([128, 64], "Stride<1, 1, 1, -1, -1>"),  # Dynamic strides from runtime
            ([256], "Stride<1, 1, 1, 1, -1>"),  # Dynamic stride for last dimension
            (
                [16, 128, 64],
                "Stride<1, 1, -1, -1, -1>",
            ),  # Dynamic strides from runtime
        ],
    )
    def test_generate_stride(self, shape, expected):
        """Test stride generation (dynamic strides) with padding to 5D."""
        converter = codegen.TypeConverter()
        assert converter.GenerateStrideType(shape) == expected


class TestPipeTypeConversion:
    """Test PipeType to C++ type conversion."""

    @pytest.mark.parametrize(
        "pipe_type,expected",
        [
            (PipeType.MTE1, "PIPE_MTE1"),
            (PipeType.MTE2, "PIPE_MTE2"),
            (PipeType.MTE3, "PIPE_MTE3"),
            (PipeType.M, "PIPE_M"),
            (PipeType.V, "PIPE_V"),
            (PipeType.S, "PIPE_S"),
            (PipeType.FIX, "PIPE_FIX"),
            (PipeType.ALL, "PIPE_ALL"),
        ],
    )
    def test_convert_pipe_type(self, pipe_type, expected):
        """Test PipeType to C++ string conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertPipeType(pipe_type) == expected


class TestEventIdConversion:
    """Test event ID to C++ string conversion."""

    @pytest.mark.parametrize("event_id,expected", [(0, "EVENT_ID0"), (1, "EVENT_ID1"), (7, "EVENT_ID7")])
    def test_convert_event_id(self, event_id, expected):
        """Test valid event ID conversion."""
        converter = codegen.TypeConverter()
        assert converter.ConvertEventId(event_id) == expected

    @pytest.mark.parametrize("invalid_id", [-1, 8])
    def test_convert_event_id_invalid(self, invalid_id):
        """Test event ID with invalid value raises error."""
        converter = codegen.TypeConverter()
        with pytest.raises(ValueError, match=rf"Event ID must be in range \[0, 7\].*got {invalid_id}"):
            converter.ConvertEventId(invalid_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
