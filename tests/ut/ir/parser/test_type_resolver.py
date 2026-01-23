# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TypeResolver."""

import ast

import pytest
from pypto.ir.parser.diagnostics import ParserTypeError
from pypto.ir.parser.type_resolver import TypeResolver
from pypto.pypto_core import DataType, ir


class TestTypeResolver:
    """Tests for TypeResolver class."""

    def test_resolve_tensor_type_subscript(self):
        """Test resolving tensor type with subscript notation."""
        resolver = TypeResolver()

        # Parse: pl.Tensor[[64, 128], pl.FP16]
        code = "pl.Tensor[[64, 128], pl.FP16]"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        # Shape elements are ConstInt expressions
        assert result.dtype == DataType.FP16

    def test_resolve_tensor_type_different_dtypes(self):
        """Test resolving tensor types with different data types."""
        resolver = TypeResolver()

        test_cases = [
            ("pl.Tensor[[64], pl.FP32]", DataType.FP32),
            ("pl.Tensor[[32, 64], pl.INT32]", DataType.INT32),
            ("pl.Tensor[[1, 2, 3], pl.FP16]", DataType.FP16),
        ]

        for code, expected_dtype in test_cases:
            node = ast.parse(code, mode="eval").body
            result = resolver.resolve_type(node)

            assert isinstance(result, ir.TensorType)
            assert result.dtype == expected_dtype

    def test_resolve_dtype_attribute(self):
        """Test resolving dtype from attribute access."""
        resolver = TypeResolver()

        # Parse: pl.FP16
        code = "pl.FP16"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_dtype(node)
        assert result == DataType.FP16

    def test_resolve_dtype_all_types(self):
        """Test all supported dtype values."""
        resolver = TypeResolver()

        dtypes = [
            ("pl.FP16", DataType.FP16),
            ("pl.FP32", DataType.FP32),
            ("pl.INT32", DataType.INT32),
            ("pl.INT64", DataType.INT64),
            ("pl.BOOL", DataType.BOOL),
        ]

        for code, expected in dtypes:
            node = ast.parse(code, mode="eval").body
            result = resolver.resolve_dtype(node)
            assert result == expected

    def test_resolve_invalid_dtype(self):
        """Test error on invalid dtype."""
        resolver = TypeResolver()

        code = "pl.INVALID_TYPE"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="Unknown dtype"):
            resolver.resolve_dtype(node)

    def test_resolve_invalid_tensor_syntax(self):
        """Test error on invalid tensor syntax."""
        resolver = TypeResolver()

        # Missing dtype
        code = "pl.Tensor[[64, 128]]"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="requires"):
            resolver.resolve_type(node)

    def test_parse_shape_list(self):
        """Test parsing shape from list literal."""
        resolver = TypeResolver()

        code = "[64, 128, 256]"
        node = ast.parse(code, mode="eval").body

        shape = resolver._parse_shape(node)
        assert len(shape) == 3
        assert shape == [64, 128, 256]

    def test_parse_shape_tuple(self):
        """Test parsing shape from tuple literal."""
        resolver = TypeResolver()

        code = "(32, 64)"
        node = ast.parse(code, mode="eval").body

        shape = resolver._parse_shape(node)
        assert len(shape) == 2
        assert shape == [32, 64]

    def test_parse_shape_invalid(self):
        """Test error on invalid shape."""
        resolver = TypeResolver()

        # Non-constant shape dimension
        code = "x"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="must be tuple or list"):
            resolver._parse_shape(node)
