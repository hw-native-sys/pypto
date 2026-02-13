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

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.type_resolver import TypeResolver
from pypto.language.typing.dynamic import DynVar
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


class TestTupleTypeResolver:
    """Tests for tuple[T1, T2, ...] return type resolution."""

    def test_resolve_tuple_two_tensors(self):
        """Test resolving tuple[pl.Tensor[...], pl.Tensor[...]]."""
        resolver = TypeResolver()

        code = "tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[128], pl.FP16]]"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_type(node)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], ir.TensorType)
        assert result[0].dtype == DataType.FP32
        assert isinstance(result[1], ir.TensorType)
        assert result[1].dtype == DataType.FP16

    def test_resolve_tuple_mixed_types(self):
        """Test resolving tuple with mixed Tensor and Scalar types."""
        resolver = TypeResolver()

        code = "tuple[pl.Tensor[[32, 64], pl.FP32], pl.Scalar[pl.INT64]]"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_type(node)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], ir.TensorType)
        assert isinstance(result[1], ir.ScalarType)
        assert result[1].dtype == DataType.INT64

    def test_resolve_tuple_single_element(self):
        """Test resolving tuple with a single element."""
        resolver = TypeResolver()

        code = "tuple[pl.Tensor[[64], pl.FP32]]"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_type(node)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ir.TensorType)

    def test_resolve_nested_tuple_error(self):
        """Test that nested tuple types raise an error."""
        resolver = TypeResolver()

        code = "tuple[tuple[pl.Tensor[[64], pl.FP32]], pl.Tensor[[128], pl.FP16]]"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="Nested tuple types"):
            resolver.resolve_type(node)


class TestDynamicShapeResolution:
    """Tests for dynamic shape dimension resolution."""

    # --- Compile-time dynamic (int variables from closure) ---

    def test_parse_shape_with_int_variable(self):
        """Int variable from closure resolves to constant dimension."""
        resolver = TypeResolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("[rows, cols]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_int_var_and_literal_mixed(self):
        """Mix of int literal and int variable in same shape."""
        resolver = TypeResolver(closure_vars={"rows": 128})
        node = ast.parse("[rows, 64]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_int_var_tuple_syntax(self):
        """Int variables work with tuple syntax too."""
        resolver = TypeResolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("(rows, cols)", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    # --- Runtime dynamic (DynVar from pl.dynamic) ---

    def test_parse_shape_with_dynvar(self):
        """DynVar creates ir.Var nodes in shape."""
        resolver = TypeResolver(closure_vars={"M": DynVar("M"), "N": DynVar("N")})
        node = ast.parse("[M, N]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert len(shape) == 2
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert isinstance(shape[1], ir.Var)
        assert shape[1].name == "N"

    def test_parse_shape_dynvar_and_literal_mixed(self):
        """Mix of DynVar and int literal in same shape."""
        resolver = TypeResolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("[M, 128]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert shape[1] == 128

    def test_parse_shape_dynvar_and_int_var_mixed(self):
        """Mix of DynVar and int variable in same shape."""
        resolver = TypeResolver(closure_vars={"M": DynVar("M"), "cols": 64})
        node = ast.parse("[M, cols]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert shape[1] == 64

    def test_dynvar_has_int64_scalar_type(self):
        """DynVar creates Var with ScalarType(INT64)."""
        resolver = TypeResolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("[M]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert isinstance(shape[0].type, ir.ScalarType)
        assert shape[0].type.dtype == DataType.INT64

    # --- Scope lookup (Scalar IR vars in function body) ---

    def test_parse_shape_with_scope_variable(self):
        """Scalar variable from parser scope used in inline annotation."""
        mock_var = ir.Var("q_tile", ir.ScalarType(DataType.UINT64), ir.Span.unknown())
        scope = {"q_tile": mock_var}
        resolver = TypeResolver(scope_lookup=lambda name: scope.get(name))
        node = ast.parse("[q_tile, 128]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape[0] is mock_var
        assert shape[1] == 128

    def test_closure_vars_take_precedence_over_scope(self):
        """Closure variables are checked before parser scope."""
        scope = {"x": ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())}
        resolver = TypeResolver(closure_vars={"x": 42}, scope_lookup=lambda name: scope.get(name))
        node = ast.parse("[x]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [42]

    # --- _to_ir_shape normalization ---

    def test_to_ir_shape_pure_int(self):
        """Pure int list passes through unchanged."""
        resolver = TypeResolver()
        result = resolver._to_ir_shape([64, 128])
        assert result == [64, 128]

    def test_to_ir_shape_mixed_converts_all_to_expr(self):
        """Mixed list converts all ints to ConstInt."""
        resolver = TypeResolver()
        var = ir.Var("M", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        result = resolver._to_ir_shape([var, 128])
        assert len(result) == 2
        assert result[0] is var
        assert isinstance(result[1], ir.ConstInt)
        assert result[1].value == 128

    # --- Full type resolution with dynamic shapes ---

    def test_resolve_tensor_type_with_int_vars(self):
        """Full TensorType resolution with int closure variables."""
        resolver = TypeResolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("pl.Tensor[[rows, cols], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.dtype == DataType.FP32

    def test_resolve_tensor_type_with_dynvar(self):
        """Full TensorType resolution with DynVar."""
        resolver = TypeResolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("pl.Tensor[[M, 128], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[0], ir.Var)
        assert result.shape[0].name == "M"

    def test_resolve_tile_type_with_dynvar(self):
        """TileType also supports dynamic shapes."""
        resolver = TypeResolver(closure_vars={"N": DynVar("N")})
        node = ast.parse("pl.Tile[[N, 64], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TileType)
        assert isinstance(result.shape[0], ir.Var)

    # --- Error cases ---

    def test_parse_shape_undefined_variable(self):
        """Undefined variable name raises error."""
        resolver = TypeResolver(closure_vars={})
        node = ast.parse("[undefined_var]", mode="eval").body
        with pytest.raises(ParserTypeError, match="Unknown shape variable"):
            resolver._parse_shape(node)

    def test_parse_shape_invalid_variable_type(self):
        """Non-int, non-DynVar closure variable raises error."""
        resolver = TypeResolver(closure_vars={"x": 3.14})
        node = ast.parse("[x]", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be int or pl.dynamic"):
            resolver._parse_shape(node)

    def test_parse_shape_string_variable_type(self):
        """String variable raises error."""
        resolver = TypeResolver(closure_vars={"x": "hello"})
        node = ast.parse("[x]", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be int or pl.dynamic"):
            resolver._parse_shape(node)


class TestDynamicShapeIntegration:
    """End-to-end tests: decorator + dynamic shapes."""

    # --- Compile-time dynamic with @pl.function ---

    def test_function_with_int_variable_shape(self):
        """@pl.function with int variables from enclosing scope."""
        rows, cols = 128, 64

        @pl.function
        def func(x: pl.Tensor[[rows, cols], pl.FP32]) -> pl.Tensor[[rows, cols], pl.FP32]:
            return x

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2

    # --- Compile-time dynamic with @pl.program ---

    def test_program_with_int_variable_shape(self):
        """@pl.program with int variables from enclosing scope."""
        rows, cols = 256, 128

        @pl.program
        class MyProgram:
            @pl.function
            def add(self, a: pl.Tensor[[rows, cols], pl.FP32]) -> pl.Tensor[[rows, cols], pl.FP32]:
                return a

        assert isinstance(MyProgram, ir.Program)
        func = list(MyProgram.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert param_type.shape[0] == rows
        assert param_type.shape[1] == cols

    # --- Runtime dynamic with @pl.function ---

    def test_function_with_dynvar_shape(self):
        """@pl.function with pl.dynamic() variables."""
        M = pl.dynamic("M")

        @pl.function
        def func(x: pl.Tensor[[M, 128], pl.FP32]) -> pl.Tensor[[M, 128], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert isinstance(param_type.shape[0], ir.Var)
        assert param_type.shape[0].name == "M"
        # Second dim is still a ConstInt
        assert isinstance(param_type.shape[1], ir.ConstInt)
        assert param_type.shape[1].value == 128

    def test_function_with_multiple_dynvars(self):
        """@pl.function with multiple pl.dynamic() variables."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")
        K = pl.dynamic("K")

        @pl.function
        def func(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[K, N], pl.FP32],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            return a

        a_type = func.params[0].type
        b_type = func.params[1].type
        assert isinstance(a_type, ir.TensorType)
        assert isinstance(b_type, ir.TensorType)
        assert isinstance(a_type.shape[0], ir.Var)
        assert isinstance(a_type.shape[1], ir.Var)
        assert isinstance(b_type.shape[0], ir.Var)
        assert isinstance(b_type.shape[1], ir.Var)
        assert a_type.shape[0].name == "M"
        assert a_type.shape[1].name == "K"
        assert b_type.shape[0].name == "K"
        assert b_type.shape[1].name == "N"

    def test_function_dynvar_return_type(self):
        """Return type also supports dynamic shapes."""
        M = pl.dynamic("M")

        @pl.function
        def func(x: pl.Tensor[[M, 64], pl.FP32]) -> pl.Tensor[[M, 64], pl.FP32]:
            return x

        ret_type = func.return_types[0]
        assert isinstance(ret_type, ir.TensorType)
        assert isinstance(ret_type.shape[0], ir.Var)
        assert ret_type.shape[0].name == "M"

    # --- Runtime dynamic with @pl.program ---

    def test_program_with_dynvar_shape(self):
        """@pl.program with pl.dynamic() variables."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.program
        class MyProgram:
            @pl.function
            def process(self, x: pl.Tensor[[M, N], pl.FP32]) -> pl.Tensor[[M, N], pl.FP32]:
                return x

        func = list(MyProgram.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert isinstance(param_type.shape[0], ir.Var)
        assert param_type.shape[0].name == "M"

    # --- Parametrized testing (issue #163 primary use case) ---

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128), (256, 256)])
    def test_parametrized_shapes(self, rows, cols):
        """pytest.mark.parametrize with variable shapes."""

        @pl.function
        def func(x: pl.Tensor[[rows, cols], pl.FP32]) -> pl.Tensor[[rows, cols], pl.FP32]:
            return x

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
