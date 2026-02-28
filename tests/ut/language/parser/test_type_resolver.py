# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TypeResolver."""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking the annotations and kwargs inside @pl.function bodies.
# pyright: reportUndefinedVariable=false

import ast
from typing import TYPE_CHECKING, Any

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.expr_evaluator import ExprEvaluator
from pypto.language.parser.type_resolver import TypeResolver
from pypto.language.typing.dynamic import DynVar

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    closure_vars: dict | None = None, scope_lookup: "Callable[[str], Any | None] | None" = None
) -> TypeResolver:
    """Create a TypeResolver with ExprEvaluator from closure_vars."""
    ev = ExprEvaluator(closure_vars=closure_vars or {})
    return TypeResolver(expr_evaluator=ev, scope_lookup=scope_lookup)


class TestTypeResolver:
    """Tests for TypeResolver class."""

    def test_resolve_tensor_type_subscript(self):
        """Test resolving tensor type with subscript notation."""
        resolver = _make_resolver()

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
        resolver = _make_resolver()

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
        resolver = _make_resolver()

        # Parse: pl.FP16
        code = "pl.FP16"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_dtype(node)
        assert result == DataType.FP16

    def test_resolve_dtype_all_types(self):
        """Test all supported dtype values."""
        resolver = _make_resolver()

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
        resolver = _make_resolver()

        code = "pl.INVALID_TYPE"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="Unknown dtype"):
            resolver.resolve_dtype(node)

    def test_resolve_invalid_tensor_syntax(self):
        """Test error on invalid tensor syntax."""
        resolver = _make_resolver()

        # Missing dtype
        code = "pl.Tensor[[64, 128]]"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="requires"):
            resolver.resolve_type(node)

    def test_parse_shape_list(self):
        """Test parsing shape from list literal."""
        resolver = _make_resolver()

        code = "[64, 128, 256]"
        node = ast.parse(code, mode="eval").body

        shape = resolver._parse_shape(node)
        assert len(shape) == 3
        assert shape == [64, 128, 256]

    def test_parse_shape_tuple(self):
        """Test parsing shape from tuple literal."""
        resolver = _make_resolver()

        code = "(32, 64)"
        node = ast.parse(code, mode="eval").body

        shape = resolver._parse_shape(node)
        assert len(shape) == 2
        assert shape == [32, 64]

    def test_parse_shape_invalid(self):
        """Test error on invalid shape (not a list, tuple, or known variable)."""
        resolver = _make_resolver()

        # Bare variable name not in closure_vars
        code = "x"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="Unknown shape variable"):
            resolver._parse_shape(node)


class TestTupleTypeResolver:
    """Tests for tuple[T1, T2, ...] return type resolution."""

    def test_resolve_tuple_two_tensors(self):
        """Test resolving tuple[pl.Tensor[...], pl.Tensor[...]]."""
        resolver = _make_resolver()

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
        resolver = _make_resolver()

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
        resolver = _make_resolver()

        code = "tuple[pl.Tensor[[64], pl.FP32]]"
        node = ast.parse(code, mode="eval").body

        result = resolver.resolve_type(node)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ir.TensorType)

    def test_resolve_nested_tuple_error(self):
        """Test that nested tuple types raise an error."""
        resolver = _make_resolver()

        code = "tuple[tuple[pl.Tensor[[64], pl.FP32]], pl.Tensor[[128], pl.FP16]]"
        node = ast.parse(code, mode="eval").body

        with pytest.raises(ParserTypeError, match="Nested tuple types"):
            resolver.resolve_type(node)


class TestDynamicShapeResolution:
    """Tests for dynamic shape dimension resolution."""

    # --- Compile-time dynamic (int variables from closure) ---

    def test_parse_shape_with_int_variable(self):
        """Int variable from closure resolves to constant dimension."""
        resolver = _make_resolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("[rows, cols]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_int_var_and_literal_mixed(self):
        """Mix of int literal and int variable in same shape."""
        resolver = _make_resolver(closure_vars={"rows": 128})
        node = ast.parse("[rows, 64]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_int_var_tuple_syntax(self):
        """Int variables work with tuple syntax too."""
        resolver = _make_resolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("(rows, cols)", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    # --- Runtime dynamic (DynVar from pl.dynamic) ---

    def test_parse_shape_with_dynvar(self):
        """DynVar creates ir.Var nodes in shape."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M"), "N": DynVar("N")})
        node = ast.parse("[M, N]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert len(shape) == 2
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert isinstance(shape[1], ir.Var)
        assert shape[1].name == "N"

    def test_parse_shape_dynvar_and_literal_mixed(self):
        """Mix of DynVar and int literal in same shape."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("[M, 128]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert shape[1] == 128

    def test_parse_shape_dynvar_and_int_var_mixed(self):
        """Mix of DynVar and int variable in same shape."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M"), "cols": 64})
        node = ast.parse("[M, cols]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert shape[1] == 64

    def test_dynvar_has_int64_scalar_type(self):
        """DynVar creates Var with ScalarType(INT64)."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M")})
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
        resolver = _make_resolver(scope_lookup=lambda name: scope.get(name))
        node = ast.parse("[q_tile, 128]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape[0] is mock_var
        assert shape[1] == 128

    def test_closure_vars_take_precedence_over_scope(self):
        """Closure variables are checked before parser scope."""
        scope = {"x": ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())}
        resolver = _make_resolver(closure_vars={"x": 42}, scope_lookup=lambda name: scope.get(name))
        node = ast.parse("[x]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [42]

    # --- _to_ir_shape normalization ---

    def test_to_ir_shape_pure_int(self):
        """Pure int list passes through unchanged."""
        resolver = _make_resolver()
        result = resolver._to_ir_shape([64, 128])
        assert result == [64, 128]

    def test_to_ir_shape_mixed_converts_all_to_expr(self):
        """Mixed list converts all ints to ConstInt."""
        resolver = _make_resolver()
        var = ir.Var("M", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        result = resolver._to_ir_shape([var, 128])
        assert len(result) == 2
        assert result[0] is var
        assert isinstance(result[1], ir.ConstInt)
        assert result[1].value == 128

    # --- Full type resolution with dynamic shapes ---

    def test_resolve_tensor_type_with_int_vars(self):
        """Full TensorType resolution with int closure variables."""
        resolver = _make_resolver(closure_vars={"rows": 128, "cols": 64})
        node = ast.parse("pl.Tensor[[rows, cols], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.dtype == DataType.FP32

    def test_resolve_tensor_type_with_dynvar(self):
        """Full TensorType resolution with DynVar."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("pl.Tensor[[M, 128], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[0], ir.Var)
        assert result.shape[0].name == "M"

    def test_resolve_tile_type_with_dynvar(self):
        """TileType also supports dynamic shapes."""
        resolver = _make_resolver(closure_vars={"N": DynVar("N")})
        node = ast.parse("pl.Tile[[N, 64], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TileType)
        assert isinstance(result.shape[0], ir.Var)

    # --- Error cases ---

    def test_parse_shape_undefined_variable(self):
        """Undefined variable name raises error."""
        resolver = _make_resolver(closure_vars={})
        node = ast.parse("[undefined_var]", mode="eval").body
        with pytest.raises(ParserTypeError, match="Unknown shape variable"):
            resolver._parse_shape(node)

    def test_parse_shape_invalid_variable_type(self):
        """Non-int, non-DynVar closure variable raises error."""
        resolver = _make_resolver(closure_vars={"x": 3.14})
        node = ast.parse("[x]", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be int or pl.dynamic"):
            resolver._parse_shape(node)

    def test_parse_shape_string_variable_type(self):
        """String variable raises error."""
        resolver = _make_resolver(closure_vars={"x": "hello"})
        node = ast.parse("[x]", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be int or pl.dynamic"):
            resolver._parse_shape(node)

    # --- Shape as a list/tuple variable (issue #205) ---

    def test_parse_shape_list_variable(self):
        """List variable from closure resolves to shape dimensions."""
        resolver = _make_resolver(closure_vars={"shape": [128, 64]})
        node = ast.parse("shape", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_tuple_variable(self):
        """Tuple variable from closure resolves to shape dimensions."""
        resolver = _make_resolver(closure_vars={"shape": (32, 64)})
        node = ast.parse("shape", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [32, 64]

    def test_parse_shape_list_variable_with_dynvar(self):
        """List variable containing DynVar creates ir.Var nodes."""
        resolver = _make_resolver(closure_vars={"shape": [DynVar("M"), DynVar("N")]})
        node = ast.parse("shape", mode="eval").body
        shape = resolver._parse_shape(node)
        assert len(shape) == 2
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert isinstance(shape[1], ir.Var)
        assert shape[1].name == "N"

    def test_parse_shape_list_variable_mixed_int_dynvar(self):
        """List variable with mixed int and DynVar."""
        resolver = _make_resolver(closure_vars={"shape": [DynVar("M"), 128]})
        node = ast.parse("shape", mode="eval").body
        shape = resolver._parse_shape(node)
        assert isinstance(shape[0], ir.Var)
        assert shape[0].name == "M"
        assert shape[1] == 128

    def test_parse_shape_unknown_variable(self):
        """Unknown shape variable raises error."""
        resolver = _make_resolver(closure_vars={})
        node = ast.parse("shape", mode="eval").body
        with pytest.raises(ParserTypeError, match="Unknown shape variable"):
            resolver._parse_shape(node)

    def test_parse_shape_non_list_variable(self):
        """Non-list/tuple variable raises error."""
        resolver = _make_resolver(closure_vars={"shape": 42})
        node = ast.parse("shape", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be a list or tuple"):
            resolver._parse_shape(node)

    def test_parse_shape_list_variable_invalid_element(self):
        """List variable with invalid element type raises error."""
        resolver = _make_resolver(closure_vars={"shape": [128, "bad"]})
        node = ast.parse("shape", mode="eval").body
        with pytest.raises(ParserTypeError, match="element 1 must be int"):
            resolver._parse_shape(node)

    # --- Dtype from closure variable (issue #205) ---

    def test_resolve_dtype_from_closure(self):
        """DataType variable from closure resolves to dtype."""
        resolver = _make_resolver(closure_vars={"dtype": DataType.FP32})
        node = ast.parse("dtype", mode="eval").body
        result = resolver.resolve_dtype(node)
        assert result == DataType.FP32

    def test_resolve_dtype_closure_invalid_type(self):
        """Non-DataType closure variable raises error."""
        resolver = _make_resolver(closure_vars={"dtype": "FP32"})
        node = ast.parse("dtype", mode="eval").body
        with pytest.raises(ParserTypeError, match="must be a DataType"):
            resolver.resolve_dtype(node)

    def test_resolve_tensor_with_shape_and_dtype_variables(self):
        """Full TensorType resolution with shape list and dtype from closure."""
        resolver = _make_resolver(closure_vars={"shape": [128, 64], "dtype": DataType.FP16})
        node = ast.parse("pl.Tensor[shape, dtype]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.dtype == DataType.FP16

    def test_resolve_tile_with_shape_variable(self):
        """TileType also supports shape list variable."""
        resolver = _make_resolver(closure_vars={"tile_shape": [64, 64]})
        node = ast.parse("pl.Tile[tile_shape, pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TileType)
        assert len(result.shape) == 2

    # --- Expression-based shapes (new with ExprEvaluator) ---

    def test_parse_shape_arithmetic_dim(self):
        """Arithmetic expression in shape dimension."""
        resolver = _make_resolver(closure_vars={"base": 64})
        node = ast.parse("[base * 2, base]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_parse_shape_len_expression(self):
        """len() call in shape dimension."""
        resolver = _make_resolver(closure_vars={"data": [1, 2, 3, 4]})
        node = ast.parse("[len(data), 64]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [4, 64]

    def test_parse_shape_subscript_expression(self):
        """Subscript access in shape dimension."""
        resolver = _make_resolver(closure_vars={"dims": [128, 64, 32]})
        node = ast.parse("[dims[0], dims[1]]", mode="eval").body
        shape = resolver._parse_shape(node)
        assert shape == [128, 64]

    def test_resolve_tensor_with_expression_dims(self):
        """Full TensorType resolution with expression-based dimensions."""
        resolver = _make_resolver(closure_vars={"base": 64})
        node = ast.parse("pl.Tensor[[base * 2, base], pl.FP32]", mode="eval").body
        result = resolver.resolve_type(node)
        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.dtype == DataType.FP32


class TestDynamicShapeIntegration:
    """End-to-end tests: decorator + dynamic shapes."""

    # --- Compile-time dynamic with @pl.function ---

    def test_function_with_int_variable_shape(self):
        """@pl.function with int variables from enclosing scope."""
        rows, cols = 128, 64

        @pl.function
        def func(
            x: pl.Tensor[[rows, cols], pl.FP32],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
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
        def func(
            x: pl.Tensor[[rows, cols], pl.FP32],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            return x

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2

    # --- Shape/dtype as variables (issue #205) ---

    def test_function_with_shape_variable(self):
        """@pl.function with shape as a list variable (issue #205)."""
        shape = [128, 128]
        dtype = pl.FP32

        @pl.function
        def func(t: pl.Tensor[shape, dtype]) -> pl.Tensor[shape, dtype]:
            return t

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert param_type.dtype == DataType.FP32

    def test_function_with_multiple_shape_variables(self):
        """@pl.function with different shape variables per param (issue #205 pattern)."""
        tensor_shape = [128, 128]
        tile_shape = [64, 64]
        dtype = pl.FP32

        @pl.function
        def func(
            t: pl.Tensor[tensor_shape, dtype], tile: pl.Tensor[tile_shape, dtype]
        ) -> pl.Tensor[tensor_shape, dtype]:
            return t

        assert isinstance(func, ir.Function)
        assert isinstance(func.params[0].type, ir.TensorType)
        assert len(func.params[0].type.shape) == 2
        assert isinstance(func.params[1].type, ir.TensorType)
        assert len(func.params[1].type.shape) == 2

    def test_program_with_shape_variable(self):
        """@pl.program with shape as a list variable."""
        shape = [256, 128]
        dtype = pl.FP16

        @pl.program
        class MyProgram:
            @pl.function
            def process(self, x: pl.Tensor[shape, dtype]) -> pl.Tensor[shape, dtype]:
                return x

        func = list(MyProgram.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert param_type.dtype == DataType.FP16

    @pytest.mark.parametrize("shape", [[64, 64], [128, 128], [256, 256]])
    def test_parametrized_shape_variable(self, shape):
        """pytest.mark.parametrize with shape as a list variable."""

        @pl.function
        def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
            return x

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2


class TestExpressionShapePatterns:
    """User patterns: computed shapes and expressions in type annotations."""

    def test_arithmetic_shape_dims(self):
        """User computes shape dims from a base size."""
        base = 64

        @pl.function
        def func(
            x: pl.Tensor[[base * 2, base], pl.FP32],
        ) -> pl.Tensor[[base * 2, base], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.shape == [128, 64]

    def test_floor_division_in_shape(self):
        """User splits a dimension with //."""
        total = 256

        @pl.function
        def func(
            x: pl.Tensor[[total // 4, total], pl.FP32],
        ) -> pl.Tensor[[total // 4, total], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.shape == [64, 256]

    def test_multi_variable_arithmetic(self):
        """User combines multiple variables in shape expressions."""
        batch = 4
        heads = 8
        seq_len = 128

        @pl.function
        def func(
            x: pl.Tensor[[batch * heads, seq_len], pl.FP16],
        ) -> pl.Tensor[[batch * heads, seq_len], pl.FP16]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.shape == [32, 128]

    def test_shape_from_dict_config(self):
        """User stores config in a dict, uses subscript for dims."""
        config = {"rows": 128, "cols": 64}

        @pl.function
        def func(
            x: pl.Tensor[[config["rows"], config["cols"]], pl.FP32],  # noqa: F821
        ) -> pl.Tensor[[config["rows"], config["cols"]], pl.FP32]:  # noqa: F821
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.shape == [128, 64]

    def test_shape_from_list_indexing(self):
        """User picks dims from a list of predefined sizes."""
        sizes = [32, 64, 128, 256]

        @pl.function
        def func(
            x: pl.Tensor[[sizes[2], sizes[1]], pl.FP32],
        ) -> pl.Tensor[[sizes[2], sizes[1]], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.shape == [128, 64]

    def test_tuple_variable_as_shape(self):
        """User passes shape as a tuple (not list)."""
        shape = (128, 64)

        @pl.function
        def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2

    def test_1d_shape_variable(self):
        """User has a 1D tensor shape."""
        shape = [256]

        @pl.function
        def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 1

    def test_3d_shape_variable(self):
        """User has a 3D tensor shape."""
        batch, rows, cols = 4, 128, 64

        @pl.function
        def func(
            x: pl.Tensor[[batch, rows, cols], pl.FP32],
        ) -> pl.Tensor[[batch, rows, cols], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 3

    def test_different_shapes_per_param(self):
        """User has different shapes for different parameters."""
        input_shape = [256, 128]
        output_shape = [256, 64]

        @pl.function
        def func(
            x: pl.Tensor[input_shape, pl.FP32],
        ) -> pl.Tensor[output_shape, pl.FP32]:
            return x

        in_type = func.params[0].type
        out_type = func.return_types[0]
        assert isinstance(in_type, ir.TensorType)
        assert isinstance(out_type, ir.TensorType)
        assert len(in_type.shape) == 2
        assert len(out_type.shape) == 2

    def test_different_dtypes_per_param(self):
        """User has different dtypes for input and output."""
        dtype_in = pl.FP16
        dtype_out = pl.FP32

        @pl.function
        def func(
            x: pl.Tensor[[128, 64], dtype_in],
        ) -> pl.Tensor[[128, 64], dtype_out]:
            return x

        in_type = func.params[0].type
        out_type = func.return_types[0]
        assert isinstance(in_type, ir.TensorType)
        assert isinstance(out_type, ir.TensorType)
        assert in_type.dtype == DataType.FP16
        assert out_type.dtype == DataType.FP32

    @pytest.mark.parametrize(
        "dtype",
        [pl.FP16, pl.FP32, pl.BF16, pl.INT32],
    )
    def test_parametrized_dtype_variable(self, dtype):
        """User parametrizes dtype across test cases."""

        @pl.function
        def func(x: pl.Tensor[[64, 64], dtype]) -> pl.Tensor[[64, 64], dtype]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.dtype == dtype

    @pytest.mark.parametrize(
        "rows,cols,dtype",
        [
            (64, 64, pl.FP16),
            (128, 256, pl.FP32),
            (32, 32, pl.BF16),
        ],
    )
    def test_parametrized_shape_and_dtype(self, rows, cols, dtype):
        """User parametrizes both shape and dtype together."""

        @pl.function
        def func(
            x: pl.Tensor[[rows, cols], dtype],
        ) -> pl.Tensor[[rows, cols], dtype]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert param_type.dtype == dtype


class TestClosureVarsInFunctionBody:
    """User patterns: closure variables used inside function body annotations and kwargs."""

    def test_tile_shape_and_dtype_from_closure(self):
        """User defines tile shape and dtype outside, uses in body annotations."""
        tensor_shape = [128, 128]
        tile_shape = [64, 64]
        dtype = pl.FP32

        @pl.function
        def func(
            t: pl.Tensor[tensor_shape, dtype], out: pl.Tensor[tensor_shape, dtype]
        ) -> pl.Tensor[tensor_shape, dtype]:
            a: pl.Tile[tile_shape, dtype] = pl.block.load(t, offsets=[0, 0], shapes=tile_shape)
            result: pl.Tensor[tensor_shape, dtype] = pl.block.store(
                a, offsets=[0, 0], shapes=tile_shape, output_tensor=out
            )
            return result

        assert isinstance(func, ir.Function)

    def test_shapes_kwarg_from_variable(self):
        """User passes shapes= kwarg as a closure variable (t.py pattern)."""
        tile_shape = [32, 32]

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[[32, 32], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=tile_shape)
            result: pl.Tensor[[128, 128], pl.FP32] = pl.block.store(
                a, offsets=[0, 0], shapes=tile_shape, output_tensor=out
            )
            return result

        assert isinstance(func, ir.Function)

    def test_int_kwarg_from_closure(self):
        """User passes an int kwarg (like axis) from closure."""
        swap_axis = 1

        @pl.function
        def func(
            x: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[128, 64], pl.FP32]:
            result: pl.Tensor[[128, 64], pl.FP32] = pl.transpose(x, axis1=0, axis2=swap_axis)
            return result

        assert isinstance(func, ir.Function)

    def test_dtype_kwarg_from_closure(self):
        """User passes dtype= kwarg from closure variable."""
        out_dtype = pl.FP16

        @pl.function
        def func(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP16]:
            result: pl.Tensor[[64, 64], pl.FP16] = pl.cast(x, target_type=out_dtype)
            return result

        assert isinstance(func, ir.Function)

    def test_full_parametrized_kernel(self):
        """Realistic pattern: fully parametrized kernel (the t.py use case)."""
        dtype = pl.FP32
        x, y = 128, 128
        shape = [128, 128]
        tile_shape = [64, 64]

        @pl.function
        def kernel_add(t: pl.Tensor[[x, y], dtype], out: pl.Tensor[shape, dtype]) -> pl.Tensor[shape, dtype]:
            a: pl.Tile[tile_shape, dtype] = pl.block.load(t, offsets=[0, 0], shapes=tile_shape)
            b: pl.Tile[tile_shape, dtype] = pl.add(a, 5)
            result: pl.Tensor[shape, dtype] = pl.block.store(
                b, offsets=[0, 0], shapes=tile_shape, output_tensor=out
            )
            return result

        assert isinstance(kernel_add, ir.Function)
        assert len(kernel_add.params) == 2
        for p in kernel_add.params:
            assert isinstance(p.type, ir.TensorType)
            assert p.type.dtype == DataType.FP32

    def test_program_with_closure_shapes_in_body(self):
        """User uses closure shapes inside @pl.program methods."""
        shape = [128, 128]
        tile_shape = [64, 64]
        dtype = pl.FP32

        @pl.program
        class Prog:
            @pl.function
            def compute(
                self, t: pl.Tensor[shape, dtype], out: pl.Tensor[shape, dtype]
            ) -> pl.Tensor[shape, dtype]:
                a: pl.Tile[tile_shape, dtype] = pl.block.load(t, offsets=[0, 0], shapes=tile_shape)
                result: pl.Tensor[shape, dtype] = pl.block.store(
                    a, offsets=[0, 0], shapes=tile_shape, output_tensor=out
                )
                return result

        assert isinstance(Prog, ir.Program)


class TestDynamicShapeEdgeCases:
    """Edge cases and error scenarios users may encounter."""

    def test_dynvar_with_variable_shape(self):
        """User combines DynVar list with variable-as-shape pattern."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")
        shape = [M, N]

        @pl.function
        def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert isinstance(param_type.shape[0], ir.Var)
        assert param_type.shape[0].name == "M"
        assert isinstance(param_type.shape[1], ir.Var)
        assert param_type.shape[1].name == "N"

    def test_dynvar_mixed_with_int_in_variable_shape(self):
        """User mixes DynVar and int in a shape variable."""
        M = pl.dynamic("M")
        shape = [M, 128]

        @pl.function
        def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert isinstance(param_type.shape[0], ir.Var)
        assert param_type.shape[0].name == "M"
        assert isinstance(param_type.shape[1], ir.ConstInt)
        assert param_type.shape[1].value == 128

    def test_dynvar_mixed_with_computed_dim(self):
        """User mixes DynVar and arithmetic expression in shape."""
        M = pl.dynamic("M")
        base = 64

        @pl.function
        def func(
            x: pl.Tensor[[M, base * 2], pl.FP32],
        ) -> pl.Tensor[[M, base * 2], pl.FP32]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert isinstance(param_type.shape[0], ir.Var)
        assert param_type.shape[0].name == "M"
        # base * 2 evaluates to 128 at compile time, then gets wrapped as ConstInt
        # because the shape is mixed (Var + int → all Expr)
        assert isinstance(param_type.shape[1], ir.ConstInt)
        assert param_type.shape[1].value == 128

    def test_same_shape_reused_across_params_and_return(self):
        """User uses same shape variable everywhere — should produce consistent types."""
        shape = [64, 64]
        dtype = pl.FP32

        @pl.function
        def func(
            a: pl.Tensor[shape, dtype],
            b: pl.Tensor[shape, dtype],
        ) -> pl.Tensor[shape, dtype]:
            return a

        all_types = [p.type for p in func.params] + func.return_types
        for t in all_types:
            assert isinstance(t, ir.TensorType)
            assert len(t.shape) == 2
            assert t.dtype == DataType.FP32

    def test_tile_type_with_variable_shape(self):
        """User uses a variable for Tile shape annotation."""
        tile_shape = [32, 32]

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[tile_shape, pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[32, 32])
            result: pl.Tensor[[128, 128], pl.FP32] = pl.block.store(
                a, offsets=[0, 0], shapes=[32, 32], output_tensor=out
            )
            return result

        assert isinstance(func, ir.Function)

    def test_shape_variable_not_defined_raises_error(self):
        """User typos variable name — should get a clear error."""
        shape = [128, 64]  # noqa: F841 — intentionally unused; typo below

        with pytest.raises(Exception, match="shaep|Cannot resolve|Unknown|undefined"):

            @pl.function
            def func(x: pl.Tensor[shaep, pl.FP32]) -> pl.Tensor[shaep, pl.FP32]:  # noqa: F821
                return x

    def test_non_list_shape_variable_raises_error(self):
        """User accidentally passes a string as shape."""
        shape = "128x64"

        with pytest.raises(Exception, match="must be a list or tuple|Failed to evaluate"):

            @pl.function
            def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
                return x

    def test_float_in_shape_raises_error(self):
        """User accidentally uses floats in shape."""
        shape = [128.0, 64.0]

        with pytest.raises(Exception, match="must be int|element"):

            @pl.function
            def func(x: pl.Tensor[shape, pl.FP32]) -> pl.Tensor[shape, pl.FP32]:
                return x

    def test_nested_function_captures_correct_scope(self):
        """Shape variables from outer function are captured by inner @pl.function."""

        def make_kernel(rows, cols, dtype):
            @pl.function
            def kernel(
                x: pl.Tensor[[rows, cols], dtype],
            ) -> pl.Tensor[[rows, cols], dtype]:
                return x

            return kernel

        k1 = make_kernel(64, 64, pl.FP16)
        k2 = make_kernel(128, 256, pl.FP32)

        assert isinstance(k1, ir.Function)
        assert isinstance(k2, ir.Function)
        k1_type = k1.params[0].type
        k2_type = k2.params[0].type
        assert isinstance(k1_type, ir.TensorType)
        assert isinstance(k2_type, ir.TensorType)
        assert k1_type.dtype == DataType.FP16
        assert k2_type.dtype == DataType.FP32

    def test_factory_with_shape_variable(self):
        """User writes a factory function that parametrizes shape."""

        def make_kernel(shape, dtype):
            @pl.function
            def kernel(x: pl.Tensor[shape, dtype]) -> pl.Tensor[shape, dtype]:
                return x

            return kernel

        k = make_kernel([128, 128], pl.FP32)
        assert isinstance(k, ir.Function)
        param_type = k.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert param_type.dtype == DataType.FP32


class TestLayoutResolution:
    """Tests for TensorLayout resolution in type annotations."""

    def test_resolve_tensor_with_layout_nz(self):
        """Tensor with NZ layout creates TensorType with TensorView."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16, pl.NZ]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.dtype == DataType.FP16
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.NZ

    def test_resolve_tensor_with_layout_dn(self):
        """Tensor with DN layout."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[16, 1], pl.FP16, pl.DN]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.DN

    def test_resolve_tensor_with_layout_nd(self):
        """Tensor with explicit ND layout."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[64, 128], pl.FP32, pl.ND]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.ND

    def test_resolve_tensor_without_layout_backward_compat(self):
        """Tensor without layout has no tensor_view (backward compatible)."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.tensor_view is None

    def test_resolve_tensor_layout_invalid(self):
        """Invalid layout raises ParserTypeError."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16, pl.INVALID]", mode="eval").body

        with pytest.raises(ParserTypeError, match="Unknown layout"):
            resolver.resolve_type(node)

    def test_tile_with_layout_raises_error(self):
        """Tile does not support layout syntax."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tile[[64, 64], pl.FP32, pl.NZ]", mode="eval").body

        with pytest.raises(ParserTypeError, match="Layout is only supported for Tensor"):
            resolver.resolve_type(node)

    def test_resolve_layout_bare_name(self):
        """Layout specified as bare name (NZ) instead of pl.NZ."""
        resolver = _make_resolver()
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16, NZ]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.NZ

    def test_resolve_layout_from_closure(self):
        """Layout from closure variable."""
        resolver = _make_resolver(closure_vars={"my_layout": ir.TensorLayout.NZ})
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16, my_layout]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.NZ

    def test_resolve_layout_closure_invalid_type(self):
        """Non-TensorLayout closure variable raises error."""
        resolver = _make_resolver(closure_vars={"my_layout": "NZ"})
        node = ast.parse("pl.Tensor[[64, 128], pl.FP16, my_layout]", mode="eval").body

        with pytest.raises(ParserTypeError, match="must be a TensorLayout"):
            resolver.resolve_type(node)

    def test_resolve_tensor_layout_with_dynamic_shape(self):
        """Layout works with dynamic shapes."""
        resolver = _make_resolver(closure_vars={"M": DynVar("M")})
        node = ast.parse("pl.Tensor[[M, 128], pl.FP16, pl.NZ]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[0], ir.Var)
        assert result.shape[0].name == "M"
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.NZ

    def test_resolve_tensor_layout_with_shape_variable(self):
        """Layout works with shape variable from closure."""
        resolver = _make_resolver(closure_vars={"shape": [64, 128]})
        node = ast.parse("pl.Tensor[shape, pl.FP16, pl.DN]", mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert len(result.shape) == 2
        assert result.tensor_view is not None
        assert result.tensor_view.layout == ir.TensorLayout.DN


class TestLayoutIntegration:
    """End-to-end tests: decorator + layout annotations."""

    def test_function_with_tensor_layout(self):
        """@pl.function with layout in parameter and return type."""

        @pl.function
        def func(
            x: pl.Tensor[[64, 128], pl.FP16, pl.NZ],
        ) -> pl.Tensor[[64, 128], pl.FP16, pl.NZ]:
            return x

        assert isinstance(func, ir.Function)
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.dtype == DataType.FP16
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.NZ

        ret_type = func.return_types[0]
        assert isinstance(ret_type, ir.TensorType)
        assert ret_type.tensor_view is not None
        assert ret_type.tensor_view.layout == ir.TensorLayout.NZ

    def test_function_mixed_layout_and_no_layout(self):
        """@pl.function with some params having layout and some not."""

        @pl.function
        def func(
            a: pl.Tensor[[64, 128], pl.FP16, pl.NZ],
            b: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            return a

        a_type = func.params[0].type
        b_type = func.params[1].type
        assert isinstance(a_type, ir.TensorType)
        assert isinstance(b_type, ir.TensorType)
        assert a_type.tensor_view is not None
        assert a_type.tensor_view.layout == ir.TensorLayout.NZ
        assert b_type.tensor_view is None

    def test_program_with_tensor_layout(self):
        """@pl.program with layout annotations."""

        @pl.program
        class MyProgram:
            @pl.function
            def compute(
                self,
                x: pl.Tensor[[64, 128], pl.FP16, pl.NZ],
            ) -> pl.Tensor[[64, 128], pl.FP16, pl.NZ]:
                return x

        assert isinstance(MyProgram, ir.Program)
        func = list(MyProgram.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.NZ

    def test_function_layout_from_closure_variable(self):
        """@pl.function with layout from closure variable."""
        layout = pl.NZ

        @pl.function
        def func(
            x: pl.Tensor[[64, 128], pl.FP16, layout],
        ) -> pl.Tensor[[64, 128], pl.FP16, layout]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.NZ

    @pytest.mark.parametrize(
        "layout,expected",
        [
            (pl.ND, ir.TensorLayout.ND),
            (pl.DN, ir.TensorLayout.DN),
            (pl.NZ, ir.TensorLayout.NZ),
        ],
    )
    def test_parametrized_layout(self, layout, expected):
        """pytest.mark.parametrize with layout."""

        @pl.function
        def func(
            x: pl.Tensor[[64, 128], pl.FP16, layout],
        ) -> pl.Tensor[[64, 128], pl.FP16, layout]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == expected

    def test_function_with_dn_layout(self):
        """@pl.function with DN layout for column-major tensors."""

        @pl.function
        def func(
            x: pl.Tensor[[16, 1], pl.FP16, pl.DN],
        ) -> pl.Tensor[[16, 1], pl.FP16, pl.DN]:
            return x

        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.DN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
