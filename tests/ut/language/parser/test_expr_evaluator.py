# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExprEvaluator."""

import ast

import pytest
from pypto import DataType
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.expr_evaluator import ExprEvaluator
from pypto.language.typing.dynamic import DynVar


def _parse_expr(code: str) -> ast.expr:
    """Parse a Python expression string to an AST node."""
    return ast.parse(code, mode="eval").body


class TestBasicVariableResolution:
    """Tests for basic variable lookups."""

    def test_resolve_int(self):
        ev = ExprEvaluator(closure_vars={"x": 42})
        assert ev.eval_expr(_parse_expr("x")) == 42

    def test_resolve_list(self):
        ev = ExprEvaluator(closure_vars={"shape": [128, 64]})
        assert ev.eval_expr(_parse_expr("shape")) == [128, 64]

    def test_resolve_tuple(self):
        ev = ExprEvaluator(closure_vars={"dims": (32, 64)})
        assert ev.eval_expr(_parse_expr("dims")) == (32, 64)

    def test_resolve_datatype(self):
        ev = ExprEvaluator(closure_vars={"dtype": DataType.FP32})
        assert ev.eval_expr(_parse_expr("dtype")) == DataType.FP32

    def test_resolve_dynvar(self):
        dv = DynVar("M")
        ev = ExprEvaluator(closure_vars={"M": dv})
        assert ev.eval_expr(_parse_expr("M")) is dv

    def test_resolve_bool(self):
        ev = ExprEvaluator(closure_vars={"flag": True})
        assert ev.eval_expr(_parse_expr("flag")) is True

    def test_resolve_none(self):
        ev = ExprEvaluator(closure_vars={"val": None})
        assert ev.eval_expr(_parse_expr("val")) is None


class TestArithmeticExpressions:
    """Tests for arithmetic expression evaluation."""

    def test_multiply(self):
        ev = ExprEvaluator(closure_vars={"base": 64})
        assert ev.eval_expr(_parse_expr("base * 2")) == 128

    def test_add(self):
        ev = ExprEvaluator(closure_vars={"a": 10, "b": 20})
        assert ev.eval_expr(_parse_expr("a + b")) == 30

    def test_subtract(self):
        ev = ExprEvaluator(closure_vars={"total": 256})
        assert ev.eval_expr(_parse_expr("total - 128")) == 128

    def test_floor_divide(self):
        ev = ExprEvaluator(closure_vars={"n": 256})
        assert ev.eval_expr(_parse_expr("n // 4")) == 64

    def test_negative(self):
        ev = ExprEvaluator(closure_vars={"x": 5})
        assert ev.eval_expr(_parse_expr("-x")) == -5


class TestBuiltinCalls:
    """Tests for safe builtin function calls."""

    def test_len(self):
        ev = ExprEvaluator(closure_vars={"shape": [128, 64, 32]})
        assert ev.eval_expr(_parse_expr("len(shape)")) == 3

    def test_max(self):
        ev = ExprEvaluator(closure_vars={"a": 10, "b": 20})
        assert ev.eval_expr(_parse_expr("max(a, b)")) == 20

    def test_min(self):
        ev = ExprEvaluator(closure_vars={"a": 10, "b": 20})
        assert ev.eval_expr(_parse_expr("min(a, b)")) == 10

    def test_abs(self):
        ev = ExprEvaluator(closure_vars={"x": -5})
        assert ev.eval_expr(_parse_expr("abs(x)")) == 5

    def test_int_conversion(self):
        ev = ExprEvaluator(closure_vars={"x": 3.7})
        assert ev.eval_expr(_parse_expr("int(x)")) == 3

    def test_list_builtin(self):
        ev = ExprEvaluator(closure_vars={"t": (1, 2, 3)})
        assert ev.eval_expr(_parse_expr("list(t)")) == [1, 2, 3]


class TestSubscriptAndAttribute:
    """Tests for subscript and attribute access."""

    def test_subscript_index(self):
        ev = ExprEvaluator(closure_vars={"dims": [128, 64, 32]})
        assert ev.eval_expr(_parse_expr("dims[1]")) == 64

    def test_subscript_slice(self):
        ev = ExprEvaluator(closure_vars={"dims": [128, 64, 32, 16]})
        assert ev.eval_expr(_parse_expr("dims[0:2]")) == [128, 64]

    def test_list_literal_with_vars(self):
        ev = ExprEvaluator(closure_vars={"a": 128, "b": 64})
        assert ev.eval_expr(_parse_expr("[a, b]")) == [128, 64]


class TestErrorCases:
    """Tests for error handling."""

    def test_undefined_variable(self):
        ev = ExprEvaluator(closure_vars={})
        with pytest.raises(ParserTypeError, match="Cannot resolve expression"):
            ev.eval_expr(_parse_expr("undefined_var"))

    def test_type_error(self):
        ev = ExprEvaluator(closure_vars={"x": "hello"})
        with pytest.raises(ParserTypeError, match="Failed to evaluate expression"):
            ev.eval_expr(_parse_expr("x + 1"))

    def test_blocked_builtin_open(self):
        ev = ExprEvaluator(closure_vars={})
        with pytest.raises(ParserTypeError, match="Cannot resolve expression"):
            ev.eval_expr(_parse_expr("open('file.txt')"))

    def test_blocked_builtin_import(self):
        ev = ExprEvaluator(closure_vars={})
        with pytest.raises(ParserTypeError):
            ev.eval_expr(_parse_expr("__import__('os')"))


class TestTryEvalExpr:
    """Tests for try_eval_expr non-throwing variant."""

    def test_success(self):
        ev = ExprEvaluator(closure_vars={"x": 42})
        success, value = ev.try_eval_expr(_parse_expr("x"))
        assert success is True
        assert value == 42

    def test_failure(self):
        ev = ExprEvaluator(closure_vars={})
        success, value = ev.try_eval_expr(_parse_expr("undefined"))
        assert success is False
        assert value is None

    def test_success_with_expression(self):
        ev = ExprEvaluator(closure_vars={"base": 64})
        success, value = ev.try_eval_expr(_parse_expr("base * 2"))
        assert success is True
        assert value == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
