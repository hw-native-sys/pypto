# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for closure variable resolution in DSL function bodies (issue #276).

Verifies that Python globals/closure variables used as positional arguments
in function calls inside @pl.function bodies are resolved correctly.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserTypeError, UndefinedVariableError


def _first_call(func: ir.Function) -> ir.Call:
    """Return the RHS of the function body's first binding, asserting it is a Call."""
    body = func.body
    stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            assert isinstance(stmt.value, ir.Call), (
                f"{stmt.var.name_hint} is {type(stmt.value).__name__}, expected Call"
            )
            return stmt.value
    raise AssertionError("No binding found in function body")


def _int_elements(expr: ir.Expr) -> list[int]:
    """Flatten a MakeTuple of integer constants into plain ints."""
    assert isinstance(expr, ir.MakeTuple), f"expected MakeTuple, got {type(expr).__name__}"
    values = []
    for element in expr.elements:
        assert isinstance(element, ir.ConstInt), f"expected ConstInt element, got {type(element).__name__}"
        values.append(element.value)
    return values


class TestClosureVarAsPositionalArg:
    """Closure variables used as positional arguments in function calls."""

    def test_list_closure_var_as_positional_arg(self):
        """List closure var works as positional arg (the original issue)."""
        OFFSET = [0, 0]
        TILE_SHAPE = [64, 64]

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)
            result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(a, OFFSET, output_tensor=out)
            return result

        # The list closure vars reach the load as tuple literals with their values
        load = _first_call(func)
        assert load.op.name == ir.get_op("tile.load").name
        assert _int_elements(load.args[1]) == [0, 0]
        assert _int_elements(load.args[2]) == [64, 64]

    def test_int_closure_var_as_positional_arg(self):
        """Int closure variable resolves to ConstInt in function body."""
        AXIS = 1

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            result: pl.Tensor[[128, 64], pl.FP32] = pl.transpose(x, axis1=0, axis2=AXIS)
            return result

        transpose = _first_call(func)
        assert transpose.op.name == ir.get_op("tensor.transpose").name
        # axis2=AXIS resolved to the closure's value 1, not to axis1's 0
        axis2 = transpose.args[2]
        assert isinstance(axis2, ir.ConstInt)
        assert axis2.value == 1

    def test_float_closure_var_as_positional_arg(self):
        """Float closure variable resolves to ConstFloat in function body."""
        SCALE = 2.0

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, SCALE)
            return result

        scale_arg = _first_call(func).args[1]
        assert isinstance(scale_arg, ir.ConstFloat)
        assert scale_arg.value == 2.0

    def test_bool_closure_var_as_positional_arg(self):
        """Bool closure variable resolves to ConstBool in function body."""
        FLAG = True

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, FLAG)
            return result

        flag_arg = _first_call(func).args[1]
        # Must stay a ConstBool — not silently widened to ConstInt(1)
        assert isinstance(flag_arg, ir.ConstBool)
        assert flag_arg.value is True

    def test_tuple_closure_var_as_positional_arg(self):
        """Tuple closure variable resolves to MakeTuple in function body."""
        OFFSET = (0, 0)
        TILE_SHAPE = (64, 64)

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)
            result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(a, OFFSET, output_tensor=out)
            return result

        # Tuple closure vars resolve identically to list ones
        load = _first_call(func)
        assert _int_elements(load.args[1]) == [0, 0]
        assert _int_elements(load.args[2]) == [64, 64]

    def test_nested_list_closure_var(self):
        """Nested list closure variable recursively converts to nested MakeTuple."""
        OFFSETS = [[0, 0], [64, 64]]

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSETS, shapes=[64, 64])  # type: ignore[arg-type]
            result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(a, [0, 0], output_tensor=out)
            return result

        # The outer list becomes a MakeTuple whose elements are themselves MakeTuples
        offsets = _first_call(func).args[1]
        assert isinstance(offsets, ir.MakeTuple)
        rows = list(offsets.elements)
        assert len(rows) == 2
        assert _int_elements(rows[0]) == [0, 0]
        assert _int_elements(rows[1]) == [64, 64]

    def test_dynvar_closure_var(self):
        """DynVar closure variable resolves to ir.Var with INDEX type."""
        M = pl.dynamic("M")

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # ``M`` is a DynVar (INDEX); cast before use as a tensor scalar operand.
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, pl.cast(M, pl.INT32))  # type: ignore[arg-type]
            return result

        cast = _first_call(func).args[1]
        assert isinstance(cast, ir.Cast)
        # The DynVar resolves to a Var named "M" carrying INDEX dtype
        operand = cast.operand
        assert isinstance(operand, ir.Var)
        assert operand.name_hint == "M"
        assert isinstance(operand.type, ir.ScalarType)
        assert operand.type.dtype == pl.INDEX


class TestClosureVarShadowing:
    """DSL scope takes priority over closure variables."""

    def test_dsl_scope_shadows_closure(self):
        """Variable defined in DSL body shadows same-named closure variable."""
        x_scale = 999.0  # noqa: F841 — deliberately shadowed by DSL assignment

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            x_scale: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x_scale, x)
            return result

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        add_stmt, mul_stmt = (s for s in body.stmts if isinstance(s, ir.AssignStmt))

        # `x_scale` in the mul refers to the DSL binding, not the 999.0 closure var
        mul_call = mul_stmt.value
        assert isinstance(mul_call, ir.Call)
        assert mul_call.args[0] is add_stmt.var


class TestClosureVarErrors:
    """Error cases for closure variable resolution."""

    def test_undefined_variable_still_raises(self):
        """Variable not in scope or closure raises UndefinedVariableError."""
        with pytest.raises(UndefinedVariableError, match="Undefined variable"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, totally_undefined)  # noqa: F821 # type: ignore
                return result

    def test_unsupported_closure_type_raises(self):
        """Unsupported closure variable type raises ParserTypeError."""
        BAD_VALUE = "not_a_number"

        with pytest.raises(ParserTypeError, match="Unsupported closure variable type"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, BAD_VALUE)  # type: ignore[arg-type]
                return result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
