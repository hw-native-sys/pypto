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
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, OFFSET, TILE_SHAPE)
            result: pl.Tensor[[128, 128], pl.FP32] = pl.block.store(a, OFFSET, TILE_SHAPE, output_tensor=out)
            return result

        assert isinstance(func, ir.Function)

    def test_int_closure_var_as_positional_arg(self):
        """Int closure variable resolves to ConstInt in function body."""
        AXIS = 1

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            result: pl.Tensor[[128, 64], pl.FP32] = pl.transpose(x, axis1=0, axis2=AXIS)
            return result

        assert isinstance(func, ir.Function)

    def test_float_closure_var_as_positional_arg(self):
        """Float closure variable resolves to ConstFloat in function body."""
        SCALE = 2.0

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, SCALE)
            return result

        assert isinstance(func, ir.Function)

    def test_bool_closure_var_as_positional_arg(self):
        """Bool closure variable resolves to ConstBool in function body."""
        FLAG = True

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, FLAG)
            return result

        assert isinstance(func, ir.Function)

    def test_tuple_closure_var_as_positional_arg(self):
        """Tuple closure variable resolves to MakeTuple in function body."""
        OFFSET = (0, 0)
        TILE_SHAPE = (64, 64)

        @pl.function
        def func(
            t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, OFFSET, TILE_SHAPE)
            result: pl.Tensor[[128, 128], pl.FP32] = pl.block.store(a, OFFSET, TILE_SHAPE, output_tensor=out)
            return result

        assert isinstance(func, ir.Function)


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

        assert isinstance(func, ir.Function)


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
