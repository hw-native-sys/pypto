# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for ``with pl.manual_scope():`` and the ``deps=[var]`` Call kwarg."""

import pypto.language as pl
import pytest
from pypto import ir


def _first_runtime_scope(stmt):
    """Return the first RuntimeScopeStmt found in a stmt subtree (DFS), or None."""
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_runtime_scope(s)
            if r is not None:
                return r
    return None


class TestManualScopeParsing:
    def test_parse_manual_scope_creates_runtime_scope_with_manual_true(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None, "expected a RuntimeScopeStmt for `with pl.manual_scope():`"
        assert scope.manual is True

    def test_parse_manual_scope_rejects_arguments(self):
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope(name="foo"):
                        return x

    def test_deps_kwarg_records_user_manual_dep_edges(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    b = self.k2(x, deps=[a])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        # Walk the scope body and inspect both Calls.
        body = scope.body
        if isinstance(body, ir.SeqStmts):
            stmts = list(body.stmts)
        else:
            stmts = [body]
        assert len(stmts) == 2
        a_assign, b_assign = stmts
        assert isinstance(a_assign, ir.AssignStmt)
        assert isinstance(b_assign, ir.AssignStmt)
        a_call = a_assign.value
        b_call = b_assign.value
        assert isinstance(a_call, ir.Call)
        assert isinstance(b_call, ir.Call)
        assert "user_manual_dep_edges" not in a_call.attrs
        b_user_deps = b_call.attrs.get("user_manual_dep_edges", [])
        assert len(b_user_deps) == 1
        assert b_user_deps[0].same_as(a_assign.var)

    def test_deps_outside_manual_scope_is_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    a = self.k1(x)
                    # No manual_scope around this — the deps= kwarg must error.
                    b = self.k1(x, deps=[a])
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
