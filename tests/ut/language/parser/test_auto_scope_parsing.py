# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser/printer tests for the ``with pl.auto_scope():`` DSL construct.

``pl.auto_scope()`` is the explicit DSL form of an AUTO ``RuntimeScopeStmt``
(``manual=False``) — the IR representation of the orchestration codegen's
``PTO2_SCOPE()`` block. It is the round-trip surface for the AUTO scopes that
the MaterializeRuntimeScopes pass inserts.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.printer import python_print


def _first_runtime_scope(stmt):
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_runtime_scope(s)
            if r is not None:
                return r
    return None


def test_parse_auto_scope_creates_runtime_scope_with_manual_false():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.auto_scope():
                a = self.k1(x)
            return a

    fn = Prog.get_function("main")
    assert fn is not None
    scope = _first_runtime_scope(fn.body)
    assert scope is not None, "expected a RuntimeScopeStmt for `with pl.auto_scope():`"
    assert scope.manual is False


def test_auto_scope_round_trips():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.auto_scope():
                a = self.k1(x)
            return a

    printed = python_print(Prog, format=False)
    assert "pl.auto_scope()" in printed
    reparsed = pl.parse(printed)
    ir.assert_structural_equal(Prog, reparsed)


def test_auto_scope_rejects_arguments():
    with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_scope(1):
                    a = x
                return a


def test_auto_scope_rejected_inside_manual_scope():
    with pytest.raises(Exception):  # noqa: B017 — runtime forbids AUTO nested in MANUAL

        @pl.program
        class _Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    with pl.auto_scope():
                        a = self.k1(x)
                return a


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
