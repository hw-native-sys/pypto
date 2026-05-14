# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for ``pl.task_id_invalid()`` / ``pl.task_id_of(...)`` and the
loosened ``deps=[...]`` kwarg that now accepts TaskId scalars."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType


def _first_runtime_scope(stmt):
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_runtime_scope(s)
            if r is not None:
                return r
    return None


def _flatten(stmt):
    if isinstance(stmt, ir.SeqStmts):
        return list(stmt.stmts)
    return [stmt]


class TestTaskIdNamespace:
    def test_pl_task_id_alias(self):
        """``pl.TASK_ID`` is the TASK_ID DataType; ``pl.TaskId`` is its scalar annotation."""
        assert pl.TASK_ID == DataType.TASK_ID
        # ``pl.TaskId`` is the annotation-only Scalar form of TASK_ID. Two
        # constructions of ``Scalar[TASK_ID]`` give different instances but
        # both should report the same dtype.
        assert pl.TaskId.dtype == DataType.TASK_ID

    def test_task_id_invalid_returns_task_id_scalar(self):
        """``pl.task_id_invalid()`` wraps a Call of result type Scalar[TASK_ID]."""
        s = pl.task_id_invalid()
        assert isinstance(s.expr, ir.Call)
        assert s.expr.op.name == "system.task_invalid"
        assert isinstance(s.expr.type, ir.ScalarType)
        assert s.expr.type.dtype == DataType.TASK_ID


class TestTaskIdOfParsing:
    def test_task_id_of_extracts_kernel_lhs_task_id(self):
        """``tid = pl.task_id_of(out)`` parses to an AssignStmt whose RHS is a
        ``Call(system.task_id_of, [out])`` and whose LHS Var has TASK_ID type."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    tid = pl.task_id_of(a)  # noqa: F841 — referenced via attr on resulting Var
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        stmts = _flatten(scope.body)
        assert len(stmts) >= 2
        # The second AssignStmt should hold the task_id_of call.
        tid_assign = stmts[1]
        assert isinstance(tid_assign, ir.AssignStmt)
        tid_call = tid_assign.value
        assert isinstance(tid_call, ir.Call)
        assert tid_call.op.name == "system.task_id_of"
        # LHS Var has TASK_ID scalar type.
        assert isinstance(tid_assign.var.type, ir.ScalarType)
        assert tid_assign.var.type.dtype == DataType.TASK_ID


class TestDepsKwargAcceptsBothShapes:
    def test_deps_accepts_tensor_var_legacy(self):
        """Legacy path: ``deps=[tensor_var]`` still works."""

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
        b_assign = _flatten(scope.body)[1]
        assert isinstance(b_assign, ir.AssignStmt)
        b_call = b_assign.value
        assert isinstance(b_call, ir.Call)
        edges = b_call.attrs.get("user_manual_dep_edges", [])
        assert len(edges) == 1
        # Tensor-typed dep var.
        assert isinstance(edges[0].type, ir.TensorType)

    def test_deps_accepts_task_id_scalar_var(self):
        """Explicit path: ``deps=[task_id_scalar]`` is accepted."""

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
                    tid = pl.task_id_of(a)
                    b = self.k2(x, deps=[tid])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        # Stmt 0: a = self.k1(x)
        # Stmt 1: tid = pl.task_id_of(a)
        # Stmt 2: b = self.k2(x, deps=[tid])
        b_assign = _flatten(scope.body)[2]
        assert isinstance(b_assign, ir.AssignStmt)
        b_call = b_assign.value
        assert isinstance(b_call, ir.Call)
        edges = b_call.attrs.get("user_manual_dep_edges", [])
        assert len(edges) == 1
        # TaskId-typed dep var (not Tensor).
        assert isinstance(edges[0].type, ir.ScalarType)
        assert edges[0].type.dtype == DataType.TASK_ID

    def test_deps_rejects_non_var_expr(self):
        """``deps=[some_scalar_int]`` (a non-TaskId, non-Tensor scalar) errors."""
        with pytest.raises(Exception):  # noqa: B017 — ParserTypeError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    x: pl.Tensor[[64], pl.FP32],
                    n: pl.Scalar[pl.INT32],
                ) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        # n is INT32, not TASK_ID and not a tensor — reject.
                        b = self.k1(x, deps=[n])
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
