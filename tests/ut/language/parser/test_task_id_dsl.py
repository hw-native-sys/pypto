# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for the TaskId DSL surface: the ``None`` sentinel, the
``pl.submit(...)`` construct, and the ``deps=[...]`` kwarg it accepts."""

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
        out = []
        for s in stmt.stmts:
            out.extend(_flatten(s))
        return out
    return [stmt]


def _calls_in(stmt):
    return [s.value for s in _flatten(stmt) if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call)]


class TestTaskIdNamespace:
    def test_pl_task_id_alias(self):
        """``pl.TASK_ID`` is the TASK_ID DataType; ``pl.TaskId`` is its scalar annotation."""
        assert pl.TASK_ID == DataType.TASK_ID
        assert pl.TaskId.dtype == DataType.TASK_ID

    def test_none_sentinel_lowers_to_task_invalid(self):
        """The Python literal ``None`` in a TaskId position lowers to ``system.task_invalid``."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    prev_tid = None  # noqa: F841 — "no producer yet" sentinel
                    a, _ = pl.submit(self.k1, x)
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        # The ``prev_tid = None`` assignment lowers to a system.task_invalid Call.
        invalid_calls = [c for c in _calls_in(scope.body) if c.op.name == "system.task_invalid"]
        assert len(invalid_calls) == 1
        assert isinstance(invalid_calls[0].type, ir.ScalarType)
        assert invalid_calls[0].type.dtype == DataType.TASK_ID


class TestSubmitParsing:
    def test_submit_augments_return_type_with_task_id(self):
        """``out, tid = pl.submit(self.k, x)`` builds one Call typed Tuple{<result>, TaskId}."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, tid = pl.submit(self.k1, x)  # noqa: F841 — tid checked via IR
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        k1_calls = [c for c in _calls_in(scope.body) if c.op.name == "k1"]
        assert len(k1_calls) == 1
        ret = k1_calls[0].type
        assert isinstance(ret, ir.TupleType)
        # Flat Tuple{<single result>, TaskId}.
        assert len(ret.types) == 2
        assert isinstance(ret.types[1], ir.ScalarType)
        assert ret.types[1].dtype == DataType.TASK_ID


class TestDepsKwargAcceptsTaskId:
    def test_deps_accepts_submit_task_id(self):
        """``deps=[tid]`` where ``tid`` is a prior submit's producer TaskId."""

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
                    a, tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, x, deps=[tid])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        k2_call = next(c for c in _calls_in(scope.body) if c.op.name == "k2")
        edges = k2_call.attrs.get("manual_dep_edges", [])
        assert len(edges) == 1
        assert isinstance(edges[0].type, ir.ScalarType)
        assert edges[0].type.dtype == DataType.TASK_ID

    def test_deps_rejects_tensor_var(self):
        """Tensor variables are not accepted in ``deps=[...]``."""
        with pytest.raises(Exception):  # noqa: B017 — ParserTypeError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a, _ = pl.submit(self.k1, x)
                        # Bare tensor — reject (deps= takes TaskIds only).
                        b, _ = pl.submit(self.k2, x, deps=[a])
                    return b

    def test_deps_rejects_non_task_id_scalar(self):
        """``deps=[int_scalar]`` (a non-TaskId scalar) errors."""
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
                        # n is INT32, not TASK_ID — reject.
                        b, _ = pl.submit(self.k1, x, deps=[n])
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
