# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for ``with pl.manual_scope():`` and the ``pl.submit(...)`` construct."""

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


def _flatten(stmt):
    """Flatten a (possibly nested) SeqStmts subtree into a list of statements."""
    if isinstance(stmt, ir.SeqStmts):
        out = []
        for s in stmt.stmts:
            out.extend(_flatten(s))
        return out
    return [stmt]


def _calls_in(stmt):
    """Collect every ir.Call that is the RHS of an AssignStmt in the subtree."""
    calls = []
    for s in _flatten(stmt):
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call):
            calls.append(s.value)
    return calls


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

    def test_submit_records_manual_dep_edges(self):
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
                    a, a_tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, x, deps=[a_tid])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _calls_in(scope.body)
        # One kernel Call per submit; each carries the flat augmented return
        # type Tuple{<kernel result>, TaskId}.
        k1_calls = [c for c in calls if c.op.name == "k1"]
        k2_calls = [c for c in calls if c.op.name == "k2"]
        assert len(k1_calls) == 1
        assert len(k2_calls) == 1
        k1_call, k2_call = k1_calls[0], k2_calls[0]
        for c in (k1_call, k2_call):
            assert isinstance(c.type, ir.TupleType)
            assert len(c.type.types) == 2
            assert isinstance(c.type.types[1], ir.ScalarType)
            assert c.type.types[1].dtype == pl.TASK_ID
        # Producer k1 has no dep edges of its own.
        assert "manual_dep_edges" not in k1_call.attrs
        # Consumer k2 records one dep edge, naming the TaskId scalar `a_tid`.
        k2_deps = k2_call.attrs.get("manual_dep_edges", [])
        assert len(k2_deps) == 1
        assert isinstance(k2_deps[0].type, ir.ScalarType)
        assert k2_deps[0].type.dtype == pl.TASK_ID

    def test_submit_none_dep_entry_dropped(self):
        """A bare ``None`` entry in ``deps=`` contributes no edge."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, _ = pl.submit(self.k1, x, deps=[None])
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        k1_call = next(c for c in _calls_in(scope.body) if c.op.name == "k1")
        # ``deps=[None]`` drops the only entry, so no edge attr is recorded.
        assert k1_call.attrs.get("manual_dep_edges", []) == []

    def test_plain_call_rejects_deps_kwarg(self):
        """``deps=`` on a plain ``self.kernel(...)`` call is rejected — use pl.submit."""
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserTypeError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        b = self.k1(x, deps=[a])
                    return b

    def test_submit_in_auto_scope_records_manual_dep_edges(self):
        """``pl.submit(..., deps=[...])`` is orthogonal to ``manual_scope``.

        The runtime's ``Arg::set_dependencies`` adds explicit edges on top of
        auto-tracked OverlapMap deps (final fanin = auto ∪ explicit), so
        ``pl.submit`` and ``deps=`` work in auto scope too — as a precision
        tool that patches the edges auto can't infer (or infers too
        conservatively). No ``with pl.manual_scope():`` required.
        """

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
                # No manual_scope wrapper — auto OverlapMap stays on; the
                # explicit deps= entry is added on top.
                a, a_tid = pl.submit(self.k1, x)
                b, _ = pl.submit(self.k2, x, deps=[a_tid])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        # No RuntimeScopeStmt: the program stays in the implicit auto scope.
        assert _first_runtime_scope(fn.body) is None
        k2_call = next(c for c in _calls_in(fn.body) if c.op.name == "k2")
        edges = k2_call.attrs.get("manual_dep_edges", [])
        assert len(edges) == 1
        assert isinstance(edges[0].type, ir.ScalarType)
        assert edges[0].type.dtype == pl.TASK_ID

    def test_submit_as_bare_expression_is_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        pl.submit(self.k1, x)
                    return x

    def test_submit_single_target_is_rejected(self):
        """pl.submit must be unpacked as exactly ``(result, task_id)``."""
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = pl.submit(self.k1, x)
                    return a

    def test_pl_at_deps_and_as_tid_attach_scope_attrs(self):
        """``with pl.at(..., deps=[d1]) as tid:`` attaches metadata to the
        synthesised ScopeStmt via ``attrs_``. The outliner later promotes
        them to the ``Call`` it synthesises for the outlined kernel.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s1") as t1:
                    y: pl.Tensor[[64], pl.FP32] = x
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2", deps=[t1]) as _t2:
                    z: pl.Tensor[[64], pl.FP32] = y
                return z

        fn = Prog.get_function("main")
        assert fn is not None
        # Parser emits placeholder ``AssignStmt(tid, system.task_invalid())``
        # before each scope to give ConvertToSSA a definition; the outliner
        # drops these once it generates the real binding.
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        # First stmt: placeholder for t1.
        assert isinstance(stmts[0], ir.AssignStmt)
        assert isinstance(stmts[0].value, ir.Call)
        assert stmts[0].value.op.name == "system.task_invalid"
        # Second stmt: the first pl.at scope. Its ``task_id_var`` attr must
        # point at the same Var bound by the placeholder above (otherwise the
        # outliner couldn't unify the synthesised ``TupleGetItem`` binding
        # with subsequent ``deps=[t1]`` uses).
        assert isinstance(stmts[1], ir.InCoreScopeStmt)
        scope1_attrs = stmts[1].attrs
        assert "task_id_var" in scope1_attrs, f"scope1 missing task_id_var: keys={list(scope1_attrs)}"
        assert scope1_attrs["task_id_var"] is stmts[0].var
        # First scope has no deps=, so manual_dep_edges is absent (not an empty list).
        assert "manual_dep_edges" not in scope1_attrs
        # Third stmt: placeholder for t2.
        assert isinstance(stmts[2], ir.AssignStmt)
        # Fourth stmt: the second pl.at scope with deps=. Both attrs are set;
        # ``manual_dep_edges`` references t1 (the producer Var from scope1's
        # ``task_id_var``).
        assert isinstance(stmts[3], ir.InCoreScopeStmt)
        scope2_attrs = stmts[3].attrs
        assert "task_id_var" in scope2_attrs
        assert scope2_attrs["task_id_var"] is stmts[2].var
        assert "manual_dep_edges" in scope2_attrs
        assert len(scope2_attrs["manual_dep_edges"]) == 1
        assert scope2_attrs["manual_dep_edges"][0] is scope1_attrs["task_id_var"]

    def test_pl_at_as_on_non_at_scope_is_rejected(self):
        """``as`` is only meaningful on ``pl.at(...)``; other constructs reject it."""
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope() as not_supported:  # noqa: F841
                        return x

    def test_submit_nested_result_tuple_is_rejected(self):
        """pl.submit result targets must be plain names — no nested tuples."""
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        # Nested tuple in the result target — must error rather
                        # than silently pass the arity check.
                        (a, (b, c)), tid = pl.submit(self.k1, x)
                    return a


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
