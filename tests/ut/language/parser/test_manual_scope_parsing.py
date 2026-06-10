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
from pypto.language.parser.diagnostics.exceptions import ParserTypeError, UnsupportedFeatureError


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
    """Collect every call-like RHS (Call OR Submit) of an AssignStmt in the
    subtree. Submit shares ``.op.name`` with Call, so most assertions remain
    valid; tests that probe dep info use ``submit.deps`` instead of
    ``call.attrs['manual_dep_edges']``."""
    calls = []
    for s in _flatten(stmt):
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, (ir.Call, ir.Submit)):
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
        assert list(k1_call.deps) == []
        # Consumer k2 records one dep edge, naming the TaskId scalar `a_tid`.
        k2_deps = list(k2_call.deps)
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
        # ``deps=[None]`` drops the only entry, so the Submit's deps_ stays empty.
        assert list(k1_call.deps) == []

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

    def test_call_rejects_manual_dep_edges_in_attrs(self):
        """``attrs={'manual_dep_edges': ...}`` is rejected on any call — deps live
        on ``Submit::deps_`` only (ManualDepsOnSubmitOnly invariant)."""
        with pytest.raises(ParserTypeError, match="manual_dep_edges"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a, a_tid = pl.submit(self.k1, x)
                        b = self.k1(x, attrs={"manual_dep_edges": [a_tid]})
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
        edges = list(k2_call.deps)
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

    def test_submit_single_target_binds_full_tuple(self):
        """``result = pl.submit(self.k, ...)`` is the single-LHS form. The
        whole flat ``Tuple{<kernel result>, TaskId}`` binds to one Var.
        Mostly emitted by the printer (round-trip path) when an IR pass
        rewrites a Submit whose LHS is a single tuple-typed Var; user code
        usually still writes the unpacked ``a, tid = pl.submit(...)`` form.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = pl.submit(self.k1, x)  # noqa: F841 — checked via IR
                return x

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        k1_calls = [c for c in _calls_in(scope.body) if c.op.name == "k1"]
        assert len(k1_calls) == 1
        submit = k1_calls[0]
        assert isinstance(submit, ir.Submit)
        # Single-LHS bind: the AssignStmt's LHS Var has the flat tuple type;
        # no separate result / TaskId projection statements are synthesised.
        assert isinstance(submit.type, ir.TupleType)
        assert len(submit.type.types) == 2
        assert isinstance(submit.type.types[1], ir.ScalarType)
        assert submit.type.types[1].dtype == pl.TASK_ID

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

    def test_pl_at_deps_and_as_tid_print_parse_roundtrip(self):
        """The Python printer must surface the ``deps=`` and ``as <tid>:``
        clauses (and suppress the parser's transient ``system.task_invalid()``
        placeholder), so a print → parse cycle reproduces the same IR.
        Without this, ``manual_dep_edges`` and ``task_id_var`` would silently
        vanish across a round-trip.
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

        printed = str(Prog)
        # Surface checks: the two attr-driven clauses must be in the dump.
        assert "as t1" in printed
        assert "deps=[t1]" in printed
        assert "as _t2" in printed
        # The placeholder must NOT round-trip — the printer omits it because
        # the reparser will recreate it from the ``as <tid>`` clause.
        assert "task_invalid" not in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Prog, reparsed)

    def test_pl_at_no_dep_args_print_parse_roundtrip(self):
        """``no_dep_args=`` survives a print → parse round-trip on its own."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[w]):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        printed = str(Prog)
        assert "no_dep_args=[w]" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Prog, reparsed)

    def test_pl_at_deps_no_dep_args_and_tid_combined_roundtrip(self):
        """All three attr-driven kwargs in one ``pl.at(...)`` round-trip
        together."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s1") as t1:
                    a: pl.Tensor[[64], pl.FP32] = x
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="s2",
                    deps=[t1],
                    no_dep_args=[w],
                ) as _t2:
                    b: pl.Tensor[[64], pl.FP32] = pl.add(a, w)
                return b

        printed = str(Prog)
        assert "as t1" in printed
        assert "deps=[t1]" in printed
        assert "no_dep_args=[w]" in printed
        assert "as _t2" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Prog, reparsed)

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


class TestPlAtNoDepArgsParsing:
    """``pl.at(no_dep_args=[t1, t2])`` marks scope-captured tensor args as NoDep."""

    def test_no_dep_args_records_arg_direction_overrides_vars(self):
        """The parser writes the resolved Var list to ScopeStmt.attrs[arg_direction_overrides_vars]."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[w]):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        fn = Prog.get_function("main")
        assert fn is not None
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        scope = next(s for s in stmts if isinstance(s, ir.InCoreScopeStmt))
        attrs = scope.attrs
        assert "arg_direction_overrides_vars" in attrs
        no_dep_vars = attrs["arg_direction_overrides_vars"]
        assert len(no_dep_vars) == 1
        assert no_dep_vars[0].name_hint == "w"
        # deps= and task_id_var are independent paths; this scope uses neither.
        assert "manual_dep_edges" not in attrs
        assert "task_id_var" not in attrs

    def test_no_dep_args_combines_with_deps_and_tid(self):
        """``deps=``, ``no_dep_args=``, and ``as tid:`` are independent and may all appear."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s1") as t1:
                    a: pl.Tensor[[64], pl.FP32] = x
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="s2",
                    deps=[t1],
                    no_dep_args=[w],
                ) as _t2:
                    b: pl.Tensor[[64], pl.FP32] = pl.add(a, w)
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        # Second pl.at is the fourth stmt (after placeholder + scope1 + placeholder for t2).
        scope2 = next(s for s in stmts[2:] if isinstance(s, ir.InCoreScopeStmt) and s.name_hint == "s2")
        attrs = scope2.attrs
        assert "manual_dep_edges" in attrs
        assert "task_id_var" in attrs
        assert "arg_direction_overrides_vars" in attrs
        assert len(attrs["arg_direction_overrides_vars"]) == 1
        assert attrs["arg_direction_overrides_vars"][0].name_hint == "w"

    def test_no_dep_args_rejects_non_list_literal(self):
        with pytest.raises(ParserTypeError, match="must be a list literal"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    nope = [x]  # noqa: F841
                    with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=nope):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_no_dep_args_rejects_non_name_entry(self):
        with pytest.raises(ParserTypeError, match="bare tensor names"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[pl.no_dep(x)]):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_no_dep_args_rejects_unknown_name(self):
        # `ghost` is intentionally undefined at the DSL level — the parser
        # must raise a clear "unknown name" error rather than crash. The
        # bare name never has to exist as a Python binding: @pl.program
        # inspects the function source (ast.parse) rather than executing
        # it, so static-checker noise about ``ghost`` is irrelevant here.
        with pytest.raises(ParserTypeError, match="unknown name"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        no_dep_args=[ghost],  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
                    ):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_no_dep_args_rejects_duplicate(self):
        with pytest.raises(ParserTypeError, match="more than once"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[x, x]):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_no_dep_args_empty_list_is_noop(self):
        """``no_dep_args=[]`` is tolerated and leaves the scope attrs clean."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[]):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        fn = Prog.get_function("main")
        assert fn is not None
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        scope = next(s for s in stmts if isinstance(s, ir.InCoreScopeStmt))
        assert "arg_direction_overrides_vars" not in scope.attrs


# Module-level Python constants used by Form-2 comprehension tests below.
# They live in the test module's globals so the parser's closure resolver
# can read them at parse time (same path used for shape constants).
_FORM2_SKIP_INDEX = 1
_FORM2_INDEX_LIST = [0, 2]


def _all_calls(stmt):
    """Collect every call-like RHS (``ir.Call`` OR ``ir.Submit``) reachable
    as the value of any ``AssignStmt`` in the subtree, recursing into ForStmt /
    IfStmt / RuntimeScopeStmt / SeqStmts bodies. Submit shares ``op.name``
    with Call, so most filter-by-name tests remain valid; dep-probing tests
    use ``submit.deps`` instead of ``call.attrs['manual_dep_edges']``.
    """
    calls: list = []

    def _walk(s):
        if s is None:
            return
        if isinstance(s, ir.SeqStmts):
            for sub in s.stmts:
                _walk(sub)
            return
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, (ir.Call, ir.Submit)):
            calls.append(s.value)
        for attr in ("body", "then_body", "else_body"):
            sub = getattr(s, attr, None)
            if sub is not None:
                _walk(sub)

    _walk(stmt)
    return calls


class TestSubmitDepsPerElementAndComprehension:
    """Tests for the Form 1 (``arr[i]``) and Form 2 (``[arr[i] for i in ...]``)
    DSL surfaces for ``pl.submit(..., deps=...)``. Both forms desugar in the
    parser into a fresh ``Array[K, TASK_ID]`` populated by N
    ``array.update_element`` calls; the existing whole-array dep codegen
    path then emits one ``is_valid()``-guarded slot per entry.
    """

    def test_form1_single_per_element_subscript(self):
        """``deps=[tids[n]]`` desugars to a size-1 fresh array."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        a, t = pl.submit(self.producer, x)
                        tids[n] = t
                    for n in pl.parallel(4):
                        b, _ = pl.submit(self.consumer, x, deps=[tids[n]])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _all_calls(scope.body)
        # The consumer call must record exactly one dep edge, and that edge
        # must be a Var of type Array[1, TASK_ID] — the synthesized buffer.
        consumer_calls = [c for c in calls if c.op.name == "consumer"]
        assert len(consumer_calls) == 1
        edges = consumer_calls[0].deps
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge, ir.Var)
        assert isinstance(edge.type, ir.ArrayType)
        assert edge.type.dtype == pl.TASK_ID

    def test_form1_mixed_scalar_and_subscript(self):
        """``deps=[scalar, tids[2*k], tids[2*k+1]]`` desugars to a size-3 fresh array."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    seed_a, seed_tid = pl.submit(self.producer, x)
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        a, t = pl.submit(self.producer, x)
                        tids[n] = t
                    for k in pl.parallel(2):
                        b, _ = pl.submit(
                            self.consumer,
                            x,
                            deps=[seed_tid, tids[2 * k], tids[2 * k + 1]],
                        )
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _all_calls(scope.body)
        consumer_calls = [c for c in calls if c.op.name == "consumer"]
        assert len(consumer_calls) == 1
        edges = consumer_calls[0].deps
        # One Array[3, TASK_ID] entry (the synthesized buffer) after desugar.
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge.type, ir.ArrayType)
        assert edge.type.dtype == pl.TASK_ID
        # The synthesizer must emit one array.create + 3 array.update_element
        # calls *before* the consumer kernel call. Look for them by op name.
        create_calls = [c for c in calls if c.op.name == "array.create"]
        update_calls = [c for c in calls if c.op.name == "array.update_element"]
        assert len(create_calls) >= 1  # one for the user `tids` plus the synth buffer
        # Two `tids[n] = t` writes (from the producer loop) plus 3 from the synthesizer.
        assert len(update_calls) >= 3

    def test_form2_comprehension_static_range(self):
        """``deps=[arr[k] for k in range(K)]`` unrolls to a size-K fresh array."""
        K = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(K, pl.TASK_ID)
                    for n in pl.parallel(K):
                        a, t = pl.submit(self.producer, x)
                        tids[n] = t
                    b, _ = pl.submit(self.consumer, x, deps=[tids[k] for k in range(K)])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _all_calls(scope.body)
        consumer_calls = [c for c in calls if c.op.name == "consumer"]
        assert len(consumer_calls) == 1
        edges = consumer_calls[0].deps
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge.type, ir.ArrayType)
        # The synthesized buffer must be size K (one slot per unrolled entry).
        if hasattr(edge.type, "extent") and isinstance(edge.type.extent, ir.ConstInt):
            assert edge.type.extent.value == K

    def test_form2_comprehension_with_static_filter(self):
        """A filter that depends only on Python-level names is honored."""

        K = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(K, pl.TASK_ID)
                    for n in pl.parallel(K):
                        a, t = pl.submit(self.producer, x)
                        tids[n] = t
                    # _FORM2_SKIP_INDEX is a module-level Python int (1) — the
                    # filter resolves at parse time and excludes one slot, so
                    # the synthesized array has K-1 entries.
                    b, _ = pl.submit(
                        self.consumer,
                        x,
                        deps=[tids[k] for k in range(K) if k != _FORM2_SKIP_INDEX],
                    )
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _all_calls(scope.body)
        consumer_call = next(c for c in calls if c.op.name == "consumer")
        edges = consumer_call.deps
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge.type, ir.ArrayType)
        if hasattr(edge.type, "extent") and isinstance(edge.type.extent, ir.ConstInt):
            # K=4, one slot filtered out by `k != 1` → expect 3.
            assert edge.type.extent.value == K - 1

    def test_form2_comprehension_global_iterable(self):
        """A module-global list is a valid iterable for Form 2."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        a, t = pl.submit(self.producer, x)
                        tids[n] = t
                    b, _ = pl.submit(self.consumer, x, deps=[tids[k] for k in _FORM2_INDEX_LIST])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        consumer_call = next(c for c in _all_calls(scope.body) if c.op.name == "consumer")
        edges = consumer_call.deps
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge.type, ir.ArrayType)
        if hasattr(edge.type, "extent") and isinstance(edge.type.extent, ir.ConstInt):
            assert edge.type.extent.value == len(_FORM2_INDEX_LIST)

    def test_form3_pl_range_iterable_rejected(self):
        """An IR `pl.range(K)` iterable in the comprehension is rejected
        with a clear error pointing at Form 1 inline."""
        with pytest.raises(ParserTypeError, match="DSL loop"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = pl.array.create(4, pl.TASK_ID)
                        for n in pl.parallel(4):
                            a, t = pl.submit(self.producer, x)
                            tids[n] = t
                        # pl.range is an IR loop — not parse-time iterable.
                        b, _ = pl.submit(
                            self.consumer,
                            x,
                            deps=[tids[k] for k in pl.range(4)],
                        )
                    return b

    def test_form3_ir_var_in_filter_rejected(self):
        """A filter that depends on an IR Var (from `pl.range`) is rejected."""
        with pytest.raises(ParserTypeError, match="IR variable"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = pl.array.create(4, pl.TASK_ID)
                        for n in pl.parallel(4):
                            a, t = pl.submit(self.producer, x)
                            tids[n] = t
                        for outer in pl.range(2):
                            # `outer` is an IR Var — filter cannot be evaluated.
                            b, _ = pl.submit(
                                self.consumer,
                                x,
                                deps=[tids[k] for k in range(4) if k != outer],
                            )
                    return b

    def test_all_scalar_deps_keep_direct_path(self):
        """When every `deps=` entry is a bare TaskId scalar Var, NO synthesizer
        fires — codegen golden output stays byte-identical."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.producer, x)
                    b, b_tid = pl.submit(self.producer, x)
                    c, _ = pl.submit(self.consumer, x, deps=[a_tid, b_tid])
                return c

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        calls = _all_calls(scope.body)
        consumer_call = next(c for c in calls if c.op.name == "consumer")
        edges = consumer_call.deps
        # Two SCALAR entries — direct path, no Array[N, TASK_ID] synthesis.
        assert len(edges) == 2
        for edge in edges:
            assert isinstance(edge.type, ir.ScalarType)
            assert edge.type.dtype == pl.TASK_ID
        # No `_submit_deps_buf`-style array.create should appear in this
        # function — the synth gate must stay off.
        synth_creates = [
            c
            for c in calls
            if c.op.name == "array.create"
            # locally-bound _submit_deps_buf would be the array.create's LHS;
            # the user wrote no array.create themselves here.
        ]
        assert len(synth_creates) == 0

    def test_intermediate_name_for_comprehension_rejected(self):
        """Decision A — the comprehension must appear inline in `deps=`.
        Binding it to a separate variable is not supported because
        ``cast_deps = <ListComp>`` is rejected as an unsupported RHS
        (``UnsupportedFeatureError: Unsupported expression type: ListComp``)
        before we ever reach ``deps=cast_deps``."""
        with pytest.raises(UnsupportedFeatureError, match="ListComp"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = pl.array.create(4, pl.TASK_ID)
                        for n in pl.parallel(4):
                            a, t = pl.submit(self.producer, x)
                            tids[n] = t
                        cast_deps = [tids[k] for k in range(4)]
                        b, _ = pl.submit(self.consumer, x, deps=cast_deps)
                    return b

    def test_mixed_whole_array_then_per_element_rejected(self):
        """``deps=[whole_arr, arr[i]]`` cannot be desugared — the synthesizer
        would feed an ArrayType Var into ``array.update_element``'s scalar
        value slot, tripping a C++ type-deducer CHECK. Reject the mixed form
        at parse time with a clear error."""
        with pytest.raises(ParserTypeError, match="cannot mix"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = pl.array.create(4, pl.TASK_ID)
                        for n in pl.parallel(4):
                            a, t = pl.submit(self.producer, x)
                            tids[n] = t
                        # Whole array first, then a per-element read — rejected.
                        b, _ = pl.submit(self.consumer, x, deps=[tids, tids[0]])
                    return b

    def test_mixed_per_element_then_whole_array_rejected(self):
        """Same correctness guard, opposite order — per-element first, whole
        array second. The synthesizer trips the same way regardless of order,
        so both orientations raise."""
        with pytest.raises(ParserTypeError, match="cannot mix"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.InCore)
                def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = pl.array.create(4, pl.TASK_ID)
                        for n in pl.parallel(4):
                            a, t = pl.submit(self.producer, x)
                            tids[n] = t
                        # Per-element first, then whole array — rejected.
                        b, _ = pl.submit(self.consumer, x, deps=[tids[0], tids])
                    return b

    def test_pl_at_form1_per_element_subscript(self):
        """``with pl.at(..., deps=[arr[i]])`` — the per-element form must
        also work on the pl.at path, since both share ``_parse_submit_deps_kwarg``.

        The synthesized ``Array[1, TASK_ID]`` Var must be emitted as
        AssignStmts BEFORE the ScopeStmt so the attr reference resolves.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s1") as t1:
                    y: pl.Tensor[[64], pl.FP32] = x
                tids = pl.array.create(1, pl.TASK_ID)
                tids[0] = t1
                # Per-element read on the pl.at deps= site.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2", deps=[tids[0]]):
                    z: pl.Tensor[[64], pl.FP32] = y
                return z

        fn = Prog.get_function("main")
        assert fn is not None
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        # The second pl.at scope is wherever the per-element form was used.
        scopes = [s for s in stmts if isinstance(s, ir.InCoreScopeStmt)]
        assert len(scopes) == 2
        scope2_attrs = scopes[1].attrs
        assert "manual_dep_edges" in scope2_attrs
        # Synthesized buffer — a single Array[N, TASK_ID] entry, not two scalars.
        edges = scope2_attrs["manual_dep_edges"]
        assert len(edges) == 1
        assert isinstance(edges[0], ir.Var)
        assert isinstance(edges[0].type, ir.ArrayType)
        assert edges[0].type.dtype == pl.TASK_ID
        # The synth array.create + update_element AssignStmts must precede the
        # ScopeStmt that references them (otherwise the attr Var dangles).
        scope2_idx = stmts.index(scopes[1])
        prior_calls = []
        for s in stmts[:scope2_idx]:
            if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call):
                prior_calls.append(s.value)
        synth_creates = [c for c in prior_calls if c.op.name == "array.create"]
        synth_updates = [c for c in prior_calls if c.op.name == "array.update_element"]
        # Two array.create calls before scope2: the user's `tids` and the synth.
        # Two array.update_element calls: `tids[0] = t1` (user) + synthesized fill.
        assert len(synth_creates) == 2
        assert len(synth_updates) == 2

    def test_pl_at_form2_comprehension(self):
        """``with pl.at(..., deps=[arr[k] for k in range(K)])`` unrolls correctly."""
        K = 3

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Seed K producer scopes; collect their TaskIds in an array.
                tids = pl.array.create(K, pl.TASK_ID)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="p0") as t0:
                    y0: pl.Tensor[[64], pl.FP32] = x
                tids[0] = t0
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="p1") as t1:
                    y1: pl.Tensor[[64], pl.FP32] = y0
                tids[1] = t1
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="p2") as t2:
                    y2: pl.Tensor[[64], pl.FP32] = y1
                tids[2] = t2
                # Consumer fences on all K producers via comprehension.
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="consumer",
                    deps=[tids[k] for k in range(K)],
                ):
                    z: pl.Tensor[[64], pl.FP32] = y2
                return z

        fn = Prog.get_function("main")
        assert fn is not None
        stmts = list(fn.body.stmts) if isinstance(fn.body, ir.SeqStmts) else [fn.body]
        scopes = [s for s in stmts if isinstance(s, ir.InCoreScopeStmt)]
        # Three producer pl.at scopes + one consumer pl.at scope.
        assert len(scopes) == 4
        consumer_scope = scopes[-1]
        # pl.at scope still uses attrs['manual_dep_edges'] — Submit replaces
        # only the Call-side dep encoding, not ScopeStmt's.
        edges = consumer_scope.attrs.get("manual_dep_edges", [])
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge, ir.Var)
        assert isinstance(edge.type, ir.ArrayType)
        assert edge.type.dtype == pl.TASK_ID
        # The synthesized buffer must be size K.
        if hasattr(edge.type, "extent") and isinstance(edge.type.extent, ir.ConstInt):
            assert edge.type.extent.value == K


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
