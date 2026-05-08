# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeriveManualScopeDeps pass."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Skip the global print -> parse -> assert_structural_equal roundtrip.

    The python_printer does not surface ``Call.attrs['manual_dep_edges']`` (an
    internal post-pass attr), so the roundtrip would always fail after this
    pass. Property verification still runs.
    """
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


def _calls_in_manual_scope(fn):
    """Return ``(lhs_var, call)`` pairs for every kernel call in a manual scope."""
    out = []

    def walk(stmt):
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        elif isinstance(stmt, ir.RuntimeScopeStmt):
            if stmt.manual:
                _walk_manual(stmt.body, out)
            else:
                walk(stmt.body)

    def _walk_manual(stmt, acc):
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                _walk_manual(s, acc)
        elif isinstance(stmt, ir.AssignStmt):
            v = stmt.value
            if isinstance(v, ir.Call) and not v.op.name.startswith(("tensor.", "tile.", "system.")):
                acc.append((stmt.var, v))

    walk(fn.body)
    return out


def _edges(call):
    return call.attrs.get("manual_dep_edges", [])


class TestDeriveManualScopeDeps:
    def test_no_manual_scope_is_noop(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)
        # No manual scope ⇒ pass is a no-op (returns same Program).
        assert ddep.same_as(ddir)

    def test_dataflow_edge_auto_derived(self):
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
                    b = self.k2(a)
                return b

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)

        fn = ddep.get_function("main")
        calls = _calls_in_manual_scope(fn)
        assert len(calls) == 2
        # k1 has no producer yet ⇒ no manual_dep_edges attr written.
        assert _edges(calls[0][1]) == []
        # k2 consumes `a`, which is a producer in the same manual scope.
        e = _edges(calls[1][1])
        assert len(e) == 1
        assert e[0].same_as(calls[0][0])

    def test_user_deps_merge_with_dataflow(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    b = self.k2(x)
                    # k3 reads `x` (no auto edge). User asks for explicit deps on a, b.
                    c = self.k3(x, deps=[a, b])
                return c

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)

        fn = ddep.get_function("main")
        calls = _calls_in_manual_scope(fn)
        assert len(calls) == 3
        a_var, _ = calls[0]
        b_var, _ = calls[1]
        c_lhs, c_call = calls[2]
        edges = _edges(c_call)
        assert len(edges) == 2
        assert edges[0].same_as(a_var)
        assert edges[1].same_as(b_var)

    def test_user_deps_only_adopted_into_manual_dep_edges(self):
        """User-supplied deps that aren't in the data flow still appear as edges."""

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

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)

        fn = ddep.get_function("main")
        calls = _calls_in_manual_scope(fn)
        a_var, _ = calls[0]
        _, b_call = calls[1]
        edges = _edges(b_call)
        assert len(edges) == 1
        assert edges[0].same_as(a_var)

    def test_tuple_unpacking_propagates_producer(self):
        """``a, b = self.kpair(x)`` makes both ``a`` and ``b`` resolve to the
        kpair task as their producer — otherwise consumers race the call.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kpair(
                self, x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return x, x

            @pl.function(type=pl.FunctionType.InCore)
            def k_add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, b = self.kpair(x)
                    c = self.k_add(a)
                return c

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)

        fn = ddep.get_function("main")
        assert fn is not None
        # Find the tuple-temp var (LHS of the only kernel call other than k_add)
        # and the k_add call.
        tuple_tmp_var = None
        kadd_call = None

        def walk(stmt):
            nonlocal tuple_tmp_var, kadd_call
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    walk(s)
            elif isinstance(stmt, ir.RuntimeScopeStmt):
                walk(stmt.body)
            elif isinstance(stmt, ir.AssignStmt):
                v = stmt.value
                if isinstance(v, ir.Call):
                    if v.op.name == "kpair":
                        tuple_tmp_var = stmt.var
                    elif v.op.name == "k_add":
                        kadd_call = v

        walk(fn.body)
        assert tuple_tmp_var is not None
        assert kadd_call is not None
        edges = kadd_call.attrs.get("manual_dep_edges", [])
        assert len(edges) == 1
        assert edges[0].same_as(tuple_tmp_var), "k_add's manual_dep_edge should point at the kpair tuple temp"

    def test_var_to_var_copy_propagates_producer(self):
        """A trivial ``c = a`` rebind shouldn't lose ``a``'s producer."""

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
                    c = a
                    d = self.k2(c)
                return d

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)

        fn = ddep.get_function("main")
        assert fn is not None
        a_var = None
        k2_call = None

        def walk(stmt):
            nonlocal a_var, k2_call
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    walk(s)
            elif isinstance(stmt, ir.RuntimeScopeStmt):
                walk(stmt.body)
            elif isinstance(stmt, ir.AssignStmt):
                v = stmt.value
                if isinstance(v, ir.Call):
                    if v.op.name == "k1":
                        a_var = stmt.var
                    elif v.op.name == "k2":
                        k2_call = v

        walk(fn.body)
        assert a_var is not None
        assert k2_call is not None
        edges = k2_call.attrs.get("manual_dep_edges", [])
        assert len(edges) == 1
        assert edges[0].same_as(a_var)


class TestManualScopeNesting:
    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_nested_manual_scope_rejected(self):
        """The runtime forbids MANUAL inside MANUAL; reject at parse time."""
        with pytest.raises(Exception, match="manual_scope"):  # noqa: B017

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        with pl.manual_scope():  # nested — must error
                            b = self.k1(a)
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
